"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + RRF fusion + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song (ThreadPoolExecutor)
    2. Nếu điểm cosine GỐC tốt nhất < threshold → fallback sang PageIndex và
       dừng luôn (không rerank, tiết kiệm một chat-completion trả phí)
    3. Fuse hai ranked list bằng RRF (Task 7)
    4. Rerank
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP:
    Nếu dùng điểm RRF đã fuse để so với ``score_threshold``, bạn sẽ gặp bug thật:
    RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan hay
    không. Vì vậy pipeline này quyết định fallback bằng ``dense_results[0]["score"]``
    — cosine similarity gốc từ Task 5 — tách hẳn khỏi điểm RRF dùng để xếp hạng.

CALIBRATION (đo thật trên corpus hiện tại: 10 documents / 1475 chunks, embedding
``text-embedding-3-small``, điểm là best cosine của ``semantic_search`` sau khi
``expand_query`` — chạy lại khi đổi corpus hoặc embedding model):

    Câu hỏi đúng chủ đề          → 0.478 … 0.745
      "Xe máy vượt đèn đỏ bị phạt bao nhiêu tiền?"        0.674
      "Người lái xe có bao nhiêu điểm giấy phép lái xe?"  0.649
      "Khi xảy ra tai nạn giao thông cần làm gì?"         0.639
      "Hồ sơ đăng ký xe gồm những giấy tờ gì?"            0.745
      "Nồng độ cồn bao nhiêu thì bị phạt?"                0.478
      "Tốc độ tối đa trong khu đông dân cư là bao nhiêu?" 0.605

    Câu rác / lạc đề             → 0.148 … 0.313
      "xyzabc123nonsense"                                 0.209
      "asdkjhasd qwe zxc"                                 0.148
      "recipe for chocolate cake"                         0.153
      "How do I return an item on Shopee?"                0.227
      "quantum chromodynamics lattice gauge"              0.161
      "giá cổ phiếu VN30 hôm nay"                         0.313

    Khoảng trống 0.313 → 0.478; chọn ngưỡng gần giữa: 0.40.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.40  # Xem CALIBRATION ở docstring module
DEFAULT_TOP_K = 5
# "llm" = LLM đóng vai cross-encoder trên pool đã fuse. Cần thiết vì corpus toàn
# các khoản phạt viết gần giống nhau; RRF thuần cho top-1 chưa đủ chính xác.
RERANK_METHOD = "llm"  # "llm" | "cross_encoder" | "mmr" | "rrf"

# Câu hỏi của người dùng dùng tiếng Việt đời thường, văn bản luật dùng thuật ngữ
# pháp lý. Glossary tối thiểu này nối hai lớp từ vựng cho CẢ dense và BM25 —
# rẻ, tất định, giải thích được (không tốn thêm API call như query rewriting).
QUERY_GLOSSARY = {
    "vượt đèn đỏ": "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
    "xe máy": "xe mô tô, xe gắn máy",
    "bằng lái": "giấy phép lái xe",
    "nồng độ cồn": "trong máu hoặc hơi thở có nồng độ cồn",
    "rượu bia": "trong máu hoặc hơi thở có nồng độ cồn",
    "mũ bảo hiểm": "không đội mũ bảo hiểm cho người đi mô tô, xe máy",
    "quá tốc độ": "chạy quá tốc độ quy định",
    "đi ngược chiều": "đi ngược chiều của đường một chiều",
    "cà vẹt": "chứng nhận đăng ký xe",
    "biển số": "biển số xe",
}


def expand_query(query: str) -> str:
    """Nối thêm thuật ngữ pháp lý tương ứng vào cuối query nếu có trong glossary."""
    lowered = query.lower()
    additions = [
        legal_term
        for colloquial, legal_term in QUERY_GLOSSARY.items()
        if colloquial in lowered and legal_term.lower() not in lowered
    ]
    return f"{query} ({'; '.join(additions)})" if additions else query


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ If dense_results[0]["score"] < threshold:
          │     └→ PageIndex Vectorless → return fallback_results
          │
          ├→ Merge (RRF) → merged_results
          └→ Rerank → reranked_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có áp dụng bước rerank sau khi fuse hay không

    Returns:
        List of {'content', 'score', 'metadata', 'source'} với
        ``source ∈ {"hybrid", "pageindex"}``.

        Trả về list RỖNG khi câu hỏi nằm ngoài corpus và PageIndex cũng không có
        gì — cố tình không trả một điều luật ngẫu nhiên để Task 10 nói thẳng là
        không xác minh được.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    # Pool rộng hơn top_k khá nhiều: một Điều luật dài bị cắt thành nhiều chunk,
    # chunk mang đúng mức phạt thường nằm quanh hạng 8-12 của dense. Fuse trên
    # pool hẹp sẽ đánh rơi nó.
    candidate_k = max(top_k * 4, 20)

    # Step 1: hai ranker độc lập, chạy song song (semantic tốn 1 API call,
    # lexical tốn CPU — overlap được thời gian chờ mạng).
    # Query rác không khớp glossary nên không bị đổi → ngưỡng fallback đã
    # calibrate ở docstring vẫn đúng.
    search_query = expand_query(query)

    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(semantic_search, search_query, candidate_k)
        sparse_future = executor.submit(lexical_search, search_query, candidate_k)
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    # Step 2: quyết định fallback bằng COSINE GỐC, không phải điểm RRF — và
    # kiểm tra TRƯỚC khi rerank, vì RERANK_METHOD="llm" tốn một chat-completion
    # trả phí mà kết quả sẽ bị vứt đi ngay sau đó.
    best_dense_score = dense_results[0]["score"] if dense_results else 0.0
    if best_dense_score < score_threshold:
        print(
            f"  ⚠ Semantic best score ({best_dense_score:.3f}) "
            f"< threshold ({score_threshold}) → PageIndex fallback"
        )
        return pageindex_search(query, top_k=top_k)[:top_k]

    # Step 3: fuse bằng RRF — cosine và BM25 không cùng thang nên chỉ fuse rank.
    merged = rerank_rrf([dense_results, sparse_results], top_k=candidate_k)
    for item in merged:
        item["source"] = "hybrid"

    # Step 4: rerank (với "rrf" trên một list là giữ nguyên thứ tự + chuẩn hoá điểm).
    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        for item in final_results:
            item["source"] = "hybrid"
    else:
        final_results = merged[:top_k]

    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "Xe máy vượt đèn đỏ bị phạt bao nhiêu?",
        "Người lái xe có bao nhiêu điểm giấy phép lái xe?",
        "Khi xảy ra tai nạn giao thông cần làm gì?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for question in test_queries:
        print(f"\nQuery: {question}")
        print("-" * 60)
        results = retrieve(question, top_k=3)
        if not results:
            print("  (không có evidence — pipeline từ chối trả lời)")
        for index, result in enumerate(results, 1):
            source_name = result["metadata"].get("source", "?")
            print(
                f"  {index}. [{result['score']:.4f}] [{result['source']}] "
                f"{source_name} :: {result['content'][:70]}..."
            )
