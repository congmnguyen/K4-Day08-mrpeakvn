"""
Task 7 — Reranking Module.

Phương pháp CHÍNH được chọn: **RRF (Reciprocal Rank Fusion)**.
    - Không cần API key, không kéo Torch/cross-encoder local.
    - Fuse được hai thang điểm không so sánh được với nhau (cosine similarity của
      Task 5 và BM25 score của Task 6) bằng cách chỉ dùng THỨ HẠNG.

Hai phương pháp phụ cũng được implement để so sánh khi demo:
    - MMR: thuần numpy, cần embedding của candidates.
    - Cross-encoder qua Jina Reranker API: chỉ chạy khi có ``JINA_API_KEY``.

⚠️ Lưu ý quan trọng về RRF (dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ
hạng, không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈
0.0164 (k=60), bất kể nội dung có thật sự liên quan hay không. Đừng dùng điểm RRF
để quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

from __future__ import annotations

import hashlib
import os


def chunk_identity(item: dict) -> str:
    """Khoá dedupe ổn định cho một chunk.

    Ưu tiên ``path + chunk_index`` từ metadata của Task 4 vì hai ranker có thể
    trả cùng chunk với content bị cắt/normalize khác nhau. Chỉ hash content khi
    metadata không đủ.
    """
    metadata = item.get("metadata") or {}
    path = metadata.get("path") or metadata.get("source")
    chunk_index = metadata.get("chunk_index")
    if path is not None and chunk_index is not None:
        return f"{path}#{chunk_index}"
    return hashlib.sha256(item.get("content", "").encode("utf-8")).hexdigest()


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates bằng Jina Reranker v2 (multilingual) qua API.

    Chỉ dùng được khi có ``JINA_API_KEY``; không cài model local để tránh Torch.

    Returns:
        List of top_k candidates, re-scored và sorted by relevance descending.
    """
    import requests

    api_key = os.getenv("JINA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "rerank_cross_encoder cần JINA_API_KEY. Dùng method='rrf' để chạy offline."
        )
    if not candidates:
        return []

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [item["content"] for item in candidates],
            "top_n": top_k,
        },
        timeout=60,
    )
    response.raise_for_status()
    return [
        {**candidates[entry["index"]], "score": float(entry["relevance_score"])}
        for entry in response.json().get("results", [])
    ][:top_k]


LLM_RERANK_PROMPT = """Bạn chấm độ liên quan giữa một câu hỏi về pháp luật giao thông
đường bộ Việt Nam và từng đoạn văn bản luật được đánh số.

Chấm mỗi đoạn theo thang 0-10:
  10 = chứa trực tiếp câu trả lời (đúng loại phương tiện, đúng hành vi, có mức phạt)
  5  = cùng chủ đề nhưng không trả lời thẳng
  0  = không liên quan

Lưu ý: câu hỏi dùng ngôn ngữ đời thường, văn bản dùng thuật ngữ pháp lý.
Ví dụ "vượt đèn đỏ" = "không chấp hành hiệu lệnh của đèn tín hiệu giao thông";
"xe máy" = "xe mô tô, xe gắn máy"; "bằng lái" = "giấy phép lái xe".

Chỉ trả JSON: {"scores": [{"index": <số đoạn>, "score": <0-10>}, ...]}"""


def rerank_llm(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """Rerank bằng LLM đóng vai cross-encoder — một API call cho cả pool.

    Corpus pháp luật có hàng nghìn khoản phạt viết gần giống nhau; cosine của
    ``text-embedding-3-small`` giữa top-1 và top-60 chỉ chênh ~0.05 nên dense
    ranking một mình không phân biệt được. LLM đọc cả câu hỏi lẫn đoạn văn nên
    bắc được cầu giữa từ đời thường và thuật ngữ pháp lý.

    Mọi lỗi (API, JSON hỏng) đều rơi về thứ tự RRF hiện có thay vì raise.
    """
    import json
    import os

    if not candidates:
        return []

    from openai import OpenAI

    numbered = "\n\n".join(
        f"[{index}] "
        + (item.get("metadata", {}) or {}).get("heading", "")
        + "\n"
        + item["content"][:900]
        for index, item in enumerate(candidates)
    )
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": LLM_RERANK_PROMPT},
                {"role": "user", "content": f"Câu hỏi: {query}\n\n{numbered}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content or "{}")
        scores = {
            int(entry["index"]): float(entry["score"])
            for entry in payload.get("scores", [])
            if 0 <= int(entry["index"]) < len(candidates)
        }
    except Exception as error:  # noqa: BLE001 — rerank hỏng không được làm chết pipeline
        print(f"  ⚠ LLM rerank lỗi ({error}); giữ thứ tự RRF")
        return candidates[:top_k]

    if not scores:
        return candidates[:top_k]

    ranked = sorted(
        range(len(candidates)),
        # Tie-break bằng thứ hạng RRF sẵn có → ổn định khi LLM chấm bằng điểm.
        key=lambda index: (-scores.get(index, 0.0), index),
    )
    return [
        {**candidates[index], "score": round(scores.get(index, 0.0) / 10.0, 4)}
        for index in ranked[:top_k]
        if scores.get(index, 0.0) > 0
    ]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content', 'score', 'embedding', 'metadata'}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)
    """
    import numpy as np

    if not candidates:
        return []
    # Candidate đi ra từ Task 5/6/9 chỉ có content/score/metadata — tự embed nốt
    # để method="mmr" dùng được ngay thay vì raise.
    missing = [i for i, item in enumerate(candidates) if "embedding" not in item]
    if missing:
        from .task4_chunking_indexing import embed_texts

        vectors = embed_texts([candidates[i]["content"] for i in missing])
        candidates = [dict(item) for item in candidates]
        for index, vector in zip(missing, vectors, strict=True):
            candidates[index]["embedding"] = vector

    matrix = np.asarray([item["embedding"] for item in candidates], dtype=float)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    query_vector = np.asarray(query_embedding, dtype=float)
    query_vector /= np.linalg.norm(query_vector) + 1e-12
    relevance = matrix @ query_vector

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining and len(selected) < top_k:
        if selected:
            redundancy = (matrix[remaining] @ matrix[selected].T).max(axis=1)
        else:
            redundancy = np.zeros(len(remaining))
        mmr = lambda_param * relevance[remaining] - (1 - lambda_param) * redundancy
        best = remaining[int(np.argmax(mmr))]
        selected.append(best)
        remaining.remove(best)

    results = []
    for index in selected:
        item = {**candidates[index], "score": float(relevance[index])}
        item.pop("embedding", None)  # vector 1536 chiều không cần đi tiếp xuống Task 10
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending. Mỗi item giữ
        điểm gốc trong ``original_score`` để Task 9 còn dùng được thang thật.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    fused: dict[str, float] = {}
    items: dict[str, dict] = {}
    order: list[str] = []

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = chunk_identity(item)
            fused[key] = fused.get(key, 0.0) + 1.0 / (k + rank)
            if key not in items:
                items[key] = item
                order.append(key)

    # Tie-break bằng thứ tự xuất hiện đầu tiên → kết quả deterministic.
    position = {key: index for index, key in enumerate(order)}
    ranking = sorted(order, key=lambda key: (-fused[key], position[key]))

    results = []
    for key in ranking[:top_k]:
        item = dict(items[key])
        item["original_score"] = item.get("score")
        item["score"] = fused[key]
        results.append(item)
    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "llm" | "cross_encoder" | "mmr" | "rrf"
    query_embedding: list[float] | None = None,
) -> list[dict]:
    """
    Unified reranking interface.

    Với ``method="rrf"``, ``candidates`` được coi là MỘT ranked list. Fuse một
    list bằng RRF chỉ giữ nguyên thứ tự và chuẩn hoá điểm về thang RRF — đúng
    ngữ nghĩa và không raise. Muốn fuse nhiều ranker (Task 9) thì gọi thẳng
    ``rerank_rrf([dense, sparse])``.
    """
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not candidates:
        return []

    if method == "llm":
        return rerank_llm(query, candidates, top_k)
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        if query_embedding is None:
            from .task5_semantic_search import embed_query

            query_embedding = embed_query(query)
        return rerank_mmr(query_embedding, candidates, top_k)
    if method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    dummy_candidates = [
        {"content": "Mức xử phạt người điều khiển xe mô tô vượt đèn đỏ", "score": 0.8, "metadata": {}},
        {"content": "Quy định về điểm của giấy phép lái xe", "score": 0.6, "metadata": {}},
        {"content": "Quy định về cấp biển số xe cơ giới", "score": 0.5, "metadata": {}},
    ]
    for result in rerank("mức phạt xe máy vượt đèn đỏ", dummy_candidates, top_k=2):
        print(f"[{result['score']:.4f}] {result['content']}")
