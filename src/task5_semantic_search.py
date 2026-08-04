"""
Task 5 — Semantic Search Module.

Dense retrieval trên ChromaDB đã index ở Task 4.

Thiết kế:
    - Query được embed bằng ĐÚNG model/dimension của Task 4 (``embed_texts``),
      nếu không vector query nằm khác không gian và điểm số vô nghĩa.
    - Collection tạo với ``hnsw:space=cosine`` nên Chroma trả cosine DISTANCE
      trong ``[0, 2]``; similarity = ``1 - distance``.
    - Cache embedding của query trong process: test và pipeline hybrid gọi lại
      cùng câu hỏi nhiều lần, cache cắt được số lần gọi API.
"""

from __future__ import annotations

from .task4_chunking_indexing import embed_texts, get_collection

_QUERY_EMBEDDING_CACHE: dict[str, list[float]] = {}
_COLLECTION = None


def collection():
    """Giữ một handle collection cho cả process.

    Mỗi lần ``get_collection()`` sẽ tạo lại ``PersistentClient``; trong app
    Streamlit điều đó lặp lại ở mọi lượt chat. Cache lại để chi phí chỉ trả một
    lần — và quan trọng hơn, lần build index đầu tiên trên deploy mới không bị
    kích hoạt lặp.
    """
    global _COLLECTION
    if _COLLECTION is None:
        _COLLECTION = get_collection()
    return _COLLECTION


def embed_query(query: str) -> list[float]:
    """Embed query, cache theo text để tránh gọi API trùng lặp trong 1 process."""
    cached = _QUERY_EMBEDDING_CACHE.get(query)
    if cached is None:
        cached = embed_texts([query])[0]
        _QUERY_EMBEDDING_CACHE[query] = cached
    return cached


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    response = collection().query(
        query_embeddings=[embed_query(query)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = response.get("documents") or [[]]
    metadatas = response.get("metadatas") or [[]]
    distances = response.get("distances") or [[]]

    results: list[dict] = []
    for content, metadata, distance in zip(
        documents[0], metadatas[0], distances[0], strict=True
    ):
        results.append(
            {
                "content": content,
                # Chroma cosine distance = 1 - cosine_similarity; clamp về [0, 1]
                # để điểm số dùng được trực tiếp làm ngưỡng fallback ở Task 9.
                "score": round(max(0.0, 1.0 - float(distance)), 4),
                "metadata": dict(metadata or {}),
            }
        )
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    for result in semantic_search("mức phạt xe máy vượt đèn đỏ", top_k=5):
        source = result["metadata"].get("source", "?")
        print(f"[{result['score']:.3f}] {source} :: {result['content'][:100]}...")
