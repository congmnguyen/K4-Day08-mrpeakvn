"""
Task 6 — Lexical Search Module (BM25).

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Ghi chú thiết kế:
    - Corpus BM25 là CHUNKS (không phải document nguyên khối) và dùng đúng
      ``load_documents`` + ``chunk_documents`` của Task 4. Nếu lexical search
      chấm điểm trên document nguyên khối trong khi semantic search chấm trên
      chunk thì RRF ở Task 7/9 sẽ fuse hai đơn vị khác nhau, kết quả lệch.
    - Tokenizer tiếng Việt: lower-case + regex Unicode ``\\w+`` giữ nguyên chữ có
      dấu và tách được số/đơn vị ("6.000.000" → "6", "000", "000"). Không dùng
      ``str.split()`` thô vì dấu câu dính vào token ("đỏ," ≠ "đỏ").
"""

from __future__ import annotations

import re

from .task4_chunking_indexing import (
    chunk_documents,
    chunk_embedding_text,
    load_documents,
)

TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_BM25_INDEX = None


def tokenize(text: str) -> list[str]:
    """Tokenizer tối thiểu nhưng ổn định cho tiếng Việt có dấu."""
    return TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    from rank_bm25 import BM25Okapi

    if not corpus:
        raise ValueError("Cannot build a BM25 index from an empty corpus")
    # Index trên chunk_embedding_text để BM25 nhìn thấy cùng ngữ cảnh (tiêu đề văn
    # bản + Điều) mà semantic search nhìn thấy — nếu không, chunk "b) Không chấp
    # hành hiệu lệnh của đèn tín hiệu giao thông;" không khớp được từ "xe máy".
    return BM25Okapi([tokenize(chunk_embedding_text(item)) for item in corpus])


def load_corpus() -> list[dict]:
    """Load chunks giống hệt Task 4 (lazy, cache trong process)."""
    global CORPUS
    if not CORPUS:
        CORPUS = chunk_documents(load_documents())
    return CORPUS


def get_bm25_index():
    """Build index một lần rồi tái sử dụng cho mọi truy vấn trong process."""
    global _BM25_INDEX
    if _BM25_INDEX is None:
        _BM25_INDEX = build_bm25_index(load_corpus())
    return _BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending. Chỉ giữ kết quả có score > 0.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    tokens = tokenize(query)
    if not tokens:
        return []

    corpus = load_corpus()
    scores = get_bm25_index().get_scores(tokens)
    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)

    results: list[dict] = []
    for index in ranked[:top_k]:
        score = float(scores[index])
        if score <= 0:
            break  # ranked giảm dần nên phần còn lại cũng ≤ 0
        results.append(
            {
                "content": corpus[index]["content"],
                "score": score,
                "metadata": dict(corpus[index]["metadata"]),
            }
        )
    return results


if __name__ == "__main__":
    for result in lexical_search("nồng độ cồn xe máy", top_k=5):
        source = result["metadata"].get("source", "?")
        print(f"[{result['score']:.3f}] {source} :: {result['content'][:100]}...")
