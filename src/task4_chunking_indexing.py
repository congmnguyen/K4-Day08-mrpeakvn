"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (ChromaDB khuyến cáo — đơn giản, local, không cần Docker)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options (chọn 1, cân nhắc đánh đổi cài đặt nặng vs cần API key):
    - sentence-transformers/all-MiniLM-L6-v2 hoặc BAAI/bge-m3 — chạy local, không
      cần API key, nhưng cài nặng (~1-2GB vì kéo theo torch)
    - Google models/text-embedding-004 (768 dim) — nhẹ, cần GEMINI_API_KEY
    - OpenAI text-embedding-3-small (1536 dim) — nhẹ, cần OPENAI_API_KEY
    Gợi ý: đọc EMBEDDING_PROVIDER từ .env (os.getenv("EMBEDDING_PROVIDER", "openai"))
    để cả nhóm có thể đổi provider mà không sửa code — nhớ đổi provider phải xoá
    chroma_db/ cũ và reindex vì dimension khác nhau (1024/768/1536) không tương thích ngược.

Vector store options:
    - ChromaDB (khuyến cáo: đơn giản, local persistent, không cần Docker)
    - Weaviate (hỗ trợ hybrid search built-in, cần Docker/Cloud)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters openai chromadb

Lưu ý quan trọng: nếu sau này đổi corpus (đổi chủ đề, thêm/bớt tài liệu), phải XÓA
chroma_db/ cũ trước khi reindex — nếu không, chunk cũ và mới sẽ tồn tại lẫn lộn
trong cùng collection, retrieval sẽ trả về kết quả rác từ dữ liệu cũ.
"""

import hashlib
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

# Nạp .env trước khi đọc bất kỳ biến môi trường nào ở module level, nếu không
# EMBEDDING_MODEL và OpenAI() chỉ hoạt động khi key được export sẵn trong shell.
load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# TODO: Chọn chunking strategy và giải thích vì sao
# 800 ký tự thường chứa trọn một khoản/nhóm điểm của văn bản pháp luật; overlap
# 120 ký tự giữ ngữ cảnh khi câu hoặc khoản nằm đúng ranh giới chunk.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# TODO: Chọn embedding model và giải thích
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM = 1536
EMBEDDING_BATCH_SIZE = 128

# TODO: Chọn vector store
VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "traffic_law_vn_docs"

# Heading kiểu "## Điều 7. Xử phạt..." do Task 3 sinh ra cho cả bản crawl HTML
# lẫn bản .doc Công báo.
HEADING_PATTERN = re.compile(r"^#{1,4}\s+(.*)$")


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        raw_content = md_file.read_text(encoding="utf-8").strip()
        metadata: dict = {
            "source": md_file.name,
            "path": str(md_file.relative_to(STANDARDIZED_DIR)),
            "type": "legal" if "legal" in md_file.parts else "news",
        }
        content = raw_content
        if raw_content.startswith("---\n"):
            _, front_matter, content = raw_content.split("---", 2)
            for line in front_matter.strip().splitlines():
                key, separator, value = line.partition(":")
                if separator and value.strip():
                    try:
                        parsed = json.loads(value.strip())
                    except json.JSONDecodeError:
                        parsed = value.strip()
                    if parsed is not None and isinstance(parsed, (str, int, float, bool)):
                        metadata[key.strip()] = parsed
        content = content.strip()
        if content:
            documents.append({"content": content, "metadata": metadata})
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Regex separators (keep_separator mặc định True nên ranh giới nằm ở ĐẦU
        # chunk sau). Ưu tiên cắt ở heading Điều, rồi ở đầu mỗi khoản "N. Phạt
        # tiền từ ..." — nhờ vậy chunk chứa danh sách hành vi vi phạm luôn mang
        # theo mức tiền phạt của khoản đó thay vì để mức phạt rơi sang chunk trước.
        separators=[
            r"\n#{1,4} Điều ",
            r"\n#{1,4} ",
            r"\n\d+\. Phạt tiền",
            r"\n\n",
            r"\n",
            r"\. ",
            r" ",
            r"",
        ],
        is_separator_regex=True,
        length_function=len,
    )
    chunks = []
    for document in documents:
        heading = ""
        for index, chunk_text in enumerate(splitter.split_text(document["content"])):
            # Contextual chunking: chunk nằm giữa một Điều thường chỉ còn "b) Không
            # chấp hành hiệu lệnh của đèn tín hiệu giao thông;" — mất hoàn toàn
            # thông tin "xe mô tô" và "phạt bao nhiêu". Ta gắn heading Điều gần
            # nhất vào metadata để embedding và BM25 nhìn thấy ngữ cảnh đó, trong
            # khi 'content' giữ nguyên để không phá giới hạn CHUNK_SIZE.
            headings_here = [
                match.group(1).strip()
                for match in (
                    HEADING_PATTERN.match(line) for line in chunk_text.splitlines()
                )
                if match
            ]
            # Chunk có heading riêng → dùng heading đầu tiên của chính nó;
            # chunk nằm giữa thân một Điều → thừa kế heading gần nhất phía trước.
            metadata = {
                **document["metadata"],
                "chunk_index": index,
                "heading": headings_here[0] if headings_here else heading,
            }
            chunks.append({"content": chunk_text, "metadata": metadata})
            if headings_here:
                heading = headings_here[-1]
    return chunks


def chunk_embedding_text(chunk: dict) -> str:
    """Text dùng để embed / đánh chỉ mục BM25 = ngữ cảnh văn bản + nội dung chunk."""
    metadata = chunk.get("metadata") or {}
    context = " — ".join(
        str(value)
        for value in (
            metadata.get("title"),
            metadata.get("document_number"),
            metadata.get("heading"),
        )
        if value
    )
    return f"{context}\n\n{chunk['content']}" if context else chunk["content"]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed một batch text bằng đúng model/dimension dùng khi index.

    Task 5 gọi hàm này để query vector luôn cùng không gian với vector đã index.
    """
    from openai import OpenAI

    if not texts:
        return []
    if not os.getenv("OPENAI_API_KEY"):
        # Fail sớm với thông điệp rõ ràng thay vì để OpenAI SDK raise sâu trong
        # thread của Task 9 — cả pipeline dựa trên embedding của provider này.
        raise RuntimeError(
            "Thiếu OPENAI_API_KEY. Toàn bộ Task 4/5/9/10 dùng OpenAI embeddings; "
            "tạo .env từ .env.example và điền key trước khi chạy pytest."
        )
    client = OpenAI()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
            dimensions=EMBEDDING_DIM,
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(batch):
            raise RuntimeError("OpenAI returned an unexpected embedding count")
        vectors.extend(item.embedding for item in ordered)
    return vectors


def get_collection(build_if_missing: bool = True):
    """Mở collection đã index.

    ``chroma_db/`` nằm trong .gitignore nên clone mới hoặc deploy (Hugging Face
    Space) không có index — mặc định tự build một lần thay vì để mọi query chết
    vì collection không tồn tại. Đặt ``build_if_missing=False`` nếu muốn lỗi rõ.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        if not build_if_missing:
            raise
        print(
            f"ℹ Chưa có collection '{COLLECTION_NAME}' — build index lần đầu "
            "từ data/standardized/ (tốn vài chục giây và gọi OpenAI embeddings)."
        )
        run_pipeline()
        return client.get_collection(COLLECTION_NAME)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks
    vectors = embed_texts([chunk_embedding_text(chunk) for chunk in chunks])
    for chunk, vector in zip(chunks, vectors, strict=True):
        chunk["embedding"] = vector
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import chromadb

    if any("embedding" not in chunk for chunk in chunks):
        raise ValueError("All chunks must be embedded before indexing")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception as error:
        # Chroma thay đổi class NotFoundError giữa các phiên bản; chỉ bỏ qua đúng
        # trường hợp collection chưa tồn tại, không che các lỗi storage khác.
        if "does not exist" not in str(error) and "not found" not in str(error).lower():
            raise
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )
    identifiers = [
        hashlib.sha256(
            f"{chunk['metadata']['path']}:{chunk['metadata']['chunk_index']}".encode()
        ).hexdigest()
        for chunk in chunks
    ]
    collection.upsert(
        ids=identifiers,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    return collection


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
