"""
Task 8 — PageIndex Vectorless RAG.

Ý tưởng của PageIndex: retrieval KHÔNG cần vector store — thay vì so khoảng cách
embedding, ta khai thác CẤU TRÚC tài liệu (cây mục lục: Chương → Mục → Điều) và
chọn ra node phù hợp nhất với câu hỏi.

Hai backend:

1. ``cloud`` — PageIndex API (https://pageindex.ai), chỉ chạy khi có
   ``PAGEINDEX_API_KEY``. Code viết theo signature THẬT của SDK ``pageindex``
   đang cài trong .venv (``submit_document(file_path)``, ``submit_query(doc_id,
   query)``, ``get_retrieval(retrieval_id)``). Lưu ý: repo này chưa có API key
   nên nhánh cloud CHƯA được chạy thật; mọi field đều parse phòng thủ bằng
   ``.get()`` và kết quả luôn gắn ``metadata.backend = "pageindex_api"`` để phân
   biệt với fallback local.

2. ``local_structure`` — fallback mặc định, không gọi mạng. Dựng cây heading từ
   ``data/standardized/*.md`` rồi chấm điểm theo cấp: tài liệu → Điều/Mục. Đây
   là vectorless thật (không embedding), và khác BM25 ở Task 6 vì đơn vị trả về
   là NGUYÊN một Điều luật chứ không phải chunk 800 ký tự cắt cứng.

Cả hai backend đều trả ``source="pageindex"`` để Task 9 nhận diện nhánh fallback.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
DOC_ID_CACHE = Path(__file__).parent.parent / "pageindex_doc_ids.json"

HEADING_PATTERN = re.compile(r"^(#{1,4})\s+(.*)$")
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
MAX_SECTION_CHARS = 2_000

# Ngưỡng tối thiểu cho backend local. Chỉ cần trùng MỘT token phổ biến ("xe",
# "giá") là section đã có điểm > 0, nên nếu không lọc thì câu lạc đề vẫn nhận
# được "evidence" và Task 10 sẽ sinh câu trả lời từ ngữ cảnh vô nghĩa — đúng
# thứ mà nhánh fallback lẽ ra phải chặn.
#
# CALIBRATION (đo thật trên corpus hiện tại):
#   Câu đúng chủ đề → 0.681 … 1.000
#     "Khi xảy ra tai nạn giao thông cần làm gì?"        0.926
#     "Hồ sơ đăng ký xe gồm những giấy tờ gì?"           0.681
#     "Cảnh sát giao thông tuần tra kiểm soát thế nào?"  0.844
#     "Đấu giá biển số xe ô tô"                          0.796
#   Câu lạc đề     → không có kết quả … 0.270
#     "giá cổ phiếu VN30 hôm nay"                        0.185
#     "How do I return an item on Shopee?"               0.270
#     "cách nấu phở bò ngon"                             0.178
#   → chọn ngưỡng giữa hai nhóm: 0.45
MIN_LOCAL_SCORE = 0.45

# Từ dừng tiếng Việt hay gặp trong câu hỏi — không mang thông tin phân biệt.
STOPWORDS = {
    "là", "của", "và", "có", "cho", "khi", "gì", "bao", "nhiêu", "nào", "được",
    "phải", "thì", "các", "những", "một", "trong", "về", "với", "tôi", "cần",
    "làm", "sao", "ai", "ở", "đâu", "hay", "hoặc", "bị", "sẽ", "này", "đó",
}

_SECTION_CACHE: list[dict] | None = None


# =============================================================================
# Local structure-aware backend (mặc định)
# =============================================================================

def tokenize(text: str) -> set[str]:
    """Token set đã bỏ stopword — dùng để đo overlap giữa query và tiêu đề/nội dung."""
    return {
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOPWORDS and len(token) > 1
    }


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Tách front matter (do Task 3 sinh) khỏi phần thân markdown."""
    if not raw.startswith("---\n"):
        return {}, raw
    _, front_matter, body = raw.split("---", 2)
    metadata: dict = {}
    for line in front_matter.strip().splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        try:
            metadata[key.strip()] = json.loads(value.strip())
        except json.JSONDecodeError:
            metadata[key.strip()] = value.strip()
    return metadata, body


def build_document_tree(markdown: str) -> list[dict]:
    """Dựng cây heading → list node phẳng, mỗi node giữ đường dẫn tiêu đề cha.

    Node là đơn vị retrieval của backend local: một Điều luật trọn vẹn.
    """
    nodes: list[dict] = []
    breadcrumb: dict[int, str] = {}
    current: dict | None = None

    for line in markdown.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            breadcrumb = {k: v for k, v in breadcrumb.items() if k < level}
            breadcrumb[level] = title
            current = {
                "title": title,
                "level": level,
                "title_path": " > ".join(
                    breadcrumb[key] for key in sorted(breadcrumb)
                ),
                "lines": [],
            }
            nodes.append(current)
        elif current is not None:
            current["lines"].append(line)

    for node in nodes:
        node["content"] = (
            node["title_path"] + "\n\n" + "\n".join(node.pop("lines")).strip()
        ).strip()
    return [node for node in nodes if node["content"]]


def split_section(content: str) -> list[str]:
    """Cắt một Điều quá dài thành các phần ≤ MAX_SECTION_CHARS tại ranh giới khoản.

    Điều 7 của Nghị định 168 dài hơn 13.000 ký tự; nếu chỉ cắt cụt thì phần cuối
    (nơi chứa nhiều khoản phạt) không bao giờ đến được Task 10.
    """
    if len(content) <= MAX_SECTION_CHARS:
        return [content]

    parts: list[str] = []
    buffer = ""
    for paragraph in content.split("\n\n"):
        candidate = f"{buffer}\n\n{paragraph}" if buffer else paragraph
        if len(candidate) > MAX_SECTION_CHARS and buffer:
            parts.append(buffer)
            buffer = paragraph
        else:
            buffer = candidate
        # Một đoạn đơn lẻ vẫn có thể vượt hạn mức → cắt cứng phần thừa.
        while len(buffer) > MAX_SECTION_CHARS:
            parts.append(buffer[:MAX_SECTION_CHARS])
            buffer = buffer[MAX_SECTION_CHARS:]
    if buffer:
        parts.append(buffer)
    return parts


def load_sections() -> list[dict]:
    """Load toàn bộ node cấu trúc từ data/standardized/ (cache trong process)."""
    global _SECTION_CACHE
    if _SECTION_CACHE is not None:
        return _SECTION_CACHE

    sections: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        metadata, body = parse_front_matter(
            md_file.read_text(encoding="utf-8").strip()
        )
        document_context = " ".join(
            str(metadata.get(key, ""))
            for key in ("title", "document_number", "document_type", "audience")
        )
        document_tokens = tokenize(f"{document_context} {md_file.stem}")
        for node in build_document_tree(body):
            parts = split_section(node["content"])
            for part_index, part in enumerate(parts):
                sections.append(
                    {
                        # Chấm điểm và trả về CÙNG một đoạn text: nếu chấm trên
                        # cả Điều dài rồi chỉ trả 2.000 ký tự đầu, Task 10 có thể
                        # nhận context không chứa khoản đã làm nó được chọn.
                        "content": part,
                        "title": node["title"],
                        "title_path": node["title_path"],
                        "level": node["level"],
                        "document_tokens": document_tokens,
                        "title_tokens": tokenize(node["title_path"]),
                        "content_tokens": tokenize(part),
                        "metadata": {
                            "source": md_file.name,
                            "path": str(md_file.relative_to(STANDARDIZED_DIR)),
                            "type": "legal" if "legal" in md_file.parts else "news",
                            "section": node["title"],
                            "section_path": node["title_path"],
                            "section_part": part_index,
                            "section_parts": len(parts),
                            # Giữ nguyên metadata pháp lý để Task 10 cite đúng
                            # "Thông tư 72/2024/TT-BCA" kèm mốc hiệu lực.
                            "document_number": metadata.get("document_number"),
                            "document_type": metadata.get("document_type"),
                            "issuing_agency": metadata.get("issuing_agency"),
                            "effective_date": metadata.get("effective_date"),
                            "backend": "local_structure",
                        },
                    }
                )
    _SECTION_CACHE = sections
    return sections


def local_structure_search(query: str, top_k: int = 5) -> list[dict]:
    """Vectorless retrieval theo cấu trúc: tài liệu → tiêu đề Điều → nội dung.

    Điểm = 3×overlap(tiêu đề) + 1×overlap(nội dung) + 0.5×overlap(metadata tài
    liệu), chuẩn hoá theo số token của query nên nằm trong khoảng [0, ~4.5];
    chia cho 4.5 để đưa về [0, 1] cho dễ đọc cùng thang với các task khác.
    Kết quả dưới ``MIN_LOCAL_SCORE`` bị loại để câu lạc đề trả về list rỗng.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scored: list[tuple[float, dict]] = []
    for section in load_sections():
        title_hits = len(query_tokens & section["title_tokens"])
        content_hits = len(query_tokens & section["content_tokens"])
        document_hits = len(query_tokens & section["document_tokens"])
        if not (title_hits or content_hits):
            continue
        raw = (
            3.0 * title_hits + 1.0 * content_hits + 0.5 * document_hits
        ) / len(query_tokens)
        score = min(raw / 4.5, 1.0)
        if score < MIN_LOCAL_SCORE:
            continue
        scored.append((score, section))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        {
            "content": section["content"],
            "score": round(score, 4),
            "metadata": dict(section["metadata"]),
            "source": "pageindex",
        }
        for score, section in scored[:top_k]
    ]


# =============================================================================
# PageIndex cloud backend (chỉ khi có API key)
# =============================================================================

UNICODE_FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
)


def unicode_font_path() -> Path:
    """TTF có glyph tiếng Việt để render PDF cho PageIndex."""
    for candidate in UNICODE_FONT_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "Không tìm thấy font Unicode để render PDF tiếng Việt. "
        "Cài: sudo apt install fonts-dejavu-core"
    )


def upload_documents() -> dict[str, str]:
    """Upload markdown (đã convert sang PDF) lên PageIndex, cache doc_id ra file.

    PageIndex nhận PDF nên ta render markdown sang PDF tối giản bằng fpdf2.
    """
    if not PAGEINDEX_API_KEY:
        raise RuntimeError(
            "upload_documents cần PAGEINDEX_API_KEY. Không có key thì "
            "pageindex_search() tự dùng backend local_structure."
        )
    from fpdf import FPDF
    from pageindex.client import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    pdf_dir = Path(__file__).parent.parent / "pageindex_pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    doc_ids: dict[str, str] = {}
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        _, body = parse_front_matter(md_file.read_text(encoding="utf-8"))
        pdf_path = pdf_dir / f"{md_file.stem}.pdf"
        document = FPDF()
        document.add_page()
        document.set_auto_page_break(auto=True, margin=15)
        # Font core của fpdf2 (Helvetica) là Latin-1, không encode được "ậ", "ữ"…
        # → phải nạp TTF Unicode, nếu không multi_cell raise khi gặp tiếng Việt.
        document.add_font("DejaVu", "", str(unicode_font_path()))
        document.set_font("DejaVu", size=10)
        document.multi_cell(0, 5, body)
        document.output(str(pdf_path))

        response = client.submit_document(str(pdf_path))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex không trả doc_id cho {md_file.name}")
        doc_ids[md_file.name] = doc_id
        print(f"  ✓ Uploaded: {md_file.name} -> {doc_id}")

    DOC_ID_CACHE.write_text(
        json.dumps(doc_ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return doc_ids


def _pageindex_api_search(query: str, top_k: int) -> list[dict]:
    """Query PageIndex API. Parse phòng thủ vì response schema có thể đổi."""
    import time

    from pageindex.client import PageIndexClient

    if not DOC_ID_CACHE.is_file():
        raise RuntimeError("Chưa có pageindex_doc_ids.json — chạy upload_documents().")

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    doc_ids = json.loads(DOC_ID_CACHE.read_text(encoding="utf-8"))
    results: list[dict] = []

    for document_order, (source_name, doc_id) in enumerate(doc_ids.items()):
        submitted = client.submit_query(doc_id=doc_id, query=query)
        retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
        if not retrieval_id:
            continue

        retrieval: dict = {}
        for _ in range(30):
            retrieval = client.get_retrieval(retrieval_id)
            if retrieval.get("status") in {"completed", "failed", "error"}:
                break
            time.sleep(2)
        if retrieval.get("status") != "completed":
            continue

        for rank, node in enumerate(retrieval.get("retrieved_nodes", []), 1):
            for group in node.get("relevant_contents", []) or []:
                for item in group or []:
                    content = (item or {}).get("relevant_content", "")
                    if not content:
                        continue
                    results.append(
                        {
                            "content": content,
                            # PageIndex không trả score → tự gán theo rank.
                            "score": round(1.0 / (1 + rank), 4),
                            "metadata": {
                                "source": source_name,
                                "section": (item or {}).get("section_title"),
                                "backend": "pageindex_api",
                            },
                            "source": "pageindex",
                            "_rank": rank,
                            "_document_order": document_order,
                        }
                    )

    # rank được đánh lại từ 1 cho MỖI document, nên sort thuần theo score sẽ ưu
    # tiên document nào được duyệt trước. Interleave round-robin theo rank để
    # kết quả hạng 1 của mọi document đứng trước hạng 2 của bất kỳ document nào.
    results.sort(key=lambda item: (item["_rank"], item["_document_order"]))
    for item in results:
        item.pop("_rank", None)
        item.pop("_document_order", None)
    return results[:top_k]


# =============================================================================
# Public interface
# =============================================================================

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval. Dùng làm fallback khi hybrid search không đủ tốt.

    Returns:
        List of {'content', 'score', 'metadata', 'source': 'pageindex'}.
        Query vô nghĩa/không khớp gì → trả list rỗng, KHÔNG raise.
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    if PAGEINDEX_API_KEY and DOC_ID_CACHE.is_file():
        try:
            return _pageindex_api_search(query, top_k)
        except Exception as error:  # noqa: BLE001 — fallback phải luôn sống sót
            print(f"  ⚠ PageIndex API lỗi ({error}); dùng backend local_structure")
    return local_structure_search(query, top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("ℹ Không có PAGEINDEX_API_KEY → chạy backend local_structure.\n")
    for question in (
        "Khi xảy ra tai nạn giao thông cần làm gì?",
        "Hồ sơ đăng ký xe gồm những giấy tờ gì?",
        "xyzabc123nonsense",
    ):
        print(f"Query: {question}")
        found = pageindex_search(question, top_k=3)
        if not found:
            print("  (không có kết quả)")
        for result in found:
            print(
                f"  [{result['score']:.3f}] {result['metadata'].get('source')} "
                f":: {result['metadata'].get('section')}"
            )
        print()
