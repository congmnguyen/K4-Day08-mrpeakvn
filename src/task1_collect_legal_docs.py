"""Thu thập văn bản pháp luật giao thông từ nguồn chính thức.

Corpus tập trung vào trật tự, an toàn giao thông đường bộ áp dụng từ năm 2025.
Mỗi tài liệu có URL nguồn và metadata trong ``sources.json`` để truy vết khi
chunk, index và sinh citation.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import certifi
import requests
from bs4 import BeautifulSoup


PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data" / "landing" / "legal"
MANIFEST_PATH = DATA_DIR / "sources.json"
TEXT_DIR = DATA_DIR.parent / "legal_text"
# Bản .doc Công báo nằm chung thư mục với PDF: đây mới là nguồn text sạch được
# commit vào repo, còn 4 PDF scan (~70MB) bị gitignore vì không trích xuất được
# ký tự nào và luôn tải lại được từ manifest.
GAZETTE_DIR = DATA_DIR
GAZETTE_MANIFEST_PATH = DATA_DIR / "gazette_sources.json"
USER_AGENT = "K4-Day08-RAG-Lab/1.0 (educational corpus; contact: local project)"
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# CDN g7.cdnchinhphu.vn chỉ gửi leaf certificate, thiếu intermediate nên chain
# không verify được bằng certifi thuần. Ta vendor đúng intermediate GlobalSign
# (root GlobalSign Root CA - R3 đã nằm trong certifi) và ghép thành bundle —
# verification vẫn bật đầy đủ, KHÔNG dùng verify=False.
EXTRA_CA_PATH = PROJECT_DIR / "certs" / "globalsign-rsa-ov-ssl-ca-2018.pem"
_CA_BUNDLE_CACHE: str | None = None

LEGAL_DOCUMENTS = [
    {
        "filename": "nghi-dinh-168-2024-nd-cp.pdf",
        "document_number": "168/2024/NĐ-CP",
        "title": "Xử phạt vi phạm hành chính về trật tự, an toàn giao thông đường bộ",
        "document_type": "decree",
        "issuing_agency": "Chính phủ",
        "effective_date": "2025-01-01",
        "audience": "all_road_users",
        "source_page_url": "https://vanban.chinhphu.vn/?classid=1&docid=212167&pageid=27160",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/01/168-nd-cp.signed.pdf",
    },
    {
        "filename": "thong-tu-72-2024-tt-bca.pdf",
        "document_number": "72/2024/TT-BCA",
        "title": "Quy trình điều tra, giải quyết tai nạn giao thông đường bộ",
        "document_type": "circular",
        "issuing_agency": "Bộ Công an",
        "effective_date": "2025-01-01",
        "audience": "traffic_enforcement",
        "source_page_url": "https://vanban.chinhphu.vn/?docid=211819&pageid=27160",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/tt72.pdf",
    },
    {
        "filename": "thong-tu-73-2024-tt-bca.pdf",
        "document_number": "73/2024/TT-BCA",
        "title": "Tuần tra, kiểm soát và xử lý vi phạm pháp luật về giao thông đường bộ",
        "document_type": "circular",
        "issuing_agency": "Bộ Công an",
        "effective_date": "2025-01-01",
        "audience": "traffic_enforcement",
        "source_page_url": "https://vanban.chinhphu.vn/?classid=1&docid=211864&orggroupid=4&pageid=27160",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/11/73-bca.pdf",
    },
    {
        "filename": "thong-tu-79-2024-tt-bca.pdf",
        "document_number": "79/2024/TT-BCA",
        "title": "Cấp, thu hồi chứng nhận đăng ký xe, biển số xe cơ giới, xe máy chuyên dùng",
        "document_type": "circular",
        "issuing_agency": "Bộ Công an",
        "effective_date": "2025-01-01",
        "audience": "vehicle_owner",
        "source_page_url": "https://vanban.chinhphu.vn/?docid=211945&pageid=27160",
        "download_url": "https://datafiles.chinhphu.vn/cpp/files/vbpq/2024/12/79-bca.signed.pdf",
    },
]

CLEAN_TEXT_SOURCES = [
    {
        "filename": "nghi-dinh-168-2024-nd-cp.json",
        "document_number": "168/2024/NĐ-CP",
        "title": "Xử phạt vi phạm hành chính về trật tự, an toàn giao thông đường bộ",
        "document_type": "decree",
        "issuing_agency": "Chính phủ",
        "effective_date": "2025-01-01",
        "audience": "all_road_users",
        "source_url": "https://xaydungchinhsach.chinhphu.vn/toan-van-nghi-dinh-168-2024-nd-cp-quy-dinh-xu-phat-vi-pham-hanh-chinh-ve-trat-tu-atgt-duong-bo-119241231164556785.htm",
        "corpus_snapshot_date": "2025-01-01",
    },
    {
        "filename": "luat-trat-tu-an-toan-giao-thong-duong-bo-36-2024-qh15.json",
        "document_number": "36/2024/QH15",
        "title": "Luật Trật tự, an toàn giao thông đường bộ",
        "document_type": "law",
        "issuing_agency": "Quốc hội",
        "effective_date": "2025-01-01",
        "audience": "all_road_users",
        "source_url": "https://xaydungchinhsach.chinhphu.vn/toan-van-luat-trat-tu-an-toan-giao-thong-duong-bo-119240909105718285.htm",
        "corpus_snapshot_date": "2025-01-01",
    },
]


# Bốn PDF ở trên đều là bản scan (MarkItDown trả 0 ký tự). Công báo điện tử của
# Văn phòng Chính phủ đăng đúng ba thông tư này dưới dạng .doc có text thật, nên
# đây là nguồn sạch chính thức để index thay cho OCR nhiễu.
GAZETTE_DOCUMENTS = [
    {
        "filename": "thong-tu-72-2024-tt-bca.doc",
        "document_number": "72/2024/TT-BCA",
        "title": "Quy trình điều tra, giải quyết tai nạn giao thông đường bộ của Cảnh sát giao thông",
        "document_type": "circular",
        "issuing_agency": "Bộ Công an",
        "issued_date": "2024-11-13",
        "effective_date": "2025-01-01",
        "audience": "traffic_enforcement",
        "gazette_page_url": "https://congbao.chinhphu.vn/van-ban/thong-tu-so-72-2024-tt-bca-43281/52935.htm",
        "publisher": "congbao.chinhphu.vn",
        "legal_status": "verified_original_text",
        "amendment_note": "Chưa xác minh được văn bản sửa đổi tính đến 2026-08-04.",
    },
    {
        "filename": "thong-tu-73-2024-tt-bca.doc",
        "document_number": "73/2024/TT-BCA",
        "title": "Công tác tuần tra, kiểm soát, xử lý vi phạm pháp luật về trật tự, an toàn giao thông đường bộ của Cảnh sát giao thông",
        "document_type": "circular",
        "issuing_agency": "Bộ Công an",
        "issued_date": "2024-11-15",
        "effective_date": "2025-01-01",
        "audience": "traffic_enforcement",
        "gazette_page_url": "https://congbao.chinhphu.vn/van-ban/thong-tu-so-73-2024-tt-bca-43336.htm",
        "publisher": "congbao.chinhphu.vn",
        "legal_status": "verified_original_text",
        "amendment_note": "Chưa xác minh được văn bản sửa đổi tính đến 2026-08-04.",
    },
    {
        "filename": "thong-tu-79-2024-tt-bca.doc",
        "document_number": "79/2024/TT-BCA",
        "title": "Cấp, thu hồi chứng nhận đăng ký xe, biển số xe cơ giới, xe máy chuyên dùng",
        "document_type": "circular",
        "issuing_agency": "Bộ Công an",
        "issued_date": "2024-11-15",
        "effective_date": "2025-01-01",
        "audience": "vehicle_owner",
        "gazette_page_url": "https://congbao.chinhphu.vn/van-ban/thong-tu-so-79-2024-tt-bca-43337.htm",
        "publisher": "congbao.chinhphu.vn",
        "legal_status": "verified_original_text_superseded_in_part",
        "amendment_note": (
            "Đã được sửa đổi, bổ sung bởi Thông tư 13/2025/TT-BCA (28/02/2025) và "
            "Thông tư 51/2025/TT-BCA. Bản index là văn bản gốc, chưa hợp nhất."
        ),
    },
]


def setup_directory() -> None:
    """Tạo thư mục đích nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    GAZETTE_DIR.mkdir(parents=True, exist_ok=True)


def ca_bundle() -> str:
    """Bundle = certifi + intermediate GlobalSign vendored trong repo."""
    global _CA_BUNDLE_CACHE
    if _CA_BUNDLE_CACHE and Path(_CA_BUNDLE_CACHE).is_file():
        return _CA_BUNDLE_CACHE
    if not EXTRA_CA_PATH.is_file():
        return certifi.where()
    handle = tempfile.NamedTemporaryFile(
        "w", suffix="-ca-bundle.pem", delete=False, encoding="utf-8"
    )
    with handle:
        handle.write(Path(certifi.where()).read_text(encoding="utf-8"))
        handle.write("\n")
        handle.write(EXTRA_CA_PATH.read_text(encoding="utf-8"))
    _CA_BUNDLE_CACHE = handle.name
    return _CA_BUNDLE_CACHE


def find_gazette_doc_url(page_url: str) -> str:
    """Lấy link .doc chính thức từ trang Công báo điện tử."""
    response = requests.get(
        page_url, headers={"User-Agent": BROWSER_USER_AGENT}, timeout=90
    )
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if "file_name=" in href and unquote(href).lower().endswith(".doc"):
            return href
    raise ValueError(f"Không tìm thấy link .doc trên trang Công báo: {page_url}")


def download_gazette_doc(document: dict) -> tuple[int, bool, str]:
    """Tải bản .doc chính thức từ Công báo; bỏ qua nếu đã có bản hợp lệ.

    Returns:
        ``(size_bytes, downloaded, doc_url)``. File .doc là OLE2 nên magic phải
        là ``\\xd0\\xcf\\x11\\xe0``; response HTML lỗi bị từ chối ngay.
    """
    destination = GAZETTE_DIR / document["filename"]
    doc_url = find_gazette_doc_url(document["gazette_page_url"])
    if destination.is_file() and destination.stat().st_size > 1024:
        with destination.open("rb") as source:
            if source.read(4) == b"\xd0\xcf\x11\xe0":
                return destination.stat().st_size, False, doc_url

    response = requests.get(
        doc_url,
        headers={"User-Agent": BROWSER_USER_AGENT},
        timeout=180,
        verify=ca_bundle(),
    )
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"\xd0\xcf\x11\xe0"):
        raise ValueError(
            f"Expected legacy Word (.doc) for {document['document_number']}, "
            f"got {response.headers.get('content-type', 'unknown')}"
        )
    if len(content) <= 1024:
        raise ValueError(f"Gazette .doc quá nhỏ: {document['document_number']}")
    destination.write_bytes(content)
    return len(content), True, doc_url


def collect_gazette_docs(retrieved_at: str) -> list[dict]:
    """Tải toàn bộ bản .doc Công báo và ghi manifest riêng."""
    previous = {
        record["filename"]: record
        for record in _read_manifest(GAZETTE_MANIFEST_PATH)
        if "filename" in record
    }
    manifest: list[dict] = []
    for index, document in enumerate(GAZETTE_DOCUMENTS, 1):
        print(f"[{index}/{len(GAZETTE_DOCUMENTS)}] {document['document_number']}")
        size_bytes, downloaded, doc_url = download_gazette_doc(document)
        prior = previous.get(document["filename"], {})
        record = {
            **document,
            "download_url": doc_url,
            "retrieved_at": retrieved_at
            if downloaded
            else prior.get("retrieved_at", retrieved_at),
            "downloaded": downloaded,
            "size_bytes": size_bytes,
        }
        if not downloaded:
            record["cache_reused_at"] = retrieved_at
        manifest.append(record)
        status = "downloaded" if downloaded else "cache reuse"
        print(f"  ✓ {document['filename']} ({size_bytes:,} bytes, {status})")

    GAZETTE_MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"✓ Manifest: {GAZETTE_MANIFEST_PATH}")
    return manifest


def download_file(url: str, destination: Path) -> int:
    """Tải một PDF và từ chối response HTML/error bị đặt nhầm đuôi file."""
    response = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=90,
    )
    response.raise_for_status()
    content = response.content
    if not content.startswith(b"%PDF-"):
        content_type = response.headers.get("content-type", "unknown")
        raise ValueError(f"Expected PDF from {url}, got {content_type}")
    if len(content) <= 1024:
        raise ValueError(f"Downloaded PDF is unexpectedly small: {url}")
    destination.write_bytes(content)
    return len(content)


def download_or_reuse_pdf(url: str, destination: Path) -> tuple[int, bool]:
    """Không tải lại file PDF hợp lệ đã có từ lần chạy trước.

    Returns:
        ``(size_bytes, downloaded)`` — ``downloaded=False`` nghĩa là bytes được
        tái sử dụng từ cache, khi đó caller phải giữ ``retrieved_at`` cũ trong
        manifest thay vì gán timestamp mới.
    """
    if destination.is_file() and destination.stat().st_size > 1024:
        with destination.open("rb") as source:
            if source.read(5) == b"%PDF-":
                return destination.stat().st_size, False
    return download_file(url, destination), True


def _read_manifest(path: Path) -> list[dict]:
    """Đọc manifest cũ; trả list rỗng nếu chưa có hoặc hỏng."""
    if not path.is_file():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return records if isinstance(records, list) else []


def load_manifest() -> dict[str, dict]:
    """Đọc manifest cũ để giữ nguyên provenance của file đã tải trước đó."""
    return {
        record["filename"]: record
        for record in _read_manifest(MANIFEST_PATH)
        if "filename" in record
    }


def collect_clean_text(source: dict, retrieved_at: str) -> int:
    """Lưu toàn văn HTML chính thức thành JSON sạch để index thay cho PDF scan."""
    response = requests.get(
        source["source_url"], headers={"User-Agent": USER_AGENT}, timeout=90
    )
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    soup = BeautifulSoup(response.text, "html.parser")
    for unwanted in soup.select("script, style, noscript, nav, footer, form, iframe"):
        unwanted.decompose()

    container = next(
        (
            soup.select_one(selector)
            for selector in (
                "article",
                ".article__body",
                ".article-body",
                ".detail-content",
                ".detail__content",
                ".content-detail",
                "main",
            )
            if soup.select_one(selector)
        ),
        None,
    )
    if container is None:
        raise ValueError(f"Could not locate legal text: {source['source_url']}")

    blocks: list[str] = []
    for element in container.find_all(["h2", "h3", "h4", "p", "li"]):
        text = " ".join(element.get_text(" ", strip=True).split())
        if len(text) < 10:
            continue
        if element.name in {"h2", "h3", "h4"}:
            blocks.append(f"## {text}")
        elif element.name == "li":
            blocks.append(f"- {text}")
        else:
            blocks.append(text)
    content_markdown = "\n\n".join(dict.fromkeys(blocks))
    if len(content_markdown) < 10_000:
        raise ValueError(
            f"Legal text too short ({len(content_markdown)} chars): {source['source_url']}"
        )

    record = {
        **source,
        "retrieved_at": retrieved_at,
        "publisher": response.url.split("/", 3)[2],
        "content_markdown": content_markdown,
    }
    destination = TEXT_DIR / source["filename"]
    destination.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(content_markdown)


def collect_all() -> list[dict]:
    """Tải toàn bộ văn bản và ghi manifest provenance."""
    setup_directory()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    previous = load_manifest()
    manifest: list[dict] = []

    for index, document in enumerate(LEGAL_DOCUMENTS, 1):
        destination = DATA_DIR / document["filename"]
        print(f"[{index}/{len(LEGAL_DOCUMENTS)}] {document['document_number']}")
        size_bytes, downloaded = download_or_reuse_pdf(
            document["download_url"], destination
        )
        prior = previous.get(document["filename"], {})
        record = {
            **document,
            # Chỉ lần thật sự tải mới cập nhật retrieved_at; cache hit giữ mốc cũ
            # và ghi nhận thời điểm tái sử dụng riêng để provenance trung thực.
            "retrieved_at": retrieved_at
            if downloaded
            else prior.get("retrieved_at", retrieved_at),
            "downloaded": downloaded,
            "size_bytes": size_bytes,
        }
        if not downloaded:
            record["cache_reused_at"] = retrieved_at
        manifest.append(record)
        status = "downloaded" if downloaded else "cache reuse"
        print(f"  ✓ {destination.name} ({size_bytes:,} bytes, {status})")

    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"✓ Manifest: {MANIFEST_PATH}")

    print("\n--- Official gazette .doc (text thật cho 3 thông tư scan) ---")
    collect_gazette_docs(retrieved_at)

    print("\n--- Clean legal text for indexing ---")
    for index, source in enumerate(CLEAN_TEXT_SOURCES, 1):
        text_length = collect_clean_text(source, retrieved_at)
        print(
            f"[{index}/{len(CLEAN_TEXT_SOURCES)}] {source['document_number']} "
            f"({text_length:,} chars)"
        )
    return manifest


if __name__ == "__main__":
    collect_all()
