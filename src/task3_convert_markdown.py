"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install "markitdown[pdf]"
    # Lưu ý: cần extra [pdf] để convert được file PDF. Chỉ "pip install markitdown"
    # (không có extra) sẽ báo MissingDependencyException khi convert PDF, dù JSON/DOCX
    # vẫn convert bình thường.

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
MIN_LEGAL_TEXT_CHARS = 1_000


def _front_matter(metadata: dict) -> str:
    """Serialize flat metadata thành YAML-compatible front matter an toàn."""
    lines = ["---"]
    for key, value in metadata.items():
        if value is not None:
            lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.extend(["---", ""])
    return "\n".join(lines)


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    manifest_path = legal_dir / "sources.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = {item["filename"]: item for item in manifest}

    for filepath in legal_dir.iterdir():
        # .doc là bản Công báo — MarkItDown không đọc được OLE2, đã có
        # convert_gazette_docs() xử lý bằng LibreOffice.
        if filepath.suffix.lower() in (".pdf", ".docx"):
            print(f"Converting: {filepath.name}")
            result = md.convert(str(filepath))
            output_path = output_dir / f"{filepath.stem}.md"
            extracted_text = result.text_content.strip()
            if len(extracted_text) < MIN_LEGAL_TEXT_CHARS:
                output_path.unlink(missing_ok=True)
                print(
                    f"  ⚠ Skipped scanned/empty text: {filepath.name} "
                    f"({len(extracted_text)} chars)"
                )
                continue
            source = sources.get(filepath.name, {})
            metadata = {
                key: source.get(key)
                for key in (
                    "document_number",
                    "title",
                    "document_type",
                    "issuing_agency",
                    "effective_date",
                    "audience",
                    "source_page_url",
                    "download_url",
                    "retrieved_at",
                )
            }
            content = _front_matter(metadata) + extracted_text + "\n"
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_clean_legal_texts():
    """Convert các bản toàn văn HTML đã crawl, dùng thay cho PDF scan."""
    legal_text_dir = LANDING_DIR / "legal_text"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    if not legal_text_dir.exists():
        return

    for filepath in sorted(legal_text_dir.glob("*.json")):
        print(f"Converting clean text: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        body = data.get("content_markdown", "").strip()
        if len(body) < MIN_LEGAL_TEXT_CHARS:
            raise ValueError(f"Clean legal text too short: {filepath}")
        metadata = {
            key: data.get(key)
            for key in (
                "document_number",
                "title",
                "document_type",
                "issuing_agency",
                "effective_date",
                "audience",
                "source_url",
                "publisher",
                "retrieved_at",
                "corpus_snapshot_date",
            )
        }
        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(
            _front_matter(metadata) + f"# {data['title']}\n\n" + body + "\n",
            encoding="utf-8",
        )
        print(f"  ✓ Saved: {output_path} ({len(body):,} chars)")


def _find_soffice() -> str | None:
    """LibreOffice là công cụ duy nhất có sẵn để đọc .doc (OLE2) chuẩn xác."""
    for name in ("soffice", "libreoffice"):
        path = shutil.which(name)
        if path:
            return path
    return None


def doc_to_text(doc_path: Path) -> str:
    """Convert legacy .doc sang plain text UTF-8 bằng LibreOffice headless."""
    soffice = _find_soffice()
    if soffice is None:
        raise RuntimeError(
            "Cần LibreOffice (soffice) để convert .doc Công báo. "
            "Cài: sudo apt install libreoffice-writer"
        )
    with tempfile.TemporaryDirectory() as workdir:
        subprocess.run(
            [
                soffice,
                "--headless",
                "--convert-to",
                "txt:Text (encoded):UTF8",
                "--outdir",
                workdir,
                str(doc_path),
            ],
            check=True,
            capture_output=True,
            timeout=600,
        )
        produced = Path(workdir) / f"{doc_path.stem}.txt"
        if not produced.is_file():
            raise RuntimeError(f"LibreOffice không tạo được text từ {doc_path.name}")
        return produced.read_text(encoding="utf-8")


def _gazette_text_to_markdown(raw_text: str) -> str:
    """Chuẩn hoá text Công báo thành Markdown có heading Chương/Điều.

    Heading giúp splitter ở Task 4 và fallback structure-aware ở Task 8 cắt
    đúng ranh giới điều luật thay vì cắt giữa câu.
    """
    lines = [line.strip() for line in raw_text.replace("﻿", "").splitlines()]
    blocks: list[str] = []
    for line in lines:
        if not line:
            continue
        if re.match(r"^Chương\s+[IVXLC]+", line):
            blocks.append(f"## {line}")
        elif re.match(r"^Mục\s+\d+", line):
            blocks.append(f"### {line}")
        elif re.match(r"^Điều\s+\d+[\.\s]", line):
            blocks.append(f"### {line}")
        else:
            blocks.append(line)
    return "\n\n".join(blocks)


def convert_gazette_docs():
    """Convert bản .doc chính thức từ Công báo — nguồn text sạch cho 3 thông tư scan."""
    gazette_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = gazette_dir / "gazette_sources.json"
    if not manifest_path.is_file():
        print("  ⚠ Chưa có gazette_sources.json — chạy Task 1 trước.")
        return

    sources = {
        item["filename"]: item
        for item in json.loads(manifest_path.read_text(encoding="utf-8"))
    }
    for filepath in sorted(gazette_dir.glob("*.doc")):
        print(f"Converting gazette: {filepath.name}")
        body = _gazette_text_to_markdown(doc_to_text(filepath))
        if len(body) < MIN_LEGAL_TEXT_CHARS:
            raise ValueError(f"Gazette text quá ngắn ({len(body)} chars): {filepath}")
        source = sources.get(filepath.name, {})
        metadata = {
            key: source.get(key)
            for key in (
                "document_number",
                "title",
                "document_type",
                "issuing_agency",
                "issued_date",
                "effective_date",
                "audience",
                "gazette_page_url",
                "publisher",
                "legal_status",
                "amendment_note",
                "retrieved_at",
            )
        }
        output_path = output_dir / f"{filepath.stem}.md"
        title = source.get("title", filepath.stem)
        output_path.write_text(
            _front_matter(metadata) + f"# {title}\n\n" + body + "\n",
            encoding="utf-8",
        )
        print(f"  ✓ Saved: {output_path} ({len(body):,} chars)")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"
            metadata = {
                key: data.get(key)
                for key in (
                    "title",
                    "url",
                    "published_at",
                    "date_crawled",
                    "publisher",
                    "category",
                    "audience",
                )
            }
            content = (
                _front_matter(metadata)
                + f"# {data.get('title', 'Unknown')}\n\n"
                + data.get("content_markdown", "").strip()
                + "\n"
            )
            output_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    # Thứ tự quan trọng: convert_legal_docs() xoá output của PDF scan (0 ký tự),
    # convert_gazette_docs() ghi đè cùng tên file bằng bản text sạch từ Công báo.
    convert_legal_docs()
    convert_gazette_docs()
    convert_clean_legal_texts()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
