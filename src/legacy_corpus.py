"""Kiểm kê hai kho văn bản giao thông cũ mà không index mù toàn bộ.

Script chỉ tạo manifest theo mặc định. Mọi tài liệu legacy đều được đánh dấu
``index_by_default=false`` cho đến khi hiệu lực pháp lý được xác minh thủ công.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


PROJECT_DIR = Path(__file__).parent.parent
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "legacy_inventory.json"
SUPPORTED_EXTENSIONS = {".doc", ".docx", ".pdf"}
EXCLUDE_PATTERNS = {
    "draft": re.compile(r"\bdu[ _-]*thao\b|\bdự[ _-]*thảo\b", re.IGNORECASE),
    "annex": re.compile(r"\bphu[ _-]*luc\b|\bphụ[ _-]*lục\b|[._ -]pl\d", re.IGNORECASE),
    "technical_regulation": re.compile(
        r"\bqcvn\b|\bquy[ _-]*chuan\b|\bquy[ _-]*chuẩn\b", re.IGNORECASE
    ),
}
FORMAT_PRIORITY = {".docx": 0, ".doc": 1, ".pdf": 2}


def _ascii_fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _document_key(filename: str) -> str:
    """Chuẩn hóa tên để ghép bản editable với bản ``VanBanGoc_*.pdf``."""
    stem = _ascii_fold(Path(filename).stem)
    stem = re.sub(r"^(vanbangoc|van-ban-goc)[ _.-]*", "", stem)
    return re.sub(r"[^a-z0-9]+", "", stem)


def _year(filename: str) -> int | None:
    years = [int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", filename)]
    return max(years) if years else None


def _exclusion_reason(filename: str) -> str | None:
    folded = _ascii_fold(filename)
    for reason, pattern in EXCLUDE_PATTERNS.items():
        if pattern.search(folded):
            return reason
    return None


def _safe_member(member: str) -> bool:
    path = PurePosixPath(member)
    return not path.is_absolute() and ".." not in path.parts


def build_inventory(archives: list[Path], minimum_year: int = 2020) -> dict:
    """Đọc metadata ZIP, phân loại và chọn một đại diện cho mỗi tên văn bản."""
    records: list[dict] = []
    for archive in archives:
        with zipfile.ZipFile(archive) as source:
            bad_member = source.testzip()
            if bad_member:
                raise ValueError(f"Corrupt member in {archive}: {bad_member}")
            for info in source.infolist():
                if info.is_dir():
                    continue
                member_path = PurePosixPath(info.filename)
                extension = Path(member_path.name).suffix.casefold()
                year = _year(member_path.name)
                exclusion = _exclusion_reason(member_path.name)
                eligible = (
                    _safe_member(info.filename)
                    and extension in SUPPORTED_EXTENSIONS
                    and exclusion is None
                    and year is not None
                    and year >= minimum_year
                )
                records.append(
                    {
                        "archive": str(archive.resolve()),
                        "member": info.filename,
                        "filename": member_path.name,
                        "extension": extension,
                        "size_bytes": info.file_size,
                        "crc32": f"{info.CRC:08x}",
                        "document_key": _document_key(member_path.name),
                        "year": year,
                        "exclusion_reason": exclusion,
                        "eligible_for_review": eligible,
                        "selected_representative": False,
                        "legal_status": "unverified",
                        "index_by_default": False,
                    }
                )

    # Cùng document_key: ưu tiên DOCX, rồi DOC, sau cùng PDF; tiếp theo bản lớn hơn.
    groups: dict[str, list[dict]] = {}
    for record in records:
        if record["eligible_for_review"]:
            groups.setdefault(record["document_key"], []).append(record)
    for candidates in groups.values():
        selected = min(
            candidates,
            key=lambda item: (FORMAT_PRIORITY[item["extension"]], -item["size_bytes"]),
        )
        selected["selected_representative"] = True

    reason_counts = Counter(
        record["exclusion_reason"] or "none" for record in records
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "minimum_year": minimum_year,
        "policy": {
            "purpose": "legacy corpus review only",
            "legal_status": "unverified",
            "index_by_default": False,
            "representative_priority": ["docx", "doc", "pdf"],
        },
        "summary": {
            "archives": len(archives),
            "files": len(records),
            "eligible_for_review": sum(item["eligible_for_review"] for item in records),
            "selected_representatives": sum(
                item["selected_representative"] for item in records
            ),
            "exclusion_reasons": dict(sorted(reason_counts.items())),
        },
        "documents": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--minimum-year", type=int, default=2020)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    missing = [str(path) for path in args.archives if not path.is_file()]
    if missing:
        parser.error(f"Archive not found: {', '.join(missing)}")

    inventory = build_inventory(args.archives, args.minimum_year)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = inventory["summary"]
    print(f"Inventory: {args.output}")
    print(
        f"Files={summary['files']}, eligible={summary['eligible_for_review']}, "
        f"representatives={summary['selected_representatives']}"
    )
    print("Legacy documents remain index_by_default=false until legal review.")


if __name__ == "__main__":
    main()
