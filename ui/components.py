"""Streamlit rendering helpers."""
from __future__ import annotations

from html import escape
from typing import Any

import streamlit as st
from markdown_it import MarkdownIt


_MARKDOWN = MarkdownIt("commonmark", {"html": False, "linkify": True})


def _safe(value: Any) -> str:
    return escape(str(value or ""), quote=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="main-header">
            <h1>🌿 Trợ Lý Luật Giao Thông Đường Bộ</h1>
            <p>🌍 Tra Cứu Pháp Luật Giao Thông Việt Nam Bằng AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 2rem; color: #44403c;">
            <h3 style="color: #047857; font-size: 2rem; margin-bottom: 1rem;">
                🌱 Chào Mừng Đến Với Trợ Lý Luật Giao Thông!
            </h3>
            <p style="font-size: 1.1rem; color: #78716c; max-width: 700px; margin: 0 auto 2rem;">
                Người đồng hành AI giúp bạn tra cứu quy định, mức xử phạt và hướng dẫn
                chấp hành pháp luật giao thông đường bộ tại Việt Nam. ♻️
            </p>
            <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap;">
                <span class="badge badge-green">🌿 Quy Tắc Giao Thông</span>
                <span class="badge badge-blue">📜 Tra Cứu Điều Luật</span>
                <span class="badge badge-green">♻️ Mức Xử Phạt</span>
                <span class="badge badge-blue">🏭 Hướng Dẫn Tuân Thủ</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_user_bubble(content: str) -> None:
    st.markdown(
        f"""
        <div class="user-message">
            <strong>👤 Bạn</strong><br>{_safe(content)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assistant_bubble(content: str, *, with_cursor: bool = False) -> None:
    cursor = "▌" if with_cursor else ""
    rendered_content = _MARKDOWN.render(content)
    st.markdown(
        f"""
        <div class="assistant-message">
            <strong>🌿 Trợ Lý</strong><br>{rendered_content}{cursor}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(documents: list[dict[str, Any]]) -> None:
    """Render a collapsible list of cited legal documents."""
    if not documents:
        return

    with st.expander("📚 Tài Liệu Pháp Lý Tham Khảo"):
        for doc in documents:
            meta = doc.get("metadata", {}) or {}
            title_parts: list[str] = []

            if meta.get("Chuong"):
                name = meta.get("Chuong_Name", "")
                title_parts.append(
                    f"📖 {_safe(meta['Chuong'])}: {_safe(name)}"
                    if name else f"📖 {_safe(meta['Chuong'])}"
                )

            if meta.get("Muc"):
                name = meta.get("Muc_Name", "")
                title_parts.append(
                    f"📑 {_safe(meta['Muc'])}: {_safe(name)}"
                    if name else f"📑 {_safe(meta['Muc'])}"
                )

            if meta.get("Dieu"):
                dieu = meta["Dieu"]
                dieu_name = meta.get("Dieu_Name", "")
                prefix = "" if str(dieu).startswith("Điều") else "Điều "
                title_parts.append(
                    f"📄 {prefix}{_safe(dieu)}: {_safe(dieu_name)}"
                    if dieu_name else f"📄 {prefix}{_safe(dieu)}"
                )

            if not title_parts:
                document_type_labels = {
                    "law": "Luật",
                    "decree": "Nghị định",
                    "circular": "Thông tư",
                    "resolution": "Nghị quyết",
                }
                document_number = meta.get("document_number")
                document_type = document_type_labels.get(meta.get("document_type"), "")
                if document_number:
                    label = f"{document_type} {document_number}".strip()
                else:
                    label = meta.get("title") or meta.get("source", "Nguồn tài liệu")
                title_parts.append(f"📄 {_safe(label)}")

            if meta.get("section"):
                title_parts.append(f"Mục: {_safe(meta['section'])}")

            title_html = "<br>".join(title_parts)
            preview = _safe(doc.get("page_content", doc.get("content", ""))[:300])
            details = []
            if meta.get("effective_date"):
                details.append(f"Hiệu lực từ {_safe(meta['effective_date'])}")
            if isinstance(doc.get("score"), (int, float)):
                details.append(f"Độ liên quan {doc['score']:.4f}")
            if doc.get("source"):
                details.append(f"Kênh {_safe(doc['source'])}")
            details_html = (
                f'<div class="source-details">{" · ".join(details)}</div>'
                if details else ""
            )
            st.markdown(
                f"""
                <div class="source-box">
                    <div class="source-title">{title_html}</div>
                    {details_html}
                    <div style="margin-top: 0.5rem; color: #44403c;">{preview}...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_status(message: str) -> None:
    st.markdown(
        f"""
        <div class="status-message">
            <span class="loading-spinner">🔄</span> {message}
        </div>
        """,
        unsafe_allow_html=True,
    )
