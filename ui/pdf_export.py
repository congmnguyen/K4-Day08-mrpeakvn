"""Generate a PDF transcript of a chat session."""
from __future__ import annotations

from typing import Any

_EMOJI_STRIPS = (
    "🌱", "✅", "⚠️", "🌿", "🌍", "♻️", "📜", "🏭",
    "👤", "📚", "📄", "🎤", "🗣️", "▌", "📖", "📑", "🔄",
)


def _strip_emojis(text: str) -> str:
    for emoji in _EMOJI_STRIPS:
        text = text.replace(emoji, "")
    try:
        return text.encode("latin-1", "replace").decode("latin-1")
    except Exception:
        return text.encode("ascii", "ignore").decode("ascii")


def create_chat_pdf(messages: list[dict[str, Any]]) -> bytes:
    # Lazy import lets the Streamlit shell open before optional packages are installed.
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "", 16)
    pdf.cell(200, 10, "Road Traffic Legal Assistant - Chat Export", align="C")
    pdf.ln(10)

    for msg in messages:
        role = "Nguoi dung" if msg["role"] == "user" else "Tro ly"
        content = _strip_emojis(msg.get("content", ""))

        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(4, 120, 87)
        pdf.cell(0, 10, f"{role}:")
        pdf.ln()

        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(0, 8, content)
        pdf.ln(5)

    return bytes(pdf.output())
