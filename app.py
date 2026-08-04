"""Streamlit UI for the road traffic legal assistant.

The UI is kept intact; only the backend adapter targets this lab's
``generate_with_citation`` contract.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ui import (  # noqa: E402
    CSS,
    create_chat_pdf,
    render_assistant_bubble,
    render_header,
    render_sources,
    render_status,
    render_user_bubble,
    render_welcome,
)

st.set_page_config(
    page_title="Trợ Lý Luật Giao Thông Đường Bộ",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

SUGGESTIONS = (
    "Xe máy vượt đèn đỏ bị phạt bao nhiêu tiền?",
    "Người lái xe có bao nhiêu điểm giấy phép lái xe?",
    "Khi xảy ra tai nạn giao thông cần làm gì?",
    "Hồ sơ cấp mới chứng nhận đăng ký xe gồm những gì?",
    "Tốc độ tối đa trong khu đông dân cư là bao nhiêu?",
)

# ============================================================
# Session state
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-eyebrow">LEGAL KNOWLEDGE SYSTEM</div>
            <div class="sidebar-title">Trợ lý Luật Giao thông</div>
            <div class="sidebar-description">
                Hệ thống tra cứu quy định pháp luật giao thông đường bộ Việt Nam.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-section-label">CẤU HÌNH TRA CỨU</div>', unsafe_allow_html=True)
    top_k = st.slider(
        "Số nguồn tham khảo",
        min_value=3,
        max_value=10,
        value=5,
        help="Số đoạn tài liệu được đưa vào bước tổng hợp câu trả lời.",
    )
    use_memory = st.toggle(
        "Ghi nhớ hội thoại",
        value=True,
        help="Cho phép hệ thống hiểu câu hỏi tiếp nối như 'còn ô tô thì sao?'.",
    )

    st.markdown('<div class="sidebar-section-label suggestion-label">CÂU HỎI GỢI Ý</div>', unsafe_allow_html=True)
    for index, suggestion in enumerate(SUGGESTIONS):
        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):
            st.session_state.pending_query = suggestion

    st.markdown('<div class="sidebar-section-label">PHIÊN LÀM VIỆC</div>', unsafe_allow_html=True)
    turn_count = sum(1 for message in st.session_state.messages if message["role"] == "user")
    st.markdown(
        f"""
        <div class="session-summary">
            <span>Cuộc hội thoại hiện tại</span>
            <strong>{turn_count} lượt hỏi</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Xóa lịch sử trò chuyện", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

    if st.session_state.messages:
        try:
            st.download_button(
                label="Xuất bản ghi PDF",
                data=create_chat_pdf(st.session_state.messages),
                file_name=f"LuatGiaoThong_TroChuyen_{datetime.now():%Y%m%d_%H%M}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except ImportError:
            st.caption("Cài `fpdf2` để sử dụng chức năng xuất PDF.")

    api_ready = bool(os.getenv("OPENAI_API_KEY"))
    index_ready = (PROJECT_ROOT / "chroma_db").exists()
    backend_label = "Sẵn sàng" if api_ready else "Cần API key"
    index_label = "Đã khởi tạo" if index_ready else "Chưa khởi tạo"
    backend_class = "ready" if api_ready else "warning"
    st.markdown(
        f"""
        <div class="sidebar-section-label system-label">HỆ THỐNG</div>
        <div class="system-card">
            <div class="system-row">
                <span>Backend</span>
                <strong class="{backend_class}"><i></i>{backend_label}</strong>
            </div>
            <div class="system-row">
                <span>Chỉ mục</span>
                <strong>{index_label}</strong>
            </div>
            <div class="system-row">
                <span>Phạm vi</span>
                <strong>Luật giao thông</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Main interface
# ============================================================
render_header()
if not st.session_state.messages:
    render_welcome()

for message in st.session_state.messages:
    if message["role"] == "user":
        render_user_bubble(message["content"])
    else:
        render_assistant_bubble(message["content"])
        render_sources(message.get("sources", []))


# ============================================================
# Text input
# ============================================================
user_input = st.chat_input("Nhập câu hỏi pháp lý của bạn tại đây...")
query_input = user_input or st.session_state.pending_query

if query_input:
    st.session_state.pending_query = None
    st.session_state.messages.append({"role": "user", "content": query_input})
    st.rerun()

# ============================================================
# Process latest user turn
# ============================================================
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    query = st.session_state.messages[-1]["content"]
    history = (
        [
            {"role": message["role"], "content": message["content"]}
            for message in st.session_state.messages[:-1]
        ]
        if use_memory
        else []
    )
    response_placeholder = st.empty()
    status_placeholder = st.empty()

    with status_placeholder.container():
        render_status("Đang tìm kiếm tài liệu và tổng hợp câu trả lời...")

    try:
        from src.task10_generation import generate_with_citation

        response = generate_with_citation(query, top_k=top_k, history=history)
        answer = response.get("answer", "Chưa thể tạo câu trả lời.")
        sources = [
            {
                **source,
                "page_content": source.get("page_content", source.get("content", "")),
            }
            for source in response.get("sources", [])
        ]
        retrieval_source = response.get("retrieval_source", "none")
        search_query = response.get("search_query", query)
    except NotImplementedError:
        answer = (
            "⚠️ Giao diện Luật Giao thông đã sẵn sàng, nhưng pipeline RAG chưa được triển khai. "
            "Hãy hoàn thiện Task 10 để kết nối hệ thống tri thức pháp lý."
        )
        sources = []
        retrieval_source = "none"
        search_query = query
    except Exception as exc:
        answer = f"❌ Core System Error: {exc}"
        sources = []
        retrieval_source = "error"
        search_query = query

    status_placeholder.empty()
    with response_placeholder.container():
        render_assistant_bubble(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
            "search_query": search_query,
        }
    )
    st.rerun()
