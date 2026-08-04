"""
Task 10 — Generation Có Citation.

Pipeline: condense follow-up → retrieve (Task 9) → reorder chống "lost in the
middle" → format context có nhãn nguồn → gọi LLM → trả answer + sources.

LLM: gọi thẳng OpenAI (cùng provider với embedding ở Task 4/5, một API key duy
nhất cho cả repo). Model cấu hình qua ``OPENAI_CHAT_MODEL``.

Conversation memory (multi-turn):
    Câu hỏi tiếp nối kiểu "còn ô tô thì sao?" không thể đem đi retrieve nguyên
    văn — nó không chứa từ khoá nào để dense/BM25 bám vào. Vì vậy khi có lịch sử
    hội thoại, ``condense_query()`` viết lại thành câu hỏi độc lập TRƯỚC khi
    retrieve, còn lịch sử vẫn được đưa vào messages để câu trả lời mạch lạc.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.2 vì: tra cứu pháp luật cần bám sát văn bản, gần như không sáng tạo
TEMPERATURE = 0.2

# Model generation — đọc từ env để đổi được mà không sửa code.
LLM_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Số message lịch sử tối đa đưa vào prompt (6 = 3 lượt hỏi-đáp). Giới hạn để
# prompt không phình theo độ dài hội thoại và chi phí mỗi lượt ổn định.
HISTORY_WINDOW = 6

NO_EVIDENCE_ANSWER = (
    "Tôi không thể xác minh thông tin này từ nguồn hiện có. "
    "Corpus của hệ thống chỉ gồm Luật Trật tự, an toàn giao thông đường bộ "
    "36/2024/QH15, Nghị định 168/2024/NĐ-CP và các Thông tư 72, 73, 79/2024/TT-BCA."
)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý tra cứu pháp luật giao thông đường bộ Việt Nam. Hãy trả lời
dựa trên context được cung cấp và nêu rõ mốc hiệu lực của nguồn khi có.

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Mỗi khẳng định phải có trích dẫn ngay sau, COPY NGUYÊN VĂN chuỗi ở trường
   "Trích dẫn" của Document tương ứng, ví dụ: [Nghị định 168/2024/NĐ-CP].
   Không tự đổi loại văn bản (Thông tư ≠ Nghị định ≠ Luật).
3. Chỉ nêu mốc hiệu lực đúng như trường "Hiệu lực từ"; không suy đoán từ năm ban hành
4. Nếu context không đủ thông tin → trả lời: "Tôi không thể xác minh thông tin này từ nguồn hiện có"
5. Trả lời bằng tiếng Việt, có cấu trúc rõ ràng theo đoạn văn
6. Không suy luận hay mở rộng ngoài những gì được nêu trong context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)
    """
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]   # hạng 1, 3, 5 → đầu prompt
    back = chunks[1::2]   # hạng 2, 4    → cuối prompt, đảo ngược
    return front + back[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

# Tên loại văn bản tiếng Việt — thiếu map này LLM hay cite sai tiền tố
# ("Nghị định 72/2024/TT-BCA" thay vì "Thông tư 72/2024/TT-BCA").
DOCUMENT_TYPE_LABELS = {
    "law": "Luật",
    "decree": "Nghị định",
    "circular": "Thông tư",
    "resolution": "Nghị quyết",
}


def citation_label(metadata: dict) -> str:
    """Chuỗi citation chuẩn để LLM copy nguyên văn.

    Văn bản pháp luật → "Thông tư 72/2024/TT-BCA". Bài viết/tin (không có số
    hiệu) → tiêu đề + nguồn xuất bản. Luôn trả về một nhãn dùng được: nếu để
    trống, prompt bắt buộc trích dẫn sẽ không thể thoả mãn và LLM dễ tự bịa nhãn.
    """
    document_number = metadata.get("document_number")
    if document_number:
        prefix = DOCUMENT_TYPE_LABELS.get(metadata.get("document_type") or "")
        return f"{prefix} {document_number}" if prefix else str(document_number)

    title = metadata.get("title")
    publisher = metadata.get("publisher")
    if title:
        return f"{title} — {publisher}" if publisher else str(title)
    return str(metadata.get("source", "nguồn không xác định"))


def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có nhãn source + citation + mốc hiệu lực để LLM cite chính xác.
    """
    context_parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source", f"Source {index}")
        doc_type = metadata.get("type", "unknown")

        label = (
            f"[Document {index} | Source: {source} | Type: {doc_type}"
            f" | Trích dẫn: {citation_label(metadata)}"
        )
        if metadata.get("effective_date"):
            label += f" | Hiệu lực từ: {metadata['effective_date']}"
        # Heading của Điều là thứ phân biệt "Điều 6 ... xe ô tô" với "Điều 7 ...
        # xe mô tô, xe gắn máy". Thiếu nó, một chunk chỉ là danh sách hành vi
        # kèm mức tiền, model rất dễ gán mức phạt của ô tô cho xe máy.
        section = metadata.get("section") or metadata.get("heading")
        if section:
            label += f" | Điều/Mục: {section}"
        label += "]"
        context_parts.append(f"{label}\n{chunk['content']}\n")
    return "\n---\n".join(context_parts)


# =============================================================================
# CONVERSATION MEMORY
# =============================================================================

CONDENSE_PROMPT = """Viết lại câu hỏi cuối của người dùng thành MỘT câu hỏi độc lập,
đầy đủ ngữ cảnh, để đem đi tra cứu văn bản pháp luật giao thông đường bộ Việt Nam.

Quy tắc:
- Thay đại từ và cách nói tỉnh lược bằng danh từ cụ thể lấy từ lịch sử hội thoại
  ("còn ô tô thì sao?" sau câu hỏi về xe máy vượt đèn đỏ → "Ô tô vượt đèn đỏ bị
  phạt bao nhiêu tiền?").
- Nếu câu hỏi cuối ĐÃ độc lập, trả lại đúng nguyên văn.
- Không trả lời câu hỏi, không thêm giải thích. Chỉ xuất đúng một câu hỏi."""


def trim_history(history: list[dict] | None) -> list[dict]:
    """Giữ HISTORY_WINDOW message cuối, chỉ nhận role user/assistant hợp lệ."""
    if not history:
        return []
    cleaned = [
        {"role": message["role"], "content": message["content"]}
        for message in history
        if message.get("role") in {"user", "assistant"} and message.get("content")
    ]
    return cleaned[-HISTORY_WINDOW:]


def condense_query(query: str, history: list[dict] | None) -> str:
    """Viết lại câu hỏi tiếp nối thành câu hỏi độc lập để retrieve.

    Không có lịch sử → trả nguyên query, không tốn API call. Lỗi API/response
    rỗng → cũng trả nguyên query thay vì làm chết cả lượt chat.
    """
    recent = trim_history(history)
    if not recent:
        return query

    from openai import OpenAI

    transcript = "\n".join(
        f"{'Người dùng' if message['role'] == 'user' else 'Trợ lý'}: {message['content']}"
        for message in recent
    )
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": CONDENSE_PROMPT},
                {
                    "role": "user",
                    "content": f"Lịch sử hội thoại:\n{transcript}\n\nCâu hỏi cuối: {query}",
                },
            ],
            temperature=0,
        )
        condensed = (response.choices[0].message.content or "").strip()
    except Exception as error:  # noqa: BLE001 — memory hỏng không được chặn câu trả lời
        print(f"  ⚠ Condense query lỗi ({error}); dùng câu hỏi gốc")
        return query
    return condensed or query


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str, top_k: int = TOP_K, history: list[dict] | None = None
) -> dict:
    """
    End-to-end RAG generation có citation, hỗ trợ multi-turn.

    Args:
        query: Câu hỏi của user (có thể là câu tiếp nối)
        top_k: Số chunks đưa vào context
        history: Lịch sử hội thoại ``[{'role': 'user'|'assistant', 'content': str}]``
            KHÔNG bao gồm ``query`` hiện tại. Bỏ trống = hỏi đáp một lượt.

    Returns:
        {
            'answer': str,            # Câu trả lời có citation
            'sources': list[dict],    # Các chunks đã dùng
            'retrieval_source': str,  # 'hybrid' | 'pageindex' | 'none'
            'search_query': str       # Câu đã condense, dùng để retrieve
        }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")

    recent_history = trim_history(history)
    search_query = condense_query(query, recent_history)

    chunks = retrieve(search_query, top_k=top_k)
    if not chunks:
        # Không có evidence → KHÔNG gọi model, tránh việc LLM tự bịa từ tri thức nền.
        return {
            "answer": NO_EVIDENCE_ANSWER,
            "sources": [],
            "retrieval_source": "none",
            "search_query": search_query,
        }

    context = format_context(reorder_for_llm(chunks))
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {query}"

    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        # Lịch sử nằm GIỮA system prompt và context: model thấy được mạch hội
        # thoại nhưng evidence của lượt hiện tại vẫn là thứ gần câu hỏi nhất.
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *recent_history,
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    answer = (response.choices[0].message.content or "").strip() or NO_EVIDENCE_ANSWER

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "hybrid"),
        "search_query": search_query,
    }


if __name__ == "__main__":
    test_queries = [
        "Xe máy vượt đèn đỏ bị phạt bao nhiêu?",
        "Khi xảy ra tai nạn giao thông cần làm gì?",
        "Hồ sơ cấp mới chứng nhận đăng ký xe gồm những gì?",
        "xyzabc123nonsense",
    ]

    for question in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {question}")
        print("=" * 70)
        result = generate_with_citation(question)
        print(f"\nA: {result['answer']}")
        print(
            f"\n[Sources: {len(result['sources'])} chunks | "
            f"via {result['retrieval_source']}]"
        )

    # Multi-turn: câu thứ hai tỉnh lược, chỉ hiểu được nhờ conversation memory.
    print(f"\n{'='*70}")
    print("DEMO conversation memory")
    print("=" * 70)
    conversation: list[dict] = []
    for question in ("Xe máy vượt đèn đỏ bị phạt bao nhiêu?", "còn ô tô thì sao?"):
        result = generate_with_citation(question, history=conversation)
        print(f"\nQ: {question}")
        print(f"   → retrieve bằng: {result['search_query']!r}")
        print(f"A: {result['answer'][:300]}")
        conversation.append({"role": "user", "content": question})
        conversation.append({"role": "assistant", "content": result["answer"]})
