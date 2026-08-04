# Bài Tập Nhóm — Vietnamese Traffic Law RAG Chatbot

**Nhóm thực hiện cả 2 sản phẩm:** RAG Chatbot (Streamlit) và Evaluation Pipeline (RAGAS).

| Sản phẩm | Vị trí | Chạy thử |
|---|---|---|
| RAG Chatbot | `app.py` + `ui/` | `uv run streamlit run app.py` |
| Evaluation Pipeline | `group_project/evaluation/` | `uv run python group_project/evaluation/eval_pipeline.py` |

---

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật giao thông đường bộ Việt Nam.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

Xem code mẫu (DeepEval/RAGAS/TruLens) chi tiết trong `README.md` gốc mục "Yêu cầu 2".

### Deliverable Evaluation

- [x] File `group_project/evaluation/golden_dataset.json` — **19 cặp Q&A** (yêu cầu ≥15), mỗi câu đối chiếu trực tiếp với văn bản trong corpus (ghi rõ Điều/khoản ở `expected_context`), phủ cả 5 văn bản pháp luật lẫn bài viết
- [x] File `group_project/evaluation/eval_pipeline.py` — **RAGAS 0.1.21**, 4 metric, ép dùng đúng `gpt-4o-mini` + `text-embedding-3-small` của repo
- [x] File `group_project/evaluation/results.md` — sinh tự động từ số đo thật
- [x] So sánh A/B: **`rerank_llm`** vs **`rerank_rrf`**, patch `RERANK_METHOD` bằng context manager nên không phải sửa Task 9

```bash
uv run python group_project/evaluation/eval_pipeline.py
```

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
                  ┌──────────────────────────────────────────────┐
   Câu hỏi ──────▶│ condense_query()  (chỉ khi có lịch sử chat)   │
                  │   "còn ô tô thì sao?" → câu hỏi độc lập       │
                  └───────────────────┬──────────────────────────┘
                                      ▼
                  ┌──────────────────────────────────────────────┐
                  │ expand_query()  glossary đời thường → pháp lý │
                  └───────────────────┬──────────────────────────┘
                          ┌───────────┴───────────┐   (song song)
                          ▼                       ▼
              ┌───────────────────┐   ┌───────────────────────┐
              │ semantic_search   │   │ lexical_search        │
              │ OpenAI embedding  │   │ BM25 trên cùng chunk  │
              │ Chroma cosine     │   │ rank_bm25             │
              └─────────┬─────────┘   └───────────┬───────────┘
                        │  cosine gốc             │
                        ▼                         │
          ┌──────────────────────────┐            │
          │ best cosine < 0.40 ?     │──── có ───▶│  pageindex_search()
          │ (ngưỡng đã calibrate)    │            │  vectorless, cây Điều
          └────────────┬─────────────┘            │  → source="pageindex"
                    không                          │
                        ▼                          │
              ┌───────────────────────────────────┴──┐
              │ rerank_rrf()  fuse 2 ranked list      │
              └────────────────┬─────────────────────┘
                               ▼
              ┌──────────────────────────────────────┐
              │ rerank(method="llm")  LLM cross-enc.  │
              └────────────────┬─────────────────────┘
                               ▼
              ┌──────────────────────────────────────┐
              │ reorder_for_llm()  [1,2,3,4,5]→       │
              │                    [1,3,5,4,2]        │
              │ format_context()  nhãn trích dẫn +    │
              │                   mốc hiệu lực + Điều │
              └────────────────┬─────────────────────┘
                               ▼
                     Generation (gpt-4o-mini)
                     → answer + sources + retrieval_source

   Retrieval rỗng → trả "không thể xác minh", KHÔNG gọi model.
```

Chi tiết corpus, số liệu calibrate ngưỡng và các quyết định kỹ thuật: xem mục
**"Ghi Chú Triển Khai"** trong `README.md` ở thư mục gốc.

---

## Phân Công Công Việc

| Thành viên | MSSV | Vai trò | Nhiệm vụ | Trạng thái |
|---|---|---|---|---|
| Nguyễn Minh Công | 2A202601945 | **Role 1** — Team Leader & RAG Architect | Điều phối tiến độ, ghép code tổng hợp, Task 9 (retrieval pipeline) | ✅ Hoàn thành |
| Diệp Đức Lai | 2A202601784 | **Role 3** — Frontend & Chatbot Developer | Giao diện Streamlit `app.py` + package `ui/`, nối Task 10 | ✅ Hoàn thành |
| Nguyễn Văn Sáng | 2A202601252 | **Role 4** — Evaluation & QA Engineer | `golden_dataset.json`, `eval_pipeline.py` (RAGAS), `results.md` | ✅ Hoàn thành |

### Chi tiết đóng góp

**Role 1 — Nguyễn Minh Công**
- Task 9: hybrid retrieval (dense + BM25 chạy song song) → RRF fusion → LLM rerank, fallback PageIndex quyết định bằng cosine gốc với `SCORE_THRESHOLD = 0.40` đã calibrate bằng số đo thật.
- Ghép code: merge nhánh Role 3 (PR #1) và Role 4 (`feature/role4-evaluation`) vào `main`.
- Corpus: tìm nguồn text sạch chính thức cho 3 thông tư bản scan (bản `.doc` Công báo), contextual chunking, query glossary.

**Role 3 — Diệp Đức Lai**
- `app.py` + package `ui/` (`styles.py`, `components.py`, `pdf_export.py`): giao diện chat, hiển thị nguồn tham khảo kèm mức độ liên quan, xuất bản ghi hội thoại ra PDF.
- Nối `generate_with_citation` của Task 10, truyền `top_k` và lịch sử hội thoại (multi-turn).

**Role 4 — Nguyễn Văn Sáng**
- `golden_dataset.json`: 19 cặp Q&A, mỗi câu đối chiếu trực tiếp với văn bản trong corpus (ghi rõ Điều/khoản ở `expected_context`).
- `eval_pipeline.py`: RAGAS 0.1.21 với 4 metric, A/B `rerank_llm` vs `rerank_rrf` bằng context manager patch `RERANK_METHOD`.
- `results.md`: bảng điểm, phân tích A/B, worst performers kèm cột Failure Stage / Root Cause, và đề xuất cải tiến xếp theo metric yếu nhất.

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
uv sync
cp .env.example .env    # điền OPENAI_API_KEY

# Build index (bỏ qua cũng được — query đầu tiên sẽ tự build, nhưng chậm)
uv run python -m src.task4_chunking_indexing

# Chạy app
uv run streamlit run app.py
```

Chạy evaluation:
```bash
uv run python group_project/evaluation/eval_pipeline.py
```

---

## Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
