# RAG Evaluation Results

## Framework sử dụng

RAGAS

---

## Overall Scores

| Metric | rerank_llm | rerank_rrf | Δ |
|--------|---|---|---|
| Faithfulness | 0.865 | 0.819 | +0.046 |
| Answer Relevance | 0.429 | 0.398 | +0.031 |
| Context Recall | 1.000 | 0.912 | +0.088 |
| Context Precision | 0.965 | 0.901 | +0.064 |
| **Average** | 0.815 | 0.758 | +0.057 |

---

## A/B Comparison Analysis

**rerank_llm:** điểm trung bình 0.815

**rerank_rrf:** điểm trung bình 0.758

**Kết luận:** `rerank_llm` đạt điểm trung bình cao hơn — ưu tiên config này cho production.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Khi phát hiện vi phạm qua thiết bị kỹ thuật nghiệp vụ nhưng  | 0.500 | 0.000 | 1.000 | Generation | Answer Relevance thấp nhất (0.000) |
| 2 | Luật Trật tự, an toàn giao thông đường bộ có cho phép điều k | 0.250 | 0.384 | 1.000 | Generation | Faithfulness thấp nhất (0.250) |
| 3 | Hồ sơ cấp mới chứng nhận đăng ký xe, biển số xe gồm những gi | 0.286 | 0.562 | 1.000 | Generation | Faithfulness thấp nhất (0.286) |

### Ghi chú kiểm chứng thủ công

Đã mở lại từng câu trong bảng trên và đối chiếu tay với văn bản gốc: **các câu trả lời đều đúng nội dung và đúng nguồn**. Ví dụ câu "Hồ sơ cấp mới chứng nhận đăng ký xe" bị chấm `faithfulness` thấp nhưng liệt kê chính xác đủ 5 giấy tờ theo Điều 8 Thông tư 79/2024/TT-BCA.

Hai nguyên nhân gây điểm thấp giả:

1. `faithfulness` tách câu trả lời thành các mệnh đề rồi kiểm tra từng mệnh đề trong context. Với câu trả lời dạng danh sách và có trích dẫn số hiệu văn bản, nhiều mệnh đề bị chấm "không suy ra được" dù thông tin có thật trong context.
2. `answer_relevancy` sinh câu hỏi ngược từ câu trả lời rồi so cosine với câu hỏi gốc — trên tiếng Việt trần thực tế chỉ quanh 0.4, và bộ phân loại "noncommittal" bắn nhầm thành đúng 0.0 ở **1/19** câu.

→ Đọc `answer_relevancy` như chỉ số **so sánh tương đối giữa hai config**, không phải tỉ lệ đúng/sai tuyệt đối. Đề xuất bên dưới được sinh tự động từ metric yếu nhất nên cần đọc kèm ghi chú này.

---

## Recommendations

### Cải tiến 1
**Action:** Cải thiện prompt để bám sát đúng câu hỏi, tránh trả lời lan man ngoài trọng tâm
**Expected impact:** Tăng độ liên quan giữa câu trả lời và câu hỏi

### Cải tiến 2
**Action:** Siết system prompt yêu cầu bám sát context, hạ temperature
**Expected impact:** Giảm hallucination, tăng độ tin cậy câu trả lời

### Cải tiến 3
**Action:** Tune lại reranking / giảm top_k retrieval để lọc bớt chunk nhiễu trước khi đưa vào context
**Expected impact:** Tăng context_precision, giảm nhiễu ảnh hưởng tới faithfulness
