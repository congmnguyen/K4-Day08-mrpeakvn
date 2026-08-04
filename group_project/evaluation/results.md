# RAG Evaluation Results

## Framework sử dụng

RAGAS

---

## Overall Scores

| Metric | rerank_llm | rerank_rrf | Δ |
|--------|---|---|---|
| Faithfulness | 0.536 | 0.585 | -0.049 |
| Answer Relevance | 0.776 | 0.568 | +0.208 |
| Context Recall | 1.000 | 0.947 | +0.053 |
| Context Precision | 0.985 | 0.913 | +0.072 |
| **Average** | 0.824 | 0.753 | +0.071 |

---

## A/B Comparison Analysis

**rerank_llm:** điểm trung bình 0.824

**rerank_rrf:** điểm trung bình 0.753

**Kết luận:** `rerank_llm` đạt điểm trung bình cao hơn — ưu tiên config này cho production.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Khi phát hiện vi phạm qua thiết bị kỹ thuật nghiệp vụ nhưng  | 0.000 | 0.000 | 1.000 | Generation | Faithfulness thấp nhất (0.000) |
| 2 | Người điều khiển xe ô tô có nồng độ cồn vượt quá 80 miligam/ | 0.667 | 0.000 | 1.000 | Generation | Answer Relevance thấp nhất (0.000) |
| 3 | Luật Trật tự, an toàn giao thông đường bộ có cho phép điều k | 0.000 | 0.798 | 1.000 | Generation | Faithfulness thấp nhất (0.000) |

---

## Recommendations

### Cải tiến 1
**Action:** Siết system prompt yêu cầu bám sát context, hạ temperature
**Expected impact:** Giảm hallucination, tăng độ tin cậy câu trả lời

### Cải tiến 2
**Action:** Cải thiện prompt để bám sát đúng câu hỏi, tránh trả lời lan man ngoài trọng tâm
**Expected impact:** Tăng độ liên quan giữa câu trả lời và câu hỏi

### Cải tiến 3
**Action:** Tune lại reranking / giảm top_k retrieval để lọc bớt chunk nhiễu trước khi đưa vào context
**Expected impact:** Tăng context_precision, giảm nhiễu ảnh hưởng tới faithfulness
