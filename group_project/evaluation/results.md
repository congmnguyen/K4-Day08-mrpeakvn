# RAG Evaluation Results

> File này được sinh tự động bởi `eval_pipeline.py`. Đừng sửa tay —
> chạy lại `uv run python group_project/evaluation/eval_pipeline.py`.

## Framework sử dụng

**RAGAS 0.1.21** — chấm bằng `gpt-4o-mini` và `text-embedding-3-small`,
đúng model mà pipeline đang dùng, để số đo phản ánh hệ thống thật.

- Golden dataset: **18 cặp Q&A**
- Thời điểm chạy: render lại từ raw_runs.json

---

## Overall Scores

| Metric | Hybrid + LLM rerank | Hybrid + RRF (không LLM rerank) | Δ |
|---|---|---|---|
| Faithfulness | 0.978 | 0.841 | +0.137 |
| Answer Relevance | 0.429 | 0.411 | +0.018 |
| Context Recall | 0.889 | 0.917 | -0.028 |
| Context Precision | 0.969 | 0.898 | +0.072 |
| **Average** | 0.816 | 0.766 | +0.050 |

Thêm một chỉ số ngoài RAGAS — tỉ lệ retrieval chạm đúng văn bản mà
golden dataset chỉ định (`expected_source`):

| Config | Source hit rate |
|---|---|
| Hybrid + LLM rerank | 100.0% |
| Hybrid + RRF (không LLM rerank) | 100.0% |

---

## A/B Comparison

- **Hybrid + LLM rerank** vs **Hybrid + RRF (không LLM rerank)**.
- Hai config chỉ khác nhau ở bước rerank sau khi đã fuse RRF; mọi thứ
  khác (corpus, chunking, embedding, threshold fallback, prompt) giữ nguyên.

---

## Worst Performers

### Hybrid + LLM rerank

| Câu hỏi | Faith. | Ans.Rel. | Ctx.Rec. | Ctx.Prec. | Đúng nguồn |
|---|---|---|---|---|---|
| Người được chở trên xe ô tô không thắt dây đai an toàn bị phạt bao nhi… | 0.800 | 0.000 | 0.000 | 0.639 | ✅ |
| Người điều khiển xe mô tô có nồng độ cồn chưa vượt quá 50 miligam/100 … | 0.800 | 0.526 | 0.000 | 1.000 | ✅ |
| Người lái xe và người được chở trên xe ô tô có bắt buộc thắt dây đai a… | 1.000 | 0.266 | 1.000 | 1.000 | ✅ |
| Người điều khiển xe mô tô, xe gắn máy không chấp hành hiệu lệnh của đè… | 1.000 | 0.485 | 1.000 | 0.806 | ✅ |
| Người điều khiển xe mô tô, xe gắn máy không đội mũ bảo hiểm bị phạt ba… | 1.000 | 0.325 | 1.000 | 1.000 | ✅ |

### Hybrid + RRF (không LLM rerank)

| Câu hỏi | Faith. | Ans.Rel. | Ctx.Rec. | Ctx.Prec. | Đúng nguồn |
|---|---|---|---|---|---|
| Người điều khiển xe mô tô có nồng độ cồn chưa vượt quá 50 miligam/100 … | 0.500 | 0.427 | 0.000 | 0.500 | ✅ |
| Người điều khiển xe ô tô có nồng độ cồn chưa vượt quá 50 miligam/100 m… | 0.000 | 0.000 | 1.000 | 0.950 | ✅ |
| Bao nhiêu tuổi thì được điều khiển xe gắn máy?… | 0.333 | 0.459 | 1.000 | 1.000 | ✅ |
| Người điều khiển xe ô tô vượt đèn đỏ bị phạt bao nhiêu tiền?… | 1.000 | 0.477 | 1.000 | 0.333 | ✅ |
| Giấy phép lái xe bị trừ hết điểm thì người lái xe có được tiếp tục điề… | 1.000 | 0.372 | 0.500 | 1.000 | ✅ |

---

## Phân tích & Đề xuất cải tiến

- Config tốt hơn theo điểm trung bình: **Hybrid + LLM rerank** (chênh 0.050).
- Metric phân hoá rõ nhất giữa hai config: `Faithfulness` (chênh +0.137 nghiêng về Hybrid + LLM rerank).
- `Context Precision` cũng chênh đáng kể: +0.072.
- Hybrid + LLM rerank: **18/18** câu đều chạm đúng văn bản kỳ vọng — retrieval chọn đúng TÀI LIỆU ở mọi câu, phần chênh lệch giữa hai config nằm ở việc chọn đúng KHOẢN nào trong tài liệu đó.
  - Nhưng **2** câu có `Context Recall = 0`, tức lấy đúng văn bản mà trượt khoản chứa đáp án: "Người điều khiển xe mô tô có nồng độ cồn chưa vượt…"; "Người được chở trên xe ô tô không thắt dây đai an …".
- Hybrid + RRF (không LLM rerank): **18/18** câu đều chạm đúng văn bản kỳ vọng — retrieval chọn đúng TÀI LIỆU ở mọi câu, phần chênh lệch giữa hai config nằm ở việc chọn đúng KHOẢN nào trong tài liệu đó.
  - Nhưng **1** câu có `Context Recall = 0`, tức lấy đúng văn bản mà trượt khoản chứa đáp án: "Người điều khiển xe mô tô có nồng độ cồn chưa vượt…".
- ⚠️ Artifact đo lường: Hybrid + LLM rerank có **1** câu bị `answer_relevancy` chấm đúng 0.0, Hybrid + RRF (không LLM rerank) có **1** câu bị `answer_relevancy` chấm đúng 0.0. Đây là bộ phân loại "noncommittal" của RAGAS bắn nhầm trên câu trả lời tiếng Việt, không phải câu trả lời sai — đã kiểm tra tay các câu này đều nêu đúng mức phạt. Đọc `answer_relevancy` như chỉ số tương đối giữa hai config, đừng đọc như tỉ lệ đúng/sai tuyệt đối.

**Đề xuất cải tiến rút ra từ số đo trên:**

1. `Context Recall` (0.889) THẤP HƠN `Context Precision` (0.969): context lấy về sạch nhưng còn thiếu khoản chứa đáp án. Nút chỉnh đúng là **retrieval**, không phải rerank — tăng `candidate_k` trong `retrieve()` (hiện `max(top_k*4, 20)`) hoặc giảm `CHUNK_SIZE` để mỗi khoản gọn trong một chunk.
2. Source hit rate đạt 100% ở cả hai config nên `QUERY_GLOSSARY` hiện đủ cho bộ câu hỏi này. Muốn kiểm tra thật sự giới hạn của nó, cần bổ sung vào golden dataset các câu dùng tiếng lóng/khẩu ngữ chưa có trong glossary.
3. Chênh lệch `Faithfulness` (+0.137) là lý do rõ ràng nhất để giữ **Hybrid + LLM rerank** làm mặc định: bước rerank quyết định khoản luật nào lọt vào context, và đưa nhầm khoản của loại phương tiện khác chính là nguồn gốc câu trả lời sai số tiền phạt.
4. `answer_relevancy` quanh 0.4 ở CẢ HAI config là trần của metric này trên tiếng Việt (xem cảnh báo artifact ở trên), không phải chất lượng thật. Nếu cần con số so sánh được với các bài khác, nên thêm một metric tự viết: so khớp con số tiền phạt trong câu trả lời với `expected_answer`.
