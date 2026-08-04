"""
RAG Evaluation Pipeline.

Framework: RAGAS.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý chi phí: RAGAS gọi LLM RẤT NHIỀU LẦN (không phải 1 lần/câu hỏi mà nhiều
lần/metric/câu hỏi — LLM judge cho từng metric), cộng thêm 1 lần generation +
có thể 1 lần LLM rerank (RERANK_METHOD="llm", Task 7/9) cho mỗi câu hỏi. Pipeline
gọi thẳng OpenAI (OPENAI_API_KEY, cùng provider với Task 4/5/10, không qua
OpenRouter). Với 19 câu hỏi × 2 config A/B, tổng số lượt gọi khá lớn — nếu cần
chạy nhanh khi demo, giảm subset golden_dataset.json xuống 5 câu.
"""

import sys
from contextlib import contextmanager
from pathlib import Path

import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevance",
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
}


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAGAS Evaluation
# =============================================================================

def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]):
    """
    Evaluate RAG pipeline sử dụng RAGAS.

    Args:
        rag_pipeline: callable(question: str) -> {'answer': str, 'sources': list[dict]}
        golden_dataset: list of {'question', 'expected_answer', 'expected_context'}

    Returns:
        pandas.DataFrame — 1 dòng/câu hỏi, kèm cột điểm cho từng metric.
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in golden_dataset:
        result = rag_pipeline(item["question"])
        contexts = [c["content"] for c in result.get("sources", [])]

        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append(contexts or [""])
        eval_data["ground_truth"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )
    return result.to_pandas()


# =============================================================================
# RAG Pipeline wrappers
# =============================================================================

def _default_pipeline(question: str) -> dict:
    """Pipeline mặc định — dùng nguyên cấu hình của Task 10 (generate_with_citation)."""
    from src.task10_generation import generate_with_citation

    result = generate_with_citation(question)
    return {"answer": result["answer"], "sources": result["sources"]}


@contextmanager
def _rerank_method(method: str):
    """Tạm override hằng số module-level ``RERANK_METHOD`` của Task 9.

    ``retrieve()`` đọc ``RERANK_METHOD`` từ global của chính module task9 mỗi
    lần gọi (không nhận tham số ``method`` trực tiếp) — nên đây là cách A/B
    test rerank method mà KHÔNG sửa file task9_retrieval_pipeline.py. Dùng
    context manager để luôn khôi phục giá trị gốc kể cả khi có lỗi.
    """
    import src.task9_retrieval_pipeline as task9

    original = task9.RERANK_METHOD
    task9.RERANK_METHOD = method
    try:
        yield
    finally:
        task9.RERANK_METHOD = original


def _make_pipeline(rerank_method: str, top_k: int = 5):
    """
    Tạo 1 pipeline function cho 1 rerank method cụ thể ("llm" | "rrf" | ...).

    generate_with_citation() (Task 10) không cho chỉnh rerank method từ ngoài
    — nên A/B test phải tự lắp lại retrieve → reorder → format → generate ở
    đây, tái sử dụng đúng các hàm/hằng số Task 9 + Task 10 đã viết (không sửa
    2 file đó). Gọi OpenAI trực tiếp giống hệt cách Task 10 gọi thật (cùng
    provider, cùng model qua OPENAI_CHAT_MODEL, không phải OpenRouter).
    """
    from openai import OpenAI
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import (
        reorder_for_llm,
        format_context,
        SYSTEM_PROMPT,
        LLM_MODEL,
        TEMPERATURE,
        TOP_P,
    )

    client = OpenAI()

    def pipeline(question: str) -> dict:
        with _rerank_method(rerank_method):
            chunks = retrieve(question, top_k=top_k, use_reranking=True)
        reordered = reorder_for_llm(chunks) if chunks else chunks
        context = format_context(reordered) if reordered else ""
        user_message = f"Context:\n{context}\n\n---\n\nQuestion: {question}"

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return {"answer": response.choices[0].message.content, "sources": chunks}

    return pipeline


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> dict:
    """
    So sánh A/B giữa 2 rerank method của Task 7/9: LLM rerank (mặc định,
    RERANK_METHOD="llm") vs RRF thuần (method="rrf").

    Đây là knob có tác động lớn nhất quan sát được khi làm Task 9: corpus toàn
    các khoản phạt viết gần giống nhau nên RRF thuần (chỉ dựa thứ hạng dense +
    BM25) không đủ phân biệt top-1, trong khi LLM đọc được cả câu hỏi lẫn nội
    dung đoạn văn để bắc cầu giữa từ đời thường và thuật ngữ pháp lý.

    Returns:
        {config_name: pandas.DataFrame} — kết quả evaluate_with_ragas() cho từng config.
    """
    configs = {
        "rerank_llm": "llm",
        "rerank_rrf": "rrf",
    }

    results = {}
    for config_name, method in configs.items():
        print(f"\n--- Evaluating config: {config_name} (RERANK_METHOD={method}) ---")
        pipeline = _make_pipeline(rerank_method=method)
        results[config_name] = evaluate_with_ragas(pipeline, golden_dataset)

    return results


# =============================================================================
# Export Results
# =============================================================================

def export_results(results, comparison: dict):
    """Export evaluation results ra results.md."""
    metric_cols = list(METRIC_LABELS.keys())
    config_names = list(comparison.keys())
    config_avgs = {name: df[metric_cols].mean() for name, df in comparison.items()}

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Framework sử dụng",
        "",
        "RAGAS",
        "",
        "---",
        "",
        "## Overall Scores",
        "",
        "| Metric | " + " | ".join(config_names) + " | Δ |",
        "|--------|" + "|".join(["---"] * len(config_names)) + "|---|",
    ]

    for metric in metric_cols:
        row = [METRIC_LABELS[metric]]
        row += [f"{config_avgs[name][metric]:.3f}" for name in config_names]
        delta = config_avgs[config_names[0]][metric] - config_avgs[config_names[-1]][metric]
        row.append(f"{delta:+.3f}")
        lines.append("| " + " | ".join(row) + " |")

    overall_row = ["**Average**"]
    overall_row += [f"{config_avgs[name][metric_cols].mean():.3f}" for name in config_names]
    overall_delta = (
        config_avgs[config_names[0]][metric_cols].mean()
        - config_avgs[config_names[-1]][metric_cols].mean()
    )
    overall_row.append(f"{overall_delta:+.3f}")
    lines.append("| " + " | ".join(overall_row) + " |")

    lines += ["", "---", "", "## A/B Comparison Analysis", ""]
    for name in config_names:
        lines.append(f"**{name}:** điểm trung bình {config_avgs[name][metric_cols].mean():.3f}")
        lines.append("")

    best = max(config_names, key=lambda n: config_avgs[n][metric_cols].mean())
    lines.append(
        f"**Kết luận:** `{best}` đạt điểm trung bình cao hơn — ưu tiên config này cho production."
    )
    lines += ["", "---", ""]

    # Worst performers — lấy từ kết quả evaluate trên pipeline mặc định (results)
    lines += [
        "## Worst Performers (Bottom 3)",
        "",
        "| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |",
        "|---|----------|-------------|-----------|--------|---------------|------------|",
    ]

    results = results.copy()
    results["avg_score"] = results[metric_cols].mean(axis=1)
    worst = results.sort_values("avg_score").head(3)

    for i, (_, row) in enumerate(worst.iterrows(), 1):
        lowest_metric = min(metric_cols, key=lambda m: row[m])
        failure_stage = "Retrieval" if lowest_metric in ("context_recall", "context_precision") else "Generation"
        question_preview = str(row["question"])[:60]
        lines.append(
            f"| {i} | {question_preview} | {row['faithfulness']:.3f} | "
            f"{row['answer_relevancy']:.3f} | {row['context_recall']:.3f} | {failure_stage} | "
            f"{METRIC_LABELS[lowest_metric]} thấp nhất ({row[lowest_metric]:.3f}) |"
        )

    # Recommendations — dựa trên 3 metric có điểm trung bình thấp nhất trên toàn bộ dataset
    lines += ["", "---", "", "## Recommendations", ""]
    action_by_metric = {
        "context_precision": (
            "Tune lại reranking / giảm top_k retrieval để lọc bớt chunk nhiễu trước khi đưa vào context",
            "Tăng context_precision, giảm nhiễu ảnh hưởng tới faithfulness",
        ),
        "context_recall": (
            "Tăng top_k retrieval hoặc cải thiện chunking (chunk_size/overlap) để bao phủ đủ evidence",
            "Tăng context_recall, giảm câu trả lời thiếu bằng chứng",
        ),
        "faithfulness": (
            "Siết system prompt yêu cầu bám sát context, hạ temperature",
            "Giảm hallucination, tăng độ tin cậy câu trả lời",
        ),
        "answer_relevancy": (
            "Cải thiện prompt để bám sát đúng câu hỏi, tránh trả lời lan man ngoài trọng tâm",
            "Tăng độ liên quan giữa câu trả lời và câu hỏi",
        ),
    }
    avg_by_metric = results[metric_cols].mean().sort_values()
    for i, metric in enumerate(avg_by_metric.index[:3], 1):
        action, impact = action_by_metric[metric]
        lines += [f"### Cải tiến {i}", f"**Action:** {action}", f"**Expected impact:** {impact}", ""]

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✓ Results exported to {RESULTS_PATH}")


if __name__ == "__main__":
    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    results = evaluate_with_ragas(_default_pipeline, golden_dataset)
    comparison = compare_configs(golden_dataset)
    export_results(results, comparison)
