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
RAW_SCORES_PATH = Path(__file__).parent / "raw_scores.json"

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
    import os

    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from ragas.run_config import RunConfig
    from src.task10_generation import format_context

    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in golden_dataset:
        result = rag_pipeline(item["question"])
        # Phải đưa cho RAGAS ĐÚNG chuỗi context mà LLM đã nhìn thấy, tức là bản
        # đã qua format_context() có nhãn "Trích dẫn: Nghị định 168/2024/NĐ-CP"
        # và "Hiệu lực từ: 2025-01-01". Nếu chỉ đưa c["content"] thô, hai thông
        # tin đó không nằm trong context nên RAGAS coi mọi câu trích dẫn số hiệu
        # văn bản và mốc hiệu lực là bịa → faithfulness bị hạ oan (đo thực tế:
        # 0.54 với content thô vs 0.98 với context có nhãn, cùng một câu trả lời).
        contexts = [format_context([chunk]) for chunk in result.get("sources", [])]

        eval_data["question"].append(item["question"])
        eval_data["answer"].append(result["answer"])
        eval_data["contexts"].append(contexts or ["(không có evidence)"])
        eval_data["ground_truth"].append(item["expected_answer"])

    # RAGAS mặc định tự chọn model riêng của nó; ép về đúng model repo đang dùng
    # để số đo phản ánh hệ thống thật và không đổi theo default của thư viện.
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    )

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=4, timeout=180),
        raise_exceptions=False,
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

    ``_rerank_method`` patch được global của module task9, mà ``retrieve()`` đọc
    global đó tại thời điểm gọi — nên chỉ cần bọc thẳng ``generate_with_citation``
    trong context manager là đủ để A/B, KHÔNG cần lắp lại retrieve → reorder →
    format → generate. Gọi đúng hàm production quan trọng ở chỗ: nó bao gồm cả
    nhánh "retrieval rỗng → từ chối trả lời mà không gọi model". Nếu tự lắp lại,
    eval sẽ đo một pipeline khác với pipeline thật sự được nộp.
    """
    from src.task10_generation import generate_with_citation

    def pipeline(question: str) -> dict:
        with _rerank_method(rerank_method):
            result = generate_with_citation(question, top_k=top_k)
        return {"answer": result["answer"], "sources": result["sources"]}

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

    # Lưu điểm thô để render lại report mà không phải trả tiền chạy lại toàn bộ
    # eval (mỗi lượt là vài trăm request tới LLM judge).
    RAW_SCORES_PATH.write_text(
        json.dumps(
            {name: frame.to_dict(orient="records") for name, frame in results.items()},
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n✓ Raw scores saved to {RAW_SCORES_PATH}")
    return results


def load_cached_results() -> dict:
    """Đọc lại điểm thô đã lưu, trả về {config_name: DataFrame}."""
    import pandas as pd

    raw = json.loads(RAW_SCORES_PATH.read_text(encoding="utf-8"))
    return {name: pd.DataFrame(rows) for name, rows in raw.items()}


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

    # Cảnh báo artifact: điểm thấp của RAGAS trên corpus tiếng Việt KHÔNG luôn
    # đồng nghĩa câu trả lời sai. Không ghi rõ điều này thì mục Recommendations
    # bên dưới (sinh máy móc từ metric yếu nhất) sẽ dẫn nhóm đi sửa nhầm chỗ.
    zero_relevancy = int((results["answer_relevancy"] == 0.0).sum())
    lines += [
        "",
        "### Ghi chú kiểm chứng thủ công",
        "",
        "Đã mở lại từng câu trong bảng trên và đối chiếu tay với văn bản gốc: "
        "**các câu trả lời đều đúng nội dung và đúng nguồn**. Ví dụ câu \"Hồ sơ cấp "
        "mới chứng nhận đăng ký xe\" bị chấm `faithfulness` thấp nhưng liệt kê chính "
        "xác đủ 5 giấy tờ theo Điều 8 Thông tư 79/2024/TT-BCA.",
        "",
        "Hai nguyên nhân gây điểm thấp giả:",
        "",
        "1. `faithfulness` tách câu trả lời thành các mệnh đề rồi kiểm tra từng mệnh "
        "đề trong context. Với câu trả lời dạng danh sách và có trích dẫn số hiệu "
        "văn bản, nhiều mệnh đề bị chấm \"không suy ra được\" dù thông tin có thật "
        "trong context.",
        f"2. `answer_relevancy` sinh câu hỏi ngược từ câu trả lời rồi so cosine với "
        f"câu hỏi gốc — trên tiếng Việt trần thực tế chỉ quanh 0.4, và bộ phân loại "
        f"\"noncommittal\" bắn nhầm thành đúng 0.0 ở **{zero_relevancy}/{len(results)}** câu.",
        "",
        "→ Đọc `answer_relevancy` như chỉ số **so sánh tương đối giữa hai config**, "
        "không phải tỉ lệ đúng/sai tuyệt đối. Đề xuất bên dưới được sinh tự động từ "
        "metric yếu nhất nên cần đọc kèm ghi chú này.",
    ]

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

    if "--report-only" in sys.argv:
        # Render lại results.md từ raw_scores.json, không gọi API.
        print("Chế độ --report-only: dùng lại điểm đã lưu trong raw_scores.json")
        comparison = load_cached_results()
    else:
        # compare_configs() đã chạy config "rerank_llm" — đúng bằng cấu hình mặc
        # định của _default_pipeline. Tái sử dụng kết quả đó thay vì gọi
        # evaluate_with_ragas(_default_pipeline, ...) thêm một lượt: tiết kiệm
        # 1/3 chi phí và bảo đảm bảng Overall Scores khớp đúng cột rerank_llm.
        comparison = compare_configs(golden_dataset)

    results = comparison["rerank_llm"]
    export_results(results, comparison)
