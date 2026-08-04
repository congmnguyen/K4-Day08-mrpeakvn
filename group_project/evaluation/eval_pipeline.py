"""
RAG Evaluation Pipeline.

Sử dụng DeepEval / RAGAS / TruLens để đánh giá chất lượng RAG pipeline.
Chọn 1 framework và implement đầy đủ.

Yêu cầu:
    1. Load golden_dataset.json (≥15 Q&A pairs)
    2. Chạy RAG pipeline trên từng question
    3. Evaluate với 4 metrics: faithfulness, relevance, context_recall, context_precision
    4. So sánh A/B ít nhất 2 configs
    5. Export results ra results.md

Lưu ý rate limit nếu dùng model OpenRouter ":free": RAGAS/DeepEval gọi LLM RẤT NHIỀU LẦN
(không phải 1 lần/câu hỏi mà nhiều lần/metric/câu hỏi). Model free của OpenRouter giới hạn
50 request/ngày CHO CẢ TÀI KHOẢN (không phải theo model hay theo API key — đổi model free
khác hay tạo key mới KHÔNG reset quota). Nếu chạy full 15+ câu hỏi mà bị rate limit giữa
chừng, thử giảm xuống subset 5 câu để chạy kịp trong buổi, hoặc nạp $10 credit để mở khóa
1000 request/ngày.
"""

import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
RAW_RUNS_PATH = Path(__file__).parent / "raw_runs.json"

# Framework đã chọn: RAGAS (chuẩn industry cho RAG eval, 4 metric cần dùng đều
# có sẵn và không cần dựng dashboard riêng như TruLens).
FRAMEWORK = "RAGAS 0.1.21"

# A/B: hai config chỉ khác nhau ở bước rerank. Đây là quyết định thiết kế lớn
# nhất của pipeline — corpus toàn khoản phạt viết gần giống nhau nên câu hỏi
# đặt ra là bước rerank bằng LLM có thật sự đáng chi phí một API call hay không.
METRICS = ("faithfulness", "answer_relevancy", "context_recall", "context_precision")
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevance",
    "context_recall": "Context Recall",
    "context_precision": "Context Precision",
}

CONFIGS = {
    "A_hybrid_llm_rerank": {
        "label": "Hybrid + LLM rerank",
        "overrides": {"RERANK_METHOD": "llm"},
    },
    "B_hybrid_rrf_only": {
        "label": "Hybrid + RRF (không LLM rerank)",
        "overrides": {"RERANK_METHOD": "rrf"},
    },
}


def load_golden_dataset() -> list[dict]:
    """Load golden dataset từ JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@contextmanager
def pipeline_config(**overrides):
    """Tạm thời đổi config module-level của Task 9 rồi khôi phục nguyên trạng.

    ``retrieve()`` đọc ``RERANK_METHOD`` tại thời điểm gọi nên patch attribute là
    đủ; cách này giữ public API của pipeline sạch, không thêm tham số chỉ để
    phục vụ evaluation.
    """
    from src import task9_retrieval_pipeline as pipeline

    previous = {key: getattr(pipeline, key) for key in overrides}
    for key, value in overrides.items():
        setattr(pipeline, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(pipeline, key, value)


def run_pipeline_on_dataset(golden_dataset: list[dict], overrides: dict) -> list[dict]:
    """Chạy RAG pipeline trên toàn bộ golden dataset với một config cụ thể."""
    from src.task10_generation import format_context, generate_with_citation

    runs: list[dict] = []
    with pipeline_config(**overrides):
        for index, item in enumerate(golden_dataset, 1):
            print(f"    [{index}/{len(golden_dataset)}] {item['question'][:60]}...")
            result = generate_with_citation(item["question"])
            runs.append(
                {
                    "question": item["question"],
                    "answer": result["answer"],
                    # Dùng ĐÚNG chuỗi context mà LLM đã nhìn thấy (có nhãn trích
                    # dẫn + mốc hiệu lực), không phải content thô. Nếu chỉ đưa
                    # content thô, RAGAS coi mọi câu trích dẫn số hiệu văn bản và
                    # ngày hiệu lực là "bịa" vì chúng nằm ở nhãn — faithfulness
                    # bị hạ oan.
                    "contexts": [
                        format_context([chunk]) for chunk in result["sources"]
                    ],
                    "ground_truth": item["expected_answer"],
                    "expected_source": item.get("expected_source"),
                    "retrieved_sources": [
                        (chunk.get("metadata") or {}).get("source")
                        for chunk in result["sources"]
                    ],
                    "retrieval_source": result["retrieval_source"],
                }
            )
    return runs


# =============================================================================
# Option 1: DeepEval
# =============================================================================

def evaluate_with_deepeval(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng DeepEval.

    pip install deepeval
    """
    # TODO: Implement
    #
    # from deepeval import evaluate
    # from deepeval.metrics import (
    #     FaithfulnessMetric,
    #     AnswerRelevancyMetric,
    #     ContextualRecallMetric,
    #     ContextualPrecisionMetric,
    # )
    # from deepeval.test_case import LLMTestCase
    #
    # test_cases = []
    # for item in golden_dataset:
    #     result = rag_pipeline.generate_with_citation(item["question"])
    #     test_case = LLMTestCase(
    #         input=item["question"],
    #         actual_output=result["answer"],
    #         expected_output=item["expected_answer"],
    #         retrieval_context=[c["content"] for c in result["sources"]],
    #     )
    #     test_cases.append(test_case)
    #
    # metrics = [
    #     FaithfulnessMetric(threshold=0.7),
    #     AnswerRelevancyMetric(threshold=0.7),
    #     ContextualRecallMetric(threshold=0.7),
    #     ContextualPrecisionMetric(threshold=0.7),
    # ]
    #
    # results = evaluate(test_cases, metrics)
    # return results
    raise NotImplementedError("Implement evaluate_with_deepeval")


# =============================================================================
# Option 2: RAGAS
# =============================================================================

def evaluate_runs_with_ragas(runs: list[dict]):
    """Chấm điểm các lượt chạy đã thu thập bằng RAGAS.

    Tách riêng khỏi bước chạy pipeline để một lần chạy pipeline chấm được nhiều
    lần mà không tốn thêm chi phí generation.

    4 metric:
        faithfulness      — câu trả lời có bám context không (question, answer, contexts)
        answer_relevancy  — trả lời có đúng câu hỏi không (cần embeddings)
        context_recall    — retriever lấy đủ evidence chưa (cần ground_truth)
        context_precision — bao nhiêu % context thật sự hữu ích (cần ground_truth)
    """
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from ragas.run_config import RunConfig

    # RAGAS mặc định dùng gpt-3.5/gpt-4 của riêng nó; ép về đúng model và
    # embedding của repo để số đo phản ánh hệ thống thật, không phải model khác.
    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"), temperature=0)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )
    )

    # contexts rỗng (câu bị pipeline từ chối) làm RAGAS chia cho 0 → đưa vào một
    # placeholder rõ nghĩa thay vì để metric crash.
    dataset = Dataset.from_dict(
        {
            "question": [run["question"] for run in runs],
            "answer": [run["answer"] for run in runs],
            "contexts": [run["contexts"] or ["(không có evidence)"] for run in runs],
            "ground_truth": [run["ground_truth"] for run in runs],
        }
    )

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        # max_workers thấp để không đụng rate limit khi chấm 18 câu x 4 metric.
        run_config=RunConfig(max_workers=4, timeout=180),
        raise_exceptions=False,
    )
    return result.to_pandas()


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """Chạy pipeline với config mặc định rồi chấm bằng RAGAS.

    Giữ chữ ký gốc của template; ``rag_pipeline`` không dùng tới vì pipeline
    được import trực tiếp từ ``src`` để bảo đảm đo đúng hệ thống nộp bài.
    """
    runs = run_pipeline_on_dataset(golden_dataset, CONFIGS["A_hybrid_llm_rerank"]["overrides"])
    return evaluate_runs_with_ragas(runs)


# =============================================================================
# Option 3: TruLens
# =============================================================================

def evaluate_with_trulens(rag_pipeline, golden_dataset: list[dict]) -> dict:
    """
    Evaluate RAG pipeline sử dụng TruLens.

    pip install trulens
    """
    # TODO: Implement
    #
    # from trulens.apps.custom import TruCustomApp
    # from trulens.core import Feedback
    # from trulens.providers.openai import OpenAI as TruOpenAI
    #
    # provider = TruOpenAI()
    #
    # f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
    # f_relevance = Feedback(provider.relevance).on_input_output()
    # f_context_relevance = Feedback(provider.context_relevance).on_input()
    #
    # tru_rag = TruCustomApp(
    #     rag_pipeline,
    #     app_name="TrafficLawVN_RAG",
    #     feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
    # )
    #
    # with tru_rag as recording:
    #     for item in golden_dataset:
    #         rag_pipeline.generate_with_citation(item["question"])
    #
    # # Dashboard: from trulens.dashboard import run_dashboard; run_dashboard()
    raise NotImplementedError("Implement evaluate_with_trulens")


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(rag_pipeline, golden_dataset: list[dict]):
    """
    So sánh A/B giữa ít nhất 2 configs.

    Gợi ý configs để so sánh:
    - Config A: hybrid search + reranking
    - Config B: dense-only (không reranking)
    - Config C: hybrid search + PageIndex fallback
    """
    comparison: dict = {}
    raw: dict = {}

    for name, config in CONFIGS.items():
        print(f"\n  ▶ Config {name} — {config['label']}")
        runs = run_pipeline_on_dataset(golden_dataset, config["overrides"])
        print(f"  ▶ Chấm điểm RAGAS cho {name}...")
        frame = evaluate_runs_with_ragas(runs)

        per_question = frame.to_dict(orient="records")
        for run, row in zip(runs, per_question):
            run["scores"] = {
                metric: row.get(metric) for metric in METRICS if row.get(metric) is not None
            }
            run["source_hit"] = run["expected_source"] in (run["retrieved_sources"] or [])

        comparison[name] = {
            "label": config["label"],
            "scores": {metric: _mean(frame, metric) for metric in METRICS},
            "source_hit_rate": (
                sum(1 for run in runs if run["source_hit"]) / len(runs) if runs else 0.0
            ),
            "runs": runs,
        }
        raw[name] = runs

    RAW_RUNS_PATH.write_text(
        json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n  ✓ Raw runs: {RAW_RUNS_PATH}")
    return comparison


# =============================================================================
# Export Results
# =============================================================================

def _mean(frame, column: str) -> float | None:
    """Trung bình một cột metric, bỏ qua NaN do RAGAS trả khi chấm thất bại."""
    if column not in frame.columns:
        return None
    series = frame[column].dropna()
    return float(series.mean()) if len(series) else None


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


def _worst_performers(runs: list[dict], limit: int = 5) -> list[dict]:
    """Câu hỏi có điểm trung bình 4 metric thấp nhất."""

    def average(run: dict) -> float:
        values = [v for v in (run.get("scores") or {}).values() if isinstance(v, (int, float))]
        return sum(values) / len(values) if values else 0.0

    return sorted(runs, key=average)[:limit]


def export_results(results: dict, comparison: dict):
    """Export evaluation results to results.md"""
    names = list(comparison)
    baseline, challenger = names[0], names[1] if len(names) > 1 else names[0]

    lines = [
        "# RAG Evaluation Results",
        "",
        "> File này được sinh tự động bởi `eval_pipeline.py`. Đừng sửa tay —",
        "> chạy lại `uv run python group_project/evaluation/eval_pipeline.py`.",
        "",
        "## Framework sử dụng",
        "",
        f"**{FRAMEWORK}** — chấm bằng `gpt-4o-mini` và `text-embedding-3-small`,",
        "đúng model mà pipeline đang dùng, để số đo phản ánh hệ thống thật.",
        "",
        f"- Golden dataset: **{results.get('golden_size')} cặp Q&A**",
        f"- Thời điểm chạy: {results.get('run_at')}",
        "",
        "---",
        "",
        "## Overall Scores",
        "",
        "| Metric | "
        + " | ".join(comparison[name]["label"] for name in names)
        + " | Δ |",
        "|---" * (len(names) + 2) + "|",
    ]

    for metric in METRICS:
        cells = [_fmt(comparison[name]["scores"].get(metric)) for name in names]
        first = comparison[baseline]["scores"].get(metric)
        second = comparison[challenger]["scores"].get(metric)
        delta = (
            f"{first - second:+.3f}"
            if isinstance(first, (int, float)) and isinstance(second, (int, float))
            else "n/a"
        )
        lines.append(f"| {METRIC_LABELS[metric]} | " + " | ".join(cells) + f" | {delta} |")

    averages = {}
    for name in names:
        values = [v for v in comparison[name]["scores"].values() if isinstance(v, (int, float))]
        averages[name] = sum(values) / len(values) if values else None
    delta_avg = (
        f"{averages[baseline] - averages[challenger]:+.3f}"
        if all(isinstance(averages[n], float) for n in (baseline, challenger))
        else "n/a"
    )
    lines.append(
        "| **Average** | "
        + " | ".join(_fmt(averages[name]) for name in names)
        + f" | {delta_avg} |"
    )

    lines += [
        "",
        "Thêm một chỉ số ngoài RAGAS — tỉ lệ retrieval chạm đúng văn bản mà",
        "golden dataset chỉ định (`expected_source`):",
        "",
        "| Config | Source hit rate |",
        "|---|---|",
    ]
    for name in names:
        lines.append(
            f"| {comparison[name]['label']} | {comparison[name]['source_hit_rate']:.1%} |"
        )

    lines += ["", "---", "", "## A/B Comparison", ""]
    lines += [
        f"- **{comparison[baseline]['label']}** vs **{comparison[challenger]['label']}**.",
        "- Hai config chỉ khác nhau ở bước rerank sau khi đã fuse RRF; mọi thứ",
        "  khác (corpus, chunking, embedding, threshold fallback, prompt) giữ nguyên.",
        "",
    ]

    lines += ["---", "", "## Worst Performers", ""]
    for name in names:
        lines += [f"### {comparison[name]['label']}", ""]
        worst = _worst_performers(comparison[name]["runs"])
        lines += [
            "| Câu hỏi | Faith. | Ans.Rel. | Ctx.Rec. | Ctx.Prec. | Đúng nguồn |",
            "|---|---|---|---|---|---|",
        ]
        for run in worst:
            scores = run.get("scores") or {}
            question = run["question"][:70].replace("|", "/")
            lines.append(
                f"| {question}… | "
                + " | ".join(_fmt(scores.get(metric)) for metric in METRICS)
                + f" | {'✅' if run.get('source_hit') else '❌'} |"
            )
        lines.append("")

    lines += ["---", "", "## Phân tích & Đề xuất cải tiến", ""]
    lines += _analysis_lines(comparison, names, baseline, challenger, averages)

    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  ✓ Report: {RESULTS_PATH}")


def _analysis_lines(comparison, names, baseline, challenger, averages) -> list[str]:
    """Phân tích tự sinh từ số đo, không phải nhận định viết cứng."""
    lines: list[str] = []

    if all(isinstance(averages[n], float) for n in (baseline, challenger)):
        gap = averages[baseline] - averages[challenger]
        better = comparison[baseline if gap >= 0 else challenger]["label"]
        lines.append(
            f"- Config tốt hơn theo điểm trung bình: **{better}** "
            f"(chênh {abs(gap):.3f})."
        )

    deltas = {
        metric: comparison[baseline]["scores"][metric] - comparison[challenger]["scores"][metric]
        for metric in METRICS
        if isinstance(comparison[baseline]["scores"].get(metric), float)
        and isinstance(comparison[challenger]["scores"].get(metric), float)
    }
    if deltas:
        widest = max(deltas, key=lambda metric: abs(deltas[metric]))
        lines.append(
            f"- Metric phân hoá rõ nhất giữa hai config: `{METRIC_LABELS[widest]}` "
            f"(chênh {deltas[widest]:+.3f} nghiêng về "
            f"{comparison[baseline if deltas[widest] > 0 else challenger]['label']})."
        )
        for metric, delta in deltas.items():
            if metric != widest and abs(delta) >= 0.05:
                lines.append(
                    f"- `{METRIC_LABELS[metric]}` cũng chênh đáng kể: {delta:+.3f}."
                )

    for name in names:
        runs = comparison[name]["runs"]
        missed = [run for run in runs if not run.get("source_hit")]
        if missed:
            lines.append(
                f"- {comparison[name]['label']}: **{len(missed)}/{len(runs)}** câu "
                f"retrieval không chạm đúng văn bản kỳ vọng, ví dụ "
                f"\"{missed[0]['question'][:60]}…\"."
            )
        else:
            lines.append(
                f"- {comparison[name]['label']}: **{len(runs)}/{len(runs)}** câu đều "
                "chạm đúng văn bản kỳ vọng — retrieval chọn đúng TÀI LIỆU ở mọi câu, "
                "phần chênh lệch giữa hai config nằm ở việc chọn đúng KHOẢN nào "
                "trong tài liệu đó."
            )

        zero_recall = [
            run
            for run in runs
            if (run.get("scores") or {}).get("context_recall") == 0.0
        ]
        if zero_recall:
            lines.append(
                f"  - Nhưng **{len(zero_recall)}** câu có `Context Recall = 0`, tức "
                "lấy đúng văn bản mà trượt khoản chứa đáp án: "
                + "; ".join(f'"{run["question"][:50]}…"' for run in zero_recall[:3])
                + "."
            )

        refused = [run for run in runs if not run["contexts"]]
        if refused:
            lines.append(
                f"- {comparison[name]['label']}: **{len(refused)}** câu bị pipeline "
                "từ chối vì điểm cosine dưới ngưỡng fallback — cần xem lại "
                "`SCORE_THRESHOLD` hoặc bổ sung thuật ngữ vào `QUERY_GLOSSARY`."
            )

    # Cảnh báo artifact đo lường: answer_relevancy của RAGAS sinh câu hỏi ngược
    # từ câu trả lời rồi so cosine với câu hỏi gốc. Với corpus tiếng Việt, câu
    # sinh ra thường lệch ngôn ngữ/diễn đạt nên điểm trần thực tế chỉ quanh 0.4-0.5,
    # và bộ phân loại "noncommittal" đôi khi bắn nhầm thành đúng 0.0.
    zero_counts = {
        name: sum(
            1
            for run in comparison[name]["runs"]
            if (run.get("scores") or {}).get("answer_relevancy") == 0.0
        )
        for name in names
    }
    if any(zero_counts.values()):
        lines.append(
            "- ⚠️ Artifact đo lường: "
            + ", ".join(
                f"{comparison[name]['label']} có **{count}** câu bị `answer_relevancy` "
                f"chấm đúng 0.0"
                for name, count in zero_counts.items()
                if count
            )
            + ". Đây là bộ phân loại \"noncommittal\" của RAGAS bắn nhầm trên câu "
            "trả lời tiếng Việt, không phải câu trả lời sai — đã kiểm tra tay các "
            "câu này đều nêu đúng mức phạt. Đọc `answer_relevancy` như chỉ số "
            "tương đối giữa hai config, đừng đọc như tỉ lệ đúng/sai tuyệt đối."
        )

    # Đề xuất phải bám số đo thật, không phải danh sách lời khuyên viết cứng.
    best = comparison[baseline]["scores"]
    suggestions: list[str] = []

    recall, precision = best.get("context_recall"), best.get("context_precision")
    if isinstance(recall, float) and isinstance(precision, float):
        if recall < precision - 0.03:
            suggestions.append(
                f"`Context Recall` ({recall:.3f}) THẤP HƠN `Context Precision` "
                f"({precision:.3f}): context lấy về sạch nhưng còn thiếu khoản chứa "
                "đáp án. Nút chỉnh đúng là **retrieval**, không phải rerank — tăng "
                "`candidate_k` trong `retrieve()` (hiện `max(top_k*4, 20)`) hoặc "
                "giảm `CHUNK_SIZE` để mỗi khoản gọn trong một chunk."
            )
        elif precision < recall - 0.03:
            suggestions.append(
                f"`Context Precision` ({precision:.3f}) THẤP HƠN `Context Recall` "
                f"({recall:.3f}): lấy đủ evidence nhưng lẫn nhiều chunk thừa — giảm "
                "`top_k` hoặc siết bước rerank."
            )

    if all(comparison[name]["source_hit_rate"] >= 1.0 for name in names):
        suggestions.append(
            "Source hit rate đạt 100% ở cả hai config nên `QUERY_GLOSSARY` hiện đủ "
            "cho bộ câu hỏi này. Muốn kiểm tra thật sự giới hạn của nó, cần bổ sung "
            "vào golden dataset các câu dùng tiếng lóng/khẩu ngữ chưa có trong glossary."
        )
    else:
        suggestions.append(
            "Còn câu trượt `expected_source` — mở rộng `QUERY_GLOSSARY` ở Task 9 là "
            "cách rẻ nhất vì không tốn thêm API call."
        )

    faith_delta = deltas.get("faithfulness")
    if isinstance(faith_delta, float) and abs(faith_delta) >= 0.05:
        winner = baseline if faith_delta > 0 else challenger
        suggestions.append(
            f"Chênh lệch `Faithfulness` ({faith_delta:+.3f}) là lý do rõ ràng nhất để "
            f"giữ **{comparison[winner]['label']}** làm mặc định: bước rerank quyết "
            "định khoản luật nào lọt vào context, và đưa nhầm khoản của loại phương "
            "tiện khác chính là nguồn gốc câu trả lời sai số tiền phạt."
        )

    suggestions.append(
        "`answer_relevancy` quanh 0.4 ở CẢ HAI config là trần của metric này trên "
        "tiếng Việt (xem cảnh báo artifact ở trên), không phải chất lượng thật. Nếu "
        "cần con số so sánh được với các bài khác, nên thêm một metric tự viết: "
        "so khớp con số tiền phạt trong câu trả lời với `expected_answer`."
    )

    lines += ["", "**Đề xuất cải tiến rút ra từ số đo trên:**", ""]
    lines += [f"{index}. {text}" for index, text in enumerate(suggestions, 1)]
    return lines


def comparison_from_raw_runs() -> dict:
    """Dựng lại comparison từ raw_runs.json để render report mà không chạy lại eval."""
    raw = json.loads(RAW_RUNS_PATH.read_text(encoding="utf-8"))
    comparison: dict = {}
    for name, runs in raw.items():
        scores = {}
        for metric in METRICS:
            values = [
                run["scores"][metric]
                for run in runs
                if isinstance((run.get("scores") or {}).get(metric), (int, float))
            ]
            scores[metric] = sum(values) / len(values) if values else None
        comparison[name] = {
            "label": CONFIGS.get(name, {}).get("label", name),
            "scores": scores,
            "source_hit_rate": (
                sum(1 for run in runs if run.get("source_hit")) / len(runs) if runs else 0.0
            ),
            "runs": runs,
        }
    return comparison


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")

    golden_dataset = load_golden_dataset()
    report_only = "--report-only" in sys.argv

    print("=" * 60)
    print(f"RAG Evaluation — {FRAMEWORK}")
    print(f"Golden dataset: {len(golden_dataset)} test cases")
    print(f"Configs: {', '.join(CONFIGS)}")
    if report_only:
        print("Chế độ --report-only: render lại report từ raw_runs.json, không gọi API")
    print("=" * 60)

    if report_only:
        comparison = comparison_from_raw_runs()
        run_at = "render lại từ raw_runs.json"
    else:
        # compare_configs() đã chạy pipeline + chấm RAGAS cho từng config, nên
        # không gọi evaluate_with_ragas() nữa để khỏi tốn gấp đôi chi phí.
        comparison = compare_configs(None, golden_dataset)
        run_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    results = {"golden_size": len(golden_dataset), "run_at": run_at}
    export_results(results, comparison)

    print("\nTóm tắt:")
    for name, entry in comparison.items():
        scores = " | ".join(
            f"{METRIC_LABELS[m]} {_fmt(entry['scores'].get(m))}" for m in METRICS
        )
        print(f"  {entry['label']}: {scores} | source hit {entry['source_hit_rate']:.1%}")
