import os
import random
import time
import threading
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import gradio as gr
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import evaluation.eval as eval_module
# import implementation.answer as answer_module
import pro_implementation.answer as answer_module
from evaluation.eval import evaluate_all_retrieval, evaluate_answer
from evaluation.test import load_tests

load_dotenv(override=True)

SEED = 42
CPU_COUNT = os.cpu_count() or 4
DEFAULT_MAX_WORKERS = max(2, min(8, CPU_COUNT // 2))
DEFAULT_CONCURRENCY = max(2, min(6, DEFAULT_MAX_WORKERS))
MAX_WORKERS_LIMIT = max(4, min(16, CPU_COUNT * 2))

# Color coding thresholds - Retrieval
MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

# Color coding thresholds - Answer (1-5 scale)
ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0


def configure_reproducibility(seed: int = SEED) -> None:
    """Apply best-effort reproducibility settings without changing legacy files."""
    random.seed(seed)
    np.random.seed(seed)

    original_completion = eval_module.completion

    def seeded_completion(*args, **kwargs):
        kwargs.setdefault("seed", seed)
        return original_completion(*args, **kwargs)

    eval_module.completion = seeded_completion
    answer_module.llm = ChatOpenAI(
        temperature=0,
        model_name=answer_module.MODEL,
        seed=seed,
    )


configure_reproducibility()


def get_color(value: float, metric_type: str) -> str:
    """Get color based on metric value and type."""
    if metric_type == "mrr":
        if value >= MRR_GREEN:
            return "green"
        if value >= MRR_AMBER:
            return "orange"
        return "red"
    if metric_type == "ndcg":
        if value >= NDCG_GREEN:
            return "green"
        if value >= NDCG_AMBER:
            return "orange"
        return "red"
    if metric_type == "coverage":
        if value >= COVERAGE_GREEN:
            return "green"
        if value >= COVERAGE_AMBER:
            return "orange"
        return "red"
    if metric_type in ["accuracy", "completeness", "relevance"]:
        if value >= ANSWER_GREEN:
            return "green"
        if value >= ANSWER_AMBER:
            return "orange"
        return "red"
    return "black"


def format_metric_html(
    label: str,
    value: float,
    metric_type: str,
    is_percentage: bool = False,
    score_format: bool = False,
) -> str:
    """Format a metric with color coding."""
    color = get_color(value, metric_type)
    if is_percentage:
        value_str = f"{value:.1f}%"
    elif score_format:
        value_str = f"{value:.2f}/5"
    else:
        value_str = f"{value:.4f}"
    return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>
    """


def normalize_settings(max_workers: float, concurrency: float) -> tuple[int, int]:
    """Convert UI values into safe executor settings."""
    workers = max(1, min(int(max_workers), MAX_WORKERS_LIMIT))
    in_flight = max(1, min(int(concurrency), workers))
    return workers, in_flight


def evaluate_answer_only(test):
    """Evaluate a single answer and return only the scoring object."""
    return evaluate_answer(test)[0]


def run_retrieval_evaluation(progress=gr.Progress()):
    """Run retrieval evaluation and yield updates."""
    total_mrr = 0.0
    total_ndcg = 0.0
    total_coverage = 0.0
    category_mrr = defaultdict(list)
    count = 0

    for test, result, prog_value in evaluate_all_retrieval():
        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage

        category_mrr[test.category].append(result.mrr)
        progress(prog_value, desc=f"Evaluating test {count}...")

    avg_mrr = total_mrr / count
    avg_ndcg = total_ndcg / count
    avg_coverage = total_coverage / count

    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
        {format_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg")}
        {format_metric_html("Keyword Coverage", avg_coverage, "coverage", is_percentage=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">âœ“ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

    category_data = []
    for category, mrr_scores in category_mrr.items():
        avg_category_mrr = sum(mrr_scores) / len(mrr_scores)
        category_data.append({"Category": category, "Average MRR": avg_category_mrr})

    df = pd.DataFrame(category_data)
    return final_html, df


def build_summary_html(
    avg_accuracy: float,
    avg_completeness: float,
    avg_relevance: float,
    completed: int,
    failed: int,
    elapsed_seconds: float,
    workers: int,
    concurrency: int,
) -> str:
    """Render the final summary block."""
    status_color = "#d4edda" if failed == 0 else "#fff3cd"
    border_color = "#c3e6cb" if failed == 0 else "#ffe69c"
    text_color = "#155724" if failed == 0 else "#856404"
    status_text = (
        f"Evaluation Complete: {completed} tests"
        if failed == 0
        else f"Evaluation Complete: {completed} succeeded, {failed} failed"
    )

    return f"""
    <div style="padding: 0;">
        {format_metric_html("Accuracy", avg_accuracy, "accuracy", score_format=True)}
        {format_metric_html("Completeness", avg_completeness, "completeness", score_format=True)}
        {format_metric_html("Relevance", avg_relevance, "relevance", score_format=True)}
        <div style="margin-top: 20px; padding: 14px; background-color: {status_color}; border-radius: 5px; border: 1px solid {border_color};">
            <div style="font-size: 14px; color: {text_color}; font-weight: bold;">{status_text}</div>
            <div style="font-size: 13px; color: {text_color}; margin-top: 6px;">
                Elapsed: {elapsed_seconds:.1f}s | max_workers: {workers} | concurrency: {concurrency}
            </div>
        </div>
    </div>
    """


def run_answer_evaluation(
    max_workers: float = DEFAULT_MAX_WORKERS,
    concurrency: float = DEFAULT_CONCURRENCY,
    progress=gr.Progress(),
):
    """Run answer evaluation concurrently while preserving old scoring logic."""
    workers, in_flight_limit = normalize_settings(max_workers, concurrency)
    tests = load_tests()
    total_tests = len(tests)

    total_accuracy = 0.0
    total_completeness = 0.0
    total_relevance = 0.0
    category_accuracy = defaultdict(list)
    completed = 0
    failed = 0
    next_index = 0
    futures = {}
    lock = threading.Lock()
    start_time = time.perf_counter()

    progress(0, desc="Starting concurrent answer evaluation...")

    def submit_more(executor: ThreadPoolExecutor) -> None:
        nonlocal next_index
        while next_index < total_tests and len(futures) < in_flight_limit:
            test = tests[next_index]
            future = executor.submit(evaluate_answer_only, test)
            futures[future] = test
            next_index += 1

    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="answer-eval") as executor:
        submit_more(executor)

        while futures:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                test = futures.pop(future)
                try:
                    result = future.result()
                    with lock:
                        total_accuracy += result.accuracy
                        total_completeness += result.completeness
                        total_relevance += result.relevance
                        category_accuracy[test.category].append(result.accuracy)
                except Exception:
                    failed += 1

                completed += 1
                progress(
                    completed / total_tests,
                    desc=(
                        f"Completed {completed}/{total_tests} tests "
                        f"(workers={workers}, concurrency={in_flight_limit})"
                    ),
                )

            submit_more(executor)

    successful = completed - failed
    if successful == 0:
        raise RuntimeError("All answer evaluations failed.")

    avg_accuracy = total_accuracy / successful
    avg_completeness = total_completeness / successful
    avg_relevance = total_relevance / successful
    elapsed_seconds = time.perf_counter() - start_time

    final_html = build_summary_html(
        avg_accuracy=avg_accuracy,
        avg_completeness=avg_completeness,
        avg_relevance=avg_relevance,
        completed=successful,
        failed=failed,
        elapsed_seconds=elapsed_seconds,
        workers=workers,
        concurrency=in_flight_limit,
    )

    category_data = []
    for category, accuracy_scores in category_accuracy.items():
        avg_category_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        category_data.append({"Category": category, "Average Accuracy": avg_category_accuracy})

    df = pd.DataFrame(category_data)
    return final_html, df


def main():
    """Launch the accelerated Gradio evaluation app."""
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Evaluation Dashboard", theme=theme) as app:
        gr.Markdown("#RAG Evaluation Dashboard")
        gr.Markdown("Evaluate retrieval and answer quality for the Insurellm RAG system")

        # RETRIEVAL SECTION
        gr.Markdown("##Retrieval Evaluation")

        retrieval_button = gr.Button("Run Evaluation", variant="primary", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                retrieval_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
                )

            with gr.Column(scale=1):
                retrieval_chart = gr.BarPlot(
                    x="Category",
                    y="Average MRR",
                    title="Average MRR by Category",
                    y_lim=[0, 1],
                    height=400,
                )

        # ANSWERING SECTION
        gr.Markdown("##Answer Evaluation")

        answer_button = gr.Button("Run Evaluation", variant="primary", size="lg")

        with gr.Row():
            max_workers_input = gr.Slider(
                minimum=1,
                maximum=MAX_WORKERS_LIMIT,
                value=DEFAULT_MAX_WORKERS,
                step=1,
                label="max_workers",
                info=f"Detected CPU count: {CPU_COUNT}. Default chosen for mixed I/O workload.",
            )
            concurrency_input = gr.Slider(
                minimum=1,
                maximum=MAX_WORKERS_LIMIT,
                value=DEFAULT_CONCURRENCY,
                step=1,
                label="concurrency",
                info="Maximum number of tests kept in flight at the same time.",
            )

        with gr.Row():
            with gr.Column(scale=1):
                answer_metrics = gr.HTML(
                    "<div style='padding: 20px; text-align: center; color: #999;'>Click 'Run Evaluation' to start</div>"
                )

            with gr.Column(scale=1):
                answer_chart = gr.BarPlot(
                    x="Category",
                    y="Average Accuracy",
                    title="Average Accuracy by Category",
                    y_lim=[1, 5],
                    height=400,
                )

        retrieval_button.click(
            fn=run_retrieval_evaluation,
            outputs=[retrieval_metrics, retrieval_chart],
        )

        answer_button.click(
            fn=run_answer_evaluation,
            inputs=[max_workers_input, concurrency_input],
            outputs=[answer_metrics, answer_chart],
        )

    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()
