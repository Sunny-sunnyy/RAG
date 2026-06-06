# File: Agentic-RAG-InsureLLM/evaluator.py
# Giao diện Gradio Dashboard chạy đánh giá tuần tự của Agentic RAG và lưu trữ kết quả đánh giá câu trả lời dạng JSONL.

import gradio as gr
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
import json

from evaluation.eval import evaluate_all_retrieval, evaluate_all_answers

# Tải cấu hình biến môi trường
load_dotenv(override=True)

# Định nghĩa ngưỡng màu sắc để hiển thị kết quả cho khâu Retrieval
MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

# Định nghĩa ngưỡng màu sắc để hiển thị kết quả cho khâu sinh câu trả lời (thang điểm 1-5)
ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0


# Hàm get_color: Xác định màu sắc hiển thị dựa trên giá trị và loại metric để trực quan hoá trên giao diện.
def get_color(value: float, metric_type: str) -> str:
    """Get color based on metric value and type."""
    if metric_type == "mrr":
        if value >= MRR_GREEN:
            return "green"
        elif value >= MRR_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "ndcg":
        if value >= NDCG_GREEN:
            return "green"
        elif value >= NDCG_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type == "coverage":
        if value >= COVERAGE_GREEN:
            return "green"
        elif value >= COVERAGE_AMBER:
            return "orange"
        else:
            return "red"
    elif metric_type in ["accuracy", "completeness", "relevance"]:
        if value >= ANSWER_GREEN:
            return "green"
        elif value >= ANSWER_AMBER:
            return "orange"
        else:
            return "red"
    return "black"


# Hàm format_metric_html: Định dạng hiển thị HTML cho từng metric, đổi màu đường viền dựa trên điểm số.
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


# Hàm run_retrieval_evaluation: Chạy tuần tự đánh giá tất cả test cases cho khâu truy xuất (Retrieval).
def run_retrieval_evaluation(progress=gr.Progress()):
    """Run retrieval evaluation and yield updates."""
    total_mrr = 0.0
    total_ndcg = 0.0
    total_coverage = 0.0
    category_mrr = defaultdict(list)
    count = 0

    # Lặp qua generator đánh giá retrieval
    for test, result, prog_value in evaluate_all_retrieval():
        count += 1
        total_mrr += result.mrr
        total_ndcg += result.ndcg
        total_coverage += result.keyword_coverage

        category_mrr[test.category].append(result.mrr)

        # Cập nhật thanh tiến trình Gradio
        progress(prog_value, desc=f"Evaluating test {count}...")

    # Tính trung bình các chỉ số
    avg_mrr = total_mrr / count
    avg_ndcg = total_ndcg / count
    avg_coverage = total_coverage / count

    # Xây dựng HTML hiển thị tổng kết
    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
        {format_metric_html("Normalized Discounted Cumulative Gain (nDCG)", avg_ndcg, "ndcg")}
        {format_metric_html("Keyword Coverage", avg_coverage, "coverage", is_percentage=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

    # Chuẩn bị dữ liệu vẽ biểu đồ MRR theo danh mục
    category_data = []
    for category, mrr_scores in category_mrr.items():
        avg_cat_mrr = sum(mrr_scores) / len(mrr_scores)
        category_data.append({"Category": category, "Average MRR": avg_cat_mrr})

    df = pd.DataFrame(category_data)

    return final_html, df


# Hàm save_answer_eval_jsonl: Ghi danh sách kết quả chi tiết của phiên đánh giá câu trả lời vào file định dạng JSON Lines (.jsonl).
def save_answer_eval_jsonl(test_results, filepath="answer_eval.jsonl"):
    with open(filepath, "w", encoding="utf-8") as f:
        for row in test_results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Answer evaluation saved to {filepath}")


# Hàm run_answer_evaluation: Thực hiện chạy đánh giá chất lượng sinh câu trả lời và lưu vết kết quả ra file JSONL.
def run_answer_evaluation(progress=gr.Progress()):
    """Run answer evaluation and yield updates (async)."""
    total_accuracy = 0.0
    total_completeness = 0.0
    total_relevance = 0.0
    category_accuracy = defaultdict(list)
    count = 0
    eval_json_list = []
    # Lặp qua generator đánh giá câu trả lời để lấy điểm và dữ liệu JSON
    for test, result, prog_value, eval_json in evaluate_all_answers():
        count += 1
        total_accuracy += result.accuracy
        total_completeness += result.completeness
        total_relevance += result.relevance

        category_accuracy[test.category].append(result.accuracy)
        eval_json_list.append(eval_json)
        # Cập nhật thanh tiến trình
        progress(prog_value, desc=f"Evaluating test {count}...")

    # Tính điểm trung bình của các chỉ số chất lượng
    avg_accuracy = total_accuracy / count
    avg_completeness = total_completeness / count
    avg_relevance = total_relevance / count

    # Ghi logs kết quả chi tiết ra file JSONL phục vụ phân tích sâu
    save_answer_eval_jsonl(eval_json_list, filepath="answer_eval.jsonl")

    # Tạo HTML báo cáo
    final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Accuracy", avg_accuracy, "accuracy", score_format=True)}
        {format_metric_html("Completeness", avg_completeness, "completeness", score_format=True)}
        {format_metric_html("Relevance", avg_relevance, "relevance", score_format=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

    # Chuẩn bị dữ liệu vẽ biểu đồ cột cho điểm Accuracy
    category_data = []
    for category, accuracy_scores in category_accuracy.items():
        avg_cat_accuracy = sum(accuracy_scores) / len(accuracy_scores)
        category_data.append({"Category": category, "Average Accuracy": avg_cat_accuracy})

    df = pd.DataFrame(category_data)

    return final_html, df


# Hàm main: Khởi dựng và thiết lập các thành phần giao diện Gradio Dashboard.
def main():
    """Launch the Gradio evaluation app."""
    theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

    with gr.Blocks(title="RAG Evaluation Dashboard", theme=theme) as app:
        gr.Markdown("# 📊 RAG Evaluation Dashboard")
        gr.Markdown("Evaluate retrieval and answer quality for the Insurellm RAG system")

        # KHU VỰC RETRIEVAL EVALUATION
        gr.Markdown("## 🔍 Retrieval Evaluation")

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

        # KHU VỰC ANSWER EVALUATION
        gr.Markdown("## 💬 Answer Evaluation")

        answer_button = gr.Button("Run Evaluation", variant="primary", size="lg")

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

        # Gắn các hàm callback tương ứng khi Click chuột
        retrieval_button.click(
            fn=run_retrieval_evaluation,
            outputs=[retrieval_metrics, retrieval_chart],
        )

        answer_button.click(
            fn=run_answer_evaluation,
            outputs=[answer_metrics, answer_chart],
        )

    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()