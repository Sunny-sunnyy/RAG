# File: Agentic-RAG-InsureLLM/evaluation/eval.py
# File này chứa toàn bộ logic đánh giá cho hai phần:
# 1) Retrieval: context có được kéo đúng và đủ không
# 2) Answer: câu trả lời cuối cùng có chính xác, đầy đủ, đúng trọng tâm không

import sys
import math
from pydantic import BaseModel, Field
from litellm import completion
from dotenv import load_dotenv

from evaluation.test import TestQuestion, load_tests
from implementation.answer import answer_question, fetch_context_router


load_dotenv(override=True)

MODEL = "gpt-4.1-nano"
db_name = "preprocessed_hierarchic_db"

# Schema này gom các chỉ số dùng để đánh giá phần retrieval.
class RetrievalEval(BaseModel):
    """Evaluation metrics for retrieval performance."""

    mrr: float = Field(description="Mean Reciprocal Rank - average across all keywords")
    ndcg: float = Field(description="Normalized Discounted Cumulative Gain (binary relevance)")
    keywords_found: int = Field(description="Number of keywords found in top-k results")
    total_keywords: int = Field(description="Total number of keywords to find")
    keyword_coverage: float = Field(description="Percentage of keywords found")


# Schema này gom điểm số và feedback khi dùng LLM làm judge cho câu trả lời.
class AnswerEval(BaseModel):
    """LLM-as-a-judge evaluation of answer quality."""

    feedback: str = Field(
        description="Concise feedback on the answer quality, comparing it to the reference answer and evaluating based on the retrieved context"
    )
    accuracy: float = Field(
        description="How factually correct is the answer compared to the reference answer? 1 (wrong. any wrong answer must score 1) to 5 (ideal - perfectly accurate). An acceptable answer would score 3."
    )
    completeness: float = Field(
        description="How complete is the answer in addressing all aspects of the question? 1 (very poor - missing key information) to 5 (ideal - all the information from the reference answer is provided completely). Only answer 5 if ALL information from the reference answer is included."
    )
    relevance: float = Field(
        description="How relevant is the answer to the specific question asked? 1 (very poor - off-topic) to 5 (ideal - directly addresses question and gives no additional information). Only answer 5 if the answer is completely relevant to the question and gives no additional information."
    )


# Tính reciprocal rank cho từng keyword để biết keyword quan trọng xuất hiện sớm đến đâu.
def calculate_mrr(keyword: str, retrieved_docs: list) -> float:
    """Calculate reciprocal rank for a single keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    for rank, doc in enumerate(retrieved_docs, start=1):
        if keyword_lower in doc.parent_headline.lower() + "\n" + doc.parent_chunk.lower() + "\n" + doc.child_chunks[0].lower() :
            return 1.0 / rank
    return 0.0


# Hàm phụ này tính DCG cho một danh sách độ liên quan đã biết.
def calculate_dcg(relevances: list[int], k: int) -> float:
    """Calculate Discounted Cumulative Gain."""
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


# nDCG giúp đo không chỉ "có tìm thấy không" mà còn "tìm thấy sớm hay muộn".
def calculate_ndcg(keyword: str, retrieved_docs: list, k: int = 10) -> float:
    """Calculate nDCG for a single keyword (binary relevance, case-insensitive)."""
    keyword_lower = keyword.lower()

    # Binary relevance: 1 if keyword found, 0 otherwise
    relevances = [
        1 if keyword_lower in doc.parent_headline.lower() + doc.parent_chunk.lower() + "\n" + doc.child_chunks[0].lower() else 0 for doc in retrieved_docs[:k]
    ]

    # DCG
    dcg = calculate_dcg(relevances, k)

    # Ideal DCG (best case: keyword in first position)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


# Đánh giá phần retrieval của một câu hỏi test bằng bộ chỉ số chuẩn.
def evaluate_retrieval(test: TestQuestion, k: int = 10) -> RetrievalEval:
    """
    Evaluate retrieval performance for a test question.

    Args:
        test: TestQuestion object containing question and keywords
        k: Number of top documents to retrieve (default 10)

    Returns:
        RetrievalEval object with MRR, nDCG, and keyword coverage metrics
    """
    # Dùng chính retrieval runtime của hệ thống để phép đo bám sát hành vi thật.
    retrieved_docs = fetch_context_router(test.question)

    # Tính điểm MRR trung bình trên toàn bộ keyword vàng của test case.
    mrr_scores = [calculate_mrr(keyword, retrieved_docs) for keyword in test.keywords]
    avg_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0

    # Tính nDCG trung bình để phản ánh chất lượng thứ hạng retrieval.
    ndcg_scores = [calculate_ndcg(keyword, retrieved_docs, k) for keyword in test.keywords]
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # Coverage trả lời câu hỏi đơn giản: có kéo được đủ keyword cần thiết hay chưa.
    keywords_found = sum(1 for score in mrr_scores if score > 0)
    total_keywords = len(test.keywords)
    keyword_coverage = (keywords_found / total_keywords * 100) if total_keywords > 0 else 0.0

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keyword_coverage=keyword_coverage,
    )


# Đánh giá câu trả lời cuối cùng bằng LLM-as-a-judge.
def evaluate_answer(test: TestQuestion) -> tuple[AnswerEval, str, list]:
    """
    Evaluate answer quality using LLM-as-a-judge (async).

    Args:
        test: TestQuestion object containing question and reference answer

    Returns:
        Tuple of (AnswerEval object, generated_answer string, retrieved_docs list)
    """
    # Lấy đúng answer do hệ thống sinh ra để chấm, thay vì chấm một output giả lập.
    generated_answer, retrieved_docs = answer_question(test.question)

    # Prompt judge so sánh generated answer với reference answer theo 3 tiêu chí.
    judge_messages = [
        {
            "role": "system",
            "content": "You are an expert evaluator assessing the quality of answers. Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.",
        },
        {
            "role": "user",
            "content": f"""Question:
{test.question}

Generated Answer:
{generated_answer}

Reference Answer:
{test.reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.""",
        },
    ]

    # Yêu cầu output có cấu trúc để parse ổn định sang schema AnswerEval.
    judge_response = completion(model=MODEL, messages=judge_messages, response_format=AnswerEval)

    answer_eval = AnswerEval.model_validate_json(judge_response.choices[0].message.content)

    return answer_eval, generated_answer, retrieved_docs


# Generator này yield kết quả retrieval theo từng test để dashboard cập nhật progress.
def evaluate_all_retrieval():
    """Evaluate all retrieval tests."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        result = evaluate_retrieval(test)
        progress = (index + 1) / total_tests
        yield test, result, progress


# Generator này yield kết quả answer evaluation và thêm JSON dùng cho lưu log.
def evaluate_all_answers():
    """Evaluate all answers to tests using batched async execution."""
    tests = load_tests()
    total_tests = len(tests)
    for index, test in enumerate(tests):
        eval = evaluate_answer(test)
        result = eval[0]
        # Lưu thêm bản ghi chi tiết để có thể hậu kiểm từng câu sau khi chạy xong.
        eval_json = {
            "question": test.question,
            "category": test.category,
            "generated_answer": eval[1],
            "reference_answer": test.reference_answer,
            "result": {
                "accuracy" : result.accuracy,
                "completeness" : result.completeness,
                "relevance" : result.relevance}
            }
        progress = (index + 1) / total_tests
        yield test, result, progress, eval_json


# Hàm CLI helper để soi chi tiết một test case cụ thể.
def run_cli_evaluation(test_number: int):
    """Run evaluation for a specific test (async helper for CLI)."""
    # Lấy toàn bộ bộ test rồi chọn ra một dòng theo chỉ số người dùng cung cấp.
    tests = load_tests("tests.jsonl")

    if test_number < 0 or test_number >= len(tests):
        print(f"Error: test_row_number must be between 0 and {len(tests) - 1}")
        sys.exit(1)

    # Chọn test case cần kiểm tra chi tiết.
    test = tests[test_number]

    # In thông tin đầu vào để dễ so sánh với kết quả chấm.
    print(f"\n{'=' * 80}")
    print(f"Test #{test_number}")
    print(f"{'=' * 80}")
    print(f"Question: {test.question}")
    print(f"Keywords: {test.keywords}")
    print(f"Category: {test.category}")
    print(f"Reference Answer: {test.reference_answer}")

    # Chạy phần retrieval trước để xem context có được kéo đúng không.
    print(f"\n{'=' * 80}")
    print("Retrieval Evaluation")
    print(f"{'=' * 80}")

    retrieval_result = evaluate_retrieval(test)

    print(f"MRR: {retrieval_result.mrr:.4f}")
    print(f"nDCG: {retrieval_result.ndcg:.4f}")
    print(f"Keywords Found: {retrieval_result.keywords_found}/{retrieval_result.total_keywords}")
    print(f"Keyword Coverage: {retrieval_result.keyword_coverage:.1f}%")

    # Sau đó mới chấm chất lượng answer sinh ra từ context đó.
    print(f"\n{'=' * 80}")
    print("Answer Evaluation")
    print(f"{'=' * 80}")

    answer_result, generated_answer, retrieved_docs = evaluate_answer(test)

    print(f"\nGenerated Answer:\n{generated_answer}")
    print(f"\nFeedback:\n{answer_result.feedback}")
    print("\nScores:")
    print(f"  Accuracy: {answer_result.accuracy:.2f}/5")
    print(f"  Completeness: {answer_result.completeness:.2f}/5")
    print(f"  Relevance: {answer_result.relevance:.2f}/5")
    print(f"\n{'=' * 80}\n")


# Entry point cho chế độ dòng lệnh.
def main():
    """CLI to evaluate a specific test by row number."""
    if len(sys.argv) != 2:
        print("Usage: uv run eval.py <test_row_number>")
        sys.exit(1)

    try:
        test_number = int(sys.argv[1])
    except ValueError:
        print("Error: test_row_number must be an integer")
        sys.exit(1)

    run_cli_evaluation(test_number)


if __name__ == "__main__":
    main()
