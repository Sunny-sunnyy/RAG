# File: Agentic-RAG-InsureLLM/evaluation/test.py
# Nhiệm vụ của file này là nạp bộ câu hỏi đánh giá chuẩn từ file JSONL
# và ép kiểu chúng thành object để các bước evaluation dùng lại nhất quán.

import json
from pathlib import Path
from pydantic import BaseModel, Field

TEST_FILE = str(Path(__file__).parent / "tests.jsonl")


# Class này mô tả cấu trúc chuẩn của một test case trong bộ dữ liệu đánh giá.
class TestQuestion(BaseModel):
    """A test question with expected keywords and reference answer."""

    question: str = Field(description="The question to ask the RAG system")
    keywords: list[str] = Field(description="Keywords that must appear in retrieved context")
    reference_answer: str = Field(description="The reference answer for this question")
    category: str = Field(description="Question category (e.g., direct_fact, spanning, temporal)")


# Hàm này đọc toàn bộ test cases từ file JSONL thành danh sách object.
def load_tests() -> list[TestQuestion]:
    """Load test questions from JSONL file."""
    tests = []
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            # Mỗi dòng là một JSON object độc lập, tương ứng với một câu hỏi test.
            data = json.loads(line.strip())
            tests.append(TestQuestion(**data))
    return tests
