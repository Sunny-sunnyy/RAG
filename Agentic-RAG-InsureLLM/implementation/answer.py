# File: Agentic-RAG-InsureLLM/implementation/answer.py
# Đây là runtime chính của Agentic RAG:
# router chọn chiến lược retrieval, truy xuất child chunks,
# kéo lại parent context, rerank, generate answer, rồi tự đánh giá để retry nếu cần.

from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Dict
import sqlite3
import json
from tenacity import retry, wait_exponential


load_dotenv(override=True)
wait = wait_exponential(multiplier=1, min=10, max=240)

MODEL = "openrouter/openai/gpt-oss-120b"

# Vector DB
# Chứa embedding của child chunks để truy xuất ngữ nghĩa chi tiết.
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_hierarchic_db")
# PARENT DB connection
# SQLite chứa parent chunks để phục hồi ngữ cảnh lớn hơn sau bước search.
PARENT_DB_PATH = str(Path(__file__).parent.parent / "parent_chunks.db")
# Knowledge base path
# Dùng để quét số lượng tài liệu theo từng thư mục phục vụ adaptive retrieval.
KNOWLEDGE_BASE_PATH = str(Path(__file__).parent.parent / "knowledge-base")

collection_name = "docs"
embedding_model = "text-embedding-3-large"

openai = OpenAI()



SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
Your answer will be evaluated for accuracy, relevance and completeness, so make sure it only answers the question and fully answers it.
If you don't know the answer, say so.
For context, here are specific extracts from the Knowledge Base that might be directly relevant to the user's question:
{context}

With this context, please answer the user's question with complement and supporting details. 
Be accurate, relevant and complete. Avoid long and irrelevant answers.

IMPORTANT INSTRUCTIONS:
- Fully answer the question, not just the minimum core fact.
- Include important supporting details from the context when they are directly related to the question.
- If the question is holistic, aggregated, or asks about totals, counts, scope, or overall status, include relevant contextual details such as distribution, locations, categories, timeframe, or operating scope when available in the context.
- If a direct answer is a number, date, person, or item, also include the most relevant qualifying details that make the answer more complete.
- Do not add unrelated details.
- Do not invent information.
- Prefer a complete and specific answer over a short answer.
"""



# Result là một context item đã ghép cả child hit và parent context tương ứng.
class Result(BaseModel):
    parent_headline: str
    parent_chunk: str
    child_chunks: List[str]
    metadata: Dict



# Đếm số tài liệu trong từng thư mục để hỗ trợ chiến lược directory-wide scan.
def build_directory_doc_counts(knowledge_base_path):
    """
    Scan the knowledge base directory and return a dictionary
    mapping each subdirectory to its number of documents.
    """

    base_path = Path(knowledge_base_path)

    directory_doc_counts = {}

    for subdir in base_path.iterdir():
        if subdir.is_dir():
            doc_count = sum(1 for file in subdir.iterdir() if file.is_file())
            directory_doc_counts[subdir.name] = doc_count

    return directory_doc_counts


# Lấy hàng loạt parent chunks trong một query SQL để đỡ tốn I/O.
def fetch_parent_chunks_batch(parent_ids: List[str]) -> Dict[str, Dict]:
    placeholders = ','.join('?' for _ in parent_ids)

    with sqlite3.connect(PARENT_DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT parent_id, parent_headline, parent_chunk, metadata
            FROM parent_chunks
            WHERE parent_id IN ({placeholders})
            """,
            parent_ids
        )

        rows = cursor.fetchall()    
    
    # Chuyển kết quả SQL về dict để tra parent chunk theo parent_id thật nhanh.
    results_map = {str(row[0]): {"parent_headline": row[1], "parent_chunk": row[2], "metadata": json.loads(row[3])} for row in rows}
         
    return results_map


# Mở collection Chroma hiện tại; nếu chưa có thì tạo mới.
def get_collection():
    chroma = PersistentClient(path=DB_NAME)
    return chroma.get_or_create_collection(collection_name)


# Gộp nhiều danh sách chunks và loại trùng dựa trên child chunk match.
def merge_chunks(chunk_lists):
    merged = []
    seen = set()

    for chunks in chunk_lists:
        for chunk in chunks:
            key = chunk.child_chunks[0]

            if key not in seen:
                merged.append(chunk)
                seen.add(key)

    return merged



# Search trực tiếp trên vector DB rồi ghép lại parent context tương ứng.
def fetch_context_unranked(question, RETRIEVAL_K):
    query = openai.embeddings.create(model=embedding_model, input=[question]).data[0].embedding
    collection = get_collection()
    results = collection.query(query_embeddings=[query], n_results=RETRIEVAL_K)

    documents, metadatas = results["documents"][0], results["metadatas"][0]
    # Một parent có thể sinh nhiều child chunks, nên chỉ query mỗi parent một lần.
    parent_ids = list({meta["parent_id"] for meta in metadatas})
    results_map = fetch_parent_chunks_batch(parent_ids)

    chunks = []
    for doc, meta in zip(documents, metadatas):
        parent_id = meta.get("parent_id")
        parent_info = results_map.get(parent_id)
        # Kết quả cuối giữ cả child chunk trúng search lẫn parent chunk để LLM có đủ ngữ cảnh.
        chunks.append(Result(parent_headline = parent_info["parent_headline"], parent_chunk=parent_info["parent_chunk"], child_chunks=[doc], metadata=meta))

    return chunks



# Làm sạch thứ tự chunk do reranker trả về để tránh id lặp, sai hoặc thiếu.
def sanitize_rank_order(raw_order, n):
    seen = set()
    clean_order = []

    for i in raw_order:
        if isinstance(i, int) and 1 <= i <= n and i not in seen:
            clean_order.append(i)
            seen.add(i)

    for i in range(1, n + 1):
        if i not in seen:
            clean_order.append(i)

    return clean_order


# Schema mà mô hình reranker phải trả về.
class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


# Dùng LLM để sắp lại candidate chunks theo độ liên quan với câu hỏi gốc.
@retry(wait=wait)
def rerank(question, chunks):
    n = len(chunks)
    system_prompt = f"""
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.

STRICT RULES:
- Use only chunk ids from 1 to {n}
- Do not invent new ids
- Do not repeat ids
- Return every id exactly once
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.parent_headline + "\n" + chunk.child_chunks[0] if chunk.child_chunks else chunk.parent_chunk}\n\n"        
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = completion(model=MODEL, messages=messages, response_format=RankOrder)
        reply = response.choices[0].message.content
        raw_order = RankOrder.model_validate_json(reply).order
    except Exception:
        # Nếu reranker lỗi, rơi về thứ tự ban đầu để không làm hỏng toàn bộ request.
        raw_order = list(range(1, len(chunks) + 1))

    order = sanitize_rank_order(raw_order, len(chunks))
    # print(order)

    return [chunks[i - 1] for i in order]      


# Viết lại query để tạo một truy vấn ngắn, cụ thể và dễ match hơn trong knowledge base.
@retry(wait=wait)
def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}])
    return response.choices[0].message.content


# Chiến lược này phù hợp cho câu hỏi direct fact hoặc câu hỏi cục bộ trong một tài liệu.
def fetch_context_single_doc(original_question, RETRIEVAL_K=20, FINAL_K=10):
    chunk_lists = []
    rewritten_question = rewrite_query(original_question)
    # Search cả câu gốc và câu rewrite để tăng recall nhưng vẫn giữ ý gốc.
    chunks1 = fetch_context_unranked(original_question, RETRIEVAL_K)
    chunks2 = fetch_context_unranked(rewritten_question, RETRIEVAL_K)
    chunk_lists.append(chunks1)
    chunk_lists.append(chunks2)
    chunks = merge_chunks(chunk_lists)
    reranked = rerank(original_question, chunks)
    return reranked[:FINAL_K]



# Schema cho danh sách sub-queries khi phải phân rã một câu hỏi lớn.
class SubQueries(BaseModel):
    queries: List[str] = Field(
        description=(
            "A list of refined sub-questions derived from the user's question to optimize retrieval quality. "
            "Each item must be a complete natural-language question sentence."
        ),
        min_items=1,
        max_items=5
    )


# Tách câu hỏi thành nhiều truy vấn con để phục vụ spanning / multi-hop retrieval.
@retry(wait=wait)
def decompose_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.

You are a query decomposition assistant and your task is to transform the user's question into a small set of high-quality search queries
that will retrieve the most relevant information from a knowledge base.

This is the history of your conversation so far with the user:
{history}

And this is the user's current question:
{question}

To optimize retrieval quality, decompose the main question into independent sub-queries. 
A sub-query should be a VERY short specific question most likely to surface content. 
Focus on the question details.
If the question involves fragmented information spanning multiple documents (Spanning) or 
requires understanding the overall context of a document (Holistic), adaptively (min. 2 increase 
the number of sub-queries according to the breadth of the topic to capture important information parts.

CRITICAL RULES: 
- Each sub-query must be concise, specific, and semantically complete. 
- Don't mention the company name unless it's a general question about the company.
- Respond ONLY with the knowledgebase query, nothing else.
"""
    response = completion(model=MODEL, messages=[{"role": "system", "content": message}], response_format=SubQueries)
    reply = response.choices[0].message.content
    sub_queries = SubQueries.model_validate_json(reply).queries
    return sub_queries


# Chiến lược này phù hợp cho câu hỏi phải nối nhiều facts ở nhiều tài liệu khác nhau.
def fetch_context_multi_hop(original_question, RETRIEVAL_K=20, FINAL_K=10):
    chunk_lists = []
    rewritten_questions = decompose_query(original_question)
    chunks1 = fetch_context_unranked(original_question, RETRIEVAL_K)
    chunk_lists.append(chunks1)

    # Mỗi sub-query kéo một phần thông tin khác nhau của bài toán.
    for quest in rewritten_questions:
        chunks2 = fetch_context_unranked(quest, RETRIEVAL_K)
        chunk_lists.append(chunks2)

    chunks = merge_chunks(chunk_lists)
    reranked = rerank(original_question, chunks)
    
    return reranked[:FINAL_K]


# Chiến lược này quét rộng theo thư mục khi câu hỏi có tính tổng hợp hoặc thống kê.
def fetch_context_directory_scan(original_question, target_directory, RETRIEVAL_K=20):
    DIRECTORY_DOC_COUNTS = build_directory_doc_counts(KNOWLEDGE_BASE_PATH)    
    DIRECTORY_DOC_COUNTS["unknown"] = RETRIEVAL_K
    # Mở rộng K theo kích thước thư mục để tăng cơ hội gom đủ bức tranh tổng thể.
    adaptive_retrieval_k = DIRECTORY_DOC_COUNTS[target_directory] + 5
    chunk_lists = []
    rewritten_question = rewrite_query(original_question)
    chunks1 = fetch_context_unranked(original_question, adaptive_retrieval_k)
    chunks2 = fetch_context_unranked(rewritten_question, adaptive_retrieval_k)
    chunk_lists.append(chunks1)
    chunk_lists.append(chunks2)
    chunks = merge_chunks(chunk_lists)    
    reranked = rerank(original_question, chunks)
    
    return reranked[:adaptive_retrieval_k]


# Bộ tool schema mà router agent được phép dùng để chọn strategy.
tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_context_single_doc",
            "description": (
                "Use this tool for straightforward, localized questions targeting a single piece of information. "
                "This function uses query rewriting to optimize the vector search for highly specific facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "original_question": {
                        "type": "string",
                        "description": "The exact question provided by the user."
                    }
                },
                "required": ["original_question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_context_multi_hop",
            "description": (
                "Use this tool for complex, multi-hop questions that require spanning multiple documents. "
                "This function decomposes the main question into multiple atomic sub-queries to gather fragmented information from different regions of the knowledge base."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "original_question": {
                        "type": "string",
                        "description": "The exact question provided by the user."
                    }
                },
                "required": ["original_question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_context_directory_scan",
            "description": (
                "Use this tool ONLY when the question requires aggregating, comparing, or scanning across all documents within a specific category. "
                "Available categories: 'company', 'products', 'employees', 'contracts'. "
                "The system will adapt the retrieval breadth based on the target category's size."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "original_question": {
                        "type": "string",
                        "description": "The exact question provided by the user."
                    },
                    "target_directory": {
                        "type": "string",
                        "enum": ["company", "products", "employees", "contracts", "unknown"],
                        "description": "The specific directory the question targets. If it spans all directories, use 'unknown'."
                    }
                },
                "required": ["original_question", "target_directory"]
            }
        }
    }
]


# Nhận tool call từ router rồi gọi đúng hàm retrieval tương ứng.
def handle_tool_call(message):
    """
    Actually call the tools associated with this message
    """
    mapping = {
        "fetch_context_single_doc": fetch_context_single_doc,
        "fetch_context_multi_hop": fetch_context_multi_hop,
        "fetch_context_directory_scan": fetch_context_directory_scan,
    }

    results = []
    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        tool = mapping.get(tool_name)
        # Tool nào được router chọn thì sẽ được thực thi ở đây với arguments đã parse từ JSON.
        result = tool(**arguments) if tool else ""
        results.append({"role": "tool","content": result,"tool_call_id": tool_call.id})
    return results


# Đây là router agent: nó không trả lời người dùng mà chỉ chọn retrieval strategy phù hợp nhất.
@retry(wait=wait)
def fetch_context_router(question: str):
    ROUTER_SYSTEM_PROMPT = """
You are an intelligent retrieval router for Insurellm’s internal RAG system.

Your job is to decide how to retrieve context before answering the user’s question.
You must first determine what kind of question the user is asking, then call the most appropriate retrieval tool.

The knowledge base contains documents under these directories:
- company: 4 documents
- contracts: 32 documents
- employees: 32 documents
- products: 8 documents

You MUST evaluate the user's question and strictly choose ONE of the following retrieval strategies:

1. SINGLE FACT LOOKUP: If the question is direct, highly specific, and likely resides in a single document (e.g., "What is the deductible for the Premium Plan?", "Who is the CEO?").
2. MULTI-HOP / SPANNING SYNTHESIS: If the question requires connecting fragmented information across different documents, or requires understanding the broader context to deduce an answer (e.g., "What is the salary of the CTO who joined in 2017?", "Compare the coverage of Plan A and Plan B").
3. HOLISTIC/DIRECTORY-WIDE AGGREGATION: If the question requires scanning, filtering, or aggregating data across an entire category of documents (e.g., "What is the longest contract duration among all contracts?", "List all employees in the marketing department"). 

Do not answer the question. 
Your only job is to trigger the correct function call with the necessary parameters.
"""
    router_messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": question}]    

    # Bắt buộc mô hình phải chọn tool, không được trả lời free-form.
    router_response = completion(
        model=MODEL, 
        messages=router_messages, 
        tools=tools, 
        tool_choice="required" 
    )
    if router_response.choices[0].finish_reason=="tool_calls":
        router_message = router_response.choices[0].message
        tool_results = handle_tool_call(router_message)   
        # Hiện tại router chỉ chọn đúng một strategy, nên lấy phần tử đầu tiên.
        chunks = tool_results[0]["content"]
        
    return chunks


# Ghép system prompt với context, history và câu hỏi gốc để mô hình sinh answer.
def make_rag_messages(question, history, chunks):
    # Dù search có thể dùng query đã rewrite, lúc generate vẫn giữ câu hỏi gốc để không lệch intent.
    context = "\n\n".join(f"Extract from {chunk.metadata['source']}:\n{chunk.parent_headline}\n{chunk.parent_chunk}\n{chunk.child_chunks[0]}" for chunk in chunks)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": question}]    

 
# Schema quyết định có retry retrieval thêm một vòng hay không.
class RetryEvaluation(BaseModel):
    allow_retry: bool = Field(
        description="True if the answer states information is missing, incomplete, or not found. False if the answer fully and confidently addresses the question."
    )
    refined_query: str = Field(
        description="If allow_retry is True, provide a better, more specific search query that will help the RAG system find the missing information in the vector database. If False, output an empty string."
    )


# Dùng LLM judge để xem câu trả lời hiện tại đã đủ tốt để trả ra chưa.
def evaluate_answer_quality(question: str, answer: str) -> RetryEvaluation:
    EVALUATOR_SYSTEM_PROMPT = """
    You are an intelligent answer evaluator and query optimization agent for a RAG system.

    You will receive:
    - the user's original question
    - the generated answer based on retrieved context

    Your task is to decide whether the answer is good enough to return to the user,
    or whether the system should retry retrieval.

    Return allow_retry=True IF:
    - The answer is incomplete, vague for a specific question, or fails to answer an important part of the question
    - The answer explicitly says "I don't know", "I couldn't find", "I do not have information", "There is no" or "The context does not state".
    - The answer only partially addresses the question due to missing information.

    If allow_retry=True:
    1. Write a 'refined_query' that the system should use to search the database again. This query should specifically target the missing information.

    Return allow_retry=False IF:
    - The answer is sufficiently complete and supported by the retrieved context
    - The answer confidently and fully addresses the question.    
    """

    user_prompt = f"Question: {question}\n\nAnswer: {answer}"
        
    messages = [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # Judge chỉ trả về quyết định retry và refined query, không tự sinh một answer mới.
    response = completion(
            model=MODEL, 
            messages=messages, 
            response_format=RetryEvaluation
        )
    reply = response.choices[0].message.content
    return RetryEvaluation.model_validate_json(reply)


# Hàm runtime chính: retrieve -> generate -> judge -> retry tối đa một lần nếu cần.
@retry(wait=wait)
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list]:
    """
    Answer a question using an Agentic RAG approach.
    If the evaluator determines the answer is insufficient, it uses the generated critique 
    and refined query to retry retrieval once.
    """

    max_attempts = 2
    attempt = 0
    search_query = question # search with user's original question first

    while attempt < max_attempts:
        # Nếu judge chê answer đầu tiên chưa đủ, vòng sau sẽ search bằng refined query.
        chunks = fetch_context_router(search_query) 
        
        # Khi generate answer vẫn dùng original question để giữ đúng ý định ban đầu của user.
        messages = make_rag_messages(question, history, chunks)
        response = completion(model=MODEL, messages=messages)
        reply = response.choices[0].message.content

        if attempt == 0:
            eval_result = evaluate_answer_quality(question, reply)
            
            if eval_result.allow_retry:
                # print(f"[Agentic Retry] Answer insufficient.")
                # print(f"Retrying with new query: {eval_result.refined_query}")

                # Judge đề xuất một refined query mới rồi cho hệ thống thử lại thêm đúng một vòng.
                search_query = eval_result.refined_query
                attempt += 1
                continue

        return reply, chunks        
