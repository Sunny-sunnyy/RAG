# File: Agentic-RAG-InsureLLM/implementation/ingest.py
# File này xây pipeline ingest cho Agentic RAG:
# đọc markdown -> chia chunk cha/con bằng LLM -> lưu parent vào SQLite
# và lưu embedding của child chunks vào Chroma để phục vụ retrieval.

from openai import OpenAI
from dotenv import load_dotenv
from chromadb import PersistentClient
from tqdm import tqdm
from litellm import completion
from pydantic import BaseModel, Field
from pathlib import Path
from typing import List, Dict
import sqlite3
import json
from multiprocessing import Pool
from tenacity import retry, wait_exponential


load_dotenv(override=True)

MODEL = "openai/gpt-4.1-nano"
AVERAGE_PARENT_CHUNK_SIZE = 1000

# Parent DB
# SQLite này giữ parent chunks để khi search trúng child chunk,
# hệ thống có thể kéo lại ngữ cảnh rộng hơn của cùng parent.
PARENT_DB_PATH = str(Path(__file__).parent.parent / "parent_chunks.db")
# Vector DB
# Chroma này chỉ lưu child chunks vì child chunk phù hợp cho truy xuất chi tiết.
DB_NAME = str(Path(__file__).parent.parent / "preprocessed_hierarchic_db")

collection_name = "docs"
embedding_model = "text-embedding-3-large"
KNOWLEDGE_BASE_PATH = Path(__file__).parent.parent / "knowledge-base"

wait = wait_exponential(multiplier=1, min=10, max=240)
WORKERS = 3

openai = OpenAI()



# Hàm này tạo bảng SQLite nếu chưa tồn tại.
def init_parent_chunk_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS parent_chunks (
        parent_id INTEGER PRIMARY KEY,
        parent_headline TEXT NOT NULL,
        parent_chunk TEXT NOT NULL,
        metadata TEXT NOT NULL
    )
    """)

    conn.commit()
    return conn


# Hàm này ghi toàn bộ parent chunks vào SQLite.
def insert_parent_chunk(conn, chunks):
    cursor = conn.cursor()

    # Chỉ parent chunk mới được lưu ở đây; child chunk sẽ được lưu ở vector DB.
    data = [
        (
            int(chunk.metadata.get("parent_id")),
            chunk.parent_headline,            
            chunk.parent_chunk,
            json.dumps(chunk.metadata, ensure_ascii=False) 
        )
        for chunk in chunks
    ]

    cursor.executemany("""
        INSERT OR REPLACE INTO parent_chunks
        (parent_id, parent_headline, parent_chunk, metadata)
        VALUES (?, ?, ?, ?)
    """, data)

    conn.commit()
    
    print(f"{len(data)} parent chunks saved to DB.")



# Đọc toàn bộ markdown trong knowledge base thành danh sách document thô.
def fetch_documents():
    """A homemade version of the LangChain DirectoryLoader"""

    documents = []

    for folder in KNOWLEDGE_BASE_PATH.iterdir():
        doc_type = folder.name
        for file in folder.rglob("*.md"):
            with open(file, "r", encoding="utf-8") as f:
                relative_path = file.relative_to(KNOWLEDGE_BASE_PATH.parent)
                documents.append({"type": doc_type,"source": relative_path.as_posix(),"text": f.read()})

    print(f"Loaded {len(documents)} documents")
    return documents



# Result là kiểu dữ liệu chuẩn dùng trong các bước sau ingest.
class Result(BaseModel):
    parent_headline: str
    parent_chunk: str
    child_chunks: List[str]
    metadata: Dict


# Chunk là schema mà LLM phải trả về cho từng phần của document.
class Chunk(BaseModel):
    parent_headline: str = Field(
        description=
        "A short headline summarizing the main subject and context of the Parent Chunk. "
        "The headline must clearly include the primary entity or subject of the document "
    )
    parent_chunk: str = Field(
        description=(
            "A reasonably divided, self-contained section of the source text. "
            "It must preserve sufficient detail and context for accurate retrieval and answer generation. "
            "The parent chunk must be fully understandable in isolation. "
            "It must not contain ambiguous or unresolved pronouns or references. "
            "Whenever necessary, replace pronouns such as 'it', 'they', 'this', or 'that' with the explicit entity name from the source text, "
            "without changing the original meaning or inventing new information."
        )
    )
    child_chunks: List[str] = Field(
        description =
        "List of a smaller, more specific version of the parent chunk, broken down into smaller, sub-pieces."
        "Each of these pieces must represent a single main idea or detail to enable precise semantic searching."
        )

    def as_result(self,document):
        # Chuẩn hóa metadata để mọi nơi về sau dùng cùng một format.
        metadata = {"source": document["source"], "type": document["type"]}
        return Result(parent_headline=self.parent_headline,parent_chunk=self.parent_chunk, child_chunks=self.child_chunks, metadata=metadata)


# Wrapper schema cho danh sách chunk của một document.
class Chunks(BaseModel):
    chunks: list[Chunk]


# Tạo prompt hướng dẫn LLM chia document thành parent chunks và child chunks.
def make_prompt(document):
    how_many = (len(document["text"]) // AVERAGE_PARENT_CHUNK_SIZE) + 1

    return f"""
You are performing hierarchical semantic chunking for a knowledge base. 
Your goal is to split a document into a structured hierarchy of Parent Chunks and Child Chunks.

This hierarchical structure is explicitly designed to support a retrieval flow
that moves from highly granular, specific CHILD chunks toward broader, more
context-rich PARENT chunks.

A chatbot will use these chunks to answer questions about the company.
The document belongs to the internal knowledge base of a company called Insurellm.
Document type: {document["type"]}
Document source: {document["source"]}

Divide the document into approximately {how_many} **Parent Chunks**, but you can have more or less as appropriate.
Ensure about 10-20% overlap between consecutive Parent Chunks to prevent information loss at boundaries.
The combination of all Parent Chunks must represent the *entire* document text.
For each Parent Chunk, generate a list of corresponding **Child Chunks**.

When generating chunks, assume that retrieval will typically:
1. Match relevant CHILD chunks first
2. Then expand upward to their associated PARENT chunks
   to recover broader context and meaning.

DEFINITIONS & REQUIREMENTS:

1. Parent Chunk (The Context Layer):
   - This is a logically divided, self-contained section of the text (e.g., a comprehensive paragraph or a thematic section).
   - It must be long enough to preserve full context so the LLM can answer questions accurately later.
   - PARENT chunks must be optimized for contextual grounding and synthesis (coarse-grained retrieval, general understanding, broader scope).
   - Each Parent Chunk must include a short headline that preserves the main subject of the document and is most likely to be surfaced in a query.


2. Child Chunks (The Index Layer):
   - For each Parent Chunk, break it down into a list of Child Chunks.
   - A Child Chunk MUST be no longer than 1–2 sentences.
   - Each Child Chunk must be fully understandable when retrieved alone. If not, rewrite or split the chunk.   
   - CHILD chunks must be optimized for precise, detail-level matching (fine-grained retrieval, exact facts, narrow concepts).   
   - Each child chunk must represent a single, specific idea, fact, rule, or detail to enable precise semantic searching  
   - Every important detail in the Parent Chunk should be represented in its Child Chunks.
   - Child Chunks MUST NOT contain unresolved pronouns like it, they, them, this, that, these, those, he, she, his, her, their.
     Always replace pronouns with explicit entity names from the Parent Chunk.
     For example: *Bad:* "It costs $10." (Who is it?)
                  *Good:* "The Silver Plan costs $10." (Replace pronouns with specific nouns).   

IMPORTANT RULES:
- The entire document MUST be fully covered by the parent chunks.
- Do NOT omit any important information.
- You MUST preserve every part of the document, including the beginning, middle, and end.
- If the last portion of the document is too short to form a full chunk on its own, still include it by attaching it to the final Parent Chunk or by creating a smaller final chunk.
- Parent Chunks MUST preserve the original wording and factual structure of the document as much as possible.
- Child chunks MUST be derived from their parent chunk only.
- Do NOT reference information outside the document.
- Do NOT invent or infer information.

Here is the document:

{document["text"]}

Respond with the structured chunks strictly adhering to the schema provided.
"""


# Bọc prompt thành format messages cho LiteLLM/OpenAI-compatible API.
def make_messages(document):
    return [
        {"role": "user", "content": make_prompt(document)},
    ]


# Gọi LLM để xử lý một document duy nhất.
@retry(wait=wait)
def process_document(document):
    messages = make_messages(document)
    response = completion(
        model=MODEL, 
        messages=messages, 
        response_format=Chunks, 
    )
    reply = response.choices[0].message.content
    doc_as_chunks = Chunks.model_validate_json(reply).chunks
    return [chunk.as_result(document) for chunk in doc_as_chunks]


# Chạy chunking song song cho nhiều document để tăng tốc ingest.
def create_chunks(documents):
    """
    Create chunks using a number of workers in parallel.
    If you get a rate limit error, set the WORKERS to 1.
    """
    chunks = []
    # Mỗi document xử lý độc lập nên có thể dùng multiprocessing.
    with Pool(processes=WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(process_document, documents), total=len(documents)):
            chunks.extend(result)
    return chunks


# Gán parent_id cho từng parent chunk để nối child chunk với parent chunk tương ứng.
def assign_parent_ids(chunks):
    for i in range(len(chunks)):
        chunks[i].metadata["parent_id"] = str(i)

    return chunks    


# Chia dữ liệu lớn thành các lô nhỏ để batch embedding.
def batch(iterable, size): 
    for i in range(0, len(iterable), size): 
        yield iterable[i:i + size]


# Tạo embedding cho tất cả child chunks rồi lưu vào Chroma.
def create_embeddings(chunks, batch_size=50):
    chroma = PersistentClient(path=DB_NAME)

    if collection_name in [c.name for c in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    # Flatten child chunks vì vector DB chỉ index các mảnh nhỏ phục vụ search chính xác.
    texts = [child for chunk in chunks for child in chunk.child_chunks]
    metas = [chunk.metadata for chunk in chunks for _ in chunk.child_chunks]

    collection = chroma.get_or_create_collection(collection_name)

    current_id = 0

    # Gửi embeddings theo batch để tránh request quá lớn và dễ kiểm soát tiến trình hơn.
    for text_batch, meta_batch in zip(
        batch(texts, batch_size),
        batch(metas, batch_size)
    ):
        emb = openai.embeddings.create(model=embedding_model, input=text_batch).data

        vectors = [e.embedding for e in emb]
        ids = [str(i) for i in range(current_id, current_id + len(text_batch))]
        collection.add(ids=ids, embeddings=vectors, documents=text_batch, metadatas=meta_batch)

        current_id += len(text_batch)

    print(f"Vectorstore created with {collection.count()} documents")



if __name__ == "__main__":
    # Luồng chạy đầy đủ:
    # 1) đọc document
    # 2) chunk phân cấp
    # 3) gán parent_id
    # 4) lưu parent vào SQLite
    # 5) lưu child embeddings vào Chroma
    documents = fetch_documents()
    chunks = create_chunks(documents)
    chunks = assign_parent_ids(chunks)

    # Save the parents to DB
    conn = init_parent_chunk_db(PARENT_DB_PATH)
    insert_parent_chunk(conn, chunks)
    conn.close()

    create_embeddings(chunks, batch_size=20)
    print("Ingestion complete")


