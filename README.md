# Insurellm RAG Workspace

A Retrieval-Augmented Generation (RAG) study and implementation repository, featuring a complete exploration curriculum (Days 1–5), structured evaluation, and an interactive assistant interface.

## Project Structure

- **`day1.ipynb` to `day5.ipynb`**: Step-by-step Jupyter notebooks covering RAG curriculum (naive retrieval, chunking strategies, query rewriting, reranking, and advanced evaluation).
- **[study_guide/]**: HTML study guides (`day1` to `day5`) formatted for learning reinforcement and interview preparation.
- **[implementation/]**: Basic RAG pipeline utilizing LangChain, HuggingFace embeddings (`all-MiniLM-L6-v2`), ChromaDB, and OpenAI.
- **[pro_implementation/]**: Advanced RAG pipeline implementing:
  - LLM-based structured chunking (extracting headline, summary, original text).
  - Dual-path retrieval (searching with both original and rewritten queries).
  - Custom LLM-based reranking.
- **[evaluation/]**: RAG evaluation suite assessing retrieval metrics (MRR, nDCG, keyword coverage) and answer quality (LLM-as-a-judge scoring accuracy, completeness, and relevance).
- **[app.py]**: Gradio chatbot interface to chat with the Insurellm Expert Assistant.
- **[evaluator2.py]**: Multi-threaded Gradio evaluation dashboard supporting concurrent testing, seed configuration, and progress visualization.

## Getting Started

This project uses `uv` for dependency management.

### Installation

Install dependencies and set up the virtual environment:
```bash
uv sync
```

Set up your `.env` file by copying the example file and filling in the required API keys:
```bash
cp .env.example .env
```

The environment variables configured in `.env.example` include:
- `OPENAI_API_KEY`: Your OpenAI API key (required for OpenAI LLMs).
- `OPENROUTER_API_KEY`: Your OpenRouter API key (if calling models via OpenRouter).
- `HF_TOKEN`: Your Hugging Face access token (if using gated models/datasets).

### Usage

1. **Ingest Data**:
   ```bash
   uv run pro_implementation/ingest.py
   ```

2. **Launch Chatbot**:
   ```bash
   uv run app.py
   ```

3. **Launch Evaluation Dashboard**:
   ```bash
   uv run evaluator2.py
   ```

4. **Run CLI Evaluation**:
   ```bash
   uv run evaluation/eval.py <test_row_number>
   ```