# RAG System + Evals

Production-style **document Q&A** in Python: ingest docs, retrieve with multiple strategies, answer with **source citations**, and run an **eval table** that compares retrieval quality head-to-head.

**Default LLM: free Groq** (`openai/gpt-oss-20b`). Hugging Face Inference is also supported.

## What you get

| Capability | Details |
|---|---|
| Ingest | `.txt`, `.md`, `.pdf` → chunked + embedded into Chroma |
| Retrieve | `dense` · `bm25` · `hybrid` (RRF) · `hybrid_rerank` |
| Answer | Free Groq / Hugging Face answers grounded in chunks with `[n]` citations |
| Eval | Hit@k, MRR, context recall/precision, answer faithfulness → comparison table |

## Quick start (free)

```bash
# 1. Create env + install
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -e .

# 2. Configure a FREE LLM key
copy .env.example .env   # or: cp .env.example .env

# Recommended: Groq free tier — https://console.groq.com/keys
#   LLM_PROVIDER=groq
#   GROQ_API_KEY=gsk_...

# Or Hugging Face free tier — https://huggingface.co/settings/tokens
#   LLM_PROVIDER=huggingface
#   HF_TOKEN=hf_...

# 3. Ingest sample docs
rag ingest data/sample_docs

# 4. Ask a question (hybrid retrieval + citations)
rag ask "What is the refund policy for annual plans?"

# 5. Run retrieval evals → prints comparison table
rag eval

# See which provider is active
rag providers
```

Serve an HTTP API:

```bash
rag serve
# POST http://127.0.0.1:8000/ask  {"question": "...", "strategy": "hybrid"}
```

## Free LLM providers

| Provider | Env | Default model | Notes |
|---|---|---|---|
| **groq** (default) | `GROQ_API_KEY` | `openai/gpt-oss-20b` | Fast free tier |
| **huggingface** | `HF_TOKEN` | `Qwen/Qwen2.5-7B-Instruct:fastest` | HF Inference Router |
| openai | `OPENAI_API_KEY` | `gpt-4o-mini` | Optional paid |
| mock | `RAG_MOCK_LLM=1` | stub | Offline / CI |

Switch with `LLM_PROVIDER=groq` or `LLM_PROVIDER=huggingface` in `.env`.

## Retrieval strategies compared

1. **dense** — cosine similarity over sentence-transformer embeddings  
2. **bm25** — lexical ranking (great for exact terms / IDs)  
3. **hybrid** — Reciprocal Rank Fusion of dense + BM25  
4. **hybrid_rerank** — hybrid candidates re-scored by a cross-encoder

The eval harness measures which strategy actually retrieves the right evidence on a labeled set in `data/eval/qa_pairs.json`.

## Project layout

```
src/rag/
  ingest.py          # load + chunk documents
  embeddings.py      # local sentence-transformers
  store.py           # Chroma persistence
  retrieval/         # dense / bm25 / hybrid / rerank
  llm.py             # Groq / HF / OpenAI provider wiring
  generate.py        # LLM answer + citation formatting
  pipeline.py        # end-to-end ask()
  eval/              # metrics + runner + table
  cli.py             # typer CLI
  api.py             # FastAPI
data/sample_docs/    # demo knowledge base
data/eval/           # labeled Q&A for evals
artifacts/           # index + eval reports
```

## Typical workflow

1. Drop your docs into `data/sample_docs` (or any folder).  
2. `rag ingest <path>`  
3. `rag ask "..."` and inspect citations.  
4. Extend `data/eval/qa_pairs.json` with expected source snippets.  
5. `rag eval --strategies dense,bm25,hybrid,hybrid_rerank` and pick a winner.
