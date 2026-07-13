# NEXUS — Multi-Agent Autonomous Software Engineering Platform

> *Automatically resolves GitHub issues using a self-improving multi-agent pipeline.*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.28-purple.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is NEXUS?

NEXUS is an autonomous software engineering system that takes a GitHub issue URL and produces a code patch — with no human in the loop.

It is inspired by systems like [Devin](https://cognition.ai), [SWE-agent](https://swe-agent.com), and [OpenHands](https://github.com/All-Hands-AI/OpenHands), and is evaluated on the [SWE-bench](https://www.swebench.com) benchmark.

**Input:** `https://github.com/psf/requests/issues/7443`
**Output:** A unified diff patch that fixes the bug

---

## Architecture

```
GitHub Issue URL
      │
      ▼
┌─────────────────────────────────────────────────┐
│                 NEXUS Pipeline                  │
│                                                 │
│  Clone Repo ──▶ AST Chunker ──▶ Hybrid RAG     │
│  (GitPython)   (tree-sitter)   (BM25+Vec+HyDE) │
│                                      │          │
│                                      ▼          │
│         LangGraph Agent Graph                   │
│                                                 │
│  Planner ──▶ Engineer ──▶ Reviewer             │
│                               │    │            │
│                          pass │    │ fail       │
│                               │    ▼            │
│                               │  Reflector      │
│                               │  (loop ≤ 2x)   │
│                               ▼                 │
└───────────────────────────────────────────────  ┘
                               │
                               ▼
                      Unified Diff Patch
```

### Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **API** | FastAPI + Uvicorn | Async REST API, webhook receiver |
| **Agents** | LangGraph 0.2 | Multi-agent state machine orchestration |
| **LLM** | GPT-4o + Groq GPT-OSS-120B | Patch generation with fallback |
| **RAG** | Weaviate Cloud | Hybrid BM25 + vector search |
| **Embeddings** | OpenAI text-embedding-3-small | Code chunk embeddings |
| **Chunking** | tree-sitter | AST-based Python code chunking |
| **DB** | PostgreSQL (Supabase) | Task persistence |
| **Query Expansion** | HyDE | Hypothetical Document Embeddings |

---

## Agents

### 1. Planner Agent
Decomposes the GitHub issue into 2-4 ordered subtasks with file hints.
Uses GPT-4o structured output (Pydantic schema enforcement).

### 2. Engineer Agent
Generates the actual code patch using retrieved context from Hybrid RAG.
GPT-4o primary, Groq GPT-OSS-120B fallback.

### 3. Reviewer Agent
Scores the patch on correctness, completeness, safety, and style (0.0–1.0).
Passes patches with score >= 0.7.

### 4. Reflector Agent
If the reviewer rejects the patch, reads the feedback and generates an improved version.
Loops back to the reviewer. Maximum 2 reflection rounds.

---

## RAG Pipeline

**Reciprocal Rank Fusion (RRF)** combines BM25 and vector rankings:

```
score(d) = 1/(k + rank_bm25) + 1/(k + rank_vector)    where k=60
```

**HyDE**: Instead of embedding the raw issue text, we ask GPT-4o-mini to
write a hypothetical fix, then embed that — much closer to actual code in
embedding space.

---

## Quick Start

### Prerequisites
- Python 3.11+, OpenAI API key, Groq API key, GitHub Token
- Supabase account (free), Weaviate Cloud account (free)
- Or run Postgres + Weaviate locally: `docker compose up -d`

### Installation

```powershell
git clone https://github.com/YOUR_USERNAME/nexus.git
cd nexus
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Configuration

Create `.env` from the template and fill in your keys:

```env
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
GITHUB_TOKEN=ghp_...
DATABASE_URL=postgresql://...
WEAVIATE_URL=https://....weaviate.cloud   # or http://localhost:8080 for local
WEAVIATE_API_KEY=...                      # empty for local Weaviate
SECRET_KEY=<python -c "import secrets; print(secrets.token_hex(32))">
```

See `.env.example` for the full list (CORS origins, token budget, model choices).
With `APP_ENV=production`, startup fails unless `SECRET_KEY` is set.

### Run

```powershell
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` — the dashboard is served by the app
(create an account at `/login.html`; the first account gets the admin role).
API docs: `http://127.0.0.1:8000/docs`.

### Submit a Task

All task endpoints require a JWT (engineer role):

```bash
TOKEN=$(curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "you", "password": "..."}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl -X POST http://127.0.0.1:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"github_issue_url": "https://github.com/psf/requests/issues/7443"}'
```

Every task tracks its token usage and estimated cost, and the agent loop
stops automatically when `MAX_TOKENS_PER_TASK` is exhausted.

---

## GitHub App Integration

NEXUS auto-triggers on any repository via GitHub webhooks.

1. Go to **GitHub → Settings → Developer settings → GitHub Apps → New**
2. Set webhook URL: `https://your-domain/api/v1/webhook/github`
3. Set a webhook secret and put it in `GITHUB_WEBHOOK_SECRET` — **required**;
   the endpoint returns 503 until it is configured, and every request is
   HMAC-verified
4. Subscribe to **Issues** events
5. NEXUS triggers when an issue is labeled `nexus` (or `auto-fix`,
   `nexus-fix`, `ai-fix`). Triggering on every opened issue is deliberately
   not supported — each run costs LLM money, so runs require an explicit
   label from a collaborator.

---

## SWE-bench Evaluation

```powershell
pip install datasets
$env:NEXUS_API_TOKEN = "<jwt from /api/v1/auth/login>"
python evals/swebench_eval.py --limit 10 --output evals/results.json
```

---

## Tests

```powershell
pytest tests/ -v
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/tasks` | Submit a GitHub issue |
| `GET` | `/api/v1/tasks/{id}` | Poll task status + results |
| `GET` | `/api/v1/tasks` | List recent tasks |
| `GET` | `/api/v1/metrics` | Platform-wide statistics |
| `GET` | `/api/v1/metrics/daily` | Daily task breakdown |
| `POST` | `/api/v1/webhook/github` | GitHub App webhook receiver |
| `GET` | `/api/v1/health` | Health check |

---

## Project Structure

```
nexus/
├── app/
│   ├── agents/          # LangGraph multi-agent system
│   │   ├── state.py     # Shared agent state (TypedDict)
│   │   ├── planner.py   # Planner Agent
│   │   ├── engineer.py  # Engineer Agent
│   │   ├── reviewer.py  # Reviewer Agent
│   │   ├── reflector.py # Reflector Agent (self-healing loop)
│   │   └── graph.py     # LangGraph orchestration
│   ├── rag/             # Retrieval-Augmented Generation
│   │   ├── chunker.py   # AST-based code chunking (tree-sitter)
│   │   ├── embedder.py  # OpenAI embeddings + Weaviate indexing
│   │   └── retriever.py # Hybrid BM25 + vector + RRF + HyDE
│   ├── api/
│   │   ├── tasks.py     # Task CRUD + pipeline trigger
│   │   ├── webhook.py   # GitHub App webhook handler
│   │   └── metrics.py   # Analytics API
│   ├── auth/            # JWT auth (register/login, roles, throttling)
│   ├── core/
│   │   ├── config.py    # Pydantic settings (env-driven)
│   │   └── llm.py       # Token accounting + cost estimation helpers
│   ├── db/
│   │   ├── models.py    # SQLAlchemy Task model
│   │   └── database.py  # Supabase connection
│   └── main.py
├── evals/
│   └── swebench_eval.py # SWE-bench evaluation script
├── frontend/
│   ├── index.html       # Real-time dashboard (served at /)
│   └── login.html       # Login / register page
├── tests/               # Pytest test suite
├── Dockerfile           # Single-container deploy (Render/Railway/Fly)
├── docker-compose.yml   # Local Postgres + Weaviate + Redis
└── requirements.txt
```

---

## Deployment Notes

- **Render/Railway**: build from the `Dockerfile` (or `pip install -r
  requirements.txt` + `uvicorn app.main:app --host 0.0.0.0 --port $PORT`).
  Set all env vars from `.env.example`; use `APP_ENV=production`.
- **Model fallback**: the Engineer falls back from `PRIMARY_LLM` (GPT-4o) to
  `FALLBACK_LLM` on Groq (`openai/gpt-oss-120b` — the previous LLaMA 3.1/3.3
  versatile models were decommissioned by Groq).
- **Single worker**: run one uvicorn worker. Background pipelines run
  in-process; in-flight tasks from a previous process are marked failed at
  startup ("orphaned by server restart").
- **Cost control**: per-task token budget (`MAX_TOKENS_PER_TASK`) hard-stops
  the reflection loop; webhook runs require an explicit trigger label.

---

## Key Design Decisions

**Why LangGraph?** Gives explicit control over the agent loop — you define exactly which node runs next. Critical for the Reviewer → Reflector → Reviewer cycle.

**Why Hybrid RAG?** BM25 excels at exact token matches (function names, error messages). Vector search excels at semantic similarity. RRF fusion consistently outperforms either alone on code retrieval.

**Why HyDE?** Embedding "what code would fix this?" is semantically closer to actual code than embedding the issue description.

**Why tree-sitter?** Preserves semantic boundaries — functions never get split mid-body. The LLM always sees complete, meaningful code units.

---

*Built by Mirdula M — M.Tech CSE, SREC (2027)*
