# Adaptive RAG System for Business Workflow Question Answering

This repository contains a phased capstone implementation for an Adaptive RAG system that answers business workflow questions. It starts with an offline, testable baseline and is structured to grow into model specialization, hybrid retrieval, evaluation, and a chatbot demo.

## Project Layout

- `src/aragbiz/`: core Adaptive RAG package.
- `api/`: FastAPI service exposing the answer and feedback endpoints.
- `app/`: Streamlit chatbot prototype.
- `data/`: sample, raw, processed, and artifact data areas.
- `notebooks/`: Colab-oriented experiment notebooks.
- `scripts/`: reproducible CLI utilities.
- `tests/`: unit and integration tests.
- `docs/`: methodology, phase notes, evaluation outputs, and report materials.

## Runtime Requirement

Use Python 3.11 for local development and demos. The project metadata intentionally rejects Python 3.9/3.10 to keep the FastAPI, Transformers, Torch, and pgVector stack stable.

Check your launcher before creating the environment:

```powershell
py -3.11 --version
```

## Quick Start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev,api,app,ml,models]"
Copy-Item .env.example .env
python -m pytest
```

Before starting the API, replace the placeholder JWT secret, administrator password, and model credential encryption key in `.env`. The React login and signup screens use the FastAPI JWT endpoints; the administrator account configured by `ARAGBIZ_BOOTSTRAP_ADMIN_EMAIL` is required to manage AI Model connections.

Conversation-aware RAG defaults to three completed exchanges and 4,000 characters per saved configuration. The server-enforced ceilings are configured in `.env` with `ARAGBIZ_CONVERSATION_MAX_EXCHANGES` and `ARAGBIZ_CONVERSATION_MAX_CHARACTERS`; the supplied defaults are `6` and `10000`.


### PyTorch DLL repair

If MiniLM re-indexing fails on Windows with `c10.dll` or `[WinError 1114]`, reinstall the stable CPU PyTorch wheel, then reinstall the project extras and restart the API:

```powershell
python -m pip install --force-reinstall "torch>=2.2,<2.6" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,api,app,ml]"
python -m uvicorn api.main:app --reload --env-file .env
```

Start local PostgreSQL + pgVector for the Knowledge & data processing layer:

```powershell
docker compose up -d postgres
```

Apply database migrations when using PostgreSQL:

```powershell
alembic upgrade head
```

Run the API:

```powershell
python -m uvicorn api.main:app --reload --env-file .env
```

Run the background worker for queued ingestion, indexing, and evaluation jobs:

```powershell
python -m aragbiz.worker
```

Run the chatbot UI:

```powershell
streamlit run app/streamlit_app.py
```

Run the React RAG Studio:

```powershell
cd frontend
npm install
npm run dev
```

The React app calls the FastAPI backend at `http://127.0.0.1:8000`.

## RAG Observability Traces

Each synchronous or streaming Main-screen answer now creates a durable full-fidelity trace. PostgreSQL stores searchable trace metadata while strict UTF-8 JSON artifacts are gzip-compressed under `data/traces` by default. Apply `alembic upgrade head` before starting the updated API.

Trace retention and payload limits are configured through:

- `ARAGBIZ_TRACE_STORE=data/traces`
- `ARAGBIZ_TRACE_RETENTION_DAYS=30`
- `ARAGBIZ_TRACE_MAX_BYTES=10485760`

The Trace popup loads the artifact by `trace_id` and shows hierarchical spans, timings, a latency waterfall, full inputs/outputs, model usage links, and retrieval score diagnostics. JSON can also be retrieved through `GET /traces/{trace_id}`, `GET /chat/messages/{message_id}/trace`, or downloaded through `GET /traces/{trace_id}/download`.

Artifacts include complete prompts, responses, conversation context, and retrieval candidate text for debugging. Credential fields, authorization headers, encrypted secrets, and raw numeric embedding vectors are always redacted.

Run the full local stack with PostgreSQL, migrations, API, and worker:

```powershell
docker compose up --build
```

## AI Models

AI Models is the provider-neutral runtime layer for generation, embeddings, reranking, judging, planning, and classifiers. Built-in local deployments are seeded automatically:

- `model-local-extractive`
- `model-local-flan-t5-small`
- `model-local-hash-384`
- `model-local-minilm-384`
- `model-local-lexical-reranker`
- `model-local-distilbert`
- `model-local-t5-classifier`

The React **AI Models** wizard separates reusable provider connections from model deployments:

- **Experimentation:** OpenRouter.
- **Production direct:** OpenAI and Gemini, using their own credentials and billing.
- **Local:** Ollama, vLLM, and the in-process built-ins above.

Provider-native model IDs are stored in deployments and translated to LiteLLM IDs only at runtime. Credentials may reference `ARAGBIZ_MODEL_*` environment variables or be entered in the wizard. Entered values are stored in a versioned AES-GCM envelope and are never returned by the API. Configure `ARAGBIZ_MODEL_SECRET_KEY` before storing credentials; keep this key stable so existing credentials remain decryptable.

Run `alembic upgrade head` after pulling this version. Existing deployment IDs are preserved and assigned to migrated connection records. Remote connections and deployments must pass health tests before they can be enabled. Paid models require LiteLLM-known pricing or a positive administrator pricing override, and all calls remain subject to deployment/global budgets.

Knowledge bases select an embedding deployment and keep query embeddings tied to the active index version. Chat configurations select generator, fallback, planner, and reranker deployments. Evaluation runs can select a registered judge deployment and persist that choice with the run metadata.

Knowledge-base ingestion APIs are available under:

- `GET /knowledge-bases`
- `POST /knowledge-bases`
- `PUT /knowledge-bases/{id}`
- `DELETE /knowledge-bases/{id}`
- `POST /knowledge-bases/{id}/sources/upload`
- `POST /knowledge-bases/{id}/sources/website`
- `POST /knowledge-bases/{id}/reindex`
- `GET /knowledge-bases/{id}/documents`
- `POST /knowledge-bases/{id}/documents`
- `PUT /knowledge-bases/{id}/documents/{document_id}`
- `DELETE /knowledge-bases/{id}/documents/{document_id}`

Download and convert WixQA:

```powershell
$env:PYTHONPATH='src'
python scripts/download_wixqa.py --subset wixqa_expertwritten
```

Run the evaluation:

```powershell
python scripts/evaluate_sample.py
```

For RAGXplain LLM-judge evaluation and the embedded insights viewer, install the sibling project into the same environment:

```powershell
python -m pip install -e "..\ragxplain"
```

Set `ARAGBIZ_RAGXPLAIN_JUDGE=your_judge_module:judge` for a real judge. The configured default mock judge is intended only for integration testing.

Train the Phase 2 lightweight query complexity classifier:

```powershell
$env:PYTHONPATH='src'
python scripts/generate_synthetic_qac.py --limit 90
python scripts/train_query_classifier.py --extra-dataset data/processed/wixqa_synthetic_bootstrap_qac.jsonl
python scripts/evaluate_sample.py --limit 10
```

For a pure WixQA-only classifier run:

```powershell
$env:PYTHONPATH='src'
python scripts/train_query_classifier.py
```

Train the Hugging Face DistilBERT classifier in Colab or a GPU environment:

```powershell
python -m pip install -e ".[dev,api,app,ml]"
$env:PYTHONPATH='src'
python scripts/train_hf_query_classifier.py --extra-dataset data/processed/wixqa_synthetic_bootstrap_qac.jsonl
```

Train and compare a T5-small classifier:

```powershell
python -m pip install -e ".[dev,api,app,ml]"
$env:PYTHONPATH='src'
python scripts/train_t5_query_classifier.py --extra-dataset data/processed/wixqa_synthetic_bootstrap_qac.jsonl
python scripts/compare_query_classifiers.py --limit 50
```

When `data/artifacts/query_classifier_distilbert/` exists, the app uses it first. If it is absent, the app falls back to `data/artifacts/query_classifier_nb.json`, then to the heuristic classifier.

## Current Baseline

The first implementation is intentionally lightweight and offline:

- query complexity classification uses deterministic heuristics;
- when `data/artifacts/query_classifier_distilbert/` exists, routing uses the Hugging Face classifier artifact;
- otherwise, routing can fall back to `data/artifacts/query_classifier_nb.json` or the deterministic heuristic;
- retrieval combines BM25 and hashed dense similarity;
- generation uses retrieved context snippets;
- evaluation reports routing accuracy, retrieval relevance, faithfulness proxy, answer overlap, and latency.
- knowledge ingestion supports local files and public websites, then performs metadata extraction, deduplication, chunking, embedding, and PostgreSQL/pgVector persistence.
- each knowledge base contains many documents; documents can be added, modified, deleted, and reindexed independently from the UI/API.

By default, the app uses processed WixQA files when they exist and falls back to the small bundled sample dataset when they do not.
