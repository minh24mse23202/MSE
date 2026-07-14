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
python -m pytest
```


### PyTorch DLL repair

If MiniLM re-indexing fails on Windows with `c10.dll` or `[WinError 1114]`, reinstall the stable CPU PyTorch wheel, then reinstall the project extras and restart the API:

```powershell
python -m pip install --force-reinstall "torch>=2.2,<2.6" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,api,app,ml]"
python -m uvicorn api.main:app --reload
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
python -m uvicorn api.main:app --reload
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

Remote deployments are registered from the React **AI Models** screen or the `/model-farm/deployments` API. Deployment records store only environment-variable names with the `ARAGBIZ_MODEL_*` prefix; secrets stay in `.env` or the host environment. Paid deployments should be tested before enabling and should have pricing/budget values configured.

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
