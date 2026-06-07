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

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,api,app]"
python -m pytest
```

Run the API:

```powershell
uvicorn api.main:app --reload
```

Run the chatbot UI:

```powershell
streamlit run app/streamlit_app.py
```

Download and convert WixQA:

```powershell
$env:PYTHONPATH='src'
python scripts/download_wixqa.py --subset wixqa_expertwritten
```

Run the evaluation:

```powershell
python scripts/evaluate_sample.py
```

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
python -m pip install -e ".[ml]"
$env:PYTHONPATH='src'
python scripts/train_hf_query_classifier.py --extra-dataset data/processed/wixqa_synthetic_bootstrap_qac.jsonl
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

By default, the app uses processed WixQA files when they exist and falls back to the small bundled sample dataset when they do not.
