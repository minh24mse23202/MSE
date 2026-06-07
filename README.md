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

Run the sample evaluation:

```powershell
python scripts/evaluate_sample.py
```

## Current Baseline

The first implementation is intentionally lightweight and offline:

- query complexity classification uses deterministic heuristics;
- retrieval combines BM25 and hashed dense similarity;
- generation uses retrieved context snippets;
- evaluation reports routing accuracy, retrieval relevance, faithfulness proxy, answer overlap, and latency.

This gives every phase a runnable target before adding Hugging Face models, Colab training, WixQA preprocessing, and RAGAS-based evaluation.

