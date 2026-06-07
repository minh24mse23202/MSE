# Phase Checklist

## Phase 1: Literature Review and Dataset Preparation

- Capture notes for RAG, Adaptive RAG, hybrid retrieval, and evaluation frameworks.
- Normalize business workflow QA data into QAC JSONL.
- Define complexity labeling guidelines and a validation split.

## Phase 2: Model Specialization

- Generate synthetic QAC examples where useful.
- Fine-tune or distill a lightweight classifier.
- Track accuracy, macro F1, confusion matrix, and per-label recall.

## Phase 3: Adaptive RAG Pipeline

- Replace baseline retriever/generator pieces with stronger model-backed components.
- Add decomposition for complex queries.
- Keep the current interfaces stable.

## Phase 4: Evaluation and Optimization

- Compare adaptive and static RAG.
- Run retrieval and prompt ablations.
- Save metric outputs and config snapshots.

## Phase 5: Prototype Chatbot

- Run FastAPI and Streamlit together.
- Show answer provenance and route metadata.
- Record optional feedback.

## Phase 6: Final Report and Defense

- Prepare reproducible demo.
- Finalize slides, report, and architecture/evaluation figures.

