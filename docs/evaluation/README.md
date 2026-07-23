# Evaluation

Evaluation tracks:

- routing accuracy;
- context relevance;
- faithfulness proxy or RAGAS faithfulness when configured;
- answer overlap for benchmark answers;
- latency and runtime/cost proxy;
- ablations for `top_k`, hybrid weights, routing thresholds, and prompts.

Run the baseline sample:

```powershell
python scripts/download_wixqa.py --subset wixqa_expertwritten
python scripts/evaluate_sample.py
```


## Phase 4 live evaluation

The Phase 4 implementation adds persisted evaluation runs for the current L1/L2/L3/L4 Adaptive RAG pipeline.

- Adaptive run: `mode=adaptive`, so classifier labels route queries to L1 Direct, L2 Simple RAG, L3 Complex RAG, or L4 Advanced RAG.
- Static baseline: `mode=simple_rag`, using the same knowledge base, retrieval mode, top-k, and chat configuration.
- Results are stored in PostgreSQL when available, otherwise in `data/knowledge/evaluation_store.json`.
- The React Evaluation screen can create runs, compare metrics, inspect case-level traces, and embed run-level RAGXplain insights.

## RAGXplain judge and viewer

RAGXplain remains a sibling project. Install it into the active Python 3.11 environment:

```powershell
python -m pip install -e "..\ragxplain"
```

The default configuration uses `examples.mock_judge_impl:judge` so the integration can be smoke-tested without credentials. Configure a real judge implementation before collecting capstone results:

```powershell
$env:ARAGBIZ_RAGXPLAIN_ROOT="..\ragxplain"
$env:ARAGBIZ_RAGXPLAIN_JUDGE="your_judge_module:judge"
```

Enable **Run RAGXplain LLM Judge** on the Evaluation screen. A completed run stores native artifacts under `docs/evaluation/results/<run_id>/ragxplain/`. **Open RAGXplain insights** embeds the sibling viewer and loads `overall_insights.json` automatically; manual drag-and-drop remains available as a fallback.

The integration exposes:

- `GET /evaluation/runs/{run_id}/ragxplain/overall-insights`
- `GET /evaluation/ragxplain/viewer`

CLI snapshot example:

```powershell
python scripts/evaluate_adaptive.py --knowledge-base-id <kb-id> --limit 20 --retrieval-mode hybrid --top-k 4 --run-ragxplain
```

Snapshots are saved under `docs/evaluation/results/`.

## Four-class classifier evaluation

The four-class training scripts report accuracy, macro F1, per-label recall, advanced precision/recall/F1, expected calibration error, latency, under-routing, over-routing, and route-cost regret. Splits are source-grouped and stratified to prevent records sharing a document from appearing in both train and validation sets. The generated manifest warns when a validation class contains fewer than 50 examples.

Download all three official WixQA QA subsets, generate the separately named template supplement, and build a source-aware balanced dataset:

```powershell
python scripts/download_wixqa.py --subset all
python scripts/generate_synthetic_qac.py --limit 2400
python scripts/prepare_four_class_dataset.py
```

By default, preparation retains all unique ExpertWritten and Simulated records, deterministically samples 200 official WixQA Synthetic records, and fills each class to 600 with template records. The template generator uses disjoint source-document sets for each example. The preparation manifest records per-source selection and SHA-256 hashes; the audit verifies class coverage, duplicate and conflicting records, grouped split sizes, and train/validation source isolation.

Hybrid silver labels select the least-complex successful route using fixed L1-L4 outcome snapshots:

```powershell
python scripts/generate_silver_complexity_labels.py --dataset data/processed/input.jsonl --knowledge-base-id <kb-id> --output data/processed/four_class_qac.jsonl --provenance-output docs/evaluation/four_class_label_provenance.jsonl --judge-deployment-id <judge-id>
```

To relabel from previously captured route outcomes without making model calls:

```powershell
python scripts/build_silver_complexity_labels.py --dataset data/processed/input.jsonl --outcomes data/processed/route_outcomes.jsonl --output data/processed/four_class_qac.jsonl --provenance-output docs/evaluation/four_class_label_provenance.jsonl
```
