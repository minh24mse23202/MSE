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

The Phase 4 implementation adds persisted evaluation runs for the current L1/L2/L3 Adaptive RAG pipeline.

- Adaptive run: `mode=adaptive`, so classifier labels route queries to L1 Direct, L2 Simple RAG, or L3 Complex RAG.
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
