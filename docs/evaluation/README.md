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


## WixQA configuration benchmark

The Evaluation screen creates a durable experiment matrix from selected saved RAG Customizer configurations and selected WixQA subsets. Every configuration receives the same deterministic case sample for each subset.

Each matrix cell executes exactly one saved configuration using its configured
route (`adaptive`, L1, L2, L3, or L4). There is no implicit static-L2 baseline
call. To compare Adaptive with L2, save one configuration for each route and
include both in the same experiment; the leaderboard compares their independent
results.

The primary WixQA metrics are token F1, BLEU, ROUGE-1, ROUGE-2, LLM-judged Context Recall, and LLM-judged Factuality. The default ranking assigns 30% each to Context Recall and Factuality and distributes the remaining 40% across lexical metrics. Dataset scores are macro-averaged so Synthetic does not dominate ExpertWritten and Simulated.

Before running an experiment:

1. Apply `alembic upgrade head`.
2. Start the API and `python -m aragbiz.worker`.
3. Create an enabled Model Farm deployment with the `judge` capability.
4. Enable external processing on the WixQA Knowledge Base when the judge is remote.
5. Save at least one RAG Customizer configuration.

Experiment APIs:

- `GET /evaluation/datasets`
- `POST /evaluation/experiments`
- `GET /evaluation/experiments/{id}`
- `GET /evaluation/experiments/{id}/runs`
- `GET /evaluation/experiments/{id}/leaderboard`
- `POST /evaluation/experiments/{id}/cancel`
- `POST /evaluation/experiments/{id}/resume`

## RAGXplain judge and viewer

RAGXplain remains a sibling project. Install it into the active Python 3.11 environment:

```powershell
python -m pip install -e "..\ragxplain"
```

After a configuration-dataset run completes, select it and start a bounded RAGXplain diagnosis. The default stratified limit is 100 cases. RAGXplain uses the experiment's registered Model Farm judge instead of a separate Python judge module. A completed diagnosis stores native artifacts under `docs/evaluation/results/<run_id>/ragxplain/`. **Open RAGXplain insights** embeds the sibling viewer and loads `overall_insights.json` automatically.

Artifacts created by the legacy inline integration may identify the judge as
`examples.mock_judge_impl:judge`. The Evaluation screen marks these artifacts as
legacy and requires **Run diagnosis** again before opening them as current
insights. Restart both the API and worker after upgrading so the rerun uses the
selected Model Farm judge.

The integration exposes:

- `GET /evaluation/runs/{run_id}/ragxplain/overall-insights`
- `POST /evaluation/runs/{run_id}/ragxplain`
- `GET /evaluation/ragxplain/viewer`

CLI snapshot example:

```powershell
python scripts/evaluate_adaptive.py --knowledge-base-id <kb-id> --chat-configuration-id <config-id> --limit 20 --retrieval-mode hybrid --top-k 4
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
