# Phase 2: Model Specialization

The first specialized model is a lightweight supervised query complexity classifier. It trains on QAC records and predicts one of:

- `simple`
- `moderate`
- `complex`

Train it with:

```powershell
$env:PYTHONPATH='src'
python scripts/generate_synthetic_qac.py --limit 90
python scripts/train_query_classifier.py --extra-dataset data/processed/wixqa_synthetic_bootstrap_qac.jsonl
```

Outputs:

- `data/artifacts/query_classifier_nb.json`
- `docs/evaluation/query_classifier_metrics.json`

The app automatically uses the trained artifact when it exists. If no artifact is available, it falls back to the deterministic heuristic classifier.

The synthetic bootstrap generator creates template-based QAC examples from KB documents to balance the initial `simple`, `moderate`, and `complex` classes. This is a temporary Phase 2 bridge before Colab-based T5/DistilBERT fine-tuning.

For the Colab path, keep the same runtime interface: the trained model wrapper must expose `predict(query) -> complexity_label`.
