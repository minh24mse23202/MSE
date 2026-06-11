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

Train the DistilBERT classifier in Colab or another GPU environment:

```powershell
python -m pip install -e ".[ml]"
$env:PYTHONPATH='src'
python scripts/train_hf_query_classifier.py --extra-dataset data/processed/wixqa_synthetic_bootstrap_qac.jsonl
```

Train the T5-small seq2seq classifier:

```powershell
python -m pip install -e ".[ml]"
$env:PYTHONPATH='src'
python scripts/train_t5_query_classifier.py --extra-dataset data/processed/wixqa_synthetic_bootstrap_qac.jsonl
```

Compare available classifiers:

```powershell
$env:PYTHONPATH='src'
python scripts/compare_query_classifiers.py --limit 50
```

Outputs:

- `data/artifacts/query_classifier_nb.json`
- `data/artifacts/query_classifier_distilbert/`
- `data/artifacts/query_classifier_t5_small/`
- `docs/evaluation/query_classifier_metrics.json`
- `docs/evaluation/hf_query_classifier_metrics.json`
- `docs/evaluation/t5_query_classifier_metrics.json`
- `docs/evaluation/classifier_comparison.json`

The app uses classifier artifacts in this order:

1. Hugging Face directory at `data/artifacts/query_classifier_distilbert/`
2. Naive Bayes JSON at `data/artifacts/query_classifier_nb.json`
3. deterministic heuristic classifier

The synthetic bootstrap generator creates template-based QAC examples from KB documents to balance the initial `simple`, `moderate`, and `complex` classes. This is a temporary Phase 2 bridge before Colab-based T5/DistilBERT fine-tuning.

For the Colab path, keep the same runtime interface: the trained model wrapper must expose `predict(query) -> complexity_label`.
