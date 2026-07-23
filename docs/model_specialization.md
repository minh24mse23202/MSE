# Phase 2: Model Specialization

Use Python 3.11 for Phase 2 local training commands. The specialized model is a lightweight supervised query complexity classifier. It predicts one of:

- `simple`
- `moderate`
- `complex`
- `advanced`

Prepare the source-aware dataset with all official WixQA QA configurations:

```powershell
$env:PYTHONPATH='src'
python scripts/download_wixqa.py --subset all
python scripts/generate_synthetic_qac.py --limit 2400
python scripts/prepare_four_class_dataset.py
```

Train the DistilBERT classifier in Colab or another GPU environment:

```powershell
python -m pip install -e ".[dev,api,app,ml]"
$env:PYTHONPATH='src'
python scripts/train_hf_query_classifier.py --dataset data/processed/four_class_qac.jsonl
```

Train the T5-small seq2seq classifier:

```powershell
python -m pip install -e ".[dev,api,app,ml]"
$env:PYTHONPATH='src'
python scripts/train_t5_query_classifier.py --dataset data/processed/four_class_qac.jsonl
```

Compare available classifiers:

```powershell
$env:PYTHONPATH='src'
python scripts/compare_query_classifiers.py --limit 50
```

Outputs:

- `data/artifacts/query_classifier_nb.json`
- `data/artifacts/query_classifier_distilbert_v2/`
- `data/artifacts/query_classifier_t5_v2/`
- `docs/evaluation/query_classifier_metrics.json`
- `docs/evaluation/hf_query_classifier_metrics.json`
- `docs/evaluation/t5_query_classifier_metrics.json`
- `docs/evaluation/classifier_comparison.json`

The app uses classifier artifacts in this order:

1. Hugging Face directory at `data/artifacts/query_classifier_distilbert/`
2. Naive Bayes JSON at `data/artifacts/query_classifier_nb.json`
3. deterministic heuristic classifier

`wixqa_template_four_class_qac.jsonl` is an Aragbiz-generated balancing supplement and is distinct from the official `wixqa_synthetic` subset. The final preparation manifest records exactly how much each source contributes.

For the Colab path, keep the same runtime interface: the trained model wrapper must expose `predict(query) -> complexity_label`.
