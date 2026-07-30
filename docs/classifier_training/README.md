# Final Four-Class Classifier Training

This guide trains the final query-complexity classifiers used by Adaptive
routing:

- DistilBERT: `simple`, `moderate`, `complex`, `advanced`
- T5-small: `simple`, `moderate`, `complex`, `advanced`

Use the same prepared dataset, split parameters, and seed for both models.
Train on a GPU environment, then copy the resulting artifacts back into the
local Aragbiz repository.

## 1. Required outputs

The final local paths are:

```text
data/artifacts/query_classifier_distilbert_v2/
data/artifacts/query_classifier_t5_v2/
docs/evaluation/hf_query_classifier_v2_metrics.json
docs/evaluation/t5_query_classifier_v2_metrics.json
```

Do not overwrite the legacy three-class artifacts. The `v2` directories have
separate built-in AI Model deployment IDs.

## 2. Freeze and audit the dataset locally

Run these commands from the Aragbiz repository root using Python 3.11:

```powershell
$env:PYTHONPATH="src"
python scripts/download_wixqa.py --subset all
python scripts/generate_synthetic_qac.py --limit 2400
python scripts/prepare_four_class_dataset.py --seed 42 --validation-ratio 0.2
```

The command must complete successfully. Verify the audit:

```powershell
python scripts/audit_classifier_dataset.py `
  --dataset data/processed/four_class_qac.jsonl `
  --output docs/evaluation/four_class_dataset_audit.json `
  --validation-ratio 0.2 `
  --seed 42 `
  --minimum-validation-per-label 50
```

Expected high-level properties:

- 2,400 total records
- 600 records for each of the four labels
- source-grouped train/validation split
- no train/validation source-group leakage
- at least 50 validation records per label
- `ready_for_training` is `true`

The dataset is intentionally excluded from Git. Package it with its manifests
for transfer while preserving repository-relative paths:

```powershell
@'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

paths = [
    Path("data/processed/four_class_qac.jsonl"),
    Path("docs/evaluation/four_class_preparation_manifest.json"),
    Path("docs/evaluation/four_class_dataset_audit.json"),
]
with ZipFile("four_class_training_input.zip", "w", ZIP_DEFLATED) as archive:
    for path in paths:
        archive.write(path, path.as_posix())
'@ | python -
```

Keep this ZIP with the final experiment evidence. The preparation manifest
contains the dataset SHA-256 hash.

## 3. Start a GPU environment

### Google Colab

1. Open a new notebook.
2. Select **Runtime > Change runtime type**.
3. Select a GPU accelerator.
4. Run the following checks:

```python
import sys
print(sys.version)
assert sys.version_info[:2] == (3, 11), "Aragbiz training requires Python 3.11"
```

```python
!nvidia-smi
```

### Other GPU environments

Use a clean Python 3.11 virtual environment with a CUDA-compatible PyTorch
installation. A T4-class GPU or better is sufficient for this dataset and
these small models. CPU training is possible but is not recommended for the
final experiment.

## 4. Clone and install Aragbiz

In Colab:

```python
%cd /content
!git clone https://github.com/minh24mse23202/MSE.git aragbiz
%cd /content/aragbiz
!python -m pip install --upgrade pip setuptools wheel
!python -m pip install -e ".[ml]"
```

Confirm the ML runtime:

```python
import torch
import transformers
import datasets

print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
assert torch.cuda.is_available(), "Select a GPU runtime before final training"
```

The check must report `Runtime check passed` and `CUDA: True` before final
training.

An authenticated Hugging Face session is optional for the public base models,
but setting `HF_TOKEN` avoids unauthenticated Hub rate limits.

## 5. Transfer the frozen dataset

Upload `four_class_training_input.zip` to Colab:

```python
from google.colab import files

uploaded = files.upload()
```

Then extract it into the repository root:

```python
!unzip -o /content/aragbiz/four_class_training_input.zip -d /content/aragbiz
```

If the browser uploaded the file under `/content`, use:

```python
!unzip -o /content/four_class_training_input.zip -d /content/aragbiz
```

Verify the transferred dataset against the preparation manifest:

```python
import hashlib
import json
from pathlib import Path

dataset_path = Path("data/processed/four_class_qac.jsonl")
manifest_path = Path("docs/evaluation/four_class_preparation_manifest.json")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
actual_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
expected_hash = manifest["output_sha256"]

print("Records:", sum(1 for _ in dataset_path.open(encoding="utf-8")))
print("Expected SHA-256:", expected_hash)
print("Actual SHA-256:  ", actual_hash)
assert actual_hash == expected_hash
```

Do not regenerate the dataset independently in Colab after this point. Both
models must use this exact frozen file.

## 6. Train DistilBERT

Run:

```python
!python scripts/train_hf_query_classifier.py \
  --dataset data/processed/four_class_qac.jsonl \
  --model-name distilbert-base-uncased \
  --output-dir data/artifacts/query_classifier_distilbert_v2 \
  --metrics-output docs/evaluation/hf_query_classifier_v2_metrics.json \
  --validation-ratio 0.2 \
  --seed 42 \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 128
```

If GPU memory is insufficient, reduce `--batch-size` to `8`. Do not change the
dataset, seed, or validation ratio between models.

Check that these files exist:

```python
from pathlib import Path

distilbert_dir = Path("data/artifacts/query_classifier_distilbert_v2")
required = [
    "config.json",
    "classifier_manifest.json",
    "tokenizer_config.json",
]
for name in required:
    assert (distilbert_dir / name).exists(), name
assert (distilbert_dir / "model.safetensors").exists() or (
    distilbert_dir / "pytorch_model.bin"
).exists()
```

## 7. Train T5-small

Run:

```python
!python scripts/train_t5_query_classifier.py \
  --dataset data/processed/four_class_qac.jsonl \
  --model-name t5-small \
  --output-dir data/artifacts/query_classifier_t5_v2 \
  --metrics-output docs/evaluation/t5_query_classifier_v2_metrics.json \
  --validation-ratio 0.2 \
  --seed 42 \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 3e-4 \
  --max-length 128 \
  --target-max-length 8
```

If GPU memory is insufficient, reduce `--batch-size` to `4`.

Verify the artifact:

```python
from pathlib import Path

t5_dir = Path("data/artifacts/query_classifier_t5_v2")
required = [
    "config.json",
    "classifier_manifest.json",
    "tokenizer_config.json",
]
for name in required:
    assert (t5_dir / name).exists(), name
assert (t5_dir / "model.safetensors").exists() or (
    t5_dir / "pytorch_model.bin"
).exists()
```

## 8. Review final metrics

Each training script evaluates its model on the same deterministic,
source-grouped validation split. Review:

```python
import json
from pathlib import Path

metric_files = [
    Path("docs/evaluation/hf_query_classifier_v2_metrics.json"),
    Path("docs/evaluation/t5_query_classifier_v2_metrics.json"),
]

for path in metric_files:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    print("\n", path.name)
    print("validation records:", payload["validation_records"])
    print("accuracy:", metrics["accuracy"])
    print("macro F1:", metrics["macro_f1"])
    print("advanced recall:", metrics["per_label_recall"]["advanced"])
    print("under-routing rate:", metrics["under_routing_rate"])
    print("over-routing rate:", metrics["over_routing_rate"])
    print("calibration error:", metrics["expected_calibration_error"])
    print("average latency ms:", metrics["average_latency_ms"])
```

Select the final router using more than accuracy:

1. Macro F1.
2. Recall and F1 for `advanced`.
3. Under-routing rate.
4. Expected calibration error.
5. Inference latency.
6. Route-cost regret.

Keep both artifacts even when one becomes the preferred default. The
Evaluation screen can compare RAG configurations that select different
classifier deployments.

## 9. Smoke-test predictions

```python
from aragbiz.classifier import HuggingFaceQueryClassifier, T5QueryClassifier

models = {
    "distilbert": HuggingFaceQueryClassifier(
        "data/artifacts/query_classifier_distilbert_v2"
    ),
    "t5": T5QueryClassifier("data/artifacts/query_classifier_t5_v2"),
}

queries = [
    "What does UAT mean?",
    "How do I submit and approve a purchase request?",
    "Compare the exception and escalation procedures across these workflows.",
    "Investigate the failure, gather evidence from multiple sources, and determine the corrective plan.",
]

for query in queries:
    print("\n", query)
    for name, classifier in models.items():
        result = classifier.predict_scored(query)
        print(name, result.label, result.confidence, result.probabilities)
```

This is a runtime check, not a replacement for validation metrics.

## 10. Package and download artifacts

Create one archive containing both models and their metrics:

```python
!zip -r final_four_class_classifiers.zip \
  data/artifacts/query_classifier_distilbert_v2 \
  data/artifacts/query_classifier_t5_v2 \
  docs/evaluation/hf_query_classifier_v2_metrics.json \
  docs/evaluation/t5_query_classifier_v2_metrics.json \
  docs/evaluation/four_class_preparation_manifest.json \
  docs/evaluation/four_class_dataset_audit.json
```

Download it:

```python
from google.colab import files

files.download("final_four_class_classifiers.zip")
```

Also copy the archive to durable storage such as Google Drive before ending
the GPU session.

## 11. Install artifacts in the local repository

From the local Aragbiz repository root:

```powershell
Expand-Archive -Force `
  -Path .\final_four_class_classifiers.zip `
  -DestinationPath .
```

Confirm the directory layout. Avoid an extra nested `aragbiz` or archive-name
directory:

```powershell
Get-ChildItem data\artifacts\query_classifier_distilbert_v2
Get-ChildItem data\artifacts\query_classifier_t5_v2
```

Install the local ML runtime and smoke-test:

```powershell
python -m pip install -e ".[ml]"
$env:PYTHONPATH="src"
python -c "from aragbiz.classifier import HuggingFaceQueryClassifier; print(HuggingFaceQueryClassifier('data/artifacts/query_classifier_distilbert_v2').predict_scored('Who approves this request?'))"
python -c "from aragbiz.classifier import T5QueryClassifier; print(T5QueryClassifier('data/artifacts/query_classifier_t5_v2').predict_scored('Who approves this request?'))"
```

The default paths are already configured. Optional explicit `.env` values are:

```dotenv
ARAGBIZ_DISTILBERT_V2_CLASSIFIER_MODEL_PATH=data/artifacts/query_classifier_distilbert_v2
ARAGBIZ_T5_V2_CLASSIFIER_MODEL_PATH=data/artifacts/query_classifier_t5_v2
```

## 12. Enable the classifiers in Aragbiz

1. Restart FastAPI after copying the artifacts.
2. Open **AI Models**.
3. Find:
   - **Local DistilBERT Classifier (4-class)**
   - **Local T5-small Classifier (4-class)**
4. Test each local model.
5. Enable the deployments that pass.
6. Open the Main screen.
7. Select a saved RAG Customizer configuration.
8. Choose the classifier deployment under **Adaptive RAG**.
9. Save the configuration.
10. Run a small Adaptive evaluation before the full benchmark matrix.

The stable built-in deployment IDs are:

```text
model-local-distilbert-v2
model-local-t5-classifier-v2
```

## 13. Reproducibility checklist

Record the following in the final report and experiment evidence:

- Git commit used for training.
- Python, Torch, Transformers, and CUDA versions.
- GPU model.
- Dataset SHA-256.
- Split seed and validation ratio.
- Hyperparameters.
- Train and validation record counts.
- Complete metric JSON files.
- Artifact manifests.
- Training duration.
- Any batch-size change or interrupted/restarted run.

Do not report metrics from the training split as final classifier quality.
Do not select a classifier from a few manually written questions.

## Troubleshooting

### `TrainingArguments` rejects `evaluation_strategy`

Use the repository training scripts rather than copying older notebook code.
The scripts detect whether the installed Transformers version expects
`evaluation_strategy` or `eval_strategy`.

### `Trainer` rejects `tokenizer`

The scripts detect whether the installed Transformers version expects
`processing_class` or `tokenizer`.

### CUDA out of memory

Reduce only the batch size first:

- DistilBERT: `16` to `8`
- T5-small: `8` to `4`

Keep the seed, split, model, and sequence lengths unchanged when comparing
models.

### Colab disconnects

Colab storage is temporary. Package and copy each completed artifact to Google
Drive before training the next model.

### Artifact works in Colab but not on Windows

Verify Python 3.11 and reinstall the local ML extra. A Windows PyTorch DLL
failure is an environment problem, not a classifier artifact problem.
