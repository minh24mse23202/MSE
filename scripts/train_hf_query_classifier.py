from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from aragbiz.classifier import evaluate_classifier
from aragbiz.config import load_config
from aragbiz.data import load_qac_jsonl
from aragbiz.factory import existing_dataset_path
from aragbiz.schemas import COMPLEXITY_LABELS, QACRecord


LABEL2ID = {label: index for index, label in enumerate(COMPLEXITY_LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a Hugging Face query complexity classifier.")
    parser.add_argument("--dataset", default=None, help="Primary QAC JSONL path.")
    parser.add_argument("--extra-dataset", action="append", default=[], help="Additional QAC JSONL path.")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="data/artifacts/query_classifier_distilbert")
    parser.add_argument("--metrics-output", default="docs/evaluation/hf_query_classifier_metrics.json")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    transformers, datasets, sklearn = import_ml_dependencies()
    AutoModelForSequenceClassification = transformers.AutoModelForSequenceClassification
    AutoTokenizer = transformers.AutoTokenizer
    Trainer = transformers.Trainer
    TrainingArguments = transformers.TrainingArguments
    Dataset = datasets.Dataset
    accuracy_score = sklearn.metrics.accuracy_score
    f1_score = sklearn.metrics.f1_score

    config = load_config()
    dataset_path = args.dataset or existing_dataset_path(config)
    records = load_qac_jsonl(dataset_path)
    for extra_dataset in args.extra_dataset:
        records.extend(load_qac_jsonl(extra_dataset))
    train_records, validation_records = split_records(records, args.validation_ratio, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(COMPLEXITY_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = make_dataset(Dataset, train_records, tokenizer, args.max_length)
    validation_dataset = make_dataset(Dataset, validation_records, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda output: compute_metrics(output, accuracy_score, f1_score),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    from aragbiz.classifier import HuggingFaceQueryClassifier

    classifier = HuggingFaceQueryClassifier(args.output_dir, max_length=args.max_length)
    metrics_payload = {
        "dataset": dataset_path,
        "extra_datasets": args.extra_dataset,
        "model_name": args.model_name,
        "model_path": args.output_dir,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "metrics": evaluate_classifier(validation_records, classifier),
    }
    write_json(metrics_payload, args.metrics_output)
    print(json.dumps(metrics_payload, indent=2, sort_keys=True))


def import_ml_dependencies():
    try:
        import datasets
        import sklearn
        import transformers
    except ImportError as exc:
        raise SystemExit(
            "Missing optional ML dependencies. Install them with: python -m pip install -e \".[ml]\""
        ) from exc
    return transformers, datasets, sklearn


def make_dataset(Dataset, records: List[QACRecord], tokenizer, max_length: int):
    dataset = Dataset.from_dict(
        {
            "text": [record.question for record in records],
            "label": [LABEL2ID[record.complexity_label] for record in records],
        }
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)

    return dataset.map(tokenize, batched=True).remove_columns(["text"])


def compute_metrics(eval_pred, accuracy_score, f1_score) -> Dict[str, float]:
    predictions, labels = eval_pred
    predicted_ids = predictions.argmax(axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predicted_ids)),
        "macro_f1": float(f1_score(labels, predicted_ids, average="macro", zero_division=0)),
    }


def split_records(records: List[QACRecord], validation_ratio: float, seed: int) -> Tuple[List[QACRecord], List[QACRecord]]:
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_ratio))
    return shuffled[validation_size:], shuffled[:validation_size]


def write_json(payload: dict, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
