from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

from aragbiz.classifier import evaluate_classifier, train_naive_bayes_classifier
from aragbiz.config import load_config
from aragbiz.data import load_qac_jsonl
from aragbiz.factory import existing_dataset_path
from aragbiz.schemas import QACRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Phase 2 lightweight query complexity classifier.")
    parser.add_argument("--dataset", default=None, help="QAC JSONL path. Defaults to configured WixQA file with fallback.")
    parser.add_argument("--extra-dataset", action="append", default=[], help="Additional QAC JSONL path to include in training.")
    parser.add_argument("--output", default=None, help="Classifier artifact path.")
    parser.add_argument("--metrics-output", default="docs/evaluation/query_classifier_metrics.json")
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config()
    dataset_path = args.dataset or existing_dataset_path(config)
    output_path = args.output or config.classifier_model_path
    records = load_qac_jsonl(dataset_path)
    for extra_dataset in args.extra_dataset:
        records.extend(load_qac_jsonl(extra_dataset))
    train_records, validation_records = split_records(records, args.validation_ratio, args.seed)

    classifier = train_naive_bayes_classifier(train_records)
    classifier.save(output_path)
    metrics = evaluate_classifier(validation_records, classifier)
    metrics_payload = {
        "dataset": dataset_path,
        "model_path": output_path,
        "train_records": len(train_records),
        "validation_records": len(validation_records),
        "metrics": metrics,
    }
    write_json(metrics_payload, args.metrics_output)
    print(json.dumps(metrics_payload, indent=2, sort_keys=True))


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
