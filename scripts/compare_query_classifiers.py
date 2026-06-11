from __future__ import annotations

import argparse
import json
from pathlib import Path

from aragbiz.classifier import (
    HeuristicQueryClassifier,
    HuggingFaceQueryClassifier,
    NaiveBayesQueryClassifier,
    T5QueryClassifier,
    evaluate_classifier,
)
from aragbiz.config import load_config
from aragbiz.data import load_qac_jsonl
from aragbiz.factory import existing_dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare query complexity classifiers on one QAC validation set.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--nb-path", default="data/artifacts/query_classifier_nb.json")
    parser.add_argument("--distilbert-path", default="data/artifacts/query_classifier_distilbert")
    parser.add_argument("--t5-path", default="data/artifacts/query_classifier_t5_small")
    parser.add_argument("--output", default="docs/evaluation/classifier_comparison.json")
    args = parser.parse_args()

    config = load_config()
    dataset_path = args.dataset or existing_dataset_path(config)
    records = load_qac_jsonl(dataset_path)
    if args.limit is not None:
        records = records[: args.limit]

    classifiers = {"heuristic": HeuristicQueryClassifier()}
    if Path(args.nb_path).exists():
        classifiers["naive_bayes"] = NaiveBayesQueryClassifier.load(args.nb_path)
    if Path(args.distilbert_path).exists():
        classifiers["distilbert"] = HuggingFaceQueryClassifier(args.distilbert_path)
    if Path(args.t5_path).exists():
        classifiers["t5_small"] = T5QueryClassifier(args.t5_path)

    payload = {
        "dataset": dataset_path,
        "records": len(records),
        "classifiers": {
            name: evaluate_classifier(records, classifier)
            for name, classifier in classifiers.items()
        },
    }
    write_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


def write_json(payload: dict, path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
