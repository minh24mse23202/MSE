from __future__ import annotations

import argparse
import json

from aragbiz.config import load_config
from aragbiz.data import load_qac_jsonl
from aragbiz.evaluation import Evaluator
from aragbiz.factory import build_sample_pipeline, existing_dataset_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the configured Adaptive RAG pipeline.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of QAC records to evaluate.")
    args = parser.parse_args()

    config = load_config()
    dataset = load_qac_jsonl(existing_dataset_path(config))
    if args.limit is not None:
        dataset = dataset[: args.limit]
    metrics = Evaluator(build_sample_pipeline(config)).evaluate(dataset)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
