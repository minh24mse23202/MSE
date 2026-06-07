from __future__ import annotations

import json

from aragbiz.config import load_config
from aragbiz.data import load_qac_jsonl
from aragbiz.evaluation import Evaluator
from aragbiz.factory import build_sample_pipeline


def main() -> None:
    config = load_config()
    dataset = load_qac_jsonl(config.sample_dataset)
    metrics = Evaluator(build_sample_pipeline(config)).evaluate(dataset)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

