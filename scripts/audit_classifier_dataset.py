from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from aragbiz.data import load_qac_jsonl
from aragbiz.preprocessing import write_qac_jsonl
from aragbiz.training import audit_classifier_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit and optionally combine four-class query-classifier datasets."
    )
    parser.add_argument("--dataset", required=True, help="Primary QAC JSONL path.")
    parser.add_argument(
        "--extra-dataset",
        action="append",
        default=[],
        help="Additional QAC JSONL path. May be specified more than once.",
    )
    parser.add_argument(
        "--output",
        default="docs/evaluation/four_class_dataset_audit.json",
        help="Strict JSON audit report path.",
    )
    parser.add_argument(
        "--combined-output",
        default="",
        help="Optional combined QAC JSONL output used for training.",
    )
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-validation-per-label", type=int, default=50)
    args = parser.parse_args()

    dataset_paths = [Path(args.dataset), *(Path(path) for path in args.extra_dataset)]
    records = []
    inputs: list[dict[str, Any]] = []
    for path in dataset_paths:
        dataset_records = load_qac_jsonl(path)
        records.extend(dataset_records)
        inputs.append(
            {
                "path": str(path),
                "records": len(dataset_records),
                "sha256": file_sha256(path),
            }
        )

    report = audit_classifier_dataset(
        records,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        minimum_validation_per_label=args.minimum_validation_per_label,
    )
    report["inputs"] = inputs
    report["combined_output"] = args.combined_output or None

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if args.combined_output and report["valid_for_training"]:
        write_qac_jsonl(records, Path(args.combined_output))

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not report["ready_for_training"]:
        raise SystemExit(1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
