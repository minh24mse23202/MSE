from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from aragbiz.data import load_qac_jsonl
from aragbiz.preprocessing import write_qac_jsonl
from aragbiz.training import (
    audit_classifier_dataset,
    build_source_balanced_four_class_dataset,
)


DEFAULT_INPUTS = {
    "expertwritten": "data/processed/wixqa_expertwritten_qac.jsonl",
    "simulated": "data/processed/wixqa_simulated_qac.jsonl",
    "official_synthetic": "data/processed/wixqa_synthetic_qac.jsonl",
    "template": "data/processed/wixqa_template_four_class_qac.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and audit a source-aware balanced four-class WixQA dataset."
    )
    parser.add_argument("--expertwritten", default=DEFAULT_INPUTS["expertwritten"])
    parser.add_argument("--simulated", default=DEFAULT_INPUTS["simulated"])
    parser.add_argument("--official-synthetic", default=DEFAULT_INPUTS["official_synthetic"])
    parser.add_argument("--template", default=DEFAULT_INPUTS["template"])
    parser.add_argument("--output", default="data/processed/four_class_qac.jsonl")
    parser.add_argument(
        "--manifest-output",
        default="docs/evaluation/four_class_preparation_manifest.json",
    )
    parser.add_argument(
        "--audit-output",
        default="docs/evaluation/four_class_dataset_audit.json",
    )
    parser.add_argument("--target-per-label", type=int, default=600)
    parser.add_argument("--official-synthetic-limit", type=int, default=200)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--minimum-validation-per-label", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_paths = {
        "wixqa_expertwritten": Path(args.expertwritten),
        "wixqa_simulated": Path(args.simulated),
        "wixqa_synthetic": Path(args.official_synthetic),
        "template_four_class": Path(args.template),
    }
    loaded = {name: load_qac_jsonl(path) for name, path in input_paths.items()}
    records, preparation_manifest = build_source_balanced_four_class_dataset(
        loaded["wixqa_expertwritten"],
        loaded["wixqa_simulated"],
        loaded["wixqa_synthetic"],
        loaded["template_four_class"],
        target_per_label=args.target_per_label,
        official_synthetic_limit=args.official_synthetic_limit,
        seed=args.seed,
    )

    output_path = Path(args.output)
    write_qac_jsonl(records, output_path)
    input_manifest = [
        {
            "source": source,
            "path": str(path),
            "records": len(loaded[source]),
            "sha256": file_sha256(path),
        }
        for source, path in input_paths.items()
    ]
    preparation_manifest["inputs"] = input_manifest
    preparation_manifest["output"] = str(output_path)
    preparation_manifest["output_sha256"] = file_sha256(output_path)
    write_json(preparation_manifest, Path(args.manifest_output))

    audit = audit_classifier_dataset(
        records,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        minimum_validation_per_label=args.minimum_validation_per_label,
    )
    audit["inputs"] = input_manifest
    audit["combined_output"] = str(output_path)
    audit["preparation_manifest"] = str(args.manifest_output)
    write_json(audit, Path(args.audit_output))

    print(
        json.dumps(
            {"preparation": preparation_manifest, "audit": audit},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    if not audit["ready_for_training"]:
        raise SystemExit(1)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
