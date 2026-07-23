from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from aragbiz.data import load_qac_jsonl
from aragbiz.preprocessing import write_qac_jsonl
from aragbiz.schemas import QACRecord
from aragbiz.silver_labels import HybridSilverLabeler, ROUTE_ORDER, StrategyOutcome


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible four-class silver complexity labels.")
    parser.add_argument("--dataset", required=True, help="Input QAC JSONL.")
    parser.add_argument("--outcomes", required=True, help="JSONL containing fixed L1-L4 outcome metrics by record ID.")
    parser.add_argument("--output", required=True, help="Labeled QAC JSONL output.")
    parser.add_argument("--provenance-output", required=True, help="Full labeling provenance JSONL output.")
    args = parser.parse_args()

    outcomes = load_outcomes(args.outcomes)

    def run_strategy(record: QACRecord, route: str) -> StrategyOutcome:
        payload = outcomes.get(record.id, {}).get(route, {})
        return StrategyOutcome(
            route=route,
            answer_overlap=float(payload.get("answer_overlap") or 0.0),
            faithfulness_proxy=float(payload.get("faithfulness_proxy") or 0.0),
            latency_ms=float(payload.get("latency_ms") or 0.0),
            estimated_cost_usd=float(payload.get("estimated_cost_usd") or 0.0),
            success=payload.get("judge_success"),
            metadata=payload.get("metadata") or {},
        )

    labeler = HybridSilverLabeler(run_strategy)
    records = load_qac_jsonl(args.dataset)
    provenance = [labeler.label(record) for record in records]
    labels = {item["record_id"]: item for item in provenance}
    labeled = [
        QACRecord(
            id=record.id,
            question=record.question,
            answer=record.answer,
            context=record.context,
            complexity_label=labels[record.id]["complexity_label"],
            metadata={**record.metadata, "complexity_labeling": labels[record.id]},
        )
        for record in records
    ]
    write_qac_jsonl(labeled, Path(args.output))
    write_jsonl(provenance, args.provenance_output)
    print(json.dumps({"records": len(labeled), "routes": list(ROUTE_ORDER), "output": args.output}, indent=2))


def load_outcomes(path: str) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            values[str(payload["record_id"])] = dict(payload.get("strategies") or {})
    return values


def write_jsonl(records: list[Dict[str, Any]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
