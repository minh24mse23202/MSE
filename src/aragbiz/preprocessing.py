from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Union

from aragbiz.data import qac_from_mapping
from aragbiz.schemas import QACRecord


def normalize_rows(rows: Iterable[dict]) -> List[QACRecord]:
    records: List[QACRecord] = []
    for index, row in enumerate(rows, start=1):
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"raw_metadata": metadata}
        records.append(
            qac_from_mapping(
                {
                    "id": row.get("id") or f"row-{index}",
                    "question": row.get("question") or row.get("query"),
                    "answer": row.get("answer") or row.get("response"),
                    "context": row.get("context") or row.get("document") or row.get("passage"),
                    "complexity_label": row.get("complexity_label") or row.get("label"),
                    "metadata": metadata,
                },
                line_number=index,
            )
        )
    return records


def read_csv_dataset(path: Union[str, Path]) -> List[QACRecord]:
    with Path(path).open("r", encoding="utf-8", newline="") as file:
        return normalize_rows(csv.DictReader(file))


def write_qac_jsonl(records: Iterable[QACRecord], path: Union[str, Path]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(
                    {
                        "id": record.id,
                        "question": record.question,
                        "answer": record.answer,
                        "context": record.context,
                        "complexity_label": record.complexity_label,
                        "metadata": record.metadata,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
