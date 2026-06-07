from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Union

from aragbiz.schemas import COMPLEXITY_LABELS, ComplexityLabel, Document, QACRecord


def load_qac_jsonl(path: Union[str, Path]) -> List[QACRecord]:
    records: List[QACRecord] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(qac_from_mapping(payload, line_number=line_number))
    return records


def qac_from_mapping(payload: dict, line_number: Optional[int] = None) -> QACRecord:
    missing = {"id", "question", "answer", "context", "complexity_label"} - set(payload)
    if missing:
        location = f" on line {line_number}" if line_number else ""
        raise ValueError(f"Missing QAC fields{location}: {sorted(missing)}")
    label = payload["complexity_label"]
    if label not in COMPLEXITY_LABELS:
        raise ValueError(f"Invalid complexity label: {label!r}")
    return QACRecord(
        id=str(payload["id"]),
        question=str(payload["question"]),
        answer=str(payload["answer"]),
        context=str(payload["context"]),
        complexity_label=label,
        metadata=dict(payload.get("metadata", {})),
    )


def records_to_documents(records: Iterable[QACRecord]) -> List[Document]:
    documents: List[Document] = []
    for record in records:
        documents.append(
            Document(
                id=record.id,
                text=f"{record.question}\n{record.context}",
                metadata={
                    **record.metadata,
                    "question": record.question,
                    "answer": record.answer,
                    "complexity_label": record.complexity_label,
                },
            )
        )
    return documents


def validate_complexity_label(label: str) -> ComplexityLabel:
    if label not in COMPLEXITY_LABELS:
        raise ValueError(f"Expected one of {COMPLEXITY_LABELS}, got {label!r}")
    return label  # type: ignore[return-value]
