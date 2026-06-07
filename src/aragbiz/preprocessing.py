from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

from aragbiz.data import qac_from_mapping
from aragbiz.schemas import Document, QACRecord


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


def wixqa_complexity_label(article_ids: Sequence[str], question: str) -> str:
    if len(article_ids) >= 3:
        return "complex"
    if len(article_ids) == 2:
        return "moderate"
    if len(question.split()) >= 22:
        return "moderate"
    return "simple"


def wixqa_rows_to_qac_records(rows: Iterable[dict], kb_lookup: Dict[str, dict], source_subset: str) -> List[QACRecord]:
    records: List[QACRecord] = []
    for index, row in enumerate(rows, start=1):
        article_ids = [str(article_id) for article_id in row.get("article_ids", [])]
        context_parts = []
        for article_id in article_ids:
            kb_row = kb_lookup.get(article_id)
            if kb_row:
                context_parts.append(str(kb_row.get("contents", "")))
        context = "\n\n".join(part for part in context_parts if part).strip()
        if not context:
            context = str(row.get("answer", ""))
        record_id = str(row.get("id") or f"{source_subset}-{index:05d}")
        records.append(
            qac_from_mapping(
                {
                    "id": record_id,
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "context": context,
                    "complexity_label": wixqa_complexity_label(article_ids, str(row.get("question", ""))),
                    "metadata": {
                        "source": "Wix/WixQA",
                        "subset": source_subset,
                        "article_ids": article_ids,
                    },
                },
                line_number=index,
            )
        )
    return records


def wixqa_kb_rows_to_documents(rows: Iterable[dict]) -> List[Document]:
    documents: List[Document] = []
    for index, row in enumerate(rows, start=1):
        article_id = str(row.get("id") or f"wix-kb-{index:05d}")
        contents = str(row.get("contents", ""))
        documents.append(
            Document(
                id=article_id,
                text=contents,
                metadata={
                    "source": "Wix/WixQA",
                    "url": row.get("url"),
                    "article_type": row.get("article_type"),
                },
            )
        )
    return documents


def write_documents_jsonl(documents: Iterable[Document], path: Union[str, Path]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(
                json.dumps(
                    {
                        "id": document.id,
                        "text": document.text,
                        "metadata": document.metadata,
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
