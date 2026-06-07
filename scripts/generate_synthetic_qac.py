from __future__ import annotations

import argparse
from pathlib import Path

from aragbiz.data import load_documents_jsonl
from aragbiz.preprocessing import write_qac_jsonl
from aragbiz.schemas import QACRecord


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple synthetic QAC records from KB documents.")
    parser.add_argument("--documents", default="data/processed/wix_kb_corpus_documents.jsonl")
    parser.add_argument("--output", default="data/processed/wixqa_synthetic_bootstrap_qac.jsonl")
    parser.add_argument("--limit", type=int, default=90, help="Total synthetic records to generate.")
    args = parser.parse_args()

    documents = load_documents_jsonl(args.documents)
    records = []
    per_label = max(1, args.limit // 3)
    for index in range(per_label):
        document = documents[index % len(documents)]
        title = first_sentence(document.text)[:120]
        records.append(make_record(index + 1, "simple", [document], f"What are the key workflow steps for {title}?"))

    offset = per_label
    for index in range(per_label):
        first = documents[(index * 2) % len(documents)]
        second = documents[(index * 2 + 1) % len(documents)]
        first_title = first_sentence(first.text)[:80]
        second_title = first_sentence(second.text)[:80]
        records.append(
            make_record(
                offset + index + 1,
                "moderate",
                [first, second],
                f"How do I combine the workflow for {first_title} with {second_title}?",
            )
        )

    offset = per_label * 2
    for index in range(args.limit - len(records)):
        first = documents[(index * 3) % len(documents)]
        second = documents[(index * 3 + 1) % len(documents)]
        third = documents[(index * 3 + 2) % len(documents)]
        first_title = first_sentence(first.text)[:70]
        second_title = first_sentence(second.text)[:70]
        third_title = first_sentence(third.text)[:70]
        records.append(
            make_record(
                offset + index + 1,
                "complex",
                [first, second, third],
                f"What should I do when {first_title} depends on {second_title} and then affects {third_title}?",
            )
        )
    write_qac_jsonl(records, Path(args.output))
    print(f"Wrote {len(records)} synthetic QAC records to {args.output}")


def make_record(index, label, documents, question):
    return QACRecord(
        id=f"synthetic-bootstrap-{index:05d}",
        question=question,
        answer="\n\n".join(first_paragraph(document.text) for document in documents),
        context="\n\n".join(document.text for document in documents),
        complexity_label=label,
        metadata={
            "source": "synthetic_bootstrap",
            "article_ids": [document.id for document in documents],
            "generator": "template",
        },
    )


def first_paragraph(text: str) -> str:
    for paragraph in text.splitlines():
        if paragraph.strip():
            return paragraph.strip()
    return text[:500].strip()


def first_sentence(text: str) -> str:
    paragraph = first_paragraph(text)
    for delimiter in [".", "?", "!"]:
        if delimiter in paragraph:
            return paragraph.split(delimiter, 1)[0].strip()
    return paragraph.strip()


if __name__ == "__main__":
    main()
