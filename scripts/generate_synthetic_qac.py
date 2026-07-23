from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from aragbiz.data import load_documents_jsonl
from aragbiz.preprocessing import write_qac_jsonl
from aragbiz.schemas import Document, QACRecord


LABEL_DOCUMENT_COUNTS = {
    "simple": 0,
    "moderate": 1,
    "complex": 2,
    "advanced": 4,
}

SIMPLE_CONCEPTS = [
    ("approval", "Approval is formal permission to proceed with a workflow action."),
    ("approver", "An approver is the person or role authorized to accept or reject a request."),
    ("SLA", "An SLA is a service-level agreement that defines expected service targets."),
    ("escalation", "Escalation routes an issue to a higher authority or support level."),
    ("handoff", "A handoff transfers responsibility for work from one role or team to another."),
    ("prerequisite", "A prerequisite is a condition that must be satisfied before a step begins."),
    ("exception", "An exception is a case that cannot follow the workflow's normal path."),
    ("task owner", "A task owner is accountable for completing or coordinating a task."),
    ("workflow trigger", "A workflow trigger is an event or condition that starts a workflow."),
    ("workflow status", "Workflow status indicates the current state of a process instance."),
    ("audit trail", "An audit trail is a chronological record of workflow actions and changes."),
    ("due date", "A due date is the deadline by which a task should be completed."),
    ("work queue", "A work queue is an ordered collection of tasks waiting to be processed."),
    ("assignee", "An assignee is the person or group currently responsible for a task."),
    ("requester", "A requester is the person or system that initiates a request."),
    ("reviewer", "A reviewer examines work for correctness, completeness, or compliance."),
    ("decision point", "A decision point selects the next workflow path based on a condition."),
    ("business rule", "A business rule defines a condition or constraint applied by a process."),
    ("workflow input", "A workflow input is information or material required by a process."),
    ("workflow output", "A workflow output is the result produced by a process."),
    ("process step", "A process step is one defined unit of work in a workflow."),
    ("dependency", "A dependency is a relationship where one task relies on another."),
    ("control", "A control is a safeguard that reduces process, compliance, or operational risk."),
    ("policy", "A policy states the principles and requirements that guide decisions."),
    ("procedure", "A procedure gives the ordered instructions for performing an activity."),
    ("KPI", "A KPI is a key performance indicator used to measure process performance."),
    ("bottleneck", "A bottleneck is a constrained step that limits overall process throughput."),
    ("rework", "Rework is repeated effort needed to correct or complete prior work."),
    ("cycle time", "Cycle time is the elapsed time required to complete a process or task."),
    ("completion criterion", "A completion criterion defines when a task is considered finished."),
]

SIMPLE_QUESTION_TEMPLATES = [
    "What does {concept} mean in a business workflow?",
    "Define {concept}.",
    "Give a concise definition of {concept}.",
    "In workflow terminology, what is {concept}?",
    "What is the meaning of {concept}?",
    "Briefly explain {concept}.",
    "How would you describe {concept}?",
    "What does the term {concept} refer to?",
    "Explain the basic idea of {concept}.",
    "What is a simple explanation of {concept}?",
    "What should a workflow user understand by {concept}?",
    "Provide a one-sentence definition of {concept}.",
    "What is {concept} in process management?",
    "Describe {concept} in plain language.",
    "What does {concept} represent in a process?",
    "State the purpose of {concept} in one sentence.",
    "What is commonly meant by {concept}?",
    "Summarize the concept of {concept}.",
    "What is the role of {concept} in workflow vocabulary?",
    "Give the standard workflow meaning of {concept}.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple synthetic QAC records from KB documents.")
    parser.add_argument("--documents", default="data/processed/wix_kb_corpus_documents.jsonl")
    parser.add_argument("--output", default="data/processed/wixqa_template_four_class_qac.jsonl")
    parser.add_argument("--limit", type=int, default=90, help="Total synthetic records to generate.")
    args = parser.parse_args()

    documents = load_documents_jsonl(args.documents)
    try:
        records = generate_synthetic_records(documents, args.limit)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    write_qac_jsonl(records, Path(args.output))
    print(f"Wrote {len(records)} synthetic QAC records to {args.output}")


def generate_synthetic_records(documents: Sequence[Document], limit: int) -> list[QACRecord]:
    if limit < 4:
        raise ValueError("The synthetic record limit must be at least four.")
    label_counts = synthetic_label_counts(limit)
    required_documents = sum(
        label_counts[label] * documents_per_record
        for label, documents_per_record in LABEL_DOCUMENT_COUNTS.items()
    )
    if len(documents) < required_documents:
        raise ValueError(
            f"{required_documents} source documents are required to generate {limit} records "
            "without source overlap; "
            f"only {len(documents)} were provided."
        )

    records: list[QACRecord] = []
    document_cursor = 0
    record_index = 1

    def take_documents(count: int) -> Sequence[Document]:
        nonlocal document_cursor
        selected = documents[document_cursor : document_cursor + count]
        document_cursor += count
        return selected

    simple_variants = [
        (template.format(concept=concept), answer)
        for concept, answer in SIMPLE_CONCEPTS
        for template in SIMPLE_QUESTION_TEMPLATES
    ]
    if label_counts["simple"] > len(simple_variants):
        raise ValueError(
            f"At most {len(simple_variants)} unique no-retrieval simple examples are supported."
        )
    for question, answer in simple_variants[: label_counts["simple"]]:
        records.append(make_simple_record(record_index, question, answer))
        record_index += 1

    for _ in range(label_counts["moderate"]):
        document = take_documents(1)[0]
        title = first_sentence(document.text)[:100]
        records.append(
            make_record(
                record_index,
                "moderate",
                [document],
                f"According to the relevant workflow document, what steps apply to {title}?",
            )
        )
        record_index += 1

    for _ in range(label_counts["complex"]):
        first, second = take_documents(2)
        first_title = first_sentence(first.text)[:80]
        second_title = first_sentence(second.text)[:80]
        records.append(
            make_record(
                record_index,
                "complex",
                [first, second],
                f"How should the workflow combine {first_title} with the dependency on {second_title}?",
            )
        )
        record_index += 1

    for _ in range(label_counts["advanced"]):
        selected = list(take_documents(4))
        titles = [first_sentence(document.text)[:55] for document in selected]
        records.append(
            make_record(
                record_index,
                "advanced",
                selected,
                (
                    f"Plan the end-to-end workflow when {titles[0]} depends on {titles[1]}, "
                    f"must handle an exception from {titles[2]}, and requires final evidence from {titles[3]}. "
                    "Identify conditional branches, owners, recovery actions, and approval order."
                ),
            )
        )
        record_index += 1
    return records


def synthetic_label_counts(limit: int) -> dict[str, int]:
    per_label, remainder = divmod(limit, len(LABEL_DOCUMENT_COUNTS))
    counts = {label: per_label for label in LABEL_DOCUMENT_COUNTS}
    labels = list(LABEL_DOCUMENT_COUNTS)
    for index in range(remainder):
        counts[labels[index]] += 1
    return counts


def make_record(index, label, documents, question):
    return QACRecord(
        id=f"template-four-class-{index:05d}",
        question=question,
        answer="\n\n".join(first_paragraph(document.text) for document in documents),
        context="\n\n".join(document.text for document in documents),
        complexity_label=label,
        metadata={
            "source": "aragbiz/template",
            "subset": "template_four_class",
            "article_ids": [document.id for document in documents],
            "source_group_id": f"template-four-class-{index:05d}",
            "generator": "template",
            "retrieval_required": True,
            "evidence_document_count": len(documents),
        },
    )


def make_simple_record(index: int, question: str, answer: str) -> QACRecord:
    return QACRecord(
        id=f"template-four-class-{index:05d}",
        question=question,
        answer=answer,
        context="",
        complexity_label="simple",
        metadata={
            "source": "aragbiz/template",
            "subset": "template_four_class",
            "article_ids": [],
            "source_group_id": f"template-four-class-{index:05d}",
            "generator": "template",
            "retrieval_required": False,
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
