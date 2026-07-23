import pytest

from aragbiz.schemas import Document
from scripts.generate_synthetic_qac import generate_synthetic_records


def _documents(count):
    return [
        Document(
            id=f"document-{index}",
            text=f"Workflow document {index}. Follow the documented business process.",
            metadata={},
        )
        for index in range(count)
    ]


def test_synthetic_generation_is_balanced_and_uses_disjoint_sources():
    records = generate_synthetic_records(_documents(70), limit=40)

    counts = {
        label: sum(record.complexity_label == label for record in records)
        for label in ("simple", "moderate", "complex", "advanced")
    }
    source_ids = [
        source_id
        for record in records
        for source_id in record.metadata["article_ids"]
    ]

    assert counts == {"simple": 10, "moderate": 10, "complex": 10, "advanced": 10}
    assert len(source_ids) == len(set(source_ids)) == 70
    assert all(
        not record.metadata["article_ids"]
        for record in records
        if record.complexity_label == "simple"
    )


def test_synthetic_generation_reports_required_document_count():
    with pytest.raises(ValueError, match="70 source documents"):
        generate_synthetic_records(_documents(69), limit=40)
