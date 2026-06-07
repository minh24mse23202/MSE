from aragbiz.data import load_qac_jsonl
from aragbiz.preprocessing import normalize_rows


def test_load_sample_qac_records():
    records = load_qac_jsonl("data/sample/business_workflows.jsonl")
    assert len(records) >= 3
    assert {record.complexity_label for record in records} == {"simple", "moderate", "complex"}


def test_preprocessing_creates_valid_qac_records():
    records = normalize_rows(
        [
            {
                "id": "one",
                "query": "How do I approve a PO?",
                "response": "Review and approve it.",
                "passage": "Approvers review purchase orders.",
                "label": "simple",
            }
        ]
    )
    assert records[0].question == "How do I approve a PO?"
    assert records[0].complexity_label == "simple"

