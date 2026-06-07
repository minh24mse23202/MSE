from aragbiz.data import load_documents_jsonl
from aragbiz.preprocessing import (
    wixqa_kb_rows_to_documents,
    wixqa_rows_to_qac_records,
    write_documents_jsonl,
)


def test_wixqa_rows_join_article_contexts():
    kb_rows = [
        {"id": "a1", "contents": "First article steps.", "url": "https://example.com/a1", "article_type": "article"},
        {"id": "a2", "contents": "Second article details.", "url": "https://example.com/a2", "article_type": "article"},
    ]
    qa_rows = [
        {
            "question": "How do I combine these workflow steps?",
            "answer": "Use both articles.",
            "article_ids": ["a1", "a2"],
        }
    ]
    records = wixqa_rows_to_qac_records(qa_rows, {row["id"]: row for row in kb_rows}, "wixqa_expertwritten")
    assert records[0].complexity_label == "moderate"
    assert "First article steps." in records[0].context
    assert records[0].metadata["article_ids"] == ["a1", "a2"]


def test_wixqa_documents_round_trip(tmp_path):
    documents = wixqa_kb_rows_to_documents(
        [{"id": "a1", "contents": "Article body", "url": "https://example.com", "article_type": "known_issue"}]
    )
    output_path = tmp_path / "docs.jsonl"
    write_documents_jsonl(documents, output_path)
    loaded = load_documents_jsonl(output_path)
    assert loaded[0].id == "a1"
    assert loaded[0].metadata["article_type"] == "known_issue"
