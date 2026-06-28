import json

from aragbiz.knowledge import (
    HashEmbeddingModel,
    KnowledgeService,
    OverlapChunker,
    StoredKnowledgeDocument,
    content_hash,
    load_file_documents,
    load_public_website,
)
from aragbiz.knowledge_store import JsonKnowledgeRepository


def test_metadata_hash_and_deduplication(tmp_path):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=40, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
    )
    kb = service.create_knowledge_base("Test KB")
    content = b"Approve the invoice after matching purchase order and goods receipt."
    first = service.ingest_uploaded_file(kb.id, "workflow.txt", content)
    second = service.ingest_uploaded_file(kb.id, "workflow-copy.txt", content)

    assert content_hash(content.decode("utf-8")) == service.list_documents(kb.id)[0].content_hash
    assert first.documents_added == 1
    assert second.documents_skipped == 1
    assert service.get_knowledge_base(kb.id).document_count == 1


def test_document_crud_regenerates_chunks(tmp_path):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=35, chunk_overlap=5),
        embedder=HashEmbeddingModel(dimension=16),
    )
    kb = service.create_knowledge_base("Editable KB")
    document = service.create_document(kb.id, "Policy", "First policy text " * 5)

    assert service.get_knowledge_base(kb.id).document_count == 1
    assert service.list_chunks(kb.id)

    updated = service.update_document(kb.id, document.id, "Policy v2", "Second policy text " * 6)
    assert updated.title == "Policy v2"
    assert service.get_document(kb.id, document.id).content_hash == updated.content_hash

    service.delete_document(kb.id, document.id)
    assert service.get_knowledge_base(kb.id).document_count == 0
    assert service.list_chunks(kb.id) == []


def test_knowledge_base_update_and_delete_cascades(tmp_path):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=40, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
    )
    kb = service.create_knowledge_base("Original", "Draft")
    service.create_document(kb.id, "Policy", "Approval policy text " * 4)

    updated = service.update_knowledge_base_details(kb.id, "Updated", "Published")
    assert updated.name == "Updated"
    assert updated.description == "Published"
    assert updated.document_count == 1

    service.delete_knowledge_base(kb.id)
    assert service.list_knowledge_bases() == []


def test_file_loaders_normalize_common_document_types():
    txt = load_file_documents("guide.txt", b"plain workflow text")
    md = load_file_documents("guide.md", b"# Workflow\nApprove it.")
    json_docs = load_file_documents("guide.json", json.dumps({"title": "JSON guide", "text": "json text"}).encode("utf-8"))
    jsonl_docs = load_file_documents("guide.jsonl", b'{"id":"one","text":"first"}\n{"id":"two","content":"second"}\n')

    assert txt[0].metadata["extension"] == ".txt"
    assert md[0].text.startswith("# Workflow")
    assert json_docs[0].title == "JSON guide"
    assert [document.title for document in jsonl_docs] == ["one", "two"]


def test_custom_ara_file_types_route_to_placeholder_loader():
    documents = load_file_documents("workflow.aratxt", b"custom chunk structure later")

    assert documents[0].metadata["chunking_mode"] == "custom_placeholder"
    assert documents[0].metadata["custom_loader_status"] == "schema_pending"


def test_overlap_chunker_creates_ordered_overlapping_chunks():
    document = StoredKnowledgeDocument(
        id="doc-1",
        knowledge_base_id="kb-1",
        source_id="src-1",
        title="Guide",
        content_hash="hash",
        text="abcdefghij" * 10,
    )
    chunks = OverlapChunker(chunk_size=30, chunk_overlap=5).chunk_stored_document(document)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[0].text[-5:] == chunks[1].text[:5]


def test_website_loader_extracts_readable_html(monkeypatch):
    class FakeResponse:
        headers = {"content-type": "text/html"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def read(self):
            return b"<html><head><title>Workflow</title><script>x</script></head><body><h1>Approve request</h1></body></html>"

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())
    document = load_public_website("https://example.com/workflow")

    assert document.title == "Workflow"
    assert "Approve request" in document.text
    assert "script" not in document.text.lower()
