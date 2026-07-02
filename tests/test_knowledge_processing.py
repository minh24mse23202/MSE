import json
import builtins

import pytest

from aragbiz.knowledge import (
    HashEmbeddingModel,
    KnowledgeProcessingError,
    KnowledgeService,
    OverlapChunker,
    SentenceTransformerEmbeddingModel,
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
    chunks = service.list_chunks(kb.id)
    assert chunks
    assert chunks[0].has_embedding is True
    assert chunks[0].embedding_model == "hash-embedding-384"
    assert chunks[0].embedding_dimension == 16
    trace = service.processing_trace(kb.id)
    assert [step.step for step in trace] == [
        "Knowledge base selected",
        "Data source loading",
        "Metadata extraction and deduplication",
        "Chunking",
        "Embedding",
        "Storage",
    ]

    updated = service.update_document(kb.id, document.id, "Policy v2", "Second policy text " * 6)
    assert updated.title == "Policy v2"
    assert service.get_document(kb.id, document.id).content_hash == updated.content_hash

    service.delete_document(kb.id, document.id)
    assert service.get_knowledge_base(kb.id).document_count == 0
    assert service.list_chunks(kb.id) == []


def test_knowledge_base_configuration_controls_chunking(tmp_path):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=800, chunk_overlap=120),
        embedder=HashEmbeddingModel(dimension=16),
    )
    kb = service.create_knowledge_base(
        "Configured KB",
        configuration={
            "chunking_strategy": "fixed_size",
            "chunk_size": 100,
            "chunk_overlap": 40,
            "embedding_provider": "Local",
            "embedding_model": "hash-embedding-384",
        },
    )
    service.ingest_uploaded_file(kb.id, "workflow.txt", ("approval workflow " * 40).encode("utf-8"))

    refreshed = service.get_knowledge_base(kb.id)
    assert refreshed.metadata["configuration"]["chunking_strategy"] == "fixed_size"
    assert refreshed.metadata["configuration"]["chunk_overlap"] == 0
    chunks = service.list_chunks(kb.id)
    assert len(chunks) > 1
    assert chunks[0].metadata["chunk_size"] == 100
    assert chunks[0].metadata["chunk_overlap"] == 0
    assert chunks[0].metadata["embedding_provider"] == "Local"
    assert chunks[0].metadata["embedding_model_requested"] == "hash-embedding-384"
    assert chunks[0].embedding_model == "hash-embedding-384"


def test_sentence_transformer_configuration_resolves_runtime_embedder(tmp_path):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=800, chunk_overlap=120),
        embedder=HashEmbeddingModel(dimension=16),
    )
    kb = service.create_knowledge_base(
        "Transformer KB",
        configuration={
            "embedding_provider": "Local",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    )

    embedder = service._embedder_for_configuration(kb.metadata["configuration"])

    assert isinstance(embedder, SentenceTransformerEmbeddingModel)
    assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert embedder.dimension == 16


def test_sentence_transformer_runtime_errors_are_processing_errors(monkeypatch):
    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name == "transformers":
            raise TypeError("broken optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    embedder = SentenceTransformerEmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", dimension=16)

    with pytest.raises(KnowledgeProcessingError, match="Unable to initialize transformer embedding model"):
        embedder.embed(["workflow text"])


def test_unsupported_embedding_configuration_is_rejected(tmp_path):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=800, chunk_overlap=120),
        embedder=HashEmbeddingModel(dimension=16),
    )

    with pytest.raises(KnowledgeProcessingError, match="Local"):
        service.create_knowledge_base(
            "Unsupported KB",
            configuration={
                "embedding_provider": "Cohere",
                "embedding_model": "embed-english-v3.0",
            },
        )


def test_existing_unsupported_knowledge_base_fails_embedding_paths(tmp_path):
    repository = JsonKnowledgeRepository(str(tmp_path / "knowledge.json"))
    service = KnowledgeService(
        repository=repository,
        chunker=OverlapChunker(chunk_size=800, chunk_overlap=120),
        embedder=HashEmbeddingModel(dimension=16),
    )
    repository.initialize()
    kb = repository.create_knowledge_base(
        "Legacy Unsupported KB",
        metadata={
            "configuration": {
                "embedding_provider": "Cohere",
                "embedding_model": "embed-english-v3.0",
            },
        },
    )

    with pytest.raises(KnowledgeProcessingError, match="Modify the knowledge base to use Local"):
        service.ingest_uploaded_file(kb.id, "workflow.txt", b"approval workflow")
    with pytest.raises(KnowledgeProcessingError, match="Modify the knowledge base to use Local"):
        service.reindex(kb.id)


def test_reindex_uses_updated_knowledge_base_embedding_configuration(tmp_path, monkeypatch):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=60, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
    )
    kb = service.create_knowledge_base("Reindex KB")
    service.create_document(kb.id, "Policy", "Approval policy text " * 10)
    assert service.list_chunks(kb.id)[0].embedding_model == "hash-embedding-384"

    def fake_embed(self, texts):
        return [[0.0] * self.dimension for _ in texts]

    monkeypatch.setattr(SentenceTransformerEmbeddingModel, "embed", fake_embed)
    service.update_knowledge_base_details(
        kb.id,
        "Reindex KB",
        configuration={
            "embedding_provider": "Local",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    )
    service.reindex(kb.id)

    chunks = service.list_chunks(kb.id)
    assert chunks[0].embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert chunks[0].metadata["embedding_model_requested"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert service.get_knowledge_base(kb.id).embedding_model == "sentence-transformers/all-MiniLM-L6-v2"


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
