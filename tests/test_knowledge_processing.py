import json
import builtins

import pytest

import aragbiz.knowledge as knowledge_module
from aragbiz.knowledge import (
    HashEmbeddingModel,
    KnowledgeProcessingError,
    KnowledgeService,
    OverlapChunker,
    SentenceTransformerEmbeddingModel,
    StructureAwareRecursiveChunker,
    StoredKnowledgeDocument,
    WIXQA_MINILM_CHUNKING_OPTIONS,
    _embedding_input_for_chunk,
    content_hash,
    load_file_documents,
    load_prepared_wixqa_corpus,
    load_public_website,
    normalize_knowledge_base_configuration,
    validate_knowledge_base_configuration,
)
from aragbiz.knowledge_store import JsonKnowledgeRepository


class _FakeWordPieceTokenizer:
    def __init__(self):
        self.tokens = {}
        self.reverse = {}

    def encode(self, text, add_special_tokens=False):
        import re

        ids = []
        for token in re.findall(r"\w+|[^\w\s]", text.lower()):
            if token not in self.tokens:
                token_id = len(self.tokens) + 10
                self.tokens[token] = token_id
                self.reverse[token_id] = token
            ids.append(self.tokens[token])
        return ([1, *ids, 2] if add_special_tokens else ids)

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return " ".join(self.reverse[token_id] for token_id in token_ids if token_id in self.reverse)

    def num_special_tokens_to_add(self, pair=False):
        return 2


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


def test_wixqa_minilm_profile_normalizes_nested_token_settings():
    configuration = normalize_knowledge_base_configuration(
        {
            "chunking_strategy": "structure_aware_recursive",
            "chunking_profile_id": "wixqa_minilm_structure_v1",
            "embedding_provider": "Local",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_options": {"hard_max_wordpieces": 240},
            "chunking_options": WIXQA_MINILM_CHUNKING_OPTIONS,
        }
    )

    assert configuration["chunking_profile_id"] == "wixqa_minilm_structure_v1"
    assert configuration["embedding_options"]["hard_max_wordpieces"] == 240
    assert configuration["chunking_options"]["target_body_tokens"] == 180
    assert configuration["chunking_options"]["soft_max_body_tokens"] == 210
    assert configuration["chunking_options"]["separators"][-1] == "token"


def test_wixqa_minilm_profile_resolves_server_defaults_and_article_isolation():
    configuration = normalize_knowledge_base_configuration(
        {
            "chunking_profile_id": "wixqa_minilm_structure_v1",
            "chunking_options": {
                "rules": {"never_merge_across_articles": False},
            },
        }
    )

    assert configuration["chunking_strategy"] == "structure_aware_recursive"
    assert configuration["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert configuration["embedding_deployment_id"] == "model-local-minilm-384"
    assert configuration["embedding_options"]["hard_max_wordpieces"] == 240
    assert configuration["chunking_options"]["rules"]["never_merge_across_articles"] is True


def test_unknown_chunking_profile_is_rejected():
    with pytest.raises(KnowledgeProcessingError, match="Unknown chunking profile"):
        validate_knowledge_base_configuration(
            {
                "chunking_profile_id": "unregistered-profile",
                "embedding_model": "hash-embedding-384",
            }
        )


def test_structure_aware_chunker_enforces_hard_limit_and_embedding_only_prefix():
    tokenizer = _FakeWordPieceTokenizer()
    options = {
        **WIXQA_MINILM_CHUNKING_OPTIONS,
        "target_body_tokens": 8,
        "soft_max_body_tokens": 10,
        "minimum_chunk_tokens": 3,
        "overlap_tokens": 2,
        "metadata_prefix": {
            **WIXQA_MINILM_CHUNKING_OPTIONS["metadata_prefix"],
            "maximum_tokens": 4,
        },
    }
    document = StoredKnowledgeDocument(
        id="doc-wix",
        knowledge_base_id="kb-wix",
        source_id="source-wix",
        title="Invoice approval",
        content_hash="hash",
        text=(
            "Invoice approval\n"
            "# Required checks\n"
            "1. Match the purchase order.\n"
            "2. Match the goods receipt.\n\n"
            "Confirm the supplier and invoice amount before approval.\n"
            "## Exceptions\n"
            "Escalate mismatches to the workflow owner before posting."
        ),
        metadata={
            "source_record_id": "article-123",
            "source_type": "wixqa_corpus",
            "article_type": "article",
            "embedding_model_requested": "sentence-transformers/all-MiniLM-L6-v2",
        },
    )
    chunker = StructureAwareRecursiveChunker(
        tokenizer,
        options=options,
        embedding_options={"hard_max_wordpieces": 16},
        profile_id="wixqa_minilm_structure_v1",
    )

    chunks = chunker.chunk_stored_document(document)

    assert len(chunks) >= 2
    assert all(chunk.metadata["article_id"] == "article-123" for chunk in chunks)
    assert all(chunk.metadata["parent_document_id"] == "article-123" for chunk in chunks)
    assert all(chunk.metadata["embedding_wordpiece_count"] <= 16 for chunk in chunks)
    assert all(not chunk.text.startswith("Title:") for chunk in chunks)
    assert all(chunk.metadata["embedding_prefix"] for chunk in chunks)
    assert all(_embedding_input_for_chunk(
        chunk,
        {"chunking_strategy": "structure_aware_recursive"},
    ).startswith(chunk.metadata["embedding_prefix"]) for chunk in chunks)
    assert all(chunk.token_count == chunk.metadata["body_wordpiece_count"] for chunk in chunks)
    exception_chunks = [
        chunk for chunk in chunks if chunk.metadata["heading_path"] == ["Required checks", "Exceptions"]
    ]
    assert exception_chunks
    assert exception_chunks[0].metadata["overlap_wordpiece_count"] == 0


def test_structure_aware_chunker_preserves_numbered_list_when_it_fits():
    tokenizer = _FakeWordPieceTokenizer()
    options = {
        **WIXQA_MINILM_CHUNKING_OPTIONS,
        "target_body_tokens": 20,
        "soft_max_body_tokens": 24,
        "minimum_chunk_tokens": 2,
        "overlap_tokens": 0,
        "metadata_prefix": {
            **WIXQA_MINILM_CHUNKING_OPTIONS["metadata_prefix"],
            "maximum_tokens": 2,
        },
    }
    document = StoredKnowledgeDocument(
        id="doc-list",
        knowledge_base_id="kb-wix",
        source_id="source-wix",
        title="Approval",
        content_hash="hash",
        text="Approval\nSteps:\n1. Review request\n2. Approve request\n3. Notify owner",
        metadata={"source_record_id": "article-list"},
    )

    chunks = StructureAwareRecursiveChunker(
        tokenizer,
        options=options,
        embedding_options={"hard_max_wordpieces": 32},
        profile_id="wixqa_minilm_structure_v1",
    ).chunk_stored_document(document)

    assert any(
        "1." in chunk.text and "2." in chunk.text and "3." in chunk.text
        for chunk in chunks
    )


def test_structure_aware_profile_rejects_non_minilm_embedding(tmp_path):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=800, chunk_overlap=120),
        embedder=HashEmbeddingModel(dimension=16),
    )

    with pytest.raises(KnowledgeProcessingError, match="requires 'sentence-transformers/all-MiniLM-L6-v2'"):
        service.create_knowledge_base(
            "Invalid structure profile",
            configuration={
                "chunking_strategy": "structure_aware_recursive",
                "embedding_provider": "Local",
                "embedding_model": "hash-embedding-384",
            },
        )


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
        if name == "torch":
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


def test_query_embedding_uses_active_index_configuration_while_reindex_is_pending(tmp_path, monkeypatch):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=80, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
    )
    kb = service.create_knowledge_base("Active configuration")
    service.ingest_uploaded_file(kb.id, "workflow.txt", b"Approve the request before publishing.")
    active_version = next(
        version for version in service.list_index_versions(kb.id) if version.status == "active"
    )
    service.update_knowledge_base_details(
        kb.id,
        kb.name,
        configuration={
            "chunking_strategy": "structure_aware_recursive",
            "chunking_profile_id": "wixqa_minilm_structure_v1",
            "embedding_provider": "Local",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_options": {"hard_max_wordpieces": 240},
            "chunking_options": WIXQA_MINILM_CHUNKING_OPTIONS,
        },
    )
    captured = {}

    class CapturingEmbedder:
        model_name = "hash-embedding-384"
        dimension = 16

        def embed(self, texts):
            captured["texts"] = texts
            return [[0.0] * 16 for _ in texts]

    def capture_embedder(configuration):
        captured["configuration"] = configuration
        return CapturingEmbedder()

    monkeypatch.setattr(service, "_embedder_for_configuration", capture_embedder)
    service.embed_query(kb.id, "What happens next?")

    assert captured["configuration"]["chunking_strategy"] == "sliding_window_overlap"
    assert captured["texts"] == ["What happens next?"]
    assert next(
        version for version in service.list_index_versions(kb.id) if version.status == "active"
    ).id == active_version.id


def test_failed_structure_aware_reindex_keeps_previous_index_active(tmp_path, monkeypatch):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=80, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
    )
    kb = service.create_knowledge_base("Atomic index")
    service.ingest_uploaded_file(kb.id, "workflow.txt", b"Approve the request before publishing.")
    active_version_id = next(
        version.id for version in service.list_index_versions(kb.id) if version.status == "active"
    )
    service.update_knowledge_base_details(
        kb.id,
        kb.name,
        configuration={
            "chunking_strategy": "structure_aware_recursive",
            "chunking_profile_id": "wixqa_minilm_structure_v1",
            "embedding_provider": "Local",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_options": {"hard_max_wordpieces": 240},
            "chunking_options": WIXQA_MINILM_CHUNKING_OPTIONS,
        },
    )
    monkeypatch.setattr(
        knowledge_module,
        "_load_wordpiece_tokenizer",
        lambda model_name: (_ for _ in ()).throw(KnowledgeProcessingError("tokenizer unavailable")),
    )

    with pytest.raises(KnowledgeProcessingError, match="tokenizer unavailable"):
        service.reindex(kb.id)

    versions = service.list_index_versions(kb.id)
    assert next(version.id for version in versions if version.status == "active") == active_version_id


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
    assert [document.metadata["source_record_id"] for document in jsonl_docs] == ["one", "two"]


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


def test_prepared_wixqa_loader_validates_and_maps_records(tmp_path):
    corpus_path = tmp_path / "wixqa.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "article-1",
                        "text": "Invoice approval\nMatch the purchase order first.",
                        "metadata": {
                            "source": "Wix/WixQA",
                            "url": "https://support.wix.com/article-1",
                            "article_type": "article",
                        },
                    }
                ),
                json.dumps(
                    {
                        "id": "article-2",
                        "text": "Refund workflow\nValidate the payment before refunding.",
                        "metadata": {
                            "source": "Wix/WixQA",
                            "url": "https://support.wix.com/article-2",
                            "article_type": "article",
                        },
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    documents, checksum = load_prepared_wixqa_corpus(corpus_path, expected_documents=2)

    assert [document.title for document in documents] == ["Invoice approval", "Refund workflow"]
    assert documents[0].source_type == "wixqa_corpus"
    assert documents[0].metadata["source_record_id"] == "article-1"
    assert documents[0].metadata["article_type"] == "article"
    assert len(checksum) == 64


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"id": "one", "text": "Text", "metadata": {}}], "exactly 2"),
        (
            [
                {"id": "same", "text": "First", "metadata": {}},
                {"id": "same", "text": "Second", "metadata": {}},
            ],
            "duplicate record id",
        ),
        (
            [
                {"id": "one", "text": "Same", "metadata": {}},
                {"id": "two", "text": "Same", "metadata": {}},
            ],
            "duplicate content",
        ),
    ],
)
def test_prepared_wixqa_loader_rejects_invalid_corpus(tmp_path, rows, message):
    corpus_path = tmp_path / "wixqa.jsonl"
    corpus_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    with pytest.raises(KnowledgeProcessingError, match=message):
        load_prepared_wixqa_corpus(corpus_path, expected_documents=2)


def test_prepared_wixqa_import_is_idempotent_and_supports_paging(tmp_path):
    corpus_path = tmp_path / "wixqa.jsonl"
    corpus_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "invoice",
                        "text": "Invoice approval\n" + ("Match invoice and purchase order. " * 5),
                        "metadata": {"url": "https://support.wix.com/invoice", "article_type": "article"},
                    }
                ),
                json.dumps(
                    {
                        "id": "refund",
                        "text": "Refund workflow\n" + ("Validate payment and customer request. " * 5),
                        "metadata": {"url": "https://support.wix.com/refund", "article_type": "article"},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=80, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
        prepared_corpus_path=str(corpus_path),
        prepared_corpus_expected_documents=2,
        embedding_batch_size=2,
    )
    kb = service.create_knowledge_base(
        "WixQA KB",
        configuration={
            "chunk_size": 80,
            "chunk_overlap": 10,
            "embedding_provider": "Local",
            "embedding_model": "hash-embedding-384",
        },
    )

    catalog = service.prepared_source_catalog()
    first = service.ingest_prepared_wixqa_corpus(
        kb.id,
        expected_checksum=catalog["sha256"],
        document_limit=1,
    )
    second = service.ingest_prepared_wixqa_corpus(
        kb.id,
        expected_checksum=catalog["sha256"],
        document_limit=2,
    )
    third = service.ingest_prepared_wixqa_corpus(
        kb.id,
        expected_checksum=catalog["sha256"],
        document_limit=2,
    )
    invoice_page, invoice_total = service.list_documents_page(kb.id, query="invoice", limit=25)

    assert catalog["available"] is True
    assert catalog["document_count"] == 2
    assert first.documents_added == 1
    assert first.documents_skipped == 0
    assert second.documents_added == 1
    assert second.documents_skipped == 1
    assert third.documents_added == 0
    assert third.documents_skipped == 2
    assert service.get_knowledge_base(kb.id).document_count == 2
    assert service.list_wixqa_source_record_ids(kb.id) == ["invoice", "refund"]
    assert invoice_total == 1
    assert invoice_page[0].metadata["source_record_id"] == "invoice"
    chunks = service.list_document_chunks(kb.id, invoice_page[0].id)
    assert chunks
    assert all(chunk.document_id == invoice_page[0].id for chunk in chunks)
    assert service.get_knowledge_base(kb.id).metadata["prepared_corpus"]["status"] == "completed"
    assert service.get_knowledge_base(kb.id).metadata["prepared_corpus"]["selected_document_count"] == 2


def test_prepared_wixqa_import_rejects_changed_checksum(tmp_path):
    corpus_path = tmp_path / "wixqa.jsonl"
    corpus_path.write_text(
        "\n".join(
            json.dumps({"id": f"article-{index}", "text": f"Content {index}", "metadata": {}})
            for index in range(2)
        ),
        encoding="utf-8",
    )
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=80, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
        prepared_corpus_path=str(corpus_path),
        prepared_corpus_expected_documents=2,
    )
    kb = service.create_knowledge_base("WixQA KB")

    with pytest.raises(KnowledgeProcessingError, match="changed after the import was queued"):
        service.ingest_prepared_wixqa_corpus(kb.id, expected_checksum="not-the-current-checksum")
