import os

import pytest

os.environ["ARAGBIZ_USE_TRAINED_CLASSIFIER"] = "false"

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import api.main as api_main
from aragbiz.knowledge import HashEmbeddingModel, KnowledgeService, OverlapChunker, SentenceTransformerEmbeddingModel
from aragbiz.knowledge_store import JsonKnowledgeRepository

app = api_main.app


def test_answer_endpoint_returns_route_metadata():
    client = TestClient(app)
    response = client.post("/answer", json={"question": "Can I start accepting payments while Wix Payments is under verification?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["contexts"]
    assert payload["metadata"]["complexity_label"] in {"simple", "moderate", "complex"}


def test_knowledge_base_endpoints_ingest_upload(monkeypatch, tmp_path):
    pytest.importorskip("multipart")
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=60, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
    )
    monkeypatch.setattr(api_main, "knowledge_service", service)
    client = TestClient(app)

    rejected = client.post(
        "/knowledge-bases",
        json={
            "name": "Unsupported KB",
            "description": "Test",
            "configuration": {
                "embedding_provider": "Cohere",
                "embedding_model": "embed-english-v3.0",
            },
        },
    )
    assert rejected.status_code == 400
    assert "Local" in rejected.json()["detail"]

    rejected_model = client.post(
        "/knowledge-bases",
        json={
            "name": "Unsupported Model KB",
            "description": "Test",
            "configuration": {
                "embedding_provider": "Local",
                "embedding_model": "unsupported-local-model",
            },
        },
    )
    assert rejected_model.status_code == 400
    assert "unsupported-local-model" in rejected_model.json()["detail"]

    created_response = client.post(
        "/knowledge-bases",
        json={
            "name": "Workflow KB",
            "description": "Test",
            "configuration": {
                "chunking_strategy": "fixed_size",
                "chunk_size": 100,
                "chunk_overlap": 20,
                "embedding_provider": "Local",
                "embedding_model": "hash-embedding-384",
            },
        },
    )
    assert created_response.status_code == 200
    created = created_response.json()
    assert created["metadata"]["configuration"]["chunking_strategy"] == "fixed_size"
    assert created["metadata"]["configuration"]["chunk_overlap"] == 0
    response = client.post(
        f"/knowledge-bases/{created['id']}/sources/upload",
        files={"files": ("workflow.txt", ("Approve invoices after matching purchase order and goods receipt. " * 8).encode("utf-8"), "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["documents_added"] == 1

    listed = client.get("/knowledge-bases").json()
    assert listed[0]["document_count"] == 1
    assert listed[0]["chunk_count"] >= 1
    chunks = client.get(f"/knowledge-bases/{created['id']}/chunks").json()
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["has_embedding"] is True
    assert chunks[0]["embedding_dimension"] == 16
    assert chunks[0]["metadata"]["chunk_size"] == 100
    assert chunks[0]["metadata"]["chunk_overlap"] == 0
    assert chunks[0]["metadata"]["embedding_provider"] == "Local"
    assert chunks[0]["metadata"]["embedding_model_requested"] == "hash-embedding-384"
    assert chunks[0]["embedding_model"] == "hash-embedding-384"

    trace = client.get(f"/knowledge-bases/{created['id']}/processing-trace").json()
    assert any(step["step"] == "Chunking" for step in trace)
    assert any(step["step"] == "Embedding" for step in trace)


def test_upload_embedding_runtime_failure_returns_bad_request(monkeypatch, tmp_path):
    pytest.importorskip("multipart")
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=60, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
    )
    monkeypatch.setattr(api_main, "knowledge_service", service)

    def broken_embed(self, texts):
        raise RuntimeError("broken optional dependency")

    monkeypatch.setattr(SentenceTransformerEmbeddingModel, "embed", broken_embed)
    client = TestClient(app)
    created = client.post(
        "/knowledge-bases",
        json={
            "name": "Workflow KB",
            "description": "Test",
            "configuration": {
                "embedding_provider": "Local",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        },
    ).json()

    response = client.post(
        f"/knowledge-bases/{created['id']}/sources/upload",
        files={"files": ("workflow.txt", b"Approve invoices after matching purchase order.", "text/plain")},
    )

    assert response.status_code == 400
    assert "broken optional dependency" in response.json()["detail"]
    assert client.get(f"/knowledge-bases/{created['id']}").json()["status"] == "failed"


def test_knowledge_document_crud_and_answer_selection(monkeypatch, tmp_path):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=60, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
    )
    monkeypatch.setattr(api_main, "knowledge_service", service)
    client = TestClient(app)

    kb = client.post("/knowledge-bases", json={"name": "Selected KB", "description": "Docs"}).json()
    created = client.post(
        f"/knowledge-bases/{kb['id']}/documents",
        json={"title": "Runbook", "text": "Escalate invoice mismatches to finance operations.", "metadata": {"owner": "finance"}},
    )
    assert created.status_code == 200
    document = created.json()
    assert document["title"] == "Runbook"

    updated = client.put(
        f"/knowledge-bases/{kb['id']}/documents/{document['id']}",
        json={"title": "Runbook v2", "text": "Escalate invoice mismatches after goods receipt matching.", "metadata": {"owner": "finance"}},
    )
    assert updated.status_code == 200
    assert updated.json()["title"] == "Runbook v2"

    answer = client.post("/answer", json={"question": "How do I handle invoice mismatch?", "knowledge_base_id": kb["id"]}).json()
    assert answer["metadata"]["knowledge_base_id"] == kb["id"]
    assert answer["metadata"]["knowledge_base_name"] == "Selected KB"

    deleted = client.delete(f"/knowledge-bases/{kb['id']}/documents/{document['id']}")
    assert deleted.status_code == 200
    assert client.get(f"/knowledge-bases/{kb['id']}/documents").json() == []


def test_knowledge_base_update_and_delete_endpoints(monkeypatch, tmp_path):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=60, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
    )
    monkeypatch.setattr(api_main, "knowledge_service", service)
    client = TestClient(app)

    kb = client.post("/knowledge-bases", json={"name": "Original KB", "description": "Draft"}).json()
    client.post(
        f"/knowledge-bases/{kb['id']}/documents",
        json={"title": "Policy", "text": "Policy document text for cascade delete.", "metadata": {}},
    )

    updated = client.put("/knowledge-bases/" + kb["id"], json={"name": "Updated KB", "description": "Published"})
    assert updated.status_code == 200
    assert updated.json()["name"] == "Updated KB"
    assert updated.json()["document_count"] == 1

    rejected = client.put(
        "/knowledge-bases/" + kb["id"],
        json={
            "name": "Updated KB",
            "description": "Configured",
            "configuration": {
                "chunking_strategy": "recursive",
                "chunk_size": 120,
                "chunk_overlap": 30,
                "embedding_provider": "Jina",
                "embedding_model": "jina-embeddings-v3",
            },
        },
    )
    assert rejected.status_code == 400
    assert "Local" in rejected.json()["detail"]

    configured = client.put(
        "/knowledge-bases/" + kb["id"],
        json={
            "name": "Updated KB",
            "description": "Configured",
            "configuration": {
                "chunking_strategy": "recursive",
                "chunk_size": 120,
                "chunk_overlap": 30,
                "embedding_provider": "Local",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        },
    )
    assert configured.status_code == 200
    assert configured.json()["metadata"]["configuration"]["chunking_strategy"] == "recursive"
    assert configured.json()["metadata"]["configuration"]["embedding_provider"] == "Local"
    assert configured.json()["metadata"]["configuration"]["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"

    deleted = client.delete("/knowledge-bases/" + kb["id"])
    assert deleted.status_code == 200
    assert client.get("/knowledge-bases").json() == []
    assert client.get(f"/knowledge-bases/{kb['id']}/documents").status_code == 404
