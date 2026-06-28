import os

import pytest

os.environ["ARAGBIZ_USE_TRAINED_CLASSIFIER"] = "false"

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import api.main as api_main
from aragbiz.knowledge import HashEmbeddingModel, KnowledgeService, OverlapChunker
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

    created = client.post("/knowledge-bases", json={"name": "Workflow KB", "description": "Test"}).json()
    response = client.post(
        f"/knowledge-bases/{created['id']}/sources/upload",
        files={"files": ("workflow.txt", b"Approve invoices after matching purchase order and goods receipt.", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["documents_added"] == 1

    listed = client.get("/knowledge-bases").json()
    assert listed[0]["document_count"] == 1
    assert listed[0]["chunk_count"] >= 1
    assert client.get(f"/knowledge-bases/{created['id']}/chunks").json()[0]["chunk_index"] == 0


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

    deleted = client.delete("/knowledge-bases/" + kb["id"])
    assert deleted.status_code == 200
    assert client.get("/knowledge-bases").json() == []
    assert client.get(f"/knowledge-bases/{kb['id']}/documents").status_code == 404
