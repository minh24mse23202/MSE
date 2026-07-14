import json
import os
import subprocess
from pathlib import Path

import pytest

os.environ["ARAGBIZ_USE_TRAINED_CLASSIFIER"] = "false"

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

import api.main as api_main
from aragbiz.answering import AdaptiveRAGAnswerService
from aragbiz.chat import ChatService, JsonChatRepository
from aragbiz.evaluation import EvaluationService, JsonEvaluationRepository
from aragbiz.knowledge import HashEmbeddingModel, KnowledgeService, OverlapChunker, SentenceTransformerEmbeddingModel
from aragbiz.knowledge_store import JsonKnowledgeRepository
from aragbiz.model_farm import JsonModelFarmRepository, ModelFarmError, ModelFarmService, ModelGateway, ModelStreamEvent
from aragbiz.ragxplain import RagxplainRunner

app = api_main.app


@pytest.fixture(autouse=True)
def isolated_chat_service(monkeypatch, tmp_path):
    service = ChatService(JsonChatRepository(str(tmp_path / "chat.json")))
    monkeypatch.setattr(api_main, "chat_service", service)
    return service


def test_answer_endpoint_returns_direct_route_metadata():
    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "question": "Can I start accepting payments while Wix Payments is under verification?",
            "mode": "direct",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"]
    assert payload["conversation_id"]
    assert payload["contexts"] == []
    assert payload["metadata"]["route_level"] == "l1_direct"
    assert payload["metadata"]["retrieval_used"] is False
    assert payload["metadata"]["trace_steps"]
    messages = client.get(f"/chat/conversations/{payload['conversation_id']}/messages").json()
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["metadata"]["question"] == payload["question"]


def test_answer_endpoint_rejects_adaptive_without_knowledge_base():
    client = TestClient(app)
    response = client.post("/answer", json={"question": "How do I handle invoice mismatch?"})
    assert response.status_code == 400
    assert "Select a knowledge base" in response.json()["detail"]


def test_answer_endpoint_rejects_complex_without_knowledge_base():
    client = TestClient(app)
    response = client.post(
        "/answer",
        json={
            "question": "After payment verification, how should invoice mismatches be handled and who approves follow-up?",
            "mode": "complex_rag",
        },
    )
    assert response.status_code == 400
    assert "L3 Complex RAG" in response.json()["detail"]


def test_chat_conversation_crud_and_library_sections():
    client = TestClient(app)
    created = client.post("/chat/conversations", json={"title": "Invoice approval workflow"}).json()
    assert created["title"] == "Invoice approval workflow"
    assert created["pinned"] is False

    recents = client.get("/chat/conversations?section=recents").json()
    assert [item["id"] for item in recents] == [created["id"]]
    assert client.get("/chat/conversations?query=invoice").json()[0]["id"] == created["id"]

    pinned = client.patch(f"/chat/conversations/{created['id']}", json={"pinned": True}).json()
    assert pinned["pinned"] is True
    assert client.get("/chat/conversations?section=recents").json() == []
    assert client.get("/chat/conversations?section=library").json()[0]["id"] == created["id"]

    deleted = client.delete(f"/chat/conversations/{created['id']}")
    assert deleted.status_code == 200
    assert client.get("/chat/conversations").json() == []


def test_chat_configuration_crud_and_answer_snapshot():
    client = TestClient(app)
    defaults = client.get("/chat/configurations").json()
    assert defaults
    assert defaults[0]["generator_model"] == "extractive"

    created_response = client.post(
        "/chat/configurations",
        json={
            "name": "Formal finance assistant",
            "description": "Finance workflow tone",
            "generator_provider": "Local",
            "generator_model": "extractive",
            "response_structure": "Step-by-step workflow guidance",
            "tone": "Formal",
            "humor_level": 1,
            "system_prompt": "Answer as a finance workflow assistant.",
            "predefined_prompt": "Use numbered steps.",
        },
    )
    assert created_response.status_code == 200
    created = created_response.json()
    assert created["name"] == "Formal finance assistant"

    updated = client.patch(
        f"/chat/configurations/{created['id']}",
        json={"tone": "Technical", "humor_level": 0},
    ).json()
    assert updated["tone"] == "Technical"
    assert updated["humor_level"] == 0

    answer_response = client.post(
        "/answer",
        json={
            "question": "Can I start accepting payments while Wix Payments is under verification?",
            "mode": "direct",
            "chat_configuration_id": created["id"],
        },
    )
    assert answer_response.status_code == 200
    answer = answer_response.json()
    assert answer["metadata"]["chat_configuration_id"] == created["id"]
    assert answer["metadata"]["chat_configuration"]["tone"] == "Technical"
    assert answer["metadata"]["configured_generator"]["model"] == "extractive"
    assert answer["metadata"]["actual_generator"] == {"provider": "Local", "model": "extractive"}
    assert answer["metadata"]["generation_status"] == "completed"
    assert any(step["step"] == "Prompt builder" for step in answer["metadata"]["trace_steps"])
    assert any(step["step"] == "Generator execution" for step in answer["metadata"]["trace_steps"])

    conversation = client.get(f"/chat/conversations/{answer['conversation_id']}").json()
    assert conversation["chat_configuration_id"] == created["id"]
    assert conversation["metadata"]["chat_configuration"]["generator_model"] == "extractive"

    unsupported = client.post(
        "/chat/configurations",
        json={
            "name": "OpenAI draft",
            "generator_provider": "OpenAI",
            "generator_model": "gpt-4.1-mini",
        },
    ).json()
    unsupported_answer = client.post(
        "/answer",
        json={
            "question": "Can I start accepting payments while Wix Payments is under verification?",
            "mode": "direct",
            "chat_configuration_id": unsupported["id"],
        },
    )
    assert unsupported_answer.status_code == 400
    assert "not implemented" in unsupported_answer.json()["detail"]

    deleted = client.delete(f"/chat/configurations/{created['id']}")
    assert deleted.status_code == 200
    missing = client.post(
        "/answer",
        json={
            "question": "Can I start accepting payments while Wix Payments is under verification?",
            "mode": "direct",
            "chat_configuration_id": created["id"],
        },
    )
    assert missing.status_code == 404

def test_answer_appends_to_existing_conversation():
    client = TestClient(app)
    conversation = client.post("/chat/conversations", json={"title": "Existing chat"}).json()
    response = client.post(
        "/answer",
        json={
            "conversation_id": conversation["id"],
            "question": "Can I start accepting payments while Wix Payments is under verification?",
            "mode": "direct",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["conversation_id"] == conversation["id"]
    messages = client.get(f"/chat/conversations/{conversation['id']}/messages").json()
    assert len(messages) == 2
    assert messages[0]["content"].startswith("Can I start accepting")


def test_answer_stream_emits_deltas_and_persists_completed_messages():
    client = TestClient(app)

    response = client.post(
        "/answer/stream",
        json={
            "question": "Can I start accepting payments while Wix Payments is under verification?",
            "mode": "direct",
        },
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert [event["type"] for event in events][0] == "started"
    assert any(event["type"] == "trace" for event in events)
    assert any(event["type"] == "delta" for event in events)
    assert events[-1]["type"] == "completed"
    started = next(event["data"] for event in events if event["type"] == "started" and event["data"].get("assistant_message_id"))
    completed = events[-1]["data"]
    assert completed["conversation_id"] == started["conversation_id"]
    assert completed["metadata"]["assistant_message_id"] == started["assistant_message_id"]

    messages = client.get(f"/chat/conversations/{completed['conversation_id']}/messages").json()
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert [message["status"] for message in messages] == ["completed", "completed"]
    assert messages[1]["id"] == started["assistant_message_id"]
    assert messages[1]["content"] == completed["answer"]
    assert messages[1]["request_id"] == started["request_id"]


def test_answer_stream_persists_failed_partial_message(monkeypatch):
    async def broken_stream(*args, **kwargs):
        yield ModelStreamEvent("delta", {"text": "partial answer"})
        raise ModelFarmError("provider stopped")

    monkeypatch.setattr(api_main.model_gateway, "stream", broken_stream)
    client = TestClient(app)

    response = client.post(
        "/answer/stream",
        json={
            "question": "Can I start accepting payments while Wix Payments is under verification?",
            "mode": "direct",
        },
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert any(event["type"] == "delta" for event in events)
    assert events[-1]["type"] == "error"
    assert "provider stopped" in events[-1]["data"]["detail"]
    started = next(event["data"] for event in events if event["type"] == "started" and event["data"].get("assistant_message_id"))
    messages = client.get(f"/chat/conversations/{started['conversation_id']}/messages").json()
    assert messages[1]["status"] == "failed"
    assert messages[1]["content"] == "partial answer"
    assert "provider stopped" in messages[1]["metadata"]["error"]


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


def test_upload_with_model_gateway_embedding_runs_outside_active_event_loop(monkeypatch, tmp_path):
    pytest.importorskip("multipart")
    model_farm_service = ModelFarmService(JsonModelFarmRepository(str(tmp_path / "models.json")))
    model_gateway = ModelGateway(model_farm_service)
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=60, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
        model_farm_service=model_farm_service,
        model_gateway=model_gateway,
    )
    monkeypatch.setattr(api_main, "knowledge_service", service)
    client = TestClient(app)
    created = client.post(
        "/knowledge-bases",
        json={
            "name": "Model Farm KB",
            "description": "Test",
            "configuration": {
                "embedding_deployment_id": "model-local-hash-384",
                "embedding_provider": "Local",
                "embedding_model": "hash-embedding-384",
            },
        },
    ).json()

    response = client.post(
        f"/knowledge-bases/{created['id']}/sources/upload",
        files={"files": ("workflow.txt", b"Approve invoices after matching purchase order.", "text/plain")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["documents_added"] == 1
    chunks = client.get(f"/knowledge-bases/{created['id']}/chunks").json()
    assert chunks[0]["embedding_model"] == "hash-embedding-384"


def test_model_farm_template_api_creates_disabled_deployment(monkeypatch, tmp_path):
    model_farm_service = ModelFarmService(JsonModelFarmRepository(str(tmp_path / "models.json")))
    model_gateway = ModelGateway(model_farm_service)
    monkeypatch.setattr(api_main, "model_farm_service", model_farm_service)
    monkeypatch.setattr(api_main, "model_gateway", model_gateway)
    monkeypatch.setattr(api_main, "_require_user", lambda authorization: None)
    monkeypatch.setattr(api_main, "_require_admin", lambda authorization: None)
    client = TestClient(app)

    providers = client.get("/model-farm/providers")
    assert providers.status_code == 200
    assert {item["id"] for item in providers.json()} >= {"openai-generation", "openai-compatible"}

    created = client.post(
        "/model-farm/deployments/from-template",
        json={
            "template_id": "openai-generation",
            "name": "OpenAI GPT deployment",
            "credential_env_refs": {"api_key": "ARAGBIZ_MODEL_OPENAI_API_KEY"},
            "capabilities": ["generation", "judge"],
        },
    )
    assert created.status_code == 200
    payload = created.json()
    assert payload["enabled"] is False
    assert payload["health_status"] == "untested"
    assert payload["metadata"]["template_id"] == "openai-generation"

    enable = client.patch(f"/model-farm/deployments/{payload['id']}", json={"enabled": True})
    assert enable.status_code == 400
    assert "Test a remote deployment" in enable.json()["detail"]


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

    answer_response = client.post(
        "/answer",
        json={
            "question": "How do I handle invoice mismatch?",
            "knowledge_base_id": kb["id"],
            "mode": "simple_rag",
            "retrieval_mode": "bm25",
            "top_k": 2,
        },
    )
    assert answer_response.status_code == 200
    answer = answer_response.json()
    assert answer["metadata"]["knowledge_base_id"] == kb["id"]
    assert answer["metadata"]["knowledge_base_name"] == "Selected KB"
    assert answer["metadata"]["route_level"] == "l2_simple_rag"
    assert answer["metadata"]["retrieval_mode"] == "bm25"
    assert answer["contexts"]

    complex_question = "After invoice mismatch detection, what follow-up steps are needed and who owns approvals?"
    complex_response = client.post(
        "/answer",
        json={
            "question": complex_question,
            "knowledge_base_id": kb["id"],
            "mode": "complex_rag",
            "retrieval_mode": "bm25",
            "top_k": 2,
        },
    )
    assert complex_response.status_code == 200
    complex_answer = complex_response.json()
    assert complex_answer["metadata"]["route_level"] == "l3_complex_rag"
    assert complex_answer["metadata"]["multi_step"] is True
    assert complex_answer["metadata"]["decomposed_queries"][-1] == complex_question
    assert complex_answer["metadata"]["retrieval_steps"]
    assert complex_answer["metadata"]["aggregation_summary"]["selected_context_count"] == len(complex_answer["contexts"])
    assert complex_answer["contexts"]
    assert complex_answer["contexts"][0]["metadata"]["source_subquery"]
    assert complex_answer["contexts"][0]["metadata"]["aggregated_rank"] == 1

    deleted = client.delete(f"/knowledge-bases/{kb['id']}/documents/{document['id']}")
    assert deleted.status_code == 200
    assert client.get(f"/knowledge-bases/{kb['id']}/documents").json() == []


def test_evaluation_run_endpoints(monkeypatch, tmp_path):
    service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=60, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
    )
    monkeypatch.setattr(api_main, "knowledge_service", service)
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"id":"q1","question":"After goods receipt, how should invoice mismatches be escalated and who approves follow-up?","answer":"Escalate invoice mismatches to finance operations.","context":"Invoice mismatch","complexity_label":"complex","metadata":{}}\n',
        encoding="utf-8",
    )
    answer_service = AdaptiveRAGAnswerService(
        router=api_main.pipeline.router,
        generator=api_main.pipeline.generator,
        knowledge_service=service,
        bm25_weight=api_main.config.bm25_weight,
        dense_weight=api_main.config.dense_weight,
    )
    ragxplain_root = tmp_path / "ragxplain"
    (ragxplain_root / "ragxplain").mkdir(parents=True)
    (ragxplain_root / "ragxplain" / "cli.py").write_text("", encoding="utf-8")
    (ragxplain_root / "viewer").mkdir()
    (ragxplain_root / "viewer" / "insights-viewer.html").write_text("<html>RAGXplain viewer</html>", encoding="utf-8")

    def successful_process(command, **kwargs):
        output_dir = Path(command[command.index("--out") + 1])
        (output_dir / "results.csv").write_text("question,candidate_answer\nq,a\n", encoding="utf-8")
        (output_dir / "metrics_insights.json").write_text("{}", encoding="utf-8")
        (output_dir / "overall_insights.json").write_text(
            json.dumps({"analysis": {"executive_summary": "Evaluation summary", "insights": []}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")

    ragxplain_runner = RagxplainRunner(
        str(ragxplain_root),
        str(tmp_path / "results"),
        "examples.mock_judge_impl:judge",
        process_runner=successful_process,
    )
    monkeypatch.setattr(
        api_main,
        "evaluation_service",
        EvaluationService(
            JsonEvaluationRepository(str(tmp_path / "evaluation.json")),
            answer_service,
            str(dataset_path),
            ragxplain_runner=ragxplain_runner,
        ),
    )
    client = TestClient(app)

    missing_kb = client.post("/evaluation/runs", json={"knowledge_base_id": ""})
    assert missing_kb.status_code == 400
    assert "knowledge base" in missing_kb.json()["detail"].lower()

    kb = client.post("/knowledge-bases", json={"name": "Eval KB", "description": "Docs"}).json()
    client.post(
        f"/knowledge-bases/{kb['id']}/documents",
        json={
            "title": "Invoice workflow",
            "text": "Invoice mismatches after goods receipt should be escalated to finance operations for approval.",
            "metadata": {},
        },
    )

    created = client.post(
        "/evaluation/runs",
        json={
            "knowledge_base_id": kb["id"],
            "retrieval_mode": "bm25",
            "top_k": 2,
            "limit": 1,
            "compare_baseline": True,
        },
    )
    assert created.status_code == 200
    run = created.json()
    assert run["status"] == "completed"
    assert run["metadata"]["ragxplain"]["status"] == "not_requested"
    assert run["metrics"]["average_retrieved_contexts"] >= 1
    assert run["baseline_metrics"]["average_retrieved_contexts"] >= 1

    listed = client.get("/evaluation/runs").json()
    assert listed[0]["id"] == run["id"]
    cases = client.get(f"/evaluation/runs/{run['id']}/cases").json()
    assert len(cases) == 1
    assert cases[0]["adaptive_answer"]
    assert cases[0]["adaptive_contexts"]
    assert cases[0]["adaptive_metadata"]["trace_steps"]
    assert client.get(f"/evaluation/runs/{run['id']}/ragxplain/overall-insights").status_code == 409

    ragxplain_created = client.post(
        "/evaluation/runs",
        json={
            "knowledge_base_id": kb["id"],
            "retrieval_mode": "bm25",
            "top_k": 2,
            "limit": 1,
            "compare_baseline": False,
            "run_ragxplain": True,
        },
    )
    assert ragxplain_created.status_code == 200
    ragxplain_run = ragxplain_created.json()
    assert ragxplain_run["metadata"]["ragxplain"]["status"] == "completed"
    insights = client.get(f"/evaluation/runs/{ragxplain_run['id']}/ragxplain/overall-insights")
    assert insights.status_code == 200
    assert insights.json()["analysis"]["executive_summary"] == "Evaluation summary"
    viewer = client.get("/evaluation/ragxplain/viewer")
    assert viewer.status_code == 200
    assert "RAGXplain viewer" in viewer.text
    artifact_dir = Path(ragxplain_run["metadata"]["ragxplain"]["output_dir"])
    assert artifact_dir.exists()

    missing = client.get("/evaluation/runs/eval-missing")
    assert missing.status_code == 404
    assert client.get("/evaluation/runs/eval-missing/ragxplain/overall-insights").status_code == 404

    ragxplain_deleted = client.delete(f"/evaluation/runs/{ragxplain_run['id']}")
    assert ragxplain_deleted.status_code == 200
    assert not artifact_dir.exists()

    deleted = client.delete(f"/evaluation/runs/{run['id']}")
    assert deleted.status_code == 200
    assert client.get(f"/evaluation/runs/{run['id']}").status_code == 404


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


def _sse_events(body: str):
    events = []
    for part in body.strip().split("\n\n"):
        if not part.strip():
            continue
        event_type = "message"
        data_lines = []
        for line in part.splitlines():
            if line.startswith("event:"):
                event_type = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].strip())
        if data_lines:
            events.append({"type": event_type, "data": json.loads("\n".join(data_lines))})
    return events
