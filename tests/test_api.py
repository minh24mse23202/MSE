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
from aragbiz.evaluation import EvaluationRunRecord, EvaluationService, JsonEvaluationRepository
from aragbiz.evaluation_experiments import EvaluationExperimentRecord
from aragbiz.jobs import BackgroundJob
from aragbiz.knowledge import HashEmbeddingModel, KnowledgeService, OverlapChunker, SentenceTransformerEmbeddingModel
from aragbiz.knowledge_store import JsonKnowledgeRepository
from aragbiz.model_farm import JsonModelFarmRepository, ModelFarmError, ModelFarmService, ModelGateway, ModelStreamEvent
from aragbiz.ragxplain import RagxplainRunner
from aragbiz.tracing import FileTraceRepository, TraceArtifactStore, TraceService

app = api_main.app


@pytest.fixture(autouse=True)
def isolated_chat_service(monkeypatch, tmp_path):
    service = ChatService(JsonChatRepository(str(tmp_path / "chat.json")))
    monkeypatch.setattr(api_main, "chat_service", service)
    trace_service = TraceService(
        FileTraceRepository(str(tmp_path / "traces")),
        TraceArtifactStore(str(tmp_path / "traces")),
    )
    monkeypatch.setattr(api_main, "trace_service", trace_service)
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


def test_answer_trace_can_be_loaded_by_id_message_and_downloaded():
    client = TestClient(app)
    response = client.post(
        "/answer",
        json={"question": "What is the workflow status?", "mode": "direct"},
    )
    assert response.status_code == 200
    answer = response.json()
    trace_id = answer["metadata"]["trace_id"]
    message_id = answer["metadata"]["assistant_message_id"]

    by_id = client.get(f"/traces/{trace_id}")
    assert by_id.status_code == 200
    report = by_id.json()
    assert report["schema_version"] == "1.0"
    assert report["status"] == "completed"
    assert report["message_id"] == message_id
    assert any(span["name"] == "Generator execution" for span in report["spans"])

    by_message = client.get(f"/chat/messages/{message_id}/trace?version_number=1")
    assert by_message.status_code == 200
    assert by_message.json()["trace_id"] == trace_id

    downloaded = client.get(f"/traces/{trace_id}/download")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-disposition"].endswith(f'"{trace_id}.json"')
    assert downloaded.json()["trace_id"] == trace_id
    assert client.get("/traces/trace-missing").status_code == 404


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


def test_answer_endpoint_accepts_advanced_mode_and_requires_knowledge_base():
    client = TestClient(app)
    response = client.post(
        "/answer",
        json={"question": "Research the full exception workflow.", "mode": "advanced_rag"},
    )
    assert response.status_code == 400
    assert "L4 Advanced RAG" in response.json()["detail"]


def test_agent_tools_endpoint_reports_available_and_placeholder_tools():
    client = TestClient(app)
    response = client.get("/rag/agent-tools?public_web_enabled=false")

    assert response.status_code == 200
    tools = {item["name"]: item for item in response.json()}
    assert tools["search_knowledge_base"]["available"] is True
    assert tools["fetch_public_url"]["available"] is False
    assert tools["search_google_drive"]["available"] is False


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


def test_chat_configuration_limits_are_exposed_and_enforced():
    client = TestClient(app)
    limits_response = client.get("/chat/configuration-limits")

    assert limits_response.status_code == 200
    limits = limits_response.json()
    assert limits == {
        "default_completed_exchanges": 3,
        "default_characters": 4000,
        "max_completed_exchanges": 6,
        "max_characters": 10000,
    }

    rejected = client.post(
        "/chat/configurations",
        json={
            "name": "Invalid conversation memory",
            "metadata": {
                "conversation_history_exchanges": 7,
                "conversation_history_characters": 10001,
            },
        },
    )
    assert rejected.status_code == 400
    assert "Completed exchanges" in rejected.json()["detail"]

    accepted = client.post(
        "/chat/configurations",
        json={
            "name": "Maximum conversation memory",
            "metadata": {
                "conversation_history_exchanges": 6,
                "conversation_history_characters": 10000,
            },
        },
    )
    assert accepted.status_code == 200
    assert accepted.json()["metadata"]["conversation_history_exchanges"] == 6
    assert accepted.json()["metadata"]["conversation_history_characters"] == 10000


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


def test_answer_uses_completed_history_for_second_turn():
    client = TestClient(app)
    first = client.post(
        "/answer",
        json={"question": "Explain the invoice mismatch approval workflow.", "mode": "direct"},
    ).json()

    second_response = client.post(
        "/answer",
        json={
            "conversation_id": first["conversation_id"],
            "question": "Who approves it?",
            "mode": "direct",
        },
    )

    assert second_response.status_code == 200
    second = second_response.json()
    assert second["question"] == "Who approves it?"
    assert second["metadata"]["history_exchange_count"] == 1
    assert second["metadata"]["query_rewritten"] is True
    assert "invoice mismatch approval workflow" in second["metadata"]["standalone_query"]
    messages = client.get(f"/chat/conversations/{first['conversation_id']}/messages").json()
    assert [message["role"] for message in messages] == ["user", "assistant", "user", "assistant"]


def test_answer_uses_saved_conversation_history_limits():
    client = TestClient(app)
    configuration = client.post(
        "/chat/configurations",
        json={
            "name": "Short conversation memory",
            "metadata": {
                "conversation_awareness_enabled": True,
                "conversation_history_exchanges": 1,
                "conversation_history_characters": 500,
            },
        },
    ).json()
    first = client.post(
        "/answer",
        json={
            "question": "Explain invoice mismatch review.",
            "mode": "direct",
            "chat_configuration_id": configuration["id"],
        },
    ).json()
    client.post(
        "/answer",
        json={
            "conversation_id": first["conversation_id"],
            "question": "Explain the approval owner.",
            "mode": "direct",
            "chat_configuration_id": configuration["id"],
        },
    )

    third = client.post(
        "/answer",
        json={
            "conversation_id": first["conversation_id"],
            "question": "What happens after that?",
            "mode": "direct",
            "chat_configuration_id": configuration["id"],
        },
    ).json()

    assert third["metadata"]["history_exchange_limit"] == 1
    assert third["metadata"]["history_character_limit"] == 500
    assert third["metadata"]["history_exchange_count"] == 1
    assert third["metadata"]["history_character_count"] <= 500
    assert "approval owner" in third["metadata"]["standalone_query"]


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


def test_answer_stream_uses_history_loaded_before_current_message():
    client = TestClient(app)
    first = client.post(
        "/answer",
        json={"question": "Describe the purchase order approval workflow.", "mode": "direct"},
    ).json()

    response = client.post(
        "/answer/stream",
        json={
            "conversation_id": first["conversation_id"],
            "question": "What happens after that?",
            "mode": "direct",
        },
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    completed = next(event["data"] for event in events if event["type"] == "completed")
    assert completed["metadata"]["history_exchange_count"] == 1
    assert completed["metadata"]["query_rewritten"] is True
    trace_names = [step["step"] for step in completed["metadata"]["trace_steps"]]
    assert "Conversation context" in trace_names
    assert "Query reformulation" in trace_names


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


def test_cancel_endpoint_tracks_active_and_terminal_requests(monkeypatch):
    coordinator = api_main.CancellationCoordinator()
    monkeypatch.setattr(api_main, "answer_cancellations", coordinator)
    client = TestClient(app)
    token = coordinator.register("request-active")

    accepted = client.post("/answer/requests/request-active/cancel")

    assert accepted.status_code == 202
    assert accepted.json()["status"] == "cancel_requested"
    assert token.is_cancelled is True
    coordinator.finish("request-active", "cancelled")
    assert client.post("/answer/requests/request-active/cancel").status_code == 409
    assert client.post("/answer/requests/request-missing/cancel").status_code == 404


def test_regenerate_stream_creates_version_without_adding_history_exchange():
    client = TestClient(app)
    initial_response = client.post(
        "/answer/stream",
        json={"question": "What is UAT?", "mode": "direct"},
    )
    initial_events = _sse_events(initial_response.text)
    initial = next(event["data"] for event in initial_events if event["type"] == "completed")
    message_id = initial["metadata"]["assistant_message_id"]

    response = client.post(
        f"/chat/messages/{message_id}/regenerate/stream",
        json={"mode": "direct", "request_id": "request-regenerate"},
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    started = next(event["data"] for event in events if event["type"] == "started")
    completed = next(event["data"] for event in events if event["type"] == "completed")
    assert started["message_version_number"] == 2
    assert completed["metadata"]["message_version_number"] == 2
    assert completed["metadata"]["regenerated"] is True

    versions = client.get(f"/chat/messages/{message_id}/versions").json()
    assert [version["version_number"] for version in versions] == [1, 2]
    assert [version["status"] for version in versions] == ["completed", "completed"]
    messages = client.get(f"/chat/conversations/{initial['conversation_id']}/messages").json()
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["content"] == completed["answer"]
    assert messages[1]["version_count"] == 2
    assert messages[1]["latest_version_number"] == 2


def test_regenerate_rejects_non_latest_assistant_message():
    client = TestClient(app)
    first = client.post(
        "/answer",
        json={"question": "What is UAT?", "mode": "direct"},
    ).json()
    first_message = client.get(
        f"/chat/conversations/{first['conversation_id']}/messages"
    ).json()[1]
    client.post(
        "/answer",
        json={
            "conversation_id": first["conversation_id"],
            "question": "Who approves it?",
            "mode": "direct",
        },
    )

    response = client.post(
        f"/chat/messages/{first_message['id']}/regenerate/stream",
        json={"mode": "direct"},
    )

    assert response.status_code == 409
    assert "latest completed" in response.json()["detail"]


def test_failed_regeneration_preserves_canonical_answer(monkeypatch):
    client = TestClient(app)
    initial = client.post(
        "/answer",
        json={"question": "What is UAT?", "mode": "direct"},
    ).json()
    message = client.get(
        f"/chat/conversations/{initial['conversation_id']}/messages"
    ).json()[1]
    original_content = message["content"]

    async def broken_stream(*args, **kwargs):
        yield ModelStreamEvent("delta", {"text": "partial retry"})
        raise ModelFarmError("retry provider stopped")

    monkeypatch.setattr(api_main.model_gateway, "stream", broken_stream)
    response = client.post(
        f"/chat/messages/{message['id']}/regenerate/stream",
        json={"mode": "direct"},
    )

    assert response.status_code == 200
    events = _sse_events(response.text)
    assert events[-1]["type"] == "error"
    versions = client.get(f"/chat/messages/{message['id']}/versions").json()
    assert versions[-1]["status"] == "failed"
    assert versions[-1]["content"] == "partial retry"
    canonical = client.get(
        f"/chat/conversations/{initial['conversation_id']}/messages"
    ).json()[1]
    assert canonical["status"] == "completed"
    assert canonical["content"] == original_content


def test_retry_failed_answer_creates_version_and_reuses_exchange(monkeypatch):
    original_stream = api_main.model_gateway.stream

    async def broken_stream(*args, **kwargs):
        yield ModelStreamEvent("delta", {"text": "partial failed answer"})
        raise ModelFarmError("provider stopped")

    monkeypatch.setattr(api_main.model_gateway, "stream", broken_stream)
    client = TestClient(app)
    failed_response = client.post(
        "/answer/stream",
        json={"question": "What is UAT?", "mode": "direct", "request_id": "request-failed-initial"},
    )
    failed_events = _sse_events(failed_response.text)
    started = next(
        event["data"]
        for event in failed_events
        if event["type"] == "started" and event["data"].get("assistant_message_id")
    )
    assert failed_events[-1]["type"] == "error"

    monkeypatch.setattr(api_main.model_gateway, "stream", original_stream)
    retry_response = client.post(
        f"/chat/messages/{started['assistant_message_id']}/retry/stream",
        json={"mode": "direct", "request_id": "request-retry-success"},
    )

    assert retry_response.status_code == 200, retry_response.text
    retry_events = _sse_events(retry_response.text)
    retry_started = next(event["data"] for event in retry_events if event["type"] == "started")
    completed = next(event["data"] for event in retry_events if event["type"] == "completed")
    assert retry_started["message_version_number"] == 2
    assert completed["metadata"]["retried"] is True
    versions = client.get(f"/chat/messages/{started['assistant_message_id']}/versions").json()
    assert [version["status"] for version in versions] == ["failed", "completed"]
    messages = client.get(f"/chat/conversations/{started['conversation_id']}/messages").json()
    assert [message["role"] for message in messages] == ["user", "assistant"]
    assert messages[1]["status"] == "completed"
    assert messages[1]["content"] == completed["answer"]


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
    model_farm_service = ModelFarmService(
        JsonModelFarmRepository(str(tmp_path / "models.json")),
        secret_key="unit-test-model-secret-key",
    )
    model_gateway = ModelGateway(model_farm_service)
    monkeypatch.setattr(api_main, "model_farm_service", model_farm_service)
    monkeypatch.setattr(api_main, "model_gateway", model_gateway)
    monkeypatch.setattr(api_main, "_require_user", lambda authorization: None)
    monkeypatch.setattr(api_main, "_require_admin", lambda authorization: None)
    client = TestClient(app)

    providers = client.get("/model-farm/providers")
    assert providers.status_code == 200
    assert {item["id"] for item in providers.json()} >= {
        "openai-generation", "openrouter-generation", "gemini-generation", "ollama-generation", "vllm-generation"
    }

    connection_response = client.post(
        "/model-farm/connections",
        json={
            "name": "OpenAI production",
            "provider": "openai",
            "access_path": "production",
            "api_base": "https://api.openai.com/v1",
            "credential_env_refs": {"api_key": "ARAGBIZ_MODEL_OPENAI_API_KEY"},
        },
    )
    assert connection_response.status_code == 200
    connection = connection_response.json()
    assert connection["credential_status"]["references"] == ["ARAGBIZ_MODEL_OPENAI_API_KEY"]
    assert "credential_secrets" not in connection

    created = client.post(
        "/model-farm/deployments/from-template",
        json={
            "template_id": "openai-generation",
            "connection_id": connection["id"],
            "name": "OpenAI GPT deployment",
            "capabilities": ["generation", "judge"],
        },
    )
    assert created.status_code == 200
    payload = created.json()
    assert payload["enabled"] is False
    assert payload["health_status"] == "untested"
    assert payload["connection_id"] == connection["id"]
    assert payload["gateway_model"] == "gpt-4.1-mini"
    assert payload["metadata"]["template_id"] == "openai-generation"

    enable = client.patch(f"/model-farm/deployments/{payload['id']}", json={"enabled": True})
    assert enable.status_code == 400
    assert "Test a remote deployment" in enable.json()["detail"]


def test_model_farm_draft_endpoint_tests_without_persisting(monkeypatch, tmp_path):
    model_farm_service = ModelFarmService(
        JsonModelFarmRepository(str(tmp_path / "models.json")),
        secret_key="unit-test-model-secret-key",
    )
    model_gateway = ModelGateway(model_farm_service)
    monkeypatch.setattr(api_main, "model_farm_service", model_farm_service)
    monkeypatch.setattr(api_main, "model_gateway", model_gateway)
    monkeypatch.setattr(api_main, "_require_user", lambda authorization: None)
    monkeypatch.setattr(api_main, "_require_admin", lambda authorization: None)
    class FakeLiteLLM:
        async def acompletion(self, **kwargs):
            assert kwargs["model"] == "openrouter/google/gemma-3-27b-it:free"
            return {
                "choices": [{"message": {"content": "ready"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            }

    monkeypatch.setattr(model_gateway.litellm_adapter, "_module", lambda: FakeLiteLLM())
    client = TestClient(app)

    connection_response = client.post(
        "/model-farm/connections",
        json={
            "name": "OpenRouter experimentation",
            "provider": "openrouter",
            "access_path": "experimentation",
            "api_base": "https://openrouter.ai/api/v1",
            "credential_secrets": {"api_key": "sk-test"},
        },
    )
    assert connection_response.status_code == 200
    connection_id = connection_response.json()["id"]

    response = client.post(
        "/model-farm/deployments/test-draft",
        json={
            "template_id": "openrouter-generation",
            "connection_id": connection_id,
            "name": "Draft OpenRouter model",
            "model": "google/gemma-3-27b-it:free",
            "capabilities": ["generation", "judge", "planner"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["sample"] == "ready"
    assert payload["deployment"]["provider"] == "openrouter"
    assert payload["deployment"]["capabilities"] == ["generation", "judge", "planner"]
    assert payload["deployment"]["connection_id"] == connection_id
    deployments = client.get("/model-farm/deployments").json()
    assert all(item["name"] != "Draft OpenRouter model" for item in deployments)


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
        },
    )
    assert created.status_code == 200
    run = created.json()
    assert run["status"] == "completed"
    assert run["metadata"]["ragxplain"]["status"] == "not_requested"
    assert run["metrics"]["average_retrieved_contexts"] >= 1
    assert run["metadata"]["comparison_model"] == "configuration_matrix"

    listed = client.get("/evaluation/runs").json()
    assert listed[0]["id"] == run["id"]
    cases = client.get(f"/evaluation/runs/{run['id']}/cases").json()
    assert len(cases) == 1
    assert cases[0]["answer"]
    assert cases[0]["contexts"]
    assert cases[0]["answer_metadata"]["trace_steps"]
    assert client.get(f"/evaluation/runs/{run['id']}/ragxplain/overall-insights").status_code == 409

    viewer = client.get("/evaluation/ragxplain/viewer")
    assert viewer.status_code == 200
    assert "RAGXplain viewer" in viewer.text

    missing = client.get("/evaluation/runs/eval-missing")
    assert missing.status_code == 404
    assert client.get("/evaluation/runs/eval-missing/ragxplain/overall-insights").status_code == 404

    deleted = client.delete(f"/evaluation/runs/{run['id']}")
    assert deleted.status_code == 200
    assert client.get(f"/evaluation/runs/{run['id']}").status_code == 404


def test_evaluation_experiment_endpoint_queues_matrix_job(monkeypatch, isolated_chat_service):
    configuration = isolated_chat_service.create_configuration("Benchmark configuration")
    knowledge_base = type(
        "KnowledgeBase",
        (),
        {
            "id": "kb-wixqa",
            "name": "WixQA",
            "document_count": 6221,
            "metadata": {"active_index_version_id": "index-1"},
        },
    )()
    monkeypatch.setattr(api_main.knowledge_service, "get_knowledge_base", lambda _id: knowledge_base)
    monkeypatch.setattr(api_main.model_farm_service, "resolve", lambda *_args, **_kwargs: object())
    created = EvaluationExperimentRecord(
        id="experiment-1",
        name="Matrix",
        status="queued",
        knowledge_base_id="kb-wixqa",
        knowledge_base_name="WixQA",
        configuration_ids=[configuration.id],
        datasets={"expertwritten": 2},
        judge_deployment_id="judge-1",
        created_by="dev-admin",
        created_at="2026-07-24T00:00:00+00:00",
        updated_at="2026-07-24T00:00:00+00:00",
    )
    monkeypatch.setattr(api_main.evaluation_experiment_service, "create", lambda **_kwargs: created)
    monkeypatch.setattr(
        api_main.job_service,
        "enqueue",
        lambda *_args, **_kwargs: BackgroundJob(
            id="job-1",
            job_type="evaluation_experiment",
            status="queued",
            payload={"experiment_id": "experiment-1"},
            created_at="2026-07-24T00:00:00+00:00",
            updated_at="2026-07-24T00:00:00+00:00",
        ),
    )
    response = TestClient(app).post(
        "/evaluation/experiments",
        json={
            "name": "Matrix",
            "knowledge_base_id": "kb-wixqa",
            "configuration_ids": [configuration.id],
            "datasets": {"expertwritten": 2},
            "judge_deployment_id": "judge-1",
        },
    )
    assert response.status_code == 202
    assert response.json()["experiment"]["id"] == "experiment-1"
    assert response.json()["job"]["job_type"] == "evaluation_experiment"


def test_ragxplain_diagnosis_accepts_limit_and_seed(monkeypatch):
    run = EvaluationRunRecord(
        id="eval-1",
        name="Configuration result",
        dataset_name="WixQA ExpertWritten",
        status="completed",
        knowledge_base_id="kb-1",
        metadata={
            "judge_deployment_id": "judge-1",
            "ragxplain": {"status": "not_requested"},
        },
    )
    queued = EvaluationRunRecord(
        **{
            **run.__dict__,
            "metadata": {
                **run.metadata,
                "ragxplain": {
                    "status": "queued",
                    "judge": "judge-1",
                    "case_count": 25,
                    "random_state": 7,
                },
            },
        }
    )
    monkeypatch.setattr(api_main.evaluation_service, "get_run", lambda _run_id: run)
    monkeypatch.setattr(api_main.evaluation_service, "queue_ragxplain", lambda *_args, **_kwargs: queued)
    monkeypatch.setattr(
        api_main.job_service,
        "enqueue",
        lambda *_args, **_kwargs: BackgroundJob(
            id="job-ragxplain",
            job_type="evaluation_ragxplain",
            status="queued",
            payload={"run_id": "eval-1", "limit": 25, "seed": 7},
            created_at="2026-07-25T00:00:00+00:00",
            updated_at="2026-07-25T00:00:00+00:00",
        ),
    )

    response = TestClient(app).post(
        "/evaluation/runs/eval-1/ragxplain",
        json={"limit": 25, "seed": 7},
    )

    assert response.status_code == 202
    assert response.json()["ragxplain"]["status"] == "queued"
    assert response.json()["ragxplain"]["case_count"] == 25


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
