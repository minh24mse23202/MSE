import json
from datetime import datetime, timedelta, timezone

import pytest

from aragbiz.analytics import AnalyticsFilters, AnalyticsService, JsonAnalyticsRepository
from aragbiz.chat import ChatService, JsonChatRepository


def _analytics_fixture(tmp_path):
    now = datetime.now(timezone.utc)
    created_at = now.isoformat()
    chat_path = tmp_path / "chat.json"
    chat = ChatService(JsonChatRepository(str(chat_path)))
    configuration = chat.create_configuration("Evaluation assistant", metadata={"configuration_id": "CfgTest00001"})
    conversation = chat.create_conversation(
        "Invoice workflow",
        owner_user_id="user-1",
        knowledge_base_id="kb-1",
        chat_configuration_id=configuration.id,
    )
    user_message = chat.append_message(
        conversation.id,
        "user",
        "How is an invoice approved?",
        user_id="user-1",
    )
    assistant = chat.append_message(
        conversation.id,
        "assistant",
        "The manager approves the invoice.",
        metadata={"trace_id": "trace-1", "question": user_message.content},
    )

    model_path = tmp_path / "models.json"
    model_path.write_text(
        json.dumps(
            {
                "connections": {},
                "deployments": {
                    "model-1": {"id": "model-1", "name": "Generator One"},
                    "judge-1": {"id": "judge-1", "name": "Judge One"},
                },
                "usage": [
                    {
                        "id": "usage-1",
                        "deployment_id": "model-1",
                        "provider": "Local",
                        "model": "extractive",
                        "capability": "generation",
                        "purpose": "answer_generation",
                        "status": "completed",
                        "user_id": "user-1",
                        "conversation_id": conversation.id,
                        "knowledge_base_id": "kb-1",
                        "evaluation_run_id": "",
                        "chat_configuration_id": configuration.id,
                        "input_tokens": 20,
                        "output_tokens": 10,
                        "total_tokens": 30,
                        "latency_ms": 12,
                        "estimated_cost_usd": 0.01,
                        "created_at": created_at,
                    },
                    {
                        "id": "usage-2",
                        "deployment_id": "judge-1",
                        "provider": "OpenAI",
                        "model": "judge",
                        "capability": "judge",
                        "purpose": "evaluation_wixqa_factuality",
                        "status": "failed",
                        "user_id": "user-1",
                        "conversation_id": "",
                        "knowledge_base_id": "kb-1",
                        "evaluation_run_id": "eval-1",
                        "chat_configuration_id": "",
                        "input_tokens": 40,
                        "output_tokens": 5,
                        "total_tokens": 45,
                        "latency_ms": 30,
                        "estimated_cost_usd": 0.02,
                        "created_at": created_at,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    knowledge_path = tmp_path / "knowledge.json"
    knowledge_path.write_text(
        json.dumps({"knowledge_bases": {"kb-1": {"id": "kb-1", "name": "Invoice KB"}}}),
        encoding="utf-8",
    )
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(
        json.dumps(
            {
                "evaluation_runs": {
                    "eval-1": {
                        "id": "eval-1",
                        "chat_configuration_id": configuration.id,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "users": {
                    "user-1": {
                        "id": "user-1",
                        "email": "user@example.com",
                        "first_name": "Test",
                        "last_name": "User",
                        "active": True,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    repository = JsonAnalyticsRepository(
        model_farm_path=str(model_path),
        chat_path=str(chat_path),
        knowledge_path=str(knowledge_path),
        evaluation_path=str(evaluation_path),
        auth_path=str(auth_path),
        feedback_path=str(tmp_path / "analytics.json"),
    )
    service = AnalyticsService(repository)
    filters = AnalyticsFilters(
        from_at=(now - timedelta(days=1)).isoformat(),
        to_at=(now + timedelta(days=1)).isoformat(),
    )
    return service, filters, assistant, configuration


def test_json_analytics_aggregates_usage_engagement_and_scopes(tmp_path):
    service, filters, _, configuration = _analytics_fixture(tmp_path)

    overview = service.overview(filters)
    assert overview["usage"]["calls"] == 2
    assert overview["usage"]["total_tokens"] == 75
    assert overview["usage"]["estimated_cost_usd"] == pytest.approx(0.03)
    assert overview["usage"]["failed_calls"] == 1
    assert overview["engagement"]["active_chats"] == 1
    assert overview["engagement"]["active_users"] == 1
    assert overview["engagement"]["total_messages"] == 2

    events = service.usage_events(filters)
    assert events["total"] == 2
    assert {item["activity_scope"] for item in events["items"]} == {"chat", "evaluation"}
    assert all(item["chat_configuration_id"] == configuration.id for item in events["items"])

    breakdowns = service.usage_breakdowns(filters, "tokens")
    assert breakdowns["models"]
    assert breakdowns["knowledge_bases"][0]["label"] == "Invoice KB"
    assert breakdowns["configurations"][0]["label"].startswith("CfgTest00001 |")


def test_json_analytics_consolidates_unavailable_knowledge_base_options(tmp_path):
    service, filters, _, _ = _analytics_fixture(tmp_path)
    model_path = service.repository.model_farm_path
    state = json.loads(model_path.read_text(encoding="utf-8"))
    template = dict(state["usage"][0])
    state["usage"].extend(
        [
            {**template, "id": "usage-orphan-1", "knowledge_base_id": "kb-deleted-1"},
            {**template, "id": "usage-orphan-2", "knowledge_base_id": "kb-deleted-2"},
        ]
    )
    model_path.write_text(json.dumps(state), encoding="utf-8")

    options = service.filter_options(filters)["knowledge_bases"]
    unavailable = [item for item in options if item["id"] == "__unavailable__"]
    assert unavailable == [
        {
            "id": "__unavailable__",
            "label": "Deleted or unavailable (2 Knowledge Bases)",
        }
    ]
    assert all(item["label"] != "Unknown" for item in options)

    unavailable_events = service.usage_events(
        AnalyticsFilters(
            from_at=filters.from_at,
            to_at=filters.to_at,
            knowledge_base_id="__unavailable__",
        )
    )
    assert unavailable_events["total"] == 2
    assert {item["knowledge_base_name"] for item in unavailable_events["items"]} == {"Deleted or unavailable"}


def test_json_feedback_upserts_one_rating_per_answer_version(tmp_path):
    service, filters, assistant, _ = _analytics_fixture(tmp_path)

    created = service.upsert_feedback(
        assistant_message_id=assistant.id,
        version_number=1,
        user_id="user-1",
        user_name="Test User",
        user_email="user@example.com",
        rating="up",
        comment="",
    )
    updated = service.upsert_feedback(
        assistant_message_id=assistant.id,
        version_number=1,
        user_id="user-1",
        user_name="Test User",
        user_email="user@example.com",
        rating="down",
        comment="The approval role is unclear.",
    )

    assert updated["id"] == created["id"]
    assert updated["rating"] == "down"
    page = service.feedback(filters)
    assert page["total"] == 1
    assert page["items"][0]["trace_id"] == "trace-1"
    assert page["items"][0]["question_snapshot"] == "How is an invoice approved?"
    assert page["items"][0]["comment"] == "The approval role is unclear."
    overview = service.overview(filters)
    assert overview["engagement"]["thumbs_up"] == 0
    assert overview["engagement"]["thumbs_down"] == 1

    rating_only_update = service.upsert_feedback(
        assistant_message_id=assistant.id,
        version_number=1,
        user_id="user-1",
        user_name="Test User",
        user_email="user@example.com",
        rating="up",
        comment=None,
    )
    assert rating_only_update["rating"] == "up"
    assert rating_only_update["comment"] == "The approval role is unclear."
    assert service.list_message_feedback([assistant.id], "user-1")[0]["comment"] == "The approval role is unclear."


def test_analytics_filters_validate_dates_scope_and_rating():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError, match="later"):
        AnalyticsFilters(from_at=now.isoformat(), to_at=now.isoformat()).normalized()
    with pytest.raises(ValueError, match="scope"):
        AnalyticsFilters(
            from_at=(now - timedelta(days=1)).isoformat(),
            to_at=now.isoformat(),
            scope="invalid",
        ).normalized()
    with pytest.raises(ValueError, match="rating"):
        AnalyticsFilters(
            from_at=(now - timedelta(days=1)).isoformat(),
            to_at=now.isoformat(),
            rating="neutral",
        ).normalized()
