from types import SimpleNamespace

from aragbiz.conversation import (
    QueryReformulator,
    build_conversation_history,
    conversation_history_characters,
)
from aragbiz.model_farm import ModelCallContext, ModelFarmError, ModelGenerationResult


def _message(message_id, role, content, status="completed"):
    return SimpleNamespace(id=message_id, role=role, content=content, status=status)


def test_history_uses_latest_three_completed_exchanges_and_excludes_failures():
    records = []
    for index in range(1, 5):
        records.extend(
            [
                _message(f"u{index}", "user", f"Question {index}"),
                _message(f"a{index}", "assistant", f"Answer {index}"),
            ]
        )
    records.extend(
        [
            _message("u-failed", "user", "Failed question"),
            _message("a-failed", "assistant", "Partial answer", status="failed"),
        ]
    )

    history = build_conversation_history(records)

    assert [message["message_id"] for message in history] == ["u2", "a2", "u3", "a3", "u4", "a4"]
    assert all("Failed" not in message["content"] for message in history)


def test_history_enforces_character_budget_without_losing_exchange_roles():
    records = [
        _message("u1", "user", "u" * 5000),
        _message("a1", "assistant", "a" * 5000),
    ]

    history = build_conversation_history(records)

    assert conversation_history_characters(history) == 4000
    assert [message["role"] for message in history] == ["user", "assistant"]


def test_standalone_question_is_not_rewritten():
    result = QueryReformulator().reformulate(
        "What is user acceptance testing?",
        [{"role": "user", "content": "Explain release approval."}, {"role": "assistant", "content": "It validates readiness."}],
        enabled=True,
    )

    assert result.rewritten is False
    assert result.strategy == "standalone"


def test_follow_up_uses_deterministic_reformulation_without_planner():
    result = QueryReformulator().reformulate(
        "Who approves it?",
        [{"role": "user", "content": "Explain the invoice mismatch workflow."}, {"role": "assistant", "content": "Finance reviews mismatches."}],
        enabled=True,
    )

    assert result.rewritten is True
    assert result.follow_up_detected is True
    assert result.strategy == "deterministic"
    assert "invoice mismatch workflow" in result.standalone_query
    assert "Who approves it?" in result.standalone_query


def test_follow_up_uses_selected_planner_capability():
    gateway = PlannerGateway()

    result = QueryReformulator().reformulate(
        "What happens after that?",
        [{"role": "user", "content": "How is an invoice mismatch reviewed?"}, {"role": "assistant", "content": "Finance investigates it."}],
        enabled=True,
        planner_deployment_id="planner-1",
        model_gateway=gateway,
        call_context=ModelCallContext(purpose="conversation_rewrite", request_id="req-1"),
    )

    assert result.standalone_query == "What happens after finance reviews an invoice mismatch?"
    assert result.strategy == "planner"
    assert gateway.capability == "planner"
    assert gateway.context.purpose == "conversation_rewrite"


def test_planner_failure_falls_back_to_deterministic_rewrite():
    result = QueryReformulator().reformulate(
        "What about the next step?",
        [{"role": "user", "content": "Describe purchase order approval."}, {"role": "assistant", "content": "A manager approves it."}],
        enabled=True,
        planner_deployment_id="planner-1",
        model_gateway=FailingPlannerGateway(),
    )

    assert result.rewritten is True
    assert result.strategy == "deterministic_fallback"
    assert "failed open" in result.warning


class PlannerGateway:
    capability = ""
    context = None

    def generate_sync(self, messages, deployment_id, **kwargs):
        self.capability = kwargs["capability"]
        self.context = kwargs["context"]
        assert deployment_id == "planner-1"
        assert "Current follow-up" in messages[1]["content"]
        return ModelGenerationResult(
            text='{"standalone_query":"What happens after finance reviews an invoice mismatch?"}',
            deployment_id=deployment_id,
            provider="Local",
            model="planner",
            status="completed",
            input_tokens=20,
            output_tokens=8,
        )


class FailingPlannerGateway:
    def generate_sync(self, *args, **kwargs):
        raise ModelFarmError("planner unavailable")
