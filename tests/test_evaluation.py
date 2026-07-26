import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from aragbiz.answering import AdaptiveRAGAnswerService, AnswerOptions
from aragbiz.evaluation import (
    EvaluationCaseRecord,
    EvaluationRunConfig,
    EvaluationService,
    JsonEvaluationRepository,
    evaluate_predictions,
)
from aragbiz.generation import ExtractiveGenerator
from aragbiz.knowledge import HashEmbeddingModel, KnowledgeService, OverlapChunker
from aragbiz.knowledge_store import JsonKnowledgeRepository
from aragbiz.model_farm import ModelGenerationResult
from aragbiz.routing import AdaptiveRouter
from aragbiz.ragxplain import (
    RagxplainError,
    RagxplainRunner,
    _ModelGatewayJudge,
    _normalize_response_schema,
    _validate_semantic_metric_artifacts,
)
from aragbiz.schemas import AnswerResult, Document, QACRecord, RetrievedContext


class StubClassifier:
    def __init__(self, label="complex"):
        self.label = label

    def predict(self, query):
        return self.label


def build_answer_service(tmp_path, label="complex"):
    knowledge_service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=80, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
    )
    kb = knowledge_service.create_knowledge_base("Workflow KB")
    knowledge_service.create_document(
        kb.id,
        "Invoice workflow",
        "Invoice mismatches after goods receipt should be escalated to finance operations for approval. " * 4,
        {"article_id": "invoice-workflow"},
    )
    service = AdaptiveRAGAnswerService(
        router=AdaptiveRouter(StubClassifier(label)),
        generator=ExtractiveGenerator(),
        knowledge_service=knowledge_service,
    )
    return service, kb


def test_evaluate_predictions_empty_schema():
    metrics = evaluate_predictions([], [])
    assert metrics == {
        "routing_accuracy": 0.0,
        "context_relevance": 0.0,
        "faithfulness_proxy": 0.0,
        "answer_overlap": 0.0,
        "average_latency_ms": 0.0,
    }


def test_evaluate_predictions_non_empty_runtime_metrics():
    record = QACRecord(
        id="q1",
        question="How do I handle invoice mismatch?",
        answer="Escalate invoice mismatches to finance operations.",
        context="Invoice mismatch context",
        complexity_label="complex",
        metadata={"article_ids": ["doc-1"]},
    )
    result = AnswerResult(
        question=record.question,
        answer="Escalate invoice mismatches to finance operations.",
        contexts=[
            RetrievedContext(
                document=Document(
                    id="chunk-1",
                    text="Escalate invoice mismatches to finance operations.",
                    metadata={"document_id": "doc-1"},
                ),
                score=1.0,
                rank=1,
                mode="bm25",
            )
        ],
        metadata={"complexity_label": "complex", "latency_ms": 12.5},
    )

    metrics = evaluate_predictions([record], [result])

    assert metrics["routing_accuracy"] == 1.0
    assert metrics["context_relevance"] == 1.0
    assert metrics["faithfulness_proxy"] == 1.0
    assert metrics["answer_overlap"] == 1.0
    assert metrics["average_latency_ms"] == 12.5


def test_json_evaluation_repository_crud(tmp_path):
    service, kb = build_answer_service(tmp_path)
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"id":"q1","question":"How do I handle invoice mismatch?","answer":"Escalate to finance operations.","context":"Invoice mismatch","complexity_label":"complex","metadata":{}}\n',
        encoding="utf-8",
    )
    evaluation = EvaluationService(JsonEvaluationRepository(str(tmp_path / "eval.json")), service, str(dataset_path))
    run = evaluation.run(EvaluationRunConfig(knowledge_base_id=kb.id, retrieval_mode="bm25", top_k=2, limit=1))

    assert evaluation.get_run(run.id).id == run.id
    assert len(evaluation.list_runs()) == 1
    assert len(evaluation.list_cases(run.id)) == 1

    evaluation.delete_run(run.id)

    assert evaluation.list_runs() == []


def test_evaluation_service_executes_the_saved_configuration_route_once(tmp_path):
    service, kb = build_answer_service(tmp_path, label="complex")
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '\n'.join([
            '{"id":"q1","question":"After invoice mismatch, who approves follow-up?","answer":"Finance operations approves follow-up.","context":"Invoice mismatch","complexity_label":"complex","metadata":{}}',
            '{"id":"q2","question":"How are invoice mismatches escalated?","answer":"Escalate invoice mismatches to finance operations.","context":"Invoice mismatch","complexity_label":"complex","metadata":{}}',
        ]) + '\n',
        encoding="utf-8",
    )
    evaluation = EvaluationService(JsonEvaluationRepository(str(tmp_path / "eval.json")), service, str(dataset_path))

    run = evaluation.run(
        EvaluationRunConfig(
            knowledge_base_id=kb.id,
            retrieval_mode="bm25",
            top_k=2,
            limit=2,
            chat_configuration={"metadata": {"route_mode": "simple_rag"}},
        )
    )
    cases = evaluation.list_cases(run.id)

    assert run.status == "completed"
    assert run.metrics["average_retrieved_contexts"] >= 1
    assert run.baseline_metrics == {}
    assert run.route_distribution["l2_simple_rag"] == 2
    assert run.baseline_route_distribution == {}
    assert run.metadata["evaluation_mode"] == "simple_rag"
    assert len(cases) == 2
    assert cases[0].adaptive_metadata["trace_steps"]
    assert cases[0].static_metadata == {}
    assert cases[0].metrics["result"]["wixqa"]


def test_evaluation_runs_each_case_without_conversation_context(tmp_path):
    service, kb = build_answer_service(tmp_path, label="complex")
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '\n'.join([
            '{"id":"q1","question":"How are invoice mismatches escalated?","answer":"Escalate to finance operations.","context":"Invoice mismatch","complexity_label":"complex","metadata":{}}',
            '{"id":"q2","question":"Who approves it?","answer":"Finance operations.","context":"Invoice mismatch","complexity_label":"complex","metadata":{}}',
        ]) + '\n',
        encoding="utf-8",
    )
    saved_configuration = {
        "metadata": {
            "conversation_awareness_enabled": True,
            "conversation_history_exchanges": 6,
            "conversation_history_characters": 10000,
            "agent_public_web_enabled": True,
        }
    }
    evaluation = EvaluationService(JsonEvaluationRepository(str(tmp_path / "eval.json")), service, str(dataset_path))

    run = evaluation.run(
        EvaluationRunConfig(
            knowledge_base_id=kb.id,
            retrieval_mode="bm25",
            top_k=2,
            limit=2,
            chat_configuration=saved_configuration,
        )
    )
    cases = evaluation.list_cases(run.id)

    assert saved_configuration["metadata"]["conversation_awareness_enabled"] is True
    assert saved_configuration["metadata"]["agent_public_web_enabled"] is True
    assert run.metadata["conversation_context"]["enabled"] is False
    assert run.metadata["chat_configuration"]["metadata"]["agent_public_web_enabled"] is False
    for case in cases:
        metadata = case.adaptive_metadata
        assert metadata["conversation_awareness_enabled"] is False
        assert metadata["history_exchange_count"] == 0
        assert metadata["history_character_count"] == 0
        assert metadata["query_rewritten"] is False
        assert metadata["standalone_query"] == metadata["original_query"]
        conversation_step = next(
            step for step in metadata["trace_steps"] if step["step"] == "Conversation context"
        )
        assert conversation_step["status"] == "skipped"
        assert conversation_step["metadata"]["enabled"] is False


def test_wixqa_judge_metrics_use_model_gateway_with_metric_purposes(tmp_path):
    service, kb = build_answer_service(tmp_path, label="moderate")
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"id":"q1","question":"How are invoice mismatches handled?","answer":"Escalate to finance operations.",'
        '"context":"Invoice mismatch","complexity_label":"moderate","metadata":{}}\n',
        encoding="utf-8",
    )

    class JudgeGateway:
        def __init__(self):
            self.calls = []

        def generate_sync(self, _messages, deployment_id, **kwargs):
            self.calls.append((deployment_id, kwargs["context"].purpose))
            return ModelGenerationResult(
                text='{"score": 0.8, "explanation": "Supported"}',
                deployment_id=deployment_id,
                provider="test",
                model="judge",
                status="completed",
                metadata={"usage_event_id": f"usage-{len(self.calls)}"},
            )

    gateway = JudgeGateway()
    evaluation = EvaluationService(
        JsonEvaluationRepository(str(tmp_path / "eval.json")),
        service,
        str(dataset_path),
        model_gateway=gateway,  # type: ignore[arg-type]
    )
    run = evaluation.run(
        EvaluationRunConfig(
            knowledge_base_id=kb.id,
            retrieval_mode="bm25",
            top_k=2,
            limit=1,
            judge_deployment_id="judge-1",
        )
    )
    wixqa = evaluation.list_cases(run.id)[0].metrics["adaptive"]["wixqa"]

    assert wixqa["context_recall"] == pytest.approx(0.8)
    assert wixqa["factuality"] == pytest.approx(0.8)
    assert gateway.calls == [
        ("judge-1", "evaluation_wixqa_context_recall"),
        ("judge-1", "evaluation_wixqa_factuality"),
    ]


def _ragxplain_case():
    return EvaluationCaseRecord(
        id="case-1",
        run_id="eval-1",
        record_id="q1",
        question="Who approves an invoice mismatch?",
        expected_answer="Finance operations approves it.",
        complexity_label="moderate",
        adaptive_answer="Finance operations approves the mismatch.",
        adaptive_contexts=[
            {
                "id": "chunk-1",
                "text": "Invoice mismatches are escalated to finance operations for approval.",
                "metadata": {"title": "Invoice workflow"},
            }
        ],
        metrics={
            "adaptive": {
                "routing_match": 1.0,
                "context_relevance": 0.8,
                "faithfulness_proxy": 0.9,
                "answer_overlap": 0.75,
                "latency_ms": 12.0,
            }
        },
    )


def _create_ragxplain_root(tmp_path):
    root = tmp_path / "ragxplain"
    (root / "ragxplain").mkdir(parents=True)
    (root / "ragxplain" / "cli.py").write_text("", encoding="utf-8")
    (root / "viewer").mkdir()
    (root / "viewer" / "insights-viewer.html").write_text("<html>viewer</html>", encoding="utf-8")
    return root


def _successful_ragxplain_process(command, **kwargs):
    output_dir = Path(command[command.index("--out") + 1])
    (output_dir / "results.csv").write_text("question,candidate_answer\nq,a\n", encoding="utf-8")
    (output_dir / "metrics_insights.json").write_text("{}", encoding="utf-8")
    (output_dir / "overall_insights.json").write_text(
        json.dumps({"analysis": {"executive_summary": "Healthy run", "insights": []}}),
        encoding="utf-8",
    )
    return subprocess.CompletedProcess(command, 0, stdout="done", stderr="")


def test_ragxplain_runner_exports_and_validates_artifacts(tmp_path):
    root = _create_ragxplain_root(tmp_path)
    runner = RagxplainRunner(
        str(root),
        str(tmp_path / "results"),
        "examples.mock_judge_impl:judge",
        process_runner=_successful_ragxplain_process,
    )

    metadata = runner.run("eval-1", "Evaluation", [_ragxplain_case()], {"top_k": 4})
    exported = json.loads(Path(metadata["input_path"]).read_text(encoding="utf-8").splitlines()[0])

    assert exported["question"] == "Who approves an invoice mismatch?"
    assert exported["candidate_answer"].startswith("Finance operations")
    assert exported["gt_answer"] == "Finance operations approves it."
    assert "[1] Invoice workflow" in exported["contexts"]
    assert exported["faithfulness_proxy"] == 0.9
    assert metadata["status"] == "completed"
    assert runner.load_overall_insights(metadata["overall_insights_path"])["analysis"]["executive_summary"] == "Healthy run"


def test_ragxplain_runner_rejects_invalid_output_and_unsafe_deletion(tmp_path):
    root = _create_ragxplain_root(tmp_path)

    def invalid_process(command, **kwargs):
        output_dir = Path(command[command.index("--out") + 1])
        (output_dir / "results.csv").write_text("", encoding="utf-8")
        (output_dir / "metrics_insights.json").write_text("{}", encoding="utf-8")
        (output_dir / "overall_insights.json").write_text('{"analysis": "invalid"}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    runner = RagxplainRunner(
        str(root),
        str(tmp_path / "results"),
        "examples.mock_judge_impl:judge",
        process_runner=invalid_process,
    )

    with pytest.raises(RagxplainError, match="analysis object"):
        runner.run("eval-1", "Evaluation", [_ragxplain_case()], {})
    with pytest.raises(RagxplainError, match="Refusing to delete"):
        runner.delete_artifacts("eval-1", str(tmp_path / "outside"))


def test_ragxplain_failure_does_not_discard_evaluation_cases(tmp_path):
    service, kb = build_answer_service(tmp_path)
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"id":"q1","question":"Who approves invoice mismatches?","answer":"Finance operations.","context":"Invoice mismatch","complexity_label":"complex","metadata":{}}\n',
        encoding="utf-8",
    )
    root = _create_ragxplain_root(tmp_path)

    runner = RagxplainRunner(
        str(root),
        str(tmp_path / "results"),
        "legacy-judge-unused",
    )
    runner.run_with_model_gateway = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        RagxplainError("Judge unavailable")
    )

    class JudgeGateway:
        def generate_sync(self, _messages, deployment_id, **_kwargs):
            return ModelGenerationResult(
                text='{"score": 0.8, "explanation": "Supported"}',
                deployment_id=deployment_id,
                provider="test",
                model="judge",
                status="completed",
            )

    evaluation = EvaluationService(
        JsonEvaluationRepository(str(tmp_path / "eval.json")),
        service,
        str(dataset_path),
        ragxplain_runner=runner,
        model_gateway=JudgeGateway(),  # type: ignore[arg-type]
    )

    run = evaluation.run(
        EvaluationRunConfig(
            knowledge_base_id=kb.id,
            limit=1,
            judge_deployment_id="judge-1",
        )
    )
    run = evaluation.run_ragxplain(run.id, limit=1)

    assert run.status == "completed"
    assert run.metadata["ragxplain"]["status"] == "failed"
    assert "Judge unavailable" in run.metadata["ragxplain"]["error"]
    assert len(evaluation.list_cases(run.id)) == 1


def test_model_gateway_judge_normalizes_fenced_schema_json():
    class Gateway:
        def __init__(self):
            self.calls = []

        async def generate(self, messages, deployment_id, **kwargs):
            self.calls.append((messages, deployment_id, kwargs))
            return ModelGenerationResult(
                text='```json\n{"analysis":{"executive_summary":"Healthy"}}\n```',
                deployment_id=deployment_id,
                provider="openai",
                model="gpt-test",
                status="completed",
            )

    gateway = Gateway()
    judge = _ModelGatewayJudge(gateway, "judge-1", "eval-1", "kb-1", True)
    request = SimpleNamespace(
        system_prompt="Return JSON.",
        user_prompt="Analyze this run.",
        metadata={
            "prompt_key": "overall_insight",
            "response_schema": {
                "name": "prompts/overall_insight.md",
                "schema": {
                    "type": "object",
                    "properties": {"analysis": {"type": "object"}},
                    "required": ["analysis"],
                },
            },
        },
    )

    result = asyncio.run(judge.run(request))

    assert json.loads(result)["analysis"]["executive_summary"] == "Healthy"
    assert len(gateway.calls) == 1
    parameters = gateway.calls[0][2]["parameters"]
    assert parameters["max_tokens"] == 6000
    assert parameters["response_format"]["type"] == "json_schema"
    assert parameters["response_format"]["json_schema"]["name"] == "prompts_overall_insight_md"


def test_model_gateway_judge_retries_incomplete_schema_json():
    class Gateway:
        def __init__(self):
            self.calls = []

        async def generate(self, messages, deployment_id, **kwargs):
            self.calls.append((messages, deployment_id, kwargs))
            text = (
                '{"analysis":{"executive_summary":"Truncated"'
                if len(self.calls) == 1
                else '{"analysis":{"executive_summary":"Recovered"}}'
            )
            return ModelGenerationResult(
                text=text,
                deployment_id=deployment_id,
                provider="openai",
                model="gpt-test",
                status="completed",
            )

    gateway = Gateway()
    judge = _ModelGatewayJudge(gateway, "judge-1", "eval-1", "kb-1", True)
    request = SimpleNamespace(
        system_prompt="Return JSON.",
        user_prompt="Analyze this run.",
        metadata={
            "prompt_key": "overall_insight",
            "response_schema": {
                "name": "ragxplain_overall_insight",
                "schema": {
                    "type": "object",
                    "properties": {"analysis": {"type": "object"}},
                    "required": ["analysis"],
                },
            },
        },
    )

    result = asyncio.run(judge.run(request))

    assert json.loads(result)["analysis"]["executive_summary"] == "Recovered"
    assert len(gateway.calls) == 2
    assert gateway.calls[1][2]["parameters"]["max_tokens"] == 7000
    assert gateway.calls[1][2]["context"].purpose == "ragxplain_overall_insight_json_retry"


def test_ragxplain_response_schema_name_is_normalized_without_mutation():
    original = {
        "name": "prompts/context_relevancy.md",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"context_relevancy_score": {"type": "string"}},
        },
    }

    normalized = _normalize_response_schema(original)

    assert normalized["name"] == "prompts_context_relevancy_md"
    assert normalized["schema"] == original["schema"]
    assert normalized["schema"] is not original["schema"]
    assert original["name"] == "prompts/context_relevancy.md"
    assert _normalize_response_schema({"name": "overall_insight_response"})["name"] == "overall_insight_response"
    assert _normalize_response_schema({"name": ".../"})["name"] == "ragxplain_response"


def test_ragxplain_semantic_metric_validation_accepts_all_six_metrics(tmp_path):
    expected = [
        "context_relevancy",
        "context_adherence",
        "answer_relevancy",
        "context_recall",
        "factuality",
        "grading_note",
    ]
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        ",".join(["question", *[f"{metric}_score" for metric in expected]])
        + "\n"
        + ",".join(["Question", *(["0.8"] * len(expected))])
        + "\n",
        encoding="utf-8",
    )
    overall_insights = {
        "analysis": {"executive_summary": "Healthy"},
        "prompt": {
            "insights": {
                metric: {"metric_name": metric, "avg_score": 0.8}
                for metric in expected
            }
        },
    }

    coverage = _validate_semantic_metric_artifacts(results_path, overall_insights, expected)

    assert coverage["status"] == "completed"
    assert coverage["completed"] == expected
    assert coverage["missing"] == []


def test_ragxplain_semantic_metric_validation_reports_partial_coverage(tmp_path):
    expected = ["context_relevancy", "context_adherence", "answer_relevancy"]
    results_path = tmp_path / "results.csv"
    results_path.write_text(
        "question,context_relevancy_score,context_adherence_score\nQuestion,0.8,0.7\n",
        encoding="utf-8",
    )
    overall_insights = {
        "analysis": {},
        "prompt": {
            "insights": {
                "context_relevancy": {"avg_score": 0.8},
                "context_adherence": {"avg_score": 0.7},
            }
        },
    }

    coverage = _validate_semantic_metric_artifacts(results_path, overall_insights, expected)

    assert coverage["status"] == "partial"
    assert coverage["completed"] == ["context_relevancy", "context_adherence"]
    assert coverage["missing"] == ["answer_relevancy"]
    assert coverage["missing_score_columns"] == ["answer_relevancy"]
    assert coverage["missing_insight_summaries"] == ["answer_relevancy"]


def test_ragxplain_semantic_metric_validation_rejects_empty_coverage(tmp_path):
    results_path = tmp_path / "results.csv"
    results_path.write_text("question,candidate_answer\nQuestion,Answer\n", encoding="utf-8")

    with pytest.raises(RagxplainError, match="produced no semantic metrics"):
        _validate_semantic_metric_artifacts(
            results_path,
            {"analysis": {}, "prompt": {"insights": {}}},
            ["context_relevancy", "context_adherence"],
        )
