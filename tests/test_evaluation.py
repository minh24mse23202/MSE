import json
import subprocess
from pathlib import Path

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
from aragbiz.routing import AdaptiveRouter
from aragbiz.ragxplain import RagxplainError, RagxplainRunner
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


def test_evaluation_service_runs_adaptive_and_static_baseline(tmp_path):
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

    run = evaluation.run(EvaluationRunConfig(knowledge_base_id=kb.id, retrieval_mode="bm25", top_k=2, limit=2, compare_baseline=True))
    cases = evaluation.list_cases(run.id)

    assert run.status == "completed"
    assert run.metrics["average_retrieved_contexts"] >= 1
    assert run.baseline_metrics["average_retrieved_contexts"] >= 1
    assert run.route_distribution["l3_complex_rag"] == 2
    assert run.baseline_route_distribution["l2_simple_rag"] == 2
    assert len(cases) == 2
    assert cases[0].adaptive_metadata["trace_steps"]
    assert cases[0].static_metadata["route_level"] == "l2_simple_rag"


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

    def failed_process(command, **kwargs):
        return subprocess.CompletedProcess(command, 2, stdout="", stderr="Judge unavailable")

    runner = RagxplainRunner(
        str(root),
        str(tmp_path / "results"),
        "examples.real_judge:judge",
        process_runner=failed_process,
    )
    evaluation = EvaluationService(
        JsonEvaluationRepository(str(tmp_path / "eval.json")),
        service,
        str(dataset_path),
        ragxplain_runner=runner,
    )

    run = evaluation.run(EvaluationRunConfig(knowledge_base_id=kb.id, limit=1, run_ragxplain=True))

    assert run.status == "completed"
    assert run.metadata["ragxplain"]["status"] == "failed"
    assert "Judge unavailable" in run.metadata["ragxplain"]["error"]
    assert len(evaluation.list_cases(run.id)) == 1
