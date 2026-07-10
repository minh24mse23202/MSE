from aragbiz.answering import AdaptiveRAGAnswerService, AnswerOptions
from aragbiz.evaluation import (
    EvaluationRunConfig,
    EvaluationService,
    JsonEvaluationRepository,
    evaluate_predictions,
)
from aragbiz.generation import ExtractiveGenerator
from aragbiz.knowledge import HashEmbeddingModel, KnowledgeService, OverlapChunker
from aragbiz.knowledge_store import JsonKnowledgeRepository
from aragbiz.routing import AdaptiveRouter
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
