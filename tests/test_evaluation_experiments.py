from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import aragbiz.evaluation_experiments as experiments_module
from aragbiz.evaluation import EvaluationRunRecord
from aragbiz.evaluation_experiments import (
    EvaluationExperimentService,
    JsonEvaluationExperimentRepository,
)
from aragbiz.evaluation_metrics import (
    DEFAULT_QUALITY_WEIGHTS,
    aggregate_wixqa_metrics,
    deterministic_case_metrics,
    quality_score,
    token_f1,
)


def test_wixqa_deterministic_metrics_and_quality_score():
    exact = deterministic_case_metrics("Save the site, then publish it.", "Save the site, then publish it.")
    assert exact["token_f1"] == pytest.approx(1.0)
    assert exact["rouge_1"] == pytest.approx(1.0)
    assert exact["rouge_2"] == pytest.approx(1.0)
    assert exact["bleu"] == pytest.approx(1.0)
    assert token_f1("one two two", "one two") == pytest.approx(0.8)

    rows = [
        {
            "wixqa": {
                **exact,
                "reference_answer": "Save the site, then publish it.",
                "candidate_answer": "Save the site, then publish it.",
                "context_recall": 0.8,
                "factuality": 0.9,
                "retrieval": {
                    "available": True,
                    "precision_at_k": 0.5,
                    "recall_at_k": 1.0,
                    "hit_at_k": 1.0,
                    "mrr": 1.0,
                    "ndcg_at_k": 1.0,
                },
            }
        }
    ]
    aggregate = aggregate_wixqa_metrics(rows)
    assert aggregate["judge_coverage"] == 1.0
    assert aggregate["retrieval_coverage"] == 1.0
    assert aggregate["precision_at_k"] == pytest.approx(0.5)
    assert aggregate["recall_at_k"] == pytest.approx(1.0)
    assert quality_score(aggregate, DEFAULT_QUALITY_WEIGHTS) == pytest.approx(0.91)


class _FakeEvaluationService:
    def __init__(self):
        self.runs = {}

    def get_run(self, run_id):
        if run_id not in self.runs:
            raise KeyError(run_id)
        return self.runs[run_id]

    def run(self, config):
        score = 0.9 if config.chat_configuration_id == "config-a" else 0.6
        record = EvaluationRunRecord(
            id=config.run_id,
            name=config.name,
            dataset_name=config.dataset_name,
            status="completed",
            knowledge_base_id=config.knowledge_base_id,
            knowledge_base_name="WixQA",
            chat_configuration_id=config.chat_configuration_id,
            retrieval_mode=config.retrieval_mode,
            top_k=config.top_k,
            limit=config.limit,
            metrics={
                "wixqa": {
                    "context_recall": score,
                    "factuality": score,
                    "token_f1": score,
                    "bleu": score,
                    "rouge_1": score,
                    "rouge_2": score,
                },
                "average_cost_per_case_usd": 0.01,
                "average_latency_ms": 500,
            },
            metadata={"record_count": config.limit, "experiment_id": config.experiment_id},
            created_at="2026-07-24T00:00:00+00:00",
            finished_at="2026-07-24T00:00:01+00:00",
        )
        self.runs[record.id] = record
        return record

    def delete_run(self, run_id):
        self.runs.pop(run_id, None)


class _FakeKnowledgeService:
    def __init__(self, article_ids):
        self.article_ids = list(article_ids)

    def list_wixqa_source_record_ids(self, knowledge_base_id):
        return list(self.article_ids)


def test_experiment_matrix_uses_shared_samples_and_ranks_configurations(monkeypatch, tmp_path):
    dataset = tmp_path / "expert.jsonl"
    dataset.write_text(
        "\n".join(
            json.dumps({"id": f"record-{index}", "question": "q", "answer": "a", "context": "c", "complexity_label": "moderate"})
            for index in range(5)
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        experiments_module,
        "DATASET_DEFINITIONS",
        {"expertwritten": {"name": "WixQA ExpertWritten", "path": str(dataset), "count": 5}},
    )
    fake = _FakeEvaluationService()
    service = EvaluationExperimentService(
        JsonEvaluationExperimentRepository(str(tmp_path / "evaluation.json")),
        fake,  # type: ignore[arg-type]
    )
    knowledge_base = SimpleNamespace(
        id="kb-wixqa",
        name="WixQA",
        document_count=6221,
        metadata={"active_index_version_id": "index-1"},
    )
    experiment = service.create(
        name="Matrix",
        knowledge_base=knowledge_base,
        configuration_snapshots=[
            {"id": "config-a", "name": "A", "metadata": {"retrieval_mode": "hybrid", "top_k": 4}},
            {"id": "config-b", "name": "B", "metadata": {"retrieval_mode": "bm25", "top_k": 8}},
        ],
        datasets={"expertwritten": 3},
        judge_deployment_id="judge-1",
    )
    completed = service.execute(experiment.id)
    assert completed.status == "completed"
    assert len(completed.run_ids) == 2
    assert completed.leaderboard[0]["configuration_id"] == "config-a"
    assert completed.leaderboard[0]["winner"] is True
    assert completed.metadata["knowledge_base_compatibility"]["status"] == "compatible"

    # Resume is idempotent because deterministic child run IDs are reused.
    resumed = service.execute(experiment.id)
    assert resumed.run_ids == completed.run_ids
    assert len(fake.runs) == 2


def test_failed_experiment_cells_report_incomplete_results(monkeypatch, tmp_path):
    dataset = tmp_path / "expert.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "record-1",
                "question": "q",
                "answer": "a",
                "context": "c",
                "complexity_label": "moderate",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        experiments_module,
        "DATASET_DEFINITIONS",
        {"expertwritten": {"name": "WixQA ExpertWritten", "path": str(dataset), "count": 1}},
    )

    class FailingEvaluationService(_FakeEvaluationService):
        def run(self, config):
            raise ValueError("Judge output is not valid JSON")

    service = EvaluationExperimentService(
        JsonEvaluationExperimentRepository(str(tmp_path / "evaluation.json")),
        FailingEvaluationService(),  # type: ignore[arg-type]
    )
    experiment = service.create(
        name="Failed matrix",
        knowledge_base=SimpleNamespace(
            id="kb-wixqa",
            name="WixQA",
            document_count=6221,
            metadata={"active_index_version_id": "index-1"},
        ),
        configuration_snapshots=[{"id": "config-a", "name": "A", "metadata": {}}],
        datasets={"expertwritten": 1},
        judge_deployment_id="judge-1",
    )

    completed = service.execute(experiment.id)

    assert completed.status == "failed"
    assert completed.leaderboard[0]["eligible"] is False
    assert completed.leaderboard[0]["constraint_violations"] == [
        "incomplete results (0/1)"
    ]
    assert "Judge output is not valid JSON" in completed.error


def test_partial_wixqa_kb_limits_evaluation_to_exact_article_coverage(monkeypatch, tmp_path):
    dataset = tmp_path / "synthetic.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({
                    "id": "record-a",
                    "question": "q",
                    "answer": "a",
                    "context": "c",
                    "complexity_label": "moderate",
                    "metadata": {"article_ids": ["article-a"]},
                }),
                json.dumps({
                    "id": "record-b",
                    "question": "q",
                    "answer": "a",
                    "context": "c",
                    "complexity_label": "moderate",
                    "metadata": {"article_ids": ["article-b"]},
                }),
                json.dumps({
                    "id": "record-cross",
                    "question": "q",
                    "answer": "a",
                    "context": "c",
                    "complexity_label": "complex",
                    "metadata": {"article_ids": ["article-a", "article-b"]},
                }),
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        experiments_module,
        "DATASET_DEFINITIONS",
        {"synthetic": {"name": "WixQA Synthetic", "path": str(dataset), "count": 3}},
    )
    fake_evaluation = _FakeEvaluationService()
    fake_knowledge = _FakeKnowledgeService(["article-a"])
    service = EvaluationExperimentService(
        JsonEvaluationExperimentRepository(str(tmp_path / "evaluation.json")),
        fake_evaluation,  # type: ignore[arg-type]
        fake_knowledge,  # type: ignore[arg-type]
    )
    knowledge_base = SimpleNamespace(
        id="kb-partial",
        name="Partial WixQA",
        document_count=1,
        metadata={"active_index_version_id": "index-1"},
    )

    datasets = service.datasets(knowledge_base.id)
    assert datasets[0]["compatible_record_count"] == 1
    assert datasets[0]["knowledge_base_document_count"] == 1

    with pytest.raises(ValueError, match="cannot exceed the 1 records compatible"):
        service.create(
            name="Too many",
            knowledge_base=knowledge_base,
            configuration_snapshots=[{"id": "config-a", "name": "A", "metadata": {}}],
            datasets={"synthetic": 2},
            judge_deployment_id="judge-1",
        )

    experiment = service.create(
        name="Exact subset",
        knowledge_base=knowledge_base,
        configuration_snapshots=[{"id": "config-a", "name": "A", "metadata": {}}],
        datasets={"synthetic": 1},
        judge_deployment_id="judge-1",
    )
    completed = service.execute(experiment.id)

    assert completed.status == "completed"
    assert completed.metadata["knowledge_base_compatibility"]["wixqa_document_count"] == 1
    assert completed.metadata["knowledge_base_compatibility"]["eligible_record_counts"] == {
        "synthetic": 1
    }
    assert next(iter(fake_evaluation.runs.values())).limit == 1


def test_evaluation_rejects_kb_document_changes_after_experiment_creation(monkeypatch, tmp_path):
    dataset = tmp_path / "synthetic.jsonl"
    dataset.write_text(
        json.dumps({
            "id": "record-a",
            "question": "q",
            "answer": "a",
            "context": "c",
            "complexity_label": "moderate",
            "metadata": {"article_ids": ["article-a"]},
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        experiments_module,
        "DATASET_DEFINITIONS",
        {"synthetic": {"name": "WixQA Synthetic", "path": str(dataset), "count": 1}},
    )
    fake_knowledge = _FakeKnowledgeService(["article-a"])
    service = EvaluationExperimentService(
        JsonEvaluationExperimentRepository(str(tmp_path / "evaluation.json")),
        _FakeEvaluationService(),  # type: ignore[arg-type]
        fake_knowledge,  # type: ignore[arg-type]
    )
    knowledge_base = SimpleNamespace(
        id="kb-partial",
        name="Partial WixQA",
        document_count=1,
        metadata={"active_index_version_id": "index-1"},
    )
    experiment = service.create(
        name="Stable subset",
        knowledge_base=knowledge_base,
        configuration_snapshots=[{"id": "config-a", "name": "A", "metadata": {}}],
        datasets={"synthetic": 1},
        judge_deployment_id="judge-1",
    )

    fake_knowledge.article_ids.append("article-b")
    with pytest.raises(ValueError, match="changed after this experiment was created"):
        service.execute(experiment.id)
    assert service.get(experiment.id).status == "failed"
