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
