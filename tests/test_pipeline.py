from aragbiz.config import AppConfig
from aragbiz.data import load_qac_jsonl
from aragbiz.evaluation import Evaluator
from aragbiz.factory import build_sample_pipeline, existing_dataset_path


def test_pipeline_answers_sample_query():
    config = AppConfig(sample_dataset="data/sample/business_workflows.jsonl", kb_corpus="data/processed/missing_docs.jsonl")
    pipeline = build_sample_pipeline(config)
    result = pipeline.answer("How should we handle an invoice mismatch after goods are received?")
    assert result.answer
    assert result.contexts
    assert result.metadata["complexity_label"] == "complex"
    assert result.metadata["multi_step"] is True


def test_evaluator_returns_expected_metric_schema():
    config = AppConfig(sample_dataset="data/sample/business_workflows.jsonl", kb_corpus="data/processed/missing_docs.jsonl")
    dataset = load_qac_jsonl(existing_dataset_path(config))
    metrics = Evaluator(build_sample_pipeline(config)).evaluate(dataset)
    assert set(metrics) == {
        "routing_accuracy",
        "context_relevance",
        "faithfulness_proxy",
        "answer_overlap",
        "average_latency_ms",
    }
