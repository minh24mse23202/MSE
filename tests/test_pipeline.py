from aragbiz.config import AppConfig
from aragbiz.data import load_qac_jsonl
from aragbiz.evaluation import Evaluator
from aragbiz.factory import build_sample_pipeline


def test_pipeline_answers_sample_query():
    pipeline = build_sample_pipeline(AppConfig())
    result = pipeline.answer("How should we handle an invoice mismatch after goods are received?")
    assert result.answer
    assert result.contexts
    assert result.metadata["complexity_label"] == "complex"
    assert result.metadata["multi_step"] is True


def test_evaluator_returns_expected_metric_schema():
    config = AppConfig()
    dataset = load_qac_jsonl(config.sample_dataset)
    metrics = Evaluator(build_sample_pipeline(config)).evaluate(dataset)
    assert set(metrics) == {
        "routing_accuracy",
        "context_relevance",
        "faithfulness_proxy",
        "answer_overlap",
        "average_latency_ms",
    }

