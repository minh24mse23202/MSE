from aragbiz.classifier import (
    NaiveBayesQueryClassifier,
    evaluate_classifier,
    train_naive_bayes_classifier,
)
from aragbiz.data import load_qac_jsonl


def test_naive_bayes_classifier_trains_saves_and_loads(tmp_path):
    records = load_qac_jsonl("data/sample/business_workflows.jsonl")
    classifier = train_naive_bayes_classifier(records)
    artifact_path = tmp_path / "classifier.json"
    classifier.save(artifact_path)

    loaded = NaiveBayesQueryClassifier.load(artifact_path)
    prediction = loaded.predict("How should finance resolve an invoice mismatch?")
    assert prediction in {"simple", "moderate", "complex"}


def test_classifier_evaluation_metrics_schema():
    records = load_qac_jsonl("data/sample/business_workflows.jsonl")
    classifier = train_naive_bayes_classifier(records)
    metrics = evaluate_classifier(records, classifier)
    assert metrics["total"] == len(records)
    assert "macro_f1" in metrics
    assert set(metrics["per_label_recall"]) == {"simple", "moderate", "complex"}
