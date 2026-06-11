import json

from aragbiz.classifier import T5QueryClassifier
from aragbiz.config import AppConfig
from aragbiz.factory import build_query_classifier
from scripts.train_t5_query_classifier import format_t5_input


def test_t5_classifier_artifact_detection(tmp_path):
    model_dir = tmp_path / "t5_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"model_type": "t5"}), encoding="utf-8")
    config = AppConfig(classifier_model_path=str(model_dir), classifier_fallback_model_path=str(tmp_path / "missing.json"))
    classifier = build_query_classifier(config)
    assert isinstance(classifier, T5QueryClassifier)


def test_t5_input_format_is_stable():
    assert format_t5_input("How do I approve a PO?") == "classify query complexity: How do I approve a PO?"
