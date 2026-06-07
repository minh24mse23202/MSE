import json
import builtins

import pytest

from aragbiz.classifier import HuggingFaceQueryClassifier
from aragbiz.config import AppConfig
from aragbiz.factory import build_query_classifier
from scripts.train_hf_query_classifier import make_trainer


def test_hf_classifier_reads_id2label_config(tmp_path):
    model_dir = tmp_path / "hf_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"id2label": {"0": "simple", "1": "moderate", "2": "complex"}}),
        encoding="utf-8",
    )
    classifier = HuggingFaceQueryClassifier(model_dir)
    assert classifier.id2label["2"] == "complex"


def test_factory_prefers_hf_directory_artifact(tmp_path):
    model_dir = tmp_path / "hf_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"id2label": {"0": "simple", "1": "moderate", "2": "complex"}}),
        encoding="utf-8",
    )
    config = AppConfig(classifier_model_path=str(model_dir), classifier_fallback_model_path=str(tmp_path / "missing.json"))
    classifier = build_query_classifier(config)
    assert isinstance(classifier, HuggingFaceQueryClassifier)


def test_hf_classifier_has_clear_missing_dependency_error(tmp_path, monkeypatch):
    model_dir = tmp_path / "hf_model"
    model_dir.mkdir()
    classifier = HuggingFaceQueryClassifier(model_dir)

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("No module named transformers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="optional ML dependencies"):
        classifier.predict("How do I approve a purchase order?")


def test_make_trainer_supports_processing_class_signature():
    captured = {}

    class FakeTrainer:
        def __init__(
            self,
            model,
            args,
            train_dataset,
            eval_dataset,
            compute_metrics,
            processing_class=None,
        ):
            captured["processing_class"] = processing_class

    trainer = make_trainer(
        FakeTrainer,
        model=object(),
        training_args=object(),
        train_dataset=[],
        validation_dataset=[],
        tokenizer="tokenizer",
        compute_metrics=lambda output: {},
    )
    assert isinstance(trainer, FakeTrainer)
    assert captured["processing_class"] == "tokenizer"
