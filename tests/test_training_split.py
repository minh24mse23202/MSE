from dataclasses import replace

from aragbiz.schemas import QACRecord
from aragbiz.training import (
    audit_classifier_dataset,
    build_source_balanced_four_class_dataset,
    grouped_stratified_split,
)


def _record(index, label, source):
    return QACRecord(
        id=f"record-{index}",
        question=f"Question {index}",
        answer="Answer",
        context="Context",
        complexity_label=label,
        metadata={"article_ids": [source]},
    )


def test_grouped_split_prevents_source_leakage_and_reports_small_classes():
    records = []
    labels = ["simple", "moderate", "complex", "advanced"]
    for index in range(24):
        records.append(_record(index, labels[index % 4], f"source-{index // 2}"))

    train, validation, manifest = grouped_stratified_split(records, validation_ratio=0.25, seed=7)
    train_sources = {source for record in train for source in record.metadata["article_ids"]}
    validation_sources = {source for record in validation for source in record.metadata["article_ids"]}

    assert train
    assert validation
    assert train_sources.isdisjoint(validation_sources)
    assert manifest["strategy"] == "source-grouped-stratified"
    assert manifest["warnings"]


def test_classifier_dataset_audit_reports_training_readiness():
    labels = ["simple", "moderate", "complex", "advanced"]
    records = [
        _record(index, labels[index % 4], f"unique-source-{index}")
        for index in range(40)
    ]

    report = audit_classifier_dataset(
        records,
        validation_ratio=0.25,
        seed=7,
        minimum_validation_per_label=1,
    )

    assert report["ready_for_training"] is True
    assert report["source_leakage_count"] == 0
    assert report["class_counts"] == {
        "simple": 10,
        "moderate": 10,
        "complex": 10,
        "advanced": 10,
    }


def test_classifier_dataset_audit_rejects_conflicting_duplicate_questions():
    records = [
        _record(1, "simple", "source-1"),
        _record(2, "moderate", "source-2"),
        _record(3, "complex", "source-3"),
        _record(4, "advanced", "source-4"),
    ]
    records[1] = replace(records[1], question=f"  {records[0].question.upper()}  ")

    report = audit_classifier_dataset(
        records,
        validation_ratio=0.5,
        minimum_validation_per_label=0,
    )

    assert report["valid_for_training"] is False
    assert report["conflicting_question_group_count"] == 1


def test_source_balanced_builder_uses_all_official_sources_and_fills_with_templates():
    expertwritten = [
        _source_record(1, "moderate", "expert-1", "wixqa_expertwritten"),
        _source_record(2, "complex", "expert-2", "wixqa_expertwritten"),
    ]
    simulated = [
        _source_record(3, "moderate", "simulated-1", "wixqa_simulated"),
        _source_record(4, "complex", "simulated-2", "wixqa_simulated"),
    ]
    official_synthetic = [
        _source_record(index, "moderate", f"synthetic-{index}", "wixqa_synthetic")
        for index in range(5, 10)
    ]
    templates = [
        _source_record(
            100 + index,
            label,
            f"template-{label}-{index}",
            "template_four_class",
        )
        for label in ("simple", "moderate", "complex", "advanced")
        for index in range(3)
    ]

    records, manifest = build_source_balanced_four_class_dataset(
        expertwritten,
        simulated,
        official_synthetic,
        templates,
        target_per_label=3,
        official_synthetic_limit=1,
        seed=7,
    )

    assert manifest["selected_class_counts"] == {
        "simple": 3,
        "moderate": 3,
        "complex": 3,
        "advanced": 3,
    }
    assert manifest["selected_source_counts"] == {
        "template_four_class": 7,
        "wixqa_expertwritten": 2,
        "wixqa_simulated": 2,
        "wixqa_synthetic": 1,
    }
    assert manifest["selected_source_class_counts"]["wixqa_synthetic"] == {
        "moderate": 1
    }
    assert len(records) == 12


def _source_record(index, label, source, subset):
    return QACRecord(
        id=f"{subset}-{source}-{index}",
        question=f"{subset} {source} question {index}",
        answer="Answer",
        context="Context",
        complexity_label=label,
        metadata={"article_ids": [source], "subset": subset},
    )
