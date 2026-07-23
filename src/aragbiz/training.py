from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from aragbiz.schemas import COMPLEXITY_LABELS, QACRecord


def grouped_stratified_split(
    records: Sequence[QACRecord],
    validation_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[QACRecord], List[QACRecord], Dict[str, object]]:
    """Split records without placing records that share source documents in both sets."""
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    records = list(records)
    if len(records) < 2:
        raise ValueError("At least two records are required for a train/validation split.")

    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    owner_by_source: Dict[str, int] = {}
    for index, record in enumerate(records):
        for source_id in _source_ids(record):
            previous = owner_by_source.get(source_id)
            if previous is None:
                owner_by_source[source_id] = index
            else:
                union(index, previous)

    groups: Dict[int, List[QACRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[find(index)].append(record)

    groups_by_label: Dict[str, List[List[QACRecord]]] = defaultdict(list)
    for group in groups.values():
        label_counts = Counter(record.complexity_label for record in group)
        label = max(COMPLEXITY_LABELS, key=lambda item: (label_counts[item], -COMPLEXITY_LABELS.index(item)))
        groups_by_label[label].append(group)

    rng = random.Random(seed)
    validation_ids: set[str] = set()
    for label in COMPLEXITY_LABELS:
        label_groups = groups_by_label.get(label, [])
        rng.shuffle(label_groups)
        target = max(1, round(sum(len(group) for group in label_groups) * validation_ratio)) if label_groups else 0
        selected_count = 0
        for group in label_groups:
            if selected_count >= target and selected_count > 0:
                break
            validation_ids.update(record.id for record in group)
            selected_count += len(group)

    validation = [record for record in records if record.id in validation_ids]
    train = [record for record in records if record.id not in validation_ids]
    if not train:
        largest_validation_group = max(groups.values(), key=len)
        move_ids = {record.id for record in largest_validation_group}
        validation = [record for record in validation if record.id not in move_ids]
        train = [*train, *largest_validation_group]
    if not validation:
        validation = [train.pop()]

    validation_counts = Counter(record.complexity_label for record in validation)
    warnings = [
        f"Validation class {label!r} has {validation_counts[label]} examples; at least 50 are recommended."
        for label in COMPLEXITY_LABELS
        if validation_counts[label] < 50
    ]
    manifest: Dict[str, object] = {
        "strategy": "source-grouped-stratified",
        "seed": seed,
        "validation_ratio": validation_ratio,
        "group_count": len(groups),
        "train_records": len(train),
        "validation_records": len(validation),
        "train_class_counts": dict(Counter(record.complexity_label for record in train)),
        "validation_class_counts": dict(validation_counts),
        "warnings": warnings,
    }
    return train, validation, manifest


def audit_classifier_dataset(
    records: Sequence[QACRecord],
    validation_ratio: float = 0.2,
    seed: int = 42,
    minimum_validation_per_label: int = 50,
) -> Dict[str, object]:
    records = list(records)
    label_counts = Counter(record.complexity_label for record in records)
    id_counts = Counter(record.id for record in records)
    duplicate_record_ids = sorted(record_id for record_id, count in id_counts.items() if count > 1)

    records_by_question: Dict[str, List[QACRecord]] = defaultdict(list)
    for record in records:
        records_by_question[_normalize_question(record.question)].append(record)
    duplicate_question_groups = [
        {
            "question": group[0].question,
            "record_ids": [record.id for record in group],
            "labels": sorted({record.complexity_label for record in group}),
        }
        for group in records_by_question.values()
        if len(group) > 1
    ]
    conflicting_question_groups = [
        group for group in duplicate_question_groups if len(group["labels"]) > 1
    ]

    missing_labels = [label for label in COMPLEXITY_LABELS if label_counts[label] == 0]
    errors = []
    if len(records) < 2:
        errors.append("At least two records are required.")
    if missing_labels:
        errors.append(f"Missing complexity labels: {', '.join(missing_labels)}.")
    if duplicate_record_ids:
        errors.append(f"Duplicate record IDs were found: {len(duplicate_record_ids)}.")
    if conflicting_question_groups:
        errors.append(
            f"Questions with conflicting labels were found: {len(conflicting_question_groups)}."
        )

    split_manifest: Dict[str, object] = {}
    source_leakage: List[str] = []
    if len(records) >= 2 and not duplicate_record_ids:
        train, validation, split_manifest = grouped_stratified_split(
            records,
            validation_ratio=validation_ratio,
            seed=seed,
        )
        train_sources = {
            source_id for record in train for source_id in source_ids_for_record(record)
        }
        validation_sources = {
            source_id for record in validation for source_id in source_ids_for_record(record)
        }
        source_leakage = sorted(train_sources & validation_sources)
        if source_leakage:
            errors.append(f"Source leakage was found: {len(source_leakage)} shared sources.")

    validation_counts = Counter(split_manifest.get("validation_class_counts", {}))
    underrepresented_validation_labels = [
        label
        for label in COMPLEXITY_LABELS
        if validation_counts[label] < minimum_validation_per_label
    ]
    warnings = list(split_manifest.get("warnings", []))
    if duplicate_question_groups:
        warnings.append(
            f"Duplicate normalized questions were found: {len(duplicate_question_groups)} groups."
        )

    valid_for_training = not errors
    recommended_validation_size_met = not underrepresented_validation_labels
    return {
        "schema_version": 1,
        "records": len(records),
        "class_counts": {label: label_counts[label] for label in COMPLEXITY_LABELS},
        "class_percentages": {
            label: round(label_counts[label] / len(records), 6) if records else 0.0
            for label in COMPLEXITY_LABELS
        },
        "source_count": len(
            {
                source_id
                for record in records
                for source_id in source_ids_for_record(record)
            }
        ),
        "duplicate_record_ids": duplicate_record_ids[:100],
        "duplicate_record_id_count": len(duplicate_record_ids),
        "duplicate_question_groups": duplicate_question_groups[:100],
        "duplicate_question_group_count": len(duplicate_question_groups),
        "conflicting_question_groups": conflicting_question_groups[:100],
        "conflicting_question_group_count": len(conflicting_question_groups),
        "missing_labels": missing_labels,
        "split": split_manifest,
        "source_leakage": source_leakage[:100],
        "source_leakage_count": len(source_leakage),
        "minimum_validation_per_label": minimum_validation_per_label,
        "underrepresented_validation_labels": underrepresented_validation_labels,
        "valid_for_training": valid_for_training,
        "recommended_validation_size_met": recommended_validation_size_met,
        "ready_for_training": valid_for_training and recommended_validation_size_met,
        "warnings": warnings,
        "errors": errors,
    }


def build_source_balanced_four_class_dataset(
    expertwritten: Sequence[QACRecord],
    simulated: Sequence[QACRecord],
    official_synthetic: Sequence[QACRecord],
    template_records: Sequence[QACRecord],
    target_per_label: int = 600,
    official_synthetic_limit: int = 200,
    seed: int = 42,
) -> Tuple[List[QACRecord], Dict[str, object]]:
    if target_per_label < 1:
        raise ValueError("target_per_label must be positive.")
    if official_synthetic_limit < 1:
        raise ValueError("official_synthetic_limit must be positive.")

    rng = random.Random(seed)
    synthetic_candidates = list(official_synthetic)
    rng.shuffle(synthetic_candidates)
    prioritized_sources = [
        ("wixqa_expertwritten", list(expertwritten)),
        ("wixqa_simulated", list(simulated)),
        ("wixqa_synthetic", synthetic_candidates[:official_synthetic_limit]),
    ]

    selected: List[QACRecord] = []
    selected_questions: set[str] = set()
    selected_ids: set[str] = set()
    dropped_duplicates: Counter[str] = Counter()

    def append_unique(record: QACRecord, source_name: str) -> bool:
        normalized_question = _normalize_question(record.question)
        if record.id in selected_ids or normalized_question in selected_questions:
            dropped_duplicates[source_name] += 1
            return False
        selected.append(record)
        selected_ids.add(record.id)
        selected_questions.add(normalized_question)
        return True

    for source_name, source_records in prioritized_sources:
        for record in source_records:
            append_unique(record, source_name)

    selected_counts = Counter(record.complexity_label for record in selected)
    overflowing_labels = {
        label: selected_counts[label]
        for label in COMPLEXITY_LABELS
        if selected_counts[label] > target_per_label
    }
    if overflowing_labels:
        details = ", ".join(
            f"{label}={count}" for label, count in sorted(overflowing_labels.items())
        )
        raise ValueError(
            f"Official source selection exceeds target_per_label={target_per_label}: {details}."
        )

    templates_by_label: Dict[str, List[QACRecord]] = defaultdict(list)
    for record in template_records:
        templates_by_label[record.complexity_label].append(record)
    for label in COMPLEXITY_LABELS:
        candidates = templates_by_label[label]
        rng.shuffle(candidates)
        needed = target_per_label - selected_counts[label]
        if needed == 0:
            continue
        added = 0
        for record in candidates:
            if append_unique(record, "template_four_class"):
                added += 1
                if added == needed:
                    break
        if added < needed:
            raise ValueError(
                f"Insufficient unique template records for {label!r}: "
                f"needed {needed}, added {added}."
            )
        selected_counts[label] += added

    rng.shuffle(selected)
    source_counts = Counter(_dataset_source_name(record) for record in selected)
    source_class_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for record in selected:
        source_class_counts[_dataset_source_name(record)][record.complexity_label] += 1
    return selected, {
        "schema_version": 1,
        "seed": seed,
        "target_per_label": target_per_label,
        "official_synthetic_limit": official_synthetic_limit,
        "input_counts": {
            "wixqa_expertwritten": len(expertwritten),
            "wixqa_simulated": len(simulated),
            "wixqa_synthetic": len(official_synthetic),
            "template_four_class": len(template_records),
        },
        "selected_source_counts": dict(sorted(source_counts.items())),
        "selected_source_class_counts": {
            source: {
                label: counts[label]
                for label in COMPLEXITY_LABELS
                if counts[label]
            }
            for source, counts in sorted(source_class_counts.items())
        },
        "selected_class_counts": {
            label: selected_counts[label] for label in COMPLEXITY_LABELS
        },
        "dropped_duplicate_counts": dict(sorted(dropped_duplicates.items())),
        "records": len(selected),
    }


def _source_ids(record: QACRecord) -> Iterable[str]:
    return source_ids_for_record(record)


def source_ids_for_record(record: QACRecord) -> List[str]:
    metadata = record.metadata or {}
    values = metadata.get("article_ids") or metadata.get("document_ids") or metadata.get("source_ids") or []
    if isinstance(values, str):
        values = [values]
    source_ids = [str(value).strip() for value in values if str(value).strip()]
    if source_ids:
        return source_ids
    source = str(metadata.get("source_document_id") or metadata.get("source_id") or "").strip()
    return [source] if source else [f"record:{record.id}"]


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", str(question or "").strip().lower())


def _dataset_source_name(record: QACRecord) -> str:
    metadata = record.metadata or {}
    subset = str(metadata.get("subset") or "").strip()
    if subset:
        return subset
    return str(metadata.get("source") or "unknown").strip() or "unknown"
