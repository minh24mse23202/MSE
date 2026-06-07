from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Protocol, Union

from aragbiz.schemas import COMPLEXITY_LABELS, ComplexityLabel, QACRecord


class QueryClassifier(Protocol):
    def predict(self, query: str) -> ComplexityLabel:
        """Return a query complexity label."""


class HeuristicQueryClassifier:
    """Deterministic baseline classifier for early routing experiments."""

    complex_terms = {
        "before",
        "depends",
        "dependency",
        "mismatch",
        "multi-step",
        "multiple",
        "resolve",
        "when",
        "while",
    }
    moderate_terms = {
        "next",
        "happens",
        "process",
        "review",
        "reject",
        "if",
        "then",
    }

    def predict(self, query: str) -> ComplexityLabel:
        tokens = set(_tokens(query))
        word_count = len(tokens)
        if tokens & self.complex_terms or word_count >= 14:
            return "complex"
        if tokens & self.moderate_terms or word_count >= 8:
            return "moderate"
        return "simple"


class NaiveBayesQueryClassifier:
    """Small supervised text classifier saved as a portable JSON artifact."""

    def __init__(
        self,
        label_log_priors: Dict[str, float],
        token_log_likelihoods: Dict[str, Dict[str, float]],
        unknown_log_likelihoods: Dict[str, float],
    ):
        self.label_log_priors = label_log_priors
        self.token_log_likelihoods = token_log_likelihoods
        self.unknown_log_likelihoods = unknown_log_likelihoods

    def predict(self, query: str) -> ComplexityLabel:
        tokens = _tokens(query)
        scores = {}
        for label in COMPLEXITY_LABELS:
            label_key = str(label)
            score = self.label_log_priors.get(label_key, float("-inf"))
            likelihoods = self.token_log_likelihoods.get(label_key, {})
            unknown = self.unknown_log_likelihoods.get(label_key, -20.0)
            for token in tokens:
                score += likelihoods.get(token, unknown)
            scores[label_key] = score
        best_label = max(scores, key=scores.get)
        return best_label  # type: ignore[return-value]

    def save(self, path: Union[str, Path]) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "model_type": "multinomial_naive_bayes",
                    "labels": list(COMPLEXITY_LABELS),
                    "label_log_priors": self.label_log_priors,
                    "token_log_likelihoods": self.token_log_likelihoods,
                    "unknown_log_likelihoods": self.unknown_log_likelihoods,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "NaiveBayesQueryClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_type") != "multinomial_naive_bayes":
            raise ValueError(f"Unsupported classifier artifact: {payload.get('model_type')}")
        return cls(
            label_log_priors={str(key): float(value) for key, value in payload["label_log_priors"].items()},
            token_log_likelihoods={
                str(label): {str(token): float(value) for token, value in token_scores.items()}
                for label, token_scores in payload["token_log_likelihoods"].items()
            },
            unknown_log_likelihoods={
                str(key): float(value) for key, value in payload["unknown_log_likelihoods"].items()
            },
        )


def train_naive_bayes_classifier(records: Iterable[QACRecord], alpha: float = 1.0) -> NaiveBayesQueryClassifier:
    records = list(records)
    if not records:
        raise ValueError("Cannot train classifier without records")

    label_counts = Counter(record.complexity_label for record in records)
    token_counts: Dict[str, Counter] = defaultdict(Counter)
    total_tokens: Counter = Counter()
    vocabulary = set()
    for record in records:
        label = record.complexity_label
        tokens = _tokens(record.question)
        token_counts[label].update(tokens)
        total_tokens[label] += len(tokens)
        vocabulary.update(tokens)

    vocab_size = max(len(vocabulary), 1)
    label_log_priors = {
        label: math.log((label_counts[label] + alpha) / (len(records) + alpha * len(COMPLEXITY_LABELS)))
        for label in COMPLEXITY_LABELS
    }
    token_log_likelihoods: Dict[str, Dict[str, float]] = {}
    unknown_log_likelihoods: Dict[str, float] = {}
    for label in COMPLEXITY_LABELS:
        denominator = total_tokens[label] + alpha * (vocab_size + 1)
        unknown_log_likelihoods[label] = math.log(alpha / denominator)
        token_log_likelihoods[label] = {
            token: math.log((token_counts[label][token] + alpha) / denominator)
            for token in vocabulary
        }
    return NaiveBayesQueryClassifier(label_log_priors, token_log_likelihoods, unknown_log_likelihoods)


def evaluate_classifier(records: Iterable[QACRecord], classifier: QueryClassifier) -> Dict[str, object]:
    records = list(records)
    confusion = {
        expected: {predicted: 0 for predicted in COMPLEXITY_LABELS}
        for expected in COMPLEXITY_LABELS
    }
    correct = 0
    for record in records:
        predicted = classifier.predict(record.question)
        confusion[record.complexity_label][predicted] += 1
        correct += int(predicted == record.complexity_label)
    total = len(records)
    per_label_recall = {}
    f1_scores: List[float] = []
    for label in COMPLEXITY_LABELS:
        true_positive = confusion[label][label]
        false_negative = sum(confusion[label].values()) - true_positive
        false_positive = sum(confusion[other][label] for other in COMPLEXITY_LABELS if other != label)
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label_recall[label] = recall
        f1_scores.append(f1)
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "per_label_recall": per_label_recall,
        "confusion_matrix": confusion,
        "total": total,
    }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
