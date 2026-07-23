from __future__ import annotations

import json
import math
import re
import inspect
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Protocol, Union

from aragbiz.schemas import COMPLEXITY_LABELS, ComplexityLabel, QACRecord


class QueryClassifier(Protocol):
    def predict(self, query: str) -> ComplexityLabel:
        """Return a query complexity label."""


@dataclass(frozen=True)
class ClassificationPrediction:
    label: ComplexityLabel
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    margin: float = 1.0
    supported_labels: List[str] = field(default_factory=lambda: list(COMPLEXITY_LABELS))
    latency_ms: float = 0.0


def predict_scored(classifier: QueryClassifier, query: str) -> ClassificationPrediction:
    started = time.perf_counter()
    scorer = getattr(classifier, "predict_scored", None)
    if callable(scorer):
        result = scorer(query)
        if isinstance(result, ClassificationPrediction):
            if result.latency_ms > 0:
                return result
            return ClassificationPrediction(
                result.label,
                result.probabilities,
                result.confidence,
                result.margin,
                result.supported_labels,
                round((time.perf_counter() - started) * 1000, 3),
            )
    label = classifier.predict(query)
    return _prediction_from_probabilities(
        {candidate: float(candidate == label) for candidate in COMPLEXITY_LABELS},
        supported_labels=list(COMPLEXITY_LABELS),
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
    )


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
    advanced_terms = {
        "cross-system",
        "investigate",
        "root-cause",
        "across",
        "exceptions",
        "alternatives",
    }

    def predict(self, query: str) -> ComplexityLabel:
        return self.predict_scored(query).label

    def predict_scored(self, query: str) -> ClassificationPrediction:
        tokens = set(_tokens(query))
        word_count = len(tokens)
        connector_count = len(re.findall(r"\b(?:and|then|before|after|while|unless|except|across)\b", query.lower()))
        if tokens & self.advanced_terms or connector_count >= 4 or word_count >= 28:
            label: ComplexityLabel = "advanced"
            probabilities = {"simple": 0.02, "moderate": 0.08, "complex": 0.25, "advanced": 0.65}
        elif tokens & self.complex_terms or word_count >= 14:
            label = "complex"
            probabilities = {"simple": 0.04, "moderate": 0.16, "complex": 0.68, "advanced": 0.12}
        elif tokens & self.moderate_terms or word_count >= 8:
            label = "moderate"
            probabilities = {"simple": 0.16, "moderate": 0.70, "complex": 0.11, "advanced": 0.03}
        else:
            label = "simple"
            probabilities = {"simple": 0.78, "moderate": 0.16, "complex": 0.05, "advanced": 0.01}
        return _prediction_from_probabilities(probabilities, preferred_label=label)


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
        return self.predict_scored(query).label

    def predict_scored(self, query: str) -> ClassificationPrediction:
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
        probabilities = _softmax_scores(scores)
        supported_labels = [
            label for label in COMPLEXITY_LABELS if label in self.label_log_priors
        ]
        return _prediction_from_probabilities(probabilities, supported_labels=supported_labels)

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


class HuggingFaceQueryClassifier:
    """Runtime wrapper for a local Hugging Face sequence-classification artifact."""

    def __init__(self, model_dir: Union[str, Path], max_length: int = 128):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self._runtime = None
        self.id2label = self._load_id2label()

    def predict(self, query: str) -> ComplexityLabel:
        return self.predict_scored(query).label

    def predict_scored(self, query: str) -> ClassificationPrediction:
        started = time.perf_counter()
        tokenizer, model, torch, accepted_inputs, device = self._load_runtime()
        encoded = tokenizer(
            query,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in accepted_inputs
        }
        with torch.no_grad():
            output = model(**model_inputs)
        values = output.logits[0].detach().cpu().tolist()
        labels = [self.id2label.get(str(index), f"LABEL_{index}").lower() for index in range(len(values))]
        invalid = [label for label in labels if label not in COMPLEXITY_LABELS]
        if invalid:
            raise ValueError(f"Hugging Face classifier returned unsupported labels: {invalid!r}")
        probabilities = _softmax_scores(dict(zip(labels, values)))
        return _prediction_from_probabilities(
            probabilities,
            supported_labels=labels,
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    def _load_runtime(self):
        if self._runtime is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "HuggingFaceQueryClassifier requires the optional ML dependencies. "
                    "Install them with: python -m pip install -e \".[dev,api,app,ml]\""
                ) from exc
            except Exception as exc:
                raise ImportError(
                    "HuggingFaceQueryClassifier could not initialize the optional ML runtime. "
                    "If this is a Windows PyTorch DLL error, reinstall CPU PyTorch with: "
                    "python -m pip install --force-reinstall \"torch>=2.2,<2.6\" --index-url https://download.pytorch.org/whl/cpu"
                ) from exc
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            accepted_inputs = set(inspect.signature(model.forward).parameters)
            self._runtime = (tokenizer, model, torch, accepted_inputs, device)
        return self._runtime

    def _load_id2label(self) -> Dict[str, str]:
        config_path = self.model_dir / "config.json"
        if not config_path.exists():
            return {}
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return {str(key): str(value) for key, value in payload.get("id2label", {}).items()}


class T5QueryClassifier:
    """Runtime wrapper for a local T5-style seq2seq classifier artifact."""

    def __init__(self, model_dir: Union[str, Path], max_length: int = 128, generation_max_length: int = 8):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.generation_max_length = generation_max_length
        self.supported_labels = self._load_supported_labels()
        self._runtime = None

    def predict(self, query: str) -> ComplexityLabel:
        return self.predict_scored(query).label

    def predict_scored(self, query: str) -> ClassificationPrediction:
        started = time.perf_counter()
        tokenizer, model, torch, accepted_inputs, device = self._load_runtime()
        encoded = tokenizer(
            _format_t5_input(query),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        model_inputs = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in accepted_inputs
        }
        scores: Dict[str, float] = {}
        with torch.no_grad():
            for label in self.supported_labels:
                target = tokenizer(
                    label,
                    truncation=True,
                    max_length=self.generation_max_length,
                    return_tensors="pt",
                )
                labels = target["input_ids"].to(device)
                output = model(**model_inputs, labels=labels)
                token_count = max(int(labels.ne(getattr(tokenizer, "pad_token_id", -1)).sum().item()), 1)
                scores[label] = -float(output.loss.item()) * token_count
        return _prediction_from_probabilities(
            _softmax_scores(scores),
            supported_labels=list(self.supported_labels),
            latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    def _load_supported_labels(self) -> List[str]:
        manifest_path = self.model_dir / "classifier_manifest.json"
        if manifest_path.exists():
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            labels = [str(label).lower() for label in payload.get("complexity_labels", [])]
            if labels and all(label in COMPLEXITY_LABELS for label in labels):
                return labels
        return ["simple", "moderate", "complex"]

    def _load_runtime(self):
        if self._runtime is None:
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "T5QueryClassifier requires the optional ML dependencies. "
                    "Install them with: python -m pip install -e \".[ml]\""
                ) from exc
            tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
            model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_dir))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            accepted_inputs = set(inspect.signature(model.forward).parameters)
            self._runtime = (tokenizer, model, torch, accepted_inputs, device)
        return self._runtime


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
    confidences: List[float] = []
    confidence_correctness: List[float] = []
    latencies: List[float] = []
    under_routed = 0
    over_routed = 0
    route_cost_regret = 0.0
    label_rank = {label: index for index, label in enumerate(COMPLEXITY_LABELS)}
    for record in records:
        prediction = predict_scored(classifier, record.question)
        predicted = prediction.label
        confusion[record.complexity_label][predicted] += 1
        correct += int(predicted == record.complexity_label)
        confidences.append(prediction.confidence)
        confidence_correctness.append(float(predicted == record.complexity_label))
        latencies.append(prediction.latency_ms)
        expected_rank = label_rank[record.complexity_label]
        predicted_rank = label_rank[predicted]
        under_routed += int(predicted_rank < expected_rank)
        over_routed += int(predicted_rank > expected_rank)
        route_cost_regret += abs(predicted_rank - expected_rank)
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
    advanced_precision, advanced_recall, advanced_f1 = _label_metrics(confusion, "advanced")
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "per_label_recall": per_label_recall,
        "advanced_precision": advanced_precision,
        "advanced_recall": advanced_recall,
        "advanced_f1": advanced_f1,
        "expected_calibration_error": _expected_calibration_error(confidences, confidence_correctness),
        "average_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "under_routing_rate": under_routed / total if total else 0.0,
        "over_routing_rate": over_routed / total if total else 0.0,
        "route_cost_regret": route_cost_regret / total if total else 0.0,
        "confusion_matrix": confusion,
        "total": total,
    }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _format_t5_input(query: str) -> str:
    return f"classify query complexity: {query}"


def _normalize_generated_label(text: str) -> ComplexityLabel:
    normalized = text.strip().lower()
    for label in COMPLEXITY_LABELS:
        if normalized == label or label in normalized.split():
            return label
    if "complex" in normalized:
        return "complex"
    if "moderate" in normalized:
        return "moderate"
    return "simple"


def _softmax_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    maximum = max(scores.values())
    exponentials = {label: math.exp(max(min(value - maximum, 60.0), -60.0)) for label, value in scores.items()}
    total = sum(exponentials.values()) or 1.0
    return {label: value / total for label, value in exponentials.items()}


def _prediction_from_probabilities(
    probabilities: Dict[str, float],
    *,
    preferred_label: ComplexityLabel | None = None,
    supported_labels: List[str] | None = None,
    latency_ms: float = 0.0,
) -> ClassificationPrediction:
    normalized = {label: max(float(probabilities.get(label, 0.0)), 0.0) for label in COMPLEXITY_LABELS}
    total = sum(normalized.values())
    if total <= 0:
        normalized = {label: float(label == (preferred_label or "simple")) for label in COMPLEXITY_LABELS}
    else:
        normalized = {label: value / total for label, value in normalized.items()}
    ordered = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    label = preferred_label or ordered[0][0]
    confidence = normalized[str(label)]
    second = ordered[1][1] if len(ordered) > 1 else 0.0
    return ClassificationPrediction(
        label=label,  # type: ignore[arg-type]
        probabilities={key: round(value, 8) for key, value in normalized.items()},
        confidence=round(confidence, 8),
        margin=round(max(confidence - second, 0.0), 8),
        supported_labels=list(supported_labels or COMPLEXITY_LABELS),
        latency_ms=latency_ms,
    )


def _label_metrics(confusion: Dict[str, Dict[str, int]], label: str) -> tuple[float, float, float]:
    true_positive = confusion[label][label]
    false_negative = sum(confusion[label].values()) - true_positive
    false_positive = sum(confusion[other][label] for other in confusion if other != label)
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _expected_calibration_error(confidences: List[float], correctness: List[float], bins: int = 10) -> float:
    if not confidences:
        return 0.0
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            item
            for item, value in enumerate(confidences)
            if lower <= value and (value <= upper if index == bins - 1 else value < upper)
        ]
        if not members:
            continue
        average_confidence = sum(confidences[item] for item in members) / len(members)
        average_accuracy = sum(correctness[item] for item in members) / len(members)
        error += (len(members) / len(confidences)) * abs(average_accuracy - average_confidence)
    return error
