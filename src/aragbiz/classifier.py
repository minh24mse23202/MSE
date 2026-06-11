from __future__ import annotations

import json
import math
import re
import inspect
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


class HuggingFaceQueryClassifier:
    """Runtime wrapper for a local Hugging Face sequence-classification artifact."""

    def __init__(self, model_dir: Union[str, Path], max_length: int = 128):
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self._runtime = None
        self.id2label = self._load_id2label()

    def predict(self, query: str) -> ComplexityLabel:
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
        label_id = int(output.logits.argmax(dim=-1).item())
        label = self.id2label.get(str(label_id), f"LABEL_{label_id}")
        if label not in COMPLEXITY_LABELS:
            raise ValueError(f"Hugging Face classifier returned unsupported label: {label!r}")
        return label  # type: ignore[return-value]

    def _load_runtime(self):
        if self._runtime is None:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "HuggingFaceQueryClassifier requires the optional ML dependencies. "
                    "Install them with: python -m pip install -e \".[ml]\""
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
        self._runtime = None

    def predict(self, query: str) -> ComplexityLabel:
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
        with torch.no_grad():
            generated_ids = model.generate(**model_inputs, max_length=self.generation_max_length)
        decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return _normalize_generated_label(decoded)

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
