from __future__ import annotations

import math
import re
from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence


DEFAULT_QUALITY_WEIGHTS: Dict[str, float] = {
    "context_recall": 0.30,
    "factuality": 0.30,
    "token_f1": 0.10,
    "bleu": 0.05,
    "rouge_1": 0.15,
    "rouge_2": 0.10,
}


def normalize_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def token_f1(reference: str, candidate: str) -> float:
    reference_tokens = normalize_tokens(reference)
    candidate_tokens = normalize_tokens(candidate)
    if not reference_tokens or not candidate_tokens:
        return float(reference_tokens == candidate_tokens)
    overlap = sum((Counter(reference_tokens) & Counter(candidate_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_n(reference: str, candidate: str, n: int) -> float:
    reference_ngrams = _ngrams(normalize_tokens(reference), n)
    candidate_ngrams = _ngrams(normalize_tokens(candidate), n)
    if not reference_ngrams or not candidate_ngrams:
        return float(reference_ngrams == candidate_ngrams)
    overlap = sum((Counter(reference_ngrams) & Counter(candidate_ngrams)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(candidate_ngrams)
    recall = overlap / len(reference_ngrams)
    return 2 * precision * recall / (precision + recall)


def sentence_bleu(reference: str, candidate: str, max_order: int = 4) -> float:
    return corpus_bleu([reference], [candidate], max_order=max_order)


def corpus_bleu(
    references: Sequence[str],
    candidates: Sequence[str],
    *,
    max_order: int = 4,
) -> float:
    if len(references) != len(candidates):
        raise ValueError("references and candidates must have the same length")
    if not references:
        return 0.0
    matches = [0] * max_order
    possible = [0] * max_order
    reference_length = 0
    candidate_length = 0
    for reference, candidate in zip(references, candidates):
        reference_tokens = normalize_tokens(reference)
        candidate_tokens = normalize_tokens(candidate)
        reference_length += len(reference_tokens)
        candidate_length += len(candidate_tokens)
        for order in range(1, max_order + 1):
            reference_ngrams = Counter(_ngrams(reference_tokens, order))
            candidate_ngrams = Counter(_ngrams(candidate_tokens, order))
            matches[order - 1] += sum((reference_ngrams & candidate_ngrams).values())
            possible[order - 1] += max(len(candidate_tokens) - order + 1, 0)
    if candidate_length == 0:
        return 0.0
    # Add-one smoothing keeps short enterprise answers comparable without returning
    # zero solely because a higher-order n-gram is unavailable.
    precisions = [
        (matches[index] + 1.0) / (possible[index] + 1.0)
        for index in range(max_order)
    ]
    geo_mean = math.exp(sum(math.log(value) for value in precisions) / max_order)
    brevity_penalty = 1.0
    if candidate_length < reference_length:
        brevity_penalty = math.exp(1.0 - reference_length / max(candidate_length, 1))
    return brevity_penalty * geo_mean


def deterministic_case_metrics(reference: str, candidate: str) -> Dict[str, float]:
    return {
        "token_f1": token_f1(reference, candidate),
        "bleu": sentence_bleu(reference, candidate),
        "rouge_1": rouge_n(reference, candidate, 1),
        "rouge_2": rouge_n(reference, candidate, 2),
    }


def aggregate_wixqa_metrics(case_metrics: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    rows = list(case_metrics)
    if not rows:
        return {
            "token_f1": 0.0,
            "bleu": 0.0,
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "context_recall": 0.0,
            "factuality": 0.0,
            "judge_coverage": 0.0,
            "retrieval_coverage": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
        }
    deterministic = [
        dict(row.get("wixqa") or row.get("adaptive", {}).get("wixqa") or {})
        for row in rows
    ]
    references = [str(row.get("reference_answer") or "") for row in deterministic]
    candidates = [str(row.get("candidate_answer") or "") for row in deterministic]
    judged = [
        row for row in deterministic
        if _is_number(row.get("context_recall")) and _is_number(row.get("factuality"))
    ]
    retrieval_rows = [
        dict(row.get("retrieval") or {})
        for row in deterministic
        if isinstance(row.get("retrieval"), dict) and row["retrieval"].get("available")
    ]
    result = {
        "token_f1": mean(float(row.get("token_f1", 0.0)) for row in deterministic),
        "bleu": corpus_bleu(references, candidates),
        "rouge_1": mean(float(row.get("rouge_1", 0.0)) for row in deterministic),
        "rouge_2": mean(float(row.get("rouge_2", 0.0)) for row in deterministic),
        "context_recall": mean(float(row["context_recall"]) for row in judged) if judged else 0.0,
        "factuality": mean(float(row["factuality"]) for row in judged) if judged else 0.0,
        "judge_coverage": len(judged) / len(deterministic),
        "retrieval_coverage": len(retrieval_rows) / len(deterministic),
    }
    for metric in ("precision_at_k", "recall_at_k", "hit_at_k", "mrr", "ndcg_at_k"):
        result[metric] = mean(float(row.get(metric, 0.0)) for row in retrieval_rows) if retrieval_rows else 0.0
    return result


def quality_score(
    metrics: Dict[str, Any],
    weights: Dict[str, float] | None = None,
) -> float:
    selected = dict(weights or DEFAULT_QUALITY_WEIGHTS)
    total = sum(float(value) for value in selected.values())
    if total <= 0:
        raise ValueError("Quality metric weights must total more than zero.")
    return sum(
        float(metrics.get(metric, 0.0)) * float(weight) / total
        for metric, weight in selected.items()
    )


def validate_quality_weights(weights: Dict[str, float]) -> Dict[str, float]:
    unknown = set(weights) - set(DEFAULT_QUALITY_WEIGHTS)
    if unknown:
        raise ValueError(f"Unsupported quality metrics: {', '.join(sorted(unknown))}")
    normalized = {
        metric: float(weights.get(metric, 0.0))
        for metric in DEFAULT_QUALITY_WEIGHTS
    }
    if any(value < 0 for value in normalized.values()):
        raise ValueError("Quality metric weights cannot be negative.")
    total = sum(normalized.values())
    if not math.isclose(total, 1.0, abs_tol=0.001):
        raise ValueError("Quality metric weights must total 1.0.")
    return normalized


def _ngrams(tokens: Sequence[str], n: int) -> List[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
