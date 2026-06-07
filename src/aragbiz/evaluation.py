from __future__ import annotations

from statistics import mean
from typing import Dict, Iterable, List

from aragbiz.pipeline import RAGPipeline
from aragbiz.schemas import AnswerResult, QACRecord


class Evaluator:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def evaluate(self, dataset: Iterable[QACRecord]) -> Dict[str, float]:
        records = list(dataset)
        results = [self.pipeline.answer(record.question) for record in records]
        return evaluate_predictions(records, results)


def evaluate_predictions(records: List[QACRecord], results: List[AnswerResult]) -> Dict[str, float]:
    if len(records) != len(results):
        raise ValueError("records and results must have the same length")
    if not records:
        return {
            "routing_accuracy": 0.0,
            "context_relevance": 0.0,
            "faithfulness_proxy": 0.0,
            "answer_overlap": 0.0,
            "average_latency_ms": 0.0,
        }
    return {
        "routing_accuracy": mean(_routing_match(record, result) for record, result in zip(records, results)),
        "context_relevance": mean(_context_relevance(record, result) for record, result in zip(records, results)),
        "faithfulness_proxy": mean(_faithfulness_proxy(result) for result in results),
        "answer_overlap": mean(_answer_overlap(record.answer, result.answer) for record, result in zip(records, results)),
        "average_latency_ms": mean(float(result.metadata.get("latency_ms", 0.0)) for result in results),
    }


def _routing_match(record: QACRecord, result: AnswerResult) -> float:
    return float(record.complexity_label == result.metadata.get("complexity_label"))


def _context_relevance(record: QACRecord, result: AnswerResult) -> float:
    retrieved_ids = {context.document.id for context in result.contexts}
    article_ids = set(record.metadata.get("article_ids", []))
    if article_ids:
        return float(bool(article_ids & retrieved_ids))
    return float(record.id in retrieved_ids)


def _faithfulness_proxy(result: AnswerResult) -> float:
    context_text = " ".join(context.document.text for context in result.contexts).lower()
    answer_terms = set(_tokens(result.answer))
    if not answer_terms:
        return 0.0
    supported = {term for term in answer_terms if term in context_text}
    return len(supported) / len(answer_terms)


def _answer_overlap(expected: str, actual: str) -> float:
    expected_terms = set(_tokens(expected))
    actual_terms = set(_tokens(actual))
    if not expected_terms:
        return 0.0
    return len(expected_terms & actual_terms) / len(expected_terms)


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())
