from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal

ComplexityLabel = Literal["simple", "moderate", "complex"]
RetrievalMode = Literal["bm25", "dense", "hybrid"]

COMPLEXITY_LABELS: tuple[ComplexityLabel, ...] = ("simple", "moderate", "complex")


@dataclass(frozen=True)
class QACRecord:
    id: str
    question: str
    answer: str
    context: str
    complexity_label: ComplexityLabel
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievedContext:
    document: Document
    score: float
    rank: int
    mode: RetrievalMode


@dataclass(frozen=True)
class RAGStrategy:
    complexity_label: ComplexityLabel
    retrieval_mode: RetrievalMode
    top_k: int
    multi_step: bool


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    contexts: List[RetrievedContext]
    metadata: Dict[str, Any]

