from __future__ import annotations

import re
from typing import Protocol

from aragbiz.schemas import ComplexityLabel


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


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
