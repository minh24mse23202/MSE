from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from aragbiz.classifier import QueryClassifier
from aragbiz.schemas import RAGStrategy


@dataclass(frozen=True)
class RouterConfig:
    simple_top_k: int = 2
    moderate_top_k: int = 4
    complex_top_k: int = 6


class AdaptiveRouter:
    def __init__(self, classifier: QueryClassifier, config: Optional[RouterConfig] = None):
        self.classifier = classifier
        self.config = config or RouterConfig()

    def route(self, query: str) -> RAGStrategy:
        label = self.classifier.predict(query)
        if label == "simple":
            return RAGStrategy(complexity_label=label, retrieval_mode="bm25", top_k=self.config.simple_top_k, multi_step=False)
        if label == "moderate":
            return RAGStrategy(complexity_label=label, retrieval_mode="hybrid", top_k=self.config.moderate_top_k, multi_step=False)
        return RAGStrategy(complexity_label=label, retrieval_mode="hybrid", top_k=self.config.complex_top_k, multi_step=True)
