from __future__ import annotations

import time
from typing import List

from aragbiz.generation import Generator
from aragbiz.retrieval import Retriever
from aragbiz.routing import AdaptiveRouter
from aragbiz.schemas import AnswerResult, RetrievedContext


class RAGPipeline:
    def __init__(self, router: AdaptiveRouter, retriever: Retriever, generator: Generator):
        self.router = router
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str) -> AnswerResult:
        start = time.perf_counter()
        strategy = self.router.route(query)
        contexts = self.retriever.search(query, top_k=strategy.top_k, mode=strategy.retrieval_mode)
        if strategy.multi_step:
            contexts = self._multi_step_expand(query, contexts, strategy.top_k)
        answer = self.generator.generate(query, contexts)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
        return AnswerResult(
            question=query,
            answer=answer,
            contexts=contexts,
            metadata={
                "complexity_label": strategy.complexity_label,
                "retrieval_mode": strategy.retrieval_mode,
                "top_k": strategy.top_k,
                "multi_step": strategy.multi_step,
                "latency_ms": elapsed_ms,
            },
        )

    def _multi_step_expand(self, query: str, initial_contexts: List[RetrievedContext], top_k: int) -> List[RetrievedContext]:
        expansion_terms = " ".join(context.document.metadata.get("question", "") for context in initial_contexts[:2])
        expanded_query = f"{query} {expansion_terms}".strip()
        expanded = self.retriever.search(expanded_query, top_k=top_k, mode="hybrid")
        by_id = {context.document.id: context for context in initial_contexts}
        for context in expanded:
            by_id.setdefault(context.document.id, context)
        return sorted(by_id.values(), key=lambda context: (context.rank, -context.score))[:top_k]

