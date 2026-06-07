from __future__ import annotations

from typing import List, Protocol

from aragbiz.schemas import RetrievedContext


class Generator(Protocol):
    def generate(self, query: str, contexts: List[RetrievedContext]) -> str:
        """Generate an answer from query and retrieved contexts."""


class ExtractiveGenerator:
    def __init__(self, max_context_chars: int = 900):
        self.max_context_chars = max_context_chars

    def generate(self, query: str, contexts: List[RetrievedContext]) -> str:
        if not contexts:
            return "I could not find enough workflow context to answer the question."
        best = contexts[0].document
        answer = best.metadata.get("answer")
        if isinstance(answer, str) and answer:
            return answer
        combined = " ".join(context.document.text for context in contexts)
        clipped = combined[: self.max_context_chars].strip()
        return f"Based on the retrieved workflow context: {clipped}"

