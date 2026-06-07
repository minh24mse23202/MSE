from __future__ import annotations

import math
import re
import hashlib
from collections import Counter
from typing import Dict, Iterable, List, Protocol

from aragbiz.schemas import Document, RetrievedContext, RetrievalMode


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 4, mode: RetrievalMode = "hybrid") -> List[RetrievedContext]:
        """Return ranked contexts for a query."""


class InMemoryHybridRetriever:
    def __init__(self, documents: Iterable[Document], bm25_weight: float = 0.65, dense_weight: float = 0.35):
        self.documents = list(documents)
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self._doc_tokens = {doc.id: _tokens(doc.text) for doc in self.documents}
        self._doc_freqs = self._build_doc_freqs()
        self._avg_doc_len = _safe_average(len(tokens) for tokens in self._doc_tokens.values())
        self._dense_vectors = {doc.id: _hashed_vector(doc.text) for doc in self.documents}

    def search(self, query: str, top_k: int = 4, mode: RetrievalMode = "hybrid") -> List[RetrievedContext]:
        if mode not in {"bm25", "dense", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {mode}")
        bm25_scores = self._bm25_scores(query)
        dense_scores = self._dense_scores(query)
        ranked = []
        for doc in self.documents:
            if mode == "bm25":
                score = bm25_scores[doc.id]
            elif mode == "dense":
                score = dense_scores[doc.id]
            else:
                score = self.bm25_weight * bm25_scores[doc.id] + self.dense_weight * dense_scores[doc.id]
            ranked.append((score, doc))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedContext(document=doc, score=score, rank=rank, mode=mode)
            for rank, (score, doc) in enumerate(ranked[:top_k], start=1)
        ]

    def _build_doc_freqs(self) -> Dict[str, int]:
        freqs: Dict[str, int] = {}
        for tokens in self._doc_tokens.values():
            for token in set(tokens):
                freqs[token] = freqs.get(token, 0) + 1
        return freqs

    def _bm25_scores(self, query: str) -> Dict[str, float]:
        query_terms = _tokens(query)
        total_docs = max(len(self.documents), 1)
        scores: Dict[str, float] = {}
        k1 = 1.5
        b = 0.75
        for doc in self.documents:
            tokens = self._doc_tokens[doc.id]
            counts = Counter(tokens)
            doc_len = max(len(tokens), 1)
            score = 0.0
            for term in query_terms:
                freq = counts[term]
                if freq == 0:
                    continue
                doc_freq = self._doc_freqs.get(term, 0)
                idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                denominator = freq + k1 * (1 - b + b * doc_len / max(self._avg_doc_len, 1))
                score += idf * (freq * (k1 + 1)) / denominator
            scores[doc.id] = score
        return _normalize(scores)

    def _dense_scores(self, query: str) -> Dict[str, float]:
        query_vector = _hashed_vector(query)
        return {
            doc.id: _cosine_similarity(query_vector, self._dense_vectors[doc.id])
            for doc in self.documents
        }


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _safe_average(values: Iterable[int]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    if math.isclose(max_score, min_score):
        return {key: 0.0 for key in scores}
    return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}


def _hashed_vector(text: str, dimensions: int = 128) -> List[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        vector[int(digest[:8], 16) % dimensions] += 1.0
    return vector


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
