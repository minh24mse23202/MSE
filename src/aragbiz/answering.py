from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from aragbiz.generation import Generator
from aragbiz.knowledge import KnowledgeBaseRecord, KnowledgeService, StoredKnowledgeChunk
from aragbiz.retrieval import InMemoryHybridRetriever
from aragbiz.routing import AdaptiveRouter
from aragbiz.schemas import AnswerResult, ComplexityLabel, Document, RetrievedContext, RetrievalMode

AnswerMode = Literal["adaptive", "direct", "simple_rag"]
RouteLevel = Literal["l1_direct", "l2_simple_rag"]


class AnsweringError(ValueError):
    """Raised when an answer request cannot be executed."""


@dataclass(frozen=True)
class AnswerOptions:
    mode: AnswerMode = "adaptive"
    knowledge_base_id: Optional[str] = None
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = 4


class AdaptiveRAGAnswerService:
    def __init__(
        self,
        router: AdaptiveRouter,
        generator: Generator,
        knowledge_service: KnowledgeService,
        bm25_weight: float = 0.65,
        dense_weight: float = 0.35,
    ):
        self.router = router
        self.generator = generator
        self.knowledge_service = knowledge_service
        self.retriever = KnowledgeBaseRetriever(
            knowledge_service=knowledge_service,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    def answer(self, query: str, options: Optional[AnswerOptions] = None) -> AnswerResult:
        options = options or AnswerOptions()
        start = time.perf_counter()
        top_k = max(1, min(int(options.top_k), 50))
        complexity_label = self.router.classifier.predict(query)
        route_level = self._resolve_route(options.mode, complexity_label)
        knowledge_base = self._selected_knowledge_base(options.knowledge_base_id)
        if options.mode == "adaptive" and knowledge_base is None:
            raise AnsweringError("Select a knowledge base before using Adaptive mode.")
        if route_level == "l2_simple_rag" and knowledge_base is None:
            raise AnsweringError("Select a knowledge base before using L2 Simple RAG.")

        trace_steps = [
            _trace_step("Chat input", "completed", query, {"characters": len(query)}),
            _trace_step(
                "Query complexity classifier",
                "completed",
                f"Predicted {complexity_label} query complexity.",
                {"complexity_label": complexity_label, "classifier": type(self.router.classifier).__name__},
            ),
            _trace_step(
                "Route decision",
                "completed",
                f"Selected {_route_label(route_level)} from requested mode {options.mode}.",
                {"requested_mode": options.mode, "route_level": route_level},
            ),
        ]

        if route_level == "l1_direct":
            answer = (
                "L1 Direct Generation is not configured yet. Select L2 Simple RAG with a knowledge base "
                "to answer from indexed workflow documents."
            )
            contexts: List[RetrievedContext] = []
            trace_steps.append(
                _trace_step(
                    "Direct generator adapter",
                    "not_configured",
                    "No external or local direct-generation provider is configured in this slice.",
                    {"retrieval_used": False},
                )
            )
            retrieval_mode: str = "none"
            retrieval_used = False
            generator_name = "direct-adapter-not-configured"
        else:
            assert knowledge_base is not None
            contexts = self.retriever.search(
                query=query,
                knowledge_base_id=knowledge_base.id,
                top_k=top_k,
                mode=options.retrieval_mode,
            )
            trace_steps.append(
                _trace_step(
                    "Knowledge base retrieval",
                    "completed" if contexts else "empty",
                    f"Retrieved {len(contexts)} chunk(s) from {knowledge_base.name}.",
                    {
                        "knowledge_base_id": knowledge_base.id,
                        "knowledge_base_name": knowledge_base.name,
                        "retrieval_mode": options.retrieval_mode,
                        "top_k": top_k,
                        "context_ids": [context.document.id for context in contexts],
                    },
                )
            )
            answer = self.generator.generate(query, contexts) if contexts else (
                f"No indexed chunks were found in knowledge base '{knowledge_base.name}'. "
                "Add documents or re-index the knowledge base before asking this question."
            )
            trace_steps.append(
                _trace_step(
                    "Extractive answer generator",
                    "completed" if contexts else "skipped",
                    "Generated an answer from retrieved workflow chunks." if contexts else "Skipped because retrieval returned no chunks.",
                    {"context_count": len(contexts)},
                )
            )
            retrieval_mode = options.retrieval_mode
            retrieval_used = True
            generator_name = "extractive"

        elapsed_ms = round((time.perf_counter() - start) * 1000, 3)
        metadata: Dict[str, Any] = {
            "requested_mode": options.mode,
            "route_level": route_level,
            "route_label": _route_label(route_level),
            "complexity_label": complexity_label,
            "retrieval_used": retrieval_used,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "multi_step": False,
            "latency_ms": elapsed_ms,
            "generator": generator_name,
            "trace_steps": trace_steps,
        }
        if knowledge_base is not None:
            metadata.update(
                {
                    "knowledge_base_id": knowledge_base.id,
                    "knowledge_base_name": knowledge_base.name,
                    "knowledge_base_status": knowledge_base.status,
                    "knowledge_base_chunk_count": knowledge_base.chunk_count,
                    "knowledge_base_document_count": knowledge_base.document_count,
                }
            )
        return AnswerResult(question=query, answer=answer, contexts=contexts, metadata=metadata)

    def _resolve_route(self, mode: AnswerMode, complexity_label: ComplexityLabel) -> RouteLevel:
        if mode == "direct":
            return "l1_direct"
        if mode == "simple_rag":
            return "l2_simple_rag"
        if mode == "adaptive":
            return "l1_direct" if complexity_label == "simple" else "l2_simple_rag"
        raise AnsweringError(f"Unsupported answer mode: {mode}")

    def _selected_knowledge_base(self, knowledge_base_id: Optional[str]) -> Optional[KnowledgeBaseRecord]:
        if not knowledge_base_id:
            return None
        return self.knowledge_service.get_knowledge_base(knowledge_base_id)


class KnowledgeBaseRetriever:
    def __init__(self, knowledge_service: KnowledgeService, bm25_weight: float = 0.65, dense_weight: float = 0.35):
        self.knowledge_service = knowledge_service
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def search(self, query: str, knowledge_base_id: str, top_k: int = 4, mode: RetrievalMode = "hybrid") -> List[RetrievedContext]:
        if mode == "bm25":
            return self._bm25_search(query, knowledge_base_id, top_k)
        if mode == "dense":
            return self._dense_search(query, knowledge_base_id, top_k)
        if mode == "hybrid":
            return self._hybrid_search(query, knowledge_base_id, top_k)
        raise AnsweringError(f"Unsupported retrieval mode: {mode}")

    def _bm25_search(self, query: str, knowledge_base_id: str, top_k: int) -> List[RetrievedContext]:
        chunks = self.knowledge_service.list_chunks(knowledge_base_id, limit=10000)
        if not chunks:
            return []
        retriever = InMemoryHybridRetriever(
            [_document_from_chunk(chunk) for chunk in chunks],
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
        )
        return retriever.search(query, top_k=min(top_k, len(chunks)), mode="bm25")

    def _dense_search(self, query: str, knowledge_base_id: str, top_k: int) -> List[RetrievedContext]:
        if not self.knowledge_service.list_chunks(knowledge_base_id, limit=1):
            return []
        query_embedding, embedding_model = self.knowledge_service.embed_query(knowledge_base_id, query)
        matches = self.knowledge_service.repository.search_chunks_by_embedding(
            knowledge_base_id,
            query_embedding,
            limit=top_k,
        )
        contexts: List[RetrievedContext] = []
        for rank, (chunk, score) in enumerate(matches, start=1):
            document = _document_from_chunk(chunk, {"query_embedding_model": embedding_model})
            contexts.append(RetrievedContext(document=document, score=score, rank=rank, mode="dense"))
        return contexts

    def _hybrid_search(self, query: str, knowledge_base_id: str, top_k: int) -> List[RetrievedContext]:
        candidate_limit = max(top_k * 4, 25)
        bm25_contexts = self._bm25_search(query, knowledge_base_id, candidate_limit)
        dense_contexts = self._dense_search(query, knowledge_base_id, candidate_limit)
        bm25_scores = {context.document.id: context.score for context in bm25_contexts}
        dense_scores = {context.document.id: context.score for context in dense_contexts}
        documents = {context.document.id: context.document for context in [*bm25_contexts, *dense_contexts]}
        normalized_bm25 = _normalize_scores(bm25_scores)
        normalized_dense = _normalize_scores(dense_scores)
        ranked = []
        for document_id, document in documents.items():
            score = self.bm25_weight * normalized_bm25.get(document_id, 0.0) + self.dense_weight * normalized_dense.get(document_id, 0.0)
            ranked.append((score, document))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedContext(document=document, score=score, rank=rank, mode="hybrid")
            for rank, (score, document) in enumerate(ranked[:top_k], start=1)
        ]


def _document_from_chunk(chunk: StoredKnowledgeChunk, extra_metadata: Optional[Dict[str, Any]] = None) -> Document:
    return Document(
        id=chunk.id,
        text=chunk.text,
        metadata={
            **chunk.metadata,
            **(extra_metadata or {}),
            "knowledge_base_id": chunk.knowledge_base_id,
            "document_id": chunk.document_id,
            "chunk_id": chunk.id,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "embedding_model": chunk.embedding_model,
            "embedding_dimension": chunk.embedding_dimension,
            "has_embedding": chunk.has_embedding,
        },
    )


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    minimum = min(scores.values())
    maximum = max(scores.values())
    if maximum == minimum:
        return {key: 1.0 if value > 0 else 0.0 for key, value in scores.items()}
    return {key: (value - minimum) / (maximum - minimum) for key, value in scores.items()}


def _route_label(route_level: RouteLevel) -> str:
    if route_level == "l1_direct":
        return "L1 Direct Generation"
    return "L2 Simple RAG"


def _trace_step(step: str, status: str, detail: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "step": step,
        "status": status,
        "detail": detail,
        "metadata": metadata or {},
    }