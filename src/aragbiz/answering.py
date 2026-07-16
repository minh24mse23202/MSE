from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Tuple

from aragbiz.generation import (
    GenerationRequest,
    Generator,
    GeneratorConfigurationError,
    GeneratorExecutionError,
    GeneratorResolver,
    GeneratorResult,
    PromptBuildResult,
    PromptBuilder,
)
from aragbiz.knowledge import KnowledgeBaseRecord, KnowledgeService, StoredKnowledgeChunk
from aragbiz.model_farm import ModelCallContext, ModelFarmError, ModelFarmService, ModelGateway
from aragbiz.retrieval import InMemoryHybridRetriever
from aragbiz.routing import AdaptiveRouter
from aragbiz.schemas import AnswerResult, ComplexityLabel, Document, RetrievedContext, RetrievalMode

AnswerMode = Literal["adaptive", "direct", "simple_rag", "complex_rag"]
RouteLevel = Literal["l1_direct", "l2_simple_rag", "l3_complex_rag"]


class AnsweringError(ValueError):
    """Raised when an answer request cannot be executed."""


@dataclass(frozen=True)
class AnswerOptions:
    mode: AnswerMode = "adaptive"
    knowledge_base_id: Optional[str] = None
    document_ids: List[str] = field(default_factory=list)
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = 4
    chat_configuration: Optional[Dict[str, Any]] = None
    request_id: str = ""
    user_id: str = ""
    conversation_id: str = ""


@dataclass
class PreparedAnswer:
    query: str
    options: AnswerOptions
    start: float
    top_k: int
    chat_configuration: Dict[str, Any]
    complexity_label: ComplexityLabel
    route_level: RouteLevel
    knowledge_base: Optional[KnowledgeBaseRecord]
    document_ids: List[str]
    contexts: List[RetrievedContext]
    retrieval_mode: str
    retrieval_used: bool
    external_processing_allowed: bool
    decomposed_queries: List[str]
    retrieval_steps: List[Dict[str, Any]]
    aggregation_summary: Dict[str, Any]
    prompt: PromptBuildResult
    trace_steps: List[Dict[str, Any]]


@dataclass(frozen=True)
class AnswerStreamEvent:
    type: str
    data: Dict[str, Any]


class AdaptiveRAGAnswerService:
    def __init__(
        self,
        router: AdaptiveRouter,
        generator: Generator,
        knowledge_service: KnowledgeService,
        bm25_weight: float = 0.65,
        dense_weight: float = 0.35,
        prompt_builder: Optional[PromptBuilder] = None,
        generator_resolver: Optional[GeneratorResolver] = None,
        query_decomposer: Optional["QueryDecomposer"] = None,
        model_farm_service: Optional[ModelFarmService] = None,
        model_gateway: Optional[ModelGateway] = None,
    ):
        self.router = router
        self.generator = generator
        self.knowledge_service = knowledge_service
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.model_farm_service = model_farm_service
        self.model_gateway = model_gateway
        self.generator_resolver = generator_resolver or GeneratorResolver(
            generator if hasattr(generator, "max_context_chars") else None,
            model_farm_service=model_farm_service,
            model_gateway=model_gateway,
        )
        self.decomposer = query_decomposer or QueryDecomposer()
        self.retriever = KnowledgeBaseRetriever(
            knowledge_service=knowledge_service,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )

    def answer(self, query: str, options: Optional[AnswerOptions] = None) -> AnswerResult:
        prepared = self._prepare_answer(query, options or AnswerOptions())
        generation = self._generate_answer(prepared)
        return self._finalize_answer(prepared, generation)

    async def answer_stream(self, query: str, options: Optional[AnswerOptions] = None) -> AsyncIterator[AnswerStreamEvent]:
        prepared = await _to_thread(self._prepare_answer, query, options or AnswerOptions())
        for step in prepared.trace_steps:
            yield AnswerStreamEvent("trace", step)
        yield AnswerStreamEvent("sources", {"contexts": prepared.contexts})

        answer_parts: List[str] = []
        model_completed: Dict[str, Any] = {}
        try:
            async for event in self._stream_generation(prepared):
                if event.type == "delta":
                    answer_parts.append(str(event.data.get("text") or ""))
                elif event.type == "model_completed":
                    model_completed = event.data
                yield event
        except (GeneratorConfigurationError, GeneratorExecutionError, ModelFarmError) as exc:
            raise AnsweringError(str(exc)) from exc

        answer_text = "".join(answer_parts)
        generation = GeneratorResult(
            answer=answer_text,
            provider=str(model_completed.get("provider") or "Local"),
            model=str(model_completed.get("model") or prepared.chat_configuration.get("generator_model") or "extractive"),
            status=str(model_completed.get("status") or "completed"),
            prompt_preview=prepared.prompt.prompt_preview,
            input_chars=prepared.prompt.input_chars,
            output_chars=len(answer_text),
            metadata={
                **dict(model_completed.get("metadata") or {}),
                "deployment_id": model_completed.get("deployment_id", ""),
                "input_tokens": int(model_completed.get("input_tokens") or 0),
                "output_tokens": int(model_completed.get("output_tokens") or 0),
                "estimated_cost_usd": float(model_completed.get("estimated_cost_usd") or 0.0),
                "finish_reason": model_completed.get("finish_reason", ""),
            },
        )
        result = self._finalize_answer(prepared, generation)
        for step in result.metadata.get("trace_steps", [])[len(prepared.trace_steps) :]:
            yield AnswerStreamEvent("trace", step)
        yield AnswerStreamEvent("completed", {"result": result})

    def _prepare_answer(self, query: str, options: AnswerOptions) -> PreparedAnswer:
        options = options or AnswerOptions()
        start = time.perf_counter()
        top_k = max(1, min(int(options.top_k), 50))
        selected_document_ids = _normalize_document_ids(options.document_ids)
        chat_configuration = dict(options.chat_configuration or {})
        complexity_label = self.router.classifier.predict(query)
        route_level = self._resolve_route(options.mode, complexity_label)
        knowledge_base = self._selected_knowledge_base(options.knowledge_base_id)
        if options.mode == "adaptive" and knowledge_base is None:
            raise AnsweringError("Select a knowledge base before using Adaptive mode.")
        if route_level == "l2_simple_rag" and knowledge_base is None:
            raise AnsweringError("Select a knowledge base before using L2 Simple RAG.")
        if route_level == "l3_complex_rag" and knowledge_base is None:
            raise AnsweringError("Select a knowledge base before using L3 Complex RAG.")
        if route_level in {"l2_simple_rag", "l3_complex_rag"}:
            selected_document_ids = self._validate_document_filter(knowledge_base, selected_document_ids)
        else:
            selected_document_ids = []

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

        contexts: List[RetrievedContext] = []
        retrieval_mode: str = "none"
        retrieval_used = route_level in {"l2_simple_rag", "l3_complex_rag"}
        external_processing_allowed = _external_processing_allowed(knowledge_base)
        decomposed_queries: List[str] = []
        retrieval_steps: List[Dict[str, Any]] = []
        aggregation_summary: Dict[str, Any] = {}
        if route_level == "l2_simple_rag":
            assert knowledge_base is not None
            contexts = self.retriever.search(
                query=query,
                knowledge_base_id=knowledge_base.id,
                top_k=top_k,
                mode=options.retrieval_mode,
                document_ids=selected_document_ids,
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
                        "document_filter_ids": selected_document_ids,
                        "document_filter_count": len(selected_document_ids),
                        "context_ids": [context.document.id for context in contexts],
                    },
                )
            )
            retrieval_mode = options.retrieval_mode
        elif route_level == "l3_complex_rag":
            assert knowledge_base is not None
            decomposed_queries = self.decomposer.decompose(query)
            trace_steps.append(
                _trace_step(
                    "Query decomposition",
                    "completed",
                    f"Created {len(decomposed_queries)} deterministic retrieval subquery(s).",
                    {"decomposed_queries": decomposed_queries, "strategy": "deterministic_rules"},
                )
            )
            contexts, retrieval_steps, aggregation_summary = self._retrieve_multi_step(
                decomposed_queries=decomposed_queries,
                knowledge_base=knowledge_base,
                top_k=top_k,
                mode=options.retrieval_mode,
                document_ids=selected_document_ids,
            )
            trace_steps.append(
                _trace_step(
                    "Multi-step retrieval",
                    "completed" if aggregation_summary.get("candidate_count", 0) else "empty",
                    f"Ran {len(retrieval_steps)} retrieval step(s) against {knowledge_base.name}.",
                    {
                        "knowledge_base_id": knowledge_base.id,
                        "knowledge_base_name": knowledge_base.name,
                        "retrieval_mode": options.retrieval_mode,
                        "document_filter_ids": selected_document_ids,
                        "document_filter_count": len(selected_document_ids),
                        "retrieval_steps": retrieval_steps,
                    },
                )
            )
            trace_steps.append(
                _trace_step(
                    "Context aggregation",
                    "completed" if contexts else "empty",
                    f"Selected {len(contexts)} unique chunk(s) after deduplication and reranking.",
                    aggregation_summary,
                )
            )
            retrieval_mode = options.retrieval_mode

        contexts = _with_source_labels(contexts)
        reranker_deployment_id = str(chat_configuration.get("reranker_deployment_id") or "").strip()
        if contexts and reranker_deployment_id and self.model_gateway is not None:
            original_contexts = contexts
            try:
                reranked = self.model_gateway.rerank_sync(
                    query,
                    [context.document.text for context in contexts],
                    reranker_deployment_id,
                    top_n=min(top_k, len(contexts)),
                    context=ModelCallContext(
                        purpose="query_rerank",
                        request_id=options.request_id,
                        user_id=options.user_id,
                        conversation_id=options.conversation_id,
                        knowledge_base_id=knowledge_base.id if knowledge_base else "",
                    ),
                    external_processing_allowed=external_processing_allowed,
                )
                contexts = [
                    RetrievedContext(
                        document=original_contexts[item.index].document,
                        score=item.score,
                        rank=rank,
                        mode=original_contexts[item.index].mode,
                    )
                    for rank, item in enumerate(reranked.items, start=1)
                    if 0 <= item.index < len(original_contexts)
                ]
                contexts = _with_source_labels(contexts)
                trace_steps.append(
                    _trace_step(
                        "Context reranking",
                        "completed",
                        f"Reranked {len(contexts)} context chunk(s).",
                        {"deployment_id": reranker_deployment_id, **reranked.metadata},
                    )
                )
            except ModelFarmError as exc:
                trace_steps.append(
                    _trace_step(
                        "Context reranking",
                        "warning",
                        "Reranker failed open; original retrieval order was retained.",
                        {"deployment_id": reranker_deployment_id, "error": str(exc)},
                    )
                )

        prompt = self.prompt_builder.build(query, contexts, chat_configuration, route_level=route_level)
        trace_steps.append(
            _trace_step(
                "Prompt builder",
                "completed",
                f"Built generator prompt with {prompt.context_count} context chunk(s).",
                {**prompt.metadata, "input_chars": prompt.input_chars, "prompt_preview": prompt.prompt_preview},
            )
        )
        return PreparedAnswer(
            query=query,
            options=options,
            start=start,
            top_k=top_k,
            chat_configuration=chat_configuration,
            complexity_label=complexity_label,
            route_level=route_level,
            knowledge_base=knowledge_base,
            document_ids=selected_document_ids,
            contexts=contexts,
            retrieval_mode=retrieval_mode,
            retrieval_used=retrieval_used,
            external_processing_allowed=external_processing_allowed,
            decomposed_queries=decomposed_queries,
            retrieval_steps=retrieval_steps,
            aggregation_summary=aggregation_summary,
            prompt=prompt,
            trace_steps=trace_steps,
        )

    def _generation_request(self, prepared: PreparedAnswer) -> GenerationRequest:
        return GenerationRequest(
            query=prepared.query,
            contexts=prepared.contexts,
            chat_configuration=prepared.chat_configuration,
            prompt=prepared.prompt.prompt,
            prompt_preview=prepared.prompt.prompt_preview,
            input_chars=prepared.prompt.input_chars,
            route_level=prepared.route_level,
        )

    def _call_context(self, prepared: PreparedAnswer) -> ModelCallContext:
        return ModelCallContext(
            purpose="answer_generation",
            request_id=prepared.options.request_id,
            user_id=prepared.options.user_id,
            conversation_id=prepared.options.conversation_id,
            knowledge_base_id=prepared.knowledge_base.id if prepared.knowledge_base else "",
        )

    def _generate_answer(self, prepared: PreparedAnswer) -> GeneratorResult:
        try:
            generator = self.generator_resolver.resolve(
                prepared.chat_configuration,
                external_processing_allowed=prepared.external_processing_allowed,
                call_context=self._call_context(prepared),
            )
            return generator.generate(self._generation_request(prepared))
        except (GeneratorConfigurationError, GeneratorExecutionError) as exc:
            raise AnsweringError(str(exc)) from exc

    async def _stream_generation(self, prepared: PreparedAnswer) -> AsyncIterator[AnswerStreamEvent]:
        deployment_id = str(prepared.chat_configuration.get("generator_deployment_id") or "").strip()
        if deployment_id and self.model_gateway is not None:
            try:
                async for event in self.model_gateway.stream(
                    [{"role": "user", "content": prepared.prompt.prompt}],
                    deployment_id,
                    fallback_deployment_ids=list(prepared.chat_configuration.get("fallback_deployment_ids") or []),
                    parameters=dict(prepared.chat_configuration.get("generation_parameters") or {}),
                    context=self._call_context(prepared),
                    external_processing_allowed=prepared.external_processing_allowed,
                ):
                    yield AnswerStreamEvent(event.type, event.data)
                return
            except ModelFarmError as exc:
                raise GeneratorExecutionError(str(exc)) from exc
        generation = await _to_thread(self._generate_answer, prepared)
        for chunk in _text_chunks(generation.answer, 32):
            yield AnswerStreamEvent("delta", {"text": chunk})
        yield AnswerStreamEvent(
            "model_completed",
            {
                "provider": generation.provider,
                "model": generation.model,
                "status": generation.status,
                "metadata": generation.metadata,
                "input_tokens": generation.metadata.get("input_tokens", 0),
                "output_tokens": generation.metadata.get("output_tokens", 0),
                "estimated_cost_usd": generation.metadata.get("estimated_cost_usd", 0.0),
                "finish_reason": generation.metadata.get("finish_reason", ""),
            },
        )

    def _finalize_answer(self, prepared: PreparedAnswer, generation: GeneratorResult) -> AnswerResult:
        trace_steps = list(prepared.trace_steps)
        trace_steps.append(
            _trace_step(
                "Generator execution",
                generation.status,
                f"Executed {generation.provider}/{generation.model} generator.",
                {
                    "provider": generation.provider,
                    "model": generation.model,
                    "input_chars": generation.input_chars,
                    "output_chars": generation.output_chars,
                    **generation.metadata,
                },
            )
        )

        citation_validation = _validate_citations(generation.answer, prepared.contexts)
        trace_steps.append(
            _trace_step(
                "Citation validation",
                citation_validation["status"],
                citation_validation["detail"],
                citation_validation,
            )
        )

        elapsed_ms = round((time.perf_counter() - prepared.start) * 1000, 3)
        metadata: Dict[str, Any] = {
            "requested_mode": prepared.options.mode,
            "route_level": prepared.route_level,
            "route_label": _route_label(prepared.route_level),
            "complexity_label": prepared.complexity_label,
            "retrieval_used": prepared.retrieval_used,
            "retrieval_mode": prepared.retrieval_mode,
            "top_k": prepared.top_k,
            "document_filter_ids": prepared.document_ids,
            "document_filter_count": len(prepared.document_ids),
            "multi_step": prepared.route_level == "l3_complex_rag",
            "decomposed_queries": prepared.decomposed_queries,
            "retrieval_steps": prepared.retrieval_steps,
            "aggregation_summary": prepared.aggregation_summary,
            "latency_ms": elapsed_ms,
            "generator": generation.model,
            "configured_generator": {
                "provider": prepared.chat_configuration.get("generator_provider", "Local"),
                "model": prepared.chat_configuration.get("generator_model", "extractive"),
            },
            "actual_generator": {
                "provider": generation.provider,
                "model": generation.model,
            },
            "generation_status": generation.status,
            "prompt_preview": generation.prompt_preview,
            "input_chars": generation.input_chars,
            "output_chars": generation.output_chars,
            "chat_configuration": prepared.chat_configuration,
            "trace_steps": trace_steps,
            "request_id": prepared.options.request_id,
            "citation_validation": citation_validation,
            "external_processing_allowed": prepared.external_processing_allowed,
        }
        if prepared.knowledge_base is not None:
            metadata.update(
                {
                    "knowledge_base_id": prepared.knowledge_base.id,
                    "knowledge_base_name": prepared.knowledge_base.name,
                    "knowledge_base_status": prepared.knowledge_base.status,
                    "knowledge_base_chunk_count": prepared.knowledge_base.chunk_count,
                    "knowledge_base_document_count": prepared.knowledge_base.document_count,
                }
            )
        return AnswerResult(question=prepared.query, answer=generation.answer, contexts=prepared.contexts, metadata=metadata)

    def _retrieve_multi_step(
        self,
        decomposed_queries: List[str],
        knowledge_base: KnowledgeBaseRecord,
        top_k: int,
        mode: RetrievalMode,
        document_ids: Optional[List[str]] = None,
    ) -> Tuple[List[RetrievedContext], List[Dict[str, Any]], Dict[str, Any]]:
        per_step_top_k = max(2, min(top_k, 8))
        candidates: Dict[str, Dict[str, Any]] = {}
        retrieval_steps: List[Dict[str, Any]] = []
        total_retrieved = 0

        for subquery_index, subquery in enumerate(decomposed_queries, start=1):
            step_contexts = self.retriever.search(
                query=subquery,
                knowledge_base_id=knowledge_base.id,
                top_k=per_step_top_k,
                mode=mode,
                document_ids=document_ids,
            )
            total_retrieved += len(step_contexts)
            retrieval_steps.append(
                {
                    "retrieval_step": f"step-{subquery_index}",
                    "subquery_index": subquery_index,
                    "query": subquery,
                    "retrieved_count": len(step_contexts),
                    "document_filter_count": len(document_ids or []),
                    "context_ids": [context.document.id for context in step_contexts],
                }
            )
            for context in step_contexts:
                chunk_id = str(context.document.metadata.get("chunk_id") or context.document.id)
                rank = max(int(context.rank or 1), 1)
                score = float(context.score or 0.0)
                contribution = score / rank
                entry = candidates.setdefault(
                    chunk_id,
                    {
                        "best_context": context,
                        "best_contribution": contribution,
                        "best_subquery": subquery,
                        "best_subquery_index": subquery_index,
                        "best_original_rank": rank,
                        "combined_score": 0.0,
                        "coverage": set(),
                        "subquery_scores": [],
                    },
                )
                entry["combined_score"] += contribution
                entry["coverage"].add(subquery_index)
                entry["subquery_scores"].append(
                    {
                        "subquery_index": subquery_index,
                        "retrieval_step": f"step-{subquery_index}",
                        "rank": rank,
                        "score": round(score, 6),
                    }
                )
                if contribution > entry["best_contribution"]:
                    entry["best_context"] = context
                    entry["best_contribution"] = contribution
                    entry["best_subquery"] = subquery
                    entry["best_subquery_index"] = subquery_index
                    entry["best_original_rank"] = rank

        ranked_candidates: List[Dict[str, Any]] = []
        for entry in candidates.values():
            coverage_count = len(entry["coverage"])
            entry["aggregated_score"] = float(entry["combined_score"]) * (1.0 + 0.15 * max(coverage_count - 1, 0))
            ranked_candidates.append(entry)
        ranked_candidates.sort(
            key=lambda item: (item["aggregated_score"], len(item["coverage"]), -int(item["best_original_rank"])),
            reverse=True,
        )

        selected: List[RetrievedContext] = []
        for aggregated_rank, entry in enumerate(ranked_candidates[:top_k], start=1):
            best_context: RetrievedContext = entry["best_context"]
            metadata = dict(best_context.document.metadata)
            metadata.update(
                {
                    "source_subquery": entry["best_subquery"],
                    "subquery_index": entry["best_subquery_index"],
                    "retrieval_step": f"step-{entry['best_subquery_index']}",
                    "original_rank": entry["best_original_rank"],
                    "aggregated_rank": aggregated_rank,
                    "subquery_coverage": len(entry["coverage"]),
                    "matched_subquery_indexes": sorted(entry["coverage"]),
                    "subquery_scores": entry["subquery_scores"],
                    "aggregation_score": round(entry["aggregated_score"], 6),
                }
            )
            document = Document(id=best_context.document.id, text=best_context.document.text, metadata=metadata)
            selected.append(
                RetrievedContext(
                    document=document,
                    score=round(entry["aggregated_score"], 6),
                    rank=aggregated_rank,
                    mode=mode,
                )
            )

        aggregation_summary = {
            "subquery_count": len(decomposed_queries),
            "candidate_count": total_retrieved,
            "unique_context_count": len(candidates),
            "selected_context_count": len(selected),
            "deduplicated_count": max(total_retrieved - len(candidates), 0),
            "top_k": top_k,
            "per_step_top_k": per_step_top_k,
            "selected_context_ids": [context.document.id for context in selected],
        }
        return selected, retrieval_steps, aggregation_summary

    def _validate_document_filter(
        self,
        knowledge_base: Optional[KnowledgeBaseRecord],
        document_ids: List[str],
    ) -> List[str]:
        if not document_ids:
            return []
        if knowledge_base is None:
            return []
        available = {document.id for document in self.knowledge_service.list_documents(knowledge_base.id)}
        selected = [document_id for document_id in document_ids if document_id in available]
        if not selected:
            raise AnsweringError("Selected document filter does not match any document in the knowledge base.")
        return selected

    def _resolve_route(self, mode: AnswerMode, complexity_label: ComplexityLabel) -> RouteLevel:
        if mode == "direct":
            return "l1_direct"
        if mode == "simple_rag":
            return "l2_simple_rag"
        if mode == "complex_rag":
            return "l3_complex_rag"
        if mode == "adaptive":
            if complexity_label == "simple":
                return "l1_direct"
            if complexity_label == "moderate":
                return "l2_simple_rag"
            return "l3_complex_rag"
        raise AnsweringError(f"Unsupported answer mode: {mode}")

    def _selected_knowledge_base(self, knowledge_base_id: Optional[str]) -> Optional[KnowledgeBaseRecord]:
        if not knowledge_base_id:
            return None
        return self.knowledge_service.get_knowledge_base(knowledge_base_id)


class QueryDecomposer:
    _separator_pattern = re.compile(r"[?;\n]+")
    _connector_pattern = re.compile(
        r"\b(?:and then|then|after|before|when|if|while|where|also|plus|compare|versus|vs\.?|and)\b",
        re.IGNORECASE,
    )

    def decompose(self, query: str, max_subqueries: int = 4) -> List[str]:
        original = " ".join(query.split())
        if not original:
            return []
        subqueries: List[str] = []
        for fragment in self._fragments(original):
            cleaned = self._clean_fragment(fragment)
            if len(cleaned) < 12:
                continue
            if cleaned.lower() == original.lower():
                continue
            subqueries.append(self._as_search_query(cleaned))

        if len(subqueries) < 2:
            topic = self._topic(original)
            subqueries.extend(
                [
                    f"Find workflow steps and responsibilities for {topic}",
                    f"Find dependencies, exceptions, approvals, or follow-up actions for {topic}",
                ]
            )

        unique: List[str] = []
        seen = set()
        for subquery in subqueries:
            normalized = subquery.lower()
            if normalized == original.lower() or normalized in seen:
                continue
            unique.append(subquery)
            seen.add(normalized)
            if len(unique) >= max(max_subqueries - 1, 1):
                break
        unique.append(original)
        return unique[:max_subqueries]

    def _fragments(self, query: str) -> List[str]:
        fragments: List[str] = []
        for section in self._separator_pattern.split(query):
            fragments.extend(self._connector_pattern.split(section))
        return fragments

    def _clean_fragment(self, fragment: str) -> str:
        return " ".join(fragment.strip(" ,.:;!?-()").split())

    def _as_search_query(self, fragment: str) -> str:
        if fragment.lower().startswith(("what ", "how ", "why ", "when ", "who ", "which ")):
            return fragment
        return f"Find information about {fragment}"

    def _topic(self, query: str) -> str:
        words = re.findall(r"[A-Za-z0-9_/-]+", query)
        return " ".join(words[:12]) or query


class KnowledgeBaseRetriever:
    def __init__(self, knowledge_service: KnowledgeService, bm25_weight: float = 0.65, dense_weight: float = 0.35):
        self.knowledge_service = knowledge_service
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def search(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 4,
        mode: RetrievalMode = "hybrid",
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedContext]:
        document_ids = _normalize_document_ids(document_ids)
        if mode == "bm25":
            return self._bm25_search(query, knowledge_base_id, top_k, document_ids=document_ids)
        if mode == "dense":
            return self._dense_search(query, knowledge_base_id, top_k, document_ids=document_ids)
        if mode == "hybrid":
            return self._hybrid_search(query, knowledge_base_id, top_k, document_ids=document_ids)
        raise AnsweringError(f"Unsupported retrieval mode: {mode}")

    def _bm25_search(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedContext]:
        chunks = self._filtered_chunks(knowledge_base_id, document_ids)
        if not chunks:
            return []
        retriever = InMemoryHybridRetriever(
            [_document_from_chunk(chunk) for chunk in chunks],
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
        )
        return retriever.search(query, top_k=min(top_k, len(chunks)), mode="bm25")

    def _dense_search(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedContext]:
        if not self.knowledge_service.list_chunks(knowledge_base_id, limit=1):
            return []
        query_embedding, embedding_model = self.knowledge_service.embed_query(knowledge_base_id, query)
        document_id_set = set(_normalize_document_ids(document_ids))
        matches = self.knowledge_service.repository.search_chunks_by_embedding(
            knowledge_base_id,
            query_embedding,
            limit=10000 if document_id_set else top_k,
        )
        contexts: List[RetrievedContext] = []
        for chunk, score in matches:
            if document_id_set and chunk.document_id not in document_id_set:
                continue
            document = _document_from_chunk(chunk, {"query_embedding_model": embedding_model})
            contexts.append(RetrievedContext(document=document, score=score, rank=len(contexts) + 1, mode="dense"))
            if len(contexts) >= top_k:
                break
        return contexts

    def _hybrid_search(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedContext]:
        candidate_limit = max(top_k * 4, 25)
        bm25_contexts = self._bm25_search(query, knowledge_base_id, candidate_limit, document_ids=document_ids)
        dense_contexts = self._dense_search(query, knowledge_base_id, candidate_limit, document_ids=document_ids)
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

    def _filtered_chunks(self, knowledge_base_id: str, document_ids: Optional[List[str]] = None) -> List[StoredKnowledgeChunk]:
        chunks = self.knowledge_service.list_chunks(knowledge_base_id, limit=10000)
        document_id_set = set(_normalize_document_ids(document_ids))
        if not document_id_set:
            return chunks
        return [chunk for chunk in chunks if chunk.document_id in document_id_set]


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


def _normalize_document_ids(document_ids: Optional[List[str]]) -> List[str]:
    selected: List[str] = []
    seen: set[str] = set()
    for document_id in document_ids or []:
        cleaned = str(document_id or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        selected.append(cleaned)
    return selected


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
    if route_level == "l2_simple_rag":
        return "L2 Simple RAG"
    return "L3 Complex RAG"


def _trace_step(step: str, status: str, detail: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "step": step,
        "status": status,
        "detail": detail,
        "metadata": metadata or {},
    }


def _external_processing_allowed(knowledge_base: Optional[KnowledgeBaseRecord]) -> bool:
    if knowledge_base is None:
        return True
    configuration = knowledge_base.metadata.get("configuration") if isinstance(knowledge_base.metadata, dict) else {}
    return bool((configuration or {}).get("external_processing_allowed", False))


def _with_source_labels(contexts: List[RetrievedContext]) -> List[RetrievedContext]:
    labeled: List[RetrievedContext] = []
    for rank, context in enumerate(contexts, start=1):
        metadata = dict(context.document.metadata)
        metadata["source_label"] = f"S{rank}"
        labeled.append(
            RetrievedContext(
                document=Document(id=context.document.id, text=context.document.text, metadata=metadata),
                score=context.score,
                rank=rank,
                mode=context.mode,
            )
        )
    return labeled


def _validate_citations(answer: str, contexts: List[RetrievedContext]) -> Dict[str, Any]:
    valid_labels = {str(context.document.metadata.get("source_label") or f"S{context.rank}") for context in contexts}
    cited = set(re.findall(r"\[(S\d+)\]", answer or ""))
    invalid = sorted(cited - valid_labels)
    if not contexts:
        return {"status": "not_applicable", "detail": "No retrieval context required citations.", "cited": sorted(cited), "invalid": invalid}
    if invalid:
        return {"status": "warning", "detail": "The answer contains citations that do not match retrieved sources.", "cited": sorted(cited), "invalid": invalid, "available": sorted(valid_labels)}
    if not cited:
        return {"status": "warning", "detail": "Retrieved context was used, but the answer contains no source labels.", "cited": [], "invalid": [], "available": sorted(valid_labels)}
    return {"status": "completed", "detail": "All answer citations match retrieved sources.", "cited": sorted(cited), "invalid": [], "available": sorted(valid_labels)}


async def _to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


def _text_chunks(text: str, size: int) -> List[str]:
    return [text[index : index + max(size, 1)] for index in range(0, len(text), max(size, 1))]
