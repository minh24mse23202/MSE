from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
import json
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Tuple

from aragbiz.cancellation import AnswerCancelled, CancellationToken
from aragbiz.conversation import (
    DEFAULT_HISTORY_MAX_CHARACTERS,
    DEFAULT_HISTORY_MAX_EXCHANGES,
    QueryReformulationResult,
    QueryReformulator,
    conversation_history_characters,
    conversation_history_exchange_count,
    normalize_conversation_history,
)
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
from aragbiz.tracing import SpanHandle, TraceRecorder

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
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    conversation_history_max_exchanges: int = DEFAULT_HISTORY_MAX_EXCHANGES
    conversation_history_max_characters: int = DEFAULT_HISTORY_MAX_CHARACTERS
    cancellation_token: Optional[CancellationToken] = None
    trace_recorder: Optional[TraceRecorder] = None


@dataclass(frozen=True)
class QueryClassificationExecution:
    label: ComplexityLabel
    configured: Dict[str, Any]
    actual: Dict[str, Any]
    fallback_used: bool = False
    warning: str = ""


@dataclass(frozen=True)
class QueryDecompositionResult:
    queries: List[str]
    strategy: str
    configured: Dict[str, Any] = field(default_factory=dict)
    actual: Dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False
    warning: str = ""


@dataclass
class PreparedAnswer:
    query: str
    standalone_query: str
    options: AnswerOptions
    start: float
    top_k: int
    chat_configuration: Dict[str, Any]
    conversation_awareness_enabled: bool
    conversation_history: List[Dict[str, str]]
    conversation_history_max_exchanges: int
    conversation_history_max_characters: int
    reformulation: QueryReformulationResult
    classification: QueryClassificationExecution
    complexity_label: ComplexityLabel
    route_level: RouteLevel
    knowledge_base: Optional[KnowledgeBaseRecord]
    document_ids: List[str]
    contexts: List[RetrievedContext]
    retrieval_mode: str
    retrieval_used: bool
    external_processing_allowed: bool
    decomposed_queries: List[str]
    decomposition: QueryDecompositionResult
    retrieval_steps: List[Dict[str, Any]]
    aggregation_summary: Dict[str, Any]
    retrieval_diagnostics: Dict[str, Any]
    query_embedding: Dict[str, Any]
    citations_enabled: bool
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
        query_reformulator: Optional[QueryReformulator] = None,
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
        self.query_reformulator = query_reformulator or QueryReformulator()
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
        resolved_options = options or AnswerOptions()
        _raise_if_cancelled(resolved_options)
        prepared = await _to_thread(self._prepare_answer, query, resolved_options)
        _raise_if_cancelled(resolved_options)
        for step in prepared.trace_steps:
            _raise_if_cancelled(resolved_options)
            yield AnswerStreamEvent("trace", step)
        _raise_if_cancelled(resolved_options)
        yield AnswerStreamEvent("sources", {"contexts": prepared.contexts})

        answer_parts: List[str] = []
        model_completed: Dict[str, Any] = {}
        generation_span = _begin_span(
            resolved_options,
            "Generator execution",
            "generation",
            {
                "messages": [{"role": "user", "content": prepared.prompt.prompt}],
                "parameters": prepared.chat_configuration.get("generation_parameters", {}),
                "configured_deployment_id": prepared.chat_configuration.get("generator_deployment_id", ""),
                "fallback_deployment_ids": prepared.chat_configuration.get("fallback_deployment_ids", []),
            },
        )
        try:
            async for event in self._stream_generation(prepared):
                _raise_if_cancelled(resolved_options)
                if event.type == "delta":
                    answer_parts.append(str(event.data.get("text") or ""))
                elif event.type == "model_completed":
                    model_completed = event.data
                elif event.type == "model_fallback":
                    fallback_step = _trace_step(
                        "Generator fallback",
                        "warning",
                        (
                            f"{event.data.get('deployment_name') or event.data.get('deployment_id')} failed "
                            f"with {event.data.get('error_category') or 'a provider error'}; "
                            f"continuing with {event.data.get('next_deployment_name') or event.data.get('next_deployment_id')}."
                        ),
                        dict(event.data),
                    )
                    prepared.trace_steps.append(fallback_step)
                    yield AnswerStreamEvent("trace", fallback_step)
                    continue
                yield event
        except AnswerCancelled:
            raise
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
        _finish_span(
            resolved_options,
            generation_span,
            status=str(generation.status or "completed"),
            detail=f"Streamed {generation.provider}/{generation.model} generator output.",
            output_payload={
                "answer": answer_text,
                "provider": generation.provider,
                "model": generation.model,
                "finish_reason": generation.metadata.get("finish_reason", ""),
                "metadata": generation.metadata,
            },
            metrics=_generation_metrics(generation),
            model_usage_event_ids=_usage_event_ids(generation.metadata),
        )
        result = self._finalize_answer(prepared, generation)
        for step in result.metadata.get("trace_steps", [])[len(prepared.trace_steps) :]:
            yield AnswerStreamEvent("trace", step)
        yield AnswerStreamEvent("completed", {"result": result})

    def _prepare_answer(self, query: str, options: AnswerOptions) -> PreparedAnswer:
        options = options or AnswerOptions()
        start = time.perf_counter()
        original_query = str(query or "").strip()
        if options.trace_recorder is not None:
            options.trace_recorder.add_instant_span(
                "Chat input",
                "input",
                input_payload={"query": original_query, "characters": len(original_query)},
                output_payload={"accepted": bool(original_query)},
            )
        top_k = max(1, min(int(options.top_k), 50))
        selected_document_ids = _normalize_document_ids(options.document_ids)
        chat_configuration = dict(options.chat_configuration or {})
        knowledge_base = self._selected_knowledge_base(options.knowledge_base_id)
        if options.mode == "adaptive" and knowledge_base is None:
            raise AnsweringError("Select a knowledge base before using Adaptive mode.")
        if options.mode == "simple_rag" and knowledge_base is None:
            raise AnsweringError("Select a knowledge base before using L2 Simple RAG.")
        if options.mode == "complex_rag" and knowledge_base is None:
            raise AnsweringError("Select a knowledge base before using L3 Complex RAG.")
        external_processing_allowed = _external_processing_allowed(knowledge_base)
        conversation_awareness_enabled = _conversation_awareness_enabled(chat_configuration)
        conversation_history_max_exchanges = max(1, int(options.conversation_history_max_exchanges))
        conversation_history_max_characters = max(1, int(options.conversation_history_max_characters))
        conversation_history = (
            normalize_conversation_history(
                options.conversation_history,
                max_exchanges=conversation_history_max_exchanges,
                max_characters=conversation_history_max_characters,
            )
            if conversation_awareness_enabled
            else []
        )
        if options.trace_recorder is not None:
            options.trace_recorder.add_instant_span(
                "Conversation context",
                "conversation",
                status="completed" if conversation_history else "skipped",
                input_payload={
                    "enabled": conversation_awareness_enabled,
                    "max_exchanges": conversation_history_max_exchanges,
                    "max_characters": conversation_history_max_characters,
                },
                output_payload={
                    "history": conversation_history,
                    "exchange_count": conversation_history_exchange_count(conversation_history),
                    "character_count": conversation_history_characters(conversation_history),
                },
            )
        reformulation_span = _begin_span(
            options,
            "Query reformulation",
            "planning",
            {
                "original_query": original_query,
                "conversation_history": conversation_history,
                "planner_deployment_id": chat_configuration.get("planner_deployment_id", ""),
            },
        )
        reformulation = self.query_reformulator.reformulate(
            original_query,
            conversation_history,
            enabled=conversation_awareness_enabled,
            planner_deployment_id=str(chat_configuration.get("planner_deployment_id") or "").strip(),
            model_gateway=self.model_gateway,
            call_context=ModelCallContext(
                purpose="conversation_rewrite",
                request_id=options.request_id,
                user_id=options.user_id,
                conversation_id=options.conversation_id,
                knowledge_base_id=knowledge_base.id if knowledge_base else "",
            ),
            external_processing_allowed=external_processing_allowed,
            history_max_exchanges=conversation_history_max_exchanges,
            history_max_characters=conversation_history_max_characters,
        )
        standalone_query = reformulation.standalone_query
        _finish_span(
            options,
            reformulation_span,
            status="warning" if reformulation.warning else ("completed" if reformulation.rewritten else "skipped"),
            detail=reformulation.warning or reformulation.strategy,
            output_payload={
                "standalone_query": standalone_query,
                "rewritten": reformulation.rewritten,
                "follow_up_detected": reformulation.follow_up_detected,
                "strategy": reformulation.strategy,
                "planner": reformulation.planner_metadata,
            },
            warning=reformulation.warning,
        )
        classification_span = _begin_span(
            options,
            "Query complexity classifier",
            "classification",
            {
                "query": standalone_query,
                "configured_deployment_id": _configuration_value(chat_configuration, "classifier_deployment_id", ""),
            },
        )
        classification = self._classify_query(
            standalone_query,
            chat_configuration,
            options,
            knowledge_base,
            external_processing_allowed,
        )
        complexity_label = classification.label
        _finish_span(
            options,
            classification_span,
            status="warning" if classification.warning else "completed",
            detail=classification.warning or f"Predicted {complexity_label} complexity.",
            output_payload={
                "label": complexity_label,
                "configured": classification.configured,
                "actual": classification.actual,
                "fallback_used": classification.fallback_used,
            },
            metrics={
                "latency_ms": classification.actual.get("latency_ms", 0),
                "input_tokens": classification.actual.get("input_tokens", 0),
                "output_tokens": classification.actual.get("output_tokens", 0),
                "estimated_cost_usd": classification.actual.get("estimated_cost_usd", 0.0),
            },
            model_usage_event_ids=_usage_event_ids(classification.actual),
            warning=classification.warning,
        )
        route_level = self._resolve_route(options.mode, complexity_label)
        if options.trace_recorder is not None:
            options.trace_recorder.add_instant_span(
                "Route decision",
                "routing",
                input_payload={"requested_mode": options.mode, "complexity_label": complexity_label},
                output_payload={"route_level": route_level, "route_label": _route_label(route_level)},
            )
        if route_level == "l2_simple_rag" and knowledge_base is None:
            raise AnsweringError("Select a knowledge base before using L2 Simple RAG.")
        if route_level == "l3_complex_rag" and knowledge_base is None:
            raise AnsweringError("Select a knowledge base before using L3 Complex RAG.")
        if route_level in {"l2_simple_rag", "l3_complex_rag"}:
            selected_document_ids = self._validate_document_filter(knowledge_base, selected_document_ids)
        else:
            selected_document_ids = []

        history_exchange_count = conversation_history_exchange_count(conversation_history)
        history_character_count = conversation_history_characters(conversation_history)
        reformulation_status = "warning" if reformulation.warning else ("completed" if reformulation.rewritten else "skipped")
        if reformulation.warning:
            reformulation_detail = reformulation.warning
        elif reformulation.rewritten:
            reformulation_detail = f"Resolved the follow-up into a standalone query using {reformulation.strategy} reformulation."
        elif reformulation.strategy == "disabled":
            reformulation_detail = "Conversation awareness is disabled for this RAG configuration."
        elif reformulation.strategy == "no_history":
            reformulation_detail = "No completed prior exchanges were available."
        else:
            reformulation_detail = "The current question was already standalone."

        trace_steps = [
            _trace_step("Chat input", "completed", original_query, {"characters": len(original_query)}),
            _trace_step(
                "Conversation context",
                "completed" if conversation_history else "skipped",
                (
                    f"Loaded {history_exchange_count} completed exchange(s) using {history_character_count} characters."
                    if conversation_history
                    else "No conversation history was added to this answer."
                ),
                {
                    "enabled": conversation_awareness_enabled,
                    "exchange_count": history_exchange_count,
                    "message_count": len(conversation_history),
                    "character_count": history_character_count,
                    "exchange_limit": conversation_history_max_exchanges,
                    "character_limit": conversation_history_max_characters,
                    "message_ids": [message.get("message_id", "") for message in conversation_history if message.get("message_id")],
                },
            ),
            _trace_step(
                "Query reformulation",
                reformulation_status,
                reformulation_detail,
                {
                    "original_query": original_query,
                    "standalone_query": standalone_query,
                    "query_rewritten": reformulation.rewritten,
                    "follow_up_detected": reformulation.follow_up_detected,
                    "strategy": reformulation.strategy,
                    **reformulation.planner_metadata,
                },
            ),
            _trace_step(
                "Query complexity classifier",
                "warning" if classification.warning else "completed",
                classification.warning or f"Predicted {complexity_label} query complexity.",
                {
                    "complexity_label": complexity_label,
                    "configured_classifier": classification.configured,
                    "actual_classifier": classification.actual,
                    "fallback_used": classification.fallback_used,
                    "classified_query": standalone_query,
                },
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
        decomposed_queries: List[str] = []
        decomposition = QueryDecompositionResult([], "not_applicable")
        retrieval_steps: List[Dict[str, Any]] = []
        aggregation_summary: Dict[str, Any] = {}
        retrieval_diagnostics: Dict[str, Any] = {}
        query_embedding = self.knowledge_service.query_embedding_details(knowledge_base.id) if knowledge_base else {}
        query_embedding = {
            **query_embedding,
            "used": bool(
                route_level in {"l2_simple_rag", "l3_complex_rag"}
                and options.retrieval_mode in {"dense", "hybrid"}
            ),
            "retrieval_mode": options.retrieval_mode if retrieval_used else "none",
        }
        if route_level == "l2_simple_rag":
            assert knowledge_base is not None
            retrieval_span = _begin_span(
                options,
                "Knowledge base retrieval",
                "retrieval",
                {
                    "query": standalone_query,
                    "knowledge_base_id": knowledge_base.id,
                    "mode": options.retrieval_mode,
                    "top_k": top_k,
                    "document_ids": selected_document_ids,
                    "query_embedding": query_embedding,
                },
            )
            contexts, retrieval_diagnostics = self.retriever.search_with_diagnostics(
                query=standalone_query,
                knowledge_base_id=knowledge_base.id,
                top_k=top_k,
                mode=options.retrieval_mode,
                document_ids=selected_document_ids,
            )
            _finish_span(
                options,
                retrieval_span,
                status="completed" if contexts else "empty",
                detail=f"Retrieved {len(contexts)} chunk(s) from {knowledge_base.name}.",
                output_payload={"contexts": _context_payloads(contexts), **retrieval_diagnostics},
                metrics={
                    "candidate_count": retrieval_diagnostics.get("candidate_count", 0),
                    "selected_count": len(contexts),
                    "bm25_weight": self.retriever.bm25_weight,
                    "dense_weight": self.retriever.dense_weight,
                },
            )
            _record_retrieval_component_spans(options, retrieval_diagnostics, retrieval_span)
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
                        "query_embedding": query_embedding,
                    },
                )
            )
            retrieval_mode = options.retrieval_mode
        elif route_level == "l3_complex_rag":
            assert knowledge_base is not None
            decomposition_span = _begin_span(
                options,
                "Query decomposition",
                "planning",
                {
                    "standalone_query": standalone_query,
                    "planner_deployment_id": _configuration_value(chat_configuration, "planner_deployment_id", ""),
                },
            )
            decomposition = self.decomposer.decompose_with_planner(
                standalone_query,
                planner_deployment_id=str(
                    _configuration_value(chat_configuration, "planner_deployment_id", "")
                ).strip(),
                model_gateway=self.model_gateway,
                call_context=ModelCallContext(
                    purpose="query_decomposition",
                    request_id=options.request_id,
                    user_id=options.user_id,
                    conversation_id=options.conversation_id,
                    knowledge_base_id=knowledge_base.id,
                ),
                external_processing_allowed=external_processing_allowed,
            )
            decomposed_queries = decomposition.queries
            _finish_span(
                options,
                decomposition_span,
                status="warning" if decomposition.warning else "completed",
                detail=decomposition.warning or decomposition.strategy,
                output_payload={
                    "queries": decomposed_queries,
                    "strategy": decomposition.strategy,
                    "configured": decomposition.configured,
                    "actual": decomposition.actual,
                    "fallback_used": decomposition.fallback_used,
                },
                metrics={
                    "subquery_count": len(decomposed_queries),
                    "input_tokens": decomposition.actual.get("input_tokens", 0),
                    "output_tokens": decomposition.actual.get("output_tokens", 0),
                    "estimated_cost_usd": decomposition.actual.get("estimated_cost_usd", 0.0),
                },
                model_usage_event_ids=_usage_event_ids(decomposition.actual),
                warning=decomposition.warning,
            )
            trace_steps.append(
                _trace_step(
                    "Query decomposition",
                    "warning" if decomposition.warning else "completed",
                    decomposition.warning or f"Created {len(decomposed_queries)} retrieval subquery(s) using {decomposition.strategy}.",
                    {
                        "decomposed_queries": decomposed_queries,
                        "strategy": decomposition.strategy,
                        "configured_planner": decomposition.configured,
                        "actual_planner": decomposition.actual,
                        "fallback_used": decomposition.fallback_used,
                    },
                )
            )
            retrieval_span = _begin_span(
                options,
                "Multi-step retrieval",
                "retrieval",
                {
                    "subqueries": decomposed_queries,
                    "knowledge_base_id": knowledge_base.id,
                    "mode": options.retrieval_mode,
                    "top_k": top_k,
                    "document_ids": selected_document_ids,
                    "query_embedding": query_embedding,
                },
            )
            contexts, retrieval_steps, aggregation_summary, retrieval_diagnostics = self._retrieve_multi_step(
                decomposed_queries=decomposed_queries,
                knowledge_base=knowledge_base,
                top_k=top_k,
                mode=options.retrieval_mode,
                document_ids=selected_document_ids,
            )
            _finish_span(
                options,
                retrieval_span,
                status="completed" if aggregation_summary.get("candidate_count", 0) else "empty",
                detail=f"Ran {len(retrieval_steps)} retrieval step(s).",
                output_payload={
                    "retrieval_steps": retrieval_steps,
                    "diagnostics": retrieval_diagnostics,
                    "selected_contexts": _context_payloads(contexts),
                },
                metrics=aggregation_summary,
            )
            for diagnostic_step in list(retrieval_diagnostics.get("steps") or []):
                _record_retrieval_component_spans(options, diagnostic_step, retrieval_span)
            if options.trace_recorder is not None:
                options.trace_recorder.add_instant_span(
                    "Context aggregation",
                    "aggregation",
                    status="completed" if contexts else "empty",
                    input_payload={"retrieval_steps": retrieval_steps},
                    output_payload={"contexts": _context_payloads(contexts)},
                    metrics=aggregation_summary,
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
                        "query_embedding": query_embedding,
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
            rerank_span = _begin_span(
                options,
                "Context reranking",
                "reranking",
                {
                    "query": standalone_query,
                    "deployment_id": reranker_deployment_id,
                    "candidates": _context_payloads(original_contexts),
                },
            )
            try:
                reranked = self.model_gateway.rerank_sync(
                    standalone_query,
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
                _finish_span(
                    options,
                    rerank_span,
                    detail=f"Reranked {len(contexts)} context chunk(s).",
                    output_payload={
                        "before": _context_payloads(original_contexts),
                        "after": _context_payloads(contexts),
                        "metadata": reranked.metadata,
                    },
                    metrics={
                        "candidate_count": len(original_contexts),
                        "selected_count": len(contexts),
                        "input_tokens": reranked.metadata.get("input_tokens", 0),
                        "output_tokens": reranked.metadata.get("output_tokens", 0),
                        "estimated_cost_usd": reranked.metadata.get("estimated_cost_usd", 0.0),
                    },
                    model_usage_event_ids=_usage_event_ids(reranked.metadata),
                )
                trace_steps.append(
                    _trace_step(
                        "Context reranking",
                        "completed",
                        f"Reranked {len(contexts)} context chunk(s).",
                        {"deployment_id": reranker_deployment_id, **reranked.metadata},
                    )
                )
            except ModelFarmError as exc:
                _finish_span(
                    options,
                    rerank_span,
                    status="warning",
                    detail="Reranker failed open; original retrieval order was retained.",
                    output_payload={"contexts": _context_payloads(original_contexts)},
                    warning=str(exc),
                )
                trace_steps.append(
                    _trace_step(
                        "Context reranking",
                        "warning",
                        "Reranker failed open; original retrieval order was retained.",
                        {"deployment_id": reranker_deployment_id, "error": str(exc)},
                    )
                )

        citations_enabled = _citations_enabled(chat_configuration)
        prompt_span = _begin_span(
            options,
            "Prompt builder",
            "prompt",
            {
                "original_query": original_query,
                "standalone_query": standalone_query,
                "conversation_history": conversation_history,
                "contexts": _context_payloads(contexts),
                "chat_configuration": chat_configuration,
                "route_level": route_level,
            },
        )
        prompt = self.prompt_builder.build(
            original_query,
            contexts,
            chat_configuration,
            route_level=route_level,
            conversation_history=conversation_history,
            standalone_query=standalone_query,
        )
        _finish_span(
            options,
            prompt_span,
            detail=f"Built generator prompt with {prompt.context_count} context chunk(s).",
            output_payload={
                "prompt": prompt.prompt,
                "prompt_preview": prompt.prompt_preview,
                "metadata": prompt.metadata,
            },
            metrics={"input_chars": prompt.input_chars, "context_count": prompt.context_count},
        )
        trace_steps.append(
            _trace_step(
                "Prompt builder",
                "completed",
                f"Built generator prompt with {prompt.context_count} context chunk(s).",
                {**prompt.metadata, "input_chars": prompt.input_chars, "prompt_preview": prompt.prompt_preview},
            )
        )
        return PreparedAnswer(
            query=original_query,
            standalone_query=standalone_query,
            options=options,
            start=start,
            top_k=top_k,
            chat_configuration=chat_configuration,
            conversation_awareness_enabled=conversation_awareness_enabled,
            conversation_history=conversation_history,
            conversation_history_max_exchanges=conversation_history_max_exchanges,
            conversation_history_max_characters=conversation_history_max_characters,
            reformulation=reformulation,
            classification=classification,
            complexity_label=complexity_label,
            route_level=route_level,
            knowledge_base=knowledge_base,
            document_ids=selected_document_ids,
            contexts=contexts,
            retrieval_mode=retrieval_mode,
            retrieval_used=retrieval_used,
            external_processing_allowed=external_processing_allowed,
            decomposed_queries=decomposed_queries,
            decomposition=decomposition,
            retrieval_steps=retrieval_steps,
            aggregation_summary=aggregation_summary,
            retrieval_diagnostics=retrieval_diagnostics,
            query_embedding=query_embedding,
            citations_enabled=citations_enabled,
            prompt=prompt,
            trace_steps=trace_steps,
        )

    def _classify_query(
        self,
        query: str,
        chat_configuration: Dict[str, Any],
        options: AnswerOptions,
        knowledge_base: Optional[KnowledgeBaseRecord],
        external_processing_allowed: bool,
    ) -> QueryClassificationExecution:
        deployment_id = str(_configuration_value(chat_configuration, "classifier_deployment_id", "")).strip()
        default_descriptor = {
            "deployment_id": "",
            "provider": "Local",
            "model": type(self.router.classifier).__name__,
            "runtime": "process_default",
        }
        if not deployment_id:
            label = self.router.classifier.predict(query)
            return QueryClassificationExecution(label, dict(default_descriptor), dict(default_descriptor))

        configured = {"deployment_id": deployment_id}
        if self.model_farm_service is not None:
            try:
                deployment = self.model_farm_service.get_deployment(deployment_id)
                configured.update(
                    {
                        "name": deployment.name,
                        "provider": deployment.provider,
                        "model": deployment.model,
                    }
                )
            except KeyError:
                pass
        if self.model_gateway is None:
            label = self.router.classifier.predict(query)
            warning = (
                f"Configured classifier {configured.get('name') or deployment_id} could not run because "
                "Model Gateway is unavailable; the process-default classifier was used."
            )
            return QueryClassificationExecution(label, configured, default_descriptor, True, warning)
        try:
            result = self.model_gateway.classify_sync(
                query,
                deployment_id,
                context=ModelCallContext(
                    purpose="query_classification",
                    request_id=options.request_id,
                    user_id=options.user_id,
                    conversation_id=options.conversation_id,
                    knowledge_base_id=knowledge_base.id if knowledge_base else "",
                ),
                external_processing_allowed=external_processing_allowed,
            )
            actual = {
                "deployment_id": result.deployment_id,
                "provider": result.provider,
                "model": result.model,
                **result.metadata,
            }
            return QueryClassificationExecution(result.label, configured, actual)
        except ModelFarmError as exc:
            label = self.router.classifier.predict(query)
            warning = (
                f"Configured classifier {configured.get('name') or deployment_id} failed; "
                f"the process-default classifier predicted {label}. Error: {exc}"
            )
            return QueryClassificationExecution(label, configured, default_descriptor, True, warning)

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
        generation_span = _begin_span(
            prepared.options,
            "Generator execution",
            "generation",
            {
                "messages": [{"role": "user", "content": prepared.prompt.prompt}],
                "parameters": prepared.chat_configuration.get("generation_parameters", {}),
                "configured_deployment_id": prepared.chat_configuration.get("generator_deployment_id", ""),
                "fallback_deployment_ids": prepared.chat_configuration.get("fallback_deployment_ids", []),
            },
        )
        try:
            generator = self.generator_resolver.resolve(
                prepared.chat_configuration,
                external_processing_allowed=prepared.external_processing_allowed,
                call_context=self._call_context(prepared),
            )
            result = generator.generate(self._generation_request(prepared))
            _finish_span(
                prepared.options,
                generation_span,
                status=str(result.status or "completed"),
                detail=f"Executed {result.provider}/{result.model} generator.",
                output_payload={
                    "answer": result.answer,
                    "provider": result.provider,
                    "model": result.model,
                    "metadata": result.metadata,
                },
                metrics=_generation_metrics(result),
                model_usage_event_ids=_usage_event_ids(result.metadata),
            )
            return result
        except (GeneratorConfigurationError, GeneratorExecutionError) as exc:
            _finish_span(
                prepared.options,
                generation_span,
                status="failed",
                detail="Generator execution failed.",
                error=str(exc),
            )
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
                    cancellation_token=prepared.options.cancellation_token,
                ):
                    yield AnswerStreamEvent(event.type, event.data)
                return
            except ModelFarmError as exc:
                raise GeneratorExecutionError(str(exc)) from exc
        generation = await _to_thread(self._generate_answer, prepared)
        for chunk in _text_chunks(generation.answer, 32):
            _raise_if_cancelled(prepared.options)
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
        traced_fallback_deployments = {
            str(step.get("metadata", {}).get("deployment_id") or "")
            for step in trace_steps
            if step.get("step") == "Generator fallback"
        }
        for failed_attempt in list(generation.metadata.get("fallback_attempts") or []):
            deployment_id = str(failed_attempt.get("deployment_id") or "")
            if deployment_id in traced_fallback_deployments:
                continue
            trace_steps.append(
                _trace_step(
                    "Generator fallback",
                    "warning",
                    (
                        f"{failed_attempt.get('deployment_name') or deployment_id} failed "
                        f"with {failed_attempt.get('error_category') or 'a provider error'}; "
                        "the next configured fallback was used."
                    ),
                    dict(failed_attempt),
                )
            )
            if prepared.options.trace_recorder is not None:
                prepared.options.trace_recorder.add_observed_span(
                    "Generator fallback attempt",
                    "generation",
                    duration_ms=float(failed_attempt.get("latency_ms") or 0.0),
                    status="failed",
                    detail=(
                        f"{failed_attempt.get('deployment_name') or deployment_id} failed; "
                        "the next configured fallback was attempted."
                    ),
                    input_payload={
                        "deployment_id": deployment_id,
                        "provider": failed_attempt.get("provider", ""),
                        "model": failed_attempt.get("model", ""),
                        "fallback_index": failed_attempt.get("fallback_index", 0),
                    },
                    output_payload={"attempt": failed_attempt},
                    metrics={"latency_ms": failed_attempt.get("latency_ms", 0)},
                    model_usage_event_ids=_usage_event_ids(failed_attempt),
                    error=str(failed_attempt.get("error") or failed_attempt.get("detail") or "Provider attempt failed."),
                )
        configured_deployment_id = str(prepared.chat_configuration.get("generator_deployment_id") or "")
        configured_generator = {
            "deployment_id": configured_deployment_id,
            "provider": prepared.chat_configuration.get("generator_provider", "Local"),
            "model": prepared.chat_configuration.get("generator_model", "extractive"),
        }
        if configured_deployment_id and self.model_farm_service is not None:
            try:
                configured_deployment = self.model_farm_service.get_deployment(configured_deployment_id)
                configured_generator.update(
                    {
                        "name": configured_deployment.name,
                        "provider": configured_deployment.provider,
                        "model": configured_deployment.model,
                    }
                )
            except KeyError:
                configured_generator["name"] = "Missing deployment"
        actual_generator = {
            "provider": generation.provider,
            "model": generation.model,
        }
        actual_deployment_id = str(generation.metadata.get("deployment_id") or "")
        if actual_deployment_id:
            actual_generator["deployment_id"] = actual_deployment_id
        fallback_index = int(generation.metadata.get("fallback_index") or 0)
        generator_detail = f"Executed {generation.provider}/{generation.model} generator."
        generator_status = generation.status
        if fallback_index > 0:
            generator_detail = (
                f"Configured generator {configured_generator.get('provider')}/{configured_generator.get('model')} failed; "
                f"executed fallback {generation.provider}/{generation.model}."
            )
            generator_status = "warning"
        trace_steps.append(
            _trace_step(
                "Generator execution",
                generator_status,
                generator_detail,
                {
                    "configured_generator": configured_generator,
                    "actual_generator": actual_generator,
                    "fallback_used": fallback_index > 0,
                    "provider": generation.provider,
                    "model": generation.model,
                    "input_chars": generation.input_chars,
                    "output_chars": generation.output_chars,
                    **generation.metadata,
                },
            )
        )

        citation_validation = _validate_citations(
            generation.answer,
            prepared.contexts,
            enabled=prepared.citations_enabled,
        )
        citation_span = _begin_span(
            prepared.options,
            "Citation validation",
            "validation",
            {
                "answer": generation.answer,
                "available_sources": _citation_source_map(prepared.contexts),
                "enabled": prepared.citations_enabled,
            },
        )
        _finish_span(
            prepared.options,
            citation_span,
            status=citation_validation["status"],
            detail=citation_validation["detail"],
            output_payload=citation_validation,
            metrics={
                "cited_count": len(citation_validation.get("cited") or []),
                "invalid_count": len(citation_validation.get("invalid") or []),
            },
            warning=(
                citation_validation["detail"]
                if citation_validation["status"] == "warning"
                else ""
            ),
        )
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
            "conversation_awareness_enabled": prepared.conversation_awareness_enabled,
            "history_exchange_count": conversation_history_exchange_count(prepared.conversation_history),
            "history_character_count": conversation_history_characters(prepared.conversation_history),
            "history_exchange_limit": prepared.conversation_history_max_exchanges,
            "history_character_limit": prepared.conversation_history_max_characters,
            "original_query": prepared.query,
            "standalone_query": prepared.standalone_query,
            "query_rewritten": prepared.reformulation.rewritten,
            "reformulation_strategy": prepared.reformulation.strategy,
            "reformulation_warning": prepared.reformulation.warning,
            "planner_reformulation": prepared.reformulation.planner_metadata,
            "route_level": prepared.route_level,
            "route_label": _route_label(prepared.route_level),
            "complexity_label": prepared.complexity_label,
            "configured_classifier": prepared.classification.configured,
            "actual_classifier": prepared.classification.actual,
            "classifier_fallback_used": prepared.classification.fallback_used,
            "retrieval_used": prepared.retrieval_used,
            "retrieval_mode": prepared.retrieval_mode,
            "query_embedding": prepared.query_embedding,
            "top_k": prepared.top_k,
            "document_filter_ids": prepared.document_ids,
            "document_filter_count": len(prepared.document_ids),
            "multi_step": prepared.route_level == "l3_complex_rag",
            "decomposed_queries": prepared.decomposed_queries,
            "configured_planner": prepared.decomposition.configured,
            "actual_planner": prepared.decomposition.actual,
            "planner_fallback_used": prepared.decomposition.fallback_used,
            "retrieval_steps": prepared.retrieval_steps,
            "aggregation_summary": prepared.aggregation_summary,
            "retrieval_diagnostics": prepared.retrieval_diagnostics,
            "latency_ms": elapsed_ms,
            "generator": generation.model,
            "configured_generator": configured_generator,
            "actual_generator": actual_generator,
            "fallback_used": fallback_index > 0,
            "fallback_attempts": list(generation.metadata.get("fallback_attempts") or []),
            "generation_status": generation.status,
            "prompt_preview": generation.prompt_preview,
            "input_chars": generation.input_chars,
            "output_chars": generation.output_chars,
            "chat_configuration": prepared.chat_configuration,
            "trace_steps": trace_steps,
            "request_id": prepared.options.request_id,
            "citation_validation": citation_validation,
            "citation_sources": _citation_source_map(prepared.contexts),
            "citations_enabled": prepared.citations_enabled,
            "external_processing_allowed": prepared.external_processing_allowed,
        }
        if prepared.options.trace_recorder is not None:
            metadata["trace_id"] = prepared.options.trace_recorder.trace_id
            metadata["trace_summary"] = dict(prepared.options.trace_recorder.report.get("summary") or {})
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
    ) -> Tuple[List[RetrievedContext], List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        per_step_top_k = max(2, min(top_k, 8))
        candidates: Dict[str, Dict[str, Any]] = {}
        retrieval_steps: List[Dict[str, Any]] = []
        total_retrieved = 0
        diagnostic_steps: List[Dict[str, Any]] = []

        for subquery_index, subquery in enumerate(decomposed_queries, start=1):
            step_contexts, step_diagnostics = self.retriever.search_with_diagnostics(
                query=subquery,
                knowledge_base_id=knowledge_base.id,
                top_k=per_step_top_k,
                mode=mode,
                document_ids=document_ids,
            )
            total_retrieved += len(step_contexts)
            diagnostic_steps.append(
                {
                    "retrieval_step": f"step-{subquery_index}",
                    "subquery_index": subquery_index,
                    **step_diagnostics,
                }
            )
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
        return selected, retrieval_steps, aggregation_summary, {
            "mode": mode,
            "subquery_count": len(decomposed_queries),
            "steps": diagnostic_steps,
            "aggregation": aggregation_summary,
        }

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

    def decompose_with_planner(
        self,
        query: str,
        *,
        planner_deployment_id: str = "",
        model_gateway: Optional[ModelGateway] = None,
        call_context: Optional[ModelCallContext] = None,
        external_processing_allowed: bool = True,
        max_subqueries: int = 4,
    ) -> QueryDecompositionResult:
        deterministic = self.decompose(query, max_subqueries=max_subqueries)
        if not planner_deployment_id:
            return QueryDecompositionResult(
                deterministic,
                "deterministic_rules",
                configured={"deployment_id": ""},
                actual={"runtime": "deterministic_rules"},
            )
        configured: Dict[str, Any] = {"deployment_id": planner_deployment_id}
        if model_gateway is not None:
            try:
                gateway_service = getattr(model_gateway, "service", None)
                if gateway_service is not None:
                    deployment = gateway_service.get_deployment(planner_deployment_id)
                    configured.update(
                        {
                            "name": deployment.name,
                            "provider": deployment.provider,
                            "model": deployment.model,
                        }
                    )
            except (KeyError, ModelFarmError):
                pass
        if model_gateway is None:
            return QueryDecompositionResult(
                deterministic,
                "deterministic_rules",
                configured=configured,
                actual={"runtime": "deterministic_rules"},
                fallback_used=True,
                warning="The configured planner could not run because Model Gateway is unavailable; deterministic decomposition was used.",
            )
        try:
            generated = model_gateway.generate_sync(
                [
                    {
                        "role": "system",
                        "content": (
                            "Decompose the business-workflow question into 2 to 4 independently searchable subqueries. "
                            "Return only a JSON array of strings. Do not return markdown or an object."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                planner_deployment_id,
                parameters={"temperature": 0, "max_tokens": 240},
                context=call_context,
                external_processing_allowed=external_processing_allowed,
                capability="planner",
            )
            planned = _planned_subqueries(generated.text, query, max_subqueries)
            actual = {
                "deployment_id": generated.deployment_id,
                "provider": generated.provider,
                "model": generated.model,
                **generated.metadata,
            }
            return QueryDecompositionResult(planned, "model_planner", configured=configured, actual=actual)
        except (ModelFarmError, ValueError) as exc:
            return QueryDecompositionResult(
                deterministic,
                "deterministic_rules",
                configured=configured,
                actual={"runtime": "deterministic_rules"},
                fallback_used=True,
                warning=f"Configured planner failed validation or execution; deterministic decomposition was used. Error: {exc}",
            )

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
        contexts, _ = self.search_with_diagnostics(
            query,
            knowledge_base_id,
            top_k=top_k,
            mode=mode,
            document_ids=document_ids,
        )
        return contexts

    def search_with_diagnostics(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int = 4,
        mode: RetrievalMode = "hybrid",
        document_ids: Optional[List[str]] = None,
    ) -> Tuple[List[RetrievedContext], Dict[str, Any]]:
        document_ids = _normalize_document_ids(document_ids)
        candidate_limit = max(top_k * 4, 25) if mode == "hybrid" else top_k
        bm25_contexts: List[RetrievedContext] = []
        dense_contexts: List[RetrievedContext] = []
        bm25_raw: Dict[str, float] = {}
        bm25_duration_ms = 0.0
        dense_duration_ms = 0.0
        hybrid_duration_ms = 0.0
        if mode == "bm25":
            component_started = time.perf_counter()
            bm25_contexts, bm25_raw = self._bm25_search_details(
                query, knowledge_base_id, candidate_limit, document_ids=document_ids
            )
            bm25_duration_ms = (time.perf_counter() - component_started) * 1000
            contexts = bm25_contexts[:top_k]
        elif mode == "dense":
            component_started = time.perf_counter()
            dense_contexts = self._dense_search(query, knowledge_base_id, candidate_limit, document_ids=document_ids)
            dense_duration_ms = (time.perf_counter() - component_started) * 1000
            contexts = dense_contexts[:top_k]
        elif mode == "hybrid":
            component_started = time.perf_counter()
            bm25_contexts, bm25_raw = self._bm25_search_details(
                query, knowledge_base_id, candidate_limit, document_ids=document_ids
            )
            bm25_duration_ms = (time.perf_counter() - component_started) * 1000
            component_started = time.perf_counter()
            dense_contexts = self._dense_search(query, knowledge_base_id, candidate_limit, document_ids=document_ids)
            dense_duration_ms = (time.perf_counter() - component_started) * 1000
            component_started = time.perf_counter()
            contexts = self._combine_hybrid(bm25_contexts, dense_contexts, top_k)
            hybrid_duration_ms = (time.perf_counter() - component_started) * 1000
        else:
            raise AnsweringError(f"Unsupported retrieval mode: {mode}")

        bm25_scores = {context.document.id: context.score for context in bm25_contexts}
        dense_scores = {context.document.id: context.score for context in dense_contexts}
        normalized_bm25 = _normalize_scores(bm25_scores)
        normalized_dense = _normalize_scores(dense_scores)
        bm25_ranks = {context.document.id: context.rank for context in bm25_contexts}
        dense_ranks = {context.document.id: context.rank for context in dense_contexts}
        selected_ranks = {context.document.id: context.rank for context in contexts}
        documents = {context.document.id: context.document for context in [*bm25_contexts, *dense_contexts, *contexts]}
        candidates = []
        for document_id, document in documents.items():
            normalized_lexical = normalized_bm25.get(document_id, 0.0)
            normalized_vector = normalized_dense.get(document_id, 0.0)
            hybrid_score = (
                self.bm25_weight * normalized_lexical + self.dense_weight * normalized_vector
                if mode == "hybrid"
                else (normalized_lexical if mode == "bm25" else normalized_vector)
            )
            candidates.append(
                {
                    "chunk_id": document_id,
                    "document_id": document.metadata.get("document_id", ""),
                    "text": document.text,
                    "metadata": document.metadata,
                    "bm25_raw_score": round(float(bm25_raw.get(document_id, bm25_scores.get(document_id, 0.0))), 8),
                    "bm25_normalized_score": round(float(normalized_lexical), 8),
                    "bm25_rank": bm25_ranks.get(document_id),
                    "dense_raw_score": round(float(dense_scores.get(document_id, 0.0)), 8),
                    "dense_normalized_score": round(float(normalized_vector), 8),
                    "dense_rank": dense_ranks.get(document_id),
                    "bm25_weighted_score": round(self.bm25_weight * normalized_lexical, 8),
                    "dense_weighted_score": round(self.dense_weight * normalized_vector, 8),
                    "hybrid_score": round(float(hybrid_score), 8),
                    "selected": document_id in selected_ranks,
                    "selected_rank": selected_ranks.get(document_id),
                }
            )
        candidates.sort(key=lambda item: (item["selected_rank"] is not None, item["hybrid_score"]), reverse=True)
        return contexts, {
            "query": query,
            "mode": mode,
            "top_k": top_k,
            "candidate_limit": candidate_limit,
            "document_filter_ids": document_ids,
            "bm25_weight": self.bm25_weight,
            "dense_weight": self.dense_weight,
            "candidate_count": len(candidates),
            "selected_count": len(contexts),
            "bm25_duration_ms": round(bm25_duration_ms, 3),
            "dense_duration_ms": round(dense_duration_ms, 3),
            "hybrid_duration_ms": round(hybrid_duration_ms, 3),
            "candidates": candidates,
        }

    def _bm25_search(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedContext]:
        contexts, _ = self._bm25_search_details(query, knowledge_base_id, top_k, document_ids)
        return contexts

    def _bm25_search_details(
        self,
        query: str,
        knowledge_base_id: str,
        top_k: int,
        document_ids: Optional[List[str]] = None,
    ) -> Tuple[List[RetrievedContext], Dict[str, float]]:
        chunks = self._filtered_chunks(knowledge_base_id, document_ids)
        if not chunks:
            return [], {}
        retriever = InMemoryHybridRetriever(
            [_document_from_chunk(chunk) for chunk in chunks],
            bm25_weight=self.bm25_weight,
            dense_weight=self.dense_weight,
        )
        contexts = retriever.search(query, top_k=min(top_k, len(chunks)), mode="bm25")
        raw_scores = retriever.score_diagnostics(query)["bm25_raw"]
        return contexts, raw_scores

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
        return self._combine_hybrid(bm25_contexts, dense_contexts, top_k)

    def _combine_hybrid(
        self,
        bm25_contexts: List[RetrievedContext],
        dense_contexts: List[RetrievedContext],
        top_k: int,
    ) -> List[RetrievedContext]:
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


def _begin_span(
    options: AnswerOptions,
    name: str,
    category: str,
    input_payload: Optional[Dict[str, Any]] = None,
) -> Optional[SpanHandle]:
    if options.trace_recorder is None:
        return None
    return options.trace_recorder.begin_span(name, category, input_payload=input_payload)


def _finish_span(
    options: AnswerOptions,
    handle: Optional[SpanHandle],
    *,
    status: str = "completed",
    detail: str = "",
    output_payload: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    model_usage_event_ids: Optional[List[str]] = None,
    warning: str = "",
    error: str = "",
) -> Optional[Dict[str, Any]]:
    if options.trace_recorder is None or handle is None:
        return None
    return options.trace_recorder.finish_span(
        handle,
        status=status,
        detail=detail,
        output_payload=output_payload,
        metrics=metrics,
        model_usage_event_ids=model_usage_event_ids,
        warning=warning,
        error=error,
    )


def _context_payloads(contexts: List[RetrievedContext]) -> List[Dict[str, Any]]:
    return [
        {
            "id": context.document.id,
            "rank": context.rank,
            "score": context.score,
            "mode": context.mode,
            "text": context.document.text,
            "metadata": context.document.metadata,
        }
        for context in contexts
    ]


def _usage_event_ids(metadata: Dict[str, Any]) -> List[str]:
    values = metadata.get("usage_event_ids")
    if isinstance(values, list):
        return [str(value) for value in values if value]
    value = metadata.get("usage_event_id")
    return [str(value)] if value else []


def _generation_metrics(generation: GeneratorResult) -> Dict[str, Any]:
    return {
        "input_chars": generation.input_chars,
        "output_chars": generation.output_chars,
        "input_tokens": int(generation.metadata.get("input_tokens") or 0),
        "output_tokens": int(generation.metadata.get("output_tokens") or 0),
        "estimated_cost_usd": float(generation.metadata.get("estimated_cost_usd") or 0.0),
        "latency_ms": float(generation.metadata.get("latency_ms") or 0.0),
        "fallback_index": int(generation.metadata.get("fallback_index") or 0),
    }


def _record_retrieval_component_spans(
    options: AnswerOptions,
    diagnostics: Dict[str, Any],
    parent: Optional[SpanHandle],
) -> None:
    recorder = options.trace_recorder
    if recorder is None:
        return
    mode = str(diagnostics.get("mode") or "")
    candidates = list(diagnostics.get("candidates") or [])
    parent_span_id = parent.span_id if parent else ""
    common_input = {
        "query": diagnostics.get("query", ""),
        "retrieval_step": diagnostics.get("retrieval_step", ""),
        "subquery_index": diagnostics.get("subquery_index"),
        "document_filter_ids": diagnostics.get("document_filter_ids", []),
    }
    if mode in {"bm25", "hybrid"}:
        recorder.add_observed_span(
            "BM25 retrieval",
            "retrieval.bm25",
            duration_ms=float(diagnostics.get("bm25_duration_ms") or 0.0),
            input_payload=common_input,
            output_payload={"candidates": candidates},
            metrics={
                "candidate_count": len(candidates),
                "selected_count": sum(1 for item in candidates if item.get("selected")),
            },
            parent_span_id=parent_span_id,
        )
    if mode in {"dense", "hybrid"}:
        recorder.add_observed_span(
            "Dense retrieval",
            "retrieval.dense",
            duration_ms=float(diagnostics.get("dense_duration_ms") or 0.0),
            input_payload=common_input,
            output_payload={"candidates": candidates},
            metrics={
                "candidate_count": len(candidates),
                "selected_count": sum(1 for item in candidates if item.get("selected")),
            },
            parent_span_id=parent_span_id,
        )
    if mode == "hybrid":
        recorder.add_observed_span(
            "Hybrid score normalization",
            "retrieval.hybrid",
            duration_ms=float(diagnostics.get("hybrid_duration_ms") or 0.0),
            input_payload={
                **common_input,
                "bm25_weight": diagnostics.get("bm25_weight", 0.0),
                "dense_weight": diagnostics.get("dense_weight", 0.0),
            },
            output_payload={"candidates": candidates},
            metrics={
                "candidate_count": len(candidates),
                "selected_count": sum(1 for item in candidates if item.get("selected")),
            },
            parent_span_id=parent_span_id,
        )


def _conversation_awareness_enabled(chat_configuration: Dict[str, Any]) -> bool:
    metadata = chat_configuration.get("metadata")
    value = chat_configuration.get("conversation_awareness_enabled")
    if value is None and isinstance(metadata, dict):
        value = metadata.get("conversation_awareness_enabled")
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _configuration_value(chat_configuration: Dict[str, Any], key: str, default: Any = None) -> Any:
    value = chat_configuration.get(key)
    metadata = chat_configuration.get("metadata")
    if value is None and isinstance(metadata, dict):
        value = metadata.get(key)
    return default if value is None else value


def _citations_enabled(chat_configuration: Dict[str, Any]) -> bool:
    value = _configuration_value(chat_configuration, "citations_enabled", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _planned_subqueries(value: str, original_query: str, max_subqueries: int) -> List[str]:
    text = str(value or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Planner output was not valid JSON.") from exc
    if not isinstance(payload, list):
        raise ValueError("Planner output must be a JSON array of strings.")
    original = " ".join(original_query.split())
    unique: List[str] = []
    seen: set[str] = set()
    for item in payload:
        query = " ".join(str(item or "").split()).strip()
        normalized = query.casefold()
        if not query or normalized in seen or normalized == original.casefold():
            continue
        unique.append(query)
        seen.add(normalized)
    if not unique:
        raise ValueError("Planner output did not contain a usable retrieval subquery.")
    selected = unique[: max(max_subqueries - 1, 1)]
    selected.append(original)
    if len(selected) < 2 or len(selected) > max_subqueries:
        raise ValueError(f"Planner output must resolve to between 2 and {max_subqueries} subqueries.")
    return selected


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


def _validate_citations(
    answer: str,
    contexts: List[RetrievedContext],
    enabled: bool = True,
) -> Dict[str, Any]:
    valid_labels = {str(context.document.metadata.get("source_label") or f"S{context.rank}") for context in contexts}
    cited = set(re.findall(r"\[(S\d+)\]", answer or ""))
    invalid = sorted(cited - valid_labels)
    if not enabled:
        return {
            "status": "disabled",
            "detail": "Citation validation is disabled for this RAG configuration.",
            "cited": sorted(cited),
            "invalid": [],
            "available": sorted(valid_labels),
        }
    if not contexts:
        return {"status": "not_applicable", "detail": "No retrieval context required citations.", "cited": sorted(cited), "invalid": invalid}
    if invalid:
        return {"status": "warning", "detail": "The answer contains citations that do not match retrieved sources.", "cited": sorted(cited), "invalid": invalid, "available": sorted(valid_labels)}
    if not cited:
        return {"status": "warning", "detail": "Retrieved context was used, but the answer contains no source labels.", "cited": [], "invalid": [], "available": sorted(valid_labels)}
    return {"status": "completed", "detail": "All answer citations match retrieved sources.", "cited": sorted(cited), "invalid": [], "available": sorted(valid_labels)}


def _citation_source_map(contexts: List[RetrievedContext]) -> Dict[str, Dict[str, Any]]:
    sources: Dict[str, Dict[str, Any]] = {}
    for context in contexts:
        metadata = context.document.metadata
        label = str(metadata.get("source_label") or f"S{context.rank}")
        sources[label] = {
            "context_id": context.document.id,
            "chunk_id": metadata.get("chunk_id") or context.document.id,
            "document_id": metadata.get("document_id", ""),
            "title": metadata.get("title", ""),
            "chunk_index": metadata.get("chunk_index"),
            "rank": context.rank,
        }
    return sources


async def _to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    import asyncio

    return await asyncio.to_thread(func, *args, **kwargs)


def _raise_if_cancelled(options: AnswerOptions) -> None:
    if options.cancellation_token:
        options.cancellation_token.raise_if_cancelled()


def _text_chunks(text: str, size: int) -> List[str]:
    return [text[index : index + max(size, 1)] for index in range(0, len(text), max(size, 1))]
