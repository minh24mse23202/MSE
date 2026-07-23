import asyncio

import pytest

from aragbiz.answering import AdaptiveRAGAnswerService, AnswerOptions
from aragbiz.classifier import ClassificationPrediction
from aragbiz.cancellation import AnswerCancelled, CancellationToken
from aragbiz.generation import ExtractiveGenerator
from aragbiz.knowledge import HashEmbeddingModel, KnowledgeService, OverlapChunker
from aragbiz.knowledge_store import JsonKnowledgeRepository
from aragbiz.model_farm import (
    ModelClassificationResult,
    ModelFarmError,
    ModelGenerationResult,
    ModelStreamEvent,
)
from aragbiz.routing import AdaptiveRouter


class StubClassifier:
    def __init__(self, label):
        self.label = label

    def predict(self, query):
        return self.label


class CapturingClassifier(StubClassifier):
    def __init__(self, label):
        super().__init__(label)
        self.queries = []

    def predict(self, query):
        self.queries.append(query)
        return super().predict(query)


class ScoredClassifier(StubClassifier):
    def __init__(self, label, probabilities):
        super().__init__(label)
        self.probabilities = probabilities

    def predict_scored(self, query):
        values = sorted(self.probabilities.values(), reverse=True)
        return ClassificationPrediction(
            label=self.label,
            probabilities=self.probabilities,
            confidence=values[0],
            margin=values[0] - values[1],
            supported_labels=["simple", "moderate", "complex", "advanced"],
        )


def build_service(tmp_path, label="moderate"):
    knowledge_service = KnowledgeService(
        repository=JsonKnowledgeRepository(str(tmp_path / "knowledge.json")),
        chunker=OverlapChunker(chunk_size=80, chunk_overlap=10),
        embedder=HashEmbeddingModel(dimension=16),
    )
    kb = knowledge_service.create_knowledge_base("Workflow KB")
    knowledge_service.create_document(
        kb.id,
        "Payments runbook",
        "Wix Payments verification controls whether merchants can accept payments. " * 4,
        {"owner": "finance"},
    )
    knowledge_service.create_document(
        kb.id,
        "Invoice runbook",
        "Invoice mismatches after goods receipt should be escalated to finance operations. " * 4,
        {"owner": "procurement"},
    )
    answer_service = AdaptiveRAGAnswerService(
        router=AdaptiveRouter(StubClassifier(label)),
        generator=ExtractiveGenerator(),
        knowledge_service=knowledge_service,
        bm25_weight=0.65,
        dense_weight=0.35,
    )
    return answer_service, kb


def test_l1_direct_generation_uses_local_generator(tmp_path):
    service, _ = build_service(tmp_path, label="simple")

    result = service.answer("What is Wix Payments?", AnswerOptions(mode="direct"))

    assert result.contexts == []
    assert result.metadata["route_level"] == "l1_direct"
    assert result.metadata["retrieval_used"] is False
    assert result.metadata["generator"] == "extractive"
    assert result.metadata["generation_status"] == "completed"
    assert result.metadata["actual_generator"] == {"provider": "Local", "model": "extractive"}
    assert "prompt_preview" in result.metadata
    assert any(step["step"] == "Prompt builder" for step in result.metadata["trace_steps"])
    assert any(step["step"] == "Generator execution" for step in result.metadata["trace_steps"])


def test_stream_rejects_a_request_cancelled_before_preparation(tmp_path):
    service, _ = build_service(tmp_path, label="simple")
    token = CancellationToken("request-cancelled")
    token.cancel("Stopped before preparation.")

    async def collect():
        return [
            event
            async for event in service.answer_stream(
                "What is Wix Payments?",
                AnswerOptions(mode="direct", cancellation_token=token),
            )
        ]

    with pytest.raises(AnswerCancelled, match="Stopped before preparation"):
        asyncio.run(collect())


def test_adaptive_simple_routes_to_l1(tmp_path):
    service, kb = build_service(tmp_path, label="simple")

    result = service.answer("What is Wix Payments?", AnswerOptions(mode="adaptive", knowledge_base_id=kb.id))

    assert result.metadata["complexity_label"] == "simple"
    assert result.metadata["route_level"] == "l1_direct"
    assert result.contexts == []


def test_stream_trace_explains_generator_fallback(tmp_path):
    service, _ = build_service(tmp_path, label="simple")
    service.model_gateway = FallbackGateway()

    async def collect():
        return [
            event
            async for event in service.answer_stream(
                "What is Wix Payments?",
                AnswerOptions(
                    mode="direct",
                    chat_configuration={
                        "generator_deployment_id": "remote-gemma",
                        "generator_provider": "OpenRouter",
                        "generator_model": "google/gemma-free",
                        "fallback_deployment_ids": ["model-local-extractive"],
                    },
                ),
            )
        ]

    events = asyncio.run(collect())
    completed = next(event for event in events if event.type == "completed")
    metadata = completed.data["result"].metadata
    fallback_trace = next(step for step in metadata["trace_steps"] if step["step"] == "Generator fallback")
    execution_trace = next(step for step in metadata["trace_steps"] if step["step"] == "Generator execution")

    assert fallback_trace["metadata"]["deployment_id"] == "remote-gemma"
    assert execution_trace["status"] == "warning"
    assert metadata["configured_generator"]["model"] == "google/gemma-free"
    assert metadata["actual_generator"]["model"] == "extractive"
    assert metadata["fallback_used"] is True


def test_adaptive_moderate_routes_to_l2_with_bm25_contexts(tmp_path):
    service, kb = build_service(tmp_path, label="moderate")

    result = service.answer(
        "How do I handle invoice mismatch after goods receipt?",
        AnswerOptions(mode="adaptive", knowledge_base_id=kb.id, retrieval_mode="bm25", top_k=2),
    )

    assert result.metadata["route_level"] == "l2_simple_rag"
    assert result.metadata["retrieval_used"] is True
    assert result.metadata["retrieval_mode"] == "bm25"
    assert len(result.contexts) == 2
    assert result.contexts[0].rank == 1
    assert "document_id" in result.contexts[0].document.metadata


def test_adaptive_complex_routes_to_l3_with_multi_step_retrieval(tmp_path):
    service, kb = build_service(tmp_path, label="complex")
    question = "After Wix Payments verification, how should invoice mismatches be handled and who owns follow-up approvals?"

    result = service.answer(
        question,
        AnswerOptions(mode="adaptive", knowledge_base_id=kb.id, retrieval_mode="bm25", top_k=2),
    )

    assert result.metadata["complexity_label"] == "complex"
    assert result.metadata["route_level"] == "l3_complex_rag"
    assert result.metadata["route_label"] == "L3 Complex RAG"
    assert result.metadata["retrieval_used"] is True
    assert result.metadata["multi_step"] is True
    assert result.metadata["decomposed_queries"][-1] == question
    assert len(result.metadata["retrieval_steps"]) == len(result.metadata["decomposed_queries"])
    assert result.metadata["aggregation_summary"]["selected_context_count"] == len(result.contexts)
    assert result.metadata["aggregation_summary"]["unique_context_count"] <= result.metadata["aggregation_summary"]["candidate_count"]
    assert len(result.contexts) == 2
    assert result.contexts[0].document.metadata["source_subquery"]
    assert result.contexts[0].document.metadata["retrieval_step"].startswith("step-")
    assert result.contexts[0].document.metadata["original_rank"] >= 1
    assert result.contexts[0].document.metadata["aggregated_rank"] == 1
    assert any(step["step"] == "Query decomposition" for step in result.metadata["trace_steps"])
    assert any(step["step"] == "Multi-step retrieval" for step in result.metadata["trace_steps"])
    assert any(step["step"] == "Context aggregation" for step in result.metadata["trace_steps"])


def test_low_classifier_confidence_escalates_adaptive_route(tmp_path):
    service, kb = build_service(tmp_path, label="simple")
    service.router.classifier = ScoredClassifier(
        "simple",
        {"simple": 0.42, "moderate": 0.38, "complex": 0.12, "advanced": 0.08},
    )

    result = service.answer(
        "What is Wix Payments?",
        AnswerOptions(mode="adaptive", knowledge_base_id=kb.id, retrieval_mode="bm25", top_k=1),
    )

    assert result.metadata["predicted_complexity_label"] == "simple"
    assert result.metadata["routed_complexity_label"] == "moderate"
    assert result.metadata["route_level"] == "l2_simple_rag"
    assert result.metadata["classifier_escalated"] is True


def test_explicit_l4_runs_bounded_agent_tool_loop(tmp_path):
    service, kb = build_service(tmp_path, label="advanced")
    service.model_gateway = RuntimeControlGateway(
        label="advanced",
        planner_text='{"action":"search_knowledge_base","arguments":{"query":"invoice approval"},"reason":"Gather evidence"}',
    )

    result = service.answer(
        "Research the dependent workflow and exception approvals.",
        AnswerOptions(
            mode="advanced_rag",
            knowledge_base_id=kb.id,
            retrieval_mode="bm25",
            top_k=2,
            chat_configuration={
                "planner_deployment_id": "planner-test",
                "metadata": {"agent_max_iterations": 2, "agent_max_tool_calls": 2},
            },
        ),
    )

    assert result.metadata["route_level"] == "l4_advanced_rag"
    assert result.metadata["agent_tool_calls"] == 2
    assert result.metadata["agent_stopping_reason"] == "iteration_limit"
    assert result.metadata["agent"]["completion_status"] == "bounded_completion"
    assert result.contexts


def test_l4_stream_emits_agent_progress_before_generation(tmp_path):
    service, kb = build_service(tmp_path, label="advanced")
    service.model_gateway = RuntimeControlGateway(
        planner_text='{"action":"search_knowledge_base","arguments":{"query":"invoice approval"},"reason":"Gather evidence"}',
    )

    async def collect():
        return [
            event
            async for event in service.answer_stream(
                "Research the exception workflow.",
                AnswerOptions(
                    mode="advanced_rag",
                    knowledge_base_id=kb.id,
                    retrieval_mode="bm25",
                    top_k=1,
                    chat_configuration={
                        "planner_deployment_id": "planner-test",
                        "metadata": {"agent_max_iterations": 1, "agent_max_tool_calls": 1},
                    },
                ),
            )
        ]

    events = asyncio.run(collect())
    progress_index = next(
        index
        for index, event in enumerate(events)
        if event.type == "trace" and event.data["step"].startswith("L4 agent iteration")
    )
    first_delta_index = next(index for index, event in enumerate(events) if event.type == "delta")

    assert progress_index < first_delta_index
    assert any(event.type == "sources" and event.data["contexts"] for event in events[:first_delta_index])


def test_l2_dense_uses_repository_embedding_search(tmp_path):
    service, kb = build_service(tmp_path, label="moderate")

    result = service.answer(
        "invoice mismatch finance operations",
        AnswerOptions(mode="simple_rag", knowledge_base_id=kb.id, retrieval_mode="dense", top_k=1),
    )

    assert result.metadata["route_level"] == "l2_simple_rag"
    assert len(result.contexts) == 1
    assert result.contexts[0].mode == "dense"
    assert result.contexts[0].document.metadata["query_embedding_model"] == "hash-embedding-384"


def test_l2_hybrid_combines_bm25_and_dense_scores(tmp_path):
    service, kb = build_service(tmp_path, label="complex")

    result = service.answer(
        "payments verification accept payments",
        AnswerOptions(mode="simple_rag", knowledge_base_id=kb.id, retrieval_mode="hybrid", top_k=2),
    )

    assert result.metadata["retrieval_mode"] == "hybrid"
    assert len(result.contexts) == 2
    assert all(context.mode == "hybrid" for context in result.contexts)
    diagnostics = result.metadata["retrieval_diagnostics"]
    assert diagnostics["mode"] == "hybrid"
    assert diagnostics["candidate_count"] >= len(result.contexts)
    assert all("bm25_raw_score" in item for item in diagnostics["candidates"])
    assert all("dense_normalized_score" in item for item in diagnostics["candidates"])
    assert all("hybrid_score" in item for item in diagnostics["candidates"])
    assert sum(1 for item in diagnostics["candidates"] if item["selected"]) == len(result.contexts)


class RuntimeControlGateway:
    def __init__(self, *, label="complex", planner_text='["Check verification", "Check approval owner"]', fail_classifier=False):
        self.label = label
        self.planner_text = planner_text
        self.fail_classifier = fail_classifier
        self.classified_queries = []
        self.planner_calls = []

    def classify_sync(self, query, deployment_id, **kwargs):
        self.classified_queries.append((query, deployment_id, kwargs))
        if self.fail_classifier:
            raise ModelFarmError("classifier unavailable")
        return ModelClassificationResult(
            label=self.label,
            deployment_id=deployment_id,
            provider="Local",
            model="query_classifier_t5",
            metadata={"runtime": "test-classifier", "latency_ms": 1.2},
        )

    def generate_sync(self, messages, deployment_id, **kwargs):
        self.planner_calls.append((messages, deployment_id, kwargs))
        return ModelGenerationResult(
            text=self.planner_text,
            deployment_id=deployment_id,
            provider="OpenRouter",
            model="planner-model",
            status="completed",
            metadata={"runtime": "test-planner", "latency_ms": 2.4},
        )


def test_selected_classifier_controls_adaptive_route(tmp_path):
    service, kb = build_service(tmp_path, label="simple")
    gateway = RuntimeControlGateway(label="complex")
    service.model_gateway = gateway

    result = service.answer(
        "Compare the verification and approval workflow.",
        AnswerOptions(
            mode="adaptive",
            knowledge_base_id=kb.id,
            retrieval_mode="bm25",
            top_k=2,
            chat_configuration={"metadata": {"classifier_deployment_id": "model-local-t5-classifier"}},
        ),
    )

    assert result.metadata["route_level"] == "l3_complex_rag"
    assert result.metadata["configured_classifier"]["deployment_id"] == "model-local-t5-classifier"
    assert result.metadata["actual_classifier"]["runtime"] == "test-classifier"
    assert result.metadata["classifier_fallback_used"] is False
    assert gateway.classified_queries[0][1] == "model-local-t5-classifier"


def test_selected_classifier_failure_falls_back_to_process_default(tmp_path):
    service, kb = build_service(tmp_path, label="moderate")
    service.model_gateway = RuntimeControlGateway(fail_classifier=True)

    result = service.answer(
        "How should the invoice mismatch be handled?",
        AnswerOptions(
            mode="adaptive",
            knowledge_base_id=kb.id,
            retrieval_mode="bm25",
            chat_configuration={"metadata": {"classifier_deployment_id": "broken-classifier"}},
        ),
    )

    classifier_trace = next(
        step for step in result.metadata["trace_steps"] if step["step"] == "Query complexity classifier"
    )
    assert result.metadata["route_level"] == "l2_simple_rag"
    assert result.metadata["classifier_fallback_used"] is True
    assert result.metadata["actual_classifier"]["runtime"] == "process_default"
    assert classifier_trace["status"] == "warning"


def test_selected_planner_controls_l3_decomposition(tmp_path):
    service, kb = build_service(tmp_path, label="complex")
    gateway = RuntimeControlGateway()
    service.model_gateway = gateway
    original = "Compare verification with invoice approval."

    result = service.answer(
        original,
        AnswerOptions(
            mode="complex_rag",
            knowledge_base_id=kb.id,
            retrieval_mode="bm25",
            top_k=2,
            chat_configuration={"planner_deployment_id": "remote-planner"},
        ),
    )

    assert result.metadata["decomposed_queries"] == [
        "Check verification",
        "Check approval owner",
        original,
    ]
    assert result.metadata["configured_planner"]["deployment_id"] == "remote-planner"
    assert result.metadata["actual_planner"]["runtime"] == "test-planner"
    assert result.metadata["planner_fallback_used"] is False
    assert gateway.planner_calls[0][2]["capability"] == "planner"
    assert gateway.planner_calls[0][2]["context"].purpose == "query_decomposition"


def test_invalid_planner_output_falls_back_to_deterministic_decomposition(tmp_path):
    service, kb = build_service(tmp_path, label="complex")
    service.model_gateway = RuntimeControlGateway(planner_text="not-json")
    original = "Compare verification with invoice approval."

    result = service.answer(
        original,
        AnswerOptions(
            mode="complex_rag",
            knowledge_base_id=kb.id,
            retrieval_mode="bm25",
            top_k=2,
            chat_configuration={"planner_deployment_id": "invalid-planner"},
        ),
    )

    planner_trace = next(step for step in result.metadata["trace_steps"] if step["step"] == "Query decomposition")
    assert result.metadata["decomposed_queries"][-1] == original
    assert result.metadata["planner_fallback_used"] is True
    assert result.metadata["actual_planner"]["runtime"] == "deterministic_rules"
    assert planner_trace["status"] == "warning"


def test_query_embedding_is_inherited_from_active_index_and_ignored_for_bm25(tmp_path):
    service, kb = build_service(tmp_path, label="moderate")

    dense = service.answer(
        "invoice mismatch",
        AnswerOptions(
            mode="simple_rag",
            knowledge_base_id=kb.id,
            retrieval_mode="dense",
            chat_configuration={"metadata": {"query_embedding_deployment_id": "stale-override"}},
        ),
    )
    bm25 = service.answer(
        "invoice mismatch",
        AnswerOptions(
            mode="simple_rag",
            knowledge_base_id=kb.id,
            retrieval_mode="bm25",
            chat_configuration={"metadata": {"query_embedding_deployment_id": "stale-override"}},
        ),
    )

    assert dense.metadata["query_embedding"]["used"] is True
    assert dense.metadata["query_embedding"]["deployment_id"] != "stale-override"
    assert dense.metadata["query_embedding"]["active_index_version_id"]
    assert bm25.metadata["query_embedding"]["used"] is False


def test_citation_validation_is_advisory_and_can_be_disabled(tmp_path):
    service, kb = build_service(tmp_path, label="moderate")

    enabled = service.answer(
        "How should the invoice mismatch be handled?",
        AnswerOptions(
            mode="simple_rag",
            knowledge_base_id=kb.id,
            retrieval_mode="bm25",
            chat_configuration={"citations_enabled": True},
        ),
    )
    disabled = service.answer(
        "How should the invoice mismatch be handled?",
        AnswerOptions(
            mode="simple_rag",
            knowledge_base_id=kb.id,
            retrieval_mode="bm25",
            chat_configuration={"citations_enabled": False},
        ),
    )

    assert enabled.answer
    assert enabled.metadata["citation_validation"]["status"] == "warning"
    assert set(enabled.metadata["citation_sources"]) == {
        context.document.metadata["source_label"] for context in enabled.contexts
    }
    first_source = next(iter(enabled.metadata["citation_sources"].values()))
    assert first_source["context_id"]
    assert first_source["document_id"]
    assert disabled.metadata["citation_validation"]["status"] == "disabled"


def test_follow_up_uses_standalone_query_for_classifier_and_retrieval(tmp_path):
    service, kb = build_service(tmp_path, label="moderate")
    classifier = CapturingClassifier("moderate")
    service.router.classifier = classifier
    invoice_document = next(
        document for document in service.knowledge_service.list_documents(kb.id)
        if document.title == "Invoice runbook"
    )

    result = service.answer(
        "Who approves it?",
        AnswerOptions(
            mode="simple_rag",
            knowledge_base_id=kb.id,
            retrieval_mode="bm25",
            top_k=1,
            conversation_history=[
                {"role": "user", "content": "Explain the invoice mismatch workflow."},
                {"role": "assistant", "content": "Finance operations reviews invoice mismatches."},
            ],
        ),
    )

    assert result.question == "Who approves it?"
    assert result.metadata["query_rewritten"] is True
    assert result.metadata["history_exchange_count"] == 1
    assert "invoice mismatch workflow" in result.metadata["standalone_query"]
    assert classifier.queries == [result.metadata["standalone_query"]]
    assert result.contexts[0].document.metadata["document_id"] == invoice_document.id
    assert any(step["step"] == "Conversation context" for step in result.metadata["trace_steps"])
    assert any(step["step"] == "Query reformulation" for step in result.metadata["trace_steps"])


def test_conversation_awareness_can_be_disabled(tmp_path):
    service, _ = build_service(tmp_path, label="simple")

    result = service.answer(
        "Who approves it?",
        AnswerOptions(
            mode="direct",
            conversation_history=[
                {"role": "user", "content": "Explain the invoice mismatch workflow."},
                {"role": "assistant", "content": "Finance operations reviews invoice mismatches."},
            ],
            chat_configuration={"metadata": {"conversation_awareness_enabled": False}},
        ),
    )

    assert result.metadata["conversation_awareness_enabled"] is False
    assert result.metadata["history_exchange_count"] == 0
    assert result.metadata["query_rewritten"] is False
    assert result.metadata["reformulation_strategy"] == "disabled"


class FallbackGateway:
    async def stream(self, *args, **kwargs):
        failed_attempt = {
            "deployment_id": "remote-gemma",
            "deployment_name": "Google Gemma",
            "provider": "OpenRouter",
            "model": "google/gemma-free",
            "fallback_index": 0,
            "error_category": "rate_limit",
            "retryable": True,
            "error": "Provider returned 429",
        }
        yield ModelStreamEvent(
            "model_fallback",
            {
                **failed_attempt,
                "next_deployment_id": "model-local-extractive",
                "next_deployment_name": "Local Extractive",
                "next_provider": "Local",
                "next_model": "extractive",
            },
        )
        yield ModelStreamEvent("delta", {"text": "Fallback answer"})
        yield ModelStreamEvent(
            "model_completed",
            {
                "deployment_id": "model-local-extractive",
                "provider": "Local",
                "model": "extractive",
                "status": "completed",
                "input_tokens": 3,
                "output_tokens": 2,
                "metadata": {
                    "runtime": "deterministic-extractive",
                    "fallback_index": 1,
                    "fallback_attempts": [failed_attempt],
                },
            },
        )


def test_l2_retrieval_can_be_filtered_to_selected_documents(tmp_path):
    service, kb = build_service(tmp_path, label="moderate")
    documents = service.knowledge_service.list_documents(kb.id)
    invoice_document = next(document for document in documents if document.title == "Invoice runbook")

    result = service.answer(
        "payments verification accept payments",
        AnswerOptions(
            mode="simple_rag",
            knowledge_base_id=kb.id,
            document_ids=[invoice_document.id],
            retrieval_mode="bm25",
            top_k=3,
        ),
    )

    assert result.metadata["document_filter_ids"] == [invoice_document.id]
    assert result.metadata["document_filter_count"] == 1
    assert result.contexts
    assert {context.document.metadata["document_id"] for context in result.contexts} == {invoice_document.id}
