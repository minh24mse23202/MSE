from aragbiz.answering import AdaptiveRAGAnswerService, AnswerOptions
from aragbiz.generation import ExtractiveGenerator
from aragbiz.knowledge import HashEmbeddingModel, KnowledgeService, OverlapChunker
from aragbiz.knowledge_store import JsonKnowledgeRepository
from aragbiz.routing import AdaptiveRouter


class StubClassifier:
    def __init__(self, label):
        self.label = label

    def predict(self, query):
        return self.label


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


def test_adaptive_simple_routes_to_l1(tmp_path):
    service, kb = build_service(tmp_path, label="simple")

    result = service.answer("What is Wix Payments?", AnswerOptions(mode="adaptive", knowledge_base_id=kb.id))

    assert result.metadata["complexity_label"] == "simple"
    assert result.metadata["route_level"] == "l1_direct"
    assert result.contexts == []


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