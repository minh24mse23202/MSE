from __future__ import annotations

from pathlib import Path
from typing import Optional

from aragbiz.classifier import (
    HeuristicQueryClassifier,
    HuggingFaceQueryClassifier,
    NaiveBayesQueryClassifier,
    QueryClassifier,
    T5QueryClassifier,
)
from aragbiz.answering import AdaptiveRAGAnswerService
from aragbiz.auth import AuthService, JsonAuthRepository, PostgresAuthRepository
from aragbiz.chat import ChatService, JsonChatRepository, PostgresChatRepository
from aragbiz.config import AppConfig, load_config
from aragbiz.data import load_documents_jsonl, load_qac_jsonl, records_to_documents
from aragbiz.evaluation import EvaluationService, JsonEvaluationRepository, PostgresEvaluationRepository
from aragbiz.generation import ExtractiveGenerator
from aragbiz.knowledge import KnowledgeService, OverlapChunker, build_embedder
from aragbiz.knowledge_store import JsonKnowledgeRepository, PostgresKnowledgeRepository
from aragbiz.jobs import JobService, JsonJobRepository, LocalBlobStore, PostgresJobRepository
from aragbiz.model_farm import (
    JsonModelFarmRepository,
    ModelFarmService,
    ModelGateway,
    PostgresModelFarmRepository,
)
from aragbiz.pipeline import RAGPipeline
from aragbiz.retrieval import InMemoryHybridRetriever
from aragbiz.routing import AdaptiveRouter, RouterConfig
from aragbiz.ragxplain import RagxplainRunner


def build_sample_pipeline(config: Optional[AppConfig] = None) -> RAGPipeline:
    config = config or load_config()
    dataset_path = _existing_path(config.sample_dataset, config.fallback_sample_dataset)
    records = load_qac_jsonl(dataset_path)
    if Path(config.kb_corpus).exists():
        documents = load_documents_jsonl(config.kb_corpus)
    else:
        documents = records_to_documents(records)
    classifier = build_query_classifier(config)
    router = AdaptiveRouter(
        classifier,
        RouterConfig(
            simple_top_k=config.simple_top_k,
            moderate_top_k=config.moderate_top_k,
            complex_top_k=config.complex_top_k,
        ),
    )
    retriever = InMemoryHybridRetriever(
        documents,
        bm25_weight=config.bm25_weight,
        dense_weight=config.dense_weight,
    )
    generator = ExtractiveGenerator(max_context_chars=config.max_context_chars)
    return RAGPipeline(router=router, retriever=retriever, generator=generator)


def existing_dataset_path(config: AppConfig) -> str:
    return _existing_path(config.sample_dataset, config.fallback_sample_dataset)


def build_query_classifier(config: AppConfig) -> QueryClassifier:
    model_path = Path(config.classifier_model_path)
    if config.use_trained_classifier and model_path.exists():
        if model_path.is_dir():
            if _is_t5_artifact(model_path):
                return T5QueryClassifier(model_path)
            return HuggingFaceQueryClassifier(model_path)
        return NaiveBayesQueryClassifier.load(model_path)
    fallback_path = Path(config.classifier_fallback_model_path)
    if config.use_trained_classifier and fallback_path.exists():
        return NaiveBayesQueryClassifier.load(fallback_path)
    return HeuristicQueryClassifier()


def build_model_farm_service(config: Optional[AppConfig] = None) -> ModelFarmService:
    config = config or load_config()
    if config.knowledge_backend.lower() == "json":
        repository = JsonModelFarmRepository(config.model_farm_json_store)
    else:
        try:
            repository = PostgresModelFarmRepository(config.knowledge_database_url)
            repository.initialize()
        except Exception:
            repository = JsonModelFarmRepository(config.model_farm_json_store)
    return ModelFarmService(
        repository,
        global_monthly_budget_usd=config.global_model_budget_usd,
        secret_key=config.model_secret_key or config.jwt_secret,
    )


def build_model_gateway(
    config: Optional[AppConfig] = None,
    *,
    model_farm_service: Optional[ModelFarmService] = None,
) -> ModelGateway:
    return ModelGateway(model_farm_service or build_model_farm_service(config))


def build_auth_service(config: Optional[AppConfig] = None) -> AuthService:
    config = config or load_config()
    if config.knowledge_backend.lower() == "json":
        repository = JsonAuthRepository(config.auth_json_store)
    else:
        try:
            repository = PostgresAuthRepository(config.knowledge_database_url)
            repository.initialize()
        except Exception:
            repository = JsonAuthRepository(config.auth_json_store)
    return AuthService(
        repository,
        jwt_secret=config.jwt_secret,
        token_ttl_seconds=config.access_token_ttl_seconds,
        auth_required=config.auth_required,
    )


def build_job_service(config: Optional[AppConfig] = None) -> JobService:
    config = config or load_config()
    if config.knowledge_backend.lower() == "json":
        repository = JsonJobRepository(config.jobs_json_store)
    else:
        try:
            repository = PostgresJobRepository(config.knowledge_database_url)
            repository.initialize()
        except Exception:
            repository = JsonJobRepository(config.jobs_json_store)
    return JobService(repository)


def build_blob_store(config: Optional[AppConfig] = None) -> LocalBlobStore:
    config = config or load_config()
    return LocalBlobStore(config.blob_store)


def build_knowledge_service(
    config: Optional[AppConfig] = None,
    *,
    model_farm_service: Optional[ModelFarmService] = None,
    model_gateway: Optional[ModelGateway] = None,
) -> KnowledgeService:
    config = config or load_config()
    if config.knowledge_backend.lower() == "json":
        repository = JsonKnowledgeRepository(config.knowledge_json_store)
    else:
        try:
            repository = PostgresKnowledgeRepository(
                config.knowledge_database_url,
                embedding_dimension=config.embedding_dimension,
            )
        except Exception:
            repository = JsonKnowledgeRepository(config.knowledge_json_store)
    return KnowledgeService(
        repository=repository,
        chunker=OverlapChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap),
        embedder=build_embedder(
            model_name=config.embedding_model,
            dimension=config.embedding_dimension,
            use_sentence_transformers=config.use_sentence_transformers,
        ),
        model_farm_service=model_farm_service,
        model_gateway=model_gateway,
    )


def build_chat_service(config: Optional[AppConfig] = None) -> ChatService:
    config = config or load_config()
    if config.knowledge_backend.lower() == "json":
        repository = JsonChatRepository(config.chat_json_store)
    else:
        try:
            repository = PostgresChatRepository(config.knowledge_database_url)
        except Exception:
            repository = JsonChatRepository(config.chat_json_store)
    return ChatService(repository=repository)


def build_evaluation_service(
    config: Optional[AppConfig] = None,
    *,
    knowledge_service: Optional[KnowledgeService] = None,
    pipeline: Optional[RAGPipeline] = None,
    model_farm_service: Optional[ModelFarmService] = None,
    model_gateway: Optional[ModelGateway] = None,
) -> EvaluationService:
    config = config or load_config()
    knowledge_service = knowledge_service or build_knowledge_service(config)
    pipeline = pipeline or build_sample_pipeline(config)
    if config.knowledge_backend.lower() == "json":
        repository = JsonEvaluationRepository(config.evaluation_json_store)
    else:
        try:
            repository = PostgresEvaluationRepository(config.knowledge_database_url)
        except Exception:
            repository = JsonEvaluationRepository(config.evaluation_json_store)
    answer_service = AdaptiveRAGAnswerService(
        router=pipeline.router,
        generator=pipeline.generator,
        knowledge_service=knowledge_service,
        bm25_weight=config.bm25_weight,
        dense_weight=config.dense_weight,
        model_farm_service=model_farm_service,
        model_gateway=model_gateway,
    )
    return EvaluationService(
        repository=repository,
        answer_service=answer_service,
        dataset_path=existing_dataset_path(config),
        ragxplain_runner=RagxplainRunner(
            root=config.ragxplain_root,
            results_root=config.ragxplain_results_root,
            judge=config.ragxplain_judge,
            timeout_seconds=config.ragxplain_timeout_seconds,
        ),
    )


def _is_t5_artifact(model_path: Path) -> bool:
    config_path = model_path / "config.json"
    if not config_path.exists():
        return False
    return '"model_type": "t5"' in config_path.read_text(encoding="utf-8")


def _existing_path(primary: str, fallback: str) -> str:
    if Path(primary).exists():
        return primary
    return fallback
