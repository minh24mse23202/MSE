from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from aragbiz.answering import AdaptiveRAGAnswerService, AnswerOptions, AnsweringError
from aragbiz.auth import AuthenticationError, UserRecord
from aragbiz.config import load_config
from aragbiz.chat import ChatConfigurationRecord, ChatConversationRecord, ChatMessageRecord, ChatSection
from aragbiz.evaluation import EvaluationCaseRecord, EvaluationRunConfig, EvaluationRunRecord
from aragbiz.factory import (
    build_auth_service,
    build_blob_store,
    build_chat_service,
    build_evaluation_service,
    build_job_service,
    build_knowledge_service,
    build_model_farm_service,
    build_model_gateway,
    build_sample_pipeline,
)
from aragbiz.feedback import append_feedback
from aragbiz.schemas import RetrievalMode
from aragbiz.knowledge import (
    IngestionSummary,
    KnowledgeBaseRecord,
    KnowledgeIndexVersionRecord,
    KnowledgeProcessingError,
    ProcessingTraceStep,
    StoredKnowledgeChunk,
    StoredKnowledgeDocument,
)
from aragbiz.ragxplain import RagxplainError, RagxplainUnavailableError
from aragbiz.jobs import BackgroundJob, JobError, job_idempotency_key
from aragbiz.model_farm import ModelConnection, ModelDeployment, ModelFarmError, ModelUsageEvent, _gateway_model_name

config = load_config()
pipeline = build_sample_pipeline(config)
model_farm_service = build_model_farm_service(config)
model_gateway = build_model_gateway(config, model_farm_service=model_farm_service)
knowledge_service = build_knowledge_service(
    config,
    model_farm_service=model_farm_service,
    model_gateway=model_gateway,
)
chat_service = build_chat_service(config)
auth_service = build_auth_service(config)
job_service = build_job_service(config)
blob_store = build_blob_store(config)
evaluation_service = build_evaluation_service(
    config,
    knowledge_service=knowledge_service,
    pipeline=pipeline,
    model_farm_service=model_farm_service,
    model_gateway=model_gateway,
)
app = FastAPI(title="Adaptive RAG Business Workflow QA", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatConfigurationRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    generator_provider: str = "Local"
    generator_model: str = "extractive"
    response_structure: str = "Concise answer with supporting details"
    tone: str = "Professional"
    humor_level: int = Field(0, ge=0, le=5)
    system_prompt: str = ""
    predefined_prompt: str = ""
    generator_deployment_id: str = ""
    fallback_deployment_ids: List[str] = Field(default_factory=list)
    reranker_deployment_id: str = ""
    planner_deployment_id: str = ""
    generation_parameters: Dict[str, Any] = Field(default_factory=lambda: {"temperature": 0.2, "max_tokens": 500})
    citations_enabled: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatConfigurationPatchRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    generator_provider: Optional[str] = None
    generator_model: Optional[str] = None
    response_structure: Optional[str] = None
    tone: Optional[str] = None
    humor_level: Optional[int] = Field(None, ge=0, le=5)
    system_prompt: Optional[str] = None
    predefined_prompt: Optional[str] = None
    generator_deployment_id: Optional[str] = None
    fallback_deployment_ids: Optional[List[str]] = None
    reranker_deployment_id: Optional[str] = None
    planner_deployment_id: Optional[str] = None
    generation_parameters: Optional[Dict[str, Any]] = None
    citations_enabled: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    document_ids: List[str] = Field(default_factory=list)
    chat_configuration_id: Optional[str] = None
    chat_configuration: Optional[ChatConfigurationRequest] = None
    mode: Literal["adaptive", "direct", "simple_rag", "complex_rag"] = "adaptive"
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = Field(4, ge=1, le=50)
    request_id: str = ""


class ContextResponse(BaseModel):
    id: str
    score: float
    rank: int
    mode: str
    text: str
    metadata: Dict[str, Any]


class AnswerResponse(BaseModel):
    conversation_id: Optional[str] = None
    question: str
    answer: str
    contexts: List[ContextResponse]
    metadata: Dict[str, Any]


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: str
    comment: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatConversationCreateRequest(BaseModel):
    title: str = "New chat"
    knowledge_base_id: Optional[str] = None
    chat_configuration_id: Optional[str] = None
    route_mode: str = "adaptive"
    retrieval_mode: str = "hybrid"
    top_k: int = Field(4, ge=1, le=50)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatConversationPatchRequest(BaseModel):
    title: Optional[str] = None
    pinned: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatConfigurationResponse(BaseModel):
    id: str
    name: str
    description: str
    generator_provider: str
    generator_model: str
    response_structure: str
    tone: str
    humor_level: int
    system_prompt: str
    predefined_prompt: str
    generator_deployment_id: str
    fallback_deployment_ids: List[str]
    reranker_deployment_id: str
    planner_deployment_id: str
    generation_parameters: Dict[str, Any]
    citations_enabled: bool
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class ChatConversationResponse(BaseModel):
    id: str
    title: str
    pinned: bool
    knowledge_base_id: Optional[str]
    chat_configuration_id: Optional[str]
    route_mode: str
    retrieval_mode: str
    top_k: int
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class ChatMessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    contexts: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    status: str
    request_id: str
    created_at: str
    updated_at: str


class EvaluationRunRequest(BaseModel):
    name: str = "Adaptive vs Static L2 evaluation"
    knowledge_base_id: str = ""
    chat_configuration_id: Optional[str] = None
    chat_configuration: Optional[ChatConfigurationRequest] = None
    judge_deployment_id: str = ""
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = Field(4, ge=1, le=50)
    limit: int = Field(20, ge=0, le=100)
    compare_baseline: bool = True
    run_ragxplain: bool = False


class EvaluationRunResponse(BaseModel):
    id: str
    name: str
    dataset_name: str
    status: str
    knowledge_base_id: str
    knowledge_base_name: str
    chat_configuration_id: Optional[str]
    retrieval_mode: str
    top_k: int
    limit: int
    compare_baseline: bool
    metrics: Dict[str, Any]
    baseline_metrics: Dict[str, Any]
    route_distribution: Dict[str, int]
    baseline_route_distribution: Dict[str, int]
    metadata: Dict[str, Any]
    error: Optional[str]
    created_at: str
    finished_at: str


class EvaluationCaseResponse(BaseModel):
    id: str
    run_id: str
    record_id: str
    question: str
    expected_answer: str
    complexity_label: str
    adaptive_answer: str
    static_answer: str
    adaptive_contexts: List[Dict[str, Any]]
    static_contexts: List[Dict[str, Any]]
    adaptive_metadata: Dict[str, Any]
    static_metadata: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: str


class KnowledgeBaseConfigurationRequest(BaseModel):
    chunking_strategy: str = "sliding_window_overlap"
    chunk_size: int = Field(800, ge=1)
    chunk_overlap: int = Field(120, ge=0)
    embedding_provider: str = "Local"
    embedding_model: str = "hash-embedding-384"
    embedding_deployment_id: str = ""
    external_processing_allowed: bool = False


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    configuration: Optional[KnowledgeBaseConfigurationRequest] = None


class WebsiteSourceRequest(BaseModel):
    url: str = Field(..., min_length=1)


class KnowledgeDocumentRequest(BaseModel):
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeBaseResponse(BaseModel):
    id: str
    name: str
    description: str
    status: str
    document_count: int
    chunk_count: int
    embedding_model: str
    created_at: str
    updated_at: str
    metadata: Dict[str, Any]
    error: Optional[str] = None


class KnowledgeDocumentResponse(BaseModel):
    id: str
    knowledge_base_id: str
    source_id: str
    title: str
    content_hash: str
    text: str
    metadata: Dict[str, Any]


class KnowledgeChunkResponse(BaseModel):
    id: str
    knowledge_base_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    metadata: Dict[str, Any]
    embedding_model: str
    embedding_dimension: int
    has_embedding: bool


class ProcessingTraceResponse(BaseModel):
    step: str
    status: str
    detail: str
    metadata: Dict[str, Any]
    started_at: str
    finished_at: str


class IngestionResponse(BaseModel):
    knowledge_base_id: str
    source_id: Optional[str]
    status: str
    documents_added: int
    documents_skipped: int
    chunks_added: int
    error: Optional[str] = None


class AuthSignupRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)
    first_name: str = ""
    last_name: str = ""


class AuthLoginRequest(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: str
    email: str
    first_name: str
    last_name: str
    role: str
    active: bool


class AuthTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class ModelDeploymentFromTemplateRequest(BaseModel):
    template_id: str = Field(..., min_length=1)
    connection_id: str = ""
    name: str = ""
    model: str = ""
    model_id: str = ""
    api_base: str = ""
    capabilities: List[str] = Field(default_factory=list)
    credential_env_refs: Dict[str, str] = Field(default_factory=dict)
    credential_secrets: Dict[str, str] = Field(default_factory=dict)
    default_parameters: Dict[str, Any] = Field(default_factory=dict)
    limits: Dict[str, Any] = Field(default_factory=dict)
    pricing: Dict[str, Any] = Field(default_factory=dict)
    monthly_budget_usd: float = Field(0, ge=0)
    hard_budget: bool = True
    enabled: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelDeploymentDraftTestRequest(ModelDeploymentFromTemplateRequest):
    deployment_id: str = ""


class ModelDeploymentPatchRequest(BaseModel):
    name: Optional[str] = None
    api_base: Optional[str] = None
    credential_env_refs: Optional[Dict[str, str]] = None
    credential_secrets: Optional[Dict[str, str]] = None
    default_parameters: Optional[Dict[str, Any]] = None
    limits: Optional[Dict[str, Any]] = None
    pricing: Optional[Dict[str, Any]] = None
    monthly_budget_usd: Optional[float] = Field(None, ge=0)
    hard_budget: Optional[bool] = None
    enabled: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelDeploymentResponse(BaseModel):
    id: str
    name: str
    provider: str
    model: str
    model_id: str
    capabilities: List[str]
    connection_id: str
    connection_name: str
    access_path: str
    gateway_model: str
    api_base: str
    credential_status: Dict[str, Any]
    default_parameters: Dict[str, Any]
    limits: Dict[str, Any]
    pricing: Dict[str, Any]
    monthly_budget_usd: float
    hard_budget: bool
    locality: str
    enabled: bool
    health_status: str
    last_health_check: str
    last_error: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class ModelConnectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    provider: Literal["openrouter", "openai", "gemini", "ollama", "vllm"]
    access_path: Literal["experimentation", "production", "local"]
    api_base: str = ""
    credential_env_refs: Dict[str, str] = Field(default_factory=dict)
    credential_secrets: Dict[str, str] = Field(default_factory=dict)
    locality: Literal["local", "remote"] = "remote"
    enabled: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelConnectionPatchRequest(BaseModel):
    name: Optional[str] = None
    api_base: Optional[str] = None
    credential_env_refs: Optional[Dict[str, str]] = None
    credential_secrets: Optional[Dict[str, str]] = None
    enabled: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelConnectionResponse(BaseModel):
    id: str
    name: str
    provider: str
    access_path: str
    api_base: str
    credential_status: Dict[str, Any]
    locality: str
    enabled: bool
    health_status: str
    last_health_check: str
    last_error: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str


class ModelUsageResponse(BaseModel):
    id: str
    deployment_id: str
    provider: str
    model: str
    connection_id: str
    access_path: str
    gateway_model: str
    capability: str
    purpose: str
    status: str
    fallback_index: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    error_code: str
    error_category: str
    error: str
    created_at: str


class JobResponse(BaseModel):
    id: str
    job_type: str
    status: str
    progress: Dict[str, Any]
    result: Dict[str, Any]
    error: str
    attempts: int
    created_at: str
    updated_at: str
    finished_at: str


class KnowledgeIndexVersionResponse(BaseModel):
    id: str
    knowledge_base_id: str
    status: str
    chunking_configuration: Dict[str, Any]
    embedding_deployment_id: str
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int
    document_count: int
    chunk_count: int
    error: str
    created_at: str
    activated_at: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/signup", response_model=AuthTokenResponse)
def auth_signup(request: AuthSignupRequest) -> AuthTokenResponse:
    try:
        user = auth_service.signup(request.email, request.password, request.first_name, request.last_name)
        return AuthTokenResponse(access_token=auth_service.issue_token(user), user=_user_response(user))
    except AuthenticationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/auth/login", response_model=AuthTokenResponse)
def auth_login(request: AuthLoginRequest) -> AuthTokenResponse:
    try:
        user, token = auth_service.login(request.email, request.password)
        return AuthTokenResponse(access_token=token, user=_user_response(user))
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/auth/me", response_model=UserResponse)
def auth_me(authorization: str = Header(default="")) -> UserResponse:
    try:
        return _user_response(auth_service.current_user(authorization))
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/model-farm/providers", response_model=List[Dict[str, Any]])
def list_model_providers(authorization: str = Header(default="")) -> List[Dict[str, Any]]:
    _require_user(authorization)
    return model_farm_service.providers()


@app.get("/model-farm/connections", response_model=List[ModelConnectionResponse])
def list_model_connections(
    provider: str = "",
    enabled: Optional[bool] = None,
    authorization: str = Header(default=""),
) -> List[ModelConnectionResponse]:
    _require_admin(authorization)
    return [
        _model_connection_response(item)
        for item in model_farm_service.list_connections(provider=provider, enabled=enabled)
    ]


@app.post("/model-farm/connections", response_model=ModelConnectionResponse)
def create_model_connection(
    request: ModelConnectionCreateRequest,
    authorization: str = Header(default=""),
) -> ModelConnectionResponse:
    _require_admin(authorization)
    try:
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        return _model_connection_response(model_farm_service.create_connection(payload))
    except ModelFarmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/model-farm/connections/{connection_id}", response_model=ModelConnectionResponse)
def get_model_connection(connection_id: str, authorization: str = Header(default="")) -> ModelConnectionResponse:
    _require_admin(authorization)
    try:
        return _model_connection_response(model_farm_service.get_connection(connection_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/model-farm/connections/{connection_id}", response_model=ModelConnectionResponse)
def update_model_connection(
    connection_id: str,
    request: ModelConnectionPatchRequest,
    authorization: str = Header(default=""),
) -> ModelConnectionResponse:
    _require_admin(authorization)
    try:
        payload = request.model_dump(exclude_unset=True) if hasattr(request, "model_dump") else request.dict(exclude_unset=True)
        return _model_connection_response(model_farm_service.update_connection(connection_id, payload))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelFarmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/model-farm/connections/{connection_id}")
def delete_model_connection(connection_id: str, authorization: str = Header(default="")) -> Dict[str, str]:
    _require_admin(authorization)
    try:
        model_farm_service.delete_connection(connection_id)
        return {"status": "deleted"}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelFarmError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/model-farm/connections/{connection_id}/test", response_model=Dict[str, Any])
async def test_model_connection(connection_id: str, authorization: str = Header(default="")) -> Dict[str, Any]:
    _require_admin(authorization)
    try:
        result = await asyncio.to_thread(model_farm_service.test_connection, connection_id)
        result["connection"] = _model_connection_response(result["connection"])
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/model-farm/connections/{connection_id}/available-models", response_model=List[Dict[str, Any]])
async def list_connection_models(connection_id: str, authorization: str = Header(default="")) -> List[Dict[str, Any]]:
    _require_admin(authorization)
    try:
        return await asyncio.to_thread(model_farm_service.available_models, connection_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelFarmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/model-farm/deployments", response_model=List[ModelDeploymentResponse])
def list_model_deployments(
    capability: str = "",
    enabled: Optional[bool] = None,
    authorization: str = Header(default=""),
) -> List[ModelDeploymentResponse]:
    _require_user(authorization)
    return [_model_deployment_response(item) for item in model_farm_service.list_deployments(capability=capability, enabled=enabled)]


@app.post("/model-farm/deployments/from-template", response_model=ModelDeploymentResponse)
def create_model_deployment_from_template(
    request: ModelDeploymentFromTemplateRequest,
    authorization: str = Header(default=""),
) -> ModelDeploymentResponse:
    _require_admin(authorization)
    try:
        payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
        template_id = str(payload.pop("template_id"))
        model_id = str(payload.pop("model_id", "") or "")
        if model_id:
            payload["model"] = model_id
        return _model_deployment_response(model_farm_service.create_deployment_from_template(template_id, payload))
    except ModelFarmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/model-farm/deployments/test-draft", response_model=Dict[str, Any])
async def test_model_deployment_draft(
    request: ModelDeploymentDraftTestRequest,
    authorization: str = Header(default=""),
) -> Dict[str, Any]:
    _require_admin(authorization)
    try:
        payload = request.model_dump(exclude_unset=True) if hasattr(request, "model_dump") else request.dict(exclude_unset=True)
        deployment_id = str(payload.pop("deployment_id", "") or "")
        template_id = str(payload.pop("template_id", "") or "")
        model_id = str(payload.pop("model_id", "") or "")
        if model_id:
            payload["model"] = model_id
        if deployment_id:
            payload.pop("model", None)
            payload.pop("capabilities", None)
            deployment = model_farm_service.draft_update_deployment(deployment_id, payload)
        else:
            deployment = model_farm_service.draft_deployment_from_template(template_id, payload)
        result = await model_gateway.test_draft_deployment(deployment)
        result["deployment"] = _model_deployment_response(result["deployment"])
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelFarmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/model-farm/deployments/{deployment_id}", response_model=ModelDeploymentResponse)
def get_model_deployment(deployment_id: str, authorization: str = Header(default="")) -> ModelDeploymentResponse:
    _require_user(authorization)
    try:
        return _model_deployment_response(model_farm_service.get_deployment(deployment_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/model-farm/deployments/{deployment_id}", response_model=ModelDeploymentResponse)
def update_model_deployment(
    deployment_id: str,
    request: ModelDeploymentPatchRequest,
    authorization: str = Header(default=""),
) -> ModelDeploymentResponse:
    _require_admin(authorization)
    payload = request.model_dump(exclude_unset=True) if hasattr(request, "model_dump") else request.dict(exclude_unset=True)
    try:
        return _model_deployment_response(model_farm_service.update_deployment(deployment_id, payload))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelFarmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/model-farm/deployments/{deployment_id}")
def delete_model_deployment(deployment_id: str, authorization: str = Header(default="")) -> Dict[str, str]:
    _require_admin(authorization)
    try:
        model_farm_service.delete_deployment(deployment_id)
        return {"status": "deleted"}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelFarmError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/model-farm/deployments/{deployment_id}/test", response_model=Dict[str, Any])
async def test_model_deployment(deployment_id: str, authorization: str = Header(default="")) -> Dict[str, Any]:
    _require_admin(authorization)
    try:
        result = await model_gateway.test_deployment(deployment_id)
        result["deployment"] = _model_deployment_response(result["deployment"])
        return result
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelFarmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/model-farm/usage", response_model=List[ModelUsageResponse])
def list_model_usage(
    deployment_id: str = "",
    purpose: str = "",
    limit: int = Query(500, ge=1, le=5000),
    authorization: str = Header(default=""),
) -> List[ModelUsageResponse]:
    _require_admin(authorization)
    return [_model_usage_response(item) for item in model_farm_service.list_usage(deployment_id=deployment_id, purpose=purpose, limit=limit)]


@app.get("/model-farm/usage/summary", response_model=Dict[str, Any])
def model_usage_summary(authorization: str = Header(default="")) -> Dict[str, Any]:
    _require_admin(authorization)
    return model_farm_service.usage_summary()


@app.get("/jobs", response_model=List[JobResponse])
def list_jobs(status: str = "", limit: int = Query(100, ge=1, le=1000), authorization: str = Header(default="")) -> List[JobResponse]:
    _require_user(authorization)
    try:
        return [_job_response(item) for item in job_service.list(status=status, limit=limit)]
    except JobError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, authorization: str = Header(default="")) -> JobResponse:
    _require_user(authorization)
    try:
        return _job_response(job_service.get(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_job(job_id: str, authorization: str = Header(default="")) -> JobResponse:
    _require_user(authorization)
    try:
        return _job_response(job_service.cancel(job_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest, authorization: str = Header(default="")) -> AnswerResponse:
    user = _require_user(authorization)
    service = AdaptiveRAGAnswerService(
        router=pipeline.router,
        generator=pipeline.generator,
        knowledge_service=knowledge_service,
        bm25_weight=config.bm25_weight,
        dense_weight=config.dense_weight,
        model_farm_service=model_farm_service,
        model_gateway=model_gateway,
    )
    try:
        if request.conversation_id:
            chat_service.get_conversation(request.conversation_id)
        chat_configuration_id, chat_configuration = _resolve_answer_chat_configuration(request)
        result = service.answer(
            request.question,
            AnswerOptions(
                mode=request.mode,
                knowledge_base_id=request.knowledge_base_id,
                document_ids=request.document_ids,
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                chat_configuration=chat_configuration,
                request_id=request.request_id or f"request-{uuid.uuid4().hex}",
                user_id=user.id,
                conversation_id=request.conversation_id or "",
            ),
        )
        result.metadata["chat_configuration_id"] = chat_configuration_id
        result.metadata["chat_configuration"] = chat_configuration
        conversation = chat_service.ensure_conversation_for_question(
            request.conversation_id,
            request.question,
            knowledge_base_id=request.knowledge_base_id,
            route_mode=request.mode,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            chat_configuration_id=chat_configuration_id,
            chat_configuration=chat_configuration,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (AnsweringError, KnowledgeProcessingError, ModelFarmError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    contexts = [
        ContextResponse(
            id=context.document.id,
            score=context.score,
            rank=context.rank,
            mode=context.mode,
            text=context.document.text,
            metadata=context.document.metadata,
        )
        for context in result.contexts
    ]
    context_payloads = [
        context.model_dump() if hasattr(context, "model_dump") else context.dict()
        for context in contexts
    ]
    chat_service.append_answer_exchange(
        conversation.id,
        question=result.question,
        answer=result.answer,
        contexts=context_payloads,
        metadata=result.metadata,
    )
    return AnswerResponse(
        conversation_id=conversation.id,
        question=result.question,
        answer=result.answer,
        contexts=contexts,
        metadata=result.metadata,
    )


@app.post("/answer/stream")
async def answer_stream(request: AnswerRequest, authorization: str = Header(default="")) -> StreamingResponse:
    user = _require_user(authorization)
    request_id = request.request_id or f"request-{uuid.uuid4().hex}"
    request.request_id = request_id

    async def events():
        partial_answer = ""
        context_payloads: List[Dict[str, Any]] = []
        trace_steps: List[Dict[str, Any]] = []
        assistant_metadata: Dict[str, Any] = {"question": request.question, "request_id": request_id, "trace_steps": trace_steps}
        assistant_message = None
        yield _sse(
            "started",
            {
                "request_id": request_id,
                "conversation_id": request.conversation_id or "",
                "user_message_id": "",
                "assistant_message_id": "",
                "status": "accepted",
            },
        )
        try:
            if request.conversation_id:
                await asyncio.to_thread(chat_service.get_conversation, request.conversation_id)
            chat_configuration_id, chat_configuration = await asyncio.to_thread(_resolve_answer_chat_configuration, request)
            conversation = await asyncio.to_thread(
                chat_service.ensure_conversation_for_question,
                request.conversation_id,
                request.question,
                knowledge_base_id=request.knowledge_base_id,
                route_mode=request.mode,
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                chat_configuration_id=chat_configuration_id,
                chat_configuration=chat_configuration,
            )
            user_message = await asyncio.to_thread(
                chat_service.append_message,
                conversation.id,
                "user",
                request.question,
                contexts=[],
                metadata={},
                status="completed",
                request_id=request_id,
            )
            assistant_metadata = {
                "question": request.question,
                "request_id": request_id,
                "chat_configuration_id": chat_configuration_id,
                "chat_configuration": chat_configuration,
                "trace_steps": trace_steps,
            }
            assistant_message = await asyncio.to_thread(
                chat_service.append_message,
                conversation.id,
                "assistant",
                "",
                contexts=[],
                metadata=assistant_metadata,
                status="pending",
                request_id=request_id,
            )
            yield _sse(
                "started",
                {
                    "request_id": request_id,
                    "conversation_id": conversation.id,
                    "user_message_id": user_message.id,
                    "assistant_message_id": assistant_message.id,
                    "status": "persisted",
                },
            )
            service = AdaptiveRAGAnswerService(
                router=pipeline.router,
                generator=pipeline.generator,
                knowledge_service=knowledge_service,
                bm25_weight=config.bm25_weight,
                dense_weight=config.dense_weight,
                model_farm_service=model_farm_service,
                model_gateway=model_gateway,
            )
            await asyncio.to_thread(chat_service.update_message, assistant_message.id, status="streaming", metadata=assistant_metadata)
            async for event in service.answer_stream(
                request.question,
                AnswerOptions(
                    mode=request.mode,
                    knowledge_base_id=request.knowledge_base_id,
                    document_ids=request.document_ids,
                    retrieval_mode=request.retrieval_mode,
                    top_k=request.top_k,
                    chat_configuration=chat_configuration,
                    request_id=request_id,
                    user_id=user.id,
                    conversation_id=conversation.id,
                ),
            ):
                if event.type == "trace":
                    trace_steps.append(event.data)
                    assistant_metadata = {**assistant_metadata, "trace_steps": trace_steps}
                    await asyncio.to_thread(
                        chat_service.update_message,
                        assistant_message.id,
                        content=partial_answer,
                        contexts=context_payloads,
                        metadata=assistant_metadata,
                        status="streaming",
                    )
                    yield _sse("trace", event.data)
                elif event.type == "sources":
                    contexts = [_context_response(context) for context in event.data.get("contexts", [])]
                    context_payloads = [_model_dump(context) for context in contexts]
                    await asyncio.to_thread(
                        chat_service.update_message,
                        assistant_message.id,
                        content=partial_answer,
                        contexts=context_payloads,
                        metadata=assistant_metadata,
                        status="streaming",
                    )
                    yield _sse("sources", {"contexts": context_payloads})
                elif event.type == "delta":
                    text = str(event.data.get("text") or "")
                    partial_answer += text
                    await asyncio.to_thread(
                        chat_service.update_message,
                        assistant_message.id,
                        content=partial_answer,
                        contexts=context_payloads,
                        metadata=assistant_metadata,
                        status="streaming",
                    )
                    yield _sse("delta", {"text": text})
                elif event.type == "model_completed":
                    yield _sse("model_completed", event.data)
                elif event.type == "completed":
                    result = event.data["result"]
                    response = _answer_response_from_result(
                        conversation.id,
                        result,
                        chat_configuration_id=chat_configuration_id,
                        chat_configuration=chat_configuration,
                        user_message_id=user_message.id,
                        assistant_message_id=assistant_message.id,
                    )
                    await asyncio.to_thread(
                        chat_service.update_message,
                        assistant_message.id,
                        content=response.answer,
                        contexts=[_model_dump(context) for context in response.contexts],
                        metadata=response.metadata,
                        status="completed",
                    )
                    yield _sse("completed", _model_dump(response))
        except asyncio.CancelledError:
            if assistant_message is not None:
                await asyncio.to_thread(
                    chat_service.update_message,
                    assistant_message.id,
                    content=partial_answer,
                    contexts=context_payloads,
                    metadata={**assistant_metadata, "error": "Streaming request was cancelled by the client."},
                    status="cancelled",
                )
            raise
        except HTTPException as exc:
            if assistant_message is not None:
                await asyncio.to_thread(
                    chat_service.update_message,
                    assistant_message.id,
                    content=partial_answer,
                    contexts=context_payloads,
                    metadata={**assistant_metadata, "error": str(exc.detail)},
                    status="failed",
                )
            yield _sse("error", {"status": exc.status_code, "detail": exc.detail, "request_id": request_id})
        except (AnsweringError, KnowledgeProcessingError, ModelFarmError, ValueError) as exc:
            if assistant_message is not None:
                await asyncio.to_thread(
                    chat_service.update_message,
                    assistant_message.id,
                    content=partial_answer,
                    contexts=context_payloads,
                    metadata={**assistant_metadata, "error": str(exc)},
                    status="failed",
                )
            yield _sse("error", {"status": 400, "detail": str(exc), "request_id": request_id})
        except Exception as exc:
            if assistant_message is not None:
                await asyncio.to_thread(
                    chat_service.update_message,
                    assistant_message.id,
                    content=partial_answer,
                    contexts=context_payloads,
                    metadata={**assistant_metadata, "error": str(exc)},
                    status="failed",
                )
            yield _sse("error", {"status": 500, "detail": str(exc), "request_id": request_id})

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> Dict[str, str]:
    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    append_feedback(config.feedback_store, payload)
    return {"status": "recorded"}


@app.get("/chat/configurations", response_model=List[ChatConfigurationResponse])
def list_chat_configurations() -> List[ChatConfigurationResponse]:
    return [_chat_configuration_response(record) for record in chat_service.list_configurations()]


@app.post("/chat/configurations", response_model=ChatConfigurationResponse)
def create_chat_configuration(request: ChatConfigurationRequest) -> ChatConfigurationResponse:
    try:
        payload = _chat_configuration_payload(request)
        _validate_chat_configuration_payload(payload)
        return _chat_configuration_response(chat_service.create_configuration(**payload))
    except (KeyError, ModelFarmError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/chat/configurations/{configuration_id}", response_model=ChatConfigurationResponse)
def get_chat_configuration(configuration_id: str) -> ChatConfigurationResponse:
    try:
        return _chat_configuration_response(chat_service.get_configuration(configuration_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/chat/configurations/{configuration_id}", response_model=ChatConfigurationResponse)
def update_chat_configuration(configuration_id: str, request: ChatConfigurationPatchRequest) -> ChatConfigurationResponse:
    payload = request.model_dump(exclude_unset=True) if hasattr(request, "model_dump") else request.dict(exclude_unset=True)
    try:
        current = _chat_configuration_snapshot(chat_service.get_configuration(configuration_id))
        _validate_chat_configuration_payload({**current, **payload})
        return _chat_configuration_response(chat_service.update_configuration(configuration_id, **payload))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/chat/configurations/{configuration_id}")
def delete_chat_configuration(configuration_id: str) -> Dict[str, str]:
    try:
        chat_service.delete_configuration(configuration_id)
        return {"status": "deleted"}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
@app.get("/chat/conversations", response_model=List[ChatConversationResponse])
def list_chat_conversations(query: str = "", section: Optional[ChatSection] = None) -> List[ChatConversationResponse]:
    return [_chat_conversation_response(record) for record in chat_service.list_conversations(query=query, section=section)]


@app.post("/chat/conversations", response_model=ChatConversationResponse)
def create_chat_conversation(request: ChatConversationCreateRequest) -> ChatConversationResponse:
    return _chat_conversation_response(
        chat_service.create_conversation(
            request.title,
            knowledge_base_id=request.knowledge_base_id,
            route_mode=request.route_mode,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            metadata=request.metadata,
        )
    )


@app.get("/chat/conversations/{conversation_id}", response_model=ChatConversationResponse)
def get_chat_conversation(conversation_id: str) -> ChatConversationResponse:
    try:
        return _chat_conversation_response(chat_service.get_conversation(conversation_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/chat/conversations/{conversation_id}", response_model=ChatConversationResponse)
def update_chat_conversation(conversation_id: str, request: ChatConversationPatchRequest) -> ChatConversationResponse:
    try:
        return _chat_conversation_response(
            chat_service.update_conversation(
                conversation_id,
                title=request.title,
                pinned=request.pinned,
                metadata=request.metadata,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/chat/conversations/{conversation_id}")
def delete_chat_conversation(conversation_id: str) -> Dict[str, str]:
    try:
        chat_service.delete_conversation(conversation_id)
        return {"status": "deleted"}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/chat/conversations/{conversation_id}/messages", response_model=List[ChatMessageResponse])
def list_chat_messages(conversation_id: str) -> List[ChatMessageResponse]:
    try:
        return [_chat_message_response(record) for record in chat_service.list_messages(conversation_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/evaluation/runs", response_model=List[EvaluationRunResponse])
def list_evaluation_runs() -> List[EvaluationRunResponse]:
    return [_evaluation_run_response(record) for record in evaluation_service.list_runs()]


@app.post("/evaluation/runs", response_model=EvaluationRunResponse)
def create_evaluation_run(request: EvaluationRunRequest) -> EvaluationRunResponse:
    try:
        chat_configuration_id, chat_configuration = _resolve_chat_configuration(request.chat_configuration_id, request.chat_configuration)
        if request.judge_deployment_id:
            model_farm_service.resolve(request.judge_deployment_id, "judge")
        run = evaluation_service.run(
            EvaluationRunConfig(
                name=request.name,
                knowledge_base_id=request.knowledge_base_id,
                chat_configuration_id=chat_configuration_id,
                chat_configuration=chat_configuration,
                judge_deployment_id=request.judge_deployment_id,
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                limit=request.limit,
                compare_baseline=request.compare_baseline,
                run_ragxplain=request.run_ragxplain,
            )
        )
        return _evaluation_run_response(run)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (AnsweringError, KnowledgeProcessingError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/evaluation/runs/{run_id}", response_model=EvaluationRunResponse)
def get_evaluation_run(run_id: str) -> EvaluationRunResponse:
    try:
        return _evaluation_run_response(evaluation_service.get_run(run_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/evaluation/runs/{run_id}/cases", response_model=List[EvaluationCaseResponse])
def list_evaluation_cases(run_id: str) -> List[EvaluationCaseResponse]:
    try:
        return [_evaluation_case_response(record) for record in evaluation_service.list_cases(run_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/evaluation/runs/{run_id}/ragxplain/overall-insights", response_model=Dict[str, Any])
def get_ragxplain_overall_insights(run_id: str) -> Dict[str, Any]:
    try:
        return evaluation_service.get_ragxplain_insights(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelFarmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ModelFarmError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RagxplainError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/evaluation/ragxplain/viewer", response_class=FileResponse)
def get_ragxplain_viewer() -> FileResponse:
    try:
        return FileResponse(
            evaluation_service.ragxplain_viewer_path(),
            media_type="text/html",
            headers={"Cache-Control": "no-store"},
        )
    except (RagxplainUnavailableError, RagxplainError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.delete("/evaluation/runs/{run_id}")
def delete_evaluation_run(run_id: str) -> Dict[str, str]:
    try:
        evaluation_service.delete_run(run_id)
        return {"status": "deleted"}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/knowledge-bases", response_model=List[KnowledgeBaseResponse])
def list_knowledge_bases() -> List[KnowledgeBaseResponse]:
    return [_knowledge_base_response(record) for record in knowledge_service.list_knowledge_bases()]


@app.post("/knowledge-bases", response_model=KnowledgeBaseResponse)
def create_knowledge_base(request: KnowledgeBaseCreateRequest) -> KnowledgeBaseResponse:
    try:
        record = knowledge_service.create_knowledge_base(
            request.name,
            request.description,
            configuration=_configuration_payload(request.configuration),
        )
        return _knowledge_base_response(record)
    except KnowledgeProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
def get_knowledge_base(knowledge_base_id: str) -> KnowledgeBaseResponse:
    try:
        return _knowledge_base_response(knowledge_service.get_knowledge_base(knowledge_base_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/knowledge-bases/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
def update_knowledge_base(knowledge_base_id: str, request: KnowledgeBaseCreateRequest) -> KnowledgeBaseResponse:
    try:
        return _knowledge_base_response(
            knowledge_service.update_knowledge_base_details(
                knowledge_base_id,
                name=request.name,
                description=request.description,
                configuration=_configuration_payload(request.configuration),
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KnowledgeProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/knowledge-bases/{knowledge_base_id}")
def delete_knowledge_base(knowledge_base_id: str) -> Dict[str, str]:
    try:
        knowledge_service.delete_knowledge_base(knowledge_base_id)
        return {"status": "deleted"}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/knowledge-bases/{knowledge_base_id}/sources/upload")
async def upload_knowledge_source(knowledge_base_id: str, request: Request, wait: bool = True) -> Any:
    try:
        knowledge_service.get_knowledge_base(knowledge_base_id)
        form = await request.form()
        files = form.getlist("files") or form.getlist("file")
        if not files:
            raise HTTPException(status_code=400, detail="Upload at least one file using the 'files' field.")
        if not wait:
            blobs = []
            for upload in files:
                filename = getattr(upload, "filename", "") or "uploaded.txt"
                blobs.append(blob_store.put(filename, await upload.read()))
            payload = {"knowledge_base_id": knowledge_base_id, "blobs": blobs}
            job = job_service.enqueue("knowledge_upload", payload, idempotency_key=job_idempotency_key("knowledge_upload", payload))
            return JSONResponse(status_code=202, content=_model_dump(_job_response(job)))
        summaries: List[IngestionSummary] = []
        for upload in files:
            filename = getattr(upload, "filename", "") or "uploaded.txt"
            content = await upload.read()
            summaries.append(
                await asyncio.to_thread(
                    knowledge_service.ingest_uploaded_file,
                    knowledge_base_id,
                    filename,
                    content,
                )
            )
        return _merge_summaries(knowledge_base_id, summaries)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KnowledgeProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/knowledge-bases/{knowledge_base_id}/sources/website")
def ingest_website_source(knowledge_base_id: str, request: WebsiteSourceRequest, wait: bool = True) -> Any:
    try:
        knowledge_service.get_knowledge_base(knowledge_base_id)
        if not wait:
            payload = {"knowledge_base_id": knowledge_base_id, "url": request.url}
            job = job_service.enqueue("knowledge_website", payload, idempotency_key=job_idempotency_key("knowledge_website", payload))
            return JSONResponse(status_code=202, content=_model_dump(_job_response(job)))
        return _ingestion_response(knowledge_service.ingest_website(knowledge_base_id, request.url))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KnowledgeProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/knowledge-bases/{knowledge_base_id}/reindex")
def reindex_knowledge_base(knowledge_base_id: str, wait: bool = True) -> Any:
    try:
        knowledge_service.get_knowledge_base(knowledge_base_id)
        if not wait:
            configuration = knowledge_service.get_knowledge_base(knowledge_base_id).metadata.get("configuration", {})
            payload = {"knowledge_base_id": knowledge_base_id, "configuration": configuration}
            job = job_service.enqueue("knowledge_reindex", payload, idempotency_key=job_idempotency_key("knowledge_reindex", payload))
            return JSONResponse(status_code=202, content=_model_dump(_job_response(job)))
        return _ingestion_response(knowledge_service.reindex(knowledge_base_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KnowledgeProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/knowledge-bases/{knowledge_base_id}/documents", response_model=List[KnowledgeDocumentResponse])
def list_knowledge_documents(knowledge_base_id: str) -> List[KnowledgeDocumentResponse]:
    try:
        knowledge_service.get_knowledge_base(knowledge_base_id)
        return [_document_response(document) for document in knowledge_service.list_documents(knowledge_base_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/knowledge-bases/{knowledge_base_id}/documents", response_model=KnowledgeDocumentResponse)
def create_knowledge_document(knowledge_base_id: str, request: KnowledgeDocumentRequest) -> KnowledgeDocumentResponse:
    try:
        knowledge_service.get_knowledge_base(knowledge_base_id)
        return _document_response(
            knowledge_service.create_document(
                knowledge_base_id,
                title=request.title,
                text=request.text,
                metadata=request.metadata,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KnowledgeProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/knowledge-bases/{knowledge_base_id}/documents/{document_id}", response_model=KnowledgeDocumentResponse)
def get_knowledge_document(knowledge_base_id: str, document_id: str) -> KnowledgeDocumentResponse:
    try:
        return _document_response(knowledge_service.get_document(knowledge_base_id, document_id))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/knowledge-bases/{knowledge_base_id}/documents/{document_id}", response_model=KnowledgeDocumentResponse)
def update_knowledge_document(knowledge_base_id: str, document_id: str, request: KnowledgeDocumentRequest) -> KnowledgeDocumentResponse:
    try:
        return _document_response(
            knowledge_service.update_document(
                knowledge_base_id,
                document_id,
                title=request.title,
                text=request.text,
                metadata=request.metadata,
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KnowledgeProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/knowledge-bases/{knowledge_base_id}/documents/{document_id}")
def delete_knowledge_document(knowledge_base_id: str, document_id: str) -> Dict[str, str]:
    try:
        knowledge_service.delete_document(knowledge_base_id, document_id)
        return {"status": "deleted"}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/knowledge-bases/{knowledge_base_id}/chunks", response_model=List[KnowledgeChunkResponse])
def list_knowledge_chunks(knowledge_base_id: str, limit: int = 100) -> JSONResponse:
    try:
        knowledge_service.get_knowledge_base(knowledge_base_id)
        payload = [_chunk_response(chunk) for chunk in knowledge_service.list_chunks(knowledge_base_id, limit=limit)]
        return JSONResponse(content=jsonable_encoder(payload))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chunk load failed: {exc}") from exc


@app.get("/knowledge-bases/{knowledge_base_id}/processing-trace", response_model=List[ProcessingTraceResponse])
def get_processing_trace(knowledge_base_id: str) -> List[ProcessingTraceResponse]:
    try:
        knowledge_service.get_knowledge_base(knowledge_base_id)
        return [_trace_response(step) for step in knowledge_service.processing_trace(knowledge_base_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/knowledge-bases/{knowledge_base_id}/index-versions", response_model=List[KnowledgeIndexVersionResponse])
def list_knowledge_index_versions(knowledge_base_id: str) -> List[KnowledgeIndexVersionResponse]:
    try:
        return [_index_version_response(item) for item in knowledge_service.list_index_versions(knowledge_base_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _resolve_answer_chat_configuration(request: AnswerRequest) -> tuple[Optional[str], Dict[str, Any]]:
    return _resolve_chat_configuration(request.chat_configuration_id, request.chat_configuration)


def _resolve_chat_configuration(
    chat_configuration_id: Optional[str],
    chat_configuration: Optional[ChatConfigurationRequest],
) -> tuple[Optional[str], Dict[str, Any]]:
    if chat_configuration_id:
        record = chat_service.get_configuration(chat_configuration_id)
        if chat_configuration is not None:
            payload = _chat_configuration_payload(chat_configuration)
            _validate_chat_configuration_payload(payload)
            payload["id"] = record.id
            return record.id, payload
        return record.id, _chat_configuration_snapshot(record)
    if chat_configuration is not None:
        payload = _chat_configuration_payload(chat_configuration)
        _validate_chat_configuration_payload(payload)
        return None, payload
    default_record = chat_service.default_configuration()
    return default_record.id, _chat_configuration_snapshot(default_record)


def _chat_configuration_payload(configuration: ChatConfigurationRequest) -> Dict[str, Any]:
    payload = configuration.model_dump() if hasattr(configuration, "model_dump") else configuration.dict()
    payload["humor_level"] = max(0, min(int(payload.get("humor_level", 0)), 5))
    return payload


def _chat_configuration_snapshot(record: ChatConfigurationRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "name": record.name,
        "description": record.description,
        "generator_provider": record.generator_provider,
        "generator_model": record.generator_model,
        "response_structure": record.response_structure,
        "tone": record.tone,
        "humor_level": record.humor_level,
        "system_prompt": record.system_prompt,
        "predefined_prompt": record.predefined_prompt,
        "generator_deployment_id": record.generator_deployment_id,
        "fallback_deployment_ids": record.fallback_deployment_ids,
        "reranker_deployment_id": record.reranker_deployment_id,
        "planner_deployment_id": record.planner_deployment_id,
        "generation_parameters": record.generation_parameters,
        "citations_enabled": bool(record.metadata.get("citations_enabled", True)),
        "metadata": record.metadata,
    }


def _chat_configuration_response(record: ChatConfigurationRecord) -> ChatConfigurationResponse:
    return ChatConfigurationResponse(
        id=record.id,
        name=record.name,
        description=record.description,
        generator_provider=record.generator_provider,
        generator_model=record.generator_model,
        response_structure=record.response_structure,
        tone=record.tone,
        humor_level=record.humor_level,
        system_prompt=record.system_prompt,
        predefined_prompt=record.predefined_prompt,
        generator_deployment_id=record.generator_deployment_id,
        fallback_deployment_ids=record.fallback_deployment_ids,
        reranker_deployment_id=record.reranker_deployment_id,
        planner_deployment_id=record.planner_deployment_id,
        generation_parameters=record.generation_parameters,
        citations_enabled=bool(record.metadata.get("citations_enabled", True)),
        metadata=record.metadata,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )

def _evaluation_run_response(record: EvaluationRunRecord) -> EvaluationRunResponse:
    return EvaluationRunResponse(
        id=record.id,
        name=record.name,
        dataset_name=record.dataset_name,
        status=record.status,
        knowledge_base_id=record.knowledge_base_id,
        knowledge_base_name=record.knowledge_base_name,
        chat_configuration_id=record.chat_configuration_id,
        retrieval_mode=record.retrieval_mode,
        top_k=record.top_k,
        limit=record.limit,
        compare_baseline=record.compare_baseline,
        metrics=record.metrics,
        baseline_metrics=record.baseline_metrics,
        route_distribution=record.route_distribution,
        baseline_route_distribution=record.baseline_route_distribution,
        metadata=record.metadata,
        error=record.error,
        created_at=record.created_at,
        finished_at=record.finished_at,
    )


def _evaluation_case_response(record: EvaluationCaseRecord) -> EvaluationCaseResponse:
    return EvaluationCaseResponse(
        id=record.id,
        run_id=record.run_id,
        record_id=record.record_id,
        question=record.question,
        expected_answer=record.expected_answer,
        complexity_label=record.complexity_label,
        adaptive_answer=record.adaptive_answer,
        static_answer=record.static_answer,
        adaptive_contexts=record.adaptive_contexts,
        static_contexts=record.static_contexts,
        adaptive_metadata=record.adaptive_metadata,
        static_metadata=record.static_metadata,
        metrics=record.metrics,
        created_at=record.created_at,
    )


def _knowledge_base_response(record: KnowledgeBaseRecord) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(
        id=record.id,
        name=record.name,
        description=record.description,
        status=record.status,
        document_count=record.document_count,
        chunk_count=record.chunk_count,
        embedding_model=record.embedding_model,
        created_at=record.created_at,
        updated_at=record.updated_at,
        metadata=record.metadata,
        error=record.error,
    )


def _chat_conversation_response(record: ChatConversationRecord) -> ChatConversationResponse:
    return ChatConversationResponse(
        id=record.id,
        title=record.title,
        pinned=record.pinned,
        knowledge_base_id=record.knowledge_base_id,
        chat_configuration_id=record.chat_configuration_id,
        route_mode=record.route_mode,
        retrieval_mode=record.retrieval_mode,
        top_k=record.top_k,
        metadata=record.metadata,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _chat_message_response(record: ChatMessageRecord) -> ChatMessageResponse:
    return ChatMessageResponse(
        id=record.id,
        conversation_id=record.conversation_id,
        role=record.role,
        content=record.content,
        contexts=record.contexts,
        metadata=record.metadata,
        status=record.status,
        request_id=record.request_id,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _context_response(context: Any) -> ContextResponse:
    return ContextResponse(
        id=context.document.id,
        score=context.score,
        rank=context.rank,
        mode=context.mode,
        text=context.document.text,
        metadata=context.document.metadata,
    )


def _answer_response_from_result(
    conversation_id: str,
    result: Any,
    *,
    chat_configuration_id: Optional[str],
    chat_configuration: Dict[str, Any],
    user_message_id: str = "",
    assistant_message_id: str = "",
) -> AnswerResponse:
    result.metadata["chat_configuration_id"] = chat_configuration_id
    result.metadata["chat_configuration"] = chat_configuration
    if user_message_id:
        result.metadata["user_message_id"] = user_message_id
    if assistant_message_id:
        result.metadata["assistant_message_id"] = assistant_message_id
    return AnswerResponse(
        conversation_id=conversation_id,
        question=result.question,
        answer=result.answer,
        contexts=[_context_response(context) for context in result.contexts],
        metadata=result.metadata,
    )


def _document_response(document: StoredKnowledgeDocument) -> KnowledgeDocumentResponse:
    return KnowledgeDocumentResponse(
        id=document.id,
        knowledge_base_id=document.knowledge_base_id,
        source_id=document.source_id,
        title=document.title,
        content_hash=document.content_hash,
        text=document.text,
        metadata=document.metadata,
    )


def _chunk_response(chunk: StoredKnowledgeChunk) -> KnowledgeChunkResponse:
    return KnowledgeChunkResponse(
        id=chunk.id,
        knowledge_base_id=chunk.knowledge_base_id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        text=chunk.text,
        token_count=chunk.token_count,
        metadata=chunk.metadata,
        embedding_model=chunk.embedding_model,
        embedding_dimension=chunk.embedding_dimension,
        has_embedding=chunk.has_embedding,
    )


def _trace_response(step: ProcessingTraceStep) -> ProcessingTraceResponse:
    return ProcessingTraceResponse(
        step=step.step,
        status=step.status,
        detail=step.detail,
        metadata=step.metadata,
        started_at=step.started_at,
        finished_at=step.finished_at,
    )


def _ingestion_response(summary: IngestionSummary) -> IngestionResponse:
    return IngestionResponse(
        knowledge_base_id=summary.knowledge_base_id,
        source_id=summary.source_id,
        status=summary.status,
        documents_added=summary.documents_added,
        documents_skipped=summary.documents_skipped,
        chunks_added=summary.chunks_added,
        error=summary.error,
    )


def _merge_summaries(knowledge_base_id: str, summaries: List[IngestionSummary]) -> IngestionResponse:
    if not summaries:
        return IngestionResponse(
            knowledge_base_id=knowledge_base_id,
            source_id=None,
            status="empty",
            documents_added=0,
            documents_skipped=0,
            chunks_added=0,
        )
    return IngestionResponse(
        knowledge_base_id=knowledge_base_id,
        source_id=summaries[-1].source_id,
        status=summaries[-1].status,
        documents_added=sum(summary.documents_added for summary in summaries),
        documents_skipped=sum(summary.documents_skipped for summary in summaries),
        chunks_added=sum(summary.chunks_added for summary in summaries),
        error=next((summary.error for summary in summaries if summary.error), None),
    )


def _configuration_payload(configuration: Optional[KnowledgeBaseConfigurationRequest]) -> Optional[Dict[str, Any]]:
    if configuration is None:
        return None
    return configuration.model_dump() if hasattr(configuration, "model_dump") else configuration.dict()


def _validate_chat_configuration_payload(payload: Dict[str, Any]) -> None:
    deployment_id = str(payload.get("generator_deployment_id") or "").strip()
    if deployment_id:
        deployment = model_farm_service.resolve(deployment_id, "generation")
        payload["generator_provider"] = deployment.provider
        payload["generator_model"] = deployment.model
    for fallback_id in list(payload.get("fallback_deployment_ids") or []):
        model_farm_service.resolve(str(fallback_id), "generation")
    reranker_id = str(payload.get("reranker_deployment_id") or "").strip()
    if reranker_id:
        model_farm_service.resolve(reranker_id, "rerank")
    planner_id = str(payload.get("planner_deployment_id") or "").strip()
    if planner_id:
        model_farm_service.resolve(planner_id, "planner")


def _require_user(authorization: str) -> UserRecord:
    try:
        return auth_service.current_user(authorization)
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _require_admin(authorization: str) -> UserRecord:
    try:
        return auth_service.require_admin(authorization)
    except AuthenticationError as exc:
        status = 401 if "Authentication" in str(exc) or "token" in str(exc).lower() else 403
        raise HTTPException(status_code=status, detail=str(exc)) from exc


def _user_response(user: UserRecord) -> UserResponse:
    return UserResponse(
        id=user.id, email=user.email, first_name=user.first_name, last_name=user.last_name,
        role=user.role, active=user.active,
    )


def _model_deployment_response(deployment: ModelDeployment) -> ModelDeploymentResponse:
    try:
        connection = model_farm_service.connection_for_deployment(deployment, require_enabled=False)
    except (KeyError, ModelFarmError):
        connection = None
    return ModelDeploymentResponse(
        id=deployment.id, name=deployment.name, provider=deployment.provider,
        model=deployment.model, model_id=deployment.model,
        capabilities=deployment.capabilities, connection_id=deployment.connection_id,
        connection_name=connection.name if connection else "",
        access_path=connection.access_path if connection else "",
        gateway_model=_gateway_model_name(connection.provider, deployment.model, connection.api_base) if connection else deployment.model,
        api_base=connection.api_base if connection else deployment.api_base,
        credential_status=model_farm_service.credential_status(connection or deployment),
        default_parameters=deployment.default_parameters, limits=deployment.limits, pricing=deployment.pricing,
        monthly_budget_usd=deployment.monthly_budget_usd, hard_budget=deployment.hard_budget,
        locality=deployment.locality, enabled=deployment.enabled, health_status=deployment.health_status,
        last_health_check=deployment.last_health_check, last_error=deployment.last_error,
        metadata=deployment.metadata, created_at=deployment.created_at, updated_at=deployment.updated_at,
    )


def _model_connection_response(connection: ModelConnection) -> ModelConnectionResponse:
    return ModelConnectionResponse(
        id=connection.id,
        name=connection.name,
        provider=connection.provider,
        access_path=connection.access_path,
        api_base=connection.api_base,
        credential_status=model_farm_service.credential_status(connection),
        locality=connection.locality,
        enabled=connection.enabled,
        health_status=connection.health_status,
        last_health_check=connection.last_health_check,
        last_error=connection.last_error,
        metadata=connection.metadata,
        created_at=connection.created_at,
        updated_at=connection.updated_at,
    )


def _model_usage_response(event: ModelUsageEvent) -> ModelUsageResponse:
    return ModelUsageResponse(
        id=event.id, deployment_id=event.deployment_id, provider=event.provider, model=event.model,
        connection_id=event.connection_id, access_path=event.access_path, gateway_model=event.gateway_model,
        capability=event.capability, purpose=event.purpose, status=event.status,
        fallback_index=event.fallback_index,
        input_tokens=event.input_tokens, output_tokens=event.output_tokens, total_tokens=event.total_tokens,
        latency_ms=event.latency_ms, estimated_cost_usd=event.estimated_cost_usd,
        error_code=event.error_code, error_category=str(event.metadata.get("error_category") or ""),
        error=event.error, created_at=event.created_at,
    )


def _job_response(job: BackgroundJob) -> JobResponse:
    return JobResponse(
        id=job.id, job_type=job.job_type, status=job.status, progress=job.progress, result=job.result,
        error=job.error, attempts=job.attempts, created_at=job.created_at, updated_at=job.updated_at,
        finished_at=job.finished_at,
    )


def _index_version_response(version: KnowledgeIndexVersionRecord) -> KnowledgeIndexVersionResponse:
    return KnowledgeIndexVersionResponse(
        id=version.id, knowledge_base_id=version.knowledge_base_id, status=version.status,
        chunking_configuration=version.chunking_configuration,
        embedding_deployment_id=version.embedding_deployment_id,
        embedding_provider=version.embedding_provider, embedding_model=version.embedding_model,
        embedding_dimension=version.embedding_dimension, document_count=version.document_count,
        chunk_count=version.chunk_count, error=version.error, created_at=version.created_at,
        activated_at=version.activated_at,
    )


def _model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _sse(event: str, payload: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=True)}\n\n"
