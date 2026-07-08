from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aragbiz.answering import AdaptiveRAGAnswerService, AnswerOptions, AnsweringError
from aragbiz.config import load_config
from aragbiz.chat import ChatConfigurationRecord, ChatConversationRecord, ChatMessageRecord, ChatSection
from aragbiz.factory import build_chat_service, build_knowledge_service, build_sample_pipeline
from aragbiz.feedback import append_feedback
from aragbiz.schemas import RetrievalMode
from aragbiz.knowledge import (
    IngestionSummary,
    KnowledgeBaseRecord,
    KnowledgeProcessingError,
    ProcessingTraceStep,
    StoredKnowledgeChunk,
    StoredKnowledgeDocument,
)

config = load_config()
pipeline = build_sample_pipeline(config)
knowledge_service = build_knowledge_service(config)
chat_service = build_chat_service(config)
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
    metadata: Optional[Dict[str, Any]] = None


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    conversation_id: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    chat_configuration_id: Optional[str] = None
    chat_configuration: Optional[ChatConfigurationRequest] = None
    mode: Literal["adaptive", "direct", "simple_rag", "complex_rag"] = "adaptive"
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = Field(4, ge=1, le=50)


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
    created_at: str


class KnowledgeBaseConfigurationRequest(BaseModel):
    chunking_strategy: str = "sliding_window_overlap"
    chunk_size: int = Field(800, ge=1)
    chunk_overlap: int = Field(120, ge=0)
    embedding_provider: str = "Local"
    embedding_model: str = "hash-embedding-384"


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


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    service = AdaptiveRAGAnswerService(
        router=pipeline.router,
        generator=pipeline.generator,
        knowledge_service=knowledge_service,
        bm25_weight=config.bm25_weight,
        dense_weight=config.dense_weight,
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
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                chat_configuration=chat_configuration,
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
    except (AnsweringError, KnowledgeProcessingError, ValueError) as exc:
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
    return _chat_configuration_response(chat_service.create_configuration(**_chat_configuration_payload(request)))


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


@app.post("/knowledge-bases/{knowledge_base_id}/sources/upload", response_model=IngestionResponse)
async def upload_knowledge_source(knowledge_base_id: str, request: Request) -> IngestionResponse:
    try:
        knowledge_service.get_knowledge_base(knowledge_base_id)
        form = await request.form()
        files = form.getlist("files") or form.getlist("file")
        if not files:
            raise HTTPException(status_code=400, detail="Upload at least one file using the 'files' field.")
        summaries: List[IngestionSummary] = []
        for upload in files:
            filename = getattr(upload, "filename", "") or "uploaded.txt"
            content = await upload.read()
            summaries.append(knowledge_service.ingest_uploaded_file(knowledge_base_id, filename, content))
        return _merge_summaries(knowledge_base_id, summaries)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KnowledgeProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/knowledge-bases/{knowledge_base_id}/sources/website", response_model=IngestionResponse)
def ingest_website_source(knowledge_base_id: str, request: WebsiteSourceRequest) -> IngestionResponse:
    try:
        knowledge_service.get_knowledge_base(knowledge_base_id)
        return _ingestion_response(knowledge_service.ingest_website(knowledge_base_id, request.url))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KnowledgeProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/knowledge-bases/{knowledge_base_id}/reindex", response_model=IngestionResponse)
def reindex_knowledge_base(knowledge_base_id: str) -> IngestionResponse:
    try:
        knowledge_service.get_knowledge_base(knowledge_base_id)
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
def list_knowledge_chunks(knowledge_base_id: str, limit: int = 100) -> List[KnowledgeChunkResponse]:
    try:
        knowledge_service.get_knowledge_base(knowledge_base_id)
        return [_chunk_response(chunk) for chunk in knowledge_service.list_chunks(knowledge_base_id, limit=limit)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/knowledge-bases/{knowledge_base_id}/processing-trace", response_model=List[ProcessingTraceResponse])
def get_processing_trace(knowledge_base_id: str) -> List[ProcessingTraceResponse]:
    try:
        knowledge_service.get_knowledge_base(knowledge_base_id)
        return [_trace_response(step) for step in knowledge_service.processing_trace(knowledge_base_id)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _resolve_answer_chat_configuration(request: AnswerRequest) -> tuple[Optional[str], Dict[str, Any]]:
    if request.chat_configuration_id:
        record = chat_service.get_configuration(request.chat_configuration_id)
        if request.chat_configuration is not None:
            payload = _chat_configuration_payload(request.chat_configuration)
            payload["id"] = record.id
            return record.id, payload
        return record.id, _chat_configuration_snapshot(record)
    if request.chat_configuration is not None:
        return None, _chat_configuration_payload(request.chat_configuration)
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
        metadata=record.metadata,
        created_at=record.created_at,
        updated_at=record.updated_at,
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
        created_at=record.created_at,
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
