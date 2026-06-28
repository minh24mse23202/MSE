from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from aragbiz.config import load_config
from aragbiz.factory import build_knowledge_service, build_sample_pipeline
from aragbiz.feedback import append_feedback
from aragbiz.knowledge import IngestionSummary, KnowledgeBaseRecord, KnowledgeProcessingError, StoredKnowledgeChunk, StoredKnowledgeDocument

config = load_config()
pipeline = build_sample_pipeline(config)
knowledge_service = build_knowledge_service(config)
app = FastAPI(title="Adaptive RAG Business Workflow QA", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)
    knowledge_base_id: Optional[str] = None


class ContextResponse(BaseModel):
    id: str
    score: float
    rank: int
    text: str
    metadata: Dict[str, Any]


class AnswerResponse(BaseModel):
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


class KnowledgeBaseCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""


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
    knowledge_base_metadata: Dict[str, Any] = {}
    if request.knowledge_base_id:
        try:
            selected_kb = knowledge_service.get_knowledge_base(request.knowledge_base_id)
            knowledge_base_metadata = {
                "knowledge_base_id": selected_kb.id,
                "knowledge_base_name": selected_kb.name,
            }
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    result = pipeline.answer(request.question)
    return AnswerResponse(
        question=result.question,
        answer=result.answer,
        contexts=[
            ContextResponse(
                id=context.document.id,
                score=context.score,
                rank=context.rank,
                text=context.document.text,
                metadata=context.document.metadata,
            )
            for context in result.contexts
        ],
        metadata={**result.metadata, **knowledge_base_metadata},
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> Dict[str, str]:
    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    append_feedback(config.feedback_store, payload)
    return {"status": "recorded"}


@app.get("/knowledge-bases", response_model=List[KnowledgeBaseResponse])
def list_knowledge_bases() -> List[KnowledgeBaseResponse]:
    return [_knowledge_base_response(record) for record in knowledge_service.list_knowledge_bases()]


@app.post("/knowledge-bases", response_model=KnowledgeBaseResponse)
def create_knowledge_base(request: KnowledgeBaseCreateRequest) -> KnowledgeBaseResponse:
    record = knowledge_service.create_knowledge_base(request.name, request.description)
    return _knowledge_base_response(record)


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
            )
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
