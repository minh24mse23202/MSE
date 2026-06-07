from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from aragbiz.config import load_config
from aragbiz.factory import build_sample_pipeline
from aragbiz.feedback import append_feedback

config = load_config()
pipeline = build_sample_pipeline(config)
app = FastAPI(title="Adaptive RAG Business Workflow QA", version="0.1.0")


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1)


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


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
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
        metadata=result.metadata,
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest) -> Dict[str, str]:
    payload = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    append_feedback(config.feedback_store, payload)
    return {"status": "recorded"}
