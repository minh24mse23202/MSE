from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import uuid
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol


CUSTOM_CHUNKING_EXTENSIONS = {".aratxt", ".arajson", ".aramd"}
CHUNKING_STRATEGIES = {
    "fixed_size",
    "sliding_window_overlap",
    "header_based",
    "semantic",
    "recursive",
    "hierarchical_parent_child",
    "structure_aware_custom",
}
DEFAULT_KNOWLEDGE_BASE_CONFIGURATION = {
    "chunking_strategy": "sliding_window_overlap",
    "chunk_size": 800,
    "chunk_overlap": 120,
    "embedding_provider": "Local",
    "embedding_model": "hash-embedding-384",
}
LOCAL_EMBEDDING_PROVIDER = "Local"
HASH_EMBEDDING_MODEL = "hash-embedding-384"
SENTENCE_TRANSFORMER_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SUPPORTED_LOCAL_EMBEDDING_MODELS = {
    HASH_EMBEDDING_MODEL,
    SENTENCE_TRANSFORMER_MINILM_MODEL,
}
SUPPORTED_FILE_EXTENSIONS = {
    ".txt",
    ".json",
    ".jsonl",
    ".md",
    ".pdf",
    ".docx",
    *CUSTOM_CHUNKING_EXTENSIONS,
}


class KnowledgeProcessingError(ValueError):
    """Raised when a knowledge source cannot be loaded or processed."""


@dataclass(frozen=True)
class KnowledgeBaseRecord:
    id: str
    name: str
    description: str = ""
    status: str = "empty"
    document_count: int = 0
    chunk_count: int = 0
    embedding_model: str = ""
    created_at: str = ""
    updated_at: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass(frozen=True)
class DataSourceRecord:
    id: str
    knowledge_base_id: str
    source_type: str
    uri: str
    status: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeDocumentInput:
    title: str
    text: str
    source_type: str
    uri: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredKnowledgeDocument:
    id: str
    knowledge_base_id: str
    source_id: str
    title: str
    content_hash: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredKnowledgeChunk:
    id: str
    knowledge_base_id: str
    document_id: str
    chunk_index: int
    text: str
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_model: str = ""
    embedding_dimension: int = 0
    has_embedding: bool = False


@dataclass(frozen=True)
class ProcessingTraceStep:
    step: str
    status: str
    detail: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""


@dataclass(frozen=True)
class IngestionSummary:
    knowledge_base_id: str
    source_id: Optional[str]
    status: str
    documents_added: int
    documents_skipped: int
    chunks_added: int
    error: Optional[str] = None


class KnowledgeRepository(Protocol):
    def initialize(self) -> None:
        """Create storage tables or files if needed."""

    def create_knowledge_base(
        self,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBaseRecord:
        """Create and return a knowledge base."""

    def list_knowledge_bases(self) -> List[KnowledgeBaseRecord]:
        """Return knowledge bases with aggregate counts."""

    def get_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        """Return one knowledge base."""

    def update_knowledge_base(
        self,
        knowledge_base_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        embedding_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update knowledge-base status fields."""

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        """Delete a knowledge base and all child records."""

    def create_data_source(
        self,
        knowledge_base_id: str,
        source_type: str,
        uri: str,
        status: str,
        metadata: Dict[str, Any],
    ) -> DataSourceRecord:
        """Create a data-source record."""

    def existing_hashes(self, knowledge_base_id: str) -> set[str]:
        """Return known document hashes for deduplication."""

    def add_document(self, document: StoredKnowledgeDocument) -> None:
        """Persist a normalized document."""

    def list_documents(self, knowledge_base_id: str) -> List[StoredKnowledgeDocument]:
        """Return documents for a knowledge base."""

    def get_document(self, knowledge_base_id: str, document_id: str) -> StoredKnowledgeDocument:
        """Return one document in a knowledge base."""

    def update_document(self, document: StoredKnowledgeDocument) -> None:
        """Update a document record."""

    def delete_document(self, knowledge_base_id: str, document_id: str) -> None:
        """Delete a document and its chunks/embeddings."""

    def replace_chunks(self, knowledge_base_id: str, chunks: List[StoredKnowledgeChunk], embeddings: List[List[float]], model: str) -> None:
        """Replace all chunks and embeddings for a knowledge base."""

    def replace_document_chunks(
        self,
        knowledge_base_id: str,
        document_id: str,
        chunks: List[StoredKnowledgeChunk],
        embeddings: List[List[float]],
        model: str,
    ) -> None:
        """Replace chunks and embeddings for one document."""

    def append_chunks(self, chunks: List[StoredKnowledgeChunk], embeddings: List[List[float]], model: str) -> None:
        """Append chunks and embeddings."""

    def list_chunks(self, knowledge_base_id: str, limit: int = 100) -> List[StoredKnowledgeChunk]:
        """Return chunks for a knowledge base."""

    def list_ingestion_runs(self, knowledge_base_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Return recent ingestion runs for a knowledge base."""

    def record_ingestion_run(
        self,
        knowledge_base_id: str,
        status: str,
        counts: Dict[str, int],
        error: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> None:
        """Persist an ingestion run summary."""


class KnowledgeService:
    def __init__(
        self,
        repository: KnowledgeRepository,
        chunker: "OverlapChunker",
        embedder: "EmbeddingModel",
    ):
        self.repository = repository
        self.chunker = chunker
        self.default_embedder = embedder
        self._embedding_dimension = embedder.dimension
        self._embedder_cache: Dict[tuple[str, str, int], "EmbeddingModel"] = {}

    def create_knowledge_base(
        self,
        name: str,
        description: str = "",
        configuration: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBaseRecord:
        self.repository.initialize()
        normalized_configuration = validate_knowledge_base_configuration(configuration)
        return self.repository.create_knowledge_base(
            name=name,
            description=description,
            metadata={"configuration": normalized_configuration},
        )

    def list_knowledge_bases(self) -> List[KnowledgeBaseRecord]:
        self.repository.initialize()
        return self.repository.list_knowledge_bases()

    def get_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        self.repository.initialize()
        return self.repository.get_knowledge_base(knowledge_base_id)

    def update_knowledge_base_details(
        self,
        knowledge_base_id: str,
        name: str,
        description: str = "",
        configuration: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBaseRecord:
        self.repository.initialize()
        current = self.repository.get_knowledge_base(knowledge_base_id)
        metadata = dict(current.metadata)
        if configuration is not None:
            metadata["configuration"] = validate_knowledge_base_configuration(configuration)
        self.repository.update_knowledge_base(
            knowledge_base_id,
            name=name.strip(),
            description=description,
            metadata=metadata,
            error=None,
        )
        return self.repository.get_knowledge_base(knowledge_base_id)

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        self.repository.initialize()
        self.repository.get_knowledge_base(knowledge_base_id)
        self.repository.delete_knowledge_base(knowledge_base_id)

    def list_documents(self, knowledge_base_id: str) -> List[StoredKnowledgeDocument]:
        self.repository.initialize()
        return self.repository.list_documents(knowledge_base_id)

    def get_document(self, knowledge_base_id: str, document_id: str) -> StoredKnowledgeDocument:
        self.repository.initialize()
        return self.repository.get_document(knowledge_base_id, document_id)

    def list_chunks(self, knowledge_base_id: str, limit: int = 100) -> List[StoredKnowledgeChunk]:
        self.repository.initialize()
        return self.repository.list_chunks(knowledge_base_id, limit=limit)

    def processing_trace(self, knowledge_base_id: str) -> List[ProcessingTraceStep]:
        self.repository.initialize()
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        configuration = normalize_knowledge_base_configuration(knowledge_base.metadata.get("configuration"))
        documents = self.repository.list_documents(knowledge_base_id)
        chunks = self.repository.list_chunks(knowledge_base_id, limit=10000)
        runs = self.repository.list_ingestion_runs(knowledge_base_id, limit=8)
        embedded_chunks = sum(1 for chunk in chunks if chunk.has_embedding)
        chunk_embedding_models = sorted({chunk.embedding_model for chunk in chunks if chunk.embedding_model})
        embedding_dimension = next((chunk.embedding_dimension for chunk in chunks if chunk.embedding_dimension), self._embedding_dimension)
        source_types = sorted({document.metadata.get("source_type", "unknown") for document in documents})
        latest_run = runs[0] if runs else {}
        return [
            ProcessingTraceStep(
                step="Knowledge base selected",
                status=knowledge_base.status,
                detail=f"{knowledge_base.name} is active for document management.",
                metadata={
                    "knowledge_base_id": knowledge_base.id,
                    "document_count": knowledge_base.document_count,
                    "chunk_count": knowledge_base.chunk_count,
                },
                started_at=knowledge_base.created_at,
                finished_at=knowledge_base.updated_at,
            ),
            ProcessingTraceStep(
                step="Data source loading",
                status="completed" if documents else "waiting",
                detail=f"Loaded {len(documents)} document(s) from {', '.join(source_types) if source_types else 'no sources yet'}.",
                metadata={
                    "source_types": source_types,
                    "recent_runs": runs,
                },
                started_at=latest_run.get("started_at", ""),
                finished_at=latest_run.get("finished_at", ""),
            ),
            ProcessingTraceStep(
                step="Metadata extraction and deduplication",
                status="completed" if documents else "waiting",
                detail="Each document has title, source metadata, content hash and deduplication status.",
                metadata={
                    "content_hashes": [document.content_hash for document in documents],
                    "deduplication_key": "knowledge_base_id + content_hash",
                },
            ),
            ProcessingTraceStep(
                step="Chunking",
                status="completed" if chunks else "waiting",
                detail=f"Created {len(chunks)} ordered overlapping chunk(s).",
                metadata={
                    "chunk_size": configuration["chunk_size"],
                    "chunk_overlap": configuration["chunk_overlap"],
                    "configured_chunk_size": configuration["chunk_size"],
                    "configured_chunk_overlap": configuration["chunk_overlap"],
                    "chunking_strategy": configuration["chunking_strategy"],
                    "chunk_ids": [chunk.id for chunk in chunks[:20]],
                },
            ),
            ProcessingTraceStep(
                step="Embedding",
                status="completed" if embedded_chunks == len(chunks) and chunks else "waiting",
                detail=f"Embedded {embedded_chunks} of {len(chunks)} chunk(s).",
                metadata={
                    "embedding_model": knowledge_base.embedding_model or (chunk_embedding_models[0] if len(chunk_embedding_models) == 1 else ""),
                    "chunk_embedding_models": chunk_embedding_models,
                    "configured_embedding_provider": configuration["embedding_provider"],
                    "configured_embedding_model": configuration["embedding_model"],
                    "embedding_dimension": embedding_dimension,
                    "embedded_chunks": embedded_chunks,
                },
            ),
            ProcessingTraceStep(
                step="Storage",
                status="completed" if chunks else knowledge_base.status,
                detail="Documents are stored in the relational layer; chunk vectors are stored in pgVector or the JSON fallback store.",
                metadata={
                    "repository": type(self.repository).__name__,
                    "latest_run": latest_run,
                },
            ),
        ]

    def create_document(
        self,
        knowledge_base_id: str,
        title: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredKnowledgeDocument:
        self.repository.initialize()
        configuration = self._knowledge_base_configuration(knowledge_base_id)
        embedder = self._embedder_for_configuration(configuration)
        source = self.repository.create_data_source(
            knowledge_base_id=knowledge_base_id,
            source_type="manual",
            uri=title,
            status="ready",
            metadata={"created_at": utc_now(), "entrypoint": "document_editor"},
        )
        clean = clean_text(text)
        if not clean:
            raise KnowledgeProcessingError("Document text cannot be empty.")
        digest = content_hash(clean)
        if digest in self.repository.existing_hashes(knowledge_base_id):
            raise KnowledgeProcessingError("A document with the same content already exists in this knowledge base.")
        document = StoredKnowledgeDocument(
            id=f"doc-{uuid.uuid4().hex}",
            knowledge_base_id=knowledge_base_id,
            source_id=source.id,
            title=title.strip() or "Untitled document",
            content_hash=digest,
            text=clean,
            metadata={
                **(metadata or {}),
                **_processing_metadata(configuration),
                "source_type": "manual",
                "uri": title,
                "content_hash": digest,
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )
        self.repository.add_document(document)
        chunks = _chunker_from_configuration(configuration).chunk_stored_document(document)
        embeddings = embedder.embed([chunk.text for chunk in chunks]) if chunks else []
        self.repository.replace_document_chunks(knowledge_base_id, document.id, chunks, embeddings, embedder.model_name)
        self.repository.update_knowledge_base(
            knowledge_base_id,
            status="indexed" if chunks else "empty",
            embedding_model=embedder.model_name,
            error=None,
        )
        self.repository.record_ingestion_run(
            knowledge_base_id,
            "indexed" if chunks else "empty",
            {"documents_added": 1, "documents_skipped": 0, "chunks_added": len(chunks)},
            source_id=source.id,
        )
        return document

    def update_document(
        self,
        knowledge_base_id: str,
        document_id: str,
        title: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredKnowledgeDocument:
        self.repository.initialize()
        existing = self.repository.get_document(knowledge_base_id, document_id)
        configuration = self._knowledge_base_configuration(knowledge_base_id)
        embedder = self._embedder_for_configuration(configuration)
        clean = clean_text(text)
        if not clean:
            raise KnowledgeProcessingError("Document text cannot be empty.")
        digest = content_hash(clean)
        for document in self.repository.list_documents(knowledge_base_id):
            if document.id != document_id and document.content_hash == digest:
                raise KnowledgeProcessingError("A document with the same content already exists in this knowledge base.")
        updated = StoredKnowledgeDocument(
            id=existing.id,
            knowledge_base_id=existing.knowledge_base_id,
            source_id=existing.source_id,
            title=title.strip() or existing.title,
            content_hash=digest,
            text=clean,
            metadata={
                **existing.metadata,
                **(metadata or {}),
                **_processing_metadata(configuration),
                "content_hash": digest,
                "updated_at": utc_now(),
            },
        )
        self.repository.update_document(updated)
        chunks = _chunker_from_configuration(configuration).chunk_stored_document(updated)
        embeddings = embedder.embed([chunk.text for chunk in chunks]) if chunks else []
        self.repository.replace_document_chunks(knowledge_base_id, document_id, chunks, embeddings, embedder.model_name)
        self.repository.update_knowledge_base(
            knowledge_base_id,
            status="indexed" if chunks else "empty",
            embedding_model=embedder.model_name,
            error=None,
        )
        self.repository.record_ingestion_run(
            knowledge_base_id,
            "indexed" if chunks else "empty",
            {"documents_added": 0, "documents_skipped": 0, "chunks_added": len(chunks)},
            source_id=existing.source_id,
        )
        return updated

    def delete_document(self, knowledge_base_id: str, document_id: str) -> None:
        self.repository.initialize()
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        self.repository.get_document(knowledge_base_id, document_id)
        self.repository.delete_document(knowledge_base_id, document_id)
        remaining_chunks = self.repository.list_chunks(knowledge_base_id, limit=1)
        self.repository.update_knowledge_base(
            knowledge_base_id,
            status="indexed" if remaining_chunks else "empty",
            embedding_model=knowledge_base.embedding_model,
            error=None,
        )
        self.repository.record_ingestion_run(
            knowledge_base_id,
            "indexed" if remaining_chunks else "empty",
            {"documents_added": 0, "documents_skipped": 0, "chunks_added": 0},
        )

    def ingest_uploaded_file(self, knowledge_base_id: str, filename: str, content: bytes) -> IngestionSummary:
        self.repository.initialize()
        documents = load_file_documents(filename, content)
        return self._ingest_documents(
            knowledge_base_id,
            source_type="upload",
            uri=filename,
            documents=documents,
            source_metadata={"filename": filename, "size_bytes": len(content)},
        )

    def ingest_website(self, knowledge_base_id: str, url: str) -> IngestionSummary:
        self.repository.initialize()
        documents = [load_public_website(url)]
        return self._ingest_documents(
            knowledge_base_id,
            source_type="website",
            uri=url,
            documents=documents,
            source_metadata={"url": url},
        )

    def reindex(self, knowledge_base_id: str) -> IngestionSummary:
        self.repository.initialize()
        configuration = self._knowledge_base_configuration(knowledge_base_id)
        embedder = self._embedder_for_configuration(configuration)
        documents = self.repository.list_documents(knowledge_base_id)
        chunks: List[StoredKnowledgeChunk] = []
        chunker = _chunker_from_configuration(configuration)
        for document in documents:
            chunks.extend(chunker.chunk_stored_document(_document_with_processing_metadata(document, configuration)))
        embeddings = embedder.embed([chunk.text for chunk in chunks]) if chunks else []
        self.repository.replace_chunks(knowledge_base_id, chunks, embeddings, embedder.model_name)
        self.repository.update_knowledge_base(
            knowledge_base_id,
            status="indexed" if chunks else "empty",
            embedding_model=embedder.model_name,
            error=None,
        )
        summary = IngestionSummary(
            knowledge_base_id=knowledge_base_id,
            source_id=None,
            status="indexed" if chunks else "empty",
            documents_added=0,
            documents_skipped=0,
            chunks_added=len(chunks),
        )
        self.repository.record_ingestion_run(
            knowledge_base_id,
            summary.status,
            {"documents_added": 0, "documents_skipped": 0, "chunks_added": len(chunks)},
        )
        return summary

    def _ingest_documents(
        self,
        knowledge_base_id: str,
        source_type: str,
        uri: str,
        documents: Iterable[KnowledgeDocumentInput],
        source_metadata: Dict[str, Any],
    ) -> IngestionSummary:
        source = self.repository.create_data_source(
            knowledge_base_id=knowledge_base_id,
            source_type=source_type,
            uri=uri,
            status="processing",
            metadata={**source_metadata, "imported_at": utc_now()},
        )
        try:
            configuration = self._knowledge_base_configuration(knowledge_base_id)
            embedder = self._embedder_for_configuration(configuration)
            self.repository.update_knowledge_base(knowledge_base_id, status="processing", error=None)
            known_hashes = self.repository.existing_hashes(knowledge_base_id)
            stored_documents: List[StoredKnowledgeDocument] = []
            skipped = 0
            for document in documents:
                text = clean_text(document.text)
                if not text:
                    skipped += 1
                    continue
                digest = content_hash(text)
                if digest in known_hashes:
                    skipped += 1
                    continue
                known_hashes.add(digest)
                stored_document = StoredKnowledgeDocument(
                    id=f"doc-{uuid.uuid4().hex}",
                    knowledge_base_id=knowledge_base_id,
                    source_id=source.id,
                    title=document.title or uri,
                    content_hash=digest,
                    text=text,
                    metadata={
                        **document.metadata,
                        **_processing_metadata(configuration),
                        "source_type": document.source_type,
                        "uri": document.uri,
                        "content_hash": digest,
                        "imported_at": utc_now(),
                    },
                )
                self.repository.add_document(stored_document)
                stored_documents.append(stored_document)
            chunks: List[StoredKnowledgeChunk] = []
            chunker = _chunker_from_configuration(configuration)
            for document in stored_documents:
                chunks.extend(chunker.chunk_stored_document(document))
            embeddings = embedder.embed([chunk.text for chunk in chunks]) if chunks else []
            self.repository.append_chunks(chunks, embeddings, embedder.model_name)
            status = "indexed" if chunks or skipped else "empty"
            self.repository.update_knowledge_base(
                knowledge_base_id,
                status=status,
                embedding_model=embedder.model_name,
                error=None,
            )
            summary = IngestionSummary(
                knowledge_base_id=knowledge_base_id,
                source_id=source.id,
                status=status,
                documents_added=len(stored_documents),
                documents_skipped=skipped,
                chunks_added=len(chunks),
            )
            self.repository.record_ingestion_run(
                knowledge_base_id,
                status,
                {
                    "documents_added": len(stored_documents),
                    "documents_skipped": skipped,
                    "chunks_added": len(chunks),
                },
                source_id=source.id,
            )
            return summary
        except Exception as exc:
            message = str(exc)
            self.repository.update_knowledge_base(knowledge_base_id, status="failed", error=message)
            self.repository.record_ingestion_run(
                knowledge_base_id,
                "failed",
                {"documents_added": 0, "documents_skipped": 0, "chunks_added": 0},
                error=message,
                source_id=source.id,
            )
            if isinstance(exc, KnowledgeProcessingError):
                raise
            raise KnowledgeProcessingError(message) from exc

    def _knowledge_base_configuration(self, knowledge_base_id: str) -> Dict[str, Any]:
        knowledge_base = self.repository.get_knowledge_base(knowledge_base_id)
        return normalize_knowledge_base_configuration(knowledge_base.metadata.get("configuration"))

    def _embedder_for_configuration(self, configuration: Dict[str, Any]) -> "EmbeddingModel":
        normalized = validate_knowledge_base_configuration(configuration)
        provider = normalized["embedding_provider"]
        model_name = normalized["embedding_model"]
        cache_key = (provider, model_name, self._embedding_dimension)
        if cache_key not in self._embedder_cache:
            if model_name == HASH_EMBEDDING_MODEL:
                self._embedder_cache[cache_key] = HashEmbeddingModel(dimension=self._embedding_dimension)
            elif model_name == SENTENCE_TRANSFORMER_MINILM_MODEL:
                self._embedder_cache[cache_key] = SentenceTransformerEmbeddingModel(
                    model_name=model_name,
                    dimension=self._embedding_dimension,
                )
            else:
                supported = ", ".join(sorted(SUPPORTED_LOCAL_EMBEDDING_MODELS))
                raise KnowledgeProcessingError(
                    f"Embedding model {model_name!r} is not supported in v1. "
                    f"Modify the knowledge base to use Local with one of: {supported}."
                )
        return self._embedder_cache[cache_key]


class OverlapChunker:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_stored_document(self, document: StoredKnowledgeDocument) -> List[StoredKnowledgeChunk]:
        text = clean_text(document.text)
        if not text:
            return []
        chunks: List[StoredKnowledgeChunk] = []
        start = 0
        index = 0
        step = self.chunk_size - self.chunk_overlap
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    StoredKnowledgeChunk(
                        id=f"chunk-{uuid.uuid4().hex}",
                        knowledge_base_id=document.knowledge_base_id,
                        document_id=document.id,
                        chunk_index=index,
                        text=chunk_text,
                        token_count=len(simple_tokens(chunk_text)),
                        metadata={
                            "title": document.title,
                            "source_id": document.source_id,
                            "content_hash": document.content_hash,
                            "start_char": start,
                            "end_char": end,
                            "chunking_mode": document.metadata.get("chunking_mode", "overlap"),
                            "chunking_strategy": document.metadata.get("chunking_strategy", document.metadata.get("chunking_mode", "overlap")),
                            "chunk_size": self.chunk_size,
                            "chunk_overlap": self.chunk_overlap,
                            "embedding_provider": document.metadata.get("embedding_provider", ""),
                            "embedding_model_requested": document.metadata.get("embedding_model_requested", ""),
                        },
                    )
                )
                index += 1
            if end >= len(text):
                break
            start += step
        return chunks


class EmbeddingModel(Protocol):
    model_name: str
    dimension: int

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed text chunks."""


class HashEmbeddingModel:
    def __init__(self, dimension: int = 384, model_name: str = "hash-embedding-384"):
        self.dimension = dimension
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> List[float]:
        vector = [0.0] * self.dimension
        for token in simple_tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            vector[int(digest[:8], 16) % self.dimension] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class SentenceTransformerEmbeddingModel:
    def __init__(self, model_name: str, dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self._tokenizer = None
        self._model = None
        self._torch = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            try:
                import torch  # type: ignore
                from transformers import AutoModel, AutoTokenizer  # type: ignore
            except ImportError as exc:
                raise KnowledgeProcessingError("Install the ml extra to use transformer embeddings.") from exc
            except Exception as exc:
                raise KnowledgeProcessingError(_sentence_transformer_runtime_error(self.model_name, exc)) from exc
            try:
                self._torch = torch
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModel.from_pretrained(self.model_name)
                self._model.eval()
            except Exception as exc:
                raise KnowledgeProcessingError(_sentence_transformer_runtime_error(self.model_name, exc)) from exc
        try:
            encoded = self._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with self._torch.no_grad():
                output = self._model(**encoded)
            token_embeddings = output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
            normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings = normalized.cpu().tolist()
        except Exception as exc:
            raise KnowledgeProcessingError(_sentence_transformer_runtime_error(self.model_name, exc)) from exc
        return [self._fit_dimension(list(map(float, row))) for row in embeddings]

    def _fit_dimension(self, vector: List[float]) -> List[float]:
        if len(vector) == self.dimension:
            return vector
        if len(vector) > self.dimension:
            fitted = vector[: self.dimension]
        else:
            fitted = [*vector, *([0.0] * (self.dimension - len(vector)))]
        norm = sum(value * value for value in fitted) ** 0.5
        if norm == 0:
            return fitted
        return [value / norm for value in fitted]


def build_embedder(model_name: str, dimension: int, use_sentence_transformers: bool) -> EmbeddingModel:
    if use_sentence_transformers:
        return SentenceTransformerEmbeddingModel(model_name=model_name, dimension=dimension)
    return HashEmbeddingModel(dimension=dimension)


def _sentence_transformer_runtime_error(model_name: str, exc: Exception) -> str:
    return (
        f"Unable to initialize transformer embedding model {model_name!r}: {exc}. "
        "Use hash-embedding-384 or repair the ML dependencies with "
        "python -m pip install -e \".[ml]\"."
    )


def normalize_knowledge_base_configuration(configuration: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    raw = dict(configuration or {})
    strategy = str(raw.get("chunking_strategy") or DEFAULT_KNOWLEDGE_BASE_CONFIGURATION["chunking_strategy"])
    if strategy not in CHUNKING_STRATEGIES:
        strategy = DEFAULT_KNOWLEDGE_BASE_CONFIGURATION["chunking_strategy"]
    chunk_size = _bounded_int(raw.get("chunk_size"), DEFAULT_KNOWLEDGE_BASE_CONFIGURATION["chunk_size"], minimum=100, maximum=12000)
    chunk_overlap = _bounded_int(raw.get("chunk_overlap"), DEFAULT_KNOWLEDGE_BASE_CONFIGURATION["chunk_overlap"], minimum=0, maximum=chunk_size - 1)
    if strategy == "fixed_size":
        chunk_overlap = 0
    provider = str(raw.get("embedding_provider") or DEFAULT_KNOWLEDGE_BASE_CONFIGURATION["embedding_provider"]).strip() or LOCAL_EMBEDDING_PROVIDER
    if provider.lower() == LOCAL_EMBEDDING_PROVIDER.lower():
        provider = LOCAL_EMBEDDING_PROVIDER
    embedding_model = str(raw.get("embedding_model") or DEFAULT_KNOWLEDGE_BASE_CONFIGURATION["embedding_model"]).strip() or HASH_EMBEDDING_MODEL
    return {
        "chunking_strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": min(chunk_overlap, max(chunk_size - 1, 0)),
        "embedding_provider": provider,
        "embedding_model": embedding_model,
    }


def validate_knowledge_base_configuration(configuration: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    normalized = normalize_knowledge_base_configuration(configuration)
    provider = normalized["embedding_provider"]
    model_name = normalized["embedding_model"]
    supported = ", ".join(sorted(SUPPORTED_LOCAL_EMBEDDING_MODELS))
    if provider != LOCAL_EMBEDDING_PROVIDER:
        raise KnowledgeProcessingError(
            f"Embedding provider {provider!r} is visible for the roadmap but is not executable in v1. "
            f"Modify the knowledge base to use Local with one of: {supported}."
        )
    if model_name not in SUPPORTED_LOCAL_EMBEDDING_MODELS:
        raise KnowledgeProcessingError(
            f"Embedding model {model_name!r} is not supported for Local execution in v1. "
            f"Modify the knowledge base to use one of: {supported}."
        )
    return normalized


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _chunker_from_configuration(configuration: Dict[str, Any]) -> OverlapChunker:
    normalized = normalize_knowledge_base_configuration(configuration)
    return OverlapChunker(
        chunk_size=int(normalized["chunk_size"]),
        chunk_overlap=int(normalized["chunk_overlap"]),
    )


def _processing_metadata(configuration: Dict[str, Any]) -> Dict[str, Any]:
    normalized = normalize_knowledge_base_configuration(configuration)
    return {
        "chunking_strategy": normalized["chunking_strategy"],
        "chunking_mode": normalized["chunking_strategy"],
        "configured_chunk_size": normalized["chunk_size"],
        "configured_chunk_overlap": normalized["chunk_overlap"],
        "embedding_provider": normalized["embedding_provider"],
        "embedding_model_requested": normalized["embedding_model"],
    }


def _document_with_processing_metadata(
    document: StoredKnowledgeDocument,
    configuration: Dict[str, Any],
) -> StoredKnowledgeDocument:
    return StoredKnowledgeDocument(
        id=document.id,
        knowledge_base_id=document.knowledge_base_id,
        source_id=document.source_id,
        title=document.title,
        content_hash=document.content_hash,
        text=document.text,
        metadata={**document.metadata, **_processing_metadata(configuration)},
    )


def load_file_documents(filename: str, content: bytes) -> List[KnowledgeDocumentInput]:
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_FILE_EXTENSIONS:
        raise KnowledgeProcessingError(f"Unsupported file type {extension!r}.")
    if extension in {".txt", ".md"}:
        return [_text_document(filename, content, "upload", extension)]
    if extension == ".json":
        return _json_documents(filename, content)
    if extension == ".jsonl":
        return _jsonl_documents(filename, content)
    if extension == ".pdf":
        return [_pdf_document(filename, content)]
    if extension == ".docx":
        return [_docx_document(filename, content)]
    if extension in CUSTOM_CHUNKING_EXTENSIONS:
        document = _text_document(filename, content, "upload", extension)
        return [
            KnowledgeDocumentInput(
                title=document.title,
                text=document.text,
                source_type=document.source_type,
                uri=document.uri,
                metadata={
                    **document.metadata,
                    "chunking_mode": "custom_placeholder",
                    "custom_loader_status": "schema_pending",
                },
            )
        ]
    raise KnowledgeProcessingError(f"No loader registered for {extension!r}.")


def load_public_website(url: str) -> KnowledgeDocumentInput:
    if not url.startswith(("http://", "https://")):
        raise KnowledgeProcessingError("Website source must start with http:// or https://.")
    request = urllib.request.Request(url, headers={"User-Agent": "aragbiz-ingestion/0.1"})
    with urllib.request.urlopen(request, timeout=20) as response:
        content_type = response.headers.get("content-type", "text/html")
        raw = response.read()
    html = raw.decode("utf-8", errors="replace")
    title = _html_title(html) or url
    text = _html_text(html)
    return KnowledgeDocumentInput(
        title=title,
        text=text,
        source_type="website",
        uri=url,
        metadata={
            "url": url,
            "content_type": content_type,
            "size_bytes": len(raw),
            "loader": "website_html",
        },
    )


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(clean_text(text).encode("utf-8")).hexdigest()


def simple_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text_document(filename: str, content: bytes, source_type: str, extension: str) -> KnowledgeDocumentInput:
    text = content.decode("utf-8", errors="replace")
    return KnowledgeDocumentInput(
        title=Path(filename).name,
        text=text,
        source_type=source_type,
        uri=filename,
        metadata=_file_metadata(filename, content, extension, loader="text"),
    )


def _json_documents(filename: str, content: bytes) -> List[KnowledgeDocumentInput]:
    payload = json.loads(content.decode("utf-8"))
    rows = payload.get("documents", payload) if isinstance(payload, dict) else payload
    if isinstance(rows, list):
        return [_document_from_json_row(filename, row, index) for index, row in enumerate(rows, start=1)]
    return [_document_from_json_row(filename, rows, 1)]


def _jsonl_documents(filename: str, content: bytes) -> List[KnowledgeDocumentInput]:
    documents: List[KnowledgeDocumentInput] = []
    for index, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        documents.append(_document_from_json_row(filename, json.loads(line), index))
    return documents


def _document_from_json_row(filename: str, row: Any, index: int) -> KnowledgeDocumentInput:
    if isinstance(row, dict):
        text = str(row.get("text") or row.get("content") or row.get("contents") or row.get("context") or json.dumps(row, ensure_ascii=True))
        title = str(row.get("title") or row.get("id") or f"{Path(filename).stem}-{index}")
        metadata = {key: value for key, value in row.items() if key not in {"text", "content", "contents", "context"}}
    else:
        text = str(row)
        title = f"{Path(filename).stem}-{index}"
        metadata = {}
    return KnowledgeDocumentInput(
        title=title,
        text=text,
        source_type="upload",
        uri=f"{filename}#{index}",
        metadata={**_file_metadata(filename, text.encode("utf-8"), Path(filename).suffix.lower(), loader="json"), **metadata},
    )


def _pdf_document(filename: str, content: bytes) -> KnowledgeDocumentInput:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise KnowledgeProcessingError("Install the api extra to load PDF files.") from exc
    reader = PdfReader(BytesIO(content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return KnowledgeDocumentInput(
        title=Path(filename).name,
        text=text,
        source_type="upload",
        uri=filename,
        metadata={**_file_metadata(filename, content, ".pdf", loader="pdf"), "page_count": len(reader.pages)},
    )


def _docx_document(filename: str, content: bytes) -> KnowledgeDocumentInput:
    try:
        from docx import Document as DocxDocument  # type: ignore
    except ImportError as exc:
        raise KnowledgeProcessingError("Install the api extra to load DOCX files.") from exc
    document = DocxDocument(BytesIO(content))
    text = "\n".join(paragraph.text for paragraph in document.paragraphs)
    return KnowledgeDocumentInput(
        title=Path(filename).name,
        text=text,
        source_type="upload",
        uri=filename,
        metadata={**_file_metadata(filename, content, ".docx", loader="docx"), "paragraph_count": len(document.paragraphs)},
    )


def _file_metadata(filename: str, content: bytes, extension: str, loader: str) -> Dict[str, Any]:
    return {
        "filename": Path(filename).name,
        "extension": extension,
        "mime_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
        "size_bytes": len(content),
        "loader": loader,
    }


def _html_text(html: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html, "html.parser")
        for node in soup(["script", "style", "noscript"]):
            node.decompose()
        return clean_text(soup.get_text(" "))
    except ImportError:
        parser = _TextExtractor()
        parser.feed(html)
        return clean_text(" ".join(parser.parts))


def _html_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    return clean_text(match.group(1)) if match else ""


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, Optional[str]]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip())
