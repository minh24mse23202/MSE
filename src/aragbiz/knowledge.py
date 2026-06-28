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

    def create_knowledge_base(self, name: str, description: str = "") -> KnowledgeBaseRecord:
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
        self.embedder = embedder

    def create_knowledge_base(self, name: str, description: str = "") -> KnowledgeBaseRecord:
        self.repository.initialize()
        return self.repository.create_knowledge_base(name=name, description=description)

    def list_knowledge_bases(self) -> List[KnowledgeBaseRecord]:
        self.repository.initialize()
        return self.repository.list_knowledge_bases()

    def get_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        self.repository.initialize()
        return self.repository.get_knowledge_base(knowledge_base_id)

    def update_knowledge_base_details(self, knowledge_base_id: str, name: str, description: str = "") -> KnowledgeBaseRecord:
        self.repository.initialize()
        self.repository.get_knowledge_base(knowledge_base_id)
        self.repository.update_knowledge_base(
            knowledge_base_id,
            name=name.strip(),
            description=description,
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

    def create_document(
        self,
        knowledge_base_id: str,
        title: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StoredKnowledgeDocument:
        self.repository.initialize()
        self.repository.get_knowledge_base(knowledge_base_id)
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
                "source_type": "manual",
                "uri": title,
                "content_hash": digest,
                "created_at": utc_now(),
                "updated_at": utc_now(),
            },
        )
        self.repository.add_document(document)
        chunks = self.chunker.chunk_stored_document(document)
        embeddings = self.embedder.embed([chunk.text for chunk in chunks]) if chunks else []
        self.repository.replace_document_chunks(knowledge_base_id, document.id, chunks, embeddings, self.embedder.model_name)
        self.repository.update_knowledge_base(
            knowledge_base_id,
            status="indexed" if chunks else "empty",
            embedding_model=self.embedder.model_name,
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
                "content_hash": digest,
                "updated_at": utc_now(),
            },
        )
        self.repository.update_document(updated)
        chunks = self.chunker.chunk_stored_document(updated)
        embeddings = self.embedder.embed([chunk.text for chunk in chunks]) if chunks else []
        self.repository.replace_document_chunks(knowledge_base_id, document_id, chunks, embeddings, self.embedder.model_name)
        self.repository.update_knowledge_base(
            knowledge_base_id,
            status="indexed" if chunks else "empty",
            embedding_model=self.embedder.model_name,
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
        self.repository.get_document(knowledge_base_id, document_id)
        self.repository.delete_document(knowledge_base_id, document_id)
        remaining_chunks = self.repository.list_chunks(knowledge_base_id, limit=1)
        self.repository.update_knowledge_base(
            knowledge_base_id,
            status="indexed" if remaining_chunks else "empty",
            embedding_model=self.embedder.model_name,
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
        documents = self.repository.list_documents(knowledge_base_id)
        chunks: List[StoredKnowledgeChunk] = []
        for document in documents:
            chunks.extend(self.chunker.chunk_stored_document(document))
        embeddings = self.embedder.embed([chunk.text for chunk in chunks]) if chunks else []
        self.repository.replace_chunks(knowledge_base_id, chunks, embeddings, self.embedder.model_name)
        self.repository.update_knowledge_base(
            knowledge_base_id,
            status="indexed" if chunks else "empty",
            embedding_model=self.embedder.model_name,
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
                        "source_type": document.source_type,
                        "uri": document.uri,
                        "content_hash": digest,
                        "imported_at": utc_now(),
                    },
                )
                self.repository.add_document(stored_document)
                stored_documents.append(stored_document)
            chunks: List[StoredKnowledgeChunk] = []
            for document in stored_documents:
                chunks.extend(self.chunker.chunk_stored_document(document))
            embeddings = self.embedder.embed([chunk.text for chunk in chunks]) if chunks else []
            self.repository.append_chunks(chunks, embeddings, self.embedder.model_name)
            status = "indexed" if chunks or skipped else "empty"
            self.repository.update_knowledge_base(
                knowledge_base_id,
                status=status,
                embedding_model=self.embedder.model_name,
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
            raise


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
        self._model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise KnowledgeProcessingError("Install the ml extra to use sentence-transformers embeddings.") from exc
            self._model = SentenceTransformer(self.model_name)
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [list(map(float, row)) for row in embeddings]


def build_embedder(model_name: str, dimension: int, use_sentence_transformers: bool) -> EmbeddingModel:
    if use_sentence_transformers:
        return SentenceTransformerEmbeddingModel(model_name=model_name, dimension=dimension)
    return HashEmbeddingModel(dimension=dimension)


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
