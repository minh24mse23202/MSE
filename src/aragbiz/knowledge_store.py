from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from aragbiz.knowledge import (
    DataSourceRecord,
    KnowledgeBaseRecord,
    KnowledgeProcessingError,
    StoredKnowledgeChunk,
    StoredKnowledgeDocument,
    utc_now,
)


class JsonKnowledgeRepository:
    def __init__(self, path: str):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(_empty_state())

    def create_knowledge_base(self, name: str, description: str = "") -> KnowledgeBaseRecord:
        state = self._read()
        now = utc_now()
        record = KnowledgeBaseRecord(
            id=f"kb-{uuid.uuid4().hex}",
            name=name,
            description=description,
            status="empty",
            created_at=now,
            updated_at=now,
        )
        state["knowledge_bases"][record.id] = _kb_to_dict(record)
        self._write(state)
        return record

    def list_knowledge_bases(self) -> List[KnowledgeBaseRecord]:
        state = self._read()
        return [self._hydrate_kb(payload, state) for payload in state["knowledge_bases"].values()]

    def get_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        state = self._read()
        payload = state["knowledge_bases"].get(knowledge_base_id)
        if not payload:
            raise KeyError(f"Knowledge base not found: {knowledge_base_id}")
        return self._hydrate_kb(payload, state)

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
        state = self._read()
        payload = state["knowledge_bases"].get(knowledge_base_id)
        if not payload:
            raise KeyError(f"Knowledge base not found: {knowledge_base_id}")
        if name is not None:
            payload["name"] = name
        if description is not None:
            payload["description"] = description
        if status is not None:
            payload["status"] = status
        if embedding_model is not None:
            payload["embedding_model"] = embedding_model
        payload["error"] = error
        payload["updated_at"] = utc_now()
        self._write(state)

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        state = self._read()
        if knowledge_base_id not in state["knowledge_bases"]:
            raise KeyError(f"Knowledge base not found: {knowledge_base_id}")
        state["knowledge_bases"].pop(knowledge_base_id, None)
        source_ids = [
            source_id
            for source_id, source in state["data_sources"].items()
            if source["knowledge_base_id"] == knowledge_base_id
        ]
        document_ids = [
            document_id
            for document_id, document in state["documents"].items()
            if document["knowledge_base_id"] == knowledge_base_id
        ]
        chunk_ids = [
            chunk_id
            for chunk_id, chunk in state["chunks"].items()
            if chunk["knowledge_base_id"] == knowledge_base_id
        ]
        run_ids = [
            run_id
            for run_id, run in state["ingestion_runs"].items()
            if run["knowledge_base_id"] == knowledge_base_id
        ]
        for source_id in source_ids:
            state["data_sources"].pop(source_id, None)
        for document_id in document_ids:
            state["documents"].pop(document_id, None)
        for chunk_id in chunk_ids:
            state["chunks"].pop(chunk_id, None)
            state["chunk_embeddings"].pop(chunk_id, None)
        for run_id in run_ids:
            state["ingestion_runs"].pop(run_id, None)
        self._write(state)

    def create_data_source(
        self,
        knowledge_base_id: str,
        source_type: str,
        uri: str,
        status: str,
        metadata: Dict[str, Any],
    ) -> DataSourceRecord:
        self.get_knowledge_base(knowledge_base_id)
        state = self._read()
        record = DataSourceRecord(
            id=f"src-{uuid.uuid4().hex}",
            knowledge_base_id=knowledge_base_id,
            source_type=source_type,
            uri=uri,
            status=status,
            metadata=metadata,
        )
        state["data_sources"][record.id] = _source_to_dict(record)
        self._write(state)
        return record

    def existing_hashes(self, knowledge_base_id: str) -> set[str]:
        state = self._read()
        return {
            payload["content_hash"]
            for payload in state["documents"].values()
            if payload["knowledge_base_id"] == knowledge_base_id
        }

    def add_document(self, document: StoredKnowledgeDocument) -> None:
        state = self._read()
        state["documents"][document.id] = _document_to_dict(document)
        self._write(state)

    def list_documents(self, knowledge_base_id: str) -> List[StoredKnowledgeDocument]:
        state = self._read()
        return [
            _document_from_dict(payload)
            for payload in state["documents"].values()
            if payload["knowledge_base_id"] == knowledge_base_id
        ]

    def get_document(self, knowledge_base_id: str, document_id: str) -> StoredKnowledgeDocument:
        state = self._read()
        payload = state["documents"].get(document_id)
        if not payload or payload["knowledge_base_id"] != knowledge_base_id:
            raise KeyError(f"Document not found: {document_id}")
        return _document_from_dict(payload)

    def update_document(self, document: StoredKnowledgeDocument) -> None:
        state = self._read()
        if document.id not in state["documents"]:
            raise KeyError(f"Document not found: {document.id}")
        state["documents"][document.id] = _document_to_dict(document)
        if document.knowledge_base_id in state["knowledge_bases"]:
            state["knowledge_bases"][document.knowledge_base_id]["updated_at"] = utc_now()
        self._write(state)

    def delete_document(self, knowledge_base_id: str, document_id: str) -> None:
        state = self._read()
        payload = state["documents"].get(document_id)
        if not payload or payload["knowledge_base_id"] != knowledge_base_id:
            raise KeyError(f"Document not found: {document_id}")
        state["documents"].pop(document_id, None)
        chunk_ids = [
            chunk_id
            for chunk_id, chunk in state["chunks"].items()
            if chunk["knowledge_base_id"] == knowledge_base_id and chunk["document_id"] == document_id
        ]
        for chunk_id in chunk_ids:
            state["chunks"].pop(chunk_id, None)
            state["chunk_embeddings"].pop(chunk_id, None)
        if knowledge_base_id in state["knowledge_bases"]:
            state["knowledge_bases"][knowledge_base_id]["updated_at"] = utc_now()
        self._write(state)

    def replace_chunks(self, knowledge_base_id: str, chunks: List[StoredKnowledgeChunk], embeddings: List[List[float]], model: str) -> None:
        state = self._read()
        existing_chunk_ids = {
            chunk_id
            for chunk_id, payload in state["chunks"].items()
            if payload["knowledge_base_id"] == knowledge_base_id
        }
        for chunk_id in existing_chunk_ids:
            state["chunks"].pop(chunk_id, None)
            state["chunk_embeddings"].pop(chunk_id, None)
        self._append_chunks_to_state(state, chunks, embeddings, model)
        self._write(state)

    def replace_document_chunks(
        self,
        knowledge_base_id: str,
        document_id: str,
        chunks: List[StoredKnowledgeChunk],
        embeddings: List[List[float]],
        model: str,
    ) -> None:
        state = self._read()
        existing_chunk_ids = {
            chunk_id
            for chunk_id, payload in state["chunks"].items()
            if payload["knowledge_base_id"] == knowledge_base_id and payload["document_id"] == document_id
        }
        for chunk_id in existing_chunk_ids:
            state["chunks"].pop(chunk_id, None)
            state["chunk_embeddings"].pop(chunk_id, None)
        self._append_chunks_to_state(state, chunks, embeddings, model)
        self._write(state)

    def append_chunks(self, chunks: List[StoredKnowledgeChunk], embeddings: List[List[float]], model: str) -> None:
        state = self._read()
        self._append_chunks_to_state(state, chunks, embeddings, model)
        self._write(state)

    def list_chunks(self, knowledge_base_id: str, limit: int = 100) -> List[StoredKnowledgeChunk]:
        state = self._read()
        chunks = [
            _chunk_from_dict(payload)
            for payload in state["chunks"].values()
            if payload["knowledge_base_id"] == knowledge_base_id
        ]
        chunks.sort(key=lambda chunk: (chunk.document_id, chunk.chunk_index))
        return chunks[:limit]

    def record_ingestion_run(
        self,
        knowledge_base_id: str,
        status: str,
        counts: Dict[str, int],
        error: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> None:
        state = self._read()
        run_id = f"run-{uuid.uuid4().hex}"
        state["ingestion_runs"][run_id] = {
            "id": run_id,
            "knowledge_base_id": knowledge_base_id,
            "source_id": source_id,
            "status": status,
            "counts": counts,
            "error": error,
            "started_at": utc_now(),
            "finished_at": utc_now(),
        }
        self._write(state)

    def _append_chunks_to_state(
        self,
        state: Dict[str, Dict[str, Any]],
        chunks: List[StoredKnowledgeChunk],
        embeddings: List[List[float]],
        model: str,
    ) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            state["chunks"][chunk.id] = _chunk_to_dict(chunk)
            state["chunk_embeddings"][chunk.id] = {
                "chunk_id": chunk.id,
                "embedding": embedding,
                "embedding_model": model,
            }

    def _hydrate_kb(self, payload: Dict[str, Any], state: Dict[str, Dict[str, Any]]) -> KnowledgeBaseRecord:
        knowledge_base_id = payload["id"]
        document_count = sum(1 for document in state["documents"].values() if document["knowledge_base_id"] == knowledge_base_id)
        chunk_count = sum(1 for chunk in state["chunks"].values() if chunk["knowledge_base_id"] == knowledge_base_id)
        return KnowledgeBaseRecord(
            id=payload["id"],
            name=payload["name"],
            description=payload.get("description", ""),
            status=payload.get("status", "empty"),
            document_count=document_count,
            chunk_count=chunk_count,
            embedding_model=payload.get("embedding_model", ""),
            created_at=payload.get("created_at", ""),
            updated_at=payload.get("updated_at", ""),
            metadata=dict(payload.get("metadata", {})),
            error=payload.get("error"),
        )

    def _read(self) -> Dict[str, Dict[str, Any]]:
        self.initialize()
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, state: Dict[str, Dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


class PostgresKnowledgeRepository:
    def __init__(self, database_url: str, embedding_dimension: int = 384):
        try:
            from sqlalchemy import create_engine  # type: ignore
        except ImportError as exc:
            raise KnowledgeProcessingError("Install the api extra to use PostgreSQL knowledge storage.") from exc
        self.database_url = database_url
        self.embedding_dimension = embedding_dimension
        self.engine = create_engine(database_url, future=True)

    def initialize(self) -> None:
        ddl = f"""
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL,
            embedding_model TEXT NOT NULL DEFAULT '',
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            error TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS data_sources (
            id TEXT PRIMARY KEY,
            knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            source_type TEXT NOT NULL,
            uri TEXT NOT NULL,
            status TEXT NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            source_id TEXT NOT NULL REFERENCES data_sources(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            UNIQUE (knowledge_base_id, content_hash)
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            embedding vector({int(self.embedding_dimension)}) NOT NULL,
            embedding_model TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ingestion_runs (
            id TEXT PRIMARY KEY,
            knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            source_id TEXT,
            status TEXT NOT NULL,
            counts_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            error TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL
        );
        """
        with self.engine.begin() as connection:
            for statement in [part.strip() for part in ddl.split(";") if part.strip()]:
                connection.exec_driver_sql(statement)

    def create_knowledge_base(self, name: str, description: str = "") -> KnowledgeBaseRecord:
        from sqlalchemy import text

        record = KnowledgeBaseRecord(
            id=f"kb-{uuid.uuid4().hex}",
            name=name,
            description=description,
            status="empty",
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO knowledge_bases (id, name, description, status, embedding_model, metadata_json, created_at, updated_at)
                    VALUES (:id, :name, :description, :status, :embedding_model, CAST(:metadata AS JSONB), :created_at, :updated_at)
                    """
                ),
                {**_kb_to_dict(record), "metadata": json.dumps(record.metadata)},
            )
        return record

    def list_knowledge_bases(self) -> List[KnowledgeBaseRecord]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT kb.*,
                           COUNT(DISTINCT d.id) AS document_count,
                           COUNT(DISTINCT c.id) AS chunk_count
                    FROM knowledge_bases kb
                    LEFT JOIN documents d ON d.knowledge_base_id = kb.id
                    LEFT JOIN chunks c ON c.knowledge_base_id = kb.id
                    GROUP BY kb.id
                    ORDER BY kb.updated_at DESC
                    """
                )
            ).mappings()
            return [_kb_from_row(row) for row in rows]

    def get_knowledge_base(self, knowledge_base_id: str) -> KnowledgeBaseRecord:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT kb.*,
                           COUNT(DISTINCT d.id) AS document_count,
                           COUNT(DISTINCT c.id) AS chunk_count
                    FROM knowledge_bases kb
                    LEFT JOIN documents d ON d.knowledge_base_id = kb.id
                    LEFT JOIN chunks c ON c.knowledge_base_id = kb.id
                    WHERE kb.id = :id
                    GROUP BY kb.id
                    """
                ),
                {"id": knowledge_base_id},
            ).mappings().first()
        if not row:
            raise KeyError(f"Knowledge base not found: {knowledge_base_id}")
        return _kb_from_row(row)

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
        from sqlalchemy import text

        current = self.get_knowledge_base(knowledge_base_id)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE knowledge_bases
                    SET name = :name,
                        description = :description,
                        status = :status,
                        embedding_model = :embedding_model,
                        error = :error,
                        updated_at = :updated_at
                    WHERE id = :id
                    """
                ),
                {
                    "id": knowledge_base_id,
                    "name": name if name is not None else current.name,
                    "description": description if description is not None else current.description,
                    "status": status if status is not None else current.status,
                    "embedding_model": embedding_model if embedding_model is not None else current.embedding_model,
                    "error": error,
                    "updated_at": utc_now(),
                },
            )

    def delete_knowledge_base(self, knowledge_base_id: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            result = connection.execute(
                text("DELETE FROM knowledge_bases WHERE id = :id"),
                {"id": knowledge_base_id},
            )
            if result.rowcount == 0:
                raise KeyError(f"Knowledge base not found: {knowledge_base_id}")

    def create_data_source(
        self,
        knowledge_base_id: str,
        source_type: str,
        uri: str,
        status: str,
        metadata: Dict[str, Any],
    ) -> DataSourceRecord:
        from sqlalchemy import text

        record = DataSourceRecord(
            id=f"src-{uuid.uuid4().hex}",
            knowledge_base_id=knowledge_base_id,
            source_type=source_type,
            uri=uri,
            status=status,
            metadata=metadata,
        )
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO data_sources (id, knowledge_base_id, source_type, uri, status, metadata_json)
                    VALUES (:id, :knowledge_base_id, :source_type, :uri, :status, CAST(:metadata AS JSONB))
                    """
                ),
                {**_source_to_dict(record), "metadata": json.dumps(record.metadata)},
            )
        return record

    def existing_hashes(self, knowledge_base_id: str) -> set[str]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text("SELECT content_hash FROM documents WHERE knowledge_base_id = :id"),
                {"id": knowledge_base_id},
            )
        return {str(row[0]) for row in rows}

    def add_document(self, document: StoredKnowledgeDocument) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO documents (id, knowledge_base_id, source_id, title, content_hash, text, metadata_json)
                    VALUES (:id, :knowledge_base_id, :source_id, :title, :content_hash, :text, CAST(:metadata AS JSONB))
                    ON CONFLICT (knowledge_base_id, content_hash) DO NOTHING
                    """
                ),
                {**_document_to_dict(document), "metadata": json.dumps(document.metadata)},
            )

    def list_documents(self, knowledge_base_id: str) -> List[StoredKnowledgeDocument]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text("SELECT * FROM documents WHERE knowledge_base_id = :id ORDER BY title, id"),
                {"id": knowledge_base_id},
            ).mappings()
            return [_document_from_row(row) for row in rows]

    def get_document(self, knowledge_base_id: str, document_id: str) -> StoredKnowledgeDocument:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            row = connection.execute(
                text("SELECT * FROM documents WHERE knowledge_base_id = :knowledge_base_id AND id = :document_id"),
                {"knowledge_base_id": knowledge_base_id, "document_id": document_id},
            ).mappings().first()
        if not row:
            raise KeyError(f"Document not found: {document_id}")
        return _document_from_row(row)

    def update_document(self, document: StoredKnowledgeDocument) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    UPDATE documents
                    SET title = :title,
                        content_hash = :content_hash,
                        text = :text,
                        metadata_json = CAST(:metadata AS JSONB)
                    WHERE id = :id AND knowledge_base_id = :knowledge_base_id
                    """
                ),
                {**_document_to_dict(document), "metadata": json.dumps(document.metadata)},
            )
            if result.rowcount == 0:
                raise KeyError(f"Document not found: {document.id}")
            connection.execute(
                text("UPDATE knowledge_bases SET updated_at = :updated_at WHERE id = :id"),
                {"id": document.knowledge_base_id, "updated_at": utc_now()},
            )

    def delete_document(self, knowledge_base_id: str, document_id: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            result = connection.execute(
                text("DELETE FROM documents WHERE knowledge_base_id = :knowledge_base_id AND id = :document_id"),
                {"knowledge_base_id": knowledge_base_id, "document_id": document_id},
            )
            if result.rowcount == 0:
                raise KeyError(f"Document not found: {document_id}")
            connection.execute(
                text("UPDATE knowledge_bases SET updated_at = :updated_at WHERE id = :id"),
                {"id": knowledge_base_id, "updated_at": utc_now()},
            )

    def replace_chunks(self, knowledge_base_id: str, chunks: List[StoredKnowledgeChunk], embeddings: List[List[float]], model: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            connection.execute(text("DELETE FROM chunks WHERE knowledge_base_id = :id"), {"id": knowledge_base_id})
        self.append_chunks(chunks, embeddings, model)

    def replace_document_chunks(
        self,
        knowledge_base_id: str,
        document_id: str,
        chunks: List[StoredKnowledgeChunk],
        embeddings: List[List[float]],
        model: str,
    ) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            connection.execute(
                text("DELETE FROM chunks WHERE knowledge_base_id = :knowledge_base_id AND document_id = :document_id"),
                {"knowledge_base_id": knowledge_base_id, "document_id": document_id},
            )
        self.append_chunks(chunks, embeddings, model)

    def append_chunks(self, chunks: List[StoredKnowledgeChunk], embeddings: List[List[float]], model: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            for chunk, embedding in zip(chunks, embeddings):
                connection.execute(
                    text(
                        """
                        INSERT INTO chunks (id, knowledge_base_id, document_id, chunk_index, text, token_count, metadata_json)
                        VALUES (:id, :knowledge_base_id, :document_id, :chunk_index, :text, :token_count, CAST(:metadata AS JSONB))
                        """
                    ),
                    {**_chunk_to_dict(chunk), "metadata": json.dumps(chunk.metadata)},
                )
                connection.execute(
                    text(
                        """
                        INSERT INTO chunk_embeddings (chunk_id, embedding, embedding_model)
                        VALUES (:chunk_id, :embedding, :embedding_model)
                        """
                    ),
                    {
                        "chunk_id": chunk.id,
                        "embedding": _vector_literal(embedding),
                        "embedding_model": model,
                    },
                )

    def list_chunks(self, knowledge_base_id: str, limit: int = 100) -> List[StoredKnowledgeChunk]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT * FROM chunks
                    WHERE knowledge_base_id = :id
                    ORDER BY document_id, chunk_index
                    LIMIT :limit
                    """
                ),
                {"id": knowledge_base_id, "limit": limit},
            ).mappings()
            return [_chunk_from_row(row) for row in rows]

    def record_ingestion_run(
        self,
        knowledge_base_id: str,
        status: str,
        counts: Dict[str, int],
        error: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> None:
        from sqlalchemy import text

        now = utc_now()
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO ingestion_runs (id, knowledge_base_id, source_id, status, counts_json, error, started_at, finished_at)
                    VALUES (:id, :knowledge_base_id, :source_id, :status, CAST(:counts AS JSONB), :error, :started_at, :finished_at)
                    """
                ),
                {
                    "id": f"run-{uuid.uuid4().hex}",
                    "knowledge_base_id": knowledge_base_id,
                    "source_id": source_id,
                    "status": status,
                    "counts": json.dumps(counts),
                    "error": error,
                    "started_at": now,
                    "finished_at": now,
                },
            )


def _empty_state() -> Dict[str, Dict[str, Any]]:
    return {
        "knowledge_bases": {},
        "data_sources": {},
        "documents": {},
        "chunks": {},
        "chunk_embeddings": {},
        "ingestion_runs": {},
    }


def _kb_to_dict(record: KnowledgeBaseRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "name": record.name,
        "description": record.description,
        "status": record.status,
        "embedding_model": record.embedding_model,
        "metadata": record.metadata,
        "error": record.error,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _source_to_dict(record: DataSourceRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "knowledge_base_id": record.knowledge_base_id,
        "source_type": record.source_type,
        "uri": record.uri,
        "status": record.status,
        "metadata": record.metadata,
    }


def _document_to_dict(record: StoredKnowledgeDocument) -> Dict[str, Any]:
    return {
        "id": record.id,
        "knowledge_base_id": record.knowledge_base_id,
        "source_id": record.source_id,
        "title": record.title,
        "content_hash": record.content_hash,
        "text": record.text,
        "metadata": record.metadata,
    }


def _chunk_to_dict(record: StoredKnowledgeChunk) -> Dict[str, Any]:
    return {
        "id": record.id,
        "knowledge_base_id": record.knowledge_base_id,
        "document_id": record.document_id,
        "chunk_index": record.chunk_index,
        "text": record.text,
        "token_count": record.token_count,
        "metadata": record.metadata,
    }


def _document_from_dict(payload: Dict[str, Any]) -> StoredKnowledgeDocument:
    return StoredKnowledgeDocument(
        id=payload["id"],
        knowledge_base_id=payload["knowledge_base_id"],
        source_id=payload["source_id"],
        title=payload["title"],
        content_hash=payload["content_hash"],
        text=payload["text"],
        metadata=dict(payload.get("metadata", {})),
    )


def _chunk_from_dict(payload: Dict[str, Any]) -> StoredKnowledgeChunk:
    return StoredKnowledgeChunk(
        id=payload["id"],
        knowledge_base_id=payload["knowledge_base_id"],
        document_id=payload["document_id"],
        chunk_index=int(payload["chunk_index"]),
        text=payload["text"],
        token_count=int(payload["token_count"]),
        metadata=dict(payload.get("metadata", {})),
    )


def _kb_from_row(row: Any) -> KnowledgeBaseRecord:
    metadata = row.get("metadata_json") or {}
    return KnowledgeBaseRecord(
        id=row["id"],
        name=row["name"],
        description=row.get("description", ""),
        status=row.get("status", "empty"),
        document_count=int(row.get("document_count") or 0),
        chunk_count=int(row.get("chunk_count") or 0),
        embedding_model=row.get("embedding_model") or "",
        created_at=row.get("created_at") or "",
        updated_at=row.get("updated_at") or "",
        metadata=dict(metadata),
        error=row.get("error"),
    )


def _document_from_row(row: Any) -> StoredKnowledgeDocument:
    return StoredKnowledgeDocument(
        id=row["id"],
        knowledge_base_id=row["knowledge_base_id"],
        source_id=row["source_id"],
        title=row["title"],
        content_hash=row["content_hash"],
        text=row["text"],
        metadata=dict(row.get("metadata_json") or {}),
    )


def _chunk_from_row(row: Any) -> StoredKnowledgeChunk:
    return StoredKnowledgeChunk(
        id=row["id"],
        knowledge_base_id=row["knowledge_base_id"],
        document_id=row["document_id"],
        chunk_index=int(row["chunk_index"]),
        text=row["text"],
        token_count=int(row["token_count"]),
        metadata=dict(row.get("metadata_json") or {}),
    )


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"
