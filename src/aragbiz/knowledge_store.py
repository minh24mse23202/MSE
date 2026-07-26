from __future__ import annotations

import json
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from aragbiz.knowledge import (
    DataSourceRecord,
    KnowledgeBaseRecord,
    KnowledgeIndexVersionRecord,
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

    def create_knowledge_base(
        self,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBaseRecord:
        state = self._read()
        now = utc_now()
        record = KnowledgeBaseRecord(
            id=f"kb-{uuid.uuid4().hex}",
            name=name,
            description=description,
            status="empty",
            metadata=dict(metadata or {}),
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
        metadata: Optional[Dict[str, Any]] = None,
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
        if metadata is not None:
            payload["metadata"] = metadata
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
        version_ids = [
            version_id
            for version_id, version in state["index_versions"].items()
            if version["knowledge_base_id"] == knowledge_base_id
        ]
        for version_id in version_ids:
            state["index_versions"].pop(version_id, None)
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

    def update_data_source(
        self,
        source_id: str,
        *,
        status: str,
        metadata: Dict[str, Any],
    ) -> None:
        state = self._read()
        source = state["data_sources"].get(source_id)
        if not source:
            raise KeyError(f"Data source not found: {source_id}")
        source["status"] = status
        source["metadata"] = dict(metadata)
        self._write(state)

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

    def add_documents(self, documents: List[StoredKnowledgeDocument]) -> None:
        if not documents:
            return
        state = self._read()
        existing = {
            payload["content_hash"]
            for payload in state["documents"].values()
            if payload["knowledge_base_id"] == documents[0].knowledge_base_id
        }
        for document in documents:
            if document.content_hash in existing:
                continue
            existing.add(document.content_hash)
            state["documents"][document.id] = _document_to_dict(document)
        self._write(state)

    def list_documents(self, knowledge_base_id: str) -> List[StoredKnowledgeDocument]:
        state = self._read()
        return [
            _document_from_dict(payload)
            for payload in state["documents"].values()
            if payload["knowledge_base_id"] == knowledge_base_id
        ]

    def list_documents_page(
        self,
        knowledge_base_id: str,
        *,
        query: str = "",
        offset: int = 0,
        limit: int = 25,
    ) -> tuple[List[StoredKnowledgeDocument], int]:
        normalized = query.strip().lower()
        documents = self.list_documents(knowledge_base_id)
        if normalized:
            documents = [
                document
                for document in documents
                if normalized in document.title.lower()
                or normalized in str(document.metadata.get("source_record_id") or "").lower()
                or normalized in str(document.metadata.get("url") or document.metadata.get("uri") or "").lower()
            ]
        documents.sort(key=lambda document: (document.title.lower(), document.id))
        return documents[offset : offset + limit], len(documents)

    def list_wixqa_source_record_ids(self, knowledge_base_id: str) -> List[str]:
        state = self._read()
        record_ids = {
            str(payload.get("metadata", {}).get("source_record_id") or "")
            for payload in state["documents"].values()
            if payload["knowledge_base_id"] == knowledge_base_id
            and payload.get("metadata", {}).get("source_type") == "wixqa_corpus"
        }
        record_ids.discard("")
        return sorted(record_ids)

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
        target_version = next((chunk.index_version_id for chunk in chunks if chunk.index_version_id), "")
        existing_chunk_ids = {
            chunk_id
            for chunk_id, payload in state["chunks"].items()
            if payload["knowledge_base_id"] == knowledge_base_id
            and (not target_version or payload.get("index_version_id", "") == target_version)
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
        active_version = state["knowledge_bases"].get(knowledge_base_id, {}).get("active_index_version_id", "")
        chunks = [
            _chunk_from_dict(payload, state["chunk_embeddings"].get(chunk_id))
            for chunk_id, payload in state["chunks"].items()
            if payload["knowledge_base_id"] == knowledge_base_id
            and (not active_version or payload.get("index_version_id", "") == active_version)
        ]
        chunks.sort(key=lambda chunk: (chunk.document_id, chunk.chunk_index))
        return chunks[:limit]

    def list_document_chunks(self, knowledge_base_id: str, document_id: str) -> List[StoredKnowledgeChunk]:
        return self.list_active_chunks(knowledge_base_id, document_ids=[document_id])

    def list_active_chunks(
        self,
        knowledge_base_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> List[StoredKnowledgeChunk]:
        state = self._read()
        active_version = state["knowledge_bases"].get(knowledge_base_id, {}).get("active_index_version_id", "")
        selected = set(document_ids or [])
        chunks = [
            _chunk_from_dict(payload, state["chunk_embeddings"].get(chunk_id))
            for chunk_id, payload in state["chunks"].items()
            if payload["knowledge_base_id"] == knowledge_base_id
            and (not active_version or payload.get("index_version_id", "") == active_version)
            and (not selected or payload["document_id"] in selected)
        ]
        chunks.sort(key=lambda chunk: (chunk.document_id, chunk.chunk_index))
        return chunks

    def search_chunks_by_embedding(
        self,
        knowledge_base_id: str,
        embedding: List[float],
        limit: int = 10,
    ) -> List[tuple[StoredKnowledgeChunk, float]]:
        state = self._read()
        active_version = state["knowledge_bases"].get(knowledge_base_id, {}).get("active_index_version_id", "")
        matches: List[tuple[StoredKnowledgeChunk, float]] = []
        for chunk_id, payload in state["chunks"].items():
            if payload["knowledge_base_id"] != knowledge_base_id:
                continue
            if active_version and payload.get("index_version_id", "") != active_version:
                continue
            embedding_payload = state["chunk_embeddings"].get(chunk_id)
            if not embedding_payload:
                continue
            stored_embedding = _coerce_vector(embedding_payload.get("embedding", []))
            if not stored_embedding:
                continue
            matches.append((_chunk_from_dict(payload, embedding_payload), _cosine_similarity(embedding, stored_embedding)))
        matches.sort(key=lambda item: item[1], reverse=True)
        return matches[:limit]

    def list_ingestion_runs(self, knowledge_base_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        state = self._read()
        runs = [
            dict(payload)
            for payload in state["ingestion_runs"].values()
            if payload["knowledge_base_id"] == knowledge_base_id
        ]
        runs.sort(key=lambda run: run.get("finished_at", ""), reverse=True)
        return runs[:limit]

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

    def create_index_version(
        self,
        knowledge_base_id: str,
        configuration: Dict[str, Any],
        *,
        embedding_provider: str,
        embedding_model: str,
        embedding_deployment_id: str,
        embedding_dimension: int,
    ) -> KnowledgeIndexVersionRecord:
        self.get_knowledge_base(knowledge_base_id)
        state = self._read()
        record = KnowledgeIndexVersionRecord(
            id=f"index-{uuid.uuid4().hex}", knowledge_base_id=knowledge_base_id, status="building",
            chunking_configuration=dict(configuration), embedding_deployment_id=embedding_deployment_id,
            embedding_provider=embedding_provider, embedding_model=embedding_model,
            embedding_dimension=embedding_dimension, created_at=utc_now(),
        )
        state["index_versions"][record.id] = _index_version_to_dict(record)
        state["knowledge_bases"][knowledge_base_id]["pending_index_version_id"] = record.id
        self._write(state)
        return record

    def activate_index_version(self, knowledge_base_id: str, version_id: str) -> KnowledgeIndexVersionRecord:
        state = self._read()
        payload = state["index_versions"].get(version_id)
        if not payload or payload["knowledge_base_id"] != knowledge_base_id:
            raise KeyError(f"Knowledge index version not found: {version_id}")
        previous = state["knowledge_bases"][knowledge_base_id].get("active_index_version_id", "")
        if previous and previous in state["index_versions"]:
            state["index_versions"][previous]["status"] = "superseded"
        chunk_count = sum(1 for chunk in state["chunks"].values() if chunk.get("index_version_id") == version_id)
        document_count = len({chunk["document_id"] for chunk in state["chunks"].values() if chunk.get("index_version_id") == version_id})
        payload.update({"status": "active", "activated_at": utc_now(), "chunk_count": chunk_count, "document_count": document_count})
        state["knowledge_bases"][knowledge_base_id]["active_index_version_id"] = version_id
        state["knowledge_bases"][knowledge_base_id]["pending_index_version_id"] = ""
        self._write(state)
        return _index_version_from_dict(payload)

    def fail_index_version(self, knowledge_base_id: str, version_id: str, error: str) -> None:
        state = self._read()
        payload = state["index_versions"].get(version_id)
        if payload and payload["knowledge_base_id"] == knowledge_base_id:
            payload.update({"status": "failed", "error": str(error)[:2000]})
            state["knowledge_bases"][knowledge_base_id]["pending_index_version_id"] = ""
            self._write(state)

    def list_index_versions(self, knowledge_base_id: str) -> List[KnowledgeIndexVersionRecord]:
        state = self._read()
        records = [
            _index_version_from_dict(payload)
            for payload in state["index_versions"].values()
            if payload["knowledge_base_id"] == knowledge_base_id
        ]
        records.sort(key=lambda item: item.created_at, reverse=True)
        return records

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
        active_version = payload.get("active_index_version_id", "")
        document_count = sum(1 for document in state["documents"].values() if document["knowledge_base_id"] == knowledge_base_id)
        chunk_count = sum(
            1 for chunk in state["chunks"].values()
            if chunk["knowledge_base_id"] == knowledge_base_id
            and (not active_version or chunk.get("index_version_id", "") == active_version)
        )
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
        state = json.loads(self.path.read_text(encoding="utf-8"))
        for key, value in _empty_state().items():
            state.setdefault(key, value)
        return state

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
        self._initialize_lock = threading.Lock()
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._initialize_lock:
            if self._initialized:
                return
            self._initialize_schema()
            self._initialized = True

    def _initialize_schema(self) -> None:
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
            updated_at TEXT NOT NULL,
            active_index_version_id TEXT NOT NULL DEFAULT '',
            pending_index_version_id TEXT NOT NULL DEFAULT ''
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
            metadata_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            index_version_id TEXT NOT NULL DEFAULT '',
            parent_chunk_id TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS chunk_embeddings (
            chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
            embedding vector NOT NULL,
            embedding_model TEXT NOT NULL,
            embedding_deployment_id TEXT NOT NULL DEFAULT '',
            embedding_dimension INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS knowledge_index_versions (
            id TEXT PRIMARY KEY,
            knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            status TEXT NOT NULL,
            chunking_configuration_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding_deployment_id TEXT NOT NULL DEFAULT '',
            embedding_provider TEXT NOT NULL DEFAULT '',
            embedding_model TEXT NOT NULL DEFAULT '',
            embedding_dimension INTEGER NOT NULL DEFAULT 0,
            document_count INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            activated_at TEXT NOT NULL DEFAULT ''
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
        ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS active_index_version_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE knowledge_bases ADD COLUMN IF NOT EXISTS pending_index_version_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS index_version_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE chunks ADD COLUMN IF NOT EXISTS parent_chunk_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS embedding_deployment_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE chunk_embeddings ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER NOT NULL DEFAULT 0;
        CREATE INDEX IF NOT EXISTS idx_chunks_active_version ON chunks(knowledge_base_id, index_version_id);
        CREATE INDEX IF NOT EXISTS idx_index_versions_kb ON knowledge_index_versions(knowledge_base_id, created_at DESC);
        """
        with self.engine.begin() as connection:
            for statement in [part.strip() for part in ddl.split(";") if part.strip()]:
                connection.exec_driver_sql(statement)

    def create_knowledge_base(
        self,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeBaseRecord:
        from sqlalchemy import text

        record = KnowledgeBaseRecord(
            id=f"kb-{uuid.uuid4().hex}",
            name=name,
            description=description,
            status="empty",
            metadata=dict(metadata or {}),
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
                        AND (kb.active_index_version_id = '' OR c.index_version_id = kb.active_index_version_id)
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
                        AND (kb.active_index_version_id = '' OR c.index_version_id = kb.active_index_version_id)
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
        metadata: Optional[Dict[str, Any]] = None,
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
                        metadata_json = CAST(:metadata AS JSONB),
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
                    "metadata": json.dumps(metadata if metadata is not None else current.metadata),
                    "error": error,
                    "updated_at": utc_now(),
                },
            )

    def create_index_version(
        self,
        knowledge_base_id: str,
        configuration: Dict[str, Any],
        *,
        embedding_provider: str,
        embedding_model: str,
        embedding_deployment_id: str,
        embedding_dimension: int,
    ) -> KnowledgeIndexVersionRecord:
        from sqlalchemy import text

        self.get_knowledge_base(knowledge_base_id)
        record = KnowledgeIndexVersionRecord(
            id=f"index-{uuid.uuid4().hex}", knowledge_base_id=knowledge_base_id, status="building",
            chunking_configuration=dict(configuration), embedding_deployment_id=embedding_deployment_id,
            embedding_provider=embedding_provider, embedding_model=embedding_model,
            embedding_dimension=embedding_dimension, created_at=utc_now(),
        )
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO knowledge_index_versions (
                        id, knowledge_base_id, status, chunking_configuration_json,
                        embedding_deployment_id, embedding_provider, embedding_model,
                        embedding_dimension, document_count, chunk_count, error, created_at, activated_at
                    ) VALUES (
                        :id, :knowledge_base_id, :status, CAST(:configuration AS JSONB),
                        :embedding_deployment_id, :embedding_provider, :embedding_model,
                        :embedding_dimension, 0, 0, '', :created_at, ''
                    )
                    """
                ),
                {
                    "id": record.id, "knowledge_base_id": knowledge_base_id, "status": record.status,
                    "configuration": json.dumps(configuration), "embedding_deployment_id": embedding_deployment_id,
                    "embedding_provider": embedding_provider, "embedding_model": embedding_model,
                    "embedding_dimension": embedding_dimension, "created_at": record.created_at,
                },
            )
            connection.execute(
                text("UPDATE knowledge_bases SET pending_index_version_id = :version_id WHERE id = :id"),
                {"id": knowledge_base_id, "version_id": record.id},
            )
        return record

    def activate_index_version(self, knowledge_base_id: str, version_id: str) -> KnowledgeIndexVersionRecord:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            exists = connection.execute(
                text("SELECT 1 FROM knowledge_index_versions WHERE id = :version_id AND knowledge_base_id = :id"),
                {"version_id": version_id, "id": knowledge_base_id},
            ).first()
            if not exists:
                raise KeyError(f"Knowledge index version not found: {version_id}")
            connection.execute(
                text(
                    """
                    UPDATE knowledge_index_versions
                    SET status = 'superseded'
                    WHERE knowledge_base_id = :id AND status = 'active' AND id <> :version_id
                    """
                ),
                {"id": knowledge_base_id, "version_id": version_id},
            )
            connection.execute(
                text(
                    """
                    UPDATE knowledge_index_versions v
                    SET status = 'active', activated_at = :now,
                        chunk_count = (SELECT COUNT(*) FROM chunks WHERE index_version_id = v.id),
                        document_count = (SELECT COUNT(DISTINCT document_id) FROM chunks WHERE index_version_id = v.id),
                        error = ''
                    WHERE v.id = :version_id
                    """
                ),
                {"version_id": version_id, "now": utc_now()},
            )
            connection.execute(
                text(
                    """
                    UPDATE knowledge_bases
                    SET active_index_version_id = :version_id, pending_index_version_id = '', updated_at = :now
                    WHERE id = :id
                    """
                ),
                {"id": knowledge_base_id, "version_id": version_id, "now": utc_now()},
            )
        return next(item for item in self.list_index_versions(knowledge_base_id) if item.id == version_id)

    def fail_index_version(self, knowledge_base_id: str, version_id: str, error: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            connection.execute(
                text("UPDATE knowledge_index_versions SET status = 'failed', error = :error WHERE id = :version_id AND knowledge_base_id = :id"),
                {"id": knowledge_base_id, "version_id": version_id, "error": str(error)[:2000]},
            )
            connection.execute(
                text("UPDATE knowledge_bases SET pending_index_version_id = '' WHERE id = :id"), {"id": knowledge_base_id}
            )

    def list_index_versions(self, knowledge_base_id: str) -> List[KnowledgeIndexVersionRecord]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text("SELECT * FROM knowledge_index_versions WHERE knowledge_base_id = :id ORDER BY created_at DESC"),
                {"id": knowledge_base_id},
            ).mappings()
            return [_index_version_from_row(row) for row in rows]

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

    def update_data_source(
        self,
        source_id: str,
        *,
        status: str,
        metadata: Dict[str, Any],
    ) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    UPDATE data_sources
                    SET status = :status, metadata_json = CAST(:metadata AS JSONB)
                    WHERE id = :id
                    """
                ),
                {"id": source_id, "status": status, "metadata": json.dumps(metadata)},
            )
            if result.rowcount == 0:
                raise KeyError(f"Data source not found: {source_id}")

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

    def add_documents(self, documents: List[StoredKnowledgeDocument]) -> None:
        if not documents:
            return
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
                [
                    {**_document_to_dict(document), "metadata": json.dumps(document.metadata)}
                    for document in documents
                ],
            )

    def list_documents(self, knowledge_base_id: str) -> List[StoredKnowledgeDocument]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text("SELECT * FROM documents WHERE knowledge_base_id = :id ORDER BY title, id"),
                {"id": knowledge_base_id},
            ).mappings()
            return [_document_from_row(row) for row in rows]

    def list_documents_page(
        self,
        knowledge_base_id: str,
        *,
        query: str = "",
        offset: int = 0,
        limit: int = 25,
    ) -> tuple[List[StoredKnowledgeDocument], int]:
        from sqlalchemy import text

        normalized = query.strip()
        where = """
            knowledge_base_id = :id
            AND (
                :query = ''
                OR title ILIKE :pattern
                OR COALESCE(metadata_json->>'source_record_id', '') ILIKE :pattern
                OR COALESCE(metadata_json->>'url', metadata_json->>'uri', '') ILIKE :pattern
            )
        """
        params = {
            "id": knowledge_base_id,
            "query": normalized,
            "pattern": f"%{normalized}%",
            "offset": max(0, offset),
            "limit": max(1, limit),
        }
        with self.engine.begin() as connection:
            total = int(connection.execute(text(f"SELECT COUNT(*) FROM documents WHERE {where}"), params).scalar_one())
            rows = connection.execute(
                text(
                    f"""
                    SELECT * FROM documents
                    WHERE {where}
                    ORDER BY LOWER(title), id
                    OFFSET :offset LIMIT :limit
                    """
                ),
                params,
            ).mappings()
            return [_document_from_row(row) for row in rows], total

    def list_wixqa_source_record_ids(self, knowledge_base_id: str) -> List[str]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT metadata_json->>'source_record_id'
                    FROM documents
                    WHERE knowledge_base_id = :id
                      AND metadata_json->>'source_type' = 'wixqa_corpus'
                      AND COALESCE(metadata_json->>'source_record_id', '') <> ''
                    ORDER BY metadata_json->>'source_record_id'
                    """
                ),
                {"id": knowledge_base_id},
            )
            return [str(row[0]) for row in rows]

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

        target_version = next((chunk.index_version_id for chunk in chunks if chunk.index_version_id), "")
        with self.engine.begin() as connection:
            if target_version:
                connection.execute(
                    text("DELETE FROM chunks WHERE knowledge_base_id = :id AND index_version_id = :version_id"),
                    {"id": knowledge_base_id, "version_id": target_version},
                )
            else:
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
        if not chunks:
            return
        from sqlalchemy import text

        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO chunks (id, knowledge_base_id, document_id, chunk_index, text, token_count, metadata_json, index_version_id, parent_chunk_id)
                    VALUES (:id, :knowledge_base_id, :document_id, :chunk_index, :text, :token_count, CAST(:metadata AS JSONB), :index_version_id, :parent_chunk_id)
                    """
                ),
                [{**_chunk_to_dict(chunk), "metadata": json.dumps(chunk.metadata)} for chunk in chunks],
            )
            connection.execute(
                text(
                    """
                    INSERT INTO chunk_embeddings (chunk_id, embedding, embedding_model, embedding_deployment_id, embedding_dimension)
                    VALUES (:chunk_id, :embedding, :embedding_model, :embedding_deployment_id, :embedding_dimension)
                    """
                ),
                [
                    {
                        "chunk_id": chunk.id,
                        "embedding": _vector_literal(embedding),
                        "embedding_model": model,
                        "embedding_deployment_id": chunk.metadata.get("embedding_deployment_id", ""),
                        "embedding_dimension": len(embedding),
                    }
                    for chunk, embedding in zip(chunks, embeddings)
                ],
            )

    def list_chunks(self, knowledge_base_id: str, limit: int = 100) -> List[StoredKnowledgeChunk]:
        from sqlalchemy import text

        def _query() -> List[StoredKnowledgeChunk]:
            with self.engine.begin() as connection:
                active_version = connection.execute(
                    text("SELECT active_index_version_id FROM knowledge_bases WHERE id = :id"),
                    {"id": knowledge_base_id},
                ).scalar_one_or_none()
                if active_version is None:
                    raise KeyError(f"Knowledge base not found: {knowledge_base_id}")
                version_filter = "AND c.index_version_id = :active_version" if active_version else ""
                params = {"id": knowledge_base_id, "limit": limit, "active_version": active_version or ""}
                rows = connection.execute(
                    text(
                        f"""
                        SELECT c.*,
                               ce.embedding_model,
                               ce.embedding_deployment_id,
                               ce.embedding_dimension,
                               ce.embedding::text AS embedding_text
                        FROM chunks c
                        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
                        WHERE c.knowledge_base_id = :id
                          {version_filter}
                        ORDER BY c.document_id, c.chunk_index
                        LIMIT :limit
                        """
                    ),
                    params,
                ).mappings()
                return [_chunk_from_row(row) for row in rows]

        return _retry_transient_db_error(_query)

    def list_document_chunks(self, knowledge_base_id: str, document_id: str) -> List[StoredKnowledgeChunk]:
        return self.list_active_chunks(knowledge_base_id, document_ids=[document_id])

    def list_active_chunks(
        self,
        knowledge_base_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> List[StoredKnowledgeChunk]:
        from sqlalchemy import text

        selected = list(dict.fromkeys(document_ids or []))
        document_filter = "AND c.document_id = ANY(:document_ids)" if selected else ""

        def _query() -> List[StoredKnowledgeChunk]:
            with self.engine.begin() as connection:
                active_version = connection.execute(
                    text("SELECT active_index_version_id FROM knowledge_bases WHERE id = :id"),
                    {"id": knowledge_base_id},
                ).scalar_one_or_none()
                if active_version is None:
                    raise KeyError(f"Knowledge base not found: {knowledge_base_id}")
                version_filter = "AND c.index_version_id = :active_version" if active_version else ""
                rows = connection.execute(
                    text(
                        f"""
                        SELECT c.*,
                               ce.embedding_model,
                               ce.embedding_deployment_id,
                               ce.embedding_dimension,
                               ce.embedding::text AS embedding_text
                        FROM chunks c
                        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
                        WHERE c.knowledge_base_id = :id
                          {version_filter}
                          {document_filter}
                        ORDER BY c.document_id, c.chunk_index
                        """
                    ),
                    {
                        "id": knowledge_base_id,
                        "active_version": active_version or "",
                        "document_ids": selected,
                    },
                ).mappings()
                return [_chunk_from_row(row) for row in rows]

        return _retry_transient_db_error(_query)

    def search_chunks_by_embedding(
        self,
        knowledge_base_id: str,
        embedding: List[float],
        limit: int = 10,
    ) -> List[tuple[StoredKnowledgeChunk, float]]:
        from sqlalchemy import text

        def _query() -> List[tuple[StoredKnowledgeChunk, float]]:
            with self.engine.begin() as connection:
                active_version = connection.execute(
                    text("SELECT active_index_version_id FROM knowledge_bases WHERE id = :id"),
                    {"id": knowledge_base_id},
                ).scalar_one_or_none()
                if active_version is None:
                    raise KeyError(f"Knowledge base not found: {knowledge_base_id}")
                version_filter = "AND c.index_version_id = :active_version" if active_version else ""
                rows = connection.execute(
                    text(
                        f"""
                        SELECT c.*,
                               ce.embedding_model,
                               ce.embedding_deployment_id,
                               ce.embedding_dimension,
                               ce.embedding::text AS embedding_text,
                               1 - (ce.embedding <=> CAST(:embedding AS vector)) AS score
                        FROM chunks c
                        JOIN chunk_embeddings ce ON ce.chunk_id = c.id
                        WHERE c.knowledge_base_id = :id
                          {version_filter}
                          AND (ce.embedding_dimension = 0 OR ce.embedding_dimension = :dimension)
                        ORDER BY ce.embedding <=> CAST(:embedding AS vector)
                        LIMIT :limit
                        """
                    ),
                    {
                        "id": knowledge_base_id,
                        "active_version": active_version or "",
                        "embedding": _vector_literal(embedding),
                        "dimension": len(embedding),
                        "limit": limit,
                    },
                ).mappings()
                return [(_chunk_from_row(row), float(row.get("score") or 0.0)) for row in rows]

        return _retry_transient_db_error(_query)

    def _legacy_join_list_chunks(self, knowledge_base_id: str, limit: int = 100) -> List[StoredKnowledgeChunk]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT c.*,
                           ce.embedding_model,
                           ce.embedding_deployment_id,
                           ce.embedding_dimension,
                           ce.embedding::text AS embedding_text
                    FROM chunks c
                    LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
                    JOIN knowledge_bases kb ON kb.id = c.knowledge_base_id
                    WHERE c.knowledge_base_id = :id
                      AND (kb.active_index_version_id = '' OR c.index_version_id = kb.active_index_version_id)
                    ORDER BY c.document_id, c.chunk_index
                    LIMIT :limit
                    """
                ),
                {"id": knowledge_base_id, "limit": limit},
            ).mappings()
            return [_chunk_from_row(row) for row in rows]

    def _legacy_join_search_chunks_by_embedding(
        self,
        knowledge_base_id: str,
        embedding: List[float],
        limit: int = 10,
    ) -> List[tuple[StoredKnowledgeChunk, float]]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT c.*,
                           ce.embedding_model,
                           ce.embedding_deployment_id,
                           ce.embedding_dimension,
                           ce.embedding::text AS embedding_text,
                           1 - (ce.embedding <=> CAST(:embedding AS vector)) AS score
                    FROM chunks c
                    JOIN chunk_embeddings ce ON ce.chunk_id = c.id
                    JOIN knowledge_bases kb ON kb.id = c.knowledge_base_id
                    WHERE c.knowledge_base_id = :id
                      AND (kb.active_index_version_id = '' OR c.index_version_id = kb.active_index_version_id)
                      AND (ce.embedding_dimension = 0 OR ce.embedding_dimension = :dimension)
                    ORDER BY ce.embedding <=> CAST(:embedding AS vector)
                    LIMIT :limit
                    """
                ),
                {"id": knowledge_base_id, "embedding": _vector_literal(embedding), "dimension": len(embedding), "limit": limit},
            ).mappings()
            return [(_chunk_from_row(row), float(row.get("score") or 0.0)) for row in rows]

    def list_ingestion_runs(self, knowledge_base_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT *
                    FROM ingestion_runs
                    WHERE knowledge_base_id = :id
                    ORDER BY finished_at DESC
                    LIMIT :limit
                    """
                ),
                {"id": knowledge_base_id, "limit": limit},
            ).mappings()
            return [
                {
                    "id": row["id"],
                    "knowledge_base_id": row["knowledge_base_id"],
                    "source_id": row.get("source_id"),
                    "status": row["status"],
                    "counts": dict(row.get("counts_json") or {}),
                    "error": row.get("error"),
                    "started_at": row["started_at"],
                    "finished_at": row["finished_at"],
                }
                for row in rows
            ]

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
        "index_versions": {},
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
        "index_version_id": record.index_version_id or record.metadata.get("index_version_id", ""),
        "parent_chunk_id": record.parent_chunk_id or record.metadata.get("parent_chunk_id", ""),
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


def _chunk_from_dict(payload: Dict[str, Any], embedding_payload: Optional[Dict[str, Any]] = None) -> StoredKnowledgeChunk:
    embedding = embedding_payload.get("embedding", []) if embedding_payload else []
    return StoredKnowledgeChunk(
        id=payload["id"],
        knowledge_base_id=payload["knowledge_base_id"],
        document_id=payload["document_id"],
        chunk_index=int(payload["chunk_index"]),
        text=payload["text"],
        token_count=int(payload["token_count"]),
        metadata=dict(payload.get("metadata", {})),
        embedding_model=(embedding_payload or {}).get("embedding_model", ""),
        embedding_dimension=len(embedding) if isinstance(embedding, list) else _vector_dimension(str(embedding)),
        has_embedding=bool(embedding_payload),
        index_version_id=payload.get("index_version_id", ""),
        parent_chunk_id=payload.get("parent_chunk_id", ""),
    )


def _index_version_to_dict(record: KnowledgeIndexVersionRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "knowledge_base_id": record.knowledge_base_id,
        "status": record.status,
        "chunking_configuration": record.chunking_configuration,
        "embedding_deployment_id": record.embedding_deployment_id,
        "embedding_provider": record.embedding_provider,
        "embedding_model": record.embedding_model,
        "embedding_dimension": record.embedding_dimension,
        "document_count": record.document_count,
        "chunk_count": record.chunk_count,
        "error": record.error,
        "created_at": record.created_at,
        "activated_at": record.activated_at,
    }


def _index_version_from_dict(payload: Dict[str, Any]) -> KnowledgeIndexVersionRecord:
    return KnowledgeIndexVersionRecord(
        id=payload["id"], knowledge_base_id=payload["knowledge_base_id"], status=payload["status"],
        chunking_configuration=dict(payload.get("chunking_configuration", {})),
        embedding_deployment_id=payload.get("embedding_deployment_id", ""),
        embedding_provider=payload.get("embedding_provider", ""), embedding_model=payload.get("embedding_model", ""),
        embedding_dimension=int(payload.get("embedding_dimension") or 0), document_count=int(payload.get("document_count") or 0),
        chunk_count=int(payload.get("chunk_count") or 0), error=payload.get("error", ""),
        created_at=payload.get("created_at", ""), activated_at=payload.get("activated_at", ""),
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
    embedding_text = row.get("embedding_text") or ""
    return StoredKnowledgeChunk(
        id=row["id"],
        knowledge_base_id=row["knowledge_base_id"],
        document_id=row["document_id"],
        chunk_index=int(row["chunk_index"]),
        text=row["text"],
        token_count=int(row["token_count"]),
        metadata=dict(row.get("metadata_json") or {}),
        embedding_model=row.get("embedding_model") or "",
        embedding_dimension=int(row.get("embedding_dimension") or 0) or _vector_dimension(embedding_text),
        has_embedding=bool(row.get("embedding_model")),
        index_version_id=row.get("index_version_id") or "",
        parent_chunk_id=row.get("parent_chunk_id") or "",
    )


def _index_version_from_row(row: Any) -> KnowledgeIndexVersionRecord:
    return KnowledgeIndexVersionRecord(
        id=row["id"], knowledge_base_id=row["knowledge_base_id"], status=row["status"],
        chunking_configuration=dict(row.get("chunking_configuration_json") or {}),
        embedding_deployment_id=row.get("embedding_deployment_id") or "",
        embedding_provider=row.get("embedding_provider") or "", embedding_model=row.get("embedding_model") or "",
        embedding_dimension=int(row.get("embedding_dimension") or 0), document_count=int(row.get("document_count") or 0),
        chunk_count=int(row.get("chunk_count") or 0), error=row.get("error") or "",
        created_at=row.get("created_at") or "", activated_at=row.get("activated_at") or "",
    )


def _vector_literal(values: List[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _retry_transient_db_error(operation: Any, attempts: int = 3) -> Any:
    last_error: Optional[Exception] = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if not _is_transient_db_error(exc) or attempt >= attempts - 1:
                raise
            last_error = exc
            time.sleep(0.08 * (attempt + 1))
    if last_error:
        raise last_error
    return operation()


def _is_transient_db_error(error: Exception) -> bool:
    value = f"{type(error).__name__}: {error}".lower()
    return any(
        marker in value
        for marker in [
            "deadlock detected",
            "deadlockdetected",
            "lock timeout",
            "could not serialize access",
            "serializationfailure",
        ]
    )


def _vector_dimension(value: str) -> int:
    stripped = value.strip().strip("[]")
    if not stripped:
        return 0
    return stripped.count(",") + 1


def _coerce_vector(value: Any) -> List[float]:
    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip().strip("[]")
        if not stripped:
            return []
        return [float(part.strip()) for part in stripped.split(",") if part.strip()]
    return []


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
