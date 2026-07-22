from __future__ import annotations

import gzip
import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


TRACE_SCHEMA_VERSION = "1.0"
TERMINAL_TRACE_STATUSES = {"completed", "failed", "cancelled"}


class TraceError(ValueError):
    """Raised when a durable RAG trace cannot be stored or loaded."""


@dataclass(frozen=True)
class TraceRecord:
    id: str
    request_id: str
    conversation_id: str = ""
    message_id: str = ""
    message_version_id: str = ""
    status: str = "running"
    route_level: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    warning_count: int = 0
    span_count: int = 0
    artifact_path: str = ""
    artifact_sha256: str = ""
    artifact_size: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)
    expires_at: str = ""
    created_at: str = ""
    updated_at: str = ""


@dataclass(frozen=True)
class SpanHandle:
    span_id: str
    sequence: int
    name: str
    category: str
    parent_span_id: str
    started_at: str
    started_perf: float
    input_payload: Dict[str, Any]


class TraceRepository(Protocol):
    def initialize(self) -> None: ...

    def save(self, record: TraceRecord) -> TraceRecord: ...

    def get(self, trace_id: str) -> TraceRecord: ...

    def find_by_message(self, message_id: str, message_version_id: str = "") -> Optional[TraceRecord]: ...

    def list_expired(self, now: str, limit: int = 100) -> List[TraceRecord]: ...

    def delete(self, trace_id: str) -> None: ...


class FileTraceRepository:
    """Directory-based metadata repository used by offline/test mode."""

    def __init__(self, root: str):
        self.root = Path(root).resolve() / "index"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, record: TraceRecord) -> TraceRecord:
        self.initialize()
        path = self._path(record.id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
        return record

    def get(self, trace_id: str) -> TraceRecord:
        path = self._path(trace_id)
        if not path.is_file():
            raise KeyError(f"RAG trace not found: {trace_id}")
        return TraceRecord(**json.loads(path.read_text(encoding="utf-8")))

    def find_by_message(self, message_id: str, message_version_id: str = "") -> Optional[TraceRecord]:
        records = self._all()
        matches = [item for item in records if item.message_id == message_id]
        if message_version_id:
            matches = [item for item in matches if item.message_version_id == message_version_id]
        return max(matches, key=lambda item: item.updated_at, default=None)

    def list_expired(self, now: str, limit: int = 100) -> List[TraceRecord]:
        return sorted(
            [item for item in self._all() if item.expires_at and item.expires_at <= now],
            key=lambda item: item.expires_at,
        )[: max(1, min(limit, 1000))]

    def delete(self, trace_id: str) -> None:
        path = self._path(trace_id)
        if path.exists():
            path.unlink()

    def _path(self, trace_id: str) -> Path:
        safe = _safe_trace_id(trace_id)
        return self.root / f"{safe}.json"

    def _all(self) -> List[TraceRecord]:
        self.initialize()
        records: List[TraceRecord] = []
        for path in self.root.glob("trace-*.json"):
            try:
                records.append(TraceRecord(**json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return records


class PostgresTraceRepository:
    def __init__(self, database_url: str):
        try:
            from sqlalchemy import create_engine  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise TraceError("Install the api extra to use PostgreSQL trace storage.") from exc
        self.engine = create_engine(database_url, future=True)

    def initialize(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS rag_traces (
            id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL DEFAULT '',
            conversation_id TEXT NOT NULL DEFAULT '',
            message_id TEXT NOT NULL DEFAULT '',
            message_version_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'running',
            route_level TEXT NOT NULL DEFAULT '',
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL DEFAULT '',
            duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
            warning_count INTEGER NOT NULL DEFAULT 0,
            span_count INTEGER NOT NULL DEFAULT 0,
            artifact_path TEXT NOT NULL DEFAULT '',
            artifact_sha256 TEXT NOT NULL DEFAULT '',
            artifact_size BIGINT NOT NULL DEFAULT 0,
            summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_rag_traces_request ON rag_traces(request_id);
        CREATE INDEX IF NOT EXISTS idx_rag_traces_message ON rag_traces(message_id, message_version_id);
        CREATE INDEX IF NOT EXISTS idx_rag_traces_expiry ON rag_traces(expires_at);
        """
        with self.engine.begin() as connection:
            for statement in [part.strip() for part in ddl.split(";") if part.strip()]:
                connection.exec_driver_sql(statement)

    def save(self, record: TraceRecord) -> TraceRecord:
        from sqlalchemy import text

        payload = asdict(record)
        payload["summary"] = json.dumps(record.summary, ensure_ascii=False)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO rag_traces (
                        id, request_id, conversation_id, message_id, message_version_id,
                        status, route_level, started_at, finished_at, duration_ms,
                        input_tokens, output_tokens, estimated_cost_usd, warning_count, span_count,
                        artifact_path, artifact_sha256, artifact_size, summary_json,
                        expires_at, created_at, updated_at
                    ) VALUES (
                        :id, :request_id, :conversation_id, :message_id, :message_version_id,
                        :status, :route_level, :started_at, :finished_at, :duration_ms,
                        :input_tokens, :output_tokens, :estimated_cost_usd, :warning_count, :span_count,
                        :artifact_path, :artifact_sha256, :artifact_size, CAST(:summary AS JSONB),
                        :expires_at, :created_at, :updated_at
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        conversation_id = EXCLUDED.conversation_id,
                        message_id = EXCLUDED.message_id,
                        message_version_id = EXCLUDED.message_version_id,
                        status = EXCLUDED.status,
                        route_level = EXCLUDED.route_level,
                        finished_at = EXCLUDED.finished_at,
                        duration_ms = EXCLUDED.duration_ms,
                        input_tokens = EXCLUDED.input_tokens,
                        output_tokens = EXCLUDED.output_tokens,
                        estimated_cost_usd = EXCLUDED.estimated_cost_usd,
                        warning_count = EXCLUDED.warning_count,
                        span_count = EXCLUDED.span_count,
                        artifact_path = EXCLUDED.artifact_path,
                        artifact_sha256 = EXCLUDED.artifact_sha256,
                        artifact_size = EXCLUDED.artifact_size,
                        summary_json = EXCLUDED.summary_json,
                        expires_at = EXCLUDED.expires_at,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                payload,
            )
        return record

    def get(self, trace_id: str) -> TraceRecord:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            row = connection.execute(text("SELECT * FROM rag_traces WHERE id = :id"), {"id": trace_id}).mappings().first()
        if not row:
            raise KeyError(f"RAG trace not found: {trace_id}")
        return _trace_record_from_row(row)

    def find_by_message(self, message_id: str, message_version_id: str = "") -> Optional[TraceRecord]:
        from sqlalchemy import text

        where_version = "AND message_version_id = :version_id" if message_version_id else ""
        params = {"message_id": message_id, "version_id": message_version_id}
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    f"SELECT * FROM rag_traces WHERE message_id = :message_id {where_version} "
                    "ORDER BY updated_at DESC LIMIT 1"
                ),
                params,
            ).mappings().first()
        return _trace_record_from_row(row) if row else None

    def list_expired(self, now: str, limit: int = 100) -> List[TraceRecord]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text("SELECT * FROM rag_traces WHERE expires_at <= :now ORDER BY expires_at LIMIT :limit"),
                {"now": now, "limit": max(1, min(limit, 1000))},
            ).mappings()
            return [_trace_record_from_row(row) for row in rows]

    def delete(self, trace_id: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            connection.execute(text("DELETE FROM rag_traces WHERE id = :id"), {"id": trace_id})


class TraceArtifactStore:
    def __init__(self, root: str):
        self.root = (Path(root).resolve() / "artifacts")
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, trace_id: str, report: Dict[str, Any], max_bytes: int) -> Dict[str, Any]:
        safe_trace_id = _safe_trace_id(trace_id)
        safe_report = sanitize_trace_value(report)
        bounded = _bounded_report(safe_report, max_bytes)
        raw = json.dumps(bounded, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        compressed = gzip.compress(raw, compresslevel=6)
        digest = hashlib.sha256(compressed).hexdigest()
        directory = self.root / safe_trace_id[-2:]
        directory.mkdir(parents=True, exist_ok=True)
        path = (directory / f"{safe_trace_id}.json.gz").resolve()
        if self.root not in path.parents:
            raise TraceError("Trace artifact path is outside the configured trace store.")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(compressed)
        os.replace(temporary, path)
        return {
            "path": str(path),
            "sha256": digest,
            "size": len(compressed),
            "uncompressed_size": len(raw),
            "report": bounded,
        }

    def read(self, path: str) -> Dict[str, Any]:
        resolved = Path(path).resolve()
        if self.root not in resolved.parents or not resolved.is_file():
            raise TraceError("Trace artifact is missing or outside the configured trace store.")
        try:
            return json.loads(gzip.decompress(resolved.read_bytes()).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TraceError(f"Trace artifact is invalid: {exc}") from exc

    def delete(self, path: str) -> None:
        resolved = Path(path).resolve()
        if self.root not in resolved.parents:
            raise TraceError("Refusing to delete an artifact outside the configured trace store.")
        if resolved.exists():
            resolved.unlink()


class TraceService:
    def __init__(
        self,
        repository: TraceRepository,
        artifact_store: TraceArtifactStore,
        *,
        retention_days: int = 30,
        max_bytes: int = 10 * 1024 * 1024,
    ):
        self.repository = repository
        self.artifact_store = artifact_store
        self.retention_days = max(1, retention_days)
        self.max_bytes = max(65536, max_bytes)
        self.repository.initialize()

    def create_recorder(
        self,
        request_id: str,
        *,
        conversation_id: str = "",
        message_id: str = "",
        message_version_id: str = "",
    ) -> "TraceRecorder":
        self.purge_expired(limit=25)
        return TraceRecorder(
            self,
            request_id=request_id,
            conversation_id=conversation_id,
            message_id=message_id,
            message_version_id=message_version_id,
        )

    def save(self, report: Dict[str, Any]) -> TraceRecord:
        artifact = self.artifact_store.write(str(report["trace_id"]), report, self.max_bytes)
        persisted_report = artifact["report"]
        summary = dict(persisted_report.get("summary") or {})
        now = utc_now()
        record = TraceRecord(
            id=str(report["trace_id"]),
            request_id=str(report.get("request_id") or ""),
            conversation_id=str(report.get("conversation_id") or ""),
            message_id=str(report.get("message_id") or ""),
            message_version_id=str(report.get("message_version_id") or ""),
            status=str(report.get("status") or "running"),
            route_level=str(summary.get("route_level") or ""),
            started_at=str(report.get("started_at") or now),
            finished_at=str(report.get("finished_at") or ""),
            duration_ms=float(summary.get("duration_ms") or 0.0),
            input_tokens=int(summary.get("input_tokens") or 0),
            output_tokens=int(summary.get("output_tokens") or 0),
            estimated_cost_usd=float(summary.get("estimated_cost_usd") or 0.0),
            warning_count=int(summary.get("warning_count") or 0),
            span_count=len(persisted_report.get("spans") or []),
            artifact_path=artifact["path"],
            artifact_sha256=artifact["sha256"],
            artifact_size=artifact["size"],
            summary=summary,
            expires_at=str(report.get("expires_at") or _future_iso(self.retention_days)),
            created_at=str(report.get("created_at") or now),
            updated_at=now,
        )
        return self.repository.save(record)

    def get_report(self, trace_id: str) -> Dict[str, Any]:
        record = self.repository.get(trace_id)
        if record.expires_at and record.expires_at <= utc_now():
            self.delete(trace_id)
            raise KeyError(f"RAG trace not found: {trace_id}")
        return self.artifact_store.read(record.artifact_path)

    def get_record(self, trace_id: str) -> TraceRecord:
        record = self.repository.get(trace_id)
        if record.expires_at and record.expires_at <= utc_now():
            self.delete(trace_id)
            raise KeyError(f"RAG trace not found: {trace_id}")
        return record

    def find_message_report(self, message_id: str, message_version_id: str = "") -> Dict[str, Any]:
        record = self.repository.find_by_message(message_id, message_version_id)
        if record is None:
            raise KeyError(f"RAG trace not found for message: {message_id}")
        return self.get_report(record.id)

    def delete(self, trace_id: str) -> None:
        try:
            record = self.repository.get(trace_id)
        except KeyError:
            return
        if record.artifact_path:
            self.artifact_store.delete(record.artifact_path)
        self.repository.delete(trace_id)

    def purge_expired(self, limit: int = 100) -> int:
        expired = self.repository.list_expired(utc_now(), limit=limit)
        for record in expired:
            self.delete(record.id)
        return len(expired)


class TraceRecorder:
    def __init__(
        self,
        service: Optional[TraceService],
        *,
        request_id: str,
        conversation_id: str = "",
        message_id: str = "",
        message_version_id: str = "",
    ):
        now = utc_now()
        self.service = service
        self._started_perf = time.perf_counter()
        self._lock = threading.RLock()
        self._root_span_id = f"span-{uuid.uuid4().hex}"
        self.report: Dict[str, Any] = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_id": f"trace-{uuid.uuid4().hex}",
            "request_id": request_id,
            "conversation_id": conversation_id,
            "message_id": message_id,
            "message_version_id": message_version_id,
            "status": "running",
            "started_at": now,
            "finished_at": "",
            "created_at": now,
            "expires_at": _future_iso(service.retention_days if service else 30),
            "summary": {},
            "spans": [
                {
                    "span_id": self._root_span_id,
                    "parent_span_id": "",
                    "sequence": 1,
                    "name": "Adaptive RAG answer",
                    "category": "orchestration",
                    "status": "running",
                    "detail": "End-to-end answer orchestration.",
                    "started_at": now,
                    "finished_at": "",
                    "duration_ms": 0.0,
                    "input": {"request_id": request_id},
                    "output": {},
                    "metrics": {},
                    "model_usage_event_ids": [],
                    "warning": "",
                    "error": "",
                }
            ],
        }
        self.checkpoint()

    @property
    def trace_id(self) -> str:
        return str(self.report["trace_id"])

    def begin_span(
        self,
        name: str,
        category: str,
        *,
        input_payload: Optional[Dict[str, Any]] = None,
        parent_span_id: str = "",
    ) -> SpanHandle:
        with self._lock:
            return SpanHandle(
                span_id=f"span-{uuid.uuid4().hex}",
                sequence=len(self.report["spans"]) + 1,
                name=name,
                category=category,
                parent_span_id=parent_span_id or self._root_span_id,
                started_at=utc_now(),
                started_perf=time.perf_counter(),
                input_payload=dict(input_payload or {}),
            )

    def finish_span(
        self,
        handle: SpanHandle,
        *,
        status: str = "completed",
        detail: str = "",
        output_payload: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        model_usage_event_ids: Optional[List[str]] = None,
        warning: str = "",
        error: str = "",
    ) -> Dict[str, Any]:
        span = {
            "span_id": handle.span_id,
            "parent_span_id": handle.parent_span_id,
            "sequence": handle.sequence,
            "name": handle.name,
            "category": handle.category,
            "status": status,
            "detail": detail,
            "started_at": handle.started_at,
            "finished_at": utc_now(),
            "duration_ms": round((time.perf_counter() - handle.started_perf) * 1000, 3),
            "input": handle.input_payload,
            "output": dict(output_payload or {}),
            "metrics": dict(metrics or {}),
            "model_usage_event_ids": list(model_usage_event_ids or []),
            "warning": warning,
            "error": error,
        }
        with self._lock:
            self.report["spans"].append(span)
            self._refresh_summary()
            self.checkpoint()
        return span

    def add_instant_span(
        self,
        name: str,
        category: str,
        *,
        status: str = "completed",
        detail: str = "",
        input_payload: Optional[Dict[str, Any]] = None,
        output_payload: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        parent_span_id: str = "",
    ) -> Dict[str, Any]:
        handle = self.begin_span(name, category, input_payload=input_payload, parent_span_id=parent_span_id)
        return self.finish_span(
            handle,
            status=status,
            detail=detail,
            output_payload=output_payload,
            metrics=metrics,
        )

    def add_observed_span(
        self,
        name: str,
        category: str,
        *,
        duration_ms: float,
        status: str = "completed",
        detail: str = "",
        input_payload: Optional[Dict[str, Any]] = None,
        output_payload: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        model_usage_event_ids: Optional[List[str]] = None,
        warning: str = "",
        error: str = "",
        parent_span_id: str = "",
    ) -> Dict[str, Any]:
        finished = datetime.now(timezone.utc)
        observed_duration = max(float(duration_ms or 0.0), 0.0)
        started = finished - timedelta(milliseconds=observed_duration)
        with self._lock:
            span = {
                "span_id": f"span-{uuid.uuid4().hex}",
                "parent_span_id": parent_span_id or self._root_span_id,
                "sequence": len(self.report["spans"]) + 1,
                "name": name,
                "category": category,
                "status": status,
                "detail": detail,
                "started_at": started.isoformat(),
                "finished_at": finished.isoformat(),
                "duration_ms": round(observed_duration, 3),
                "input": dict(input_payload or {}),
                "output": dict(output_payload or {}),
                "metrics": dict(metrics or {}),
                "model_usage_event_ids": list(model_usage_event_ids or []),
                "warning": warning,
                "error": error,
            }
            self.report["spans"].append(span)
            self._refresh_summary()
            self.checkpoint()
            return span

    def update_links(
        self,
        *,
        conversation_id: str = "",
        message_id: str = "",
        message_version_id: str = "",
    ) -> None:
        with self._lock:
            if conversation_id:
                self.report["conversation_id"] = conversation_id
            if message_id:
                self.report["message_id"] = message_id
            if message_version_id:
                self.report["message_version_id"] = message_version_id
            self.checkpoint()

    def finalize(self, status: str, *, route_level: str = "", error: str = "") -> Dict[str, Any]:
        with self._lock:
            self.report["status"] = status
            self.report["finished_at"] = utc_now()
            if error:
                self.report["error"] = error
            root = self.report["spans"][0]
            root["status"] = status
            root["finished_at"] = self.report["finished_at"]
            root["duration_ms"] = round((time.perf_counter() - self._started_perf) * 1000, 3)
            root["output"] = {"status": status, "route_level": route_level}
            root["error"] = error
            self._refresh_summary(route_level=route_level)
            self.checkpoint()
            return dict(self.report)

    def checkpoint(self) -> None:
        if self.service is not None:
            self.service.save(self.report)

    def legacy_steps(self) -> List[Dict[str, Any]]:
        return [
            {
                "step": span["name"],
                "status": span["status"],
                "detail": span.get("detail") or "",
                "metadata": {
                    **dict(span.get("metrics") or {}),
                    "span_id": span["span_id"],
                    "parent_span_id": span.get("parent_span_id") or "",
                    "started_at": span["started_at"],
                    "finished_at": span["finished_at"],
                    "duration_ms": span["duration_ms"],
                },
            }
            for span in self.report["spans"]
        ]

    def _refresh_summary(self, *, route_level: str = "") -> None:
        spans = list(self.report.get("spans") or [])
        input_tokens = sum(int(span.get("metrics", {}).get("input_tokens") or 0) for span in spans)
        output_tokens = sum(int(span.get("metrics", {}).get("output_tokens") or 0) for span in spans)
        cost = sum(float(span.get("metrics", {}).get("estimated_cost_usd") or 0.0) for span in spans)
        current = dict(self.report.get("summary") or {})
        self.report["summary"] = {
            **current,
            "route_level": route_level or current.get("route_level", ""),
            "duration_ms": round((time.perf_counter() - self._started_perf) * 1000, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "estimated_cost_usd": round(cost, 10),
            "warning_count": sum(1 for span in spans if span.get("status") == "warning" or span.get("warning")),
            "error_count": sum(1 for span in spans if span.get("status") == "failed" or span.get("error")),
            "span_count": len(spans),
        }


def sanitize_trace_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if depth > 30:
        return "[maximum serialization depth reached]"
    normalized_key = key.casefold().replace("-", "_")
    if _is_sensitive_key(normalized_key):
        return "[REDACTED]"
    if is_dataclass(value):
        value = asdict(value)
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump()
        except Exception:
            value = str(value)
    if isinstance(value, dict):
        return {
            str(item_key): sanitize_trace_value(item_value, key=str(item_key), depth=depth + 1)
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if items and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in items) and (
            "embedding" in normalized_key or "vector" in normalized_key
        ):
            return {"redacted_numeric_vector": True, "dimension": len(items)}
        return [sanitize_trace_value(item, key=key, depth=depth + 1) for item in items]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return {"binary": True, "size": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    return str(value)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _future_iso(days: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=max(1, days))).isoformat()


def _safe_trace_id(trace_id: str) -> str:
    cleaned = "".join(char for char in str(trace_id) if char.isalnum() or char in {"-", "_"})
    if not cleaned or cleaned != trace_id:
        raise TraceError("Invalid trace ID.")
    return cleaned


def _is_sensitive_key(key: str) -> bool:
    if key in {"authorization", "proxy_authorization", "password", "secret", "api_key", "apikey", "access_token"}:
        return True
    return any(marker in key for marker in ("credential_secret", "encrypted_credential", "authorization_header", "api_key"))


def _bounded_report(report: Dict[str, Any], max_bytes: int) -> Dict[str, Any]:
    encoded = _json_bytes(report)
    if len(encoded) <= max_bytes:
        return report
    bounded = _truncate_value(report, string_limit=4000, list_limit=100)
    encoded = _json_bytes(bounded)
    if len(encoded) > max_bytes:
        bounded = _truncate_value(report, string_limit=1000, list_limit=30)
        encoded = _json_bytes(bounded)
    if len(encoded) > max_bytes:
        bounded = dict(report)
        bounded["spans"] = [
            {
                **{key: value for key, value in span.items() if key not in {"input", "output"}},
                "input": {"truncated": True},
                "output": {"truncated": True},
            }
            for span in list(report.get("spans") or [])
        ]
    truncation = {
        "truncated": True,
        "original_uncompressed_size": len(_json_bytes(report)),
        "max_uncompressed_size": max_bytes,
        "reason": "Trace payload exceeded ARAGBIZ_TRACE_MAX_BYTES.",
    }
    bounded["truncation"] = truncation
    if len(_json_bytes(bounded)) > max_bytes:
        summaries = []
        for span in list(report.get("spans") or []):
            summaries.append(
                {
                    key: _truncate_value(value, string_limit=300, list_limit=10)
                    for key, value in span.items()
                    if key not in {"input", "output"}
                }
            )
        bounded = {
            key: value
            for key, value in report.items()
            if key not in {"spans", "error"}
        }
        bounded["spans"] = summaries
        bounded["error"] = _truncate_value(report.get("error", ""), string_limit=500, list_limit=10)
        bounded["truncation"] = truncation
    while len(_json_bytes(bounded)) > max_bytes and bounded.get("spans"):
        bounded["spans"].pop()
        bounded["truncation"]["omitted_spans"] = len(report.get("spans") or []) - len(bounded["spans"])
    if len(_json_bytes(bounded)) > max_bytes:
        bounded = {
            "schema_version": report.get("schema_version", TRACE_SCHEMA_VERSION),
            "trace_id": report.get("trace_id", ""),
            "request_id": report.get("request_id", ""),
            "status": report.get("status", ""),
            "started_at": report.get("started_at", ""),
            "finished_at": report.get("finished_at", ""),
            "summary": report.get("summary", {}),
            "spans": [],
            "truncation": truncation,
        }
    return bounded


def _truncate_value(value: Any, *, string_limit: int, list_limit: int) -> Any:
    if isinstance(value, str):
        if len(value) <= string_limit:
            return value
        return value[:string_limit] + f"\n[truncated {len(value) - string_limit} characters]"
    if isinstance(value, list):
        selected = [_truncate_value(item, string_limit=string_limit, list_limit=list_limit) for item in value[:list_limit]]
        if len(value) > list_limit:
            selected.append({"truncated_items": len(value) - list_limit})
        return selected
    if isinstance(value, dict):
        return {
            key: _truncate_value(item, string_limit=string_limit, list_limit=list_limit)
            for key, item in value.items()
        }
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _trace_record_from_row(row: Any) -> TraceRecord:
    return TraceRecord(
        id=row["id"],
        request_id=row.get("request_id") or "",
        conversation_id=row.get("conversation_id") or "",
        message_id=row.get("message_id") or "",
        message_version_id=row.get("message_version_id") or "",
        status=row.get("status") or "running",
        route_level=row.get("route_level") or "",
        started_at=row.get("started_at") or "",
        finished_at=row.get("finished_at") or "",
        duration_ms=float(row.get("duration_ms") or 0.0),
        input_tokens=int(row.get("input_tokens") or 0),
        output_tokens=int(row.get("output_tokens") or 0),
        estimated_cost_usd=float(row.get("estimated_cost_usd") or 0.0),
        warning_count=int(row.get("warning_count") or 0),
        span_count=int(row.get("span_count") or 0),
        artifact_path=row.get("artifact_path") or "",
        artifact_sha256=row.get("artifact_sha256") or "",
        artifact_size=int(row.get("artifact_size") or 0),
        summary=dict(row.get("summary_json") or {}),
        expires_at=row.get("expires_at") or "",
        created_at=row.get("created_at") or "",
        updated_at=row.get("updated_at") or "",
    )
