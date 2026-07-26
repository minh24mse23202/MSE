from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


JOB_STATUSES = {"queued", "running", "completed", "failed", "cancel_requested", "cancelled"}


class JobError(ValueError):
    """Raised when a durable background job cannot be managed."""


@dataclass(frozen=True)
class BackgroundJob:
    id: str
    job_type: str
    status: str
    payload: Dict[str, Any]
    progress: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    attempts: int = 0
    max_attempts: int = 3
    idempotency_key: str = ""
    worker_id: str = ""
    lease_until: str = ""
    available_at: str = ""
    created_at: str = ""
    updated_at: str = ""
    finished_at: str = ""


class JobRepository(Protocol):
    def initialize(self) -> None: ...

    def save(self, job: BackgroundJob) -> BackgroundJob: ...

    def get(self, job_id: str) -> BackgroundJob: ...

    def list(self, *, status: str = "", limit: int = 100) -> List[BackgroundJob]: ...

    def claim(self, worker_id: str, lease_seconds: int = 120) -> Optional[BackgroundJob]: ...


class JsonJobRepository:
    def __init__(self, path: str):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"jobs": {}}, indent=2), encoding="utf-8")

    def save(self, job: BackgroundJob) -> BackgroundJob:
        state = self._read()
        state["jobs"][job.id] = asdict(job)
        self.path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
        return job

    def get(self, job_id: str) -> BackgroundJob:
        payload = self._read()["jobs"].get(job_id)
        if not payload:
            raise KeyError(f"Background job not found: {job_id}")
        return BackgroundJob(**payload)

    def list(self, *, status: str = "", limit: int = 100) -> List[BackgroundJob]:
        jobs = [BackgroundJob(**item) for item in self._read()["jobs"].values()]
        if status:
            jobs = [job for job in jobs if job.status == status]
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs[: max(1, min(limit, 1000))]

    def claim(self, worker_id: str, lease_seconds: int = 120) -> Optional[BackgroundJob]:
        now = utc_now()
        candidates = [
            job for job in self.list(limit=1000)
            if job.status == "queued" and (not job.available_at or job.available_at <= now)
        ]
        if not candidates:
            return None
        job = sorted(candidates, key=lambda item: item.created_at)[0]
        claimed = replace(
            job,
            status="running",
            worker_id=worker_id,
            attempts=job.attempts + 1,
            lease_until=(datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat(),
            updated_at=now,
        )
        return self.save(claimed)

    def _read(self) -> Dict[str, Any]:
        self.initialize()
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
        state.setdefault("jobs", {})
        return state


class PostgresJobRepository:
    def __init__(self, database_url: str):
        try:
            from sqlalchemy import create_engine  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise JobError("Install the api extra to use PostgreSQL background jobs.") from exc
        self.engine = create_engine(database_url, future=True)

    def initialize(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS background_jobs (
            id TEXT PRIMARY KEY,
            job_type TEXT NOT NULL,
            status TEXT NOT NULL,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            progress_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            result_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            error TEXT NOT NULL DEFAULT '',
            attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            idempotency_key TEXT NOT NULL DEFAULT '',
            worker_id TEXT NOT NULL DEFAULT '',
            lease_until TEXT NOT NULL DEFAULT '',
            available_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT NOT NULL DEFAULT ''
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_background_jobs_idempotency
            ON background_jobs(idempotency_key) WHERE idempotency_key <> '';
        CREATE INDEX IF NOT EXISTS idx_background_jobs_claim
            ON background_jobs(status, available_at, created_at);
        """
        with self.engine.begin() as connection:
            for statement in [part.strip() for part in ddl.split(";") if part.strip()]:
                connection.exec_driver_sql(statement)

    def save(self, job: BackgroundJob) -> BackgroundJob:
        from sqlalchemy import text

        payload = _job_db_payload(job)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO background_jobs (
                        id, job_type, status, payload_json, progress_json, result_json, error,
                        attempts, max_attempts, idempotency_key, worker_id, lease_until,
                        available_at, created_at, updated_at, finished_at
                    ) VALUES (
                        :id, :job_type, :status, CAST(:payload AS JSONB), CAST(:progress AS JSONB),
                        CAST(:result AS JSONB), :error, :attempts, :max_attempts, :idempotency_key,
                        :worker_id, :lease_until, :available_at, :created_at, :updated_at, :finished_at
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        status = EXCLUDED.status, payload_json = EXCLUDED.payload_json,
                        progress_json = EXCLUDED.progress_json, result_json = EXCLUDED.result_json,
                        error = EXCLUDED.error, attempts = EXCLUDED.attempts,
                        worker_id = EXCLUDED.worker_id, lease_until = EXCLUDED.lease_until,
                        available_at = EXCLUDED.available_at, updated_at = EXCLUDED.updated_at,
                        finished_at = EXCLUDED.finished_at
                    """
                ),
                payload,
            )
        return job

    def get(self, job_id: str) -> BackgroundJob:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            row = connection.execute(text("SELECT * FROM background_jobs WHERE id = :id"), {"id": job_id}).mappings().first()
        if not row:
            raise KeyError(f"Background job not found: {job_id}")
        return _job_from_row(row)

    def list(self, *, status: str = "", limit: int = 100) -> List[BackgroundJob]:
        from sqlalchemy import text

        where = "WHERE status = :status" if status else ""
        params: Dict[str, Any] = {"limit": max(1, min(limit, 1000))}
        if status:
            params["status"] = status
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(f"SELECT * FROM background_jobs {where} ORDER BY created_at DESC LIMIT :limit"), params
            ).mappings()
            return [_job_from_row(row) for row in rows]

    def claim(self, worker_id: str, lease_seconds: int = 120) -> Optional[BackgroundJob]:
        from sqlalchemy import text

        now = utc_now()
        lease_until = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        with self.engine.begin() as connection:
            row = connection.execute(
                text(
                    """
                    SELECT * FROM background_jobs
                    WHERE status = 'queued' AND available_at <= :now
                    ORDER BY created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                    """
                ),
                {"now": now},
            ).mappings().first()
            if not row:
                return None
            connection.execute(
                text(
                    """
                    UPDATE background_jobs
                    SET status = 'running', worker_id = :worker_id, attempts = attempts + 1,
                        lease_until = :lease_until, updated_at = :now
                    WHERE id = :id
                    """
                ),
                {"id": row["id"], "worker_id": worker_id, "lease_until": lease_until, "now": now},
            )
        return replace(_job_from_row(row), status="running", worker_id=worker_id, attempts=int(row.get("attempts") or 0) + 1, lease_until=lease_until, updated_at=now)


class JobService:
    def __init__(self, repository: JobRepository):
        self.repository = repository
        self.repository.initialize()

    def enqueue(
        self,
        job_type: str,
        payload: Dict[str, Any],
        *,
        idempotency_key: str = "",
        max_attempts: int = 3,
    ) -> BackgroundJob:
        if idempotency_key:
            for existing in self.repository.list(limit=1000):
                if existing.idempotency_key == idempotency_key and existing.status not in {"failed", "cancelled"}:
                    return existing
        now = utc_now()
        return self.repository.save(
            BackgroundJob(
                id=f"job-{uuid.uuid4().hex}", job_type=job_type, status="queued", payload=dict(payload),
                max_attempts=max(1, max_attempts), idempotency_key=idempotency_key,
                available_at=now, created_at=now, updated_at=now,
            )
        )

    def list(self, *, status: str = "", limit: int = 100) -> List[BackgroundJob]:
        if status and status not in JOB_STATUSES:
            raise JobError(f"Unsupported job status: {status}")
        return self.repository.list(status=status, limit=limit)

    def get(self, job_id: str) -> BackgroundJob:
        return self.repository.get(job_id)

    def cancel(self, job_id: str) -> BackgroundJob:
        job = self.get(job_id)
        if job.status in {"completed", "failed", "cancelled"}:
            return job
        status = "cancelled" if job.status == "queued" else "cancel_requested"
        return self.repository.save(replace(job, status=status, updated_at=utc_now(), finished_at=utc_now() if status == "cancelled" else ""))

    def mark_cancelled(self, job_id: str, result: Optional[Dict[str, Any]] = None) -> BackgroundJob:
        job = self.get(job_id)
        return self.repository.save(
            replace(
                job,
                status="cancelled",
                result=dict(result or job.result),
                worker_id="",
                lease_until="",
                updated_at=utc_now(),
                finished_at=utc_now(),
            )
        )

    def complete(self, job_id: str, result: Dict[str, Any]) -> BackgroundJob:
        job = self.get(job_id)
        return self.repository.save(replace(job, status="completed", result=dict(result), progress={"percent": 100}, error="", updated_at=utc_now(), finished_at=utc_now(), lease_until=""))

    def fail(self, job_id: str, error: str) -> BackgroundJob:
        job = self.get(job_id)
        if job.attempts < job.max_attempts:
            delay = min(2 ** max(job.attempts, 1), 60)
            return self.repository.save(replace(job, status="queued", error=_safe_error(error), worker_id="", lease_until="", available_at=(datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat(), updated_at=utc_now()))
        return self.repository.save(replace(job, status="failed", error=_safe_error(error), updated_at=utc_now(), finished_at=utc_now(), lease_until=""))

    def progress(self, job_id: str, progress: Dict[str, Any]) -> BackgroundJob:
        job = self.get(job_id)
        return self.repository.save(replace(job, progress=dict(progress), updated_at=utc_now()))


class LocalBlobStore:
    def __init__(self, root: str):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, filename: str, content: bytes) -> Dict[str, Any]:
        digest = hashlib.sha256(content).hexdigest()
        safe_name = "".join(char for char in Path(filename).name if char.isalnum() or char in {".", "-", "_"}) or "upload.bin"
        directory = self.root / digest[:2] / digest
        directory.mkdir(parents=True, exist_ok=True)
        path = (directory / safe_name).resolve()
        if self.root not in path.parents:
            raise JobError("Invalid blob path.")
        if not path.exists():
            path.write_bytes(content)
        return {"path": str(path), "sha256": digest, "size": len(content), "filename": safe_name}

    def read(self, path: str) -> bytes:
        resolved = Path(path).resolve()
        if self.root not in resolved.parents or not resolved.is_file():
            raise JobError("Blob path is outside the configured store or no longer exists.")
        return resolved.read_bytes()

    def delete(self, path: str) -> None:
        resolved = Path(path).resolve()
        if self.root not in resolved.parents:
            raise JobError("Blob path is outside the configured store.")
        if resolved.exists():
            resolved.unlink()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_idempotency_key(job_type: str, payload: Dict[str, Any]) -> str:
    stable = json.dumps({"job_type": job_type, "payload": payload}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable.encode("utf-8")).hexdigest()


def _job_db_payload(job: BackgroundJob) -> Dict[str, Any]:
    payload = asdict(job)
    for key in ["payload", "progress", "result"]:
        payload[key] = json.dumps(payload[key])
    return payload


def _job_from_row(row: Any) -> BackgroundJob:
    return BackgroundJob(
        id=row["id"], job_type=row["job_type"], status=row["status"], payload=dict(row.get("payload_json") or {}),
        progress=dict(row.get("progress_json") or {}), result=dict(row.get("result_json") or {}), error=row.get("error") or "",
        attempts=int(row.get("attempts") or 0), max_attempts=int(row.get("max_attempts") or 3),
        idempotency_key=row.get("idempotency_key") or "", worker_id=row.get("worker_id") or "", lease_until=row.get("lease_until") or "",
        available_at=row.get("available_at") or "", created_at=row.get("created_at") or "", updated_at=row.get("updated_at") or "",
        finished_at=row.get("finished_at") or "",
    )


def _safe_error(error: str) -> str:
    value = str(error)
    for key, secret in os.environ.items():
        if key.startswith("ARAGBIZ_MODEL_") and secret:
            value = value.replace(secret, "[REDACTED]")
    return value[:2000]
