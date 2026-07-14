from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from dataclasses import asdict
from typing import Any, Dict

from aragbiz.config import load_config
from aragbiz.factory import (
    build_blob_store,
    build_job_service,
    build_knowledge_service,
    build_model_farm_service,
    build_model_gateway,
)
from aragbiz.jobs import BackgroundJob, JobService
from aragbiz.knowledge import IngestionSummary, KnowledgeService


class KnowledgeJobWorker:
    def __init__(self, jobs: JobService, knowledge: KnowledgeService, blob_store: Any, *, worker_id: str = ""):
        self.jobs = jobs
        self.knowledge = knowledge
        self.blob_store = blob_store
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"

    def run_once(self) -> bool:
        job = self.jobs.repository.claim(self.worker_id)
        if job is None:
            return False
        if job.status == "cancel_requested":
            _log(f"Job {job.id} was cancel-requested; marking cancelled.")
            self.jobs.cancel(job.id)
            return True
        try:
            _log(f"Claimed {job.job_type} job {job.id} (attempt {job.attempts}).")
            result = self._execute(job)
            self.jobs.complete(job.id, result)
            _log(f"Completed job {job.id}.")
        except Exception as exc:
            self.jobs.fail(job.id, str(exc))
            _log(f"Failed job {job.id}: {exc}", stream=sys.stderr)
        return True

    def _execute(self, job: BackgroundJob) -> Dict[str, Any]:
        payload = job.payload
        knowledge_base_id = str(payload.get("knowledge_base_id") or "")
        if job.job_type == "knowledge_upload":
            summaries = []
            blobs = list(payload.get("blobs") or [])
            for index, blob in enumerate(blobs, start=1):
                self.jobs.progress(job.id, {"step": "upload", "current": index, "total": len(blobs), "percent": round(index * 90 / max(len(blobs), 1), 1)})
                content = self.blob_store.read(blob["path"])
                summaries.append(self.knowledge.ingest_uploaded_file(knowledge_base_id, blob["filename"], content))
            return _merge_summaries(summaries)
        if job.job_type == "knowledge_website":
            self.jobs.progress(job.id, {"step": "website", "percent": 20})
            return asdict(self.knowledge.ingest_website(knowledge_base_id, str(payload.get("url") or "")))
        if job.job_type == "knowledge_reindex":
            self.jobs.progress(job.id, {"step": "reindex", "percent": 10})
            return asdict(self.knowledge.reindex(knowledge_base_id))
        raise ValueError(f"Unsupported background job type: {job.job_type}")


def _merge_summaries(summaries: list[IngestionSummary]) -> Dict[str, Any]:
    if not summaries:
        return {"status": "empty", "documents_added": 0, "documents_skipped": 0, "chunks_added": 0}
    return {
        "knowledge_base_id": summaries[-1].knowledge_base_id,
        "source_id": summaries[-1].source_id,
        "status": summaries[-1].status,
        "documents_added": sum(item.documents_added for item in summaries),
        "documents_skipped": sum(item.documents_skipped for item in summaries),
        "chunks_added": summaries[-1].chunks_added,
        "error": next((item.error for item in summaries if item.error), None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run durable aragbiz background jobs.")
    parser.add_argument("--once", action="store_true", help="Claim at most one job and exit.")
    parser.add_argument("--poll-seconds", type=float, default=1.5)
    parser.add_argument("--quiet", action="store_true", help="Suppress idle/startup status messages.")
    args = parser.parse_args()
    config = load_config()
    farm = build_model_farm_service(config)
    gateway = build_model_gateway(config, model_farm_service=farm)
    jobs = build_job_service(config)
    worker = KnowledgeJobWorker(
        jobs,
        build_knowledge_service(config, model_farm_service=farm, model_gateway=gateway),
        build_blob_store(config),
    )
    if not args.quiet:
        _log(
            "Worker started "
            f"(id={worker.worker_id}, backend={config.knowledge_backend}, "
            f"database={'configured' if config.knowledge_database_url else 'not configured'}, once={args.once})."
        )
    while True:
        handled = worker.run_once()
        if args.once:
            if not handled and not args.quiet:
                _log("No queued jobs found. Exiting because --once was set.")
            return
        if not handled:
            if not args.quiet:
                _log(f"No queued jobs found. Waiting {max(args.poll_seconds, 0.1):g}s...")
            time.sleep(max(args.poll_seconds, 0.1))


def _log(message: str, *, stream: Any = sys.stdout) -> None:
    print(f"[aragbiz-worker] {message}", file=stream, flush=True)


if __name__ == "__main__":
    main()
