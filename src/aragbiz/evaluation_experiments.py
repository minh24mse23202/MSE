from __future__ import annotations

import hashlib
import json
import random
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Protocol

from aragbiz.evaluation import EvaluationRunConfig, EvaluationRunRecord, EvaluationService
from aragbiz.evaluation_metrics import (
    DEFAULT_QUALITY_WEIGHTS,
    quality_score,
    validate_quality_weights,
)
from aragbiz.knowledge import KnowledgeService, utc_now


EXPERIMENT_STATUSES = {"queued", "running", "completed", "partial", "failed", "cancelled"}
DATASET_DEFINITIONS = {
    "expertwritten": {
        "name": "WixQA ExpertWritten",
        "path": "data/processed/wixqa_expertwritten_qac.jsonl",
        "count": 200,
    },
    "simulated": {
        "name": "WixQA Simulated",
        "path": "data/processed/wixqa_simulated_qac.jsonl",
        "count": 200,
    },
    "synthetic": {
        "name": "WixQA Synthetic",
        "path": "data/processed/wixqa_synthetic_qac.jsonl",
        "count": 6221,
    },
}


@dataclass(frozen=True)
class EvaluationExperimentRecord:
    id: str
    name: str
    status: str
    knowledge_base_id: str
    knowledge_base_name: str
    configuration_ids: List[str]
    datasets: Dict[str, Optional[int]]
    judge_deployment_id: str
    quality_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_QUALITY_WEIGHTS))
    max_cost_per_case: Optional[float] = None
    max_average_latency_ms: Optional[float] = None
    seed: int = 42
    run_ids: List[str] = field(default_factory=list)
    leaderboard: List[Dict[str, Any]] = field(default_factory=list)
    progress: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_by: str = ""
    created_at: str = ""
    updated_at: str = ""
    finished_at: str = ""


class EvaluationExperimentRepository(Protocol):
    def initialize(self) -> None: ...
    def save(self, experiment: EvaluationExperimentRecord) -> EvaluationExperimentRecord: ...
    def get(self, experiment_id: str) -> EvaluationExperimentRecord: ...
    def list(self) -> List[EvaluationExperimentRecord]: ...
    def delete(self, experiment_id: str) -> None: ...


class JsonEvaluationExperimentRepository:
    def __init__(self, path: str):
        source = Path(path)
        self.path = source.with_name(f"{source.stem}_experiments.json")

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text(json.dumps({"experiments": {}}, indent=2), encoding="utf-8")

    def save(self, experiment: EvaluationExperimentRecord) -> EvaluationExperimentRecord:
        state = self._read()
        state["experiments"][experiment.id] = asdict(experiment)
        self.path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        return experiment

    def get(self, experiment_id: str) -> EvaluationExperimentRecord:
        payload = self._read()["experiments"].get(experiment_id)
        if not payload:
            raise KeyError(f"Evaluation experiment not found: {experiment_id}")
        return _experiment_from_dict(payload)

    def list(self) -> List[EvaluationExperimentRecord]:
        records = [_experiment_from_dict(item) for item in self._read()["experiments"].values()]
        return sorted(records, key=lambda item: item.created_at, reverse=True)

    def delete(self, experiment_id: str) -> None:
        state = self._read()
        if experiment_id not in state["experiments"]:
            raise KeyError(f"Evaluation experiment not found: {experiment_id}")
        state["experiments"].pop(experiment_id)
        self.path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    def _read(self) -> Dict[str, Any]:
        self.initialize()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        payload.setdefault("experiments", {})
        return payload


class PostgresEvaluationExperimentRepository:
    def __init__(self, database_url: str):
        from sqlalchemy import create_engine

        self.engine = create_engine(database_url, future=True)

    def initialize(self) -> None:
        # PostgreSQL schema is managed by Alembic.
        return None

    def save(self, experiment: EvaluationExperimentRecord) -> EvaluationExperimentRecord:
        from sqlalchemy import text

        payload = asdict(experiment)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO evaluation_experiments (
                        id, name, status, knowledge_base_id, knowledge_base_name,
                        configuration_ids_json, datasets_json, judge_deployment_id,
                        quality_weights_json, max_cost_per_case, max_average_latency_ms,
                        seed, run_ids_json, leaderboard_json, progress_json, metadata_json,
                        error, created_by, created_at, updated_at, finished_at
                    ) VALUES (
                        :id, :name, :status, :knowledge_base_id, :knowledge_base_name,
                        CAST(:configuration_ids AS JSONB), CAST(:datasets AS JSONB),
                        :judge_deployment_id, CAST(:quality_weights AS JSONB),
                        :max_cost_per_case, :max_average_latency_ms, :seed,
                        CAST(:run_ids AS JSONB), CAST(:leaderboard AS JSONB),
                        CAST(:progress AS JSONB), CAST(:metadata AS JSONB), :error,
                        :created_by, :created_at, :updated_at, :finished_at
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name, status = EXCLUDED.status,
                        run_ids_json = EXCLUDED.run_ids_json,
                        leaderboard_json = EXCLUDED.leaderboard_json,
                        progress_json = EXCLUDED.progress_json,
                        metadata_json = EXCLUDED.metadata_json, error = EXCLUDED.error,
                        updated_at = EXCLUDED.updated_at, finished_at = EXCLUDED.finished_at
                    """
                ),
                {
                    **payload,
                    "configuration_ids": json.dumps(experiment.configuration_ids),
                    "datasets": json.dumps(experiment.datasets),
                    "quality_weights": json.dumps(experiment.quality_weights),
                    "run_ids": json.dumps(experiment.run_ids),
                    "leaderboard": json.dumps(experiment.leaderboard),
                    "progress": json.dumps(experiment.progress),
                    "metadata": json.dumps(experiment.metadata),
                },
            )
        return experiment

    def get(self, experiment_id: str) -> EvaluationExperimentRecord:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            row = connection.execute(
                text("SELECT * FROM evaluation_experiments WHERE id = :id"),
                {"id": experiment_id},
            ).mappings().first()
        if not row:
            raise KeyError(f"Evaluation experiment not found: {experiment_id}")
        return _experiment_from_row(row)

    def list(self) -> List[EvaluationExperimentRecord]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(
                text("SELECT * FROM evaluation_experiments ORDER BY created_at DESC")
            ).mappings()
            return [_experiment_from_row(row) for row in rows]

    def delete(self, experiment_id: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            result = connection.execute(
                text("DELETE FROM evaluation_experiments WHERE id = :id"),
                {"id": experiment_id},
            )
        if result.rowcount == 0:
            raise KeyError(f"Evaluation experiment not found: {experiment_id}")


class EvaluationExperimentService:
    def __init__(
        self,
        repository: EvaluationExperimentRepository,
        evaluation_service: EvaluationService,
        knowledge_service: Optional[KnowledgeService] = None,
    ):
        self.repository = repository
        self.evaluation_service = evaluation_service
        self.knowledge_service = knowledge_service
        self.repository.initialize()

    def datasets(self, knowledge_base_id: str = "") -> List[Dict[str, Any]]:
        available_article_ids = self._wixqa_article_ids(knowledge_base_id) if knowledge_base_id else None
        result = []
        for key, definition in DATASET_DEFINITIONS.items():
            path = Path(definition["path"])
            payload = {
                "id": key,
                "name": definition["name"],
                "record_count": definition["count"],
                "path": str(path),
                "available": path.is_file(),
                "sha256": _file_sha256(path) if path.is_file() else "",
            }
            if available_article_ids is not None:
                compatible_ids = (
                    _compatible_record_ids(str(path), available_article_ids)
                    if path.is_file()
                    else []
                )
                payload["compatible_record_count"] = len(compatible_ids)
                payload["knowledge_base_document_count"] = len(available_article_ids)
            result.append(payload)
        return result

    def create(
        self,
        *,
        name: str,
        knowledge_base: Any,
        configuration_snapshots: List[Dict[str, Any]],
        datasets: Dict[str, Optional[int]],
        judge_deployment_id: str,
        quality_weights: Optional[Dict[str, float]] = None,
        max_cost_per_case: Optional[float] = None,
        max_average_latency_ms: Optional[float] = None,
        seed: int = 42,
        created_by: str = "",
    ) -> EvaluationExperimentRecord:
        if not configuration_snapshots:
            raise ValueError("Select at least one saved RAG configuration.")
        if not judge_deployment_id:
            raise ValueError("Select an enabled judge deployment.")
        normalized_datasets = _validate_datasets(datasets)
        weights = validate_quality_weights(quality_weights or DEFAULT_QUALITY_WEIGHTS)
        available_article_ids = self._wixqa_article_ids(knowledge_base.id)
        compatible_record_ids = {
            dataset_id: _compatible_record_ids(
                str(DATASET_DEFINITIONS[dataset_id]["path"]),
                available_article_ids,
            )
            for dataset_id in normalized_datasets
        } if available_article_ids is not None else {}
        if available_article_ids is not None:
            if not available_article_ids:
                raise ValueError(
                    "The selected Knowledge Base has no prepared WixQA corpus documents."
                )
            for dataset_id, requested_limit in normalized_datasets.items():
                compatible_count = len(compatible_record_ids[dataset_id])
                if compatible_count == 0:
                    raise ValueError(
                        f"The selected Knowledge Base has no compatible records for {dataset_id}."
                    )
                if requested_limit is not None and requested_limit > compatible_count:
                    raise ValueError(
                        f"{dataset_id} limit cannot exceed the {compatible_count} records "
                        "compatible with the selected Knowledge Base documents."
                    )
            has_non_wixqa_documents = int(knowledge_base.document_count or 0) != len(available_article_ids)
            compatibility = {
                "status": "warning" if has_non_wixqa_documents else "compatible",
                "message": (
                    f"Evaluation is restricted to records supported by "
                    f"{len(available_article_ids):,} imported WixQA documents."
                    + (
                        " Additional non-WixQA documents remain searchable and may affect retrieval."
                        if has_non_wixqa_documents
                        else ""
                    )
                ),
                "wixqa_document_count": len(available_article_ids),
                "total_document_count": int(knowledge_base.document_count or 0),
                "eligible_record_counts": {
                    key: len(value) for key, value in compatible_record_ids.items()
                },
                "source_record_ids_sha256": _stable_ids_sha256(available_article_ids),
            }
        else:
            compatibility = {
                "status": "compatible" if int(knowledge_base.document_count or 0) == 6221 else "warning",
                "message": (
                    "The selected Knowledge Base has 6,221 documents."
                    if int(knowledge_base.document_count or 0) == 6221
                    else "WixQA document compatibility could not be verified."
                ),
            }
        now = utc_now()
        record = EvaluationExperimentRecord(
            id=f"experiment-{uuid.uuid4().hex}",
            name=(name or "WixQA configuration benchmark").strip(),
            status="queued",
            knowledge_base_id=knowledge_base.id,
            knowledge_base_name=knowledge_base.name,
            configuration_ids=[str(item["id"]) for item in configuration_snapshots],
            datasets=normalized_datasets,
            judge_deployment_id=judge_deployment_id,
            quality_weights=weights,
            max_cost_per_case=max_cost_per_case,
            max_average_latency_ms=max_average_latency_ms,
            seed=int(seed),
            metadata={
                "configuration_snapshots": configuration_snapshots,
                "dataset_snapshots": self.datasets(knowledge_base.id),
                "knowledge_base_compatibility": compatibility,
                "knowledge_base_index_version_id": str(
                    knowledge_base.metadata.get("active_index_version_id") or ""
                ),
                "conversation_context_enabled": False,
                "public_web_enabled": False,
                "ranking": "equal-dataset macro average",
            },
            progress={"completed_cells": 0, "total_cells": len(configuration_snapshots) * len(normalized_datasets)},
            created_by=created_by,
            created_at=now,
            updated_at=now,
        )
        return self.repository.save(record)

    def execute(
        self,
        experiment_id: str,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancellation_requested: Optional[Callable[[], bool]] = None,
    ) -> EvaluationExperimentRecord:
        experiment = self.repository.get(experiment_id)
        experiment = self.repository.save(replace(experiment, status="running", updated_at=utc_now()))
        snapshots = list(experiment.metadata.get("configuration_snapshots") or [])
        compatibility = dict(experiment.metadata.get("knowledge_base_compatibility") or {})
        expected_source_hash = str(compatibility.get("source_record_ids_sha256") or "")
        available_article_ids = self._wixqa_article_ids(experiment.knowledge_base_id)
        if expected_source_hash and (
            available_article_ids is None
            or _stable_ids_sha256(available_article_ids) != expected_source_hash
        ):
            error = (
                "The WixQA documents in the selected Knowledge Base changed after this "
                "experiment was created. Create a new evaluation experiment."
            )
            self.repository.save(
                replace(
                    experiment,
                    status="failed",
                    error=error,
                    updated_at=utc_now(),
                    finished_at=utc_now(),
                )
            )
            raise ValueError(error)
        run_ids = list(experiment.run_ids)
        total = len(snapshots) * len(experiment.datasets)
        failures: List[str] = []
        completed = 0
        for snapshot in snapshots:
            for dataset_id, requested_limit in experiment.datasets.items():
                if cancellation_requested and cancellation_requested():
                    return self.repository.save(
                        replace(experiment, status="cancelled", run_ids=run_ids, updated_at=utc_now(), finished_at=utc_now())
                    )
                run_id = _matrix_run_id(experiment.id, str(snapshot["id"]), dataset_id)
                try:
                    try:
                        existing = self.evaluation_service.get_run(run_id)
                        if existing.status == "completed":
                            run = existing
                        else:
                            raise KeyError(run_id)
                    except KeyError:
                        definition = DATASET_DEFINITIONS[dataset_id]
                        record_ids = (
                            _sample_compatible_record_ids(
                                definition["path"],
                                available_article_ids,
                                requested_limit,
                                experiment.seed,
                            )
                            if available_article_ids is not None
                            else _sample_record_ids(
                                definition["path"],
                                requested_limit,
                                experiment.seed,
                            )
                        )
                        metadata = dict(snapshot.get("metadata") or {})
                        retrieval_mode = str(metadata.get("retrieval_mode") or "hybrid")
                        top_k = int(metadata.get("top_k") or 4)
                        run = self.evaluation_service.run(
                            EvaluationRunConfig(
                                name=f"{snapshot.get('name', snapshot['id'])} - {definition['name']}",
                                knowledge_base_id=experiment.knowledge_base_id,
                                chat_configuration_id=str(snapshot["id"]),
                                chat_configuration=snapshot,
                                judge_deployment_id=experiment.judge_deployment_id,
                                retrieval_mode=retrieval_mode,  # type: ignore[arg-type]
                                top_k=top_k,
                                limit=len(record_ids),
                                dataset_path=definition["path"],
                                dataset_name=definition["name"],
                                record_ids=record_ids,
                                run_id=run_id,
                                experiment_id=experiment.id,
                            )
                        )
                    if run_id not in run_ids:
                        run_ids.append(run_id)
                except Exception as exc:
                    failures.append(f"{snapshot['id']} / {dataset_id}: {str(exc)[:500]}")
                completed += 1
                progress = {
                    "completed_cells": completed,
                    "total_cells": total,
                    "current_configuration_id": snapshot["id"],
                    "current_dataset": dataset_id,
                    "percent": round(completed * 100 / max(total, 1), 1),
                }
                experiment = self.repository.save(
                    replace(experiment, run_ids=run_ids, progress=progress, updated_at=utc_now())
                )
                if progress_callback:
                    progress_callback(progress)
        runs = [self.evaluation_service.get_run(run_id) for run_id in run_ids]
        leaderboard = _build_leaderboard(experiment, runs, snapshots)
        final_status = "failed" if failures and not runs else ("partial" if failures else "completed")
        return self.repository.save(
            replace(
                experiment,
                status=final_status,
                run_ids=run_ids,
                leaderboard=leaderboard,
                error="; ".join(failures),
                progress={"completed_cells": completed, "total_cells": total, "percent": 100.0},
                updated_at=utc_now(),
                finished_at=utc_now(),
            )
        )

    def list(self) -> List[EvaluationExperimentRecord]:
        return self.repository.list()

    def get(self, experiment_id: str) -> EvaluationExperimentRecord:
        return self.repository.get(experiment_id)

    def runs(self, experiment_id: str) -> List[EvaluationRunRecord]:
        record = self.get(experiment_id)
        return [self.evaluation_service.get_run(run_id) for run_id in record.run_ids]

    def delete(self, experiment_id: str) -> None:
        record = self.get(experiment_id)
        for run_id in record.run_ids:
            try:
                self.evaluation_service.delete_run(run_id)
            except KeyError:
                pass
        self.repository.delete(experiment_id)

    def mark_cancelled(self, experiment_id: str) -> EvaluationExperimentRecord:
        record = self.get(experiment_id)
        if record.status in {"completed", "failed", "cancelled"}:
            return record
        return self.repository.save(
            replace(record, status="cancelled", updated_at=utc_now(), finished_at=utc_now())
        )

    def _wixqa_article_ids(self, knowledge_base_id: str) -> Optional[set[str]]:
        if self.knowledge_service is None:
            return None
        return set(self.knowledge_service.list_wixqa_source_record_ids(knowledge_base_id))


def _validate_datasets(datasets: Dict[str, Optional[int]]) -> Dict[str, Optional[int]]:
    normalized: Dict[str, Optional[int]] = {}
    for dataset_id, requested_limit in datasets.items():
        if dataset_id not in DATASET_DEFINITIONS:
            raise ValueError(f"Unsupported WixQA dataset: {dataset_id}")
        maximum = int(DATASET_DEFINITIONS[dataset_id]["count"])
        if requested_limit is None or int(requested_limit) == 0:
            normalized[dataset_id] = None
        else:
            limit = int(requested_limit)
            if limit < 1 or limit > maximum:
                raise ValueError(f"{dataset_id} limit must be between 1 and {maximum}, or 0 for all.")
            normalized[dataset_id] = limit
    if not normalized:
        raise ValueError("Select at least one WixQA dataset.")
    return normalized


def _sample_record_ids(path: str, limit: Optional[int], seed: int) -> List[str]:
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            records.append(str(payload.get("id") or ""))
    if limit is None or limit >= len(records):
        return records
    rng = random.Random(f"{seed}:{Path(path).name}")
    selected = list(records)
    rng.shuffle(selected)
    return selected[:limit]


def _compatible_record_ids(path: str, available_article_ids: set[str]) -> List[str]:
    compatible: List[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        article_ids = {
            str(value)
            for value in (payload.get("metadata") or {}).get("article_ids", [])
            if value
        }
        if article_ids and article_ids.issubset(available_article_ids):
            record_id = str(payload.get("id") or "")
            if record_id:
                compatible.append(record_id)
    return compatible


def _sample_compatible_record_ids(
    path: str,
    available_article_ids: set[str],
    limit: Optional[int],
    seed: int,
) -> List[str]:
    records = _compatible_record_ids(path, available_article_ids)
    if limit is None or limit >= len(records):
        return records
    rng = random.Random(f"{seed}:{Path(path).name}:compatible")
    selected = list(records)
    rng.shuffle(selected)
    return selected[:limit]


def _stable_ids_sha256(values: set[str]) -> str:
    serialized = "\n".join(sorted(values))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _matrix_run_id(experiment_id: str, configuration_id: str, dataset_id: str) -> str:
    digest = hashlib.sha256(f"{experiment_id}:{configuration_id}:{dataset_id}".encode("utf-8")).hexdigest()
    return f"eval-{digest[:32]}"


def _build_leaderboard(
    experiment: EvaluationExperimentRecord,
    runs: List[EvaluationRunRecord],
    snapshots: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    snapshot_names = {str(item["id"]): str(item.get("name") or item["id"]) for item in snapshots}
    snapshot_routes = {
        str(item["id"]): str(
            (item.get("metadata") or {}).get("route_strategy")
            or (item.get("metadata") or {}).get("route_mode")
            or "Adaptive"
        )
        for item in snapshots
    }
    entries: List[Dict[str, Any]] = []
    for configuration_id in experiment.configuration_ids:
        config_runs = [run for run in runs if run.chat_configuration_id == configuration_id]
        dataset_scores: Dict[str, float] = {}
        costs = []
        latencies = []
        for run in config_runs:
            wixqa = dict(run.metrics.get("wixqa") or {})
            score = quality_score(wixqa, experiment.quality_weights)
            dataset_scores[run.dataset_name] = score
            costs.append(float(run.metrics.get("average_cost_per_case_usd", 0.0)))
            latencies.append(float(run.metrics.get("average_latency_ms", 0.0)))
        aggregate = mean(dataset_scores.values()) if dataset_scores else 0.0
        average_cost = mean(costs) if costs else 0.0
        average_latency = mean(latencies) if latencies else 0.0
        violations = []
        if experiment.max_cost_per_case is not None and average_cost > experiment.max_cost_per_case:
            violations.append("cost")
        if experiment.max_average_latency_ms is not None and average_latency > experiment.max_average_latency_ms:
            violations.append("latency")
        if len(config_runs) != len(experiment.datasets):
            violations.append(
                f"incomplete results ({len(config_runs)}/{len(experiment.datasets)})"
            )
        entries.append(
            {
                "configuration_id": configuration_id,
                "configuration_name": snapshot_names.get(configuration_id, configuration_id),
                "configuration_route": snapshot_routes.get(configuration_id, "Adaptive"),
                "quality_score": aggregate,
                "dataset_scores": dataset_scores,
                "average_cost_per_case_usd": average_cost,
                "average_latency_ms": average_latency,
                "eligible": not violations,
                "constraint_violations": violations,
                "run_ids": [run.id for run in config_runs],
            }
        )
    entries.sort(
        key=lambda item: (
            not item["eligible"],
            -float(item["quality_score"]),
            float(item["average_cost_per_case_usd"]),
            float(item["average_latency_ms"]),
        )
    )
    for index, entry in enumerate(entries, start=1):
        entry["rank"] = index if entry["eligible"] else None
        entry["winner"] = bool(entry["eligible"] and index == 1)
    return entries


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _experiment_from_dict(payload: Dict[str, Any]) -> EvaluationExperimentRecord:
    return EvaluationExperimentRecord(
        id=payload["id"],
        name=payload.get("name", "WixQA configuration benchmark"),
        status=payload.get("status", "queued"),
        knowledge_base_id=payload.get("knowledge_base_id", ""),
        knowledge_base_name=payload.get("knowledge_base_name", ""),
        configuration_ids=list(payload.get("configuration_ids") or []),
        datasets=dict(payload.get("datasets") or {}),
        judge_deployment_id=payload.get("judge_deployment_id", ""),
        quality_weights=dict(payload.get("quality_weights") or DEFAULT_QUALITY_WEIGHTS),
        max_cost_per_case=payload.get("max_cost_per_case"),
        max_average_latency_ms=payload.get("max_average_latency_ms"),
        seed=int(payload.get("seed", 42)),
        run_ids=list(payload.get("run_ids") or []),
        leaderboard=list(payload.get("leaderboard") or []),
        progress=dict(payload.get("progress") or {}),
        metadata=dict(payload.get("metadata") or {}),
        error=payload.get("error", ""),
        created_by=payload.get("created_by", ""),
        created_at=payload.get("created_at", ""),
        updated_at=payload.get("updated_at", ""),
        finished_at=payload.get("finished_at", ""),
    )


def _experiment_from_row(row: Any) -> EvaluationExperimentRecord:
    return _experiment_from_dict(
        {
            "id": row["id"],
            "name": row["name"],
            "status": row["status"],
            "knowledge_base_id": row["knowledge_base_id"],
            "knowledge_base_name": row["knowledge_base_name"],
            "configuration_ids": row.get("configuration_ids_json") or [],
            "datasets": row.get("datasets_json") or {},
            "judge_deployment_id": row["judge_deployment_id"],
            "quality_weights": row.get("quality_weights_json") or {},
            "max_cost_per_case": row.get("max_cost_per_case"),
            "max_average_latency_ms": row.get("max_average_latency_ms"),
            "seed": row.get("seed", 42),
            "run_ids": row.get("run_ids_json") or [],
            "leaderboard": row.get("leaderboard_json") or [],
            "progress": row.get("progress_json") or {},
            "metadata": row.get("metadata_json") or {},
            "error": row.get("error") or "",
            "created_by": row.get("created_by") or "",
            "created_at": row.get("created_at") or "",
            "updated_at": row.get("updated_at") or "",
            "finished_at": row.get("finished_at") or "",
        }
    )
