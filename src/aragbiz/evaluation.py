from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Protocol

from aragbiz.answering import AdaptiveRAGAnswerService, AnswerOptions
from aragbiz.data import load_qac_jsonl
from aragbiz.knowledge import KnowledgeProcessingError, utc_now
from aragbiz.pipeline import RAGPipeline
from aragbiz.ragxplain import RagxplainError, RagxplainRunner, RagxplainUnavailableError
from aragbiz.schemas import AnswerResult, QACRecord, RetrievedContext, RetrievalMode


@dataclass(frozen=True)
class EvaluationRunRecord:
    id: str
    name: str
    dataset_name: str
    status: str
    knowledge_base_id: str
    knowledge_base_name: str = ""
    chat_configuration_id: Optional[str] = None
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = 4
    limit: int = 20
    compare_baseline: bool = True
    metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    route_distribution: Dict[str, int] = field(default_factory=dict)
    baseline_route_distribution: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    created_at: str = ""
    finished_at: str = ""


@dataclass(frozen=True)
class EvaluationCaseRecord:
    id: str
    run_id: str
    record_id: str
    question: str
    expected_answer: str
    complexity_label: str
    adaptive_answer: str
    static_answer: str = ""
    adaptive_contexts: List[Dict[str, Any]] = field(default_factory=list)
    static_contexts: List[Dict[str, Any]] = field(default_factory=list)
    adaptive_metadata: Dict[str, Any] = field(default_factory=dict)
    static_metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True)
class EvaluationRunConfig:
    name: str = "Adaptive vs Static L2 evaluation"
    knowledge_base_id: str = ""
    chat_configuration_id: Optional[str] = None
    chat_configuration: Dict[str, Any] = field(default_factory=dict)
    judge_deployment_id: str = ""
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = 4
    limit: int = 20
    compare_baseline: bool = True
    run_ragxplain: bool = False


class EvaluationRepository(Protocol):
    def initialize(self) -> None:
        """Initialize evaluation storage."""

    def save_run(self, run: EvaluationRunRecord, cases: List[EvaluationCaseRecord]) -> EvaluationRunRecord:
        """Persist an evaluation run and cases."""

    def list_runs(self) -> List[EvaluationRunRecord]:
        """List evaluation runs."""

    def get_run(self, run_id: str) -> EvaluationRunRecord:
        """Get one evaluation run."""

    def list_cases(self, run_id: str) -> List[EvaluationCaseRecord]:
        """List cases for one evaluation run."""

    def delete_run(self, run_id: str) -> None:
        """Delete an evaluation run and cases."""


class EvaluationService:
    def __init__(
        self,
        repository: EvaluationRepository,
        answer_service: AdaptiveRAGAnswerService,
        dataset_path: str,
        dataset_name: str = "WixQA expert-written",
        ragxplain_runner: Optional[RagxplainRunner] = None,
    ):
        self.repository = repository
        self.answer_service = answer_service
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.ragxplain_runner = ragxplain_runner

    def run(self, config: EvaluationRunConfig) -> EvaluationRunRecord:
        self.repository.initialize()
        if not config.knowledge_base_id:
            raise ValueError("Select a knowledge base before running evaluation.")
        top_k = max(1, min(int(config.top_k), 50))
        limit = max(0, min(int(config.limit), 100))
        records = load_qac_jsonl(self.dataset_path)
        if limit:
            records = records[:limit]
        else:
            records = []
        knowledge_base = self.answer_service.knowledge_service.get_knowledge_base(config.knowledge_base_id)
        evaluation_chat_configuration = _evaluation_chat_configuration(config.chat_configuration)

        adaptive_results: List[AnswerResult] = []
        static_results: List[AnswerResult] = []
        cases: List[EvaluationCaseRecord] = []
        now = utc_now()
        run_id = f"eval-{uuid.uuid4().hex}"

        for index, record in enumerate(records, start=1):
            adaptive_result = self.answer_service.answer(
                record.question,
                AnswerOptions(
                    mode="adaptive",
                    knowledge_base_id=config.knowledge_base_id,
                    retrieval_mode=config.retrieval_mode,
                    top_k=top_k,
                    chat_configuration=evaluation_chat_configuration,
                    conversation_history=[],
                ),
            )
            adaptive_results.append(adaptive_result)
            static_result: Optional[AnswerResult] = None
            if config.compare_baseline:
                static_result = self.answer_service.answer(
                    record.question,
                    AnswerOptions(
                        mode="simple_rag",
                        knowledge_base_id=config.knowledge_base_id,
                        retrieval_mode=config.retrieval_mode,
                        top_k=top_k,
                        chat_configuration=evaluation_chat_configuration,
                        conversation_history=[],
                    ),
                )
                static_results.append(static_result)
            cases.append(
                EvaluationCaseRecord(
                    id=f"case-{uuid.uuid4().hex}",
                    run_id=run_id,
                    record_id=record.id or str(index),
                    question=record.question,
                    expected_answer=record.answer,
                    complexity_label=record.complexity_label,
                    adaptive_answer=adaptive_result.answer,
                    static_answer=static_result.answer if static_result else "",
                    adaptive_contexts=[_context_to_dict(context) for context in adaptive_result.contexts],
                    static_contexts=[_context_to_dict(context) for context in static_result.contexts] if static_result else [],
                    adaptive_metadata=adaptive_result.metadata,
                    static_metadata=static_result.metadata if static_result else {},
                    metrics=_case_metrics(record, adaptive_result, static_result),
                    created_at=now,
                )
            )

        metrics = evaluate_predictions(records, adaptive_results)
        metrics.update(_runtime_metrics(adaptive_results))
        baseline_metrics: Dict[str, Any] = {}
        if config.compare_baseline:
            baseline_metrics = evaluate_predictions(records, static_results)
            baseline_metrics.update(_runtime_metrics(static_results))
        run_name = (config.name or "Adaptive vs Static L2 evaluation").strip()
        ragxplain_metadata: Dict[str, Any] = {
            "status": "not_requested",
            "output_dir": None,
            "overall_insights_path": None,
            "judge": self.ragxplain_runner.judge if self.ragxplain_runner else None,
            "error": None,
        }
        if config.run_ragxplain:
            if self.ragxplain_runner is None:
                ragxplain_metadata.update(
                    status="failed",
                    error="RAGXplain is not configured for this environment.",
                )
            else:
                try:
                    ragxplain_metadata = self.ragxplain_runner.run(
                        run_id,
                        run_name,
                        cases,
                        {
                            "run_id": run_id,
                            "name": run_name,
                            "dataset_name": self.dataset_name,
                            "knowledge_base_id": knowledge_base.id,
                            "knowledge_base_name": knowledge_base.name,
                            "chat_configuration_id": config.chat_configuration_id,
                            "chat_configuration": config.chat_configuration,
                            "retrieval_mode": config.retrieval_mode,
                            "top_k": top_k,
                            "compare_baseline": config.compare_baseline,
                            "route_distribution": _route_distribution(adaptive_results),
                        },
                    )
                except RagxplainError as exc:
                    ragxplain_metadata = {
                        "status": "failed",
                        "output_dir": str(self.ragxplain_runner.output_dir(run_id)),
                        "overall_insights_path": None,
                        "judge": self.ragxplain_runner.judge,
                        "error": str(exc),
                    }

        run = EvaluationRunRecord(
            id=run_id,
            name=run_name,
            dataset_name=self.dataset_name,
            status="completed",
            knowledge_base_id=knowledge_base.id,
            knowledge_base_name=knowledge_base.name,
            chat_configuration_id=config.chat_configuration_id,
            retrieval_mode=config.retrieval_mode,
            top_k=top_k,
            limit=limit,
            compare_baseline=config.compare_baseline,
            metrics=metrics,
            baseline_metrics=baseline_metrics,
            route_distribution=_route_distribution(adaptive_results),
            baseline_route_distribution=_route_distribution(static_results),
            metadata={
                "dataset_path": self.dataset_path,
                "record_count": len(records),
                "chat_configuration": config.chat_configuration,
                "judge_deployment_id": config.judge_deployment_id,
                "static_baseline": "simple_rag" if config.compare_baseline else "disabled",
                "conversation_context": {
                    "enabled": False,
                    "isolation": "Each evaluation record is executed as an independent query.",
                },
                "ragxplain": ragxplain_metadata,
            },
            error=None,
            created_at=now,
            finished_at=utc_now(),
        )
        return self.repository.save_run(run, cases)

    def list_runs(self) -> List[EvaluationRunRecord]:
        self.repository.initialize()
        return self.repository.list_runs()

    def get_run(self, run_id: str) -> EvaluationRunRecord:
        self.repository.initialize()
        return self.repository.get_run(run_id)

    def list_cases(self, run_id: str) -> List[EvaluationCaseRecord]:
        self.repository.initialize()
        return self.repository.list_cases(run_id)

    def delete_run(self, run_id: str) -> None:
        self.repository.initialize()
        run = self.repository.get_run(run_id)
        ragxplain = run.metadata.get("ragxplain", {})
        output_dir = ragxplain.get("output_dir") if isinstance(ragxplain, dict) else None
        if output_dir and self.ragxplain_runner is not None:
            self.ragxplain_runner.delete_artifacts(run_id, str(output_dir))
        self.repository.delete_run(run_id)

    def get_ragxplain_insights(self, run_id: str) -> Dict[str, Any]:
        self.repository.initialize()
        run = self.repository.get_run(run_id)
        ragxplain = run.metadata.get("ragxplain", {})
        if not isinstance(ragxplain, dict) or ragxplain.get("status") != "completed":
            detail = ragxplain.get("error") if isinstance(ragxplain, dict) else None
            raise RagxplainUnavailableError(detail or "RAGXplain insights are not available for this run.")
        artifact_path = ragxplain.get("overall_insights_path")
        if not artifact_path or self.ragxplain_runner is None:
            raise RagxplainUnavailableError("RAGXplain insights are not available for this run.")
        return self.ragxplain_runner.load_overall_insights(str(artifact_path))

    def ragxplain_viewer_path(self) -> Path:
        if self.ragxplain_runner is None:
            raise RagxplainUnavailableError("RAGXplain viewer is not configured for this environment.")
        return self.ragxplain_runner.viewer_path()


def _evaluation_chat_configuration(configuration: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = dict(configuration or {})
    metadata = dict(snapshot.get("metadata") or {})
    snapshot["conversation_awareness_enabled"] = False
    metadata["conversation_awareness_enabled"] = False
    snapshot["metadata"] = metadata
    return snapshot


class Evaluator:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def evaluate(self, dataset: Iterable[QACRecord]) -> Dict[str, float]:
        records = list(dataset)
        results = [self.pipeline.answer(record.question) for record in records]
        return evaluate_predictions(records, results)


def evaluate_predictions(records: List[QACRecord], results: List[AnswerResult]) -> Dict[str, float]:
    if len(records) != len(results):
        raise ValueError("records and results must have the same length")
    if not records:
        return {
            "routing_accuracy": 0.0,
            "context_relevance": 0.0,
            "faithfulness_proxy": 0.0,
            "answer_overlap": 0.0,
            "average_latency_ms": 0.0,
        }
    return {
        "routing_accuracy": mean(_routing_match(record, result) for record, result in zip(records, results)),
        "context_relevance": mean(_context_relevance(record, result) for record, result in zip(records, results)),
        "faithfulness_proxy": mean(_faithfulness_proxy(result) for result in results),
        "answer_overlap": mean(_answer_overlap(record.answer, result.answer) for record, result in zip(records, results)),
        "average_latency_ms": mean(float(result.metadata.get("latency_ms", 0.0)) for result in results),
    }


class JsonEvaluationRepository:
    def __init__(self, path: str):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write(_empty_evaluation_state())
            return
        state = self._read()
        changed = False
        for key, value in _empty_evaluation_state().items():
            if key not in state:
                state[key] = value
                changed = True
        if changed:
            self._write(state)

    def save_run(self, run: EvaluationRunRecord, cases: List[EvaluationCaseRecord]) -> EvaluationRunRecord:
        state = self._read()
        state["evaluation_runs"][run.id] = _run_to_dict(run)
        for case in cases:
            state["evaluation_cases"][case.id] = _case_to_dict(case)
        self._write(state)
        return run

    def list_runs(self) -> List[EvaluationRunRecord]:
        state = self._read()
        runs = [_run_from_dict(payload) for payload in state["evaluation_runs"].values()]
        runs.sort(key=lambda run: run.created_at, reverse=True)
        return runs

    def get_run(self, run_id: str) -> EvaluationRunRecord:
        state = self._read()
        payload = state["evaluation_runs"].get(run_id)
        if not payload:
            raise KeyError(f"Evaluation run not found: {run_id}")
        return _run_from_dict(payload)

    def list_cases(self, run_id: str) -> List[EvaluationCaseRecord]:
        self.get_run(run_id)
        state = self._read()
        cases = [
            _case_from_dict(payload)
            for payload in state["evaluation_cases"].values()
            if payload.get("run_id") == run_id
        ]
        cases.sort(key=lambda case: case.created_at)
        return cases

    def delete_run(self, run_id: str) -> None:
        state = self._read()
        if run_id not in state["evaluation_runs"]:
            raise KeyError(f"Evaluation run not found: {run_id}")
        state["evaluation_runs"].pop(run_id, None)
        for case_id in [case_id for case_id, case in state["evaluation_cases"].items() if case.get("run_id") == run_id]:
            state["evaluation_cases"].pop(case_id, None)
        self._write(state)

    def _read(self) -> Dict[str, Dict[str, Any]]:
        self.initialize() if not self.path.exists() else None
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, state: Dict[str, Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


class PostgresEvaluationRepository:
    def __init__(self, database_url: str):
        try:
            from sqlalchemy import create_engine  # type: ignore
        except ImportError as exc:
            raise KnowledgeProcessingError("Install the api extra to use PostgreSQL evaluation storage.") from exc
        self.engine = create_engine(database_url, future=True)

    def initialize(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            status TEXT NOT NULL,
            knowledge_base_id TEXT NOT NULL,
            knowledge_base_name TEXT NOT NULL DEFAULT '',
            chat_configuration_id TEXT,
            retrieval_mode TEXT NOT NULL DEFAULT 'hybrid',
            top_k INTEGER NOT NULL DEFAULT 4,
            run_limit INTEGER NOT NULL DEFAULT 20,
            compare_baseline BOOLEAN NOT NULL DEFAULT TRUE,
            metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            baseline_metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            route_distribution_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            baseline_route_distribution_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            error TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evaluation_cases (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
            record_id TEXT NOT NULL,
            question TEXT NOT NULL,
            expected_answer TEXT NOT NULL,
            complexity_label TEXT NOT NULL,
            adaptive_answer TEXT NOT NULL,
            static_answer TEXT NOT NULL DEFAULT '',
            adaptive_contexts_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            static_contexts_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            adaptive_metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            static_metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_evaluation_runs_created_at ON evaluation_runs(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_evaluation_cases_run_id ON evaluation_cases(run_id, created_at);
        """
        with self.engine.begin() as connection:
            for statement in [part.strip() for part in ddl.split(";") if part.strip()]:
                connection.exec_driver_sql(statement)

    def save_run(self, run: EvaluationRunRecord, cases: List[EvaluationCaseRecord]) -> EvaluationRunRecord:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO evaluation_runs
                        (id, name, dataset_name, status, knowledge_base_id, knowledge_base_name, chat_configuration_id, retrieval_mode, top_k, run_limit, compare_baseline, metrics_json, baseline_metrics_json, route_distribution_json, baseline_route_distribution_json, metadata_json, error, created_at, finished_at)
                    VALUES
                        (:id, :name, :dataset_name, :status, :knowledge_base_id, :knowledge_base_name, :chat_configuration_id, :retrieval_mode, :top_k, :limit, :compare_baseline, CAST(:metrics AS JSONB), CAST(:baseline_metrics AS JSONB), CAST(:route_distribution AS JSONB), CAST(:baseline_route_distribution AS JSONB), CAST(:metadata AS JSONB), :error, :created_at, :finished_at)
                    """
                ),
                _run_sql_params(run),
            )
            for case in cases:
                connection.execute(
                    text(
                        """
                        INSERT INTO evaluation_cases
                            (id, run_id, record_id, question, expected_answer, complexity_label, adaptive_answer, static_answer, adaptive_contexts_json, static_contexts_json, adaptive_metadata_json, static_metadata_json, metrics_json, created_at)
                        VALUES
                            (:id, :run_id, :record_id, :question, :expected_answer, :complexity_label, :adaptive_answer, :static_answer, CAST(:adaptive_contexts AS JSONB), CAST(:static_contexts AS JSONB), CAST(:adaptive_metadata AS JSONB), CAST(:static_metadata AS JSONB), CAST(:metrics AS JSONB), :created_at)
                        """
                    ),
                    _case_sql_params(case),
                )
        return run

    def list_runs(self) -> List[EvaluationRunRecord]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(text("SELECT * FROM evaluation_runs ORDER BY created_at DESC")).mappings()
            return [_run_from_row(row) for row in rows]

    def get_run(self, run_id: str) -> EvaluationRunRecord:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            row = connection.execute(text("SELECT * FROM evaluation_runs WHERE id = :id"), {"id": run_id}).mappings().first()
        if not row:
            raise KeyError(f"Evaluation run not found: {run_id}")
        return _run_from_row(row)

    def list_cases(self, run_id: str) -> List[EvaluationCaseRecord]:
        from sqlalchemy import text

        self.get_run(run_id)
        with self.engine.begin() as connection:
            rows = connection.execute(
                text("SELECT * FROM evaluation_cases WHERE run_id = :id ORDER BY created_at"),
                {"id": run_id},
            ).mappings()
            return [_case_from_row(row) for row in rows]

    def delete_run(self, run_id: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            result = connection.execute(text("DELETE FROM evaluation_runs WHERE id = :id"), {"id": run_id})
            if result.rowcount == 0:
                raise KeyError(f"Evaluation run not found: {run_id}")


def _case_metrics(record: QACRecord, adaptive_result: AnswerResult, static_result: Optional[AnswerResult]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "adaptive": {
            "routing_match": _routing_match(record, adaptive_result),
            "context_relevance": _context_relevance(record, adaptive_result),
            "faithfulness_proxy": _faithfulness_proxy(adaptive_result),
            "answer_overlap": _answer_overlap(record.answer, adaptive_result.answer),
            "latency_ms": float(adaptive_result.metadata.get("latency_ms", 0.0)),
            "retrieved_contexts": len(adaptive_result.contexts),
        }
    }
    if static_result is not None:
        metrics["static_l2"] = {
            "routing_match": _routing_match(record, static_result),
            "context_relevance": _context_relevance(record, static_result),
            "faithfulness_proxy": _faithfulness_proxy(static_result),
            "answer_overlap": _answer_overlap(record.answer, static_result.answer),
            "latency_ms": float(static_result.metadata.get("latency_ms", 0.0)),
            "retrieved_contexts": len(static_result.contexts),
        }
    return metrics


def _runtime_metrics(results: List[AnswerResult]) -> Dict[str, float]:
    if not results:
        return {
            "average_retrieved_contexts": 0.0,
            "average_input_chars": 0.0,
            "average_output_chars": 0.0,
            "runtime_proxy_units": 0.0,
        }
    input_chars = [float(result.metadata.get("input_chars", 0.0)) for result in results]
    output_chars = [float(result.metadata.get("output_chars", 0.0)) for result in results]
    return {
        "average_retrieved_contexts": mean(len(result.contexts) for result in results),
        "average_input_chars": mean(input_chars),
        "average_output_chars": mean(output_chars),
        "runtime_proxy_units": sum(input_chars) / 1000.0 + sum(output_chars) / 1000.0,
    }


def _route_distribution(results: List[AnswerResult]) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    for result in results:
        route = str(result.metadata.get("route_level") or "unknown")
        distribution[route] = distribution.get(route, 0) + 1
    return distribution


def _routing_match(record: QACRecord, result: AnswerResult) -> float:
    return float(record.complexity_label == result.metadata.get("complexity_label"))


def _context_relevance(record: QACRecord, result: AnswerResult) -> float:
    retrieved_ids = set()
    for context in result.contexts:
        retrieved_ids.add(context.document.id)
        retrieved_ids.add(str(context.document.metadata.get("document_id", "")))
        retrieved_ids.add(str(context.document.metadata.get("source_id", "")))
        retrieved_ids.add(str(context.document.metadata.get("chunk_id", "")))
    retrieved_ids.discard("")
    article_ids = {str(value) for value in record.metadata.get("article_ids", [])}
    if article_ids:
        return float(bool(article_ids & retrieved_ids))
    return float(record.id in retrieved_ids)


def _faithfulness_proxy(result: AnswerResult) -> float:
    context_text = " ".join(context.document.text for context in result.contexts).lower()
    answer_terms = set(_tokens(result.answer))
    if not answer_terms:
        return 0.0
    supported = {term for term in answer_terms if term in context_text}
    return len(supported) / len(answer_terms)


def _answer_overlap(expected: str, actual: str) -> float:
    expected_terms = set(_tokens(expected))
    actual_terms = set(_tokens(actual))
    if not expected_terms:
        return 0.0
    return len(expected_terms & actual_terms) / len(expected_terms)


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def _context_to_dict(context: RetrievedContext) -> Dict[str, Any]:
    return {
        "id": context.document.id,
        "score": context.score,
        "rank": context.rank,
        "mode": context.mode,
        "text": context.document.text,
        "metadata": context.document.metadata,
    }


def _empty_evaluation_state() -> Dict[str, Dict[str, Any]]:
    return {"evaluation_runs": {}, "evaluation_cases": {}}


def _run_to_dict(record: EvaluationRunRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "name": record.name,
        "dataset_name": record.dataset_name,
        "status": record.status,
        "knowledge_base_id": record.knowledge_base_id,
        "knowledge_base_name": record.knowledge_base_name,
        "chat_configuration_id": record.chat_configuration_id,
        "retrieval_mode": record.retrieval_mode,
        "top_k": record.top_k,
        "limit": record.limit,
        "compare_baseline": record.compare_baseline,
        "metrics": record.metrics,
        "baseline_metrics": record.baseline_metrics,
        "route_distribution": record.route_distribution,
        "baseline_route_distribution": record.baseline_route_distribution,
        "metadata": record.metadata,
        "error": record.error,
        "created_at": record.created_at,
        "finished_at": record.finished_at,
    }


def _run_from_dict(payload: Dict[str, Any]) -> EvaluationRunRecord:
    return EvaluationRunRecord(
        id=payload["id"],
        name=payload.get("name", "Evaluation run"),
        dataset_name=payload.get("dataset_name", "WixQA expert-written"),
        status=payload.get("status", "completed"),
        knowledge_base_id=payload.get("knowledge_base_id", ""),
        knowledge_base_name=payload.get("knowledge_base_name", ""),
        chat_configuration_id=payload.get("chat_configuration_id"),
        retrieval_mode=payload.get("retrieval_mode", "hybrid"),
        top_k=int(payload.get("top_k", 4)),
        limit=int(payload.get("limit", 20)),
        compare_baseline=bool(payload.get("compare_baseline", True)),
        metrics=dict(payload.get("metrics", {})),
        baseline_metrics=dict(payload.get("baseline_metrics", {})),
        route_distribution=dict(payload.get("route_distribution", {})),
        baseline_route_distribution=dict(payload.get("baseline_route_distribution", {})),
        metadata=dict(payload.get("metadata", {})),
        error=payload.get("error"),
        created_at=payload.get("created_at", ""),
        finished_at=payload.get("finished_at", ""),
    )


def _case_to_dict(record: EvaluationCaseRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "run_id": record.run_id,
        "record_id": record.record_id,
        "question": record.question,
        "expected_answer": record.expected_answer,
        "complexity_label": record.complexity_label,
        "adaptive_answer": record.adaptive_answer,
        "static_answer": record.static_answer,
        "adaptive_contexts": record.adaptive_contexts,
        "static_contexts": record.static_contexts,
        "adaptive_metadata": record.adaptive_metadata,
        "static_metadata": record.static_metadata,
        "metrics": record.metrics,
        "created_at": record.created_at,
    }


def _case_from_dict(payload: Dict[str, Any]) -> EvaluationCaseRecord:
    return EvaluationCaseRecord(
        id=payload["id"],
        run_id=payload["run_id"],
        record_id=payload.get("record_id", ""),
        question=payload.get("question", ""),
        expected_answer=payload.get("expected_answer", ""),
        complexity_label=payload.get("complexity_label", ""),
        adaptive_answer=payload.get("adaptive_answer", ""),
        static_answer=payload.get("static_answer", ""),
        adaptive_contexts=list(payload.get("adaptive_contexts", [])),
        static_contexts=list(payload.get("static_contexts", [])),
        adaptive_metadata=dict(payload.get("adaptive_metadata", {})),
        static_metadata=dict(payload.get("static_metadata", {})),
        metrics=dict(payload.get("metrics", {})),
        created_at=payload.get("created_at", ""),
    )


def _run_sql_params(run: EvaluationRunRecord) -> Dict[str, Any]:
    payload = _run_to_dict(run)
    return {
        **payload,
        "metrics": json.dumps(run.metrics),
        "baseline_metrics": json.dumps(run.baseline_metrics),
        "route_distribution": json.dumps(run.route_distribution),
        "baseline_route_distribution": json.dumps(run.baseline_route_distribution),
        "metadata": json.dumps(run.metadata),
    }


def _case_sql_params(case: EvaluationCaseRecord) -> Dict[str, Any]:
    payload = _case_to_dict(case)
    return {
        **payload,
        "adaptive_contexts": json.dumps(case.adaptive_contexts),
        "static_contexts": json.dumps(case.static_contexts),
        "adaptive_metadata": json.dumps(case.adaptive_metadata),
        "static_metadata": json.dumps(case.static_metadata),
        "metrics": json.dumps(case.metrics),
    }


def _json_field(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value


def _run_from_row(row: Any) -> EvaluationRunRecord:
    return EvaluationRunRecord(
        id=row["id"],
        name=row["name"],
        dataset_name=row["dataset_name"],
        status=row["status"],
        knowledge_base_id=row["knowledge_base_id"],
        knowledge_base_name=row.get("knowledge_base_name", ""),
        chat_configuration_id=row.get("chat_configuration_id"),
        retrieval_mode=row.get("retrieval_mode", "hybrid"),
        top_k=int(row.get("top_k", 4)),
        limit=int(row.get("run_limit", 20)),
        compare_baseline=bool(row.get("compare_baseline", True)),
        metrics=dict(_json_field(row.get("metrics_json"), {})),
        baseline_metrics=dict(_json_field(row.get("baseline_metrics_json"), {})),
        route_distribution=dict(_json_field(row.get("route_distribution_json"), {})),
        baseline_route_distribution=dict(_json_field(row.get("baseline_route_distribution_json"), {})),
        metadata=dict(_json_field(row.get("metadata_json"), {})),
        error=row.get("error"),
        created_at=row.get("created_at", ""),
        finished_at=row.get("finished_at", ""),
    )


def _case_from_row(row: Any) -> EvaluationCaseRecord:
    return EvaluationCaseRecord(
        id=row["id"],
        run_id=row["run_id"],
        record_id=row.get("record_id", ""),
        question=row.get("question", ""),
        expected_answer=row.get("expected_answer", ""),
        complexity_label=row.get("complexity_label", ""),
        adaptive_answer=row.get("adaptive_answer", ""),
        static_answer=row.get("static_answer", ""),
        adaptive_contexts=list(_json_field(row.get("adaptive_contexts_json"), [])),
        static_contexts=list(_json_field(row.get("static_contexts_json"), [])),
        adaptive_metadata=dict(_json_field(row.get("adaptive_metadata_json"), {})),
        static_metadata=dict(_json_field(row.get("static_metadata_json"), {})),
        metrics=dict(_json_field(row.get("metrics_json"), {})),
        created_at=row.get("created_at", ""),
    )
