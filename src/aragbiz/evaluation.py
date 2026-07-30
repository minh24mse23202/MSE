from __future__ import annotations

import json
import hashlib
import math
import re
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Protocol

from aragbiz.answering import AdaptiveRAGAnswerService, AnswerOptions
from aragbiz.data import load_qac_jsonl
from aragbiz.knowledge import KnowledgeProcessingError, utc_now
from aragbiz.evaluation_metrics import aggregate_wixqa_metrics, deterministic_case_metrics
from aragbiz.model_farm import ModelCallContext, ModelFarmError, ModelGateway, ModelUsageEvent
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
    # Legacy fields remain readable so existing persisted runs are not lost.
    compare_baseline: bool = False
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
    name: str = "RAG configuration evaluation"
    knowledge_base_id: str = ""
    chat_configuration_id: Optional[str] = None
    chat_configuration: Dict[str, Any] = field(default_factory=dict)
    judge_deployment_id: str = ""
    retrieval_mode: RetrievalMode = "hybrid"
    top_k: int = 4
    limit: int = 20
    dataset_path: str = ""
    dataset_name: str = ""
    record_ids: List[str] = field(default_factory=list)
    run_id: str = ""
    experiment_id: str = ""


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
        model_gateway: Optional[ModelGateway] = None,
    ):
        self.repository = repository
        self.answer_service = answer_service
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.ragxplain_runner = ragxplain_runner
        self.model_gateway = model_gateway

    def run(self, config: EvaluationRunConfig) -> EvaluationRunRecord:
        self.repository.initialize()
        if not config.knowledge_base_id:
            raise ValueError("Select a knowledge base before running evaluation.")
        top_k = max(1, min(int(config.top_k), 50))
        dataset_path = config.dataset_path or self.dataset_path
        dataset_name = config.dataset_name or self.dataset_name
        max_limit = 10000 if config.dataset_path else 100
        limit = max(0, min(int(config.limit), max_limit))
        records = load_qac_jsonl(dataset_path)
        if config.record_ids:
            selected_ids = set(config.record_ids)
            records = [record for record in records if record.id in selected_ids]
            order = {record_id: index for index, record_id in enumerate(config.record_ids)}
            records.sort(key=lambda record: order.get(record.id, len(order)))
        elif limit:
            records = records[:limit]
        else:
            records = []
        knowledge_base = self.answer_service.knowledge_service.get_knowledge_base(config.knowledge_base_id)
        evaluation_chat_configuration = _evaluation_chat_configuration(config.chat_configuration)
        answer_mode = _configured_answer_mode(evaluation_chat_configuration)

        results: List[AnswerResult] = []
        cases: List[EvaluationCaseRecord] = []
        now = utc_now()
        run_id = config.run_id or f"eval-{uuid.uuid4().hex}"

        for index, record in enumerate(records, start=1):
            result = self.answer_service.answer(
                record.question,
                AnswerOptions(
                    mode=answer_mode,
                    knowledge_base_id=config.knowledge_base_id,
                    retrieval_mode=config.retrieval_mode,
                    top_k=top_k,
                    chat_configuration=evaluation_chat_configuration,
                    evaluation_run_id=run_id,
                    chat_configuration_id=config.chat_configuration_id or "",
                    conversation_history=[],
                ),
            )
            results.append(result)
            case_metrics = _case_metrics(record, result)
            if config.judge_deployment_id:
                case_metrics["result"]["wixqa"].update(
                    self._judge_wixqa_case(
                        run_id,
                        record,
                        result,
                        config.judge_deployment_id,
                        external_processing_allowed=_kb_external_processing_allowed(knowledge_base.metadata),
                    )
                )
            case_id = "case-" + hashlib.sha256(f"{run_id}:{record.id or index}".encode("utf-8")).hexdigest()[:32]
            cases.append(
                EvaluationCaseRecord(
                    id=case_id,
                    run_id=run_id,
                    record_id=record.id or str(index),
                    question=record.question,
                    expected_answer=record.answer,
                    complexity_label=record.complexity_label,
                    adaptive_answer=result.answer,
                    adaptive_contexts=[_context_to_dict(context) for context in result.contexts],
                    adaptive_metadata=result.metadata,
                    metrics=case_metrics,
                    created_at=now,
                )
            )

        metrics = evaluate_predictions(records, results)
        metrics.update(_runtime_metrics(results))
        metrics.update(self._model_usage_metrics(run_id, len(results)))
        wixqa_metrics = aggregate_wixqa_metrics([case.metrics.get("result", {}) for case in cases])
        metrics["wixqa"] = wixqa_metrics
        run_name = (config.name or "RAG configuration evaluation").strip()
        ragxplain_metadata: Dict[str, Any] = {
            "status": "not_requested",
            "output_dir": None,
            "overall_insights_path": None,
            "judge": config.judge_deployment_id or None,
            "error": None,
        }

        run = EvaluationRunRecord(
            id=run_id,
            name=run_name,
            dataset_name=dataset_name,
            status="completed",
            knowledge_base_id=knowledge_base.id,
            knowledge_base_name=knowledge_base.name,
            chat_configuration_id=config.chat_configuration_id,
            retrieval_mode=config.retrieval_mode,
            top_k=top_k,
            limit=limit,
            compare_baseline=False,
            metrics=metrics,
            route_distribution=_route_distribution(results),
            metadata={
                "dataset_path": dataset_path,
                "record_count": len(records),
                "experiment_id": config.experiment_id,
                "chat_configuration": evaluation_chat_configuration,
                "judge_deployment_id": config.judge_deployment_id,
                "evaluation_mode": answer_mode,
                "comparison_model": "configuration_matrix",
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

    def _judge_wixqa_case(
        self,
        run_id: str,
        record: QACRecord,
        result: AnswerResult,
        deployment_id: str,
        *,
        external_processing_allowed: bool,
    ) -> Dict[str, Any]:
        if self.model_gateway is None:
            raise ValueError("A Model Gateway is required for WixQA judge metrics.")
        context_text = "\n\n".join(context.document.text for context in result.contexts)
        common_context = ModelCallContext(
            purpose="evaluation_wixqa_metric",
            request_id=f"{run_id}:{record.id}",
            knowledge_base_id=str(result.metadata.get("knowledge_base_id") or ""),
            evaluation_run_id=run_id,
            chat_configuration_id=str(result.metadata.get("chat_configuration_id") or ""),
        )
        recall = self._judge_score(
            deployment_id,
            "context_recall",
            (
                "Evaluate how completely the retrieved context contains the essential information "
                "needed to produce the reference answer. Ignore irrelevant extra context. Return "
                'strict JSON: {"score": number from 0 to 1, "explanation": "reason in 20 words or fewer"}.'
            ),
            {
                "question": record.question,
                "reference_answer": record.answer,
                "retrieved_context": context_text,
            },
            common_context,
            external_processing_allowed,
        )
        factuality = self._judge_score(
            deployment_id,
            "factuality",
            (
                "Evaluate factual alignment between the candidate and reference answer. Score how "
                "accurately the candidate preserves the essential facts without contradictions. "
                'Return strict JSON: {"score": number from 0 to 1, "explanation": "reason in 20 words or fewer"}.'
            ),
            {
                "question": record.question,
                "reference_answer": record.answer,
                "candidate_answer": result.answer,
            },
            common_context,
            external_processing_allowed,
        )
        return {
            "context_recall": recall["score"],
            "context_recall_explanation": recall["explanation"],
            "context_recall_usage_event_id": recall["usage_event_id"],
            "context_recall_judge_attempts": recall["attempts"],
            "context_recall_structured_output_recovered": recall["structured_output_recovered"],
            "factuality": factuality["score"],
            "factuality_explanation": factuality["explanation"],
            "factuality_usage_event_id": factuality["usage_event_id"],
            "factuality_judge_attempts": factuality["attempts"],
            "factuality_structured_output_recovered": factuality["structured_output_recovered"],
            "judge_deployment_id": deployment_id,
        }

    def _judge_score(
        self,
        deployment_id: str,
        metric: str,
        system_prompt: str,
        payload: Dict[str, str],
        context: ModelCallContext,
        external_processing_allowed: bool,
    ) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        call_context = replace(
            context,
            purpose=f"evaluation_wixqa_{metric}",
            request_id=f"{context.request_id}:{metric}",
        )
        response_format = _judge_response_format(metric)
        parameters = {
            "temperature": 0,
            "max_tokens": 800,
            "response_format": response_format,
            **_judge_provider_parameters(self.model_gateway, deployment_id),
        }
        responses = []
        recovered = False
        for attempt in range(2):
            attempt_messages = list(messages)
            if attempt:
                attempt_messages.extend(
                    [
                        {"role": "assistant", "content": responses[-1].text},
                        {
                            "role": "user",
                            "content": (
                                "The previous response was not valid JSON. Return only one complete "
                                "JSON object matching the required schema. Use no more than 12 words "
                                "for explanation. Do not use Markdown."
                            ),
                        },
                    ]
                )
            try:
                response = self.model_gateway.generate_sync(
                    attempt_messages,
                    deployment_id,
                    parameters=parameters,
                    context=replace(
                        call_context,
                        request_id=f"{call_context.request_id}:attempt-{attempt + 1}",
                    ),
                    external_processing_allowed=external_processing_allowed,
                    capability="judge",
                )
            except ModelFarmError as exc:
                raise ValueError(f"WixQA {metric} judge failed: {exc}") from exc
            responses.append(response)
            try:
                parsed = _parse_judge_payload(response.text)
                break
            except ValueError:
                if attempt:
                    partial_result = next(
                        (
                            (candidate, item)
                            for item in reversed(responses)
                            for candidate in [_parse_partial_judge_payload(item.text)]
                            if candidate is not None
                        ),
                        None,
                    )
                    if partial_result is None:
                        raise ValueError(
                            f"WixQA {metric} judge returned invalid structured output after one retry: "
                            f"{response.text[:300]}"
                        )
                    parsed, response = partial_result
                    recovered = True
        return {
            "score": parsed["score"],
            "explanation": parsed["explanation"],
            "usage_event_id": str(response.metadata.get("usage_event_id") or ""),
            "attempts": len(responses),
            "structured_output_recovered": recovered,
        }

    def list_runs(self) -> List[EvaluationRunRecord]:
        self.repository.initialize()
        return [self._with_model_usage(run) for run in self.repository.list_runs()]

    def get_run(self, run_id: str) -> EvaluationRunRecord:
        self.repository.initialize()
        return self._with_model_usage(self.repository.get_run(run_id))

    def _model_usage_metrics(self, run_id: str, case_count: int) -> Dict[str, Any]:
        if self.model_gateway is None:
            return {}
        model_farm_service = getattr(self.model_gateway, "service", None)
        if model_farm_service is None or not hasattr(model_farm_service, "list_usage"):
            return {}
        events = model_farm_service.list_usage(
            evaluation_run_id=run_id,
            limit=50000,
        )
        evaluation_events = [
            event
            for event in events
            if not event.purpose.startswith("ragxplain_")
        ]
        return _model_usage_metrics(evaluation_events, case_count)

    def _with_model_usage(self, run: EvaluationRunRecord) -> EvaluationRunRecord:
        case_count = int(run.metadata.get("record_count") or run.limit or 0)
        usage_metrics = self._model_usage_metrics(run.id, case_count)
        if not usage_metrics:
            return run
        return replace(run, metrics={**run.metrics, **usage_metrics})

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

    def run_ragxplain(
        self,
        run_id: str,
        *,
        limit: int = 100,
        seed: int = 42,
        judge_deployment_id: str = "",
    ) -> EvaluationRunRecord:
        if self.ragxplain_runner is None or self.model_gateway is None:
            raise RagxplainUnavailableError("RAGXplain Model Farm integration is not configured.")
        run = self.get_run(run_id)
        cases = self.list_cases(run_id)
        resolved_judge_deployment_id = str(
            judge_deployment_id
            or run.metadata.get("judge_deployment_id")
            or ""
        )
        if not resolved_judge_deployment_id:
            raise RagxplainUnavailableError("This evaluation run has no registered judge deployment.")
        selected_cases = _stratified_ragxplain_cases(cases, max(1, min(int(limit), len(cases))), seed)
        running_metadata = {
            **run.metadata,
            "ragxplain": {
                "status": "running",
                "output_dir": str(self.ragxplain_runner.output_dir(run.id)),
                "overall_insights_path": None,
                "judge": resolved_judge_deployment_id,
                "case_count": len(selected_cases),
                "random_state": seed,
                "error": None,
            },
        }
        run = self.repository.save_run(replace(run, metadata=running_metadata), cases)
        knowledge_base = self.answer_service.knowledge_service.get_knowledge_base(run.knowledge_base_id)
        try:
            ragxplain = self.ragxplain_runner.run_with_model_gateway(
                run.id,
                run.name,
                selected_cases,
                {
                    "run_id": run.id,
                    "name": run.name,
                    "dataset_name": run.dataset_name,
                    "knowledge_base_id": run.knowledge_base_id,
                    "knowledge_base_name": run.knowledge_base_name,
                    "chat_configuration_id": run.chat_configuration_id,
                    "chat_configuration": run.metadata.get("chat_configuration", {}),
                    "retrieval_mode": run.retrieval_mode,
                    "top_k": run.top_k,
                    "route_distribution": run.route_distribution,
                    "wixqa_metrics": run.metrics.get("wixqa", {}),
                },
                model_gateway=self.model_gateway,
                judge_deployment_id=resolved_judge_deployment_id,
                knowledge_base_id=run.knowledge_base_id,
                external_processing_allowed=_kb_external_processing_allowed(knowledge_base.metadata),
                random_state=seed,
            )
        except RagxplainError as exc:
            ragxplain = {
                "status": "failed",
                "output_dir": str(self.ragxplain_runner.output_dir(run.id)),
                "overall_insights_path": None,
                "judge": resolved_judge_deployment_id,
                "case_count": len(selected_cases),
                "random_state": seed,
                "error": str(exc),
            }
        metadata = {**run.metadata, "ragxplain": ragxplain}
        updated = replace(run, metadata=metadata)
        return self.repository.save_run(updated, cases)

    def queue_ragxplain(
        self,
        run_id: str,
        *,
        limit: int = 100,
        seed: int = 42,
        judge_deployment_id: str = "",
    ) -> EvaluationRunRecord:
        if self.ragxplain_runner is None or self.model_gateway is None:
            raise RagxplainUnavailableError("RAGXplain Model Farm integration is not configured.")
        run = self.get_run(run_id)
        cases = self.list_cases(run_id)
        resolved_judge_deployment_id = str(
            judge_deployment_id
            or run.metadata.get("judge_deployment_id")
            or ""
        )
        if not resolved_judge_deployment_id:
            raise RagxplainUnavailableError("This evaluation run has no registered judge deployment.")
        previous = dict(run.metadata.get("ragxplain") or {})
        selected_count = min(max(1, int(limit)), len(cases))
        queued = {
            "status": "queued",
            "output_dir": str(self.ragxplain_runner.output_dir(run.id)),
            "overall_insights_path": None,
            "judge": resolved_judge_deployment_id,
            "case_count": selected_count,
            "random_state": int(seed),
            "error": None,
            "replaces_legacy_artifact": bool(
                previous.get("status") == "completed"
                and previous.get("judge") != resolved_judge_deployment_id
            ),
        }
        return self.repository.save_run(
            replace(run, metadata={**run.metadata, "ragxplain": queued}),
            cases,
        )


def _evaluation_chat_configuration(configuration: Dict[str, Any]) -> Dict[str, Any]:
    snapshot = dict(configuration or {})
    metadata = dict(snapshot.get("metadata") or {})
    snapshot["conversation_awareness_enabled"] = False
    snapshot["agent_public_web_enabled"] = False
    metadata["conversation_awareness_enabled"] = False
    metadata["agent_public_web_enabled"] = False
    snapshot["metadata"] = metadata
    return snapshot


def _configured_answer_mode(configuration: Dict[str, Any]) -> str:
    metadata = configuration.get("metadata") if isinstance(configuration, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    configured = str(
        metadata.get("route_mode")
        or configuration.get("route_mode")
        or metadata.get("route_strategy")
        or configuration.get("route_strategy")
        or "adaptive"
    ).strip().lower()
    normalized = configured.replace("-", "_").replace(" ", "_")
    aliases = {
        "adaptive": "adaptive",
        "l1": "direct",
        "l1_direct": "direct",
        "l1_direct_generation": "direct",
        "direct": "direct",
        "l2": "simple_rag",
        "l2_simple_rag": "simple_rag",
        "simple_rag": "simple_rag",
        "l3": "complex_rag",
        "l3_complex_rag": "complex_rag",
        "complex_rag": "complex_rag",
        "l4": "advanced_rag",
        "l4_advanced_rag": "advanced_rag",
        "advanced_rag": "advanced_rag",
    }
    return aliases.get(normalized, "adaptive")


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
        # PostgreSQL schema is managed by Alembic.
        return None

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
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name, dataset_name = EXCLUDED.dataset_name,
                        status = EXCLUDED.status, knowledge_base_id = EXCLUDED.knowledge_base_id,
                        knowledge_base_name = EXCLUDED.knowledge_base_name,
                        chat_configuration_id = EXCLUDED.chat_configuration_id,
                        retrieval_mode = EXCLUDED.retrieval_mode, top_k = EXCLUDED.top_k,
                        run_limit = EXCLUDED.run_limit, compare_baseline = EXCLUDED.compare_baseline,
                        metrics_json = EXCLUDED.metrics_json,
                        baseline_metrics_json = EXCLUDED.baseline_metrics_json,
                        route_distribution_json = EXCLUDED.route_distribution_json,
                        baseline_route_distribution_json = EXCLUDED.baseline_route_distribution_json,
                        metadata_json = EXCLUDED.metadata_json, error = EXCLUDED.error,
                        finished_at = EXCLUDED.finished_at
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
                        ON CONFLICT (id) DO UPDATE SET
                            question = EXCLUDED.question,
                            expected_answer = EXCLUDED.expected_answer,
                            complexity_label = EXCLUDED.complexity_label,
                            adaptive_answer = EXCLUDED.adaptive_answer,
                            static_answer = EXCLUDED.static_answer,
                            adaptive_contexts_json = EXCLUDED.adaptive_contexts_json,
                            static_contexts_json = EXCLUDED.static_contexts_json,
                            adaptive_metadata_json = EXCLUDED.adaptive_metadata_json,
                            static_metadata_json = EXCLUDED.static_metadata_json,
                            metrics_json = EXCLUDED.metrics_json
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


def _case_metrics(record: QACRecord, result: AnswerResult) -> Dict[str, Any]:
    wixqa = deterministic_case_metrics(record.answer, result.answer)
    wixqa.update(
        {
            "reference_answer": record.answer,
            "candidate_answer": result.answer,
            "retrieval": _retrieval_diagnostics(record, result),
        }
    )
    result_metrics = {
            "routing_match": _routing_match(record, result),
            "context_relevance": _context_relevance(record, result),
            "faithfulness_proxy": _faithfulness_proxy(result),
            "answer_overlap": _answer_overlap(record.answer, result.answer),
            "latency_ms": float(result.metadata.get("latency_ms", 0.0)),
            "retrieved_contexts": len(result.contexts),
            "wixqa": wixqa,
    }
    # Keep the adaptive alias while old stored artifacts and the sibling
    # RAGXplain schema are migrated to the route-neutral result key.
    return {"result": result_metrics, "adaptive": result_metrics}


def _runtime_metrics(results: List[AnswerResult]) -> Dict[str, float]:
    if not results:
        return {
            "average_retrieved_contexts": 0.0,
            "average_input_chars": 0.0,
            "average_output_chars": 0.0,
            "runtime_proxy_units": 0.0,
            "average_cost_per_case_usd": 0.0,
            "total_estimated_cost_usd": 0.0,
            "total_input_tokens": 0.0,
            "total_output_tokens": 0.0,
        }
    input_chars = [float(result.metadata.get("input_chars", 0.0)) for result in results]
    output_chars = [float(result.metadata.get("output_chars", 0.0)) for result in results]
    usage = [_result_usage(result) for result in results]
    total_cost = sum(item["estimated_cost_usd"] for item in usage)
    return {
        "average_retrieved_contexts": mean(len(result.contexts) for result in results),
        "average_input_chars": mean(input_chars),
        "average_output_chars": mean(output_chars),
        "runtime_proxy_units": sum(input_chars) / 1000.0 + sum(output_chars) / 1000.0,
        "average_cost_per_case_usd": total_cost / len(results),
        "total_estimated_cost_usd": total_cost,
        "total_input_tokens": sum(item["input_tokens"] for item in usage),
        "total_output_tokens": sum(item["output_tokens"] for item in usage),
    }


def _result_usage(result: AnswerResult) -> Dict[str, float]:
    metadata = result.metadata
    generator = metadata.get("generator_metadata")
    if not isinstance(generator, dict):
        generator = next(
            (
                step.get("metadata")
                for step in reversed(list(metadata.get("trace_steps") or []))
                if isinstance(step, dict)
                and step.get("step") == "Generator execution"
                and isinstance(step.get("metadata"), dict)
            ),
            None,
        )
    if not isinstance(generator, dict):
        generator = metadata.get("actual_generator")
    if not isinstance(generator, dict):
        generator = {}
    return {
        "input_tokens": float(generator.get("input_tokens", metadata.get("input_tokens", 0.0)) or 0.0),
        "output_tokens": float(generator.get("output_tokens", metadata.get("output_tokens", 0.0)) or 0.0),
        "estimated_cost_usd": float(
            generator.get("estimated_cost_usd", metadata.get("estimated_cost_usd", 0.0)) or 0.0
        ),
    }


def _model_usage_metrics(
    events: List[ModelUsageEvent],
    case_count: int,
) -> Dict[str, Any]:
    if not events:
        return {}
    total_cost = sum(float(event.estimated_cost_usd or 0.0) for event in events)
    total_input_tokens = sum(int(event.input_tokens or 0) for event in events)
    total_output_tokens = sum(int(event.output_tokens or 0) for event in events)
    purpose_costs: Dict[str, float] = {}
    for event in events:
        purpose_costs[event.purpose] = (
            purpose_costs.get(event.purpose, 0.0)
            + float(event.estimated_cost_usd or 0.0)
        )
    divisor = max(int(case_count), 1)
    return {
        "average_cost_per_case_usd": total_cost / divisor,
        "total_estimated_cost_usd": total_cost,
        "total_input_tokens": float(total_input_tokens),
        "total_output_tokens": float(total_output_tokens),
        "model_usage_call_count": len(events),
        "model_usage_cost_by_purpose": {
            key: round(value, 10)
            for key, value in sorted(purpose_costs.items())
        },
        "cost_scope": "answer_pipeline_and_wixqa_judges",
        "ragxplain_cost_included": False,
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


def _retrieval_diagnostics(record: QACRecord, result: AnswerResult) -> Dict[str, Any]:
    gold_ids = {str(value) for value in record.metadata.get("article_ids", []) if value}
    if not gold_ids:
        return {"available": False, "reason": "No gold article IDs are available."}
    relevant_ranks: List[int] = []
    matched_ids: set[str] = set()
    for rank, context in enumerate(result.contexts, start=1):
        metadata = context.document.metadata
        candidate_ids = {
            str(context.document.id),
            str(metadata.get("id") or ""),
            str(metadata.get("article_id") or ""),
            str(metadata.get("document_id") or ""),
            str(metadata.get("source_id") or ""),
            str(metadata.get("content_hash") or ""),
            str(metadata.get("source_record_id") or ""),
        }
        matches = gold_ids & {value for value in candidate_ids if value}
        if matches:
            relevant_ranks.append(rank)
            matched_ids.update(matches)
    if not result.contexts:
        precision = 0.0
    else:
        precision = len(relevant_ranks) / len(result.contexts)
    recall = len(matched_ids) / len(gold_ids)
    reciprocal_rank = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks)
    ideal_count = min(len(gold_ids), len(result.contexts))
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return {
        "available": True,
        "gold_article_count": len(gold_ids),
        "matched_article_count": len(matched_ids),
        "precision_at_k": precision,
        "recall_at_k": recall,
        "hit_at_k": float(bool(relevant_ranks)),
        "mrr": reciprocal_rank,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
        "relevant_ranks": relevant_ranks,
    }


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


def _parse_judge_payload(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge output is not valid JSON: {raw[:300]}") from exc
    if not isinstance(payload, dict):
        raise ValueError("Judge output must be a JSON object.")
    try:
        score = float(payload.get("score"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Judge output must contain a numeric score.") from exc
    if score > 1 and score <= 5:
        score /= 5
    if score < 0 or score > 1:
        raise ValueError("Judge score must be between 0 and 1.")
    return {
        "score": score,
        "explanation": str(payload.get("explanation") or "").strip(),
    }


def _parse_partial_judge_payload(raw: str) -> Optional[Dict[str, Any]]:
    text = (raw or "").strip()
    score_match = re.search(
        r"""["']?score["']?\s*:\s*(-?(?:\d+(?:\.\d*)?|\.\d+))""",
        text,
        flags=re.IGNORECASE,
    )
    if not score_match:
        return None
    try:
        score = float(score_match.group(1))
    except ValueError:
        return None
    if score > 1 and score <= 5:
        score /= 5
    if score < 0 or score > 1:
        return None
    explanation_match = re.search(
        r"""["']?explanation["']?\s*:\s*["'](.*)""",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    explanation = ""
    if explanation_match:
        explanation = explanation_match.group(1)
        explanation = re.split(r"""["']\s*[,}]""", explanation, maxsplit=1)[0]
        explanation = explanation.rstrip("\"'` \r\n\t")
    return {
        "score": score,
        "explanation": explanation[:500].strip() or "Recovered from truncated judge output.",
    }


def _judge_provider_parameters(model_gateway: ModelGateway, deployment_id: str) -> Dict[str, Any]:
    service = getattr(model_gateway, "service", None)
    if service is None:
        return {}
    try:
        deployment = service.get_deployment(deployment_id)
    except (KeyError, ModelFarmError):
        return {}
    provider = str(getattr(deployment, "provider", "") or "").lower()
    model = str(getattr(deployment, "model", "") or "").lower()
    if provider == "gemini" and model.startswith("gemini-2.5"):
        return {"reasoning_effort": "disable"}
    return {}


def _judge_response_format(metric: str) -> Dict[str, Any]:
    safe_metric = re.sub(r"[^a-zA-Z0-9_-]+", "_", metric).strip("_")[:40] or "metric"
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"wixqa_{safe_metric}",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                    "explanation": {
                        "type": "string",
                        "description": "A concise reason using no more than 20 words.",
                    },
                },
                "required": ["score", "explanation"],
                "additionalProperties": False,
            },
        },
    }


def _kb_external_processing_allowed(metadata: Dict[str, Any]) -> bool:
    configuration = metadata.get("configuration") if isinstance(metadata, dict) else {}
    if not isinstance(configuration, dict):
        configuration = {}
    return bool(
        configuration.get(
            "external_processing_allowed",
            metadata.get("external_processing_allowed", False) if isinstance(metadata, dict) else False,
        )
    )


def _stratified_ragxplain_cases(
    cases: List[EvaluationCaseRecord],
    limit: int,
    seed: int,
) -> List[EvaluationCaseRecord]:
    if limit >= len(cases):
        return list(cases)
    import random

    def case_score(case: EvaluationCaseRecord) -> float:
        wixqa = dict((case.metrics.get("adaptive") or {}).get("wixqa") or {})
        values = [
            float(wixqa.get(name, 0.0))
            for name in ("token_f1", "bleu", "rouge_1", "rouge_2")
        ]
        return sum(values) / len(values)

    ordered = sorted(cases, key=case_score)
    thirds = [
        ordered[: max(1, len(ordered) // 3)],
        ordered[max(1, len(ordered) // 3) : max(2, 2 * len(ordered) // 3)],
        ordered[max(2, 2 * len(ordered) // 3) :],
    ]
    rng = random.Random(seed)
    for bucket in thirds:
        rng.shuffle(bucket)
    selected: List[EvaluationCaseRecord] = []
    index = 0
    while len(selected) < limit and any(index < len(bucket) for bucket in thirds):
        for bucket in thirds:
            if index < len(bucket) and len(selected) < limit:
                selected.append(bucket[index])
        index += 1
    return selected


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
        compare_baseline=bool(payload.get("compare_baseline", False)),
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
        compare_baseline=bool(row.get("compare_baseline", False)),
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
