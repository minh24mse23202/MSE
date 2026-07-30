from __future__ import annotations

import asyncio
import copy
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Sequence

from aragbiz.model_farm import ModelCallContext, ModelGateway


class RagxplainError(RuntimeError):
    """Raised when RAGXplain cannot execute or produces invalid artifacts."""


class RagxplainUnavailableError(RagxplainError):
    """Raised when a persisted evaluation has no usable RAGXplain insights."""


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]


class RagxplainRunner:
    REQUIRED_ARTIFACTS = ("results.csv", "metrics_insights.json", "overall_insights.json")

    def __init__(
        self,
        root: str,
        results_root: str,
        judge: str,
        *,
        timeout_seconds: int = 1800,
        process_runner: ProcessRunner = subprocess.run,
    ):
        self.root = Path(root).expanduser().resolve()
        self.results_root = Path(results_root).expanduser().resolve()
        self.judge = judge.strip()
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.process_runner = process_runner

    def run(
        self,
        run_id: str,
        run_name: str,
        cases: Sequence[Any],
        run_config: Mapping[str, Any],
    ) -> Dict[str, Any]:
        self._validate_installation()
        output_dir = self.output_dir(run_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        for artifact_name in self.REQUIRED_ARTIFACTS:
            artifact_path = output_dir / artifact_name
            if artifact_path.is_file():
                artifact_path.unlink()
        input_path = output_dir / "input.jsonl"
        config_path = output_dir / "rag_run_config.json"
        self.export_cases(cases, input_path)
        config_path.write_text(json.dumps(dict(run_config), indent=2, sort_keys=True), encoding="utf-8")

        command = [
            sys.executable,
            "-m",
            "ragxplain.cli",
            "run",
            "--input",
            str(input_path),
            "--out",
            str(output_dir),
            "--experiment-name",
            run_name or run_id,
            "--judge",
            self.judge,
            "--rag-run-config",
            str(config_path),
            "--stages",
            "all",
            "--no-progress",
        ]
        environment = os.environ.copy()
        existing_python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(self.root), existing_python_path) if part
        )
        try:
            completed = self.process_runner(
                command,
                cwd=str(self.root),
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RagxplainError(
                f"RAGXplain judge timed out after {self.timeout_seconds} seconds."
            ) from exc
        except OSError as exc:
            raise RagxplainError(f"Unable to start RAGXplain: {exc}") from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "Unknown RAGXplain error").strip()
            raise RagxplainError(f"RAGXplain exited with code {completed.returncode}: {detail[-4000:]}")

        artifacts = {name: output_dir / name for name in self.REQUIRED_ARTIFACTS}
        missing = [name for name, path in artifacts.items() if not path.is_file()]
        if missing:
            raise RagxplainError("RAGXplain did not create required artifacts: " + ", ".join(missing))
        insights = self._read_and_validate_insights(artifacts["overall_insights.json"])
        _attach_configuration_targets(insights)
        artifacts["overall_insights.json"].write_text(
            json.dumps(insights, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "output_dir": str(output_dir),
            "input_path": str(input_path),
            "results_path": str(artifacts["results.csv"]),
            "metrics_insights_path": str(artifacts["metrics_insights.json"]),
            "overall_insights_path": str(artifacts["overall_insights.json"]),
            "judge": self.judge,
            "error": None,
        }

    def run_with_model_gateway(
        self,
        run_id: str,
        run_name: str,
        cases: Sequence[Any],
        run_config: Mapping[str, Any],
        *,
        model_gateway: ModelGateway,
        judge_deployment_id: str,
        knowledge_base_id: str,
        external_processing_allowed: bool,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        self._validate_installation()
        output_dir = self.output_dir(run_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        for artifact_name in self.REQUIRED_ARTIFACTS:
            artifact_path = output_dir / artifact_name
            if artifact_path.is_file():
                artifact_path.unlink()
        input_path = output_dir / "input.jsonl"
        config_path = output_dir / "rag_run_config.json"
        self.export_cases(cases, input_path)
        config_path.write_text(json.dumps(dict(run_config), indent=2, sort_keys=True), encoding="utf-8")
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        try:
            from ragxplain.core.experiment_runner import ExperimentRunner, RunnerConfig
            from ragxplain.core.metrics_calculator import MetricsCalculator
        except ImportError as exc:
            raise RagxplainError(
                "RAGXplain Python dependencies are unavailable. Install the sibling project in this environment."
            ) from exc

        judge = _ModelGatewayJudge(
            model_gateway,
            judge_deployment_id,
            run_id,
            knowledge_base_id,
            external_processing_allowed,
            str(run_config.get("chat_configuration_id") or ""),
        )
        calculator = MetricsCalculator(
            judge=judge,
            metrics_config_path=str(self.root / "ragxplain" / "configs" / "metrics.yaml"),
            prompts_base_dir=str(self.root / "ragxplain"),
            max_concurrency=4,
            show_progress=False,
            request_timeout_s=300,
        )
        runner = ExperimentRunner(
            RunnerConfig(
                input_path=str(input_path),
                out_dir=str(output_dir),
                experiment_name=run_name or run_id,
                random_state=int(random_state),
                rag_run_config=dict(run_config),
            ),
            calculator,
            show_progress=False,
            metrics_parallelism=1,
        )
        try:
            asyncio.run(runner.run_analysis())
        except Exception as exc:
            raise RagxplainError(f"RAGXplain Model Farm judge failed: {exc}") from exc
        artifacts = {name: output_dir / name for name in self.REQUIRED_ARTIFACTS}
        missing = [name for name, path in artifacts.items() if not path.is_file()]
        if missing:
            raise RagxplainError("RAGXplain did not create required artifacts: " + ", ".join(missing))
        insights = self._read_and_validate_insights(artifacts["overall_insights.json"])
        semantic_metrics = _validate_semantic_metric_artifacts(
            artifacts["results.csv"],
            insights,
            runner.ragxplain_metrics,
        )
        overall_analysis_status = "completed"
        if not _is_complete_overall_analysis(insights.get("analysis")):
            original_error = _overall_analysis_error(insights.get("analysis"))
            try:
                repaired = asyncio.run(
                    _repair_overall_analysis(
                        judge,
                        insights,
                        self.root / "ragxplain" / "prompts" / "overall_insight_calculator_schema.json",
                    )
                )
                insights["analysis"] = repaired
                overall_analysis_status = "repaired"
                insights["aragbiz_overall_analysis"] = {
                    "status": "repaired",
                    "original_error": original_error,
                }
            except Exception as exc:
                insights["analysis"] = _fallback_overall_analysis(insights, original_error)
                overall_analysis_status = "metric_fallback"
                insights["aragbiz_overall_analysis"] = {
                    "status": "metric_fallback",
                    "original_error": original_error,
                    "repair_error": str(exc),
                }
        _attach_configuration_targets(insights)
        artifacts["overall_insights.json"].write_text(
            json.dumps(insights, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return {
            "status": "completed",
            "output_dir": str(output_dir),
            "input_path": str(input_path),
            "results_path": str(artifacts["results.csv"]),
            "metrics_insights_path": str(artifacts["metrics_insights.json"]),
            "overall_insights_path": str(artifacts["overall_insights.json"]),
            "judge": judge_deployment_id,
            "case_count": len(cases),
            "random_state": int(random_state),
            "semantic_metrics": semantic_metrics,
            "overall_analysis_status": overall_analysis_status,
            "error": None,
        }

    def export_cases(self, cases: Sequence[Any], path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(self._case_payload(case), ensure_ascii=False) for case in cases]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return path

    def load_overall_insights(self, artifact_path: str) -> Dict[str, Any]:
        path = Path(artifact_path).expanduser().resolve()
        self._validate_artifact_path(path, "overall_insights.json")
        if not path.is_file():
            raise RagxplainUnavailableError("RAGXplain overall insights artifact is missing.")
        insights = self._read_and_validate_insights(path)
        if _is_complete_overall_analysis(insights.get("analysis")):
            return insights
        original_error = _overall_analysis_error(insights.get("analysis"))
        viewer_payload = copy.deepcopy(insights)
        viewer_payload["analysis"] = _fallback_overall_analysis(viewer_payload, original_error)
        viewer_payload["aragbiz_overall_analysis"] = {
            "status": "metric_fallback",
            "original_error": original_error,
            "artifact_unchanged": True,
        }
        _attach_configuration_targets(viewer_payload)
        return viewer_payload

    def delete_artifacts(self, run_id: str, output_dir: str = "") -> None:
        expected = self.output_dir(run_id)
        candidate = Path(output_dir).expanduser().resolve() if output_dir else expected
        if candidate != expected:
            raise RagxplainError("Refusing to delete RAGXplain artifacts outside the run output directory.")
        self._validate_within_results_root(candidate)
        if candidate.exists():
            shutil.rmtree(candidate)
        run_directory = candidate.parent
        if run_directory.exists() and not any(run_directory.iterdir()):
            run_directory.rmdir()

    def output_dir(self, run_id: str) -> Path:
        if not run_id or any(separator in run_id for separator in ("/", "\\")):
            raise RagxplainError("Invalid evaluation run ID for RAGXplain output.")
        candidate = (self.results_root / run_id / "ragxplain").resolve()
        self._validate_within_results_root(candidate)
        return candidate

    def viewer_path(self) -> Path:
        path = (self.root / "viewer" / "insights-viewer.html").resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise RagxplainError("Invalid RAGXplain viewer path.") from exc
        if not path.is_file():
            raise RagxplainUnavailableError(f"RAGXplain viewer not found: {path}")
        return path

    def _validate_installation(self) -> None:
        if not self.root.is_dir() or not (self.root / "ragxplain" / "cli.py").is_file():
            raise RagxplainError(
                f"RAGXplain project not found at '{self.root}'. Set ARAGBIZ_RAGXPLAIN_ROOT."
            )
        if not self.judge:
            raise RagxplainError("Configure a RAGXplain judge with ARAGBIZ_RAGXPLAIN_JUDGE.")

    def _validate_within_results_root(self, path: Path) -> None:
        try:
            path.relative_to(self.results_root)
        except ValueError as exc:
            raise RagxplainError("RAGXplain artifact path is outside the configured results directory.") from exc

    def _validate_artifact_path(self, path: Path, expected_name: str) -> None:
        self._validate_within_results_root(path)
        if path.name != expected_name or path.parent.name != "ragxplain":
            raise RagxplainUnavailableError("Invalid RAGXplain artifact path.")

    @staticmethod
    def _read_and_validate_insights(path: Path) -> Dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RagxplainError(f"Invalid RAGXplain overall insights artifact: {exc}") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("analysis"), dict):
            raise RagxplainError("RAGXplain overall_insights.json must contain an analysis object.")
        return payload

    @staticmethod
    def _case_payload(case: Any) -> Dict[str, Any]:
        contexts = getattr(case, "adaptive_contexts", []) or []
        context_sections: List[str] = []
        for index, context in enumerate(contexts, start=1):
            if isinstance(context, dict):
                metadata = context.get("metadata", {})
                title = metadata.get("title") or metadata.get("document_title") or context.get("id", f"source-{index}")
                text = context.get("text", "")
            else:
                title = f"source-{index}"
                text = str(context)
            context_sections.append(f"[{index}] {title}\n{text}".strip())
        metrics = getattr(case, "metrics", {}) or {}
        adaptive_metrics = (
            metrics.get("result") or metrics.get("adaptive") or {}
            if isinstance(metrics, dict)
            else {}
        )
        return {
            "question": str(getattr(case, "question", "") or ""),
            "candidate_answer": str(getattr(case, "adaptive_answer", "") or ""),
            "contexts": "\n\n".join(context_sections),
            "gt_answer": str(getattr(case, "expected_answer", "") or ""),
            "routing_match": adaptive_metrics.get("routing_match", 0.0),
            "context_relevance_proxy": adaptive_metrics.get("context_relevance", 0.0),
            "faithfulness_proxy": adaptive_metrics.get("faithfulness_proxy", 0.0),
            "answer_overlap": adaptive_metrics.get("answer_overlap", 0.0),
            "token_f1": (adaptive_metrics.get("wixqa") or {}).get("token_f1", 0.0),
            "bleu": (adaptive_metrics.get("wixqa") or {}).get("bleu", 0.0),
            "rouge_1": (adaptive_metrics.get("wixqa") or {}).get("rouge_1", 0.0),
            "rouge_2": (adaptive_metrics.get("wixqa") or {}).get("rouge_2", 0.0),
        }


class _ModelGatewayJudge:
    def __init__(
        self,
        gateway: ModelGateway,
        deployment_id: str,
        evaluation_run_id: str,
        knowledge_base_id: str,
        external_processing_allowed: bool,
        chat_configuration_id: str = "",
    ):
        self.gateway = gateway
        self.deployment_id = deployment_id
        self.evaluation_run_id = evaluation_run_id
        self.knowledge_base_id = knowledge_base_id
        self.chat_configuration_id = chat_configuration_id
        self.external_processing_allowed = external_processing_allowed

    async def run(self, request: Any) -> str:
        request_metadata = dict(request.metadata or {})
        prompt_key = str(request_metadata.get("prompt_key") or "insight")
        response_schema = request_metadata.get("response_schema")
        messages = [
            {"role": "system", "content": str(request.system_prompt)},
            {"role": "user", "content": str(request.user_prompt)},
        ]
        max_tokens = 20000 if prompt_key.startswith("overall_insight") else 2000
        parameters: Dict[str, Any] = {"temperature": 0, "max_tokens": max_tokens}
        if isinstance(response_schema, dict) and response_schema:
            parameters["response_format"] = {
                "type": "json_schema",
                "json_schema": _normalize_response_schema(response_schema),
            }
        result = await self.gateway.generate(
            messages,
            self.deployment_id,
            parameters=parameters,
            context=self._context(prompt_key),
            external_processing_allowed=self.external_processing_allowed,
            capability="judge",
        )
        if not isinstance(response_schema, dict) or not response_schema:
            return result.text
        try:
            return _canonical_json_object(result.text)
        except RagxplainError:
            retry_messages = [
                {
                    "role": "system",
                    "content": (
                        f"{request.system_prompt}\n\n"
                        "Your previous response was not a complete JSON object. Return only one "
                        "complete JSON object matching the supplied response schema. Do not use "
                        "Markdown fences or explanatory text."
                    ),
                },
                {"role": "user", "content": str(request.user_prompt)},
            ]
            retry = await self.gateway.generate(
                retry_messages,
                self.deployment_id,
                parameters={
                    **parameters,
                    "max_tokens": max(max_tokens, 30000),
                },
                context=self._context(f"{prompt_key}_json_retry"),
                external_processing_allowed=self.external_processing_allowed,
                capability="judge",
            )
            return _canonical_json_object(retry.text)

    def _context(self, prompt_key: str) -> ModelCallContext:
        return ModelCallContext(
            purpose=f"ragxplain_{prompt_key}",
            request_id=f"ragxplain:{self.evaluation_run_id}:{prompt_key}",
            knowledge_base_id=self.knowledge_base_id,
            evaluation_run_id=self.evaluation_run_id,
            chat_configuration_id=self.chat_configuration_id,
        )

    async def aclose(self) -> None:
        return None

    def name(self) -> str:
        return self.deployment_id


def _canonical_json_object(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    candidates = [text]
    object_start = text.find("{")
    if object_start > 0:
        candidates.append(text[object_start:])
    for candidate in candidates:
        try:
            value, _end = json.JSONDecoder().raw_decode(candidate)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
    preview = text[:300].replace("\n", " ")
    raise RagxplainError(f"Judge response is not a complete JSON object: {preview}")


async def _repair_overall_analysis(
    judge: _ModelGatewayJudge,
    overall_insights: Mapping[str, Any],
    schema_path: Path,
) -> Dict[str, Any]:
    try:
        response_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RagxplainError(f"Unable to load the RAGXplain overall insight schema: {exc}") from exc

    compact_payload = _compact_overall_repair_payload(overall_insights)
    request = SimpleNamespace(
        system_prompt=(
            "You are repairing a RAGXplain overall evaluation report. Return only one "
            "complete JSON object matching the supplied schema. Produce exactly three "
            "concise, evidence-based insights. Keep every field compact, and format each "
            "recommended_protocol as exactly three numbered Markdown sections."
        ),
        user_prompt=(
            "Create the overall executive summary and actionable recommendations from "
            "this already-computed evaluation evidence:\n"
            + json.dumps(compact_payload, ensure_ascii=False, separators=(",", ":"))
        ),
        metadata={
            "prompt_key": "overall_insight_repair",
            "response_schema": response_schema,
        },
    )
    raw = await judge.run(request)
    try:
        analysis = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RagxplainError(f"Repaired overall insight is not valid JSON: {exc}") from exc
    if not _is_complete_overall_analysis(analysis):
        raise RagxplainError("Repaired overall insight is missing required executive-summary fields.")
    return analysis


def _compact_overall_repair_payload(overall_insights: Mapping[str, Any]) -> Dict[str, Any]:
    prompt = overall_insights.get("prompt")
    prompt = prompt if isinstance(prompt, Mapping) else {}
    raw_insights = prompt.get("insights")
    metric_insights: Dict[str, Any] = {}
    if isinstance(raw_insights, Mapping):
        for name, value in raw_insights.items():
            if not isinstance(value, Mapping):
                continue
            metric_insights[str(name)] = {
                key: value.get(key)
                for key in (
                    "metric_name",
                    "metric_description",
                    "avg_score",
                    "min_score",
                    "max_score",
                    "analysis",
                )
                if value.get(key) is not None
            }

    configs = prompt.get("configs")
    configs = configs if isinstance(configs, Mapping) else {}
    rag_configs = configs.get("rag_configs")
    rag_configs = rag_configs if isinstance(rag_configs, Mapping) else {}
    chat_config = rag_configs.get("chat_configuration")
    chat_config = chat_config if isinstance(chat_config, Mapping) else {}
    chat_metadata = chat_config.get("metadata")
    chat_metadata = chat_metadata if isinstance(chat_metadata, Mapping) else {}
    compact_config = {
        key: rag_configs.get(key)
        for key in (
            "name",
            "dataset_name",
            "knowledge_base_name",
            "retrieval_mode",
            "top_k",
            "route_distribution",
            "wixqa_metrics",
        )
        if rag_configs.get(key) is not None
    }
    compact_config["chat_configuration"] = {
        key: chat_config.get(key)
        for key in (
            "name",
            "generator_provider",
            "generator_model",
            "response_structure",
            "tone",
            "system_prompt",
            "predefined_prompt",
        )
        if chat_config.get(key) not in (None, "")
    }
    compact_config["runtime"] = {
        key: chat_metadata.get(key)
        for key in (
            "route_mode",
            "retrieval_mode",
            "top_k",
            "reranker_enabled",
            "citations_enabled",
            "classifier_deployment_id",
            "planner_deployment_id",
            "generator_deployment_id",
        )
        if chat_metadata.get(key) not in (None, "")
    }

    examples = prompt.get("examples")
    compact_examples: List[Dict[str, Any]] = []
    if isinstance(examples, Mapping):
        for value in list(examples.values())[:5]:
            if not isinstance(value, Mapping):
                continue
            compact_examples.append(
                {
                    key: value.get(key)
                    for key in (
                        "question",
                        "context_relevancy_score",
                        "context_adherence_score",
                        "answer_relevancy_score",
                        "context_recall_score",
                        "factuality_score",
                        "grading_note_score",
                    )
                    if value.get(key) is not None
                }
            )

    return {
        "metric_insights": metric_insights,
        "configuration": compact_config,
        "traditional_metrics": prompt.get("rag_traditional_metrics", {}),
        "sampled_case_scores": compact_examples,
    }


def _is_complete_overall_analysis(analysis: Any) -> bool:
    if not isinstance(analysis, Mapping) or analysis.get("error"):
        return False
    required_strings = ("executive_summary", "executive_summary_gist", "strategic_conclusion")
    if any(not str(analysis.get(field) or "").strip() for field in required_strings):
        return False
    insights = analysis.get("insights")
    return isinstance(insights, list) and bool(insights)


def _overall_analysis_error(analysis: Any) -> str:
    if isinstance(analysis, Mapping) and analysis.get("error"):
        return str(analysis.get("error"))
    return "RAGXplain overall analysis was incomplete."


def _fallback_overall_analysis(
    overall_insights: Mapping[str, Any],
    original_error: str,
) -> Dict[str, Any]:
    prompt = overall_insights.get("prompt")
    prompt = prompt if isinstance(prompt, Mapping) else {}
    raw_insights = prompt.get("insights")
    metrics: List[Dict[str, Any]] = []
    if isinstance(raw_insights, Mapping):
        for key, value in raw_insights.items():
            if not isinstance(value, Mapping):
                continue
            try:
                average = float(value.get("avg_score"))
            except (TypeError, ValueError):
                continue
            metrics.append(
                {
                    "key": str(key),
                    "name": str(value.get("metric_name") or key).replace("_", " ").title(),
                    "average": average,
                    "minimum": value.get("min_score"),
                    "maximum": value.get("max_score"),
                    "analysis": str(value.get("analysis") or ""),
                }
            )
    metrics.sort(key=lambda item: item["average"])
    if not metrics:
        raise RagxplainUnavailableError(
            f"RAGXplain overall analysis failed and no metric summaries are available: {original_error}"
        )

    weakest = metrics[0]
    strongest = max(metrics, key=lambda item: item["average"])
    summary = (
        "The semantic metric evaluation completed, but the final LLM-generated overall "
        "analysis was truncated. This deterministic fallback reports the available metric "
        f"evidence: **{strongest['name']}** is strongest at {strongest['average']:.3f}, "
        f"while **{weakest['name']}** is the primary improvement area at "
        f"{weakest['average']:.3f}. Rerun RAGXplain diagnosis after updating the worker "
        "to replace this fallback with a judge-generated executive summary."
    )
    fallback_insights = [
        _fallback_metric_insight(metric, index)
        for index, metric in enumerate(metrics[:3], start=1)
    ]
    return {
        "executive_summary": summary,
        "executive_summary_gist": (
            f"Overall synthesis was truncated; metric evidence identifies {weakest['name']} "
            f"({weakest['average']:.3f}) as the leading improvement area."
        ),
        "insights": fallback_insights,
        "strategic_conclusion": (
            "Use these metric-derived actions as a temporary diagnostic. A successful "
            "overall judge call is still required for the full RAGXplain synthesis."
        ),
    }


def _fallback_metric_insight(metric: Mapping[str, Any], index: int) -> Dict[str, Any]:
    score = float(metric["average"])
    priority = "critical" if score < 0.5 else "high" if score < 0.75 else "medium"
    name = str(metric["name"])
    evidence = (
        f"`{metric['key']}` averaged **{score:.3f}**"
        + (f", with a minimum of **{float(metric['minimum']):.3f}**" if metric.get("minimum") is not None else "")
        + "."
    )
    return {
        "title": f"Review {name}",
        "priority": priority,
        "problem_detection": str(metric.get("analysis") or evidence),
        "problem_detection_gist": f"{name} requires review based on an average score of {score:.3f}.",
        "root_cause_analysis": (
            "The final cross-metric judge synthesis was unavailable. Inspect low-scoring "
            "cases and the corresponding retrieval, prompt, and generation traces before "
            "assigning a definitive root cause."
        ),
        "root_cause_analysis_gist": "Root cause requires case-level trace inspection.",
        "evidence_trace": evidence,
        "evidence_trace_gist": evidence,
        "recommended_protocol": (
            f"### 1. Inspect {name} failures\n"
            "Review the lowest-scoring cases and their Source and Trace reports.\n\n"
            "### 2. Change one configuration variable\n"
            "Adjust the most relevant retrieval, prompt, or model setting while preserving "
            "the current run as a baseline.\n\n"
            "### 3. Re-evaluate\n"
            "Run the same dataset selection and compare this metric, latency, and cost."
        ),
        "recommended_protocol_gist": (
            f"- Inspect low-scoring {name} cases.\n"
            "- Change one relevant configuration variable.\n"
            "- Re-run the same benchmark."
        ),
        "fallback_order": index,
    }


def _normalize_response_schema(response_schema: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(dict(response_schema))
    raw_name = str(normalized.get("name") or "")
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_name)
    safe_name = re.sub(r"[_-]+", "_", safe_name).strip("_-")
    safe_name = safe_name[:64].rstrip("_-") or "ragxplain_response"
    normalized["name"] = safe_name
    return normalized


def _validate_semantic_metric_artifacts(
    results_path: Path,
    overall_insights: Mapping[str, Any],
    expected_metrics: Sequence[str],
) -> Dict[str, Any]:
    expected = list(dict.fromkeys(str(metric) for metric in expected_metrics if str(metric)))
    try:
        with results_path.open("r", encoding="utf-8-sig", newline="") as stream:
            columns = set(csv.DictReader(stream).fieldnames or [])
    except OSError as exc:
        raise RagxplainError(f"Unable to validate RAGXplain semantic metrics: {exc}") from exc

    prompt = overall_insights.get("prompt")
    insight_payload = prompt.get("insights") if isinstance(prompt, Mapping) else {}
    insight_names = set(insight_payload) if isinstance(insight_payload, Mapping) else set()
    scored = {metric for metric in expected if f"{metric}_score" in columns}
    summarized = {metric for metric in expected if metric in insight_names}
    completed = [metric for metric in expected if metric in scored and metric in summarized]
    missing_score_columns = [metric for metric in expected if metric not in scored]
    missing_insight_summaries = [metric for metric in expected if metric not in summarized]
    missing = [metric for metric in expected if metric not in completed]

    if expected and not completed:
        raise RagxplainError(
            "RAGXplain produced no semantic metrics. Failed metrics: "
            + ", ".join(expected)
            + ". Check Model Farm usage errors for ragxplain_* calls."
        )

    return {
        "status": "completed" if not missing else "partial",
        "expected": expected,
        "completed": completed,
        "missing": missing,
        "missing_score_columns": missing_score_columns,
        "missing_insight_summaries": missing_insight_summaries,
    }


def _attach_configuration_targets(payload: Dict[str, Any]) -> None:
    analysis = payload.get("analysis")
    if not isinstance(analysis, dict):
        return
    analysis["configuration_targets"] = [
        {
            "diagnostic_area": "Retrieval quality",
            "metrics": ["Context Relevancy", "Context Recall"],
            "sections": ["Knowledge Base", "Adaptive RAG"],
            "fields": ["chunking_strategy", "embedding_deployment_id", "retrieval_mode", "top_k", "reranker_deployment_id"],
        },
        {
            "diagnostic_area": "Faithfulness and grounding",
            "metrics": ["Context Adherence", "Factuality"],
            "sections": ["Generator target & prompts", "Adaptive RAG"],
            "fields": ["generator_deployment_id", "system_prompt", "predefined_prompt", "citations_enabled"],
        },
        {
            "diagnostic_area": "Relevance and intent",
            "metrics": ["Answer Relevancy"],
            "sections": ["Adaptive RAG", "Generator target & prompts"],
            "fields": ["route_mode", "classifier_deployment_id", "planner_deployment_id", "response_structure"],
        },
        {
            "diagnostic_area": "Structure and style",
            "metrics": ["Grading Note"],
            "sections": ["General", "Generator target & prompts"],
            "fields": ["response_structure", "tone", "humor_level", "system_prompt", "predefined_prompt"],
        },
    ]
