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
        return self._read_and_validate_insights(path)

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
        max_tokens = 6000 if prompt_key == "overall_insight" else 1600
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
                    "max_tokens": max(max_tokens, 7000),
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
