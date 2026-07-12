from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence


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
        self._read_and_validate_insights(artifacts["overall_insights.json"])
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
        adaptive_metrics = metrics.get("adaptive", {}) if isinstance(metrics, dict) else {}
        return {
            "question": str(getattr(case, "question", "") or ""),
            "candidate_answer": str(getattr(case, "adaptive_answer", "") or ""),
            "contexts": "\n\n".join(context_sections),
            "gt_answer": str(getattr(case, "expected_answer", "") or ""),
            "routing_match": adaptive_metrics.get("routing_match", 0.0),
            "context_relevance_proxy": adaptive_metrics.get("context_relevance", 0.0),
            "faithfulness_proxy": adaptive_metrics.get("faithfulness_proxy", 0.0),
            "answer_overlap": adaptive_metrics.get("answer_overlap", 0.0),
        }
