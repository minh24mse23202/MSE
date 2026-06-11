from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union


@dataclass(frozen=True)
class AppConfig:
    sample_dataset: str = "data/processed/wixqa_expertwritten_qac.jsonl"
    fallback_sample_dataset: str = "data/sample/business_workflows.jsonl"
    kb_corpus: str = "data/processed/wix_kb_corpus_documents.jsonl"
    feedback_store: str = "feedback.jsonl"
    simple_top_k: int = 2
    moderate_top_k: int = 4
    complex_top_k: int = 6
    classifier_model_path: str = "data/artifacts/query_classifier_distilbert"
    classifier_fallback_model_path: str = "data/artifacts/query_classifier_nb.json"
    use_trained_classifier: bool = True
    default_mode: str = "hybrid"
    bm25_weight: float = 0.65
    dense_weight: float = 0.35
    max_context_chars: int = 900


def load_config(path: Union[str, Path] = "config/default.toml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    raw = _load_toml(config_path)
    use_trained = bool(raw.get("classifier", {}).get("use_trained_model", AppConfig.use_trained_classifier))
    env_use_trained = os.getenv("ARAGBIZ_USE_TRAINED_CLASSIFIER")
    if env_use_trained is not None:
        use_trained = _parse_bool(env_use_trained)
    return AppConfig(
        sample_dataset=raw.get("paths", {}).get("sample_dataset", AppConfig.sample_dataset),
        fallback_sample_dataset=raw.get("paths", {}).get("fallback_sample_dataset", AppConfig.fallback_sample_dataset),
        kb_corpus=raw.get("paths", {}).get("kb_corpus", AppConfig.kb_corpus),
        feedback_store=raw.get("paths", {}).get("feedback_store", AppConfig.feedback_store),
        simple_top_k=int(raw.get("classifier", {}).get("simple_top_k", AppConfig.simple_top_k)),
        moderate_top_k=int(raw.get("classifier", {}).get("moderate_top_k", AppConfig.moderate_top_k)),
        complex_top_k=int(raw.get("classifier", {}).get("complex_top_k", AppConfig.complex_top_k)),
        classifier_model_path=raw.get("classifier", {}).get("model_path", AppConfig.classifier_model_path),
        classifier_fallback_model_path=raw.get("classifier", {}).get("fallback_model_path", AppConfig.classifier_fallback_model_path),
        use_trained_classifier=use_trained,
        default_mode=raw.get("retrieval", {}).get("default_mode", AppConfig.default_mode),
        bm25_weight=float(raw.get("retrieval", {}).get("bm25_weight", AppConfig.bm25_weight)),
        dense_weight=float(raw.get("retrieval", {}).get("dense_weight", AppConfig.dense_weight)),
        max_context_chars=int(raw.get("generator", {}).get("max_context_chars", AppConfig.max_context_chars)),
    )


def _load_toml(path: Path) -> Dict[str, Dict[str, Any]]:
    try:
        import tomllib  # type: ignore[attr-defined]

        with path.open("rb") as file:
            return tomllib.load(file)
    except ModuleNotFoundError:
        try:
            import tomli  # type: ignore

            with path.open("rb") as file:
                return tomli.load(file)
        except ModuleNotFoundError:
            return _minimal_toml(path)


def _minimal_toml(path: Path) -> Dict[str, Dict[str, Any]]:
    current_section = ""
    data: Dict[str, Dict[str, Any]] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            data.setdefault(current_section, {})
            continue
        if "=" not in line or not current_section:
            continue
        key, value = [part.strip() for part in line.split("=", 1)]
        data[current_section][key] = _parse_scalar(value)
    return data


def _parse_scalar(value: str) -> Any:
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
