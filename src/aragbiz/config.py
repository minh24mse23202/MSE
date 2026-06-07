from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union


@dataclass(frozen=True)
class AppConfig:
    sample_dataset: str = "data/sample/business_workflows.jsonl"
    feedback_store: str = "feedback.jsonl"
    simple_top_k: int = 2
    moderate_top_k: int = 4
    complex_top_k: int = 6
    default_mode: str = "hybrid"
    bm25_weight: float = 0.65
    dense_weight: float = 0.35
    max_context_chars: int = 900


def load_config(path: Union[str, Path] = "config/default.toml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()
    raw = _load_toml(config_path)
    return AppConfig(
        sample_dataset=raw.get("paths", {}).get("sample_dataset", AppConfig.sample_dataset),
        feedback_store=raw.get("paths", {}).get("feedback_store", AppConfig.feedback_store),
        simple_top_k=int(raw.get("classifier", {}).get("simple_top_k", AppConfig.simple_top_k)),
        moderate_top_k=int(raw.get("classifier", {}).get("moderate_top_k", AppConfig.moderate_top_k)),
        complex_top_k=int(raw.get("classifier", {}).get("complex_top_k", AppConfig.complex_top_k)),
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
