from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import math
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Protocol, Sequence


MODEL_CAPABILITIES = {"generation", "embedding", "rerank", "judge", "planner", "classifier"}
SECRET_ENV_PREFIX = "ARAGBIZ_MODEL_"


class ModelFarmError(ValueError):
    """Raised when a model deployment cannot be configured or executed."""


class ModelBudgetExceeded(ModelFarmError):
    """Raised before a call that would exceed a configured hard budget."""


class ModelPolicyError(ModelFarmError):
    """Raised when a model call violates data-egress policy."""


@dataclass(frozen=True)
class ModelDeployment:
    id: str
    name: str
    provider: str
    model: str
    capabilities: List[str]
    api_base: str = ""
    credential_env_refs: Dict[str, str] = field(default_factory=dict)
    credential_secrets: Dict[str, str] = field(default_factory=dict)
    default_parameters: Dict[str, Any] = field(default_factory=dict)
    limits: Dict[str, Any] = field(default_factory=dict)
    pricing: Dict[str, Any] = field(default_factory=dict)
    monthly_budget_usd: float = 0.0
    hard_budget: bool = True
    locality: str = "remote"
    enabled: bool = False
    health_status: str = "untested"
    last_health_check: str = ""
    last_error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @property
    def is_local(self) -> bool:
        return self.locality == "local" or self.provider.lower() == "local"

    @property
    def dimension(self) -> int:
        try:
            return int(self.limits.get("dimension") or 0)
        except (TypeError, ValueError):
            return 0


@dataclass(frozen=True)
class ModelCallContext:
    purpose: str
    request_id: str = ""
    user_id: str = ""
    conversation_id: str = ""
    knowledge_base_id: str = ""
    evaluation_run_id: str = ""


@dataclass(frozen=True)
class ModelUsageEvent:
    id: str
    deployment_id: str
    provider: str
    model: str
    capability: str
    purpose: str
    status: str
    request_id: str = ""
    user_id: str = ""
    conversation_id: str = ""
    knowledge_base_id: str = ""
    evaluation_run_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    fallback_index: int = 0
    error_code: str = ""
    error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass(frozen=True)
class ModelGenerationResult:
    text: str
    deployment_id: str
    provider: str
    model: str
    status: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    finish_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelEmbeddingResult:
    embeddings: List[List[float]]
    deployment_id: str
    provider: str
    model: str
    dimension: int
    input_tokens: int = 0
    estimated_cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRerankItem:
    index: int
    score: float


@dataclass(frozen=True)
class ModelRerankResult:
    items: List[ModelRerankItem]
    deployment_id: str
    provider: str
    model: str
    estimated_cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelStreamEvent:
    type: str
    data: Dict[str, Any]


class ModelFarmRepository(Protocol):
    def initialize(self) -> None: ...

    def list_deployments(self) -> List[ModelDeployment]: ...

    def get_deployment(self, deployment_id: str) -> ModelDeployment: ...

    def save_deployment(self, deployment: ModelDeployment) -> ModelDeployment: ...

    def delete_deployment(self, deployment_id: str) -> None: ...

    def append_usage(self, event: ModelUsageEvent) -> None: ...

    def list_usage(
        self,
        *,
        deployment_id: str = "",
        purpose: str = "",
        limit: int = 500,
    ) -> List[ModelUsageEvent]: ...


class JsonModelFarmRepository:
    def __init__(self, path: str):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"deployments": {}, "usage": []})

    def list_deployments(self) -> List[ModelDeployment]:
        state = self._read()
        items = [_deployment_from_dict(value) for value in state["deployments"].values()]
        return sorted(items, key=lambda item: (item.name.lower(), item.id))

    def get_deployment(self, deployment_id: str) -> ModelDeployment:
        state = self._read()
        payload = state["deployments"].get(deployment_id)
        if not payload:
            raise KeyError(f"Model deployment not found: {deployment_id}")
        return _deployment_from_dict(payload)

    def save_deployment(self, deployment: ModelDeployment) -> ModelDeployment:
        state = self._read()
        state["deployments"][deployment.id] = asdict(deployment)
        self._write(state)
        return deployment

    def delete_deployment(self, deployment_id: str) -> None:
        state = self._read()
        if deployment_id not in state["deployments"]:
            raise KeyError(f"Model deployment not found: {deployment_id}")
        if any(item.get("deployment_id") == deployment_id for item in state["usage"]):
            raise ModelFarmError("A deployment with usage history cannot be deleted; disable it instead.")
        del state["deployments"][deployment_id]
        self._write(state)

    def append_usage(self, event: ModelUsageEvent) -> None:
        state = self._read()
        state["usage"].append(asdict(event))
        self._write(state)

    def list_usage(self, *, deployment_id: str = "", purpose: str = "", limit: int = 500) -> List[ModelUsageEvent]:
        state = self._read()
        items = [_usage_from_dict(item) for item in state["usage"]]
        if deployment_id:
            items = [item for item in items if item.deployment_id == deployment_id]
        if purpose:
            items = [item for item in items if item.purpose == purpose]
        items.sort(key=lambda item: item.created_at, reverse=True)
        return items[: max(1, min(limit, 5000))]

    def _read(self) -> Dict[str, Any]:
        self.initialize()
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            state = {}
        state.setdefault("deployments", {})
        state.setdefault("usage", [])
        return state

    def _write(self, state: Dict[str, Any]) -> None:
        self.path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


class PostgresModelFarmRepository:
    def __init__(self, database_url: str):
        try:
            from sqlalchemy import create_engine  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise ModelFarmError("Install the api extra to use PostgreSQL Model Farm storage.") from exc
        self.engine = create_engine(database_url, future=True)

    def initialize(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS model_deployments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            capabilities_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            api_base TEXT NOT NULL DEFAULT '',
            credential_env_refs_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            credential_secrets_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            default_parameters_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            limits_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            pricing_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            monthly_budget_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
            hard_budget BOOLEAN NOT NULL DEFAULT TRUE,
            locality TEXT NOT NULL DEFAULT 'remote',
            enabled BOOLEAN NOT NULL DEFAULT FALSE,
            health_status TEXT NOT NULL DEFAULT 'untested',
            last_health_check TEXT NOT NULL DEFAULT '',
            last_error TEXT NOT NULL DEFAULT '',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_usage_events (
            id TEXT PRIMARY KEY,
            deployment_id TEXT NOT NULL REFERENCES model_deployments(id) ON DELETE RESTRICT,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            capability TEXT NOT NULL,
            purpose TEXT NOT NULL,
            status TEXT NOT NULL,
            request_id TEXT NOT NULL DEFAULT '',
            user_id TEXT NOT NULL DEFAULT '',
            conversation_id TEXT NOT NULL DEFAULT '',
            knowledge_base_id TEXT NOT NULL DEFAULT '',
            evaluation_run_id TEXT NOT NULL DEFAULT '',
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
            estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
            fallback_index INTEGER NOT NULL DEFAULT 0,
            error_code TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_model_usage_created_at ON model_usage_events(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_model_usage_deployment ON model_usage_events(deployment_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_model_usage_purpose ON model_usage_events(purpose, created_at DESC);
        ALTER TABLE model_deployments
            ADD COLUMN IF NOT EXISTS credential_secrets_json JSONB NOT NULL DEFAULT '{}'::jsonb;
        """
        with self.engine.begin() as connection:
            for statement in [part.strip() for part in ddl.split(";") if part.strip()]:
                connection.exec_driver_sql(statement)

    def list_deployments(self) -> List[ModelDeployment]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(text("SELECT * FROM model_deployments ORDER BY name, id")).mappings()
            return [_deployment_from_row(row) for row in rows]

    def get_deployment(self, deployment_id: str) -> ModelDeployment:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            row = connection.execute(
                text("SELECT * FROM model_deployments WHERE id = :id"), {"id": deployment_id}
            ).mappings().first()
        if not row:
            raise KeyError(f"Model deployment not found: {deployment_id}")
        return _deployment_from_row(row)

    def save_deployment(self, deployment: ModelDeployment) -> ModelDeployment:
        from sqlalchemy import text

        payload = _deployment_db_payload(deployment)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO model_deployments (
                        id, name, provider, model, capabilities_json, api_base,
                        credential_env_refs_json, credential_secrets_json, default_parameters_json, limits_json,
                        pricing_json, monthly_budget_usd, hard_budget, locality, enabled,
                        health_status, last_health_check, last_error, metadata_json,
                        created_at, updated_at
                    ) VALUES (
                        :id, :name, :provider, :model, CAST(:capabilities AS JSONB), :api_base,
                        CAST(:credential_env_refs AS JSONB), CAST(:credential_secrets AS JSONB),
                        CAST(:default_parameters AS JSONB), CAST(:limits AS JSONB),
                        CAST(:pricing AS JSONB), :monthly_budget_usd, :hard_budget, :locality, :enabled,
                        :health_status, :last_health_check, :last_error, CAST(:metadata AS JSONB),
                        :created_at, :updated_at
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name, provider = EXCLUDED.provider, model = EXCLUDED.model,
                        capabilities_json = EXCLUDED.capabilities_json, api_base = EXCLUDED.api_base,
                        credential_env_refs_json = EXCLUDED.credential_env_refs_json,
                        credential_secrets_json = EXCLUDED.credential_secrets_json,
                        default_parameters_json = EXCLUDED.default_parameters_json,
                        limits_json = EXCLUDED.limits_json, pricing_json = EXCLUDED.pricing_json,
                        monthly_budget_usd = EXCLUDED.monthly_budget_usd, hard_budget = EXCLUDED.hard_budget,
                        locality = EXCLUDED.locality, enabled = EXCLUDED.enabled,
                        health_status = EXCLUDED.health_status, last_health_check = EXCLUDED.last_health_check,
                        last_error = EXCLUDED.last_error, metadata_json = EXCLUDED.metadata_json,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                payload,
            )
        return deployment

    def delete_deployment(self, deployment_id: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            used = connection.execute(
                text("SELECT 1 FROM model_usage_events WHERE deployment_id = :id LIMIT 1"), {"id": deployment_id}
            ).first()
            if used:
                raise ModelFarmError("A deployment with usage history cannot be deleted; disable it instead.")
            result = connection.execute(text("DELETE FROM model_deployments WHERE id = :id"), {"id": deployment_id})
            if result.rowcount == 0:
                raise KeyError(f"Model deployment not found: {deployment_id}")

    def append_usage(self, event: ModelUsageEvent) -> None:
        from sqlalchemy import text

        payload = asdict(event)
        payload["metadata"] = json.dumps(payload.pop("metadata"))
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO model_usage_events (
                        id, deployment_id, provider, model, capability, purpose, status,
                        request_id, user_id, conversation_id, knowledge_base_id, evaluation_run_id,
                        input_tokens, output_tokens, total_tokens, latency_ms, estimated_cost_usd,
                        fallback_index, error_code, error, metadata_json, created_at
                    ) VALUES (
                        :id, :deployment_id, :provider, :model, :capability, :purpose, :status,
                        :request_id, :user_id, :conversation_id, :knowledge_base_id, :evaluation_run_id,
                        :input_tokens, :output_tokens, :total_tokens, :latency_ms, :estimated_cost_usd,
                        :fallback_index, :error_code, :error, CAST(:metadata AS JSONB), :created_at
                    )
                    """
                ),
                payload,
            )

    def list_usage(self, *, deployment_id: str = "", purpose: str = "", limit: int = 500) -> List[ModelUsageEvent]:
        from sqlalchemy import text

        filters: List[str] = []
        params: Dict[str, Any] = {"limit": max(1, min(limit, 5000))}
        if deployment_id:
            filters.append("deployment_id = :deployment_id")
            params["deployment_id"] = deployment_id
        if purpose:
            filters.append("purpose = :purpose")
            params["purpose"] = purpose
        where = f"WHERE {' AND '.join(filters)}" if filters else ""
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(f"SELECT * FROM model_usage_events {where} ORDER BY created_at DESC LIMIT :limit"), params
            ).mappings()
            return [_usage_from_row(row) for row in rows]


class ModelFarmService:
    def __init__(
        self,
        repository: ModelFarmRepository,
        *,
        global_monthly_budget_usd: float = 0.0,
        secret_key: str = "",
    ):
        self.repository = repository
        self.global_monthly_budget_usd = max(float(global_monthly_budget_usd), 0.0)
        self._secret_key = secret_key or os.getenv("ARAGBIZ_MODEL_SECRET_KEY") or "aragbiz-local-development-secret"
        self.repository.initialize()
        self._seed_local_deployments()

    def providers(self) -> List[Dict[str, Any]]:
        return model_provider_templates()

    def list_deployments(self, *, capability: str = "", enabled: Optional[bool] = None) -> List[ModelDeployment]:
        deployments = self.repository.list_deployments()
        if capability:
            deployments = [item for item in deployments if capability in item.capabilities]
        if enabled is not None:
            deployments = [item for item in deployments if item.enabled is enabled]
        return deployments

    def get_deployment(self, deployment_id: str) -> ModelDeployment:
        return self.repository.get_deployment(deployment_id)

    def create_deployment(self, payload: Dict[str, Any]) -> ModelDeployment:
        now = utc_now()
        deployment = _validated_deployment(
            ModelDeployment(
                id=str(payload.get("id") or f"model-{uuid.uuid4().hex}"),
                name=str(payload.get("name") or "").strip(),
                provider=str(payload.get("provider") or "").strip(),
                model=str(payload.get("model") or "").strip(),
                capabilities=list(payload.get("capabilities") or []),
                api_base=str(payload.get("api_base") or "").strip(),
                credential_env_refs=dict(payload.get("credential_env_refs") or {}),
                credential_secrets=self._encrypted_secret_payload(payload.get("credential_secrets") or {}),
                default_parameters=dict(payload.get("default_parameters") or {}),
                limits=dict(payload.get("limits") or {}),
                pricing=dict(payload.get("pricing") or {}),
                monthly_budget_usd=float(payload.get("monthly_budget_usd") or 0.0),
                hard_budget=bool(payload.get("hard_budget", True)),
                locality=str(payload.get("locality") or "remote"),
                enabled=bool(payload.get("enabled", False)),
                health_status=str(payload.get("health_status") or "untested"),
                metadata=dict(payload.get("metadata") or {}),
                created_at=now,
                updated_at=now,
            )
        )
        if deployment.enabled and not deployment.is_local and deployment.health_status != "healthy":
            raise ModelFarmError("Test a remote deployment successfully before enabling it.")
        self._ensure_unique_name(deployment.name)
        return self.repository.save_deployment(deployment)

    def create_deployment_from_template(self, template_id: str, payload: Dict[str, Any]) -> ModelDeployment:
        template = _template_by_id(template_id)
        if not template:
            raise ModelFarmError(f"Unknown provider template: {template_id}")
        if template.get("builtin_deployment_id"):
            raise ModelFarmError("Built-in local deployments are already registered; select the existing deployment instead.")
        defaults = dict(template.get("deployment_defaults") or {})
        allowed_capabilities = set(template.get("capabilities") or defaults.get("capabilities") or [])
        requested_capabilities = list(payload.get("capabilities") or defaults.get("capabilities") or [])
        unsupported = sorted(set(requested_capabilities) - allowed_capabilities)
        if unsupported:
            raise ModelFarmError(f"Template '{template_id}' does not support capabilities: {', '.join(unsupported)}.")
        merged = {
            **defaults,
            "id": f"model-{uuid.uuid4().hex}",
            "name": self._unique_name(str(payload.get("name") or defaults.get("name") or template.get("label") or "Model deployment")),
            "model": str(payload.get("model") or defaults.get("model") or "").strip(),
            "api_base": str(payload.get("api_base", defaults.get("api_base", "")) or "").strip(),
            "credential_env_refs": dict(payload.get("credential_env_refs") or defaults.get("credential_env_refs") or {}),
            "credential_secrets": self._encrypted_secret_payload(payload.get("credential_secrets") or {}),
            "default_parameters": dict(payload.get("default_parameters") or defaults.get("default_parameters") or {}),
            "limits": dict(payload.get("limits") or defaults.get("limits") or {}),
            "pricing": dict(payload.get("pricing") or defaults.get("pricing") or {}),
            "monthly_budget_usd": float(payload.get("monthly_budget_usd", defaults.get("monthly_budget_usd", 0.0)) or 0.0),
            "hard_budget": bool(payload.get("hard_budget", defaults.get("hard_budget", True))),
            "enabled": bool(payload.get("enabled", False)),
            "capabilities": requested_capabilities,
            "metadata": {
                **dict(defaults.get("metadata") or {}),
                **dict(payload.get("metadata") or {}),
                "template_id": template_id,
                "provider_label": template.get("provider_label") or template.get("label") or "",
            },
        }
        return self.create_deployment(merged)

    def update_deployment(self, deployment_id: str, payload: Dict[str, Any]) -> ModelDeployment:
        current = self.get_deployment(deployment_id)
        immutable = {"provider", "model", "capabilities", "locality"}
        for key in immutable:
            if key in payload and payload[key] != getattr(current, key):
                raise ModelFarmError(f"{key} is immutable after registration; clone the deployment instead.")
        updated = _validated_deployment(
            replace(
                current,
                name=str(payload.get("name", current.name)).strip(),
                api_base=str(payload.get("api_base", current.api_base)).strip(),
                credential_env_refs=dict(payload.get("credential_env_refs", current.credential_env_refs)),
                credential_secrets=self._updated_secret_payload(
                    current.credential_secrets,
                    payload.get("credential_secrets") if "credential_secrets" in payload else None,
                ),
                default_parameters=dict(payload.get("default_parameters", current.default_parameters)),
                limits=dict(payload.get("limits", current.limits)),
                pricing=dict(payload.get("pricing", current.pricing)),
                monthly_budget_usd=float(payload.get("monthly_budget_usd", current.monthly_budget_usd)),
                hard_budget=bool(payload.get("hard_budget", current.hard_budget)),
                enabled=bool(payload.get("enabled", current.enabled)),
                metadata=dict(payload.get("metadata", current.metadata)),
                updated_at=utc_now(),
            )
        )
        if updated.enabled and not updated.is_local and updated.health_status != "healthy":
            raise ModelFarmError("Test a remote deployment successfully before enabling it.")
        self._ensure_unique_name(updated.name, exclude_id=deployment_id)
        return self.repository.save_deployment(updated)

    def delete_deployment(self, deployment_id: str) -> None:
        deployment = self.get_deployment(deployment_id)
        if deployment.metadata.get("builtin"):
            raise ModelFarmError("Built-in local deployments cannot be deleted; disable them instead.")
        self.repository.delete_deployment(deployment_id)

    def resolve(self, deployment_id: str, capability: str, *, require_enabled: bool = True) -> ModelDeployment:
        deployment = self.get_deployment(deployment_id)
        if capability not in deployment.capabilities:
            raise ModelFarmError(f"Deployment '{deployment.name}' does not support {capability}.")
        if require_enabled and not deployment.enabled:
            raise ModelFarmError(f"Deployment '{deployment.name}' is disabled.")
        missing = self.missing_credentials(deployment)
        if missing:
            raise ModelFarmError(f"Deployment '{deployment.name}' is missing credentials: {', '.join(missing)}.")
        return deployment

    def missing_credentials(self, deployment: ModelDeployment) -> List[str]:
        if deployment.is_local or deployment.metadata.get("auth_mode") == "ambient":
            return []
        missing: List[str] = []
        secret_values = self.credential_values(deployment)
        for key, env_name in deployment.credential_env_refs.items():
            if key in secret_values:
                continue
            if env_name and os.getenv(env_name):
                continue
            missing.append(env_name or key)
        return sorted(set(missing))

    def credential_status(self, deployment: ModelDeployment) -> Dict[str, Any]:
        missing = self.missing_credentials(deployment)
        secret_keys = sorted(key for key, value in deployment.credential_secrets.items() if value)
        return {
            "configured": not missing,
            "missing": missing,
            "references": sorted(set(deployment.credential_env_refs.values())),
            "stored_secret_keys": secret_keys,
            "has_stored_secret": bool(secret_keys),
        }

    def credential_values(self, deployment: ModelDeployment) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for key, token in deployment.credential_secrets.items():
            clean_key = str(key).strip()
            if not clean_key or not token:
                continue
            values[clean_key] = _decrypt_secret(str(token), self._secret_key)
        return values

    def _encrypted_secret_payload(self, raw: Dict[str, Any]) -> Dict[str, str]:
        encrypted: Dict[str, str] = {}
        for key, value in dict(raw or {}).items():
            clean_key = str(key).strip()
            clean_value = str(value or "").strip()
            if not clean_key or not clean_value:
                continue
            encrypted[clean_key] = clean_value if clean_value.startswith("v1:") else _encrypt_secret(clean_value, self._secret_key)
        return encrypted

    def _updated_secret_payload(self, current: Dict[str, str], raw: Optional[Dict[str, Any]]) -> Dict[str, str]:
        if raw is None:
            return dict(current)
        updated = dict(current)
        for key, value in dict(raw or {}).items():
            clean_key = str(key).strip()
            if not clean_key:
                continue
            clean_value = str(value or "").strip()
            if clean_value:
                updated[clean_key] = clean_value if clean_value.startswith("v1:") else _encrypt_secret(clean_value, self._secret_key)
            else:
                updated.pop(clean_key, None)
        return updated

    def set_health(self, deployment_id: str, *, healthy: bool, error: str = "", dimension: int = 0) -> ModelDeployment:
        current = self.get_deployment(deployment_id)
        limits = dict(current.limits)
        if dimension:
            configured = int(limits.get("dimension") or 0)
            if configured and configured != dimension:
                healthy = False
                error = f"Configured dimension {configured} does not match runtime dimension {dimension}."
            else:
                limits["dimension"] = dimension
        updated = replace(
            current,
            limits=limits,
            health_status="healthy" if healthy else "unavailable",
            last_health_check=utc_now(),
            last_error="" if healthy else _safe_error(error),
            updated_at=utc_now(),
        )
        return self.repository.save_deployment(updated)

    def assert_egress_allowed(self, deployment: ModelDeployment, *, external_processing_allowed: bool, content_kind: str) -> None:
        if not deployment.is_local and not external_processing_allowed:
            raise ModelPolicyError(
                f"{content_kind} cannot be sent to remote deployment '{deployment.name}'. "
                "Enable external processing on the knowledge base first."
            )

    def assert_budget(self, deployment: ModelDeployment) -> None:
        month_prefix = datetime.now(timezone.utc).strftime("%Y-%m")
        all_usage = self.repository.list_usage(limit=5000)
        monthly = [item for item in all_usage if item.created_at.startswith(month_prefix) and item.status == "completed"]
        deployment_spend = sum(item.estimated_cost_usd for item in monthly if item.deployment_id == deployment.id)
        global_spend = sum(item.estimated_cost_usd for item in monthly)
        if deployment.hard_budget and deployment.monthly_budget_usd > 0 and deployment_spend >= deployment.monthly_budget_usd:
            raise ModelBudgetExceeded(f"Monthly budget reached for deployment '{deployment.name}'.")
        if self.global_monthly_budget_usd > 0 and global_spend >= self.global_monthly_budget_usd:
            raise ModelBudgetExceeded("Global Model Farm monthly budget has been reached.")

    def record_usage(self, event: ModelUsageEvent) -> None:
        self.repository.append_usage(event)

    def list_usage(self, *, deployment_id: str = "", purpose: str = "", limit: int = 500) -> List[ModelUsageEvent]:
        return self.repository.list_usage(deployment_id=deployment_id, purpose=purpose, limit=limit)

    def usage_summary(self) -> Dict[str, Any]:
        events = self.repository.list_usage(limit=5000)
        completed = [item for item in events if item.status == "completed"]
        by_provider: Dict[str, Dict[str, Any]] = {}
        by_purpose: Dict[str, Dict[str, Any]] = {}
        for event in completed:
            _add_usage_bucket(by_provider, event.provider, event)
            _add_usage_bucket(by_purpose, event.purpose, event)
        return {
            "calls": len(events),
            "completed_calls": len(completed),
            "failed_calls": len(events) - len(completed),
            "input_tokens": sum(item.input_tokens for item in completed),
            "output_tokens": sum(item.output_tokens for item in completed),
            "total_tokens": sum(item.total_tokens for item in completed),
            "estimated_cost_usd": round(sum(item.estimated_cost_usd for item in completed), 8),
            "average_latency_ms": round(sum(item.latency_ms for item in completed) / len(completed), 3) if completed else 0.0,
            "by_provider": by_provider,
            "by_purpose": by_purpose,
        }

    def _ensure_unique_name(self, name: str, exclude_id: str = "") -> None:
        if not name:
            raise ModelFarmError("Deployment name is required.")
        for item in self.repository.list_deployments():
            if item.id != exclude_id and item.name.lower() == name.lower():
                raise ModelFarmError(f"A deployment named '{name}' already exists.")

    def _unique_name(self, base_name: str) -> str:
        normalized = " ".join((base_name or "Model deployment").strip().split()) or "Model deployment"
        existing = {item.name.lower() for item in self.repository.list_deployments()}
        if normalized.lower() not in existing:
            return normalized
        for index in range(2, 1000):
            candidate = f"{normalized} {index}"
            if candidate.lower() not in existing:
                return candidate
        return f"{normalized} {uuid.uuid4().hex[:8]}"

    def _seed_local_deployments(self) -> None:
        existing = {item.id for item in self.repository.list_deployments()}
        for deployment in builtin_model_deployments():
            if deployment.id not in existing:
                self.repository.save_deployment(deployment)


class ModelGateway:
    def __init__(self, service: ModelFarmService):
        self.service = service

    async def generate(
        self,
        messages: List[Dict[str, str]],
        deployment_id: str,
        *,
        fallback_deployment_ids: Optional[Sequence[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[ModelCallContext] = None,
        external_processing_allowed: bool = True,
        require_enabled: bool = True,
    ) -> ModelGenerationResult:
        candidates = [deployment_id, *(fallback_deployment_ids or [])]
        last_error: Optional[Exception] = None
        for fallback_index, candidate_id in enumerate(candidates):
            deployment = self.service.resolve(candidate_id, "generation", require_enabled=require_enabled)
            self.service.assert_egress_allowed(
                deployment,
                external_processing_allowed=external_processing_allowed,
                content_kind="Prompt content",
            )
            self.service.assert_budget(deployment)
            started = time.perf_counter()
            try:
                result = await self._generate_once(deployment, messages, parameters or {})
                latency = (time.perf_counter() - started) * 1000
                self._record_completed(deployment, "generation", context, result.input_tokens, result.output_tokens, latency, result.estimated_cost_usd, fallback_index)
                return replace(result, metadata={**result.metadata, "fallback_index": fallback_index, "latency_ms": round(latency, 3)})
            except Exception as exc:  # provider exception types are optional imports
                latency = (time.perf_counter() - started) * 1000
                self._record_failed(deployment, "generation", context, latency, exc, fallback_index)
                last_error = exc
                if not _is_retryable_error(exc) or fallback_index == len(candidates) - 1:
                    break
        raise ModelFarmError(f"Generator execution failed: {_safe_error(last_error)}") from last_error

    async def stream(
        self,
        messages: List[Dict[str, str]],
        deployment_id: str,
        *,
        fallback_deployment_ids: Optional[Sequence[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[ModelCallContext] = None,
        external_processing_allowed: bool = True,
    ) -> AsyncIterator[ModelStreamEvent]:
        deployment = self.service.resolve(deployment_id, "generation")
        self.service.assert_egress_allowed(deployment, external_processing_allowed=external_processing_allowed, content_kind="Prompt content")
        self.service.assert_budget(deployment)
        if deployment.provider.lower() == "local" and deployment.model == "google/flan-t5-small":
            prompt = "\n".join(item.get("content", "") for item in messages)
            started = time.perf_counter()
            text_parts: List[str] = []
            try:
                async for delta in _stream_local_flan(deployment.model, prompt, {**deployment.default_parameters, **(parameters or {})}):
                    if delta:
                        text_parts.append(delta)
                        yield ModelStreamEvent("delta", {"text": delta})
                latency = (time.perf_counter() - started) * 1000
                output = "".join(text_parts)
                input_tokens = _rough_token_count(prompt)
                output_tokens = _rough_token_count(output)
                self._record_completed(deployment, "generation", context, input_tokens, output_tokens, latency, 0.0, 0)
                yield ModelStreamEvent(
                    "model_completed",
                    _generation_public_dict(
                        ModelGenerationResult(
                            output,
                            deployment.id,
                            deployment.provider,
                            deployment.model,
                            "completed",
                            input_tokens,
                            output_tokens,
                            0.0,
                            "stop",
                            {"runtime": "transformers-text2text-stream", "latency_ms": round(latency, 3)},
                        )
                    ),
                )
            except Exception as exc:
                latency = (time.perf_counter() - started) * 1000
                self._record_failed(deployment, "generation", context, latency, exc, 0)
                raise ModelFarmError(f"Generator streaming failed: {_safe_error(exc)}") from exc
            return
        if deployment.is_local:
            result = await self.generate(
                messages,
                deployment_id,
                fallback_deployment_ids=fallback_deployment_ids,
                parameters=parameters,
                context=context,
                external_processing_allowed=external_processing_allowed,
            )
            for chunk in _text_chunks(result.text, 32):
                yield ModelStreamEvent("delta", {"text": chunk})
            yield ModelStreamEvent("model_completed", _generation_public_dict(result))
            return
        try:
            import litellm  # type: ignore
        except ImportError as exc:
            raise ModelFarmError("Install the models extra to use remote model deployments: python -m pip install -e \".[models]\".") from exc
        kwargs = self._litellm_kwargs(deployment)
        kwargs.update(deployment.default_parameters)
        kwargs.update(parameters or {})
        kwargs.update({"model": deployment.model, "messages": messages, "stream": True})
        started = time.perf_counter()
        text_parts: List[str] = []
        try:
            response = await litellm.acompletion(**kwargs)
            async for chunk in response:
                delta = _stream_text(chunk)
                if delta:
                    text_parts.append(delta)
                    yield ModelStreamEvent("delta", {"text": delta})
            latency = (time.perf_counter() - started) * 1000
            output = "".join(text_parts)
            input_tokens = _rough_token_count("\n".join(item.get("content", "") for item in messages))
            output_tokens = _rough_token_count(output)
            cost = _estimate_cost(deployment, input_tokens, output_tokens)
            self._record_completed(deployment, "generation", context, input_tokens, output_tokens, latency, cost, 0)
            yield ModelStreamEvent(
                "model_completed",
                _generation_public_dict(
                    ModelGenerationResult(output, deployment.id, deployment.provider, deployment.model, "completed", input_tokens, output_tokens, cost)
                ),
            )
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            self._record_failed(deployment, "generation", context, latency, exc, 0)
            raise ModelFarmError(f"Generator streaming failed: {_safe_error(exc)}") from exc

    async def embed(
        self,
        texts: List[str],
        deployment_id: str,
        *,
        context: Optional[ModelCallContext] = None,
        external_processing_allowed: bool = True,
        require_enabled: bool = True,
    ) -> ModelEmbeddingResult:
        deployment = self.service.resolve(deployment_id, "embedding", require_enabled=require_enabled)
        self.service.assert_egress_allowed(deployment, external_processing_allowed=external_processing_allowed, content_kind="Document or query text")
        self.service.assert_budget(deployment)
        started = time.perf_counter()
        try:
            result = await self._embed_once(deployment, texts)
            latency = (time.perf_counter() - started) * 1000
            self._record_completed(deployment, "embedding", context, result.input_tokens, 0, latency, result.estimated_cost_usd, 0)
            return replace(result, metadata={**result.metadata, "latency_ms": round(latency, 3)})
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            self._record_failed(deployment, "embedding", context, latency, exc, 0)
            raise ModelFarmError(f"Embedding execution failed: {_safe_error(exc)}") from exc

    async def rerank(
        self,
        query: str,
        documents: List[str],
        deployment_id: str,
        *,
        top_n: int,
        context: Optional[ModelCallContext] = None,
        external_processing_allowed: bool = True,
    ) -> ModelRerankResult:
        deployment = self.service.resolve(deployment_id, "rerank")
        self.service.assert_egress_allowed(deployment, external_processing_allowed=external_processing_allowed, content_kind="Retrieved context")
        self.service.assert_budget(deployment)
        started = time.perf_counter()
        try:
            result = await self._rerank_once(deployment, query, documents, top_n)
            latency = (time.perf_counter() - started) * 1000
            input_tokens = _rough_token_count(query + "\n" + "\n".join(documents))
            self._record_completed(deployment, "rerank", context, input_tokens, 0, latency, result.estimated_cost_usd, 0)
            return replace(result, metadata={**result.metadata, "latency_ms": round(latency, 3)})
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            self._record_failed(deployment, "rerank", context, latency, exc, 0)
            raise ModelFarmError(f"Reranker execution failed: {_safe_error(exc)}") from exc

    def generate_sync(self, *args: Any, **kwargs: Any) -> ModelGenerationResult:
        return _run_sync(self.generate(*args, **kwargs))

    def embed_sync(self, *args: Any, **kwargs: Any) -> ModelEmbeddingResult:
        return _run_sync(self.embed(*args, **kwargs))

    def rerank_sync(self, *args: Any, **kwargs: Any) -> ModelRerankResult:
        return _run_sync(self.rerank(*args, **kwargs))

    async def test_deployment(self, deployment_id: str) -> Dict[str, Any]:
        deployment = self.service.get_deployment(deployment_id)
        try:
            if "embedding" in deployment.capabilities:
                result = await self.embed(["Adaptive RAG model health check"], deployment.id, require_enabled=False)
                updated = self.service.set_health(deployment.id, healthy=True, dimension=result.dimension)
                return {"status": "healthy", "dimension": result.dimension, "deployment": updated}
            if "rerank" in deployment.capabilities and "generation" not in deployment.capabilities:
                result = await self._rerank_once(deployment, "invoice approval", ["invoice approval workflow", "holiday calendar"], 1)
                updated = self.service.set_health(deployment.id, healthy=bool(result.items))
                return {"status": updated.health_status, "deployment": updated}
            result = await self._generate_once(
                deployment,
                [{"role": "user", "content": "Reply with the word ready."}],
                {"max_tokens": 12},
            )
            updated = self.service.set_health(deployment.id, healthy=bool(result.text.strip()))
            return {"status": updated.health_status, "deployment": updated}
        except Exception as exc:
            updated = self.service.set_health(deployment.id, healthy=False, error=str(exc))
            return {"status": "unavailable", "error": _safe_error(exc), "deployment": updated}

    async def _generate_once(
        self,
        deployment: ModelDeployment,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any],
    ) -> ModelGenerationResult:
        if deployment.provider.lower() == "local" and deployment.model == "extractive":
            prompt = next((item.get("content", "") for item in reversed(messages) if item.get("role") == "user"), "")
            text = prompt.split("Retrieved workflow context:", 1)[-1].strip()
            if "User question:" in text:
                context_text, question = text.split("User question:", 1)
                context_text = context_text.strip()
                question = question.split("Answer:", 1)[0].strip()
            else:
                context_text, question = "", prompt.strip()
            answer = (
                f"Based on the retrieved workflow context: {context_text[:900]}"
                if context_text and "No retrieved workflow context" not in context_text
                else f"Direct local draft for: {question or prompt[:500]}"
            )
            return ModelGenerationResult(
                answer,
                deployment.id,
                deployment.provider,
                deployment.model,
                "completed",
                _rough_token_count(prompt),
                _rough_token_count(answer),
                0.0,
                "stop",
                {"runtime": "deterministic-extractive"},
            )
        if deployment.provider.lower() == "local" and deployment.model == "google/flan-t5-small":
            prompt = "\n".join(item.get("content", "") for item in messages)
            answer = await asyncio.to_thread(_run_local_flan, deployment.model, prompt, parameters)
            return ModelGenerationResult(
                answer,
                deployment.id,
                deployment.provider,
                deployment.model,
                "completed",
                _rough_token_count(prompt),
                _rough_token_count(answer),
                0.0,
                "stop",
                {"runtime": "transformers-text2text"},
            )
        try:
            import litellm  # type: ignore
        except ImportError as exc:
            raise ModelFarmError("Install the models extra to use remote providers: python -m pip install -e \".[models]\".") from exc
        kwargs = self._litellm_kwargs(deployment)
        kwargs.update(deployment.default_parameters)
        kwargs.update(parameters)
        kwargs.update({"model": deployment.model, "messages": messages})
        response = await litellm.acompletion(**kwargs)
        text = _completion_text(response)
        usage = _completion_usage(response)
        input_tokens = usage.get("prompt_tokens") or _rough_token_count("\n".join(item.get("content", "") for item in messages))
        output_tokens = usage.get("completion_tokens") or _rough_token_count(text)
        cost = _response_cost(response) or _estimate_cost(deployment, input_tokens, output_tokens)
        return ModelGenerationResult(
            text,
            deployment.id,
            deployment.provider,
            deployment.model,
            "completed",
            input_tokens,
            output_tokens,
            cost,
            _completion_finish_reason(response),
            {"runtime": "litellm"},
        )

    async def _embed_once(self, deployment: ModelDeployment, texts: List[str]) -> ModelEmbeddingResult:
        if deployment.provider.lower() == "local":
            from aragbiz.knowledge import HashEmbeddingModel, SentenceTransformerEmbeddingModel

            dimension = deployment.dimension or 384
            if deployment.model == "hash-embedding-384":
                embedder = HashEmbeddingModel(dimension=dimension)
            elif deployment.model == "sentence-transformers/all-MiniLM-L6-v2":
                embedder = SentenceTransformerEmbeddingModel(deployment.model, dimension=dimension)
            else:
                raise ModelFarmError(f"Unsupported local embedding deployment: {deployment.model}")
            embeddings = await asyncio.to_thread(embedder.embed, texts)
            return ModelEmbeddingResult(
                embeddings,
                deployment.id,
                deployment.provider,
                deployment.model,
                len(embeddings[0]) if embeddings else dimension,
                _rough_token_count("\n".join(texts)),
                0.0,
                {"runtime": "local"},
            )
        try:
            import litellm  # type: ignore
        except ImportError as exc:
            raise ModelFarmError("Install the models extra to use remote providers: python -m pip install -e \".[models]\".") from exc
        kwargs = self._litellm_kwargs(deployment)
        kwargs.update(deployment.default_parameters)
        kwargs.update({"model": deployment.model, "input": texts})
        response = await litellm.aembedding(**kwargs)
        embeddings = _embedding_vectors(response)
        if not embeddings:
            raise ModelFarmError("The embedding provider returned no vectors.")
        usage = _completion_usage(response)
        input_tokens = usage.get("prompt_tokens") or usage.get("total_tokens") or _rough_token_count("\n".join(texts))
        cost = _response_cost(response) or _estimate_cost(deployment, input_tokens, 0)
        return ModelEmbeddingResult(
            embeddings,
            deployment.id,
            deployment.provider,
            deployment.model,
            len(embeddings[0]),
            input_tokens,
            cost,
            {"runtime": "litellm"},
        )

    async def _rerank_once(
        self,
        deployment: ModelDeployment,
        query: str,
        documents: List[str],
        top_n: int,
    ) -> ModelRerankResult:
        if deployment.provider.lower() == "local":
            query_terms = set(_tokens(query))
            ranked = []
            for index, document in enumerate(documents):
                terms = set(_tokens(document))
                score = len(query_terms & terms) / max(len(query_terms | terms), 1)
                ranked.append(ModelRerankItem(index, score))
            ranked.sort(key=lambda item: item.score, reverse=True)
            return ModelRerankResult(ranked[:top_n], deployment.id, deployment.provider, deployment.model, 0.0, {"runtime": "lexical"})
        try:
            import litellm  # type: ignore
        except ImportError as exc:
            raise ModelFarmError("Install the models extra to use remote rerankers: python -m pip install -e \".[models]\".") from exc
        kwargs = self._litellm_kwargs(deployment)
        kwargs.update(deployment.default_parameters)
        kwargs.update({"model": deployment.model, "query": query, "documents": documents, "top_n": top_n})
        if hasattr(litellm, "arerank"):
            response = await litellm.arerank(**kwargs)
        else:  # pragma: no cover - compatibility with older LiteLLM releases
            response = await asyncio.to_thread(litellm.rerank, **kwargs)
        items = _rerank_items(response)
        input_tokens = _rough_token_count(query + "\n" + "\n".join(documents))
        return ModelRerankResult(items, deployment.id, deployment.provider, deployment.model, _response_cost(response) or _estimate_cost(deployment, input_tokens, 0), {"runtime": "litellm"})

    def _litellm_kwargs(self, deployment: ModelDeployment) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if deployment.api_base:
            kwargs["api_base"] = deployment.api_base
        for key, value in self.service.credential_values(deployment).items():
            if value:
                kwargs[key] = value
        for key, env_name in deployment.credential_env_refs.items():
            if key in kwargs:
                continue
            value = os.getenv(env_name)
            if value:
                kwargs[key] = value
        timeout = deployment.limits.get("timeout_seconds")
        if timeout:
            kwargs["timeout"] = float(timeout)
        return kwargs

    def _record_completed(
        self,
        deployment: ModelDeployment,
        capability: str,
        context: Optional[ModelCallContext],
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost: float,
        fallback_index: int,
    ) -> None:
        self.service.record_usage(
            _usage_event(
                deployment,
                capability,
                context,
                "completed",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost=cost,
                fallback_index=fallback_index,
            )
        )

    def _record_failed(
        self,
        deployment: ModelDeployment,
        capability: str,
        context: Optional[ModelCallContext],
        latency_ms: float,
        exc: Exception,
        fallback_index: int,
    ) -> None:
        self.service.record_usage(
            _usage_event(
                deployment,
                capability,
                context,
                "failed",
                latency_ms=latency_ms,
                fallback_index=fallback_index,
                error=exc,
            )
        )


def model_provider_templates() -> List[Dict[str, Any]]:
    local = [
        _builtin_template("local-extractive", "Local Extractive", "model-local-extractive", "extractive", ["generation", "judge"]),
        _builtin_template("local-flan-t5-small", "Local FLAN-T5 Small", "model-local-flan-t5-small", "google/flan-t5-small", ["generation", "judge", "planner"]),
        _builtin_template("local-hash-384", "Local Hash Embedding 384", "model-local-hash-384", "hash-embedding-384", ["embedding"]),
        _builtin_template("local-minilm-384", "Local MiniLM Embedding 384", "model-local-minilm-384", "sentence-transformers/all-MiniLM-L6-v2", ["embedding"]),
        _builtin_template("local-lexical-reranker", "Local Lexical Reranker", "model-local-lexical-reranker", "lexical-overlap", ["rerank"]),
    ]
    remote = [
        _provider_template(
            "openai-generation",
            "OpenAI generation / judge",
            "OpenAI",
            "openai",
            "gpt-4.1-mini",
            ["generation", "judge", "planner"],
            {"api_key": "ARAGBIZ_MODEL_OPENAI_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 64000, "max_output_tokens": 1200, "timeout_seconds": 60},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
        ),
        _provider_template(
            "openai-embedding",
            "OpenAI embedding",
            "OpenAI",
            "openai",
            "text-embedding-3-small",
            ["embedding"],
            {"api_key": "ARAGBIZ_MODEL_OPENAI_API_KEY"},
            {},
            {"dimension": 1536, "timeout_seconds": 60},
            {"input_per_million_tokens_usd": 0},
        ),
        _provider_template(
            "azure-openai-generation",
            "Azure OpenAI generation / judge",
            "Azure OpenAI",
            "azure",
            "azure/gpt-4o-mini",
            ["generation", "judge", "planner"],
            {"api_key": "ARAGBIZ_MODEL_AZURE_OPENAI_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 64000, "max_output_tokens": 1200, "timeout_seconds": 60},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
            api_base="https://YOUR-RESOURCE.openai.azure.com",
        ),
        _provider_template(
            "openai-compatible",
            "OpenAI-compatible endpoint",
            "OpenAI-compatible",
            "custom",
            "openai/gpt-4.1-mini",
            ["generation", "embedding", "judge", "planner"],
            {"api_key": "ARAGBIZ_MODEL_CUSTOM_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 32000, "max_output_tokens": 1200, "timeout_seconds": 90},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
            api_base="https://api.openai.com/v1",
        ),
        _provider_template(
            "cohere-generation-rerank",
            "Cohere generation / rerank",
            "Cohere",
            "cohere",
            "command-r",
            ["generation", "rerank", "judge"],
            {"api_key": "ARAGBIZ_MODEL_COHERE_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 120000, "max_output_tokens": 1200, "timeout_seconds": 60},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
        ),
        _provider_template(
            "huggingface-generation",
            "Hugging Face hosted generation",
            "Hugging Face",
            "huggingface",
            "huggingface/mistralai/Mistral-7B-Instruct-v0.3",
            ["generation"],
            {"api_key": "ARAGBIZ_MODEL_HUGGINGFACE_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 32000, "max_output_tokens": 1200, "timeout_seconds": 90},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
        ),
        _provider_template(
            "bedrock-generation",
            "Amazon Bedrock generation / judge",
            "Amazon Bedrock",
            "bedrock",
            "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
            ["generation", "judge", "planner"],
            {},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 200000, "max_output_tokens": 1200, "timeout_seconds": 90},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
            metadata={"auth_mode": "ambient"},
        ),
        _provider_template(
            "mistral-generation",
            "Mistral generation",
            "Mistral",
            "mistral",
            "mistral/mistral-small-latest",
            ["generation", "judge", "planner"],
            {"api_key": "ARAGBIZ_MODEL_MISTRAL_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 32000, "max_output_tokens": 1200, "timeout_seconds": 60},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
        ),
        _provider_template(
            "gemini-generation",
            "Gemini generation",
            "Gemini",
            "gemini",
            "gemini/gemini-1.5-flash",
            ["generation", "judge", "planner"],
            {"api_key": "ARAGBIZ_MODEL_GEMINI_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 1000000, "max_output_tokens": 1200, "timeout_seconds": 60},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
        ),
        _provider_template(
            "anthropic-generation",
            "Anthropic generation / judge",
            "Anthropic",
            "anthropic",
            "anthropic/claude-3-5-haiku-latest",
            ["generation", "judge", "planner"],
            {"api_key": "ARAGBIZ_MODEL_ANTHROPIC_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 200000, "max_output_tokens": 1200, "timeout_seconds": 60},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
        ),
        _provider_template(
            "databricks-generation",
            "Databricks generation",
            "Databricks",
            "databricks",
            "databricks/databricks-meta-llama-3-1-70b-instruct",
            ["generation", "judge"],
            {"api_key": "ARAGBIZ_MODEL_DATABRICKS_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 32000, "max_output_tokens": 1200, "timeout_seconds": 90},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
            api_base="https://YOUR-WORKSPACE.cloud.databricks.com/serving-endpoints",
        ),
        _provider_template(
            "ai21-generation",
            "AI21 generation",
            "AI21",
            "ai21",
            "ai21/jamba-mini",
            ["generation", "judge"],
            {"api_key": "ARAGBIZ_MODEL_AI21_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 256000, "max_output_tokens": 1200, "timeout_seconds": 60},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
        ),
        _provider_template(
            "watsonx-generation",
            "IBM watsonx.ai generation",
            "IBM watsonx.ai",
            "watsonx",
            "watsonx/meta-llama/llama-3-2-90b-vision-instruct",
            ["generation", "judge"],
            {"api_key": "ARAGBIZ_MODEL_WATSONX_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 32000, "max_output_tokens": 1200, "timeout_seconds": 90},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
            api_base="https://YOUR-REGION.ml.cloud.ibm.com",
        ),
    ]
    return local + remote


def _builtin_template(template_id: str, label: str, deployment_id: str, model: str, capabilities: List[str]) -> Dict[str, Any]:
    return {
        "id": template_id,
        "label": label,
        "provider_label": "Local built-in",
        "provider": "Local",
        "model": model,
        "capabilities": capabilities,
        "locality": "local",
        "builtin_deployment_id": deployment_id,
        "creatable": False,
        "credential_fields": [],
        "notes": "Built-in deployment; already registered by the backend.",
        "deployment_defaults": {
            "name": label,
            "provider": "Local",
            "model": model,
            "capabilities": capabilities,
            "locality": "local",
            "enabled": True,
            "health_status": "healthy",
            "metadata": {"builtin": True, "template_id": template_id},
        },
    }


def _provider_template(
    template_id: str,
    label: str,
    provider_label: str,
    provider: str,
    model: str,
    capabilities: List[str],
    credential_env_refs: Dict[str, str],
    default_parameters: Dict[str, Any],
    limits: Dict[str, Any],
    pricing: Dict[str, Any],
    *,
    api_base: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": template_id,
        "label": label,
        "provider_label": provider_label,
        "provider": provider,
        "model": model,
        "capabilities": capabilities,
        "locality": "remote",
        "creatable": True,
        "credential_fields": sorted(credential_env_refs.keys()),
        "notes": "Remote deployment; test successfully before enabling.",
        "deployment_defaults": {
            "name": label,
            "provider": provider,
            "model": model,
            "capabilities": capabilities,
            "api_base": api_base,
            "credential_env_refs": credential_env_refs,
            "default_parameters": default_parameters,
            "limits": limits,
            "pricing": pricing,
            "monthly_budget_usd": 5,
            "hard_budget": True,
            "locality": "remote",
            "enabled": False,
            "health_status": "untested",
            "metadata": metadata or {},
        },
    }


def _template_by_id(template_id: str) -> Optional[Dict[str, Any]]:
    return next((template for template in model_provider_templates() if template["id"] == template_id), None)


def builtin_model_deployments() -> List[ModelDeployment]:
    now = utc_now()
    common = {"enabled": True, "health_status": "healthy", "locality": "local", "metadata": {"builtin": True}, "created_at": now, "updated_at": now}
    return [
        ModelDeployment("model-local-extractive", "Local Extractive", "Local", "extractive", ["generation", "judge"], limits={"context_window": 3200, "max_output_tokens": 900}, **common),
        ModelDeployment("model-local-flan-t5-small", "Local FLAN-T5 Small", "Local", "google/flan-t5-small", ["generation", "judge", "planner"], limits={"context_window": 512, "max_output_tokens": 160}, **common),
        ModelDeployment("model-local-hash-384", "Local Hash Embedding 384", "Local", "hash-embedding-384", ["embedding"], limits={"dimension": 384, "batch_size": 256}, **common),
        ModelDeployment("model-local-minilm-384", "Local MiniLM Embedding 384", "Local", "sentence-transformers/all-MiniLM-L6-v2", ["embedding"], limits={"dimension": 384, "batch_size": 64}, **common),
        ModelDeployment("model-local-lexical-reranker", "Local Lexical Reranker", "Local", "lexical-overlap", ["rerank"], **common),
        ModelDeployment("model-local-distilbert", "Local DistilBERT Classifier", "Local", "query_classifier_distilbert", ["classifier"], **common),
        ModelDeployment("model-local-t5-classifier", "Local T5-small Classifier", "Local", "query_classifier_t5", ["classifier"], **common),
    ]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validated_deployment(deployment: ModelDeployment) -> ModelDeployment:
    if not deployment.name:
        raise ModelFarmError("Deployment name is required.")
    if not deployment.provider:
        raise ModelFarmError("Deployment provider is required.")
    if not deployment.model:
        raise ModelFarmError("Deployment model identifier is required.")
    capabilities = sorted(set(str(item).strip() for item in deployment.capabilities if str(item).strip()))
    invalid = sorted(set(capabilities) - MODEL_CAPABILITIES)
    if invalid:
        raise ModelFarmError(f"Unsupported model capabilities: {', '.join(invalid)}")
    if not capabilities:
        raise ModelFarmError("Select at least one model capability.")
    refs: Dict[str, str] = {}
    for key, env_name in deployment.credential_env_refs.items():
        clean_key = str(key).strip()
        clean_name = str(env_name).strip()
        if clean_name and not clean_name.startswith(SECRET_ENV_PREFIX):
            raise ModelFarmError(f"Credential reference '{clean_name}' must start with {SECRET_ENV_PREFIX}.")
        if clean_key and clean_name:
            refs[clean_key] = clean_name
    secrets = {
        str(key).strip(): str(value)
        for key, value in deployment.credential_secrets.items()
        if str(key).strip() and str(value)
    }
    locality = deployment.locality if deployment.locality in {"local", "remote"} else "remote"
    if "embedding" in capabilities:
        dimension = deployment.limits.get("dimension")
        if dimension is not None and int(dimension) <= 0:
            raise ModelFarmError("Embedding dimension must be positive.")
    return replace(
        deployment,
        capabilities=capabilities,
        credential_env_refs=refs,
        credential_secrets=secrets,
        locality=locality,
        monthly_budget_usd=max(float(deployment.monthly_budget_usd), 0.0),
    )


def _deployment_from_dict(payload: Dict[str, Any]) -> ModelDeployment:
    allowed = {field.name for field in ModelDeployment.__dataclass_fields__.values()}
    return ModelDeployment(**{key: value for key, value in payload.items() if key in allowed})


def _usage_from_dict(payload: Dict[str, Any]) -> ModelUsageEvent:
    allowed = {field.name for field in ModelUsageEvent.__dataclass_fields__.values()}
    return ModelUsageEvent(**{key: value for key, value in payload.items() if key in allowed})


def _deployment_from_row(row: Any) -> ModelDeployment:
    return ModelDeployment(
        id=row["id"], name=row["name"], provider=row["provider"], model=row["model"],
        capabilities=list(row.get("capabilities_json") or []), api_base=row.get("api_base") or "",
        credential_env_refs=dict(row.get("credential_env_refs_json") or {}),
        credential_secrets=dict(row.get("credential_secrets_json") or {}),
        default_parameters=dict(row.get("default_parameters_json") or {}), limits=dict(row.get("limits_json") or {}),
        pricing=dict(row.get("pricing_json") or {}), monthly_budget_usd=float(row.get("monthly_budget_usd") or 0),
        hard_budget=bool(row.get("hard_budget", True)), locality=row.get("locality") or "remote",
        enabled=bool(row.get("enabled", False)), health_status=row.get("health_status") or "untested",
        last_health_check=row.get("last_health_check") or "", last_error=row.get("last_error") or "",
        metadata=dict(row.get("metadata_json") or {}), created_at=row.get("created_at") or "", updated_at=row.get("updated_at") or "",
    )


def _usage_from_row(row: Any) -> ModelUsageEvent:
    return ModelUsageEvent(
        id=row["id"], deployment_id=row["deployment_id"], provider=row["provider"], model=row["model"],
        capability=row["capability"], purpose=row["purpose"], status=row["status"], request_id=row.get("request_id") or "",
        user_id=row.get("user_id") or "", conversation_id=row.get("conversation_id") or "",
        knowledge_base_id=row.get("knowledge_base_id") or "", evaluation_run_id=row.get("evaluation_run_id") or "",
        input_tokens=int(row.get("input_tokens") or 0), output_tokens=int(row.get("output_tokens") or 0),
        total_tokens=int(row.get("total_tokens") or 0), latency_ms=float(row.get("latency_ms") or 0),
        estimated_cost_usd=float(row.get("estimated_cost_usd") or 0), fallback_index=int(row.get("fallback_index") or 0),
        error_code=row.get("error_code") or "", error=row.get("error") or "", metadata=dict(row.get("metadata_json") or {}),
        created_at=row.get("created_at") or "",
    )


def _deployment_db_payload(deployment: ModelDeployment) -> Dict[str, Any]:
    payload = asdict(deployment)
    for key in ["capabilities", "credential_env_refs", "credential_secrets", "default_parameters", "limits", "pricing", "metadata"]:
        payload[key] = json.dumps(payload[key])
    return payload


def _secret_key_bytes(secret_key: str) -> bytes:
    return hashlib.sha256((secret_key or "aragbiz-local-development-secret").encode("utf-8")).digest()


def _secret_stream(key: bytes, nonce: bytes, length: int) -> bytes:
    blocks: List[bytes] = []
    counter = 0
    while sum(len(block) for block in blocks) < length:
        blocks.append(hashlib.sha256(key + nonce + counter.to_bytes(4, "big")).digest())
        counter += 1
    return b"".join(blocks)[:length]


def _encrypt_secret(value: str, secret_key: str) -> str:
    plaintext = value.encode("utf-8")
    key = _secret_key_bytes(secret_key)
    nonce = os.urandom(16)
    stream = _secret_stream(key, nonce, len(plaintext))
    ciphertext = bytes(left ^ right for left, right in zip(plaintext, stream))
    tag = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()
    parts = [
        base64.urlsafe_b64encode(item).decode("ascii").rstrip("=")
        for item in (nonce, ciphertext, tag)
    ]
    return "v1:" + ":".join(parts)


def _decrypt_secret(token: str, secret_key: str) -> str:
    if not token.startswith("v1:"):
        return token
    try:
        _, nonce_raw, ciphertext_raw, tag_raw = token.split(":", 3)
        nonce = _urlsafe_b64decode(nonce_raw)
        ciphertext = _urlsafe_b64decode(ciphertext_raw)
        tag = _urlsafe_b64decode(tag_raw)
    except ValueError as exc:
        raise ModelFarmError("Stored model credential is malformed.") from exc
    key = _secret_key_bytes(secret_key)
    expected = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise ModelFarmError("Stored model credential could not be decrypted with the configured key.")
    stream = _secret_stream(key, nonce, len(ciphertext))
    plaintext = bytes(left ^ right for left, right in zip(ciphertext, stream))
    return plaintext.decode("utf-8")


def _urlsafe_b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _usage_event(
    deployment: ModelDeployment,
    capability: str,
    context: Optional[ModelCallContext],
    status: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    latency_ms: float = 0.0,
    cost: float = 0.0,
    fallback_index: int = 0,
    error: Optional[Exception] = None,
) -> ModelUsageEvent:
    context = context or ModelCallContext(purpose=capability)
    return ModelUsageEvent(
        id=f"usage-{uuid.uuid4().hex}", deployment_id=deployment.id, provider=deployment.provider, model=deployment.model,
        capability=capability, purpose=context.purpose or capability, status=status, request_id=context.request_id,
        user_id=context.user_id, conversation_id=context.conversation_id, knowledge_base_id=context.knowledge_base_id,
        evaluation_run_id=context.evaluation_run_id, input_tokens=input_tokens, output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens, latency_ms=round(latency_ms, 3), estimated_cost_usd=round(cost, 10),
        fallback_index=fallback_index, error_code=type(error).__name__ if error else "", error=_safe_error(error) if error else "",
        created_at=utc_now(),
    )


def _estimate_cost(deployment: ModelDeployment, input_tokens: int, output_tokens: int) -> float:
    input_rate = float(deployment.pricing.get("input_per_million_tokens_usd") or 0.0)
    output_rate = float(deployment.pricing.get("output_per_million_tokens_usd") or 0.0)
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _add_usage_bucket(target: Dict[str, Dict[str, Any]], key: str, event: ModelUsageEvent) -> None:
    bucket = target.setdefault(key or "unknown", {"calls": 0, "tokens": 0, "cost_usd": 0.0, "latency_ms": 0.0})
    bucket["calls"] += 1
    bucket["tokens"] += event.total_tokens
    bucket["cost_usd"] = round(bucket["cost_usd"] + event.estimated_cost_usd, 8)
    bucket["latency_ms"] = round(bucket["latency_ms"] + event.latency_ms, 3)


def _completion_text(response: Any) -> str:
    try:
        return str(response.choices[0].message.content or "")
    except (AttributeError, IndexError, KeyError, TypeError):
        payload = response if isinstance(response, dict) else getattr(response, "model_dump", lambda: {})()
        choices = payload.get("choices") or []
        return str(((choices[0].get("message") or {}).get("content")) if choices else "")


def _completion_finish_reason(response: Any) -> str:
    try:
        return str(response.choices[0].finish_reason or "")
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""


def _completion_usage(response: Any) -> Dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return {}
    return {key: int(value or 0) for key, value in usage.items() if key in {"prompt_tokens", "completion_tokens", "total_tokens"}}


def _response_cost(response: Any) -> float:
    hidden = getattr(response, "_hidden_params", None)
    if isinstance(hidden, dict):
        try:
            return float(hidden.get("response_cost") or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _embedding_vectors(response: Any) -> List[List[float]]:
    data = getattr(response, "data", None)
    if data is None and isinstance(response, dict):
        data = response.get("data")
    vectors: List[List[float]] = []
    for item in data or []:
        embedding = getattr(item, "embedding", None)
        if embedding is None and isinstance(item, dict):
            embedding = item.get("embedding")
        if embedding is not None:
            vectors.append([float(value) for value in embedding])
    return vectors


def _rerank_items(response: Any) -> List[ModelRerankItem]:
    results = getattr(response, "results", None)
    if results is None and isinstance(response, dict):
        results = response.get("results") or response.get("data")
    items: List[ModelRerankItem] = []
    for item in results or []:
        if isinstance(item, dict):
            index = item.get("index", 0)
            score = item.get("relevance_score", item.get("score", 0.0))
        else:
            index = getattr(item, "index", 0)
            score = getattr(item, "relevance_score", getattr(item, "score", 0.0))
        items.append(ModelRerankItem(int(index), float(score)))
    return items


def _stream_text(chunk: Any) -> str:
    try:
        return str(chunk.choices[0].delta.content or "")
    except (AttributeError, IndexError, KeyError, TypeError):
        return ""


def _generation_public_dict(result: ModelGenerationResult) -> Dict[str, Any]:
    return {
        "deployment_id": result.deployment_id, "provider": result.provider, "model": result.model,
        "status": result.status, "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "estimated_cost_usd": result.estimated_cost_usd, "finish_reason": result.finish_reason, "metadata": result.metadata,
    }


def _safe_error(error: Any) -> str:
    if error is None:
        return "unknown error"
    text = str(error)
    for env_name, value in os.environ.items():
        if env_name.startswith(SECRET_ENV_PREFIX) and value:
            text = text.replace(value, "[REDACTED]")
    return text[:1000]


def _is_retryable_error(error: Exception) -> bool:
    name = type(error).__name__.lower()
    text = str(error).lower()
    permanent = ("authentication", "permission", "invalidrequest", "badrequest", "budget", "policy")
    return not any(item in name or item in text for item in permanent)


def _rough_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _tokens(text: str) -> List[str]:
    return [part for part in "".join(char.lower() if char.isalnum() else " " for char in text).split() if part]


def _text_chunks(text: str, size: int) -> Iterable[str]:
    for index in range(0, len(text), max(size, 1)):
        yield text[index : index + size]


_LOCAL_FLAN_CACHE: Dict[str, tuple[Any, Any]] = {}


def _load_local_flan(model_name: str) -> tuple[Any, Any]:
    if model_name not in _LOCAL_FLAN_CACHE:
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise ModelFarmError("Install the ml extra to run local FLAN-T5.") from exc
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _LOCAL_FLAN_CACHE[model_name] = (tokenizer, model)
    return _LOCAL_FLAN_CACHE[model_name]


async def _stream_local_flan(model_name: str, prompt: str, parameters: Dict[str, Any]) -> AsyncIterator[str]:
    try:
        from transformers import TextIteratorStreamer  # type: ignore
    except ImportError as exc:
        raise ModelFarmError("Install the ml extra to stream local FLAN-T5.") from exc
    tokenizer, model = await asyncio.to_thread(_load_local_flan, model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    errors: List[BaseException] = []

    def run_generate() -> None:
        try:
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=int(parameters.get("max_tokens") or parameters.get("max_new_tokens") or 160),
            )
        except BaseException as exc:  # pragma: no cover - optional dependency failures vary by platform
            errors.append(exc)
            if hasattr(streamer, "on_finalized_text"):
                streamer.on_finalized_text("", stream_end=True)

    thread = threading.Thread(target=run_generate, daemon=True)
    thread.start()
    while thread.is_alive() or not errors:
        yielded = False
        for text in streamer:
            yielded = True
            yield str(text)
            await asyncio.sleep(0)
        if not thread.is_alive():
            break
        if not yielded:
            await asyncio.sleep(0.01)
    thread.join(timeout=0.1)
    if errors:
        raise ModelFarmError(str(errors[0])) from errors[0]


def _run_local_flan(model_name: str, prompt: str, parameters: Dict[str, Any]) -> str:
    tokenizer, model = _load_local_flan(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(
        **inputs,
        max_new_tokens=int(parameters.get("max_tokens") or parameters.get("max_new_tokens") or 160),
    )
    return str(tokenizer.decode(outputs[0], skip_special_tokens=True))


def _run_sync(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise ModelFarmError("A synchronous Model Gateway call cannot run inside an active event loop; use the async method.")
