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
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Protocol, Sequence

from aragbiz.cancellation import AnswerCancelled, CancellationToken


MODEL_CAPABILITIES = {"generation", "embedding", "rerank", "judge", "planner", "classifier"}
SECRET_ENV_PREFIX = "ARAGBIZ_MODEL_"
MODEL_CONNECTION_PROVIDERS = {"local_builtin", "openrouter", "openai", "gemini", "ollama", "vllm"}
MODEL_ACCESS_PATHS = {"experimentation", "production", "local"}
_KNOWN_SECRET_VALUES: set[str] = set()


class ModelFarmError(ValueError):
    """Raised when a model deployment cannot be configured or executed."""


class ModelBudgetExceeded(ModelFarmError):
    """Raised before a call that would exceed a configured hard budget."""


class ModelPolicyError(ModelFarmError):
    """Raised when a model call violates data-egress policy."""


@dataclass(frozen=True)
class ModelConnection:
    id: str
    name: str
    provider: str
    access_path: str
    api_base: str = ""
    credential_env_refs: Dict[str, str] = field(default_factory=dict)
    credential_secrets: Dict[str, str] = field(default_factory=dict)
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
        return self.locality == "local" or self.provider in {"local_builtin", "ollama", "vllm"}


@dataclass(frozen=True)
class ModelDeployment:
    id: str
    name: str
    provider: str
    model: str
    capabilities: List[str]
    connection_id: str = ""
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
    chat_configuration_id: str = ""


@dataclass(frozen=True)
class ModelUsageEvent:
    id: str
    deployment_id: str
    provider: str
    model: str
    capability: str
    purpose: str
    status: str
    connection_id: str = ""
    access_path: str = ""
    gateway_model: str = ""
    request_id: str = ""
    user_id: str = ""
    conversation_id: str = ""
    knowledge_base_id: str = ""
    evaluation_run_id: str = ""
    chat_configuration_id: str = ""
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
class ModelClassificationResult:
    label: str
    deployment_id: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    margin: float = 1.0
    supported_labels: List[str] = field(default_factory=list)
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

    def list_connections(self) -> List[ModelConnection]: ...

    def get_connection(self, connection_id: str) -> ModelConnection: ...

    def save_connection(self, connection: ModelConnection) -> ModelConnection: ...

    def delete_connection(self, connection_id: str) -> None: ...

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
            self._write({"connections": {}, "deployments": {}, "usage": []})

    def list_connections(self) -> List[ModelConnection]:
        state = self._read()
        items = [_connection_from_dict(value) for value in state["connections"].values()]
        return sorted(items, key=lambda item: (item.name.lower(), item.id))

    def get_connection(self, connection_id: str) -> ModelConnection:
        state = self._read()
        payload = state["connections"].get(connection_id)
        if not payload:
            raise KeyError(f"Model connection not found: {connection_id}")
        return _connection_from_dict(payload)

    def save_connection(self, connection: ModelConnection) -> ModelConnection:
        state = self._read()
        state["connections"][connection.id] = asdict(connection)
        self._write(state)
        return connection

    def delete_connection(self, connection_id: str) -> None:
        state = self._read()
        if connection_id not in state["connections"]:
            raise KeyError(f"Model connection not found: {connection_id}")
        if any(item.get("connection_id") == connection_id for item in state["deployments"].values()):
            raise ModelFarmError("Connection is referenced by one or more deployments; delete or move them first.")
        del state["connections"][connection_id]
        self._write(state)

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
        state["usage"] = [item for item in state["usage"] if item.get("deployment_id") != deployment_id]
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
        state.setdefault("connections", {})
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
        CREATE TABLE IF NOT EXISTS model_connections (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            provider TEXT NOT NULL,
            access_path TEXT NOT NULL,
            api_base TEXT NOT NULL DEFAULT '',
            credential_env_refs_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            credential_secrets_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            locality TEXT NOT NULL DEFAULT 'remote',
            enabled BOOLEAN NOT NULL DEFAULT FALSE,
            health_status TEXT NOT NULL DEFAULT 'untested',
            last_health_check TEXT NOT NULL DEFAULT '',
            last_error TEXT NOT NULL DEFAULT '',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_deployments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            capabilities_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            connection_id TEXT NOT NULL REFERENCES model_connections(id) ON DELETE RESTRICT,
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
            connection_id TEXT NOT NULL DEFAULT '',
            access_path TEXT NOT NULL DEFAULT '',
            gateway_model TEXT NOT NULL DEFAULT '',
            capability TEXT NOT NULL,
            purpose TEXT NOT NULL,
            status TEXT NOT NULL,
            request_id TEXT NOT NULL DEFAULT '',
            user_id TEXT NOT NULL DEFAULT '',
            conversation_id TEXT NOT NULL DEFAULT '',
            knowledge_base_id TEXT NOT NULL DEFAULT '',
            evaluation_run_id TEXT NOT NULL DEFAULT '',
            chat_configuration_id TEXT NOT NULL DEFAULT '',
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
        ALTER TABLE model_deployments
            ADD COLUMN IF NOT EXISTS connection_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE model_usage_events
            ADD COLUMN IF NOT EXISTS connection_id TEXT NOT NULL DEFAULT '';
        ALTER TABLE model_usage_events
            ADD COLUMN IF NOT EXISTS access_path TEXT NOT NULL DEFAULT '';
        ALTER TABLE model_usage_events
            ADD COLUMN IF NOT EXISTS gateway_model TEXT NOT NULL DEFAULT '';
        ALTER TABLE model_usage_events
            ADD COLUMN IF NOT EXISTS chat_configuration_id TEXT NOT NULL DEFAULT '';
        CREATE INDEX IF NOT EXISTS idx_model_usage_configuration
            ON model_usage_events(chat_configuration_id, created_at DESC);
        """
        with self.engine.begin() as connection:
            for statement in [part.strip() for part in ddl.split(";") if part.strip()]:
                connection.exec_driver_sql(statement)

    def list_connections(self) -> List[ModelConnection]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(text("SELECT * FROM model_connections ORDER BY name, id")).mappings()
            return [_connection_from_row(row) for row in rows]

    def get_connection(self, connection_id: str) -> ModelConnection:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            row = connection.execute(
                text("SELECT * FROM model_connections WHERE id = :id"), {"id": connection_id}
            ).mappings().first()
        if not row:
            raise KeyError(f"Model connection not found: {connection_id}")
        return _connection_from_row(row)

    def save_connection(self, model_connection: ModelConnection) -> ModelConnection:
        from sqlalchemy import text

        payload = _connection_db_payload(model_connection)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO model_connections (
                        id, name, provider, access_path, api_base, credential_env_refs_json,
                        credential_secrets_json, locality, enabled, health_status,
                        last_health_check, last_error, metadata_json, created_at, updated_at
                    ) VALUES (
                        :id, :name, :provider, :access_path, :api_base, CAST(:credential_env_refs AS JSONB),
                        CAST(:credential_secrets AS JSONB), :locality, :enabled, :health_status,
                        :last_health_check, :last_error, CAST(:metadata AS JSONB), :created_at, :updated_at
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name, provider = EXCLUDED.provider,
                        access_path = EXCLUDED.access_path, api_base = EXCLUDED.api_base,
                        credential_env_refs_json = EXCLUDED.credential_env_refs_json,
                        credential_secrets_json = EXCLUDED.credential_secrets_json,
                        locality = EXCLUDED.locality, enabled = EXCLUDED.enabled,
                        health_status = EXCLUDED.health_status,
                        last_health_check = EXCLUDED.last_health_check,
                        last_error = EXCLUDED.last_error, metadata_json = EXCLUDED.metadata_json,
                        updated_at = EXCLUDED.updated_at
                    """
                ),
                payload,
            )
        return model_connection

    def delete_connection(self, connection_id: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            used = connection.execute(
                text("SELECT 1 FROM model_deployments WHERE connection_id = :id LIMIT 1"), {"id": connection_id}
            ).first()
            if used:
                raise ModelFarmError("Connection is referenced by one or more deployments; delete or move them first.")
            result = connection.execute(text("DELETE FROM model_connections WHERE id = :id"), {"id": connection_id})
            if result.rowcount == 0:
                raise KeyError(f"Model connection not found: {connection_id}")

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
                        id, name, provider, model, capabilities_json, connection_id, api_base,
                        credential_env_refs_json, credential_secrets_json, default_parameters_json, limits_json,
                        pricing_json, monthly_budget_usd, hard_budget, locality, enabled,
                        health_status, last_health_check, last_error, metadata_json,
                        created_at, updated_at
                    ) VALUES (
                        :id, :name, :provider, :model, CAST(:capabilities AS JSONB), :connection_id, :api_base,
                        CAST(:credential_env_refs AS JSONB), CAST(:credential_secrets AS JSONB),
                        CAST(:default_parameters AS JSONB), CAST(:limits AS JSONB),
                        CAST(:pricing AS JSONB), :monthly_budget_usd, :hard_budget, :locality, :enabled,
                        :health_status, :last_health_check, :last_error, CAST(:metadata AS JSONB),
                        :created_at, :updated_at
                    )
                    ON CONFLICT (id) DO UPDATE SET
                        name = EXCLUDED.name, provider = EXCLUDED.provider, model = EXCLUDED.model,
                        capabilities_json = EXCLUDED.capabilities_json, connection_id = EXCLUDED.connection_id,
                        api_base = EXCLUDED.api_base,
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
            connection.execute(text("DELETE FROM model_usage_events WHERE deployment_id = :id"), {"id": deployment_id})
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
                        id, deployment_id, provider, model, connection_id, access_path, gateway_model,
                        capability, purpose, status,
                        request_id, user_id, conversation_id, knowledge_base_id, evaluation_run_id, chat_configuration_id,
                        input_tokens, output_tokens, total_tokens, latency_ms, estimated_cost_usd,
                        fallback_index, error_code, error, metadata_json, created_at
                    ) VALUES (
                        :id, :deployment_id, :provider, :model, :connection_id, :access_path, :gateway_model,
                        :capability, :purpose, :status,
                        :request_id, :user_id, :conversation_id, :knowledge_base_id, :evaluation_run_id, :chat_configuration_id,
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
        self._secret_key = secret_key or os.getenv("ARAGBIZ_MODEL_SECRET_KEY") or ""
        self.repository.initialize()
        self._seed_local_connection()
        self._seed_local_deployments()
        self._backfill_legacy_connections()

    def providers(self) -> List[Dict[str, Any]]:
        return model_provider_templates()

    def list_connections(self, *, provider: str = "", enabled: Optional[bool] = None) -> List[ModelConnection]:
        connections = self.repository.list_connections()
        if provider:
            connections = [item for item in connections if item.provider == provider.strip().lower()]
        if enabled is not None:
            connections = [item for item in connections if item.enabled is enabled]
        return connections

    def get_connection(self, connection_id: str) -> ModelConnection:
        return self.repository.get_connection(connection_id)

    def create_connection(self, payload: Dict[str, Any]) -> ModelConnection:
        now = utc_now()
        provider = str(payload.get("provider") or "").strip().lower()
        connection = _validated_connection(
            ModelConnection(
                id=str(payload.get("id") or f"connection-{uuid.uuid4().hex}"),
                name=str(payload.get("name") or "").strip(),
                provider=provider,
                access_path=str(payload.get("access_path") or _default_access_path(provider)).strip().lower(),
                api_base=str(payload.get("api_base") or _default_api_base(provider)).strip(),
                credential_env_refs=dict(payload.get("credential_env_refs") or {}),
                credential_secrets=self._encrypted_secret_payload(payload.get("credential_secrets") or {}),
                locality=str(payload.get("locality") or _default_locality(provider)),
                enabled=bool(payload.get("enabled", False)),
                health_status=str(payload.get("health_status") or "untested"),
                metadata=dict(payload.get("metadata") or {}),
                created_at=now,
                updated_at=now,
            )
        )
        if connection.enabled and not connection.is_local and connection.health_status != "healthy":
            raise ModelFarmError("Test a remote connection successfully before enabling it.")
        self._ensure_unique_connection_name(connection.name)
        return self.repository.save_connection(connection)

    def update_connection(self, connection_id: str, payload: Dict[str, Any]) -> ModelConnection:
        current = self.get_connection(connection_id)
        if current.metadata.get("builtin"):
            raise ModelFarmError("The built-in local connection cannot be modified.")
        for key in {"provider", "access_path", "locality"}:
            if key in payload and str(payload[key]).strip().lower() != getattr(current, key):
                raise ModelFarmError(f"{key} is immutable after connection registration.")
        runtime_changed = _connection_runtime_changed(current, payload)
        updated = _validated_connection(
            replace(
                current,
                name=str(payload.get("name", current.name)).strip(),
                api_base=str(payload.get("api_base", current.api_base)).strip(),
                credential_env_refs=dict(payload.get("credential_env_refs", current.credential_env_refs)),
                credential_secrets=self._updated_secret_payload(
                    current.credential_secrets,
                    payload.get("credential_secrets") if "credential_secrets" in payload else None,
                ),
                enabled=bool(payload.get("enabled", False if runtime_changed else current.enabled)),
                metadata=dict(payload.get("metadata", current.metadata)),
                health_status="untested" if runtime_changed else current.health_status,
                last_error="" if runtime_changed else current.last_error,
                updated_at=utc_now(),
            )
        )
        if updated.enabled and not updated.is_local and updated.health_status != "healthy":
            raise ModelFarmError("Test a remote connection successfully before enabling it.")
        self._ensure_unique_connection_name(updated.name, exclude_id=connection_id)
        return self.repository.save_connection(updated)

    def delete_connection(self, connection_id: str) -> None:
        connection = self.get_connection(connection_id)
        if connection.metadata.get("builtin"):
            raise ModelFarmError("The built-in local connection cannot be deleted.")
        self.repository.delete_connection(connection_id)

    def connection_for_deployment(self, deployment: ModelDeployment, *, require_enabled: bool = True) -> ModelConnection:
        connection = (
            self.get_connection(deployment.connection_id)
            if deployment.connection_id
            else _legacy_connection_for_deployment(deployment)
        )
        if deployment.api_base or deployment.credential_env_refs or deployment.credential_secrets:
            connection = replace(
                connection,
                api_base=deployment.api_base or connection.api_base,
                credential_env_refs=deployment.credential_env_refs or connection.credential_env_refs,
                credential_secrets=deployment.credential_secrets or connection.credential_secrets,
            )
        if require_enabled and not connection.enabled:
            raise ModelFarmError(f"Connection '{connection.name}' is disabled.")
        missing = self.missing_credentials(connection)
        if missing:
            raise ModelFarmError(f"Connection '{connection.name}' is missing credentials: {', '.join(missing)}.")
        return connection

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
                connection_id=str(payload.get("connection_id") or ""),
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
        if not deployment.connection_id:
            if _connection_provider(deployment) == "local_builtin":
                deployment = replace(deployment, connection_id="connection-local-builtin")
            else:
                connection = self.create_connection(
                    {
                        "name": self._unique_connection_name(f"{deployment.name} connection"),
                        "provider": _connection_provider(deployment),
                        "access_path": _default_access_path(_connection_provider(deployment)),
                        "api_base": deployment.api_base,
                        "credential_env_refs": deployment.credential_env_refs,
                        "credential_secrets": self.credential_values(deployment),
                        "locality": _default_locality(_connection_provider(deployment)),
                        "enabled": deployment.enabled,
                        "health_status": deployment.health_status,
                        "metadata": {"created_from_legacy_deployment": True},
                    }
                )
                deployment = replace(
                    deployment,
                    connection_id=connection.id,
                    api_base="",
                    credential_env_refs={},
                    credential_secrets={},
                )
        return self.repository.save_deployment(deployment)

    def create_deployment_from_template(self, template_id: str, payload: Dict[str, Any]) -> ModelDeployment:
        deployment = self.draft_deployment_from_template(template_id, payload)
        if deployment.enabled and not deployment.is_local and deployment.health_status != "healthy":
            raise ModelFarmError("Test a remote deployment successfully before enabling it.")
        self._ensure_unique_name(deployment.name)
        if _connection_provider(deployment) != "local_builtin" and not deployment.connection_id:
            connection = self.create_connection(
                {
                    "name": self._unique_connection_name(f"{deployment.name} connection"),
                    "provider": _connection_provider(deployment),
                    "access_path": _default_access_path(_connection_provider(deployment)),
                    "api_base": deployment.api_base,
                    "credential_env_refs": deployment.credential_env_refs,
                    "credential_secrets": self.credential_values(deployment),
                    "locality": _default_locality(_connection_provider(deployment)),
                    "metadata": {"created_from_template": template_id},
                }
            )
            deployment = replace(
                deployment,
                connection_id=connection.id,
                api_base="",
                credential_env_refs={},
                credential_secrets={},
            )
        return self.repository.save_deployment(deployment)

    def draft_deployment_from_template(self, template_id: str, payload: Dict[str, Any]) -> ModelDeployment:
        template = _template_by_id(template_id)
        if not template:
            raise ModelFarmError(f"Unknown provider template: {template_id}")
        if template.get("builtin_deployment_id"):
            raise ModelFarmError("Built-in local deployments are already registered; select the existing deployment instead.")
        defaults = dict(template.get("deployment_defaults") or {})
        connection_id = str(payload.get("connection_id") or "").strip()
        connection: Optional[ModelConnection] = None
        if connection_id:
            connection = self.get_connection(connection_id)
            template_provider = str(defaults.get("provider") or "").strip().lower()
            expected_provider = "openai" if template_provider in {"custom", "openai"} else template_provider
            if connection.provider != expected_provider:
                raise ModelFarmError(
                    f"Connection '{connection.name}' uses {connection.provider}, but this template requires {expected_provider}."
                )
        allowed_capabilities = set(template.get("capabilities") or defaults.get("capabilities") or [])
        requested_capabilities = list(payload.get("capabilities") or defaults.get("capabilities") or [])
        model = str(payload.get("model") or defaults.get("model") or "").strip()
        api_base = "" if connection_id else str(payload.get("api_base", defaults.get("api_base", "")) or "").strip()
        provider = str(defaults.get("provider") or "").strip()
        provider_label = str(template.get("provider_label") or template.get("label") or "").strip()
        resolved_provider = connection.provider if connection else (
            "openrouter" if _is_openrouter_fields(
                model=model, api_base=api_base, provider=provider, provider_label=provider_label,
            ) else provider
        )
        if resolved_provider == "openrouter":
            requested_capabilities = [
                capability
                for capability in requested_capabilities
                if capability in {"generation", "judge", "planner", "classifier"}
            ]
            if not requested_capabilities:
                requested_capabilities = ["generation"]
        unsupported = sorted(set(requested_capabilities) - allowed_capabilities)
        if unsupported:
            raise ModelFarmError(f"Template '{template_id}' does not support capabilities: {', '.join(unsupported)}.")
        merged = {
            **defaults,
            "id": f"model-{uuid.uuid4().hex}",
            "name": self._unique_name(str(payload.get("name") or defaults.get("name") or template.get("label") or "Model deployment")),
            "provider": resolved_provider,
            "model": model,
            "connection_id": connection_id,
            "api_base": api_base,
            "credential_env_refs": {} if connection_id else dict(payload.get("credential_env_refs") or defaults.get("credential_env_refs") or {}),
            "credential_secrets": {} if connection_id else self._encrypted_secret_payload(payload.get("credential_secrets") or {}),
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
        return self._deployment_from_payload(merged)

    def update_deployment(self, deployment_id: str, payload: Dict[str, Any]) -> ModelDeployment:
        current = self.get_deployment(deployment_id)
        connection_keys = {"api_base", "credential_env_refs", "credential_secrets"}
        if current.connection_id and any(key in payload for key in connection_keys):
            connection_payload = {key: payload[key] for key in connection_keys if key in payload}
            self.update_connection(current.connection_id, connection_payload)
            payload = {key: value for key, value in payload.items() if key not in connection_keys}
        updated = self.draft_update_deployment(deployment_id, payload)
        if updated.enabled and not updated.is_local and updated.health_status != "healthy":
            raise ModelFarmError("Test a remote deployment successfully before enabling it.")
        self._ensure_unique_name(updated.name, exclude_id=deployment_id)
        return self.repository.save_deployment(updated)

    def draft_update_deployment(self, deployment_id: str, payload: Dict[str, Any]) -> ModelDeployment:
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
        return updated

    def _deployment_from_payload(self, payload: Dict[str, Any]) -> ModelDeployment:
        now = utc_now()
        return _validated_deployment(
            ModelDeployment(
                id=str(payload.get("id") or f"model-{uuid.uuid4().hex}"),
                name=str(payload.get("name") or "").strip(),
                provider=str(payload.get("provider") or "").strip(),
                model=str(payload.get("model") or "").strip(),
                capabilities=list(payload.get("capabilities") or []),
                connection_id=str(payload.get("connection_id") or ""),
                api_base=str(payload.get("api_base") or "").strip(),
                credential_env_refs=dict(payload.get("credential_env_refs") or {}),
                credential_secrets=dict(payload.get("credential_secrets") or {}),
                default_parameters=dict(payload.get("default_parameters") or {}),
                limits=dict(payload.get("limits") or {}),
                pricing=dict(payload.get("pricing") or {}),
                monthly_budget_usd=float(payload.get("monthly_budget_usd") or 0.0),
                hard_budget=bool(payload.get("hard_budget", True)),
                locality=str(payload.get("locality") or "remote"),
                enabled=bool(payload.get("enabled", False)),
                health_status=str(payload.get("health_status") or "untested"),
                metadata=dict(payload.get("metadata") or {}),
                created_at=str(payload.get("created_at") or now),
                updated_at=str(payload.get("updated_at") or now),
            )
        )

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

    def missing_credentials(self, deployment: ModelDeployment | ModelConnection) -> List[str]:
        if isinstance(deployment, ModelDeployment) and deployment.connection_id:
            deployment = self.get_connection(deployment.connection_id)
        if deployment.is_local or deployment.metadata.get("auth_mode") == "ambient":
            return []
        missing: List[str] = []
        stored_secret_keys = {str(key) for key, value in deployment.credential_secrets.items() if value}
        provider = deployment.provider.lower()
        required_keys = {"api_key"} if provider in {"openrouter", "openai", "gemini"} else set()
        for key in required_keys:
            env_name = deployment.credential_env_refs.get(key, "")
            if key not in stored_secret_keys and not (env_name and os.getenv(env_name)):
                missing.append(env_name or key)
        for key, env_name in deployment.credential_env_refs.items():
            if key in stored_secret_keys:
                continue
            if env_name and os.getenv(env_name):
                continue
            missing.append(env_name or key)
        return sorted(set(missing))

    def credential_status(self, deployment: ModelDeployment | ModelConnection) -> Dict[str, Any]:
        missing = self.missing_credentials(deployment)
        secret_keys = sorted(key for key, value in deployment.credential_secrets.items() if value)
        return {
            "configured": not missing,
            "missing": missing,
            "references": sorted(set(deployment.credential_env_refs.values())),
            "stored_secret_keys": secret_keys,
            "has_stored_secret": bool(secret_keys),
        }

    def credential_values(self, deployment: ModelDeployment | ModelConnection) -> Dict[str, str]:
        if deployment.credential_secrets and not self._secret_key:
            raise ModelFarmError(
                "ARAGBIZ_MODEL_SECRET_KEY must be configured before stored model credentials can be used."
            )
        values: Dict[str, str] = {}
        for key, token in deployment.credential_secrets.items():
            clean_key = str(key).strip()
            if not clean_key or not token:
                continue
            values[clean_key] = _decrypt_secret(str(token), self._secret_key)
            if values[clean_key]:
                _KNOWN_SECRET_VALUES.add(values[clean_key])
        return values

    def _encrypted_secret_payload(self, raw: Dict[str, Any]) -> Dict[str, str]:
        if any(str(value or "").strip() for value in dict(raw or {}).values()) and not self._secret_key:
            raise ModelFarmError(
                "ARAGBIZ_MODEL_SECRET_KEY must be configured before model credentials can be stored."
            )
        encrypted: Dict[str, str] = {}
        for key, value in dict(raw or {}).items():
            clean_key = str(key).strip()
            clean_value = str(value or "").strip()
            if not clean_key or not clean_value:
                continue
            encrypted[clean_key] = clean_value if clean_value.startswith(("v1:", "v2:")) else _encrypt_secret(clean_value, self._secret_key)
        return encrypted

    def _updated_secret_payload(self, current: Dict[str, str], raw: Optional[Dict[str, Any]]) -> Dict[str, str]:
        if (current or any(str(value or "").strip() for value in dict(raw or {}).values())) and not self._secret_key:
            raise ModelFarmError(
                "ARAGBIZ_MODEL_SECRET_KEY must be configured before model credentials can be updated."
            )
        updated = {
            key: _encrypt_secret(_decrypt_secret(value, self._secret_key), self._secret_key)
            if str(value).startswith("v1:") else value
            for key, value in current.items()
        }
        if raw is None:
            return updated
        for key, value in dict(raw or {}).items():
            clean_key = str(key).strip()
            if not clean_key:
                continue
            clean_value = str(value or "").strip()
            if clean_value:
                updated[clean_key] = clean_value if clean_value.startswith(("v1:", "v2:")) else _encrypt_secret(clean_value, self._secret_key)
            else:
                updated.pop(clean_key, None)
        return updated

    def set_health(
        self,
        deployment_id: str,
        *,
        healthy: bool,
        error: str = "",
        dimension: int = 0,
        status: str = "",
    ) -> ModelDeployment:
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
            health_status=status or ("healthy" if healthy else "unavailable"),
            last_health_check=utc_now(),
            last_error="" if healthy else _safe_error(error),
            updated_at=utc_now(),
        )
        return self.repository.save_deployment(updated)

    def set_connection_health(self, connection_id: str, *, healthy: bool, error: str = "") -> ModelConnection:
        current = self.get_connection(connection_id)
        updated = replace(
            current,
            enabled=True if healthy else current.enabled,
            health_status="healthy" if healthy else "unavailable",
            last_health_check=utc_now(),
            last_error="" if healthy else _safe_error(error),
            updated_at=utc_now(),
        )
        return self.repository.save_connection(updated)

    def available_models(self, connection_id: str) -> List[Dict[str, Any]]:
        connection = self.get_connection(connection_id)
        missing = self.missing_credentials(connection)
        if missing:
            raise ModelFarmError(f"Connection '{connection.name}' is missing credentials: {', '.join(missing)}.")
        return _discover_connection_models(connection, self._resolved_credentials(connection))

    def test_connection(self, connection_id: str) -> Dict[str, Any]:
        connection = self.get_connection(connection_id)
        try:
            models = self.available_models(connection_id)
            updated = self.set_connection_health(connection_id, healthy=True)
            return {
                "status": "healthy",
                "model_count": len(models),
                "sample_models": models[:10],
                "connection": updated,
            }
        except Exception as exc:
            updated = self.set_connection_health(connection_id, healthy=False, error=str(exc))
            return {"status": "unavailable", "error": _safe_error(exc), "connection": updated}

    def _resolved_credentials(self, connection: ModelConnection) -> Dict[str, str]:
        values = self.credential_values(connection)
        for key, env_name in connection.credential_env_refs.items():
            if key not in values and env_name and os.getenv(env_name):
                values[key] = str(os.getenv(env_name))
        return values

    def assert_egress_allowed(self, deployment: ModelDeployment, *, external_processing_allowed: bool, content_kind: str) -> None:
        is_local = deployment.is_local
        if deployment.connection_id:
            try:
                is_local = self.get_connection(deployment.connection_id).is_local
            except KeyError:
                pass
        if not is_local and not external_processing_allowed:
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

    def record_usage(self, event: ModelUsageEvent) -> ModelUsageEvent:
        self.repository.append_usage(event)
        return event

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

    def _ensure_unique_connection_name(self, name: str, exclude_id: str = "") -> None:
        for item in self.repository.list_connections():
            if item.id != exclude_id and item.name.lower() == name.lower():
                raise ModelFarmError(f"A model connection named '{name}' already exists.")

    def _unique_connection_name(self, base_name: str) -> str:
        normalized = " ".join((base_name or "Model connection").strip().split()) or "Model connection"
        existing = {item.name.lower() for item in self.repository.list_connections()}
        if normalized.lower() not in existing:
            return normalized
        for index in range(2, 1000):
            candidate = f"{normalized} {index}"
            if candidate.lower() not in existing:
                return candidate
        return f"{normalized} {uuid.uuid4().hex[:8]}"

    def _seed_local_connection(self) -> None:
        if any(item.id == "connection-local-builtin" for item in self.repository.list_connections()):
            return
        now = utc_now()
        self.repository.save_connection(
            ModelConnection(
                id="connection-local-builtin",
                name="In-process local models",
                provider="local_builtin",
                access_path="local",
                locality="local",
                enabled=True,
                health_status="healthy",
                metadata={"builtin": True},
                created_at=now,
                updated_at=now,
            )
        )

    def _seed_local_deployments(self) -> None:
        existing = {item.id for item in self.repository.list_deployments()}
        for deployment in builtin_model_deployments():
            if deployment.id not in existing:
                self.repository.save_deployment(deployment)

    def _backfill_legacy_connections(self) -> None:
        connection_ids = {item.id for item in self.repository.list_connections()}
        for deployment in self.repository.list_deployments():
            if deployment.connection_id:
                continue
            if str(deployment.provider or "").strip().lower() == "local":
                self.repository.save_deployment(replace(deployment, connection_id="connection-local-builtin"))
                continue
            identity = json.dumps(
                {
                    "provider": _connection_provider(deployment),
                    "api_base": deployment.api_base,
                    "locality": deployment.locality,
                    "credential_env_refs": deployment.credential_env_refs,
                    "credential_secrets": deployment.credential_secrets,
                },
                sort_keys=True,
            )
            connection_id = f"connection-migrated-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]}"
            if connection_id not in connection_ids:
                now = utc_now()
                self.repository.save_connection(
                    _validated_connection(
                        ModelConnection(
                            id=connection_id,
                            name=self._unique_connection_name(f"{deployment.name} connection"),
                            provider=_connection_provider(deployment),
                            access_path=_default_access_path(_connection_provider(deployment)),
                            api_base=deployment.api_base,
                            credential_env_refs=deployment.credential_env_refs,
                            credential_secrets=deployment.credential_secrets,
                            locality=_default_locality(_connection_provider(deployment)),
                            enabled=deployment.enabled,
                            health_status=deployment.health_status,
                            last_health_check=deployment.last_health_check,
                            last_error=deployment.last_error,
                            metadata={"migrated_from_deployment": True},
                            created_at=now,
                            updated_at=now,
                        )
                    )
                )
                connection_ids.add(connection_id)
            self.repository.save_deployment(
                replace(
                    deployment,
                    connection_id=connection_id,
                    api_base="",
                    credential_env_refs={},
                    credential_secrets={},
                )
            )


@dataclass(frozen=True)
class ResolvedModel:
    deployment: ModelDeployment
    connection: ModelConnection
    gateway_model: str
    credentials: Dict[str, str]


class ModelAdapter(Protocol):
    async def generate(
        self,
        resolved: ResolvedModel,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any],
    ) -> ModelGenerationResult: ...

    async def stream(
        self,
        resolved: ResolvedModel,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncIterator[ModelStreamEvent]: ...

    async def embed(self, resolved: ResolvedModel, texts: List[str]) -> ModelEmbeddingResult: ...

    async def rerank(
        self,
        resolved: ResolvedModel,
        query: str,
        documents: List[str],
        top_n: int,
    ) -> ModelRerankResult: ...


class LocalBuiltinAdapter:
    async def generate(
        self,
        resolved: ResolvedModel,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any],
    ) -> ModelGenerationResult:
        deployment = resolved.deployment
        if deployment.model == "extractive":
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
                answer, deployment.id, deployment.provider, deployment.model, "completed",
                _rough_token_count(prompt), _rough_token_count(answer), 0.0, "stop",
                {"runtime": "deterministic-extractive", "gateway_model": resolved.gateway_model},
            )
        if deployment.model == "google/flan-t5-small":
            prompt = "\n".join(item.get("content", "") for item in messages)
            answer = await asyncio.to_thread(
                _run_local_flan,
                deployment.model,
                prompt,
                {**deployment.default_parameters, **parameters},
            )
            return ModelGenerationResult(
                answer, deployment.id, deployment.provider, deployment.model, "completed",
                _rough_token_count(prompt), _rough_token_count(answer), 0.0, "stop",
                {"runtime": "transformers-text2text", "gateway_model": resolved.gateway_model},
            )
        raise ModelFarmError(f"Unsupported in-process generation deployment: {deployment.model}")

    async def stream(
        self,
        resolved: ResolvedModel,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        deployment = resolved.deployment
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        if deployment.model == "google/flan-t5-small":
            prompt = "\n".join(item.get("content", "") for item in messages)
            parts: List[str] = []
            async for delta in _stream_local_flan(
                deployment.model,
                prompt,
                {**deployment.default_parameters, **parameters},
                cancellation_token,
            ):
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                if delta:
                    parts.append(delta)
                    yield ModelStreamEvent("delta", {"text": delta})
            text = "".join(parts)
            result = ModelGenerationResult(
                text, deployment.id, deployment.provider, deployment.model, "completed",
                _rough_token_count(prompt), _rough_token_count(text), 0.0, "stop",
                {"runtime": "transformers-text2text-stream", "gateway_model": resolved.gateway_model},
            )
        else:
            result = await self.generate(resolved, messages, parameters)
            for chunk in _text_chunks(result.text, 32):
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                yield ModelStreamEvent("delta", {"text": chunk})
        yield ModelStreamEvent("model_completed", _generation_public_dict(result))

    async def embed(self, resolved: ResolvedModel, texts: List[str]) -> ModelEmbeddingResult:
        from aragbiz.knowledge import HashEmbeddingModel, SentenceTransformerEmbeddingModel

        deployment = resolved.deployment
        dimension = deployment.dimension or 384
        if deployment.model == "hash-embedding-384":
            embedder = HashEmbeddingModel(dimension=dimension)
        elif deployment.model == "sentence-transformers/all-MiniLM-L6-v2":
            embedder = SentenceTransformerEmbeddingModel(deployment.model, dimension=dimension)
        else:
            raise ModelFarmError(f"Unsupported local embedding deployment: {deployment.model}")
        embeddings = await asyncio.to_thread(embedder.embed, texts)
        return ModelEmbeddingResult(
            embeddings, deployment.id, deployment.provider, deployment.model,
            len(embeddings[0]) if embeddings else dimension,
            _rough_token_count("\n".join(texts)), 0.0,
            {"runtime": "local", "gateway_model": resolved.gateway_model},
        )

    async def rerank(
        self,
        resolved: ResolvedModel,
        query: str,
        documents: List[str],
        top_n: int,
    ) -> ModelRerankResult:
        deployment = resolved.deployment
        query_terms = set(_tokens(query))
        ranked = []
        for index, document in enumerate(documents):
            terms = set(_tokens(document))
            score = len(query_terms & terms) / max(len(query_terms | terms), 1)
            ranked.append(ModelRerankItem(index, score))
        ranked.sort(key=lambda item: item.score, reverse=True)
        return ModelRerankResult(
            ranked[:top_n], deployment.id, deployment.provider, deployment.model, 0.0,
            {"runtime": "lexical", "gateway_model": resolved.gateway_model},
        )


class LiteLLMAdapter:
    @staticmethod
    def _module() -> Any:
        try:
            import litellm  # type: ignore
        except ImportError as exc:
            raise ModelFarmError(
                "Install the models extra to use hosted or local-server model connections: "
                "python -m pip install -e \".[models]\"."
            ) from exc
        return litellm

    def _kwargs(self, resolved: ResolvedModel) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if resolved.connection.api_base and _should_pass_litellm_api_base(resolved.connection):
            kwargs["api_base"] = resolved.connection.api_base
        kwargs.update({key: value for key, value in resolved.credentials.items() if value})
        timeout = resolved.deployment.limits.get("timeout_seconds")
        if timeout:
            kwargs["timeout"] = float(timeout)
        return kwargs

    async def generate(
        self,
        resolved: ResolvedModel,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any],
    ) -> ModelGenerationResult:
        litellm = self._module()
        deployment = resolved.deployment
        kwargs = self._kwargs(resolved)
        kwargs.update(deployment.default_parameters)
        kwargs.update(parameters)
        kwargs.update({"model": resolved.gateway_model, "messages": messages})
        response = await litellm.acompletion(**kwargs)
        text = _completion_text(response)
        usage = _completion_usage(response)
        input_tokens = usage.get("prompt_tokens") or _rough_token_count("\n".join(item.get("content", "") for item in messages))
        output_tokens = usage.get("completion_tokens") or _rough_token_count(text)
        cost = _response_cost(response) or _estimate_cost(deployment, input_tokens, output_tokens)
        pricing_known = _litellm_pricing_known(litellm, resolved) or cost > 0
        return ModelGenerationResult(
            text, deployment.id, deployment.provider, deployment.model, "completed",
            input_tokens, output_tokens, cost, _completion_finish_reason(response),
            {
                "runtime": "litellm", "gateway_model": resolved.gateway_model,
                "connection_id": resolved.connection.id, "pricing_known": pricing_known,
            },
        )

    async def stream(
        self,
        resolved: ResolvedModel,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        litellm = self._module()
        deployment = resolved.deployment
        kwargs = self._kwargs(resolved)
        kwargs.update(deployment.default_parameters)
        kwargs.update(parameters)
        kwargs.update({"model": resolved.gateway_model, "messages": messages, "stream": True})
        if cancellation_token:
            cancellation_token.raise_if_cancelled()
        response = await litellm.acompletion(**kwargs)
        parts: List[str] = []
        try:
            async for chunk in response:
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                delta = _stream_text(chunk)
                if delta:
                    parts.append(delta)
                    yield ModelStreamEvent("delta", {"text": delta})
        finally:
            close = getattr(response, "aclose", None)
            if callable(close):
                closed = close()
                if hasattr(closed, "__await__"):
                    await closed
        text = "".join(parts)
        input_tokens = _rough_token_count("\n".join(item.get("content", "") for item in messages))
        output_tokens = _rough_token_count(text)
        result = ModelGenerationResult(
            text, deployment.id, deployment.provider, deployment.model, "completed",
            input_tokens, output_tokens, _estimate_cost(deployment, input_tokens, output_tokens), "stop",
            {
                "runtime": "litellm-stream", "gateway_model": resolved.gateway_model,
                "connection_id": resolved.connection.id,
                "pricing_known": _litellm_pricing_known(litellm, resolved),
            },
        )
        yield ModelStreamEvent("model_completed", _generation_public_dict(result))

    async def embed(self, resolved: ResolvedModel, texts: List[str]) -> ModelEmbeddingResult:
        litellm = self._module()
        deployment = resolved.deployment
        kwargs = self._kwargs(resolved)
        kwargs.update(deployment.default_parameters)
        kwargs.update({"model": resolved.gateway_model, "input": texts})
        response = await litellm.aembedding(**kwargs)
        embeddings = _embedding_vectors(response)
        if not embeddings:
            raise ModelFarmError("The embedding provider returned no vectors.")
        usage = _completion_usage(response)
        input_tokens = usage.get("prompt_tokens") or usage.get("total_tokens") or _rough_token_count("\n".join(texts))
        return ModelEmbeddingResult(
            embeddings, deployment.id, deployment.provider, deployment.model, len(embeddings[0]), input_tokens,
            _response_cost(response) or _estimate_cost(deployment, input_tokens, 0),
            {
                "runtime": "litellm", "gateway_model": resolved.gateway_model,
                "connection_id": resolved.connection.id,
                "pricing_known": _litellm_pricing_known(litellm, resolved),
            },
        )

    async def rerank(
        self,
        resolved: ResolvedModel,
        query: str,
        documents: List[str],
        top_n: int,
    ) -> ModelRerankResult:
        litellm = self._module()
        deployment = resolved.deployment
        kwargs = self._kwargs(resolved)
        kwargs.update(deployment.default_parameters)
        kwargs.update({"model": resolved.gateway_model, "query": query, "documents": documents, "top_n": top_n})
        if hasattr(litellm, "arerank"):
            response = await litellm.arerank(**kwargs)
        else:  # pragma: no cover - compatibility with the pinned LiteLLM release
            response = await asyncio.to_thread(litellm.rerank, **kwargs)
        input_tokens = _rough_token_count(query + "\n" + "\n".join(documents))
        return ModelRerankResult(
            _rerank_items(response), deployment.id, deployment.provider, deployment.model,
            _response_cost(response) or _estimate_cost(deployment, input_tokens, 0),
            {
                "runtime": "litellm", "gateway_model": resolved.gateway_model,
                "connection_id": resolved.connection.id,
                "pricing_known": _litellm_pricing_known(litellm, resolved),
            },
        )


class ModelGateway:
    def __init__(
        self,
        service: ModelFarmService,
        *,
        classifier_model_paths: Optional[Dict[str, str]] = None,
    ):
        self.service = service
        self.local_adapter = LocalBuiltinAdapter()
        self.litellm_adapter = LiteLLMAdapter()
        self.classifier_model_paths = {
            "query_classifier_distilbert": "data/artifacts/query_classifier_distilbert",
            "query_classifier_t5": "data/artifacts/query_classifier_t5",
            "query_classifier_distilbert_v2": "data/artifacts/query_classifier_distilbert_v2",
            "query_classifier_t5_v2": "data/artifacts/query_classifier_t5_v2",
            **dict(classifier_model_paths or {}),
        }
        self._classifier_cache: Dict[tuple[str, str], Any] = {}

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
        capability: str = "generation",
    ) -> ModelGenerationResult:
        candidates = [deployment_id, *(fallback_deployment_ids or [])]
        last_error: Optional[Exception] = None
        fallback_attempts: List[Dict[str, Any]] = []
        for fallback_index, candidate_id in enumerate(candidates):
            deployment = self.service.resolve(candidate_id, capability, require_enabled=require_enabled)
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
                usage_event_id = self._record_completed(deployment, capability, context, result.input_tokens, result.output_tokens, latency, result.estimated_cost_usd, fallback_index)
                return replace(
                    result,
                    metadata={
                        **result.metadata,
                        "fallback_index": fallback_index,
                        "fallback_attempts": fallback_attempts,
                        "latency_ms": round(latency, 3),
                        "usage_event_id": usage_event_id,
                    },
                )
            except Exception as exc:  # provider exception types are optional imports
                latency = (time.perf_counter() - started) * 1000
                usage_event_id = self._record_failed(deployment, capability, context, latency, exc, fallback_index)
                fallback_attempts.append(
                    {**_failed_model_attempt(deployment, exc, fallback_index, latency), "usage_event_id": usage_event_id}
                )
                last_error = exc
                if not _is_retryable_error(exc) or fallback_index == len(candidates) - 1:
                    break
        operation = "Generator" if capability == "generation" else capability.replace("_", " ").title()
        raise ModelFarmError(f"{operation} execution failed: {_safe_error(last_error)}") from last_error

    async def stream(
        self,
        messages: List[Dict[str, str]],
        deployment_id: str,
        *,
        fallback_deployment_ids: Optional[Sequence[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[ModelCallContext] = None,
        external_processing_allowed: bool = True,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        candidates = [deployment_id, *(fallback_deployment_ids or [])]
        last_error: Optional[Exception] = None
        fallback_attempts: List[Dict[str, Any]] = []
        for fallback_index, candidate_id in enumerate(candidates):
            deployment = self.service.resolve(candidate_id, "generation")
            self.service.assert_egress_allowed(
                deployment,
                external_processing_allowed=external_processing_allowed,
                content_kind="Prompt content",
            )
            self.service.assert_budget(deployment)
            resolved = self._resolve_model(deployment)
            started = time.perf_counter()
            emitted_delta = False
            completed: Dict[str, Any] = {}
            output_parts: List[str] = []
            try:
                if cancellation_token:
                    cancellation_token.raise_if_cancelled()
                async for event in self._adapter(resolved).stream(
                    resolved,
                    messages,
                    parameters or {},
                    cancellation_token,
                ):
                    if cancellation_token:
                        cancellation_token.raise_if_cancelled()
                    if event.type == "delta":
                        emitted_delta = True
                        output_parts.append(str(event.data.get("text") or ""))
                        yield event
                    elif event.type == "model_completed":
                        completed = dict(event.data)
                latency = (time.perf_counter() - started) * 1000
                input_tokens = int(completed.get("input_tokens") or 0)
                output_tokens = int(completed.get("output_tokens") or 0)
                cost = float(completed.get("estimated_cost_usd") or 0.0)
                usage_event_id = self._record_completed(
                    deployment, "generation", context, input_tokens, output_tokens, latency, cost, fallback_index,
                    resolved=resolved,
                )
                completed["metadata"] = {
                    **dict(completed.get("metadata") or {}),
                    "connection_id": resolved.connection.id,
                    "access_path": resolved.connection.access_path,
                    "gateway_model": resolved.gateway_model,
                    "fallback_index": fallback_index,
                    "fallback_attempts": fallback_attempts,
                    "latency_ms": round(latency, 3),
                    "usage_event_id": usage_event_id,
                }
                yield ModelStreamEvent("model_completed", completed)
                return
            except AnswerCancelled:
                latency = (time.perf_counter() - started) * 1000
                input_tokens = _rough_token_count("\n".join(item.get("content", "") for item in messages))
                output_tokens = _rough_token_count("".join(output_parts))
                cost = _estimate_cost(deployment, input_tokens, output_tokens)
                self._record_cancelled(
                    deployment,
                    "generation",
                    context,
                    input_tokens,
                    output_tokens,
                    latency,
                    cost,
                    fallback_index,
                    resolved=resolved,
                )
                raise
            except Exception as exc:
                latency = (time.perf_counter() - started) * 1000
                usage_event_id = self._record_failed(deployment, "generation", context, latency, exc, fallback_index, resolved=resolved)
                failed_attempt = _failed_model_attempt(
                    deployment,
                    exc,
                    fallback_index,
                    latency,
                    gateway_model=resolved.gateway_model,
                    connection_id=resolved.connection.id,
                    access_path=resolved.connection.access_path,
                )
                failed_attempt["usage_event_id"] = usage_event_id
                fallback_attempts.append(failed_attempt)
                last_error = exc
                can_fallback = (
                    not emitted_delta
                    and _is_retryable_error(exc)
                    and fallback_index < len(candidates) - 1
                )
                if can_fallback:
                    next_deployment = self.service.get_deployment(candidates[fallback_index + 1])
                    yield ModelStreamEvent(
                        "model_fallback",
                        {
                            **failed_attempt,
                            "next_deployment_id": next_deployment.id,
                            "next_deployment_name": next_deployment.name,
                            "next_provider": next_deployment.provider,
                            "next_model": next_deployment.model,
                        },
                    )
                    continue
                if emitted_delta or not _is_retryable_error(exc) or fallback_index == len(candidates) - 1:
                    break
        raise ModelFarmError(f"Generator streaming failed: {_safe_error(last_error)}") from last_error

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
            result = await self._embed_once(deployment, texts, require_connection_enabled=require_enabled)
            latency = (time.perf_counter() - started) * 1000
            usage_event_id = self._record_completed(deployment, "embedding", context, result.input_tokens, 0, latency, result.estimated_cost_usd, 0)
            return replace(result, metadata={**result.metadata, "latency_ms": round(latency, 3), "usage_event_id": usage_event_id})
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            self._record_failed(deployment, "embedding", context, latency, exc, 0)
            raise ModelFarmError(f"Embedding execution failed: {_safe_error(exc)}") from exc

    async def classify(
        self,
        query: str,
        deployment_id: str,
        *,
        context: Optional[ModelCallContext] = None,
        external_processing_allowed: bool = True,
        require_enabled: bool = True,
    ) -> ModelClassificationResult:
        deployment = self.service.resolve(deployment_id, "classifier", require_enabled=require_enabled)
        self.service.assert_egress_allowed(
            deployment,
            external_processing_allowed=external_processing_allowed,
            content_kind="Query text",
        )
        self.service.assert_budget(deployment)
        started = time.perf_counter()
        try:
            resolved = self._resolve_model(deployment, require_connection_enabled=require_enabled)
            if resolved.connection.provider == "local_builtin":
                classifier = self._local_classifier(deployment)
                from aragbiz.classifier import predict_scored

                prediction = await asyncio.to_thread(predict_scored, classifier, query)
                label = prediction.label
                probabilities = prediction.probabilities
                confidence = prediction.confidence
                margin = prediction.margin
                supported_labels = prediction.supported_labels
                input_tokens = _rough_token_count(query)
                output_tokens = 1
                cost = 0.0
                runtime = (
                    "transformers-text2text-classification"
                    if deployment.model in {"query_classifier_t5", "query_classifier_t5_v2"}
                    else "huggingface-sequence-classification"
                )
                metadata = {
                    "runtime": runtime,
                    "gateway_model": resolved.gateway_model,
                    "connection_id": resolved.connection.id,
                    "access_path": resolved.connection.access_path,
                }
            else:
                generated = await self._generate_once(
                    deployment,
                    [
                        {
                            "role": "system",
                            "content": (
                                "Classify the business-workflow query complexity. "
                                "Return only JSON with keys label, confidence, and probabilities. "
                                "The label must be simple, moderate, complex, or advanced. "
                                "Probabilities must contain all four labels and sum approximately to 1."
                            ),
                        },
                        {"role": "user", "content": query},
                    ],
                    {"temperature": 0, "max_tokens": 160},
                    require_connection_enabled=require_enabled,
                )
                classification = _classification_payload(generated.text)
                label = classification["label"]
                probabilities = classification["probabilities"]
                confidence = classification["confidence"]
                margin = classification["margin"]
                supported_labels = list(probabilities)
                input_tokens = generated.input_tokens
                output_tokens = generated.output_tokens
                cost = generated.estimated_cost_usd
                metadata = {
                    **generated.metadata,
                    "runtime": generated.metadata.get("runtime", "litellm-classification"),
                    "gateway_model": resolved.gateway_model,
                    "connection_id": resolved.connection.id,
                    "access_path": resolved.connection.access_path,
                }
            latency = (time.perf_counter() - started) * 1000
            usage_event_id = self._record_completed(
                deployment,
                "classifier",
                context,
                input_tokens,
                output_tokens,
                latency,
                cost,
                0,
                resolved=resolved,
            )
            return ModelClassificationResult(
                label=str(label),
                deployment_id=deployment.id,
                provider=deployment.provider,
                model=deployment.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                probabilities=probabilities,
                confidence=confidence,
                margin=margin,
                supported_labels=supported_labels,
                metadata={**metadata, "latency_ms": round(latency, 3), "usage_event_id": usage_event_id},
            )
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            self._record_failed(deployment, "classifier", context, latency, exc, 0)
            raise ModelFarmError(f"Classifier execution failed: {_safe_error(exc)}") from exc

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
            usage_event_id = self._record_completed(deployment, "rerank", context, input_tokens, 0, latency, result.estimated_cost_usd, 0)
            return replace(result, metadata={**result.metadata, "latency_ms": round(latency, 3), "usage_event_id": usage_event_id})
        except Exception as exc:
            latency = (time.perf_counter() - started) * 1000
            self._record_failed(deployment, "rerank", context, latency, exc, 0)
            raise ModelFarmError(f"Reranker execution failed: {_safe_error(exc)}") from exc

    def generate_sync(self, *args: Any, **kwargs: Any) -> ModelGenerationResult:
        return _run_sync(self.generate(*args, **kwargs))

    def classify_sync(self, *args: Any, **kwargs: Any) -> ModelClassificationResult:
        return _run_sync(self.classify(*args, **kwargs))

    def embed_sync(self, *args: Any, **kwargs: Any) -> ModelEmbeddingResult:
        return _run_sync(self.embed(*args, **kwargs))

    def rerank_sync(self, *args: Any, **kwargs: Any) -> ModelRerankResult:
        return _run_sync(self.rerank(*args, **kwargs))

    async def test_deployment(self, deployment_id: str) -> Dict[str, Any]:
        deployment = self.service.get_deployment(deployment_id)
        try:
            if "classifier" in deployment.capabilities:
                result = await self.classify(
                    "How should an invoice mismatch be escalated and approved?",
                    deployment.id,
                    require_enabled=False,
                )
                if deployment.connection_id:
                    self.service.set_connection_health(deployment.connection_id, healthy=bool(result.label))
                updated = self.service.set_health(deployment.id, healthy=bool(result.label))
                return {"status": updated.health_status, "label": result.label, "deployment": updated}
            if "generation" in deployment.capabilities or "judge" in deployment.capabilities or "planner" in deployment.capabilities:
                result = await self._generate_once(
                    deployment,
                    [{"role": "user", "content": "Reply with the word ready."}],
                    {"max_tokens": 128},
                    require_connection_enabled=False,
                )
                self._assert_pricing_ready(deployment, result.metadata)
                if deployment.connection_id:
                    self.service.set_connection_health(deployment.connection_id, healthy=bool(result.text.strip()))
                updated = self.service.set_health(deployment.id, healthy=bool(result.text.strip()))
                return {"status": updated.health_status, "deployment": updated}
            if "embedding" in deployment.capabilities:
                result = await self.embed(["Adaptive RAG model health check"], deployment.id, require_enabled=False)
                self._assert_pricing_ready(deployment, result.metadata)
                if deployment.connection_id:
                    self.service.set_connection_health(deployment.connection_id, healthy=True)
                updated = self.service.set_health(deployment.id, healthy=True, dimension=result.dimension)
                return {"status": "healthy", "dimension": result.dimension, "deployment": updated}
            if "rerank" in deployment.capabilities:
                result = await self._rerank_once(
                    deployment, "invoice approval", ["invoice approval workflow", "holiday calendar"], 1,
                    require_connection_enabled=False,
                )
                self._assert_pricing_ready(deployment, result.metadata)
                if deployment.connection_id:
                    self.service.set_connection_health(deployment.connection_id, healthy=bool(result.items))
                updated = self.service.set_health(deployment.id, healthy=bool(result.items))
                return {"status": updated.health_status, "deployment": updated}
            raise ModelFarmError(f"Deployment '{deployment.name}' has no testable capability.")
        except Exception as exc:
            error_category = _error_category(exc)
            retryable = _is_retryable_error(exc)
            if deployment.connection_id and error_category != "rate_limit":
                self.service.set_connection_health(deployment.connection_id, healthy=False, error=str(exc))
            status = "rate_limited" if error_category == "rate_limit" else "unavailable"
            updated = self.service.set_health(
                deployment.id,
                healthy=False,
                error=str(exc),
                status=status,
            )
            return {
                "status": status,
                "error": _safe_error(exc),
                "error_category": error_category,
                "retryable": retryable,
                "deployment": updated,
            }

    async def test_draft_deployment(self, deployment: ModelDeployment) -> Dict[str, Any]:
        try:
            missing = self.service.missing_credentials(deployment)
            if missing:
                raise ModelFarmError(f"Deployment '{deployment.name}' is missing credentials: {', '.join(missing)}.")
            if (
                "generation" in deployment.capabilities
                or "judge" in deployment.capabilities
                or "planner" in deployment.capabilities
                or "classifier" in deployment.capabilities
            ):
                result = await self._generate_once(
                    deployment,
                    [{"role": "user", "content": "Reply with the word ready."}],
                    {"max_tokens": 128},
                    require_connection_enabled=False,
                )
                self._assert_pricing_ready(deployment, result.metadata)
                healthy = bool(result.text.strip())
                return {
                    "status": "healthy" if healthy else "unavailable",
                    "runtime": result.metadata.get("runtime", ""),
                    "sample": result.text.strip()[:500],
                    "deployment": replace(
                        deployment,
                        health_status="healthy" if healthy else "unavailable",
                        last_health_check=utc_now(),
                        last_error="" if healthy else "The generator returned an empty response.",
                    ),
                }
            if "embedding" in deployment.capabilities:
                result = await self._embed_once(
                    deployment, ["Adaptive RAG model health check"], require_connection_enabled=False,
                )
                self._assert_pricing_ready(deployment, result.metadata)
                return {
                    "status": "healthy",
                    "dimension": result.dimension,
                    "runtime": result.metadata.get("runtime", ""),
                    "deployment": replace(deployment, health_status="healthy", last_health_check=utc_now(), last_error=""),
                }
            if "rerank" in deployment.capabilities:
                result = await self._rerank_once(
                    deployment, "invoice approval", ["invoice approval workflow", "holiday calendar"], 1,
                    require_connection_enabled=False,
                )
                self._assert_pricing_ready(deployment, result.metadata)
                healthy = bool(result.items)
                return {
                    "status": "healthy" if healthy else "unavailable",
                    "runtime": result.metadata.get("runtime", ""),
                    "deployment": replace(
                        deployment,
                        health_status="healthy" if healthy else "unavailable",
                        last_health_check=utc_now(),
                        last_error="" if healthy else "The reranker returned no results.",
                    ),
                }
            raise ModelFarmError(f"Deployment '{deployment.name}' has no testable capability.")
        except Exception as exc:
            error_category = _error_category(exc)
            status = "rate_limited" if error_category == "rate_limit" else "unavailable"
            return {
                "status": status,
                "error": _safe_error(exc),
                "error_category": error_category,
                "retryable": _is_retryable_error(exc),
                "deployment": replace(deployment, health_status=status, last_health_check=utc_now(), last_error=_safe_error(exc)),
            }

    async def _generate_once(
        self,
        deployment: ModelDeployment,
        messages: List[Dict[str, str]],
        parameters: Dict[str, Any],
        *,
        require_connection_enabled: bool = True,
    ) -> ModelGenerationResult:
        resolved = self._resolve_model(deployment, require_connection_enabled=require_connection_enabled)
        return await self._adapter(resolved).generate(resolved, messages, parameters)

    def _local_classifier(self, deployment: ModelDeployment) -> Any:
        model_path = str(self.classifier_model_paths.get(deployment.model) or "").strip()
        if not model_path:
            raise ModelFarmError(f"No artifact path is configured for classifier deployment '{deployment.name}'.")
        resolved_path = str(Path(model_path).expanduser().resolve())
        cache_key = (deployment.model, resolved_path)
        if cache_key in self._classifier_cache:
            return self._classifier_cache[cache_key]
        path = Path(resolved_path)
        if not path.exists():
            raise ModelFarmError(
                f"Classifier artifact for '{deployment.name}' was not found at {path}. "
                "Train or extract the model artifact before selecting this deployment."
            )
        from aragbiz.classifier import HuggingFaceQueryClassifier, T5QueryClassifier

        classifier = T5QueryClassifier(path) if deployment.model in {"query_classifier_t5", "query_classifier_t5_v2"} else HuggingFaceQueryClassifier(path)
        self._classifier_cache[cache_key] = classifier
        return classifier

    async def _embed_once(
        self,
        deployment: ModelDeployment,
        texts: List[str],
        *,
        require_connection_enabled: bool = True,
    ) -> ModelEmbeddingResult:
        resolved = self._resolve_model(deployment, require_connection_enabled=require_connection_enabled)
        return await self._adapter(resolved).embed(resolved, texts)

    async def _rerank_once(
        self,
        deployment: ModelDeployment,
        query: str,
        documents: List[str],
        top_n: int,
        *,
        require_connection_enabled: bool = True,
    ) -> ModelRerankResult:
        resolved = self._resolve_model(deployment, require_connection_enabled=require_connection_enabled)
        return await self._adapter(resolved).rerank(resolved, query, documents, top_n)

    def _resolve_model(
        self,
        deployment: ModelDeployment,
        *,
        require_connection_enabled: bool = True,
    ) -> ResolvedModel:
        connection = self.service.connection_for_deployment(
            deployment,
            require_enabled=require_connection_enabled,
        )
        credentials = self.service.credential_values(connection)
        for key, env_name in connection.credential_env_refs.items():
            if key not in credentials and env_name and os.getenv(env_name):
                credentials[key] = str(os.getenv(env_name))
        return ResolvedModel(
            deployment=deployment,
            connection=connection,
            gateway_model=_gateway_model_name(connection.provider, deployment.model, connection.api_base),
            credentials=credentials,
        )

    def _adapter(self, resolved: ResolvedModel) -> ModelAdapter:
        if resolved.connection.provider == "local_builtin":
            return self.local_adapter
        return self.litellm_adapter

    def _assert_pricing_ready(self, deployment: ModelDeployment, metadata: Dict[str, Any]) -> None:
        connection = self.service.connection_for_deployment(deployment, require_enabled=False)
        if connection.is_local or metadata.get("pricing_known"):
            return
        raise ModelFarmError(
            "LiteLLM has no known pricing for this model. Configure a positive administrator pricing override "
            "before enabling paid execution."
        )

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
        *,
        resolved: Optional[ResolvedModel] = None,
    ) -> str:
        resolved = resolved or self._resolved_for_usage(deployment)
        event = self.service.record_usage(
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
                connection_id=resolved.connection.id if resolved else "",
                access_path=resolved.connection.access_path if resolved else "",
                gateway_model=resolved.gateway_model if resolved else deployment.model,
            )
        )

        return event.id

    def _record_failed(
        self,
        deployment: ModelDeployment,
        capability: str,
        context: Optional[ModelCallContext],
        latency_ms: float,
        exc: Exception,
        fallback_index: int,
        *,
        resolved: Optional[ResolvedModel] = None,
    ) -> str:
        resolved = resolved or self._resolved_for_usage(deployment)
        event = self.service.record_usage(
            _usage_event(
                deployment,
                capability,
                context,
                "failed",
                latency_ms=latency_ms,
                fallback_index=fallback_index,
                error=exc,
                connection_id=resolved.connection.id if resolved else "",
                access_path=resolved.connection.access_path if resolved else "",
                gateway_model=resolved.gateway_model if resolved else deployment.model,
            )
        )
        return event.id

    def _record_cancelled(
        self,
        deployment: ModelDeployment,
        capability: str,
        context: Optional[ModelCallContext],
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost: float,
        fallback_index: int,
        *,
        resolved: Optional[ResolvedModel] = None,
    ) -> str:
        resolved = resolved or self._resolved_for_usage(deployment)
        event = self.service.record_usage(
            _usage_event(
                deployment,
                capability,
                context,
                "cancelled",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost=cost,
                fallback_index=fallback_index,
                connection_id=resolved.connection.id if resolved else "",
                access_path=resolved.connection.access_path if resolved else "",
                gateway_model=resolved.gateway_model if resolved else deployment.model,
            )
        )
        return event.id

    def _resolved_for_usage(self, deployment: ModelDeployment) -> Optional[ResolvedModel]:
        try:
            return self._resolve_model(deployment, require_connection_enabled=False)
        except Exception:
            return None


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
            ["generation", "judge", "planner", "classifier"],
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
            ["generation", "judge", "planner", "classifier"],
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
            "openrouter-generation",
            "OpenRouter generation / judge",
            "OpenRouter",
            "openrouter",
            "google/gemma-4-31b-it:free",
            ["generation", "judge", "planner", "classifier"],
            {"api_key": "ARAGBIZ_MODEL_OPENROUTER_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 32000, "max_output_tokens": 1200, "timeout_seconds": 90},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
            api_base="https://openrouter.ai/api/v1",
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
            ["generation", "judge", "planner", "classifier"],
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
            "gemini-2.5-flash",
            ["generation", "judge", "planner", "classifier"],
            {"api_key": "ARAGBIZ_MODEL_GEMINI_API_KEY"},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 1000000, "max_output_tokens": 1200, "timeout_seconds": 60},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
        ),
        _provider_template(
            "ollama-generation",
            "Ollama local endpoint",
            "Ollama",
            "ollama",
            "llama3.1",
            ["generation", "embedding", "judge", "planner", "classifier"],
            {},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 32000, "max_output_tokens": 1200, "timeout_seconds": 120},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
            api_base="http://127.0.0.1:11434",
            metadata={"access_path": "local", "locality": "local"},
        ),
        _provider_template(
            "vllm-generation",
            "vLLM local endpoint",
            "vLLM",
            "vllm",
            "meta-llama/Llama-3.1-8B-Instruct",
            ["generation", "embedding", "rerank", "judge", "planner", "classifier"],
            {},
            {"temperature": 0.2, "max_tokens": 800},
            {"context_window": 32000, "max_output_tokens": 1200, "timeout_seconds": 120},
            {"input_per_million_tokens_usd": 0, "output_per_million_tokens_usd": 0},
            api_base="http://127.0.0.1:8001/v1",
            metadata={"access_path": "local", "locality": "local", "optional_api_key": True},
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
    supported = {
        "openai-generation", "openai-embedding", "openrouter-generation",
        "gemini-generation", "ollama-generation", "vllm-generation",
    }
    return local + [item for item in remote if item["id"] in supported]


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
    metadata = dict(metadata or {})
    locality = str(metadata.get("locality") or _default_locality(provider))
    access_path = str(metadata.get("access_path") or _default_access_path(provider))
    return {
        "id": template_id,
        "label": label,
        "provider_label": provider_label,
        "provider": provider,
        "model": model,
        "capabilities": capabilities,
        "locality": locality,
        "access_path": access_path,
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
            "locality": locality,
            "access_path": access_path,
            "enabled": False,
            "health_status": "untested",
            "metadata": metadata,
        },
    }


def _template_by_id(template_id: str) -> Optional[Dict[str, Any]]:
    return next((template for template in model_provider_templates() if template["id"] == template_id), None)


def builtin_model_deployments() -> List[ModelDeployment]:
    now = utc_now()
    common = {
        "connection_id": "connection-local-builtin",
        "enabled": True,
        "health_status": "healthy",
        "locality": "local",
        "metadata": {"builtin": True},
        "created_at": now,
        "updated_at": now,
    }
    return [
        ModelDeployment("model-local-extractive", "Local Extractive", "Local", "extractive", ["generation", "judge"], limits={"context_window": 3200, "max_output_tokens": 900}, **common),
        ModelDeployment("model-local-flan-t5-small", "Local FLAN-T5 Small", "Local", "google/flan-t5-small", ["generation", "judge", "planner"], limits={"context_window": 512, "max_output_tokens": 160}, **common),
        ModelDeployment("model-local-hash-384", "Local Hash Embedding 384", "Local", "hash-embedding-384", ["embedding"], limits={"dimension": 384, "batch_size": 256}, **common),
        ModelDeployment("model-local-minilm-384", "Local MiniLM Embedding 384", "Local", "sentence-transformers/all-MiniLM-L6-v2", ["embedding"], limits={"dimension": 384, "batch_size": 64}, **common),
        ModelDeployment("model-local-lexical-reranker", "Local Lexical Reranker", "Local", "lexical-overlap", ["rerank"], **common),
        ModelDeployment("model-local-distilbert", "Local DistilBERT Classifier", "Local", "query_classifier_distilbert", ["classifier"], **common),
        ModelDeployment("model-local-t5-classifier", "Local T5-small Classifier", "Local", "query_classifier_t5", ["classifier"], **common),
        ModelDeployment(
            "model-local-distilbert-v2",
            "Local DistilBERT Classifier (4-class)",
            "Local",
            "query_classifier_distilbert_v2",
            ["classifier"],
            enabled=False,
            health_status="untested",
            metadata={
                "builtin": True,
                "complexity_labels": ["simple", "moderate", "complex", "advanced"],
                "artifact_version": 2,
            },
            **{key: value for key, value in common.items() if key not in {"metadata", "enabled", "health_status"}},
        ),
        ModelDeployment(
            "model-local-t5-classifier-v2",
            "Local T5-small Classifier (4-class)",
            "Local",
            "query_classifier_t5_v2",
            ["classifier"],
            enabled=False,
            health_status="untested",
            metadata={
                "builtin": True,
                "complexity_labels": ["simple", "moderate", "complex", "advanced"],
                "artifact_version": 2,
            },
            **{key: value for key, value in common.items() if key not in {"metadata", "enabled", "health_status"}},
        ),
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


def _validated_connection(connection: ModelConnection) -> ModelConnection:
    name = " ".join(str(connection.name or "").split())
    provider = str(connection.provider or "").strip().lower()
    access_path = str(connection.access_path or "").strip().lower()
    if not name:
        raise ModelFarmError("Connection name is required.")
    if provider not in MODEL_CONNECTION_PROVIDERS:
        raise ModelFarmError(f"Unsupported model connection provider: {provider or 'missing'}.")
    if access_path not in MODEL_ACCESS_PATHS:
        raise ModelFarmError(f"Unsupported model access path: {access_path or 'missing'}.")
    api_base = str(connection.api_base or _default_api_base(provider)).strip().rstrip("/")
    parsed = urllib.parse.urlparse(api_base) if api_base else None
    if parsed and (parsed.username or parsed.password):
        raise ModelFarmError("Do not include credentials in the model connection URL.")
    if provider in {"openrouter", "openai", "gemini"} and parsed and parsed.scheme != "https":
        raise ModelFarmError("Remote provider connections require an HTTPS API base URL.")
    if provider in {"ollama", "vllm"} and parsed and parsed.scheme not in {"http", "https"}:
        raise ModelFarmError("Local model connection URLs must use HTTP or HTTPS.")
    refs: Dict[str, str] = {}
    for key, env_name in connection.credential_env_refs.items():
        clean_key = str(key).strip()
        clean_name = str(env_name).strip()
        if clean_name and not clean_name.startswith(SECRET_ENV_PREFIX):
            raise ModelFarmError(f"Credential reference '{clean_name}' must start with {SECRET_ENV_PREFIX}.")
        if clean_key and clean_name:
            refs[clean_key] = clean_name
    locality = connection.locality if connection.locality in {"local", "remote"} else "remote"
    return replace(
        connection, name=name, provider=provider, access_path=access_path,
        api_base=api_base, credential_env_refs=refs, locality=locality,
    )


def _connection_provider(deployment: ModelDeployment) -> str:
    provider = str(deployment.provider or "").strip().lower()
    api_base = str(deployment.api_base or "").lower()
    model = str(deployment.model or "").lower()
    if provider == "local":
        return "local_builtin"
    if provider == "openrouter" or "openrouter.ai" in api_base or model.startswith("openrouter/") or model.endswith(":free"):
        return "openrouter"
    if provider in {"gemini", "google"}:
        return "gemini"
    if provider in {"ollama", "ollama_chat"}:
        return "ollama"
    if provider in {"vllm", "hosted_vllm"}:
        return "vllm"
    return "openai"


def _default_access_path(provider: Any) -> str:
    normalized = str(provider or "").strip().lower()
    if normalized == "openrouter":
        return "experimentation"
    if normalized in {"local_builtin", "ollama", "vllm"}:
        return "local"
    return "production"


def _default_locality(provider: Any) -> str:
    return "local" if str(provider or "").strip().lower() in {"local_builtin", "ollama", "vllm"} else "remote"


def _default_api_base(provider: Any) -> str:
    return {
        "openrouter": "https://openrouter.ai/api/v1",
        "openai": "https://api.openai.com/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta",
        "ollama": "http://127.0.0.1:11434",
        "vllm": "http://127.0.0.1:8001/v1",
    }.get(str(provider or "").strip().lower(), "")


def _should_pass_litellm_api_base(connection: ModelConnection) -> bool:
    if connection.provider != "gemini":
        return True
    hostname = (urllib.parse.urlparse(connection.api_base).hostname or "").lower()
    # LiteLLM's native Gemini adapter owns the Google AI Studio endpoint path.
    # Supplying the official REST catalog base can incorrectly select Vertex AI.
    return hostname != "generativelanguage.googleapis.com"


def _connection_runtime_changed(connection: ModelConnection, payload: Dict[str, Any]) -> bool:
    return any(
        key in payload and payload[key] != getattr(connection, key)
        for key in {"api_base", "credential_env_refs", "credential_secrets"}
    )


def _legacy_connection_for_deployment(deployment: ModelDeployment) -> ModelConnection:
    now = deployment.updated_at or deployment.created_at or utc_now()
    provider = _connection_provider(deployment)
    return ModelConnection(
        id=f"connection-legacy-{deployment.id}",
        name=f"{deployment.name} connection",
        provider=provider,
        access_path=_default_access_path(provider),
        api_base=deployment.api_base,
        credential_env_refs=deployment.credential_env_refs,
        credential_secrets=deployment.credential_secrets,
        locality=_default_locality(provider),
        enabled=deployment.enabled,
        health_status=deployment.health_status,
        last_health_check=deployment.last_health_check,
        last_error=deployment.last_error,
        metadata={"legacy": True},
        created_at=now,
        updated_at=now,
    )


def _discover_connection_models(connection: ModelConnection, credentials: Dict[str, str]) -> List[Dict[str, Any]]:
    if connection.provider == "local_builtin":
        return []
    api_key = str(credentials.get("api_key") or "").strip()
    headers = {"Accept": "application/json"}
    if api_key and connection.provider != "gemini":
        headers["Authorization"] = f"Bearer {api_key}"
    if connection.provider == "openrouter":
        base = (connection.api_base or "https://openrouter.ai/api/v1").rstrip("/")
        url = base if base.endswith("/models") else f"{base}/models"
    elif connection.provider == "openai":
        base = (connection.api_base or "https://api.openai.com/v1").rstrip("/")
        url = base if base.endswith("/models") else f"{base}/models"
    elif connection.provider == "gemini":
        if not api_key:
            raise ModelFarmError(f"Connection '{connection.name}' is missing credentials: api_key.")
        base = (connection.api_base or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        url = base if base.endswith("/models") else f"{base}/models"
        url = f"{url}?{urllib.parse.urlencode({'key': api_key})}"
    elif connection.provider == "ollama":
        base = connection.api_base.rstrip("/")
        if not base:
            raise ModelFarmError(f"Connection '{connection.name}' requires an API base URL.")
        url = f"{base}/api/tags"
    elif connection.provider == "vllm":
        base = connection.api_base.rstrip("/")
        if not base:
            raise ModelFarmError(f"Connection '{connection.name}' requires an API base URL.")
        url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
    else:
        raise ModelFarmError(f"Model discovery is not supported for provider '{connection.provider}'.")
    payload = _connection_request_json(url, headers, timeout=30)
    raw_items = payload.get("models") if connection.provider in {"gemini", "ollama"} else payload.get("data")
    if not isinstance(raw_items, list):
        raise ModelFarmError("The provider model catalog returned an unexpected response.")
    models: List[Dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or item.get("name") or item.get("model") or "").strip()
        if connection.provider == "gemini":
            model_id = model_id.removeprefix("models/")
            methods = list(item.get("supportedGenerationMethods") or [])
            if methods and not any(method in {"generateContent", "embedContent", "batchEmbedContents"} for method in methods):
                continue
        if not model_id:
            continue
        models.append(
            {
                "id": model_id,
                "name": str(item.get("displayName") or item.get("name") or model_id),
                "context_length": int(item.get("context_length") or item.get("inputTokenLimit") or 0),
                "metadata": {
                    key: item[key]
                    for key in ("owned_by", "pricing", "supported_parameters", "supportedGenerationMethods")
                    if key in item
                },
            }
        )
    models.sort(key=lambda item: item["id"].lower())
    return models


def _connection_request_json(url: str, headers: Dict[str, str], *, timeout: float) -> Dict[str, Any]:
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ModelFarmError(f"Connection request failed ({exc.code}): {_provider_error_detail(detail)}") from exc
    except urllib.error.URLError as exc:
        raise ModelFarmError(f"Connection request failed: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise ModelFarmError("The provider returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise ModelFarmError("The provider returned an unexpected response.")
    return payload


def _provider_error_detail(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return _safe_error(raw)
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        return _safe_error(error.get("message") or error.get("code") or error)
    return _safe_error(payload)


def _connection_from_dict(payload: Dict[str, Any]) -> ModelConnection:
    allowed = {item.name for item in ModelConnection.__dataclass_fields__.values()}
    return ModelConnection(**{key: value for key, value in payload.items() if key in allowed})


def _deployment_from_dict(payload: Dict[str, Any]) -> ModelDeployment:
    allowed = {field.name for field in ModelDeployment.__dataclass_fields__.values()}
    return ModelDeployment(**{key: value for key, value in payload.items() if key in allowed})


def _usage_from_dict(payload: Dict[str, Any]) -> ModelUsageEvent:
    allowed = {field.name for field in ModelUsageEvent.__dataclass_fields__.values()}
    return ModelUsageEvent(**{key: value for key, value in payload.items() if key in allowed})


def _deployment_from_row(row: Any) -> ModelDeployment:
    return ModelDeployment(
        id=row["id"], name=row["name"], provider=row["provider"], model=row["model"],
        capabilities=list(row.get("capabilities_json") or []), connection_id=row.get("connection_id") or "",
        api_base=row.get("api_base") or "",
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
        capability=row["capability"], purpose=row["purpose"], status=row["status"],
        connection_id=row.get("connection_id") or "", access_path=row.get("access_path") or "",
        gateway_model=row.get("gateway_model") or "", request_id=row.get("request_id") or "",
        user_id=row.get("user_id") or "", conversation_id=row.get("conversation_id") or "",
        knowledge_base_id=row.get("knowledge_base_id") or "", evaluation_run_id=row.get("evaluation_run_id") or "",
        chat_configuration_id=row.get("chat_configuration_id") or "",
        input_tokens=int(row.get("input_tokens") or 0), output_tokens=int(row.get("output_tokens") or 0),
        total_tokens=int(row.get("total_tokens") or 0), latency_ms=float(row.get("latency_ms") or 0),
        estimated_cost_usd=float(row.get("estimated_cost_usd") or 0), fallback_index=int(row.get("fallback_index") or 0),
        error_code=row.get("error_code") or "", error=row.get("error") or "", metadata=dict(row.get("metadata_json") or {}),
        created_at=row.get("created_at") or "",
    )


def _connection_from_row(row: Any) -> ModelConnection:
    return ModelConnection(
        id=row["id"], name=row["name"], provider=row["provider"], access_path=row["access_path"],
        api_base=row.get("api_base") or "", credential_env_refs=dict(row.get("credential_env_refs_json") or {}),
        credential_secrets=dict(row.get("credential_secrets_json") or {}), locality=row.get("locality") or "remote",
        enabled=bool(row.get("enabled", False)), health_status=row.get("health_status") or "untested",
        last_health_check=row.get("last_health_check") or "", last_error=row.get("last_error") or "",
        metadata=dict(row.get("metadata_json") or {}), created_at=row.get("created_at") or "",
        updated_at=row.get("updated_at") or "",
    )


def _gateway_model_name(provider: str, model: str, api_base: str = "") -> str:
    normalized_provider = str(provider or "").strip().lower()
    normalized_model = str(model or "").strip()
    if normalized_provider == "local_builtin":
        return normalized_model
    if normalized_provider == "openrouter":
        return normalized_model if normalized_model.startswith("openrouter/") else f"openrouter/{normalized_model}"
    if normalized_provider == "gemini":
        native = normalized_model.removeprefix("gemini/")
        return f"gemini/{native}"
    if normalized_provider == "ollama":
        native = normalized_model.removeprefix("ollama_chat/").removeprefix("ollama/")
        return f"ollama_chat/{native}"
    if normalized_provider == "vllm":
        native = normalized_model.removeprefix("hosted_vllm/").removeprefix("vllm/")
        return f"hosted_vllm/{native}"
    if normalized_provider == "openai":
        native = normalized_model.removeprefix("openai/")
        if api_base and "api.openai.com" not in api_base.lower():
            return f"openai/{native}"
        return native
    raise ModelFarmError(f"Unsupported model connection provider: {normalized_provider or 'missing'}.")


def _is_openrouter_fields(*, model: str = "", api_base: str = "", provider: str = "", provider_label: str = "") -> bool:
    normalized_model = str(model or "").lower().strip()
    normalized_base = str(api_base or "").lower().strip()
    normalized_provider = str(provider or "").lower().strip()
    normalized_label = str(provider_label or "").lower().strip()
    return (
        normalized_provider == "openrouter"
        or "openrouter.ai" in normalized_base
        or "openrouter" in normalized_label
        or normalized_model.startswith("openrouter/")
        or normalized_model.endswith(":free")
    )


def _deployment_db_payload(deployment: ModelDeployment) -> Dict[str, Any]:
    payload = asdict(deployment)
    for key in ["capabilities", "credential_env_refs", "credential_secrets", "default_parameters", "limits", "pricing", "metadata"]:
        payload[key] = json.dumps(payload[key])
    return payload


def _connection_db_payload(connection: ModelConnection) -> Dict[str, Any]:
    payload = asdict(connection)
    for key in ["credential_env_refs", "credential_secrets", "metadata"]:
        payload[key] = json.dumps(payload[key])
    return payload


def _secret_key_bytes(secret_key: str) -> bytes:
    if not secret_key:
        raise ModelFarmError("ARAGBIZ_MODEL_SECRET_KEY is required for stored model credentials.")
    if len(secret_key) < 16:
        raise ModelFarmError("ARAGBIZ_MODEL_SECRET_KEY must contain at least 16 characters.")
    return hashlib.sha256(secret_key.encode("utf-8")).digest()


def _secret_stream(key: bytes, nonce: bytes, length: int) -> bytes:
    blocks: List[bytes] = []
    counter = 0
    while sum(len(block) for block in blocks) < length:
        blocks.append(hashlib.sha256(key + nonce + counter.to_bytes(4, "big")).digest())
        counter += 1
    return b"".join(blocks)[:length]


def _encrypt_secret(value: str, secret_key: str) -> str:
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency is included in the API extra
        raise ModelFarmError("Install the api extra to encrypt stored model credentials.") from exc
    _KNOWN_SECRET_VALUES.add(value)
    nonce = os.urandom(12)
    ciphertext = AESGCM(_secret_key_bytes(secret_key)).encrypt(
        nonce,
        value.encode("utf-8"),
        b"aragbiz:model-secret:v2",
    )
    return "v2:" + ":".join(
        base64.urlsafe_b64encode(item).decode("ascii").rstrip("=")
        for item in (nonce, ciphertext)
    )


def _decrypt_secret(token: str, secret_key: str) -> str:
    if token.startswith("v2:"):
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
            _, nonce_raw, ciphertext_raw = token.split(":", 2)
            plaintext = AESGCM(_secret_key_bytes(secret_key)).decrypt(
                _urlsafe_b64decode(nonce_raw),
                _urlsafe_b64decode(ciphertext_raw),
                b"aragbiz:model-secret:v2",
            )
            return plaintext.decode("utf-8")
        except Exception as exc:
            raise ModelFarmError("Stored model credential could not be decrypted with the configured key.") from exc
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
    connection_id: str = "",
    access_path: str = "",
    gateway_model: str = "",
) -> ModelUsageEvent:
    context = context or ModelCallContext(purpose=capability)
    return ModelUsageEvent(
        id=f"usage-{uuid.uuid4().hex}", deployment_id=deployment.id, provider=deployment.provider, model=deployment.model,
        capability=capability, purpose=context.purpose or capability, status=status, request_id=context.request_id,
        connection_id=connection_id, access_path=access_path, gateway_model=gateway_model,
        user_id=context.user_id, conversation_id=context.conversation_id, knowledge_base_id=context.knowledge_base_id,
        evaluation_run_id=context.evaluation_run_id, chat_configuration_id=context.chat_configuration_id,
        input_tokens=input_tokens, output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens, latency_ms=round(latency_ms, 3), estimated_cost_usd=round(cost, 10),
        fallback_index=fallback_index, error_code=type(error).__name__ if error else "", error=_safe_error(error) if error else "",
        metadata={"error_category": _error_category(error)} if error else {},
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


def _litellm_pricing_known(litellm: Any, resolved: ResolvedModel) -> bool:
    if resolved.deployment.model.lower().endswith(":free"):
        return True
    if any(float(value or 0) > 0 for value in resolved.deployment.pricing.values() if isinstance(value, (int, float))):
        return True
    model_cost = getattr(litellm, "model_cost", {})
    if not isinstance(model_cost, dict):
        return False
    candidates = {
        resolved.gateway_model,
        resolved.deployment.model,
        resolved.gateway_model.removeprefix("openrouter/"),
        resolved.gateway_model.removeprefix("gemini/"),
    }
    return any(candidate in model_cost for candidate in candidates)


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
        payload = chunk if isinstance(chunk, dict) else getattr(chunk, "model_dump", lambda: {})()
        choices = payload.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        return str(delta.get("content") or "")


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
    for value in tuple(_KNOWN_SECRET_VALUES):
        if value:
            text = text.replace(value, "[REDACTED]")
    return text[:1000]


def _failed_model_attempt(
    deployment: ModelDeployment,
    error: Exception,
    fallback_index: int,
    latency_ms: float,
    *,
    gateway_model: str = "",
    connection_id: str = "",
    access_path: str = "",
) -> Dict[str, Any]:
    return {
        "deployment_id": deployment.id,
        "deployment_name": deployment.name,
        "provider": deployment.provider,
        "model": deployment.model,
        "gateway_model": gateway_model,
        "connection_id": connection_id or deployment.connection_id,
        "access_path": access_path,
        "fallback_index": fallback_index,
        "latency_ms": round(latency_ms, 3),
        "error_category": _error_category(error),
        "retryable": _is_retryable_error(error),
        "error": _safe_error(error),
    }


def _is_retryable_error(error: Exception) -> bool:
    return _error_category(error) in {"rate_limit", "timeout", "unavailable"}


def _error_category(error: Any) -> str:
    if error is None:
        return ""
    name = type(error).__name__.lower()
    text = str(error).lower()
    combined = f"{name} {text}"
    if "budget" in combined:
        return "budget"
    if "policy" in combined or "egress" in combined:
        return "policy"
    if any(value in combined for value in ("authentication", "unauthorized", "invalid api key", "permission")):
        return "authentication"
    if any(value in combined for value in ("ratelimit", "rate limit", "too many requests", "429")):
        return "rate_limit"
    if any(value in combined for value in ("timeout", "timed out")):
        return "timeout"
    if any(value in combined for value in ("connection", "unavailable", "service unavailable", "502", "503", "504")):
        return "unavailable"
    if any(value in combined for value in ("badrequest", "bad request", "invalidrequest", "invalid request", "not found", "404")):
        return "invalid_request"
    return "unknown"


def _rough_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4)) if text else 0


def _classification_label(value: Any) -> str:
    return str(_classification_payload(value)["label"])


def _classification_payload(value: Any) -> Dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = text
    raw_payload = payload if isinstance(payload, dict) else {"label": payload}
    label = str(raw_payload.get("label") or "").strip().lower().strip("\"'")
    labels = ("simple", "moderate", "complex", "advanced")
    if label not in labels:
        raise ModelFarmError(
            "Classifier returned an invalid label. Expected one of: simple, moderate, complex, advanced."
        )
    raw_probabilities = raw_payload.get("probabilities")
    if isinstance(raw_probabilities, dict):
        probabilities = {candidate: max(float(raw_probabilities.get(candidate, 0.0)), 0.0) for candidate in labels}
        total = sum(probabilities.values())
        probabilities = {candidate: value / total for candidate, value in probabilities.items()} if total else {}
    else:
        probabilities = {}
    if not probabilities:
        supplied_confidence = max(0.0, min(float(raw_payload.get("confidence", 1.0)), 1.0))
        remainder = (1.0 - supplied_confidence) / (len(labels) - 1)
        probabilities = {candidate: supplied_confidence if candidate == label else remainder for candidate in labels}
    ordered = sorted(probabilities.values(), reverse=True)
    confidence = probabilities[label]
    margin = max(confidence - (ordered[1] if len(ordered) > 1 else 0.0), 0.0)
    return {
        "label": label,
        "probabilities": probabilities,
        "confidence": confidence,
        "margin": margin,
    }


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


async def _stream_local_flan(
    model_name: str,
    prompt: str,
    parameters: Dict[str, Any],
    cancellation_token: Optional[CancellationToken] = None,
) -> AsyncIterator[str]:
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer  # type: ignore
    except ImportError as exc:
        raise ModelFarmError("Install the ml extra to stream local FLAN-T5.") from exc
    tokenizer, model = await asyncio.to_thread(_load_local_flan, model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    errors: List[BaseException] = []

    class CancellationStoppingCriteria(StoppingCriteria):
        def __call__(self, *args: Any, **kwargs: Any) -> bool:
            return bool(cancellation_token and cancellation_token.is_cancelled)

    def run_generate() -> None:
        try:
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=int(parameters.get("max_tokens") or parameters.get("max_new_tokens") or 160),
                stopping_criteria=StoppingCriteriaList([CancellationStoppingCriteria()]),
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
            if cancellation_token:
                cancellation_token.raise_if_cancelled()
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
