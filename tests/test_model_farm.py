import asyncio
import base64
import hashlib
import hmac
import json

import pytest

from aragbiz.cancellation import AnswerCancelled, CancellationToken
from aragbiz.model_farm import (
    JsonModelFarmRepository,
    ModelCallContext,
    ModelConnection,
    ModelFarmError,
    ModelFarmService,
    ModelGateway,
    _gateway_model_name,
)


TEST_SECRET_KEY = "unit-test-model-secret-key"


def build_service(tmp_path, *, secret_key=TEST_SECRET_KEY):
    return ModelFarmService(
        JsonModelFarmRepository(tmp_path / "model_farm.json"),
        secret_key=secret_key,
    )


def create_remote_connection(service, provider="openrouter", *, name="Test connection"):
    defaults = {
        "openrouter": ("experimentation", "https://openrouter.ai/api/v1"),
        "openai": ("production", "https://api.openai.com/v1"),
        "gemini": ("production", "https://generativelanguage.googleapis.com/v1beta"),
    }
    access_path, api_base = defaults[provider]
    return service.create_connection(
        {
            "name": name,
            "provider": provider,
            "access_path": access_path,
            "api_base": api_base,
            "credential_secrets": {"api_key": "sk-unit-test"},
            "health_status": "healthy",
            "enabled": True,
        }
    )


def create_remote_deployment(service, connection, *, model="google/gemma-3-27b-it:free", name="Test model"):
    template_id = {
        "openrouter": "openrouter-generation",
        "openai": "openai-generation",
        "gemini": "gemini-generation",
    }[connection.provider]
    deployment = service.create_deployment_from_template(
        template_id,
        {
            "connection_id": connection.id,
            "name": name,
            "model": model,
            "capabilities": ["generation", "judge", "planner"],
        },
    )
    service.set_health(deployment.id, healthy=True)
    return service.update_deployment(deployment.id, {"enabled": True})


def test_local_deployments_and_connection_are_seeded(tmp_path):
    service = build_service(tmp_path)

    assert {item.id for item in service.list_deployments()} >= {
        "model-local-extractive",
        "model-local-hash-384",
        "model-local-lexical-reranker",
    }
    connection = service.get_connection("connection-local-builtin")
    assert connection.provider == "local_builtin"
    assert connection.enabled is True
    assert service.resolve("model-local-extractive", "generation").connection_id == connection.id


def test_provider_templates_cover_supported_access_paths(tmp_path):
    templates = build_service(tmp_path).providers()
    ids = {item["id"] for item in templates}

    assert ids >= {
        "local-extractive",
        "openrouter-generation",
        "openai-generation",
        "openai-embedding",
        "gemini-generation",
        "ollama-generation",
        "vllm-generation",
    }
    assert next(item for item in templates if item["id"] == "openrouter-generation")["access_path"] == "experimentation"
    assert next(item for item in templates if item["id"] == "openai-generation")["access_path"] == "production"
    assert next(item for item in templates if item["id"] == "ollama-generation")["access_path"] == "local"


@pytest.mark.parametrize(
    ("provider", "model", "api_base", "expected"),
    [
        ("openrouter", "google/gemma-3-27b-it:free", "https://openrouter.ai/api/v1", "openrouter/google/gemma-3-27b-it:free"),
        ("openai", "gpt-4.1-mini", "https://api.openai.com/v1", "gpt-4.1-mini"),
        ("gemini", "gemini-2.5-flash", "https://generativelanguage.googleapis.com/v1beta", "gemini/gemini-2.5-flash"),
        ("ollama", "llama3.1", "http://127.0.0.1:11434", "ollama_chat/llama3.1"),
        ("vllm", "meta-llama/Llama-3.1-8B-Instruct", "http://127.0.0.1:8001/v1", "hosted_vllm/meta-llama/Llama-3.1-8B-Instruct"),
    ],
)
def test_gateway_model_name_normalization(provider, model, api_base, expected):
    assert _gateway_model_name(provider, model, api_base) == expected


def test_connection_secret_is_aes_gcm_encrypted_and_redacted(tmp_path):
    service = build_service(tmp_path)
    connection = create_remote_connection(service)
    raw_store = (tmp_path / "model_farm.json").read_text(encoding="utf-8")

    assert "sk-unit-test" not in raw_store
    assert connection.credential_secrets["api_key"].startswith("v2:")
    assert service.credential_values(connection)["api_key"] == "sk-unit-test"
    assert service.credential_status(connection)["stored_secret_keys"] == ["api_key"]


def test_stored_credentials_require_configured_encryption_key(tmp_path):
    service = build_service(tmp_path, secret_key="")

    with pytest.raises(ModelFarmError, match="ARAGBIZ_MODEL_SECRET_KEY"):
        service.create_connection(
            {
                "name": "No encryption key",
                "provider": "openrouter",
                "access_path": "experimentation",
                "credential_secrets": {"api_key": "must-not-be-stored"},
            }
        )


def test_v1_credentials_remain_readable_and_upgrade_on_save(tmp_path):
    service = build_service(tmp_path)
    connection = create_remote_connection(service)
    legacy_token = _legacy_v1_encrypt("legacy-key", TEST_SECRET_KEY)
    service.repository.save_connection(
        ModelConnection(
            **{
                **connection.__dict__,
                "credential_secrets": {"api_key": legacy_token},
            }
        )
    )

    assert service.credential_values(service.get_connection(connection.id))["api_key"] == "legacy-key"
    updated = service.update_connection(connection.id, {"name": "Updated connection"})
    assert updated.credential_secrets["api_key"].startswith("v2:")


def test_template_deployment_reuses_connection_and_keeps_native_model_id(tmp_path):
    service = build_service(tmp_path)
    connection = create_remote_connection(service)
    deployment = service.create_deployment_from_template(
        "openrouter-generation",
        {
            "connection_id": connection.id,
            "name": "OpenRouter Gemma",
            "model": "google/gemma-3-27b-it:free",
            "capabilities": ["generation", "judge"],
        },
    )

    assert deployment.connection_id == connection.id
    assert deployment.model == "google/gemma-3-27b-it:free"
    assert deployment.api_base == ""
    assert deployment.credential_secrets == {}
    assert service.connection_for_deployment(deployment, require_enabled=False).api_base == "https://openrouter.ai/api/v1"


def test_remote_connection_cannot_enable_before_successful_test(tmp_path):
    service = build_service(tmp_path)
    with pytest.raises(ModelFarmError, match="Test a remote connection"):
        service.create_connection(
            {
                "name": "Untested OpenAI",
                "provider": "openai",
                "access_path": "production",
                "credential_env_refs": {"api_key": "ARAGBIZ_MODEL_OPENAI_API_KEY"},
                "enabled": True,
            }
        )


def test_litellm_generation_uses_connection_mapping_and_records_usage(tmp_path):
    service = build_service(tmp_path)
    connection = create_remote_connection(service)
    deployment = create_remote_deployment(service, connection)
    gateway = ModelGateway(service)
    fake = FakeLiteLLM()
    gateway.litellm_adapter._module = lambda: fake

    result = asyncio.run(
        gateway.generate(
            [{"role": "user", "content": "Hello"}],
            deployment.id,
            context=ModelCallContext(purpose="unit-test", request_id="request-1"),
        )
    )

    assert result.text == "ready"
    assert fake.calls[0]["model"] == "openrouter/google/gemma-3-27b-it:free"
    assert fake.calls[0]["api_base"] == "https://openrouter.ai/api/v1"
    assert fake.calls[0]["api_key"] == "sk-unit-test"
    usage = service.list_usage(purpose="unit-test")[0]
    assert usage.connection_id == connection.id
    assert usage.access_path == "experimentation"
    assert usage.gateway_model == "openrouter/google/gemma-3-27b-it:free"


def test_gateway_generation_can_record_planner_capability(tmp_path):
    service = build_service(tmp_path)
    connection = create_remote_connection(service)
    deployment = create_remote_deployment(service, connection)
    gateway = ModelGateway(service)
    gateway.litellm_adapter._module = lambda: FakeLiteLLM()

    result = asyncio.run(
        gateway.generate(
            [{"role": "user", "content": "Rewrite this follow-up."}],
            deployment.id,
            capability="planner",
            context=ModelCallContext(purpose="conversation_rewrite", request_id="request-planner"),
        )
    )

    assert result.text == "ready"
    usage = service.list_usage(purpose="conversation_rewrite")[0]
    assert usage.capability == "planner"
    assert usage.request_id == "request-planner"


def test_local_classifier_deployment_returns_normalized_result_and_records_usage(monkeypatch, tmp_path):
    service = build_service(tmp_path)
    gateway = ModelGateway(service)

    class StubLocalClassifier:
        def predict(self, query):
            assert "invoice" in query.lower()
            return "moderate"

    monkeypatch.setattr(gateway, "_local_classifier", lambda deployment: StubLocalClassifier())
    result = asyncio.run(
        gateway.classify(
            "How should an invoice mismatch be handled?",
            "model-local-distilbert",
            context=ModelCallContext(purpose="query_classification", request_id="request-classifier"),
        )
    )

    assert result.label == "moderate"
    assert result.deployment_id == "model-local-distilbert"
    assert result.metadata["runtime"] == "huggingface-sequence-classification"
    usage = service.list_usage(purpose="query_classification")[0]
    assert usage.capability == "classifier"
    assert usage.request_id == "request-classifier"


def test_remote_classifier_accepts_only_normalized_complexity_labels(tmp_path):
    service = build_service(tmp_path)
    connection = create_remote_connection(service)
    deployment = service.create_deployment_from_template(
        "openrouter-generation",
        {
            "connection_id": connection.id,
            "name": "Remote classifier",
            "model": "google/gemma-3-27b-it:free",
            "capabilities": ["classifier"],
        },
    )
    service.set_health(deployment.id, healthy=True)
    deployment = service.update_deployment(deployment.id, {"enabled": True})
    gateway = ModelGateway(service)
    gateway.litellm_adapter._module = lambda: FakeLiteLLM(response_text='{"label":"complex"}')

    result = asyncio.run(gateway.classify("Compare approval paths.", deployment.id))

    assert result.label == "complex"
    assert result.metadata["gateway_model"].startswith("openrouter/")


def test_remote_classifier_rejects_unconstrained_output(tmp_path):
    service = build_service(tmp_path)
    connection = create_remote_connection(service)
    deployment = service.create_deployment_from_template(
        "openrouter-generation",
        {
            "connection_id": connection.id,
            "name": "Invalid remote classifier",
            "model": "google/gemma-3-27b-it:free",
            "capabilities": ["classifier"],
        },
    )
    service.set_health(deployment.id, healthy=True)
    deployment = service.update_deployment(deployment.id, {"enabled": True})
    gateway = ModelGateway(service)
    gateway.litellm_adapter._module = lambda: FakeLiteLLM(response_text="This looks moderately difficult.")

    with pytest.raises(ModelFarmError, match="invalid label"):
        asyncio.run(gateway.classify("Compare approval paths.", deployment.id))


def test_litellm_stream_forwards_provider_deltas(tmp_path):
    service = build_service(tmp_path)
    connection = create_remote_connection(service)
    deployment = create_remote_deployment(service, connection)
    gateway = ModelGateway(service)
    gateway.litellm_adapter._module = lambda: FakeLiteLLM(stream_parts=["Hel", "lo"])

    async def collect():
        return [event async for event in gateway.stream([{"role": "user", "content": "Hello"}], deployment.id)]

    events = asyncio.run(collect())
    assert [event.data["text"] for event in events if event.type == "delta"] == ["Hel", "lo"]
    completed = next(event for event in events if event.type == "model_completed")
    assert completed.data["metadata"]["gateway_model"].startswith("openrouter/")


def test_official_gemini_api_base_is_not_forwarded_to_litellm(tmp_path):
    service = build_service(tmp_path)
    connection = create_remote_connection(service, provider="gemini")
    deployment = create_remote_deployment(
        service,
        connection,
        model="gemini-2.5-flash",
        name="Gemini Flash",
    )
    gateway = ModelGateway(service)
    fake = FakeLiteLLM()
    fake.model_cost = {"gemini/gemini-2.5-flash": {}}
    gateway.litellm_adapter._module = lambda: fake

    result = asyncio.run(gateway.test_deployment(deployment.id))

    assert result["status"] == "healthy"
    assert fake.calls[0]["model"] == "gemini/gemini-2.5-flash"
    assert fake.calls[0]["api_key"] == "sk-unit-test"
    assert fake.calls[0]["max_tokens"] == 128
    assert "api_base" not in fake.calls[0]


def test_rate_limited_model_test_preserves_healthy_connection(tmp_path):
    service = build_service(tmp_path)
    connection = create_remote_connection(service)
    deployment = create_remote_deployment(service, connection)
    gateway = ModelGateway(service)
    gateway.litellm_adapter._module = lambda: RateLimitedLiteLLM()

    result = asyncio.run(gateway.test_deployment(deployment.id))

    assert result["status"] == "rate_limited"
    assert result["error_category"] == "rate_limit"
    assert result["retryable"] is True
    assert service.get_deployment(deployment.id).health_status == "rate_limited"
    assert service.get_connection(connection.id).health_status == "healthy"


def test_rate_limited_draft_test_returns_retryable_status(tmp_path):
    service = build_service(tmp_path)
    connection = create_remote_connection(service)
    deployment = create_remote_deployment(service, connection)
    gateway = ModelGateway(service)
    gateway.litellm_adapter._module = lambda: RateLimitedLiteLLM()

    result = asyncio.run(gateway.test_draft_deployment(deployment))

    assert result["status"] == "rate_limited"
    assert result["retryable"] is True
    assert result["deployment"].health_status == "rate_limited"


def test_generation_fallback_uses_next_deployment_only_for_retryable_failure(tmp_path):
    service = build_service(tmp_path)
    first_connection = create_remote_connection(service, name="Primary connection")
    second_connection = create_remote_connection(service, provider="openai", name="Fallback connection")
    first = create_remote_deployment(service, first_connection, name="Primary model")
    second = create_remote_deployment(service, second_connection, model="gpt-4.1-mini", name="Fallback model")
    gateway = ModelGateway(service)
    fake = FakeLiteLLM(fail_models={"openrouter/google/gemma-3-27b-it:free"})
    gateway.litellm_adapter._module = lambda: fake

    result = asyncio.run(
        gateway.generate(
            [{"role": "user", "content": "Hello"}],
            first.id,
            fallback_deployment_ids=[second.id],
        )
    )

    assert result.deployment_id == second.id
    assert result.metadata["fallback_index"] == 1
    assert result.metadata["fallback_attempts"][0]["deployment_id"] == first.id
    assert result.metadata["fallback_attempts"][0]["error_category"] == "timeout"
    assert [call["model"] for call in fake.calls] == [
        "openrouter/google/gemma-3-27b-it:free",
        "gpt-4.1-mini",
    ]


def test_local_generation_embedding_and_reranking_record_usage(tmp_path):
    service = build_service(tmp_path)
    gateway = ModelGateway(service)
    context = ModelCallContext(purpose="local-test", request_id="req-1")

    generation = gateway.generate_sync(
        [{"role": "user", "content": "What is UAT?\nContext: UAT validates readiness before go-live."}],
        "model-local-extractive",
        context=context,
    )
    embedding = gateway.embed_sync(["What is UAT?"], "model-local-hash-384", context=context)
    reranked = gateway.rerank_sync(
        "UAT readiness",
        ["Payroll setup", "UAT validates readiness before go-live"],
        "model-local-lexical-reranker",
        top_n=2,
        context=context,
    )

    assert "UAT" in generation.text
    assert embedding.dimension == 384
    assert reranked.items[0].index == 1
    assert all(item.connection_id == "connection-local-builtin" for item in service.list_usage(purpose="local-test"))


def test_connection_with_deployments_cannot_be_deleted(tmp_path):
    service = build_service(tmp_path)
    connection = create_remote_connection(service)
    create_remote_deployment(service, connection)

    with pytest.raises(ModelFarmError, match="referenced"):
        service.delete_connection(connection.id)


@pytest.mark.parametrize(
    ("provider", "api_base", "payload", "expected_url", "expected_id"),
    [
        ("ollama", "http://127.0.0.1:11434", {"models": [{"name": "llama3.1"}]}, "http://127.0.0.1:11434/api/tags", "llama3.1"),
        ("vllm", "http://127.0.0.1:8001/v1", {"data": [{"id": "meta-llama/test"}]}, "http://127.0.0.1:8001/v1/models", "meta-llama/test"),
    ],
)
def test_local_server_model_discovery_uses_documented_catalog(monkeypatch, tmp_path, provider, api_base, payload, expected_url, expected_id):
    service = build_service(tmp_path)
    connection = service.create_connection(
        {
            "name": f"{provider} server",
            "provider": provider,
            "access_path": "local",
            "api_base": api_base,
            "locality": "local",
        }
    )
    captured = {}

    def fake_request(url, headers, *, timeout):
        captured.update({"url": url, "headers": headers, "timeout": timeout})
        return payload

    monkeypatch.setattr("aragbiz.model_farm._connection_request_json", fake_request)
    models = service.available_models(connection.id)

    assert captured["url"] == expected_url
    assert models[0]["id"] == expected_id


def test_stream_fallback_stops_after_first_delta(tmp_path):
    service = build_service(tmp_path)
    first_connection = create_remote_connection(service, name="Streaming primary connection")
    second_connection = create_remote_connection(service, provider="openai", name="Streaming fallback connection")
    first = create_remote_deployment(service, first_connection, name="Streaming primary")
    second = create_remote_deployment(service, second_connection, model="gpt-4.1-mini", name="Streaming fallback")
    gateway = ModelGateway(service)
    fake = FailingStreamLiteLLM()
    gateway.litellm_adapter._module = lambda: fake

    async def collect():
        return [
            event
            async for event in gateway.stream(
                [{"role": "user", "content": "Hello"}],
                first.id,
                fallback_deployment_ids=[second.id],
            )
        ]

    with pytest.raises(ModelFarmError, match="streaming failed"):
        asyncio.run(collect())
    assert fake.called_models == ["openrouter/google/gemma-3-27b-it:free"]


def test_stream_fallback_is_allowed_before_first_delta(tmp_path):
    service = build_service(tmp_path)
    first_connection = create_remote_connection(service, name="Pre-delta primary connection")
    second_connection = create_remote_connection(service, provider="openai", name="Pre-delta fallback connection")
    first = create_remote_deployment(service, first_connection, name="Pre-delta primary")
    second = create_remote_deployment(service, second_connection, model="gpt-4.1-mini", name="Pre-delta fallback")
    gateway = ModelGateway(service)
    fake = PreDeltaFailLiteLLM()
    gateway.litellm_adapter._module = lambda: fake

    async def collect():
        return [
            event
            async for event in gateway.stream(
                [{"role": "user", "content": "Hello"}],
                first.id,
                fallback_deployment_ids=[second.id],
            )
        ]

    events = asyncio.run(collect())
    assert [event.data["text"] for event in events if event.type == "delta"] == ["fallback"]
    fallback = next(event for event in events if event.type == "model_fallback")
    assert fallback.data["deployment_id"] == first.id
    assert fallback.data["next_deployment_id"] == second.id
    completed = next(event for event in events if event.type == "model_completed")
    assert completed.data["metadata"]["fallback_attempts"][0]["deployment_id"] == first.id
    assert fake.called_models == ["openrouter/google/gemma-3-27b-it:free", "gpt-4.1-mini"]


def test_stream_cancellation_closes_provider_and_prevents_fallback(tmp_path):
    service = build_service(tmp_path)
    first_connection = create_remote_connection(service, name="Cancellation primary connection")
    second_connection = create_remote_connection(service, provider="openai", name="Cancellation fallback connection")
    first = create_remote_deployment(service, first_connection, name="Cancellation primary")
    second = create_remote_deployment(service, second_connection, model="gpt-4.1-mini", name="Cancellation fallback")
    gateway = ModelGateway(service)
    fake = CancellableLiteLLM()
    gateway.litellm_adapter._module = lambda: fake
    token = CancellationToken("request-cancel-stream")

    async def collect():
        events = []
        async for event in gateway.stream(
            [{"role": "user", "content": "Hello"}],
            first.id,
            fallback_deployment_ids=[second.id],
            context=ModelCallContext(purpose="chat_generation", request_id=token.request_id),
            cancellation_token=token,
        ):
            events.append(event)
            if event.type == "delta":
                token.cancel("Stopped after partial output.")
        return events

    with pytest.raises(AnswerCancelled, match="Stopped after partial output"):
        asyncio.run(collect())

    assert fake.called_models == ["openrouter/google/gemma-3-27b-it:free"]
    assert fake.stream.closed is True
    usage = service.list_usage(purpose="chat_generation")
    assert len(usage) == 1
    assert usage[0].status == "cancelled"
    assert usage[0].request_id == token.request_id
    assert usage[0].output_tokens >= 1


class FakeLiteLLM:
    def __init__(self, *, stream_parts=None, fail_models=None, response_text="ready"):
        self.calls = []
        self.stream_parts = list(stream_parts or ["ready"])
        self.fail_models = set(fail_models or [])
        self.response_text = response_text

    async def acompletion(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["model"] in self.fail_models:
            raise TimeoutError("temporary provider timeout")
        if kwargs.get("stream"):
            return _FakeStream(self.stream_parts)
        return {
            "choices": [{"message": {"content": self.response_text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        }


class RateLimitError(Exception):
    pass


class RateLimitedLiteLLM:
    async def acompletion(self, **kwargs):
        raise RateLimitError("Provider returned 429 too many requests")


class _FakeStream:
    def __init__(self, parts):
        self._parts = iter(parts)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            part = next(self._parts)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        return {"choices": [{"delta": {"content": part}}]}


class FailingStreamLiteLLM:
    model_cost = {"gpt-4.1-mini": {}}

    def __init__(self):
        self.called_models = []

    async def acompletion(self, **kwargs):
        self.called_models.append(kwargs["model"])
        return _PartThenFailureStream()


class _PartThenFailureStream:
    def __init__(self):
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.index += 1
        if self.index == 1:
            return {"choices": [{"delta": {"content": "partial"}}]}
        raise TimeoutError("provider timed out after sending content")


class PreDeltaFailLiteLLM:
    model_cost = {"gpt-4.1-mini": {}}

    def __init__(self):
        self.called_models = []

    async def acompletion(self, **kwargs):
        self.called_models.append(kwargs["model"])
        if kwargs["model"].startswith("openrouter/"):
            return _ImmediateFailureStream()
        return _FakeStream(["fallback"])


class _ImmediateFailureStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise TimeoutError("provider timed out before content")


class CancellableLiteLLM:
    model_cost = {"gpt-4.1-mini": {}}

    def __init__(self):
        self.called_models = []
        self.stream = _CancellableStream()

    async def acompletion(self, **kwargs):
        self.called_models.append(kwargs["model"])
        return self.stream


class _CancellableStream:
    def __init__(self):
        self.index = 0
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.index += 1
        if self.index == 1:
            return {"choices": [{"delta": {"content": "partial response"}}]}
        return {"choices": [{"delta": {"content": "should not be emitted"}}]}

    async def aclose(self):
        self.closed = True


def _legacy_v1_encrypt(value, secret_key):
    key = hashlib.sha256(secret_key.encode("utf-8")).digest()
    nonce = b"legacy-unit-test"
    blocks = []
    counter = 0
    while sum(len(block) for block in blocks) < len(value.encode("utf-8")):
        blocks.append(hashlib.sha256(key + nonce + counter.to_bytes(4, "big")).digest())
        counter += 1
    plaintext = value.encode("utf-8")
    stream = b"".join(blocks)[: len(plaintext)]
    ciphertext = bytes(left ^ right for left, right in zip(plaintext, stream))
    tag = hmac.new(key, nonce + ciphertext, hashlib.sha256).digest()
    encode = lambda item: base64.urlsafe_b64encode(item).decode("ascii").rstrip("=")
    return f"v1:{encode(nonce)}:{encode(ciphertext)}:{encode(tag)}"
