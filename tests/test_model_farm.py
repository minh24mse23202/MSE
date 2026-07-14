import pytest

from aragbiz.model_farm import (
    JsonModelFarmRepository,
    ModelCallContext,
    ModelFarmError,
    ModelFarmService,
    ModelGateway,
)


def build_service(tmp_path):
    return ModelFarmService(JsonModelFarmRepository(tmp_path / "model_farm.json"))


def test_local_deployments_are_seeded(tmp_path):
    service = build_service(tmp_path)

    deployments = service.list_deployments()

    assert {item.id for item in deployments} >= {
        "model-local-extractive",
        "model-local-hash-384",
        "model-local-lexical-reranker",
    }
    assert service.resolve("model-local-extractive", "generation").enabled is True


def test_provider_templates_are_available(tmp_path):
    service = build_service(tmp_path)

    templates = service.providers()

    assert {item["id"] for item in templates} >= {
        "local-extractive",
        "openai-generation",
        "azure-openai-generation",
        "openai-compatible",
        "cohere-generation-rerank",
        "huggingface-generation",
        "bedrock-generation",
    }
    assert next(item for item in templates if item["id"] == "local-extractive")["creatable"] is False
    assert next(item for item in templates if item["id"] == "openai-generation")["creatable"] is True


def test_create_deployment_from_template_uses_unique_disabled_record(tmp_path):
    service = build_service(tmp_path)

    first = service.create_deployment_from_template(
        "openai-generation",
        {
            "name": "OpenAI GPT deployment",
            "credential_env_refs": {"api_key": "ARAGBIZ_MODEL_OPENAI_API_KEY"},
            "capabilities": ["generation", "judge"],
        },
    )
    second = service.create_deployment_from_template(
        "openai-generation",
        {
            "name": "OpenAI GPT deployment",
            "credential_env_refs": {"api_key": "ARAGBIZ_MODEL_OPENAI_API_KEY"},
            "capabilities": ["generation"],
        },
    )

    assert first.enabled is False
    assert first.health_status == "untested"
    assert second.name == "OpenAI GPT deployment 2"
    with pytest.raises(ModelFarmError, match="Test a remote deployment"):
        service.update_deployment(first.id, {"enabled": True})


def test_create_deployment_encrypts_stored_api_key(tmp_path):
    service = build_service(tmp_path)

    deployment = service.create_deployment_from_template(
        "openai-generation",
        {
            "name": "OpenAI secret deployment",
            "credential_secrets": {"api_key": "sk-unit-test"},
            "capabilities": ["generation"],
        },
    )

    raw_store = (tmp_path / "model_farm.json").read_text(encoding="utf-8")
    status = service.credential_status(deployment)

    assert "sk-unit-test" not in raw_store
    assert deployment.credential_secrets["api_key"].startswith("v1:")
    assert status["configured"] is True
    assert status["stored_secret_keys"] == ["api_key"]


def test_builtin_template_cannot_create_duplicate_local_deployment(tmp_path):
    service = build_service(tmp_path)

    with pytest.raises(ModelFarmError, match="already registered"):
        service.create_deployment_from_template("local-extractive", {})


def test_remote_credential_references_must_use_project_prefix(tmp_path):
    service = build_service(tmp_path)

    with pytest.raises(ModelFarmError, match="ARAGBIZ_MODEL_"):
        service.create_deployment(
            {
                "name": "Bad credential",
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "capabilities": ["generation"],
                "credential_env_refs": {"api_key": "OPENAI_API_KEY"},
            }
        )


def test_local_generation_embedding_and_reranking_record_usage(tmp_path):
    service = build_service(tmp_path)
    gateway = ModelGateway(service)
    context = ModelCallContext(purpose="unit-test", request_id="req-1")

    generation = gateway.generate_sync(
        [{"role": "user", "content": "What is UAT?\nContext: UAT validates readiness before go-live."}],
        "model-local-extractive",
        context=context,
    )
    embedding = gateway.embed_sync(["What is UAT?", "UAT validates go-live readiness."], "model-local-hash-384", context=context)
    reranked = gateway.rerank_sync(
        "UAT readiness",
        ["Payroll setup", "UAT validates readiness before go-live"],
        "model-local-lexical-reranker",
        top_n=2,
        context=context,
    )

    assert "UAT" in generation.text
    assert embedding.dimension == 384
    assert len(embedding.embeddings) == 2
    assert reranked.items[0].index == 1
    usage = service.list_usage(purpose="unit-test")
    assert [event.capability for event in usage] == ["rerank", "embedding", "generation"]


def test_disabled_deployment_is_not_resolved_for_runtime(tmp_path):
    service = build_service(tmp_path)
    deployment = service.create_deployment(
        {
            "name": "Disabled judge",
            "provider": "custom",
            "model": "mock-judge",
            "capabilities": ["judge"],
            "credential_env_refs": {},
            "locality": "local",
            "enabled": False,
        }
    )

    with pytest.raises(ModelFarmError, match="disabled"):
        service.resolve(deployment.id, "judge")
