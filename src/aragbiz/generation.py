from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from aragbiz.schemas import RetrievedContext
from aragbiz.model_farm import ModelCallContext, ModelFarmError, ModelFarmService, ModelGateway


class GeneratorConfigurationError(ValueError):
    """Raised when a selected generator provider/model cannot be used."""


class GeneratorExecutionError(ValueError):
    """Raised when a configured generator fails at runtime."""


@dataclass(frozen=True)
class PromptBuildResult:
    prompt: str
    prompt_preview: str
    input_chars: int
    context_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationRequest:
    query: str
    contexts: List[RetrievedContext]
    chat_configuration: Dict[str, Any]
    prompt: str
    prompt_preview: str
    input_chars: int
    route_level: str


@dataclass(frozen=True)
class GeneratorResult:
    answer: str
    provider: str
    model: str
    status: str
    prompt_preview: str
    input_chars: int
    output_chars: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class Generator(Protocol):
    def generate(self, request: GenerationRequest) -> GeneratorResult:
        """Generate an answer from a prepared prompt and optional contexts."""


class PromptBuilder:
    def __init__(self, max_context_chars: int = 3200, prompt_preview_chars: int = 1200):
        self.max_context_chars = max_context_chars
        self.prompt_preview_chars = prompt_preview_chars

    def build(
        self,
        query: str,
        contexts: List[RetrievedContext],
        chat_configuration: Optional[Dict[str, Any]] = None,
        *,
        route_level: str = "l2_simple_rag",
    ) -> PromptBuildResult:
        configuration = _normalized_configuration(chat_configuration)
        context_block = self._context_block(contexts)
        prompt_parts = [
            configuration["system_prompt"],
            "",
            "Runtime instructions:",
            f"- Response structure: {configuration['response_structure']}",
            f"- Tone: {configuration['tone']}",
            f"- Humor level: {configuration['humor_level']}/5. Keep humor appropriate for business workflow support.",
            f"- Route level: {route_level}",
            configuration["predefined_prompt"],
            "",
            "Retrieved workflow context:",
            context_block or "No retrieved workflow context was provided.",
            "",
            "User question:",
            query.strip(),
            "",
            "Answer:",
        ]
        prompt = "\n".join(part for part in prompt_parts if part is not None).strip()
        return PromptBuildResult(
            prompt=prompt,
            prompt_preview=prompt[: self.prompt_preview_chars],
            input_chars=len(prompt),
            context_count=len(contexts),
            metadata={
                "response_structure": configuration["response_structure"],
                "tone": configuration["tone"],
                "humor_level": configuration["humor_level"],
                "context_count": len(contexts),
            },
        )

    def _context_block(self, contexts: List[RetrievedContext]) -> str:
        if not contexts:
            return ""
        parts: List[str] = []
        remaining = self.max_context_chars
        for context in contexts:
            metadata = context.document.metadata
            title = metadata.get("title") or metadata.get("document_title") or metadata.get("source") or metadata.get("document_id") or context.document.id
            chunk_index = metadata.get("chunk_index", "-")
            source_label = metadata.get("source_label") or f"S{context.rank}"
            header = f"[{source_label}] {title} | score={context.score:.4f} | chunk={chunk_index}"
            text = " ".join(context.document.text.split())
            segment = f"{header}\n{text}"
            if len(segment) > remaining:
                segment = segment[: max(remaining, 0)].rstrip()
            if segment:
                parts.append(segment)
                remaining -= len(segment)
            if remaining <= 0:
                break
        return "\n\n".join(parts)


class ExtractiveGenerator:
    def __init__(self, max_context_chars: int = 900):
        self.max_context_chars = max_context_chars

    def generate(self, request: GenerationRequest) -> GeneratorResult:
        if request.contexts:
            best = request.contexts[0].document
            answer = best.metadata.get("answer")
            if not isinstance(answer, str) or not answer:
                combined = " ".join(context.document.text for context in request.contexts)
                clipped = combined[: self.max_context_chars].strip()
                answer = f"Based on the retrieved workflow context: {clipped}"
        else:
            answer = (
                "Direct generation is running with the Local/extractive fallback. "
                "No knowledge base context was retrieved, so use this as ungrounded draft guidance and switch to L2 Simple RAG for cited workflow evidence. "
                f"Question: {request.query.strip()}"
            )
        return GeneratorResult(
            answer=answer,
            provider="Local",
            model="extractive",
            status="completed",
            prompt_preview=request.prompt_preview,
            input_chars=request.input_chars,
            output_chars=len(answer),
            metadata={"context_count": len(request.contexts), "runtime": "deterministic-extractive"},
        )


class LocalFlanT5Generator:
    _pipelines: Dict[str, Any] = {}

    def __init__(self, model_name: str = "google/flan-t5-small", max_new_tokens: int = 160):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def generate(self, request: GenerationRequest) -> GeneratorResult:
        try:
            generator = self._pipeline()
            outputs = generator(request.prompt, max_new_tokens=self.max_new_tokens, truncation=True)
        except Exception as exc:  # pragma: no cover - exact optional dependency failures vary by platform
            raise GeneratorExecutionError(
                f"Unable to run local generator '{self.model_name}': {exc}. "
                "Use Local/extractive or repair ML dependencies with python -m pip install -e \".[ml]\"."
            ) from exc
        answer = _generated_text(outputs).strip() or "The local generator returned an empty answer."
        return GeneratorResult(
            answer=answer,
            provider="Local",
            model=self.model_name,
            status="completed",
            prompt_preview=request.prompt_preview,
            input_chars=request.input_chars,
            output_chars=len(answer),
            metadata={"context_count": len(request.contexts), "runtime": "transformers-text2text"},
        )

    def _pipeline(self) -> Any:
        if self.model_name not in self._pipelines:
            try:
                from transformers import pipeline  # type: ignore
            except ImportError as exc:
                raise GeneratorExecutionError(
                    "Install the ml extra to use Local/google/flan-t5-small: python -m pip install -e \".[ml]\"."
                ) from exc
            self._pipelines[self.model_name] = pipeline("text2text-generation", model=self.model_name, tokenizer=self.model_name)
        return self._pipelines[self.model_name]


class ModelFarmGenerator:
    def __init__(
        self,
        gateway: ModelGateway,
        deployment_id: str,
        *,
        fallback_deployment_ids: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        external_processing_allowed: bool = True,
        call_context: Optional[ModelCallContext] = None,
    ):
        self.gateway = gateway
        self.deployment_id = deployment_id
        self.fallback_deployment_ids = fallback_deployment_ids or []
        self.parameters = parameters or {}
        self.external_processing_allowed = external_processing_allowed
        self.call_context = call_context

    def generate(self, request: GenerationRequest) -> GeneratorResult:
        try:
            result = self.gateway.generate_sync(
                [{"role": "user", "content": request.prompt}],
                self.deployment_id,
                fallback_deployment_ids=self.fallback_deployment_ids,
                parameters=self.parameters,
                context=self.call_context,
                external_processing_allowed=self.external_processing_allowed,
            )
        except ModelFarmError as exc:
            raise GeneratorExecutionError(str(exc)) from exc
        return GeneratorResult(
            answer=result.text,
            provider=result.provider,
            model=result.model,
            status=result.status,
            prompt_preview=request.prompt_preview,
            input_chars=request.input_chars,
            output_chars=len(result.text),
            metadata={
                **result.metadata,
                "deployment_id": result.deployment_id,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "estimated_cost_usd": result.estimated_cost_usd,
                "finish_reason": result.finish_reason,
            },
        )


class GeneratorResolver:
    def __init__(
        self,
        extractive_generator: Optional[ExtractiveGenerator] = None,
        *,
        model_farm_service: Optional[ModelFarmService] = None,
        model_gateway: Optional[ModelGateway] = None,
    ):
        self.extractive_generator = extractive_generator or ExtractiveGenerator()
        self.model_farm_service = model_farm_service
        self.model_gateway = model_gateway

    def resolve(
        self,
        chat_configuration: Optional[Dict[str, Any]],
        *,
        external_processing_allowed: bool = True,
        call_context: Optional[ModelCallContext] = None,
    ) -> Generator:
        configuration = _normalized_configuration(chat_configuration)
        deployment_id = str(configuration.get("generator_deployment_id") or "").strip()
        if deployment_id:
            if self.model_gateway is not None and self.model_farm_service is not None:
                self.model_farm_service.resolve(deployment_id, "generation")
                return ModelFarmGenerator(
                    self.model_gateway,
                    deployment_id,
                    fallback_deployment_ids=list(configuration.get("fallback_deployment_ids") or []),
                    parameters=dict(configuration.get("generation_parameters") or {}),
                    external_processing_allowed=external_processing_allowed,
                    call_context=call_context,
                )
            if deployment_id not in {"model-local-extractive", "model-local-flan-t5-small"}:
                raise GeneratorConfigurationError("The Model Farm gateway is not configured for this runtime.")
        provider = configuration["generator_provider"]
        model = configuration["generator_model"]
        if provider != "Local":
            raise GeneratorConfigurationError(
                f"Generator provider '{provider}' is not implemented yet. Use Local/extractive or Local/google/flan-t5-small."
            )
        if model == "extractive":
            return self.extractive_generator
        if model == "google/flan-t5-small":
            return LocalFlanT5Generator(model)
        raise GeneratorConfigurationError(
            f"Local generator model '{model}' is not supported yet. Use extractive or google/flan-t5-small."
        )


def default_chat_configuration() -> Dict[str, Any]:
    return {
        "name": "Balanced workflow assistant",
        "generator_provider": "Local",
        "generator_model": "extractive",
        "generator_deployment_id": "model-local-extractive",
        "fallback_deployment_ids": [],
        "reranker_deployment_id": "",
        "planner_deployment_id": "",
        "generation_parameters": {"temperature": 0.2, "max_tokens": 500},
        "response_structure": "Concise answer with bullets and cited workflow context",
        "tone": "Professional",
        "humor_level": 0,
        "system_prompt": "You are an Adaptive RAG assistant for business workflow question answering. Answer using retrieved workflow context when available.",
        "predefined_prompt": "Answer clearly, mention uncertainty, and cite relevant workflow evidence when retrieval is used.",
        "metadata": {},
    }


def _normalized_configuration(chat_configuration: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = default_chat_configuration()
    normalized.update(chat_configuration or {})
    normalized["generator_provider"] = str(normalized.get("generator_provider") or "Local")
    normalized["generator_model"] = str(normalized.get("generator_model") or "extractive")
    normalized["response_structure"] = str(normalized.get("response_structure") or default_chat_configuration()["response_structure"])
    normalized["tone"] = str(normalized.get("tone") or "Professional")
    try:
        humor_level = int(normalized.get("humor_level", 0))
    except (TypeError, ValueError):
        humor_level = 0
    normalized["humor_level"] = max(0, min(humor_level, 5))
    normalized["system_prompt"] = str(normalized.get("system_prompt") or default_chat_configuration()["system_prompt"])
    normalized["predefined_prompt"] = str(normalized.get("predefined_prompt") or default_chat_configuration()["predefined_prompt"])
    return normalized


def _generated_text(outputs: Any) -> str:
    if isinstance(outputs, list) and outputs:
        first = outputs[0]
        if isinstance(first, dict):
            return str(first.get("generated_text") or first.get("summary_text") or "")
        return str(first)
    if isinstance(outputs, dict):
        return str(outputs.get("generated_text") or outputs.get("summary_text") or "")
    return str(outputs or "")
