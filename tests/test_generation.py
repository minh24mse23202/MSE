from aragbiz.generation import (
    GenerationRequest,
    GeneratorConfigurationError,
    GeneratorResolver,
    LocalFlanT5Generator,
    PromptBuilder,
)
from aragbiz.schemas import Document, RetrievedContext


def sample_contexts():
    return [
        RetrievedContext(
            document=Document(
                id="chunk-1",
                text="Invoice mismatches after goods receipt should be escalated to finance operations.",
                metadata={"document_id": "doc-1", "chunk_index": 0},
            ),
            score=0.87,
            rank=1,
            mode="bm25",
        )
    ]


def test_prompt_builder_includes_configuration_question_and_context():
    prompt = PromptBuilder().build(
        "How do I handle an invoice mismatch?",
        sample_contexts(),
        {
            "system_prompt": "You are a workflow assistant.",
            "predefined_prompt": "Use numbered steps.",
            "response_structure": "Step-by-step workflow guidance",
            "tone": "Technical",
            "humor_level": 1,
        },
        route_level="l2_simple_rag",
    )

    assert "You are a workflow assistant." in prompt.prompt
    assert "Use numbered steps." in prompt.prompt
    assert "Step-by-step workflow guidance" in prompt.prompt
    assert "Technical" in prompt.prompt
    assert "Humor level: 1/5" in prompt.prompt
    assert "How do I handle an invoice mismatch?" in prompt.prompt
    assert "Invoice mismatches" in prompt.prompt
    assert prompt.context_count == 1


def test_prompt_builder_labels_history_as_untrusted_and_keeps_original_question():
    prompt = PromptBuilder().build(
        "Who approves it?",
        sample_contexts(),
        {"system_prompt": "You are a workflow assistant."},
        route_level="l2_simple_rag",
        conversation_history=[
            {"role": "user", "content": "Explain the invoice mismatch workflow."},
            {"role": "assistant", "content": "Finance operations reviews the mismatch."},
        ],
        standalone_query="Who approves the invoice mismatch workflow?",
    )

    assert "untrusted conversational context" in prompt.prompt
    assert "User: Explain the invoice mismatch workflow." in prompt.prompt
    assert "Assistant: Finance operations reviews the mismatch." in prompt.prompt
    assert "Who approves the invoice mismatch workflow?" in prompt.prompt
    assert "User question:\nWho approves it?" in prompt.prompt
    assert prompt.metadata["history_exchange_count"] == 1


def test_generator_resolver_rejects_unsupported_provider():
    resolver = GeneratorResolver()

    try:
        resolver.resolve({"generator_provider": "OpenAI", "generator_model": "gpt-4.1-mini"})
    except GeneratorConfigurationError as exc:
        assert "not implemented" in str(exc)
    else:
        raise AssertionError("Expected unsupported provider to raise")


def test_flan_t5_generator_uses_cached_pipeline(monkeypatch):
    def fake_pipeline(prompt, **kwargs):
        assert "User question" in prompt
        return [{"generated_text": "Mock FLAN answer"}]

    monkeypatch.setattr(LocalFlanT5Generator, "_pipelines", {"google/flan-t5-small": fake_pipeline})
    generator = GeneratorResolver().resolve({"generator_provider": "Local", "generator_model": "google/flan-t5-small"})
    prompt = PromptBuilder().build("What is UAT?", [], {}, route_level="l1_direct")

    result = generator.generate(
        GenerationRequest(
            query="What is UAT?",
            contexts=[],
            chat_configuration={"generator_provider": "Local", "generator_model": "google/flan-t5-small"},
            prompt=prompt.prompt,
            prompt_preview=prompt.prompt_preview,
            input_chars=prompt.input_chars,
            route_level="l1_direct",
        )
    )

    assert result.answer == "Mock FLAN answer"
    assert result.provider == "Local"
    assert result.model == "google/flan-t5-small"
    assert result.status == "completed"
