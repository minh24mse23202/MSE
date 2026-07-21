import pytest

from aragbiz.chat import ChatService, JsonChatRepository


def build_chat_service(tmp_path):
    return ChatService(JsonChatRepository(str(tmp_path / "chat.json")))


def test_assistant_message_creates_initial_version(tmp_path):
    service = build_chat_service(tmp_path)
    conversation = service.create_conversation("Version test")
    assistant = service.append_message(
        conversation.id,
        "assistant",
        "Initial answer",
        contexts=[{"id": "chunk-1"}],
        metadata={"question": "What is UAT?"},
        status="completed",
        request_id="request-1",
    )

    versions = service.list_message_versions(assistant.id)

    assert len(versions) == 1
    assert versions[0].version_number == 1
    assert versions[0].content == "Initial answer"
    assert versions[0].request_id == "request-1"
    assert service.message_version_summary(assistant.id) == (1, 1)


def test_failed_version_does_not_replace_canonical_answer(tmp_path):
    service = build_chat_service(tmp_path)
    conversation = service.create_conversation("Failed retry")
    assistant = service.append_message(
        conversation.id,
        "assistant",
        "Working answer",
        status="completed",
    )
    failed = service.create_message_version(
        assistant.id,
        content="Partial retry",
        status="pending",
        request_id="request-2",
    )
    service.update_message_version(failed.id, status="failed")

    canonical = service.get_message(assistant.id)
    versions = service.list_message_versions(assistant.id)

    assert canonical.content == "Working answer"
    assert canonical.status == "completed"
    assert [version.status for version in versions] == ["completed", "failed"]


def test_completed_version_can_be_activated(tmp_path):
    service = build_chat_service(tmp_path)
    conversation = service.create_conversation("Completed retry")
    assistant = service.append_message(
        conversation.id,
        "assistant",
        "Version one",
        status="completed",
    )
    second = service.create_message_version(
        assistant.id,
        content="Version two",
        contexts=[{"id": "chunk-2"}],
        metadata={"message_version_number": 2},
        status="completed",
        request_id="request-2",
    )

    activated = service.activate_message_version(second.id)

    assert activated.content == "Version two"
    assert activated.contexts == [{"id": "chunk-2"}]
    assert activated.request_id == "request-2"
    assert service.message_version_summary(assistant.id) == (2, 2)


def test_active_version_blocks_concurrent_regeneration(tmp_path):
    service = build_chat_service(tmp_path)
    conversation = service.create_conversation("Concurrent retry")
    assistant = service.append_message(
        conversation.id,
        "assistant",
        "Version one",
        status="completed",
    )
    service.create_message_version(assistant.id, status="streaming", request_id="request-2")

    with pytest.raises(ValueError, match="already being generated"):
        service.create_message_version(assistant.id, status="pending", request_id="request-3")
