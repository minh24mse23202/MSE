from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol

from aragbiz.knowledge import KnowledgeProcessingError, utc_now

ChatSection = Literal["recents", "library"]
ChatRole = Literal["user", "assistant"]


@dataclass
class ChatConfigurationRecord:
    id: str
    name: str
    description: str = ""
    generator_provider: str = "Local"
    generator_model: str = "extractive"
    response_structure: str = "Concise answer with supporting details"
    tone: str = "Professional"
    humor_level: int = 0
    system_prompt: str = ""
    predefined_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ChatConversationRecord:
    id: str
    title: str
    pinned: bool = False
    knowledge_base_id: Optional[str] = None
    chat_configuration_id: Optional[str] = None
    route_mode: str = "adaptive"
    retrieval_mode: str = "hybrid"
    top_k: int = 4
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""


@dataclass
class ChatMessageRecord:
    id: str
    conversation_id: str
    role: ChatRole
    content: str
    contexts: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


class ChatRepository(Protocol):
    def initialize(self) -> None:
        """Initialize chat storage."""

    def create_conversation(
        self,
        title: str,
        *,
        knowledge_base_id: Optional[str] = None,
        chat_configuration_id: Optional[str] = None,
        route_mode: str = "adaptive",
        retrieval_mode: str = "hybrid",
        top_k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConversationRecord:
        """Create a conversation."""

    def list_conversations(self, *, query: str = "", section: Optional[ChatSection] = None) -> List[ChatConversationRecord]:
        """List conversations."""

    def get_conversation(self, conversation_id: str) -> ChatConversationRecord:
        """Get a conversation."""

    def update_conversation(
        self,
        conversation_id: str,
        *,
        title: Optional[str] = None,
        pinned: Optional[bool] = None,
        knowledge_base_id: Optional[str] = None,
        chat_configuration_id: Optional[str] = None,
        route_mode: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConversationRecord:
        """Update a conversation."""

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and its messages."""

    def append_message(
        self,
        conversation_id: str,
        role: ChatRole,
        content: str,
        *,
        contexts: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessageRecord:
        """Append a message."""

    def list_messages(self, conversation_id: str) -> List[ChatMessageRecord]:
        """List messages for a conversation."""

    def create_configuration(
        self,
        name: str,
        *,
        description: str = "",
        generator_provider: str = "Local",
        generator_model: str = "extractive",
        response_structure: str = "Concise answer with supporting details",
        tone: str = "Professional",
        humor_level: int = 0,
        system_prompt: str = "",
        predefined_prompt: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConfigurationRecord:
        """Create a reusable chat configuration."""

    def list_configurations(self) -> List[ChatConfigurationRecord]:
        """List reusable chat configurations."""

    def get_configuration(self, configuration_id: str) -> ChatConfigurationRecord:
        """Get a reusable chat configuration."""

    def update_configuration(
        self,
        configuration_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        generator_provider: Optional[str] = None,
        generator_model: Optional[str] = None,
        response_structure: Optional[str] = None,
        tone: Optional[str] = None,
        humor_level: Optional[int] = None,
        system_prompt: Optional[str] = None,
        predefined_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConfigurationRecord:
        """Update a reusable chat configuration."""

    def delete_configuration(self, configuration_id: str) -> None:
        """Delete a reusable chat configuration."""


class ChatService:
    def __init__(self, repository: ChatRepository):
        self.repository = repository

    def create_conversation(
        self,
        title: str = "New chat",
        *,
        knowledge_base_id: Optional[str] = None,
        chat_configuration_id: Optional[str] = None,
        route_mode: str = "adaptive",
        retrieval_mode: str = "hybrid",
        top_k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConversationRecord:
        self.repository.initialize()
        return self.repository.create_conversation(
            _clean_title(title),
            knowledge_base_id=knowledge_base_id,
            chat_configuration_id=chat_configuration_id,
            route_mode=route_mode,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            metadata=metadata,
        )

    def list_conversations(self, query: str = "", section: Optional[ChatSection] = None) -> List[ChatConversationRecord]:
        self.repository.initialize()
        return self.repository.list_conversations(query=query, section=section)

    def get_conversation(self, conversation_id: str) -> ChatConversationRecord:
        self.repository.initialize()
        return self.repository.get_conversation(conversation_id)

    def update_conversation(
        self,
        conversation_id: str,
        *,
        title: Optional[str] = None,
        pinned: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConversationRecord:
        self.repository.initialize()
        return self.repository.update_conversation(
            conversation_id,
            title=_clean_title(title) if title is not None else None,
            pinned=pinned,
            metadata=metadata,
        )

    def create_configuration(
        self,
        name: str,
        *,
        description: str = "",
        generator_provider: str = "Local",
        generator_model: str = "extractive",
        response_structure: str = "Concise answer with supporting details",
        tone: str = "Professional",
        humor_level: int = 0,
        system_prompt: str = "",
        predefined_prompt: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConfigurationRecord:
        self.repository.initialize()
        return self.repository.create_configuration(
            _clean_title(name),
            description=description,
            generator_provider=generator_provider,
            generator_model=generator_model,
            response_structure=response_structure,
            tone=tone,
            humor_level=_clean_humor_level(humor_level),
            system_prompt=system_prompt,
            predefined_prompt=predefined_prompt,
            metadata=metadata,
        )

    def list_configurations(self) -> List[ChatConfigurationRecord]:
        self.repository.initialize()
        return self.repository.list_configurations()

    def get_configuration(self, configuration_id: str) -> ChatConfigurationRecord:
        self.repository.initialize()
        return self.repository.get_configuration(configuration_id)

    def default_configuration(self) -> ChatConfigurationRecord:
        self.repository.initialize()
        configurations = self.repository.list_configurations()
        if configurations:
            return configurations[0]
        return self.repository.create_configuration(**_default_chat_configuration_payload())

    def update_configuration(
        self,
        configuration_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        generator_provider: Optional[str] = None,
        generator_model: Optional[str] = None,
        response_structure: Optional[str] = None,
        tone: Optional[str] = None,
        humor_level: Optional[int] = None,
        system_prompt: Optional[str] = None,
        predefined_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConfigurationRecord:
        self.repository.initialize()
        return self.repository.update_configuration(
            configuration_id,
            name=_clean_title(name) if name is not None else None,
            description=description,
            generator_provider=generator_provider,
            generator_model=generator_model,
            response_structure=response_structure,
            tone=tone,
            humor_level=_clean_humor_level(humor_level) if humor_level is not None else None,
            system_prompt=system_prompt,
            predefined_prompt=predefined_prompt,
            metadata=metadata,
        )

    def delete_configuration(self, configuration_id: str) -> None:
        self.repository.initialize()
        self.repository.delete_configuration(configuration_id)

    def delete_conversation(self, conversation_id: str) -> None:
        self.repository.initialize()
        self.repository.delete_conversation(conversation_id)

    def list_messages(self, conversation_id: str) -> List[ChatMessageRecord]:
        self.repository.initialize()
        self.repository.get_conversation(conversation_id)
        return self.repository.list_messages(conversation_id)

    def ensure_conversation_for_question(
        self,
        conversation_id: Optional[str],
        question: str,
        *,
        knowledge_base_id: Optional[str],
        route_mode: str,
        retrieval_mode: str,
        top_k: int,
        chat_configuration_id: Optional[str] = None,
        chat_configuration: Optional[Dict[str, Any]] = None,
    ) -> ChatConversationRecord:
        self.repository.initialize()
        metadata: Dict[str, Any] = {}
        if chat_configuration_id:
            metadata["chat_configuration_id"] = chat_configuration_id
        if chat_configuration:
            metadata["chat_configuration"] = chat_configuration
        if conversation_id:
            conversation = self.repository.get_conversation(conversation_id)
            merged_metadata = {**conversation.metadata, **metadata}
            return self.repository.update_conversation(
                conversation.id,
                knowledge_base_id=knowledge_base_id,
                chat_configuration_id=chat_configuration_id,
                route_mode=route_mode,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                metadata=merged_metadata,
            )
        return self.repository.create_conversation(
            title_from_question(question),
            knowledge_base_id=knowledge_base_id,
            chat_configuration_id=chat_configuration_id,
            route_mode=route_mode,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            metadata=metadata,
        )

    def append_answer_exchange(
        self,
        conversation_id: str,
        *,
        question: str,
        answer: str,
        contexts: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> None:
        self.repository.initialize()
        self.repository.append_message(conversation_id, "user", question, contexts=[], metadata={})
        self.repository.append_message(
            conversation_id,
            "assistant",
            answer,
            contexts=contexts,
            metadata={**metadata, "question": question},
        )


class JsonChatRepository:
    def __init__(self, path: str):
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        changed = False
        if self.path.exists():
            state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            state = _empty_chat_state()
            changed = True
        for key, value in _empty_chat_state().items():
            if key not in state:
                state[key] = value
                changed = True
        if not state["chat_configurations"]:
            default_record = _default_chat_configuration_record()
            state["chat_configurations"][default_record.id] = _configuration_to_dict(default_record)
            changed = True
        if changed:
            self._write(state)

    def create_conversation(
        self,
        title: str,
        *,
        knowledge_base_id: Optional[str] = None,
        chat_configuration_id: Optional[str] = None,
        route_mode: str = "adaptive",
        retrieval_mode: str = "hybrid",
        top_k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConversationRecord:
        state = self._read()
        now = utc_now()
        record = ChatConversationRecord(
            id=f"chat-{uuid.uuid4().hex}",
            title=_clean_title(title),
            pinned=False,
            knowledge_base_id=knowledge_base_id,
            chat_configuration_id=chat_configuration_id,
            route_mode=route_mode,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        state["chat_conversations"][record.id] = _conversation_to_dict(record)
        self._write(state)
        return record

    def list_conversations(self, *, query: str = "", section: Optional[ChatSection] = None) -> List[ChatConversationRecord]:
        state = self._read()
        query_normalized = query.strip().lower()
        conversations = [_conversation_from_dict(payload) for payload in state["chat_conversations"].values()]
        if section == "library":
            conversations = [conversation for conversation in conversations if conversation.pinned]
        elif section == "recents":
            conversations = [conversation for conversation in conversations if not conversation.pinned]
        if query_normalized:
            conversations = [conversation for conversation in conversations if query_normalized in conversation.title.lower()]
        conversations.sort(key=lambda conversation: conversation.updated_at, reverse=True)
        return conversations

    def get_conversation(self, conversation_id: str) -> ChatConversationRecord:
        state = self._read()
        payload = state["chat_conversations"].get(conversation_id)
        if not payload:
            raise KeyError(f"Conversation not found: {conversation_id}")
        return _conversation_from_dict(payload)

    def update_conversation(
        self,
        conversation_id: str,
        *,
        title: Optional[str] = None,
        pinned: Optional[bool] = None,
        knowledge_base_id: Optional[str] = None,
        chat_configuration_id: Optional[str] = None,
        route_mode: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConversationRecord:
        state = self._read()
        payload = state["chat_conversations"].get(conversation_id)
        if not payload:
            raise KeyError(f"Conversation not found: {conversation_id}")
        if title is not None:
            payload["title"] = _clean_title(title)
        if pinned is not None:
            payload["pinned"] = pinned
        if knowledge_base_id is not None:
            payload["knowledge_base_id"] = knowledge_base_id
        if chat_configuration_id is not None:
            payload["chat_configuration_id"] = chat_configuration_id
        if route_mode is not None:
            payload["route_mode"] = route_mode
        if retrieval_mode is not None:
            payload["retrieval_mode"] = retrieval_mode
        if top_k is not None:
            payload["top_k"] = top_k
        if metadata is not None:
            payload["metadata"] = metadata
        payload["updated_at"] = utc_now()
        self._write(state)
        return _conversation_from_dict(payload)

    def delete_conversation(self, conversation_id: str) -> None:
        state = self._read()
        if conversation_id not in state["chat_conversations"]:
            raise KeyError(f"Conversation not found: {conversation_id}")
        state["chat_conversations"].pop(conversation_id, None)
        message_ids = [
            message_id
            for message_id, message in state["chat_messages"].items()
            if message["conversation_id"] == conversation_id
        ]
        for message_id in message_ids:
            state["chat_messages"].pop(message_id, None)
        self._write(state)

    def append_message(
        self,
        conversation_id: str,
        role: ChatRole,
        content: str,
        *,
        contexts: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessageRecord:
        state = self._read()
        if conversation_id not in state["chat_conversations"]:
            raise KeyError(f"Conversation not found: {conversation_id}")
        now = utc_now()
        record = ChatMessageRecord(
            id=f"msg-{uuid.uuid4().hex}",
            conversation_id=conversation_id,
            role=role,
            content=content,
            contexts=list(contexts or []),
            metadata=dict(metadata or {}),
            created_at=now,
        )
        state["chat_messages"][record.id] = _message_to_dict(record)
        state["chat_conversations"][conversation_id]["updated_at"] = now
        self._write(state)
        return record

    def list_messages(self, conversation_id: str) -> List[ChatMessageRecord]:
        state = self._read()
        if conversation_id not in state["chat_conversations"]:
            raise KeyError(f"Conversation not found: {conversation_id}")
        messages = [
            _message_from_dict(payload)
            for payload in state["chat_messages"].values()
            if payload["conversation_id"] == conversation_id
        ]
        messages.sort(key=lambda message: message.created_at)
        return messages

    def create_configuration(
        self,
        name: str,
        *,
        description: str = "",
        generator_provider: str = "Local",
        generator_model: str = "extractive",
        response_structure: str = "Concise answer with supporting details",
        tone: str = "Professional",
        humor_level: int = 0,
        system_prompt: str = "",
        predefined_prompt: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConfigurationRecord:
        state = self._read()
        now = utc_now()
        record = ChatConfigurationRecord(
            id=f"cfg-{uuid.uuid4().hex}",
            name=_clean_title(name),
            description=description,
            generator_provider=generator_provider,
            generator_model=generator_model,
            response_structure=response_structure,
            tone=tone,
            humor_level=_clean_humor_level(humor_level),
            system_prompt=system_prompt,
            predefined_prompt=predefined_prompt,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        state["chat_configurations"][record.id] = _configuration_to_dict(record)
        self._write(state)
        return record

    def list_configurations(self) -> List[ChatConfigurationRecord]:
        state = self._read()
        configurations = [_configuration_from_dict(payload) for payload in state["chat_configurations"].values()]
        configurations.sort(key=lambda configuration: configuration.updated_at, reverse=True)
        return configurations

    def get_configuration(self, configuration_id: str) -> ChatConfigurationRecord:
        state = self._read()
        payload = state["chat_configurations"].get(configuration_id)
        if not payload:
            raise KeyError(f"Chat configuration not found: {configuration_id}")
        return _configuration_from_dict(payload)

    def update_configuration(
        self,
        configuration_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        generator_provider: Optional[str] = None,
        generator_model: Optional[str] = None,
        response_structure: Optional[str] = None,
        tone: Optional[str] = None,
        humor_level: Optional[int] = None,
        system_prompt: Optional[str] = None,
        predefined_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConfigurationRecord:
        state = self._read()
        payload = state["chat_configurations"].get(configuration_id)
        if not payload:
            raise KeyError(f"Chat configuration not found: {configuration_id}")
        if name is not None:
            payload["name"] = _clean_title(name)
        if description is not None:
            payload["description"] = description
        if generator_provider is not None:
            payload["generator_provider"] = generator_provider
        if generator_model is not None:
            payload["generator_model"] = generator_model
        if response_structure is not None:
            payload["response_structure"] = response_structure
        if tone is not None:
            payload["tone"] = tone
        if humor_level is not None:
            payload["humor_level"] = _clean_humor_level(humor_level)
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt
        if predefined_prompt is not None:
            payload["predefined_prompt"] = predefined_prompt
        if metadata is not None:
            payload["metadata"] = metadata
        payload["updated_at"] = utc_now()
        self._write(state)
        return _configuration_from_dict(payload)

    def delete_configuration(self, configuration_id: str) -> None:
        state = self._read()
        if configuration_id not in state["chat_configurations"]:
            raise KeyError(f"Chat configuration not found: {configuration_id}")
        state["chat_configurations"].pop(configuration_id, None)
        for payload in state["chat_conversations"].values():
            if payload.get("chat_configuration_id") == configuration_id:
                payload["chat_configuration_id"] = None
        self._write(state)

    def _read(self) -> Dict[str, Dict[str, Any]]:
        self.initialize()
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, state: Dict[str, Dict[str, Any]]) -> None:
        self.path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")


class PostgresChatRepository:
    def __init__(self, database_url: str):
        try:
            from sqlalchemy import create_engine  # type: ignore
        except ImportError as exc:
            raise KnowledgeProcessingError("Install the api extra to use PostgreSQL chat storage.") from exc
        self.engine = create_engine(database_url, future=True)

    def initialize(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS chat_configurations (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            generator_provider TEXT NOT NULL DEFAULT 'Local',
            generator_model TEXT NOT NULL DEFAULT 'extractive',
            response_structure TEXT NOT NULL DEFAULT 'Concise answer with supporting details',
            tone TEXT NOT NULL DEFAULT 'Professional',
            humor_level INTEGER NOT NULL DEFAULT 0,
            system_prompt TEXT NOT NULL DEFAULT '',
            predefined_prompt TEXT NOT NULL DEFAULT '',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chat_conversations (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            pinned BOOLEAN NOT NULL DEFAULT FALSE,
            knowledge_base_id TEXT,
            chat_configuration_id TEXT REFERENCES chat_configurations(id) ON DELETE SET NULL,
            route_mode TEXT NOT NULL DEFAULT 'adaptive',
            retrieval_mode TEXT NOT NULL DEFAULT 'hybrid',
            top_k INTEGER NOT NULL DEFAULT 4,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        ALTER TABLE chat_conversations ADD COLUMN IF NOT EXISTS chat_configuration_id TEXT;
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES chat_conversations(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            contexts_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_chat_configurations_updated_at ON chat_configurations(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_chat_conversations_updated_at ON chat_conversations(updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation_id ON chat_messages(conversation_id, created_at);
        """
        with self.engine.begin() as connection:
            for statement in [part.strip() for part in ddl.split(";") if part.strip()]:
                connection.exec_driver_sql(statement)
            default_record = _default_chat_configuration_record()
            from sqlalchemy import text

            connection.execute(
                text(
                    """
                    INSERT INTO chat_configurations
                        (id, name, description, generator_provider, generator_model, response_structure, tone, humor_level, system_prompt, predefined_prompt, metadata_json, created_at, updated_at)
                    SELECT :id, :name, :description, :generator_provider, :generator_model, :response_structure, :tone, :humor_level, :system_prompt, :predefined_prompt, CAST(:metadata AS JSONB), :created_at, :updated_at
                    WHERE NOT EXISTS (SELECT 1 FROM chat_configurations)
                    """
                ),
                {**_configuration_to_dict(default_record), "metadata": json.dumps(default_record.metadata)},
            )

    def create_conversation(
        self,
        title: str,
        *,
        knowledge_base_id: Optional[str] = None,
        chat_configuration_id: Optional[str] = None,
        route_mode: str = "adaptive",
        retrieval_mode: str = "hybrid",
        top_k: int = 4,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConversationRecord:
        from sqlalchemy import text

        now = utc_now()
        record = ChatConversationRecord(
            id=f"chat-{uuid.uuid4().hex}",
            title=_clean_title(title),
            pinned=False,
            knowledge_base_id=knowledge_base_id,
            chat_configuration_id=chat_configuration_id,
            route_mode=route_mode,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO chat_conversations
                        (id, title, pinned, knowledge_base_id, chat_configuration_id, route_mode, retrieval_mode, top_k, metadata_json, created_at, updated_at)
                    VALUES
                        (:id, :title, :pinned, :knowledge_base_id, :chat_configuration_id, :route_mode, :retrieval_mode, :top_k, CAST(:metadata AS JSONB), :created_at, :updated_at)
                    """
                ),
                {**_conversation_to_dict(record), "metadata": json.dumps(record.metadata)},
            )
        return record

    def list_conversations(self, *, query: str = "", section: Optional[ChatSection] = None) -> List[ChatConversationRecord]:
        from sqlalchemy import text

        filters = []
        params: Dict[str, Any] = {}
        if section == "library":
            filters.append("pinned = TRUE")
        elif section == "recents":
            filters.append("pinned = FALSE")
        if query.strip():
            filters.append("LOWER(title) LIKE :query")
            params["query"] = f"%{query.strip().lower()}%"
        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        with self.engine.begin() as connection:
            rows = connection.execute(
                text(f"SELECT * FROM chat_conversations {where_clause} ORDER BY updated_at DESC"),
                params,
            ).mappings()
            return [_conversation_from_row(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> ChatConversationRecord:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            row = connection.execute(
                text("SELECT * FROM chat_conversations WHERE id = :id"),
                {"id": conversation_id},
            ).mappings().first()
        if not row:
            raise KeyError(f"Conversation not found: {conversation_id}")
        return _conversation_from_row(row)

    def update_conversation(
        self,
        conversation_id: str,
        *,
        title: Optional[str] = None,
        pinned: Optional[bool] = None,
        knowledge_base_id: Optional[str] = None,
        chat_configuration_id: Optional[str] = None,
        route_mode: Optional[str] = None,
        retrieval_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConversationRecord:
        from sqlalchemy import text

        current = self.get_conversation(conversation_id)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE chat_conversations
                    SET title = :title,
                        pinned = :pinned,
                        knowledge_base_id = :knowledge_base_id,
                        chat_configuration_id = :chat_configuration_id,
                        route_mode = :route_mode,
                        retrieval_mode = :retrieval_mode,
                        top_k = :top_k,
                        metadata_json = CAST(:metadata AS JSONB),
                        updated_at = :updated_at
                    WHERE id = :id
                    """
                ),
                {
                    "id": conversation_id,
                    "title": title if title is not None else current.title,
                    "pinned": pinned if pinned is not None else current.pinned,
                    "knowledge_base_id": knowledge_base_id if knowledge_base_id is not None else current.knowledge_base_id,
                    "chat_configuration_id": chat_configuration_id if chat_configuration_id is not None else current.chat_configuration_id,
                    "route_mode": route_mode if route_mode is not None else current.route_mode,
                    "retrieval_mode": retrieval_mode if retrieval_mode is not None else current.retrieval_mode,
                    "top_k": top_k if top_k is not None else current.top_k,
                    "metadata": json.dumps(metadata if metadata is not None else current.metadata),
                    "updated_at": utc_now(),
                },
            )
        return self.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            result = connection.execute(text("DELETE FROM chat_conversations WHERE id = :id"), {"id": conversation_id})
            if result.rowcount == 0:
                raise KeyError(f"Conversation not found: {conversation_id}")

    def append_message(
        self,
        conversation_id: str,
        role: ChatRole,
        content: str,
        *,
        contexts: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessageRecord:
        from sqlalchemy import text

        self.get_conversation(conversation_id)
        now = utc_now()
        record = ChatMessageRecord(
            id=f"msg-{uuid.uuid4().hex}",
            conversation_id=conversation_id,
            role=role,
            content=content,
            contexts=list(contexts or []),
            metadata=dict(metadata or {}),
            created_at=now,
        )
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO chat_messages
                        (id, conversation_id, role, content, contexts_json, metadata_json, created_at)
                    VALUES
                        (:id, :conversation_id, :role, :content, CAST(:contexts AS JSONB), CAST(:metadata AS JSONB), :created_at)
                    """
                ),
                {**_message_to_dict(record), "contexts": json.dumps(record.contexts), "metadata": json.dumps(record.metadata)},
            )
            connection.execute(
                text("UPDATE chat_conversations SET updated_at = :updated_at WHERE id = :id"),
                {"id": conversation_id, "updated_at": now},
            )
        return record

    def list_messages(self, conversation_id: str) -> List[ChatMessageRecord]:
        from sqlalchemy import text

        self.get_conversation(conversation_id)
        with self.engine.begin() as connection:
            rows = connection.execute(
                text("SELECT * FROM chat_messages WHERE conversation_id = :id ORDER BY created_at"),
                {"id": conversation_id},
            ).mappings()
            return [_message_from_row(row) for row in rows]

    def create_configuration(
        self,
        name: str,
        *,
        description: str = "",
        generator_provider: str = "Local",
        generator_model: str = "extractive",
        response_structure: str = "Concise answer with supporting details",
        tone: str = "Professional",
        humor_level: int = 0,
        system_prompt: str = "",
        predefined_prompt: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConfigurationRecord:
        from sqlalchemy import text

        now = utc_now()
        record = ChatConfigurationRecord(
            id=f"cfg-{uuid.uuid4().hex}",
            name=_clean_title(name),
            description=description,
            generator_provider=generator_provider,
            generator_model=generator_model,
            response_structure=response_structure,
            tone=tone,
            humor_level=_clean_humor_level(humor_level),
            system_prompt=system_prompt,
            predefined_prompt=predefined_prompt,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO chat_configurations
                        (id, name, description, generator_provider, generator_model, response_structure, tone, humor_level, system_prompt, predefined_prompt, metadata_json, created_at, updated_at)
                    VALUES
                        (:id, :name, :description, :generator_provider, :generator_model, :response_structure, :tone, :humor_level, :system_prompt, :predefined_prompt, CAST(:metadata AS JSONB), :created_at, :updated_at)
                    """
                ),
                {**_configuration_to_dict(record), "metadata": json.dumps(record.metadata)},
            )
        return record

    def list_configurations(self) -> List[ChatConfigurationRecord]:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            rows = connection.execute(text("SELECT * FROM chat_configurations ORDER BY updated_at DESC")).mappings()
            return [_configuration_from_row(row) for row in rows]

    def get_configuration(self, configuration_id: str) -> ChatConfigurationRecord:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            row = connection.execute(
                text("SELECT * FROM chat_configurations WHERE id = :id"),
                {"id": configuration_id},
            ).mappings().first()
        if not row:
            raise KeyError(f"Chat configuration not found: {configuration_id}")
        return _configuration_from_row(row)

    def update_configuration(
        self,
        configuration_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        generator_provider: Optional[str] = None,
        generator_model: Optional[str] = None,
        response_structure: Optional[str] = None,
        tone: Optional[str] = None,
        humor_level: Optional[int] = None,
        system_prompt: Optional[str] = None,
        predefined_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatConfigurationRecord:
        from sqlalchemy import text

        current = self.get_configuration(configuration_id)
        with self.engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE chat_configurations
                    SET name = :name,
                        description = :description,
                        generator_provider = :generator_provider,
                        generator_model = :generator_model,
                        response_structure = :response_structure,
                        tone = :tone,
                        humor_level = :humor_level,
                        system_prompt = :system_prompt,
                        predefined_prompt = :predefined_prompt,
                        metadata_json = CAST(:metadata AS JSONB),
                        updated_at = :updated_at
                    WHERE id = :id
                    """
                ),
                {
                    "id": configuration_id,
                    "name": name if name is not None else current.name,
                    "description": description if description is not None else current.description,
                    "generator_provider": generator_provider if generator_provider is not None else current.generator_provider,
                    "generator_model": generator_model if generator_model is not None else current.generator_model,
                    "response_structure": response_structure if response_structure is not None else current.response_structure,
                    "tone": tone if tone is not None else current.tone,
                    "humor_level": _clean_humor_level(humor_level) if humor_level is not None else current.humor_level,
                    "system_prompt": system_prompt if system_prompt is not None else current.system_prompt,
                    "predefined_prompt": predefined_prompt if predefined_prompt is not None else current.predefined_prompt,
                    "metadata": json.dumps(metadata if metadata is not None else current.metadata),
                    "updated_at": utc_now(),
                },
            )
        return self.get_configuration(configuration_id)

    def delete_configuration(self, configuration_id: str) -> None:
        from sqlalchemy import text

        with self.engine.begin() as connection:
            connection.execute(
                text("UPDATE chat_conversations SET chat_configuration_id = NULL WHERE chat_configuration_id = :id"),
                {"id": configuration_id},
            )
            result = connection.execute(text("DELETE FROM chat_configurations WHERE id = :id"), {"id": configuration_id})
            if result.rowcount == 0:
                raise KeyError(f"Chat configuration not found: {configuration_id}")


def title_from_question(question: str, max_length: int = 64) -> str:
    title = " ".join(question.strip().split())
    if not title:
        return "New chat"
    if len(title) <= max_length:
        return title
    return title[: max_length - 1].rstrip() + "窶ｦ"


def _clean_title(value: Optional[str]) -> str:
    title = " ".join((value or "New chat").strip().split())
    return title or "New chat"


def _clean_humor_level(value: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(parsed, 5))


def _empty_chat_state() -> Dict[str, Dict[str, Any]]:
    return {"chat_conversations": {}, "chat_messages": {}, "chat_configurations": {}}


def _conversation_to_dict(record: ChatConversationRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "title": record.title,
        "pinned": record.pinned,
        "knowledge_base_id": record.knowledge_base_id,
        "chat_configuration_id": record.chat_configuration_id,
        "route_mode": record.route_mode,
        "retrieval_mode": record.retrieval_mode,
        "top_k": record.top_k,
        "metadata": record.metadata,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _default_chat_configuration_payload() -> Dict[str, Any]:
    return {
        "name": "Balanced workflow assistant",
        "description": "Default configuration for concise business workflow answers.",
        "generator_provider": "Local",
        "generator_model": "extractive",
        "response_structure": "Concise answer with bullets and cited workflow context",
        "tone": "Professional",
        "humor_level": 0,
        "system_prompt": "You are an Adaptive RAG assistant for business workflow question answering. Answer using retrieved workflow context when available.",
        "predefined_prompt": "Answer clearly, mention uncertainty, and cite relevant workflow evidence when retrieval is used.",
        "metadata": {"runtime": "local-generator", "actual_generator": "extractive"},
    }


def _default_chat_configuration_record() -> ChatConfigurationRecord:
    now = utc_now()
    return ChatConfigurationRecord(
        id="cfg-default-balanced-workflow",
        created_at=now,
        updated_at=now,
        **_default_chat_configuration_payload(),
    )


def _configuration_to_dict(record: ChatConfigurationRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "name": record.name,
        "description": record.description,
        "generator_provider": record.generator_provider,
        "generator_model": record.generator_model,
        "response_structure": record.response_structure,
        "tone": record.tone,
        "humor_level": record.humor_level,
        "system_prompt": record.system_prompt,
        "predefined_prompt": record.predefined_prompt,
        "metadata": record.metadata,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def _configuration_from_dict(payload: Dict[str, Any]) -> ChatConfigurationRecord:
    defaults = _default_chat_configuration_payload()
    return ChatConfigurationRecord(
        id=payload["id"],
        name=payload.get("name") or defaults["name"],
        description=payload.get("description", defaults["description"]),
        generator_provider=payload.get("generator_provider") or defaults["generator_provider"],
        generator_model=payload.get("generator_model") or defaults["generator_model"],
        response_structure=payload.get("response_structure") or defaults["response_structure"],
        tone=payload.get("tone") or defaults["tone"],
        humor_level=_clean_humor_level(payload.get("humor_level", defaults["humor_level"])),
        system_prompt=payload.get("system_prompt", defaults["system_prompt"]),
        predefined_prompt=payload.get("predefined_prompt", defaults["predefined_prompt"]),
        metadata=dict(payload.get("metadata") or {}),
        created_at=payload.get("created_at", ""),
        updated_at=payload.get("updated_at", ""),
    )


def _message_to_dict(record: ChatMessageRecord) -> Dict[str, Any]:
    return {
        "id": record.id,
        "conversation_id": record.conversation_id,
        "role": record.role,
        "content": record.content,
        "contexts": record.contexts,
        "metadata": record.metadata,
        "created_at": record.created_at,
    }


def _conversation_from_dict(payload: Dict[str, Any]) -> ChatConversationRecord:
    return ChatConversationRecord(
        id=payload["id"],
        title=payload.get("title") or "New chat",
        pinned=bool(payload.get("pinned", False)),
        knowledge_base_id=payload.get("knowledge_base_id"),
        chat_configuration_id=payload.get("chat_configuration_id"),
        route_mode=payload.get("route_mode", "adaptive"),
        retrieval_mode=payload.get("retrieval_mode", "hybrid"),
        top_k=int(payload.get("top_k") or 4),
        metadata=dict(payload.get("metadata") or {}),
        created_at=payload.get("created_at", ""),
        updated_at=payload.get("updated_at", ""),
    )


def _message_from_dict(payload: Dict[str, Any]) -> ChatMessageRecord:
    return ChatMessageRecord(
        id=payload["id"],
        conversation_id=payload["conversation_id"],
        role=payload["role"],
        content=payload["content"],
        contexts=list(payload.get("contexts") or []),
        metadata=dict(payload.get("metadata") or {}),
        created_at=payload.get("created_at", ""),
    )


def _conversation_from_row(row: Any) -> ChatConversationRecord:
    return ChatConversationRecord(
        id=row["id"],
        title=row["title"],
        pinned=bool(row.get("pinned")),
        knowledge_base_id=row.get("knowledge_base_id"),
        chat_configuration_id=row.get("chat_configuration_id"),
        route_mode=row.get("route_mode") or "adaptive",
        retrieval_mode=row.get("retrieval_mode") or "hybrid",
        top_k=int(row.get("top_k") or 4),
        metadata=dict(row.get("metadata_json") or {}),
        created_at=row.get("created_at") or "",
        updated_at=row.get("updated_at") or "",
    )


def _message_from_row(row: Any) -> ChatMessageRecord:
    return ChatMessageRecord(
        id=row["id"],
        conversation_id=row["conversation_id"],
        role=row["role"],
        content=row["content"],
        contexts=list(row.get("contexts_json") or []),
        metadata=dict(row.get("metadata_json") or {}),
        created_at=row.get("created_at") or "",
    )


def _configuration_from_row(row: Any) -> ChatConfigurationRecord:
    defaults = _default_chat_configuration_payload()
    return ChatConfigurationRecord(
        id=row["id"],
        name=row.get("name") or defaults["name"],
        description=row.get("description") or defaults["description"],
        generator_provider=row.get("generator_provider") or defaults["generator_provider"],
        generator_model=row.get("generator_model") or defaults["generator_model"],
        response_structure=row.get("response_structure") or defaults["response_structure"],
        tone=row.get("tone") or defaults["tone"],
        humor_level=_clean_humor_level(row.get("humor_level", defaults["humor_level"])),
        system_prompt=row.get("system_prompt") or defaults["system_prompt"],
        predefined_prompt=row.get("predefined_prompt") or defaults["predefined_prompt"],
        metadata=dict(row.get("metadata_json") or {}),
        created_at=row.get("created_at") or "",
        updated_at=row.get("updated_at") or "",
    )
