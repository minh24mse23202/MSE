from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List


class AgentActionError(ValueError):
    """Raised when a planner returns an unsafe or malformed action."""


@dataclass(frozen=True)
class AgentToolDescriptor:
    name: str
    description: str
    capability: str
    locality: str
    available: bool
    unavailable_reason: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentAction:
    action: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


class AgentToolRegistry:
    """Provider-neutral catalog exposed to L4 planners and the Main UI."""

    def __init__(self) -> None:
        self._tools = [
            AgentToolDescriptor(
                "search_knowledge_base",
                "Search the selected knowledge base using the configured retrieval mode and document filter.",
                "retrieval",
                "local",
                True,
                parameters={"query": "string", "top_k": "integer"},
            ),
            AgentToolDescriptor(
                "inspect_source",
                "Inspect a retrieved chunk by its chunk identifier.",
                "inspection",
                "local",
                True,
                parameters={"chunk_id": "string"},
            ),
            AgentToolDescriptor(
                "rerank_evidence",
                "Rerank the evidence collected by earlier tool calls.",
                "rerank",
                "mixed",
                True,
                parameters={"query": "string"},
            ),
            AgentToolDescriptor(
                "fetch_public_url",
                "Fetch readable content from an explicit public HTTP or HTTPS URL.",
                "web",
                "remote",
                True,
                parameters={"url": "string"},
            ),
            AgentToolDescriptor(
                "finish",
                "Stop gathering evidence and proceed to final answer generation.",
                "control",
                "local",
                True,
                parameters={},
            ),
            AgentToolDescriptor(
                "search_google_drive",
                "Search an authenticated Google Drive connector.",
                "connector",
                "remote",
                False,
                "Google Drive query connector is not implemented.",
            ),
            AgentToolDescriptor(
                "search_onedrive",
                "Search an authenticated OneDrive connector.",
                "connector",
                "remote",
                False,
                "OneDrive query connector is not implemented.",
            ),
            AgentToolDescriptor(
                "query_database",
                "Run an approved read-only query through a database connector.",
                "connector",
                "mixed",
                False,
                "Read-only database query connectors are not implemented.",
            ),
        ]

    def list_tools(self, *, public_web_enabled: bool = False) -> List[AgentToolDescriptor]:
        tools: List[AgentToolDescriptor] = []
        for tool in self._tools:
            if tool.name == "fetch_public_url" and not public_web_enabled:
                tools.append(
                    AgentToolDescriptor(
                        **{
                            **tool.to_dict(),
                            "available": False,
                            "unavailable_reason": "Public web tools are disabled in this RAG configuration.",
                        }
                    )
                )
            else:
                tools.append(tool)
        return tools

    def planner_tools(self, *, public_web_enabled: bool = False) -> List[Dict[str, Any]]:
        return [tool.to_dict() for tool in self.list_tools(public_web_enabled=public_web_enabled) if tool.available]

    def available_names(self, *, public_web_enabled: bool = False) -> set[str]:
        return {tool["name"] for tool in self.planner_tools(public_web_enabled=public_web_enabled)}


def parse_agent_action(value: str, allowed_actions: Iterable[str]) -> AgentAction:
    text = str(value or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        payload = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AgentActionError("Planner action must be one JSON object.") from exc
    if not isinstance(payload, dict):
        raise AgentActionError("Planner action must be a JSON object.")
    action = str(payload.get("action") or "").strip()
    allowed = set(allowed_actions)
    if action not in allowed:
        raise AgentActionError(f"Planner selected unavailable action: {action or '<empty>'}.")
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise AgentActionError("Planner action arguments must be a JSON object.")
    return AgentAction(action, arguments, str(payload.get("reason") or "").strip()[:500])

