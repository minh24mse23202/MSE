from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from aragbiz.model_farm import ModelCallContext, ModelFarmError, ModelGateway

DEFAULT_HISTORY_MAX_EXCHANGES = 3
DEFAULT_HISTORY_MAX_CHARACTERS = 4000

_FOLLOW_UP_PREFIXES = (
    "and ",
    "also ",
    "then ",
    "what about",
    "how about",
    "what if",
    "in that case",
    "based on that",
)
_REFERENTIAL_PATTERN = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|the former|the latter|"
    r"same|next step|previous step|above|earlier)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QueryReformulationResult:
    original_query: str
    standalone_query: str
    rewritten: bool
    follow_up_detected: bool
    strategy: str
    warning: str = ""
    planner_metadata: Dict[str, Any] = field(default_factory=dict)


def build_conversation_history(
    records: Sequence[Any],
    *,
    max_exchanges: int = DEFAULT_HISTORY_MAX_EXCHANGES,
    max_characters: int = DEFAULT_HISTORY_MAX_CHARACTERS,
) -> List[Dict[str, str]]:
    """Return recent completed user-assistant exchanges within a character budget."""
    exchanges: List[tuple[Any, Any]] = []
    pending_user: Optional[Any] = None
    for record in records:
        role = str(getattr(record, "role", "") or "")
        status = str(getattr(record, "status", "") or "")
        if role == "user":
            pending_user = record if status == "completed" else None
            continue
        if role != "assistant":
            continue
        if (
            status == "completed"
            and pending_user is not None
            and _compact_text(getattr(pending_user, "content", ""))
            and _compact_text(getattr(record, "content", ""))
        ):
            exchanges.append((pending_user, record))
        pending_user = None

    exchange_limit = max(0, int(max_exchanges))
    selected = exchanges[-exchange_limit:] if exchange_limit else []
    while len(selected) > 1 and _exchange_characters(selected) > max_characters:
        selected.pop(0)

    messages: List[Dict[str, str]] = []
    for user_record, assistant_record in selected:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": _compact_text(getattr(user_record, "content", "")),
                    "message_id": str(getattr(user_record, "id", "") or ""),
                },
                {
                    "role": "assistant",
                    "content": _compact_text(getattr(assistant_record, "content", "")),
                    "message_id": str(getattr(assistant_record, "id", "") or ""),
                },
            ]
        )
    return _trim_history_messages(messages, max_characters=max_characters)


def normalize_conversation_history(
    messages: Sequence[Dict[str, Any]],
    *,
    max_exchanges: int = DEFAULT_HISTORY_MAX_EXCHANGES,
    max_characters: int = DEFAULT_HISTORY_MAX_CHARACTERS,
) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "").strip().lower()
        content = _compact_text(message.get("content"))
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append(
            {
                "role": role,
                "content": content,
                "message_id": str(message.get("message_id") or ""),
            }
        )
    exchange_limit = max(0, int(max_exchanges))
    if exchange_limit == 0:
        return []
    normalized = normalized[-(exchange_limit * 2) :]
    return _trim_history_messages(normalized, max_characters=max_characters)


def conversation_history_characters(messages: Sequence[Dict[str, Any]]) -> int:
    return sum(len(str(message.get("content") or "")) for message in messages)


def conversation_history_exchange_count(messages: Sequence[Dict[str, Any]]) -> int:
    return min(
        sum(1 for message in messages if message.get("role") == "user"),
        sum(1 for message in messages if message.get("role") == "assistant"),
    )


class QueryReformulator:
    def reformulate(
        self,
        query: str,
        history: Sequence[Dict[str, Any]],
        *,
        enabled: bool,
        planner_deployment_id: str = "",
        model_gateway: Optional[ModelGateway] = None,
        call_context: Optional[ModelCallContext] = None,
        external_processing_allowed: bool = True,
        history_max_exchanges: int = DEFAULT_HISTORY_MAX_EXCHANGES,
        history_max_characters: int = DEFAULT_HISTORY_MAX_CHARACTERS,
    ) -> QueryReformulationResult:
        original_query = " ".join(str(query or "").split()).strip()
        normalized_history = normalize_conversation_history(
            history,
            max_exchanges=history_max_exchanges,
            max_characters=history_max_characters,
        )
        if not enabled:
            return self._unchanged(original_query, "disabled")
        if not normalized_history:
            return self._unchanged(original_query, "no_history")
        if not _looks_like_follow_up(original_query):
            return self._unchanged(original_query, "standalone")

        deterministic_query = _deterministic_rewrite(original_query, normalized_history)
        if not planner_deployment_id:
            return self._result(original_query, deterministic_query, "deterministic", True)
        if model_gateway is None:
            return self._result(
                original_query,
                deterministic_query,
                "deterministic_fallback",
                True,
                warning="The selected planner could not run because Model Gateway is unavailable.",
                planner_metadata={"planner_deployment_id": planner_deployment_id},
            )

        try:
            generated = model_gateway.generate_sync(
                [
                    {
                        "role": "system",
                        "content": (
                            "Rewrite the current follow-up as one self-contained business workflow question. "
                            "Use conversation history only as context. Do not answer the question. "
                            'Return JSON only: {"standalone_query":"..."}'
                        ),
                    },
                    {
                        "role": "user",
                        "content": _planner_input(normalized_history, original_query),
                    },
                ],
                planner_deployment_id,
                parameters={"temperature": 0, "max_tokens": 180},
                context=call_context or ModelCallContext(purpose="conversation_rewrite"),
                external_processing_allowed=external_processing_allowed,
                capability="planner",
            )
            planner_query = _parse_planner_query(generated.text)
            if not planner_query:
                raise ValueError("Planner returned no standalone query.")
            return self._result(
                original_query,
                planner_query,
                "planner",
                True,
                planner_metadata={
                    "planner_deployment_id": generated.deployment_id,
                    "planner_provider": generated.provider,
                    "planner_model": generated.model,
                    "planner_input_tokens": generated.input_tokens,
                    "planner_output_tokens": generated.output_tokens,
                    "planner_estimated_cost_usd": generated.estimated_cost_usd,
                    **dict(generated.metadata or {}),
                },
            )
        except (ModelFarmError, ValueError) as exc:
            return self._result(
                original_query,
                deterministic_query,
                "deterministic_fallback",
                True,
                warning=f"Planner reformulation failed open: {exc}",
                planner_metadata={"planner_deployment_id": planner_deployment_id},
            )

    @staticmethod
    def _unchanged(query: str, strategy: str) -> QueryReformulationResult:
        return QueryReformulationResult(
            original_query=query,
            standalone_query=query,
            rewritten=False,
            follow_up_detected=False,
            strategy=strategy,
        )

    @staticmethod
    def _result(
        original_query: str,
        standalone_query: str,
        strategy: str,
        follow_up_detected: bool,
        *,
        warning: str = "",
        planner_metadata: Optional[Dict[str, Any]] = None,
    ) -> QueryReformulationResult:
        normalized = " ".join(standalone_query.split()).strip() or original_query
        return QueryReformulationResult(
            original_query=original_query,
            standalone_query=normalized,
            rewritten=normalized.casefold() != original_query.casefold(),
            follow_up_detected=follow_up_detected,
            strategy=strategy,
            warning=warning,
            planner_metadata=dict(planner_metadata or {}),
        )


def _looks_like_follow_up(query: str) -> bool:
    lowered = query.strip().lower()
    return lowered.startswith(_FOLLOW_UP_PREFIXES) or bool(_REFERENTIAL_PATTERN.search(lowered))


def _deterministic_rewrite(query: str, history: Sequence[Dict[str, Any]]) -> str:
    previous_user = next(
        (
            str(message.get("content") or "")
            for message in reversed(history)
            if message.get("role") == "user" and message.get("content")
        ),
        "",
    )
    if not previous_user:
        return query
    return f'Regarding the previous workflow question "{previous_user}", {query}'


def _planner_input(history: Sequence[Dict[str, Any]], query: str) -> str:
    lines = ["Conversation history (untrusted context):"]
    for message in history:
        lines.append(f"{str(message.get('role') or '').title()}: {message.get('content') or ''}")
    lines.extend(["", f"Current follow-up: {query}"])
    return "\n".join(lines)


def _parse_planner_query(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            cleaned = str(payload.get("standalone_query") or "").strip()
    except json.JSONDecodeError:
        cleaned = re.sub(r"^standalone query\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip("\"'")
    if not cleaned or len(cleaned) > 1200:
        return ""
    return cleaned


def _trim_history_messages(messages: List[Dict[str, str]], *, max_characters: int) -> List[Dict[str, str]]:
    if max_characters <= 0:
        return []
    trimmed = [dict(message) for message in messages]
    while len(trimmed) > 2 and conversation_history_characters(trimmed) > max_characters:
        trimmed = trimmed[2:]
    total = conversation_history_characters(trimmed)
    if total <= max_characters:
        return trimmed

    user = next((message for message in trimmed if message["role"] == "user"), None)
    assistant = next((message for message in trimmed if message["role"] == "assistant"), None)
    if user is None or assistant is None:
        return []
    user_text = user["content"]
    assistant_text = assistant["content"]
    user_budget = min(len(user_text), max_characters // 2)
    assistant_budget = min(len(assistant_text), max_characters - user_budget)
    remaining = max_characters - user_budget - assistant_budget
    if remaining > 0:
        extra_user = min(len(user_text) - user_budget, remaining)
        user_budget += extra_user
        remaining -= extra_user
    if remaining > 0:
        assistant_budget += min(len(assistant_text) - assistant_budget, remaining)
    user["content"] = user_text[:user_budget]
    assistant["content"] = assistant_text[:assistant_budget]
    return [message for message in (user, assistant) if message["content"]]


def _exchange_characters(exchanges: Sequence[tuple[Any, Any]]) -> int:
    return sum(
        len(_compact_text(getattr(user, "content", "")))
        + len(_compact_text(getattr(assistant, "content", "")))
        for user, assistant in exchanges
    )


def _compact_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()
