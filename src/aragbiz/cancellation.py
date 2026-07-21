from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Literal, Optional


class AnswerCancelled(Exception):
    """Raised when an in-flight answer operation is cancelled."""


@dataclass
class CancellationToken:
    request_id: str
    _event: asyncio.Event = field(default_factory=asyncio.Event)
    reason: str = "Answer generation was stopped by the user."

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self, reason: str = "") -> None:
        if reason:
            self.reason = reason
        self._event.set()

    async def wait(self) -> None:
        await self._event.wait()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise AnswerCancelled(self.reason)


CancellationStatus = Literal["active", "cancel_requested", "completed", "failed", "cancelled"]


@dataclass
class CancellationEntry:
    token: CancellationToken
    status: CancellationStatus = "active"
    updated_at: float = field(default_factory=time.monotonic)


class CancellationCoordinator:
    def __init__(self, max_entries: int = 1000):
        self.max_entries = max(10, max_entries)
        self._entries: Dict[str, CancellationEntry] = {}
        self._lock = threading.Lock()

    def register(self, request_id: str) -> CancellationToken:
        with self._lock:
            current = self._entries.get(request_id)
            if current and current.status in {"active", "cancel_requested"}:
                raise ValueError(f"An answer operation with request ID {request_id!r} is already active.")
            token = CancellationToken(request_id)
            self._entries[request_id] = CancellationEntry(token)
            self._prune()
            return token

    def token(self, request_id: str) -> Optional[CancellationToken]:
        with self._lock:
            entry = self._entries.get(request_id)
            return entry.token if entry else None

    def request_cancel(self, request_id: str, reason: str = "") -> CancellationStatus:
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                raise KeyError(request_id)
            if entry.status in {"completed", "failed", "cancelled"}:
                raise ValueError(f"Answer operation {request_id!r} is already {entry.status}.")
            entry.status = "cancel_requested"
            entry.updated_at = time.monotonic()
            entry.token.cancel(reason)
            return entry.status

    def finish(self, request_id: str, status: CancellationStatus) -> None:
        with self._lock:
            entry = self._entries.get(request_id)
            if entry is None:
                return
            entry.status = status
            entry.updated_at = time.monotonic()

    def status(self, request_id: str) -> Optional[CancellationStatus]:
        with self._lock:
            entry = self._entries.get(request_id)
            return entry.status if entry else None

    def _prune(self) -> None:
        if len(self._entries) <= self.max_entries:
            return
        terminal = sorted(
            (
                (request_id, entry.updated_at)
                for request_id, entry in self._entries.items()
                if entry.status in {"completed", "failed", "cancelled"}
            ),
            key=lambda item: item[1],
        )
        for request_id, _ in terminal[: len(self._entries) - self.max_entries]:
            self._entries.pop(request_id, None)
