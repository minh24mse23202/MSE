from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

from aragbiz.chat import PostgresChatRepository
from aragbiz.knowledge_store import PostgresKnowledgeRepository


class _FakeConnection:
    def exec_driver_sql(self, statement):
        return None

    def execute(self, statement, parameters=None):
        return None


class _FakeTransaction:
    def __init__(self, engine):
        self.engine = engine

    def __enter__(self):
        with self.engine.lock:
            self.engine.begin_count += 1
        time.sleep(0.01)
        return _FakeConnection()

    def __exit__(self, exc_type, exc, traceback):
        return False


class _FakeEngine:
    def __init__(self):
        self.begin_count = 0
        self.lock = threading.Lock()

    def begin(self):
        return _FakeTransaction(self)


def test_postgres_chat_schema_initializes_once_under_concurrency() -> None:
    repository = PostgresChatRepository.__new__(PostgresChatRepository)
    repository.engine = _FakeEngine()
    repository._initialize_lock = threading.Lock()
    repository._initialized = False

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: repository.initialize(), range(16)))

    assert repository.engine.begin_count == 1
    assert repository._initialized is True


def test_postgres_knowledge_schema_initializes_once_under_concurrency() -> None:
    repository = PostgresKnowledgeRepository.__new__(PostgresKnowledgeRepository)
    repository.engine = _FakeEngine()
    repository._initialize_lock = threading.Lock()
    repository._initialized = False

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: repository.initialize(), range(16)))

    assert repository.engine.begin_count == 1
    assert repository._initialized is True
