"""The fake provider and the platform ledger each billing scenario runs on."""

from collections.abc import AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from backend.copilot import database_manager_fixtures
from backend.copilot.sdk import service as sdk_service

from .ledger import PlatformLedger
from .provider import FakeProvider

# No database manager runs in the copilot-sdk shard (see sdk/conftest.py).
fake_database_manager_clients = database_manager_fixtures.fake_database_manager_clients
database_manager = database_manager_fixtures.database_manager


@pytest_asyncio.fixture(loop_scope="function")
async def provider() -> AsyncIterator[FakeProvider]:
    async with FakeProvider() as fake:
        yield fake


@pytest.fixture
def ledger(monkeypatch: pytest.MonkeyPatch) -> PlatformLedger:
    ledger = PlatformLedger()
    ledger.install(monkeypatch)
    return ledger


@pytest.fixture(autouse=True)
def no_memory_services(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Billing harnesses must isolate memory work before it reaches a service."""
    recall = AsyncMock(name="memory recall", return_value="")
    refresh = MagicMock(name="memory refresh", return_value=None)
    ingest = AsyncMock(name="memory ingestion")
    learn = AsyncMock(name="skill learning")
    monkeypatch.setattr(sdk_service, "fetch_warm_context", recall)
    monkeypatch.setattr(sdk_service.context_refresh, "start_refresh", refresh)
    monkeypatch.setattr(sdk_service, "_enqueue_graphiti_turn", ingest)
    monkeypatch.setattr(sdk_service, "capture_chat_turn", learn)
    yield
    for boundary in (recall, refresh, ingest, learn):
        boundary.assert_not_called()
