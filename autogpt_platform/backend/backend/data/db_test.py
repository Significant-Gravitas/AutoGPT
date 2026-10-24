import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from backend.data import db as db_module
from backend.data.db import DISCONNECT_TIMEOUT, disconnect, retry_on_dead_engine


# The root conftest spins a full test server for every test via an autouse
# session fixture. These two tests are pure unit tests over a mocked client,
# so shadow it here the way `backend/util/conftest.py` does for its directory
# -- narrowed to this module so the rest of `backend/data` is unaffected.
@pytest.fixture(scope="session")
def server():
    yield None


async def test_disconnect_bounds_the_engine_shutdown(monkeypatch):
    """Prisma's shutdown wait is unbounded unless a timeout is handed to it.

    The blocking wait sits inside an async function, so an engine that ignores
    the SIGINT takes the event loop down with it.
    """
    prisma = MagicMock()
    prisma.is_connected.side_effect = [True, False]
    prisma.disconnect = AsyncMock()
    monkeypatch.setattr(db_module, "prisma", prisma)

    await disconnect()

    prisma.disconnect.assert_awaited_once_with(DISCONNECT_TIMEOUT)


async def test_disconnect_is_a_noop_when_not_connected(monkeypatch):
    prisma = MagicMock()
    prisma.is_connected.return_value = False
    prisma.disconnect = AsyncMock()
    monkeypatch.setattr(db_module, "prisma", prisma)

    await disconnect()

    prisma.disconnect.assert_not_awaited()


async def test_callers_failing_on_one_dead_engine_restart_it_once(monkeypatch):
    """A second restart would kill the engine the first caller is retrying on."""
    monkeypatch.setattr(db_module, "disconnect", AsyncMock())
    connect = AsyncMock()
    monkeypatch.setattr(db_module, "connect", connect)
    calls = {"a": 0, "b": 0}
    both_failed = asyncio.Event()

    async def query(name: str) -> str:
        calls[name] += 1
        if calls[name] == 1:
            if all(calls.values()):
                both_failed.set()
            await both_failed.wait()
            raise httpx.ConnectError("All connection attempts failed")
        return name

    results = await asyncio.gather(
        retry_on_dead_engine(lambda: query("a")),
        retry_on_dead_engine(lambda: query("b")),
    )

    assert results == ["a", "b"]
    connect.assert_awaited_once()
