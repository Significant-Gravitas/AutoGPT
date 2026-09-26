"""A record write is bounded: a DatabaseManager that stalls or refuses costs a
pass the write deadline, retries included, and the abandoned request is
cancelled rather than left running.

Each test drives the real DatabaseManager RPC client; only the HTTP transport
under it is replaced."""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone

import httpx
import pytest

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_manager import DatabaseManagerAsyncClient
from backend.util.service import get_service_client

from . import store
from .schemas import DreamOperations

_DEADLINE = 0.2
# Generous against a slow CI box, and far below one retry backoff (~1 s) or
# the client's own 300 s request timeout.
_BOUND = 2.0


class _Hanging(httpx.AsyncBaseTransport):
    """Accepts every request and never answers."""

    def __init__(self) -> None:
        self.requests = 0
        self.cancelled = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        raise AssertionError("a hanging transport never answers")


class _Refusing(httpx.AsyncBaseTransport):
    """Refuses every connection, as a DatabaseManager that is down would."""

    def __init__(self) -> None:
        self.requests = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        raise httpx.ConnectError("connection refused", request=request)


@pytest.fixture
def deadline(monkeypatch: pytest.MonkeyPatch) -> float:
    monkeypatch.setattr(store, "RECORD_WRITE_TIMEOUT_SECONDS", _DEADLINE)
    return _DEADLINE


@pytest.fixture
async def rpc_over(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[[httpx.AsyncBaseTransport, bool], None]]:
    """Point the store at a real DatabaseManager client whose requests go to
    a given transport, retrying or not."""
    clients: list[DatabaseManagerAsyncClient] = []

    def use(transport: httpx.AsyncBaseTransport, retry: bool) -> None:
        client = get_service_client(DatabaseManagerAsyncClient, request_retry=retry)

        def over_transport() -> httpx.AsyncClient:
            return httpx.AsyncClient(transport=transport, base_url=client.base_url)

        # A client the RPC layer rebuilds after connection failures must
        # still go to the fake transport, never to a real address.
        monkeypatch.setattr(client, "_create_async_client", over_transport)
        clients.append(client)
        monkeypatch.setattr(store, "dream_db", lambda: client)

    yield use
    for client in clients:
        await client.aclose()


@pytest.mark.parametrize("retry", [True, False], ids=["retrying", "no-retry"])
async def test_a_stalled_update_costs_the_deadline_and_is_cancelled(
    deadline, rpc_over, caplog, retry
):
    hanging = _Hanging()
    rpc_over(hanging, retry)
    loop = asyncio.get_running_loop()

    started = loop.time()
    with caplog.at_level(logging.WARNING, logger=store.logger.name):
        await store.record_next_batch("p1", "recombine", "batch-1")

    assert loop.time() - started < _BOUND
    assert (hanging.requests, hanging.cancelled) == (1, 1)
    assert "could not record the next batch" in caplog.text


async def test_a_stalled_insert_costs_the_deadline(deadline, rpc_over, caplog):
    hanging = _Hanging()
    rpc_over(hanging, True)
    loop = asyncio.get_running_loop()

    started = loop.time()
    with caplog.at_level(logging.WARNING, logger=store.logger.name):
        await store.start_pass(
            "p1",
            MemoryScope.for_user("u1"),
            route="sync_baseline",
            trigger="cron",
            started_at=datetime.now(timezone.utc),
        )

    assert loop.time() - started < _BOUND
    assert hanging.cancelled == 1
    assert "could not insert its record" in caplog.text


async def test_the_deadline_covers_the_clients_retries(deadline, rpc_over, caplog):
    """A retrying client backs off for seconds between attempts, up to a
    hundred of them; the write deadline cuts all of that short."""
    refusing = _Refusing()
    rpc_over(refusing, True)
    loop = asyncio.get_running_loop()

    started = loop.time()
    with caplog.at_level(logging.WARNING, logger=store.logger.name):
        await store.record_applying("p1", DreamOperations())

    assert loop.time() - started < _BOUND
    assert refusing.requests >= 1
    assert "could not record the start of apply" in caplog.text
