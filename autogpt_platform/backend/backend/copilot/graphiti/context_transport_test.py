"""Warm context against a FalkorDB that is slow to answer, on real sockets.

Building a FalkorDB client sends a synchronous INFO (falkordb's cluster
probe). Built on the event loop, a server that does not answer stalls the
whole loop, deadline timers included, so a refresh overruns its budget and
every other coroutine on the loop waits with it. These run the real refresh
path with a loop heartbeat: the loop must keep ticking, the refresh must
return within its budget with no block, and the build it gave up on must
still end (its transport deadlines) or, when the server does answer, serve
the next turn from the cache. A query slower than the refresh's budget ends
the read at the budget, not at the socket, and a write through the same
kind of client still completes.

Run the live case with FalkorDB reachable (see ``conftest.py``)::

    poetry run pytest -m integration backend/copilot/graphiti/context_transport_test.py
"""

import asyncio
import contextlib
import select
import socket
import threading
import time
from collections.abc import AsyncIterator, Iterator
from unittest.mock import AsyncMock

import pytest

from . import client, context, context_refresh
from .config import graphiti_config
from .falkordb_driver import open_driver
from .recall_integration_fixtures import ALICE, ingest_facts

# A refresh's budget in these tests, and what scheduling may add to it.
_BUDGET = 0.3
_SLACK = 0.3
# The longest the loop may go without a tick. A construction on the loop
# would stall it for the whole socket deadline (a second here).
_MAX_STALL = 0.2
_MESSAGE = "who works on the Atlas project now"


class _Heartbeat:
    """Ticks every 10 ms on the loop and keeps the longest gap it saw."""

    def __init__(self) -> None:
        self.max_gap = 0.0

    async def beat(self) -> None:
        last = time.perf_counter()
        while True:
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            self.max_gap = max(self.max_gap, now - last)
            last = now


@contextlib.asynccontextmanager
async def _heartbeat() -> AsyncIterator[_Heartbeat]:
    heartbeat = _Heartbeat()
    task = asyncio.create_task(heartbeat.beat())
    await asyncio.sleep(0.02)
    try:
        yield heartbeat
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@contextlib.contextmanager
def _silent_server() -> Iterator[int]:
    """A TCP port that completes the handshake and never answers: the
    operating system accepts into the backlog, nothing reads or replies."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    try:
        yield listener.getsockname()[1]
    finally:
        listener.close()


@pytest.fixture
def silent_falkordb(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    with _silent_server() as port:
        monkeypatch.setattr(graphiti_config, "falkordb_host", "127.0.0.1")
        monkeypatch.setattr(graphiti_config, "falkordb_port", port)
        monkeypatch.setattr(graphiti_config, "falkordb_socket_connect_timeout", 0.5)
        monkeypatch.setattr(graphiti_config, "falkordb_socket_timeout", 1.0)
        monkeypatch.setattr(graphiti_config, "context_refresh_timeout", _BUDGET)
        yield


@pytest.mark.asyncio
async def test_a_silent_falkordb_neither_stalls_the_loop_nor_outlasts_the_budget(
    silent_falkordb: None,
) -> None:
    async with _heartbeat() as heartbeat:
        started = time.perf_counter()
        block = await context_refresh.refresh_warm_context("user-silent", _MESSAGE)
        elapsed = time.perf_counter() - started
        build = client._get_loop_state().building["user_user-silent"]

    assert block is None
    assert elapsed < _BUDGET + _SLACK, f"the refresh took {elapsed:.3f}s"
    assert heartbeat.max_gap < _MAX_STALL, f"the loop stalled {heartbeat.max_gap:.3f}s"
    # The build the refresh stopped waiting for ends at its socket deadline
    # (a second here), with the server's silence as the error; it is not left
    # holding a thread.
    done, _ = await asyncio.wait({build}, timeout=3)
    assert build in done, "the abandoned build is still waiting on the server"
    assert build.exception() is not None


class _DelayingProxy:
    """A transparent TCP proxy that holds replies back.

    By default it holds the server's first reply (the first client build's
    probe); with ``queries_only`` it holds every reply to a graph query
    instead. Threads, not the event loop, so a stall on the loop cannot hide
    it.
    """

    def __init__(
        self, upstream: tuple[str, int], delay: float, *, queries_only: bool = False
    ) -> None:
        self._upstream = upstream
        self._delay = delay
        self._queries_only = queries_only
        self._delayed = False
        self._claim = threading.Lock()
        self._done = threading.Event()
        self._sockets: list[socket.socket] = []
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(16)
        self._listener.settimeout(0.2)
        self.port = self._listener.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def close(self) -> None:
        self._done.set()
        self._listener.close()
        for sock in self._sockets:
            sock.close()

    def _serve(self) -> None:
        while not self._done.is_set():
            try:
                downstream, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            upstream = socket.create_connection(self._upstream, timeout=2)
            self._sockets += [downstream, upstream]
            threading.Thread(
                target=self._relay, args=(downstream, upstream), daemon=True
            ).start()

    def _relay(self, downstream: socket.socket, upstream: socket.socket) -> None:
        query_sent = False
        try:
            while not self._done.is_set():
                ready, _, _ = select.select([downstream, upstream], [], [], 0.2)
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    if source is upstream:
                        if not self._queries_only:
                            self._hold_first_reply()
                        elif query_sent:
                            query_sent = False
                            time.sleep(self._delay)
                        downstream.sendall(data)
                    else:
                        query_sent = query_sent or _is_graph_query(data)
                        upstream.sendall(data)
        except OSError:
            return

    def _hold_first_reply(self) -> None:
        with self._claim:
            first, self._delayed = not self._delayed, True
        if first:
            time.sleep(self._delay)


def _is_graph_query(request: bytes) -> bool:
    return b"GRAPH.QUERY" in request or b"GRAPH.RO_QUERY" in request


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_slow_falkordb_reply_does_not_stall_the_loop_and_the_build_serves_the_next_turn(
    scope_graph, stub_graphiti_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first reply of the first connection (the client build's probe)
    comes back 1.2 s late, inside the socket deadline: the refresh gives up
    at its budget without stalling the loop, the build finishes behind it
    and caches the client, and the next refresh reads through it."""
    driver, scope = scope_graph
    body = "Quick note: Alice works on Atlas, she joined in March"
    await ingest_facts(driver, scope, stub_graphiti_client, [ALICE], body=body)
    proxy = _DelayingProxy(
        (graphiti_config.falkordb_host, graphiti_config.falkordb_port), delay=1.2
    )
    monkeypatch.setattr(graphiti_config, "falkordb_host", "127.0.0.1")
    monkeypatch.setattr(graphiti_config, "falkordb_port", proxy.port)
    monkeypatch.setattr(graphiti_config, "context_refresh_timeout", _BUDGET)
    # The episode read goes through the proxied client; the fact search
    # would call the embedding API.
    monkeypatch.setattr(context, "search_facts", AsyncMock(return_value=[]))
    try:
        async with _heartbeat() as heartbeat:
            started = time.perf_counter()
            first = await context_refresh.refresh_warm_context(
                scope.owner_user_id, _MESSAGE
            )
            elapsed = time.perf_counter() - started
            build = client._get_loop_state().building[scope.group_id]
            await asyncio.wait_for(build, timeout=10)

        assert first is None
        assert elapsed < _BUDGET + _SLACK, f"the refresh took {elapsed:.3f}s"
        assert (
            heartbeat.max_gap < _MAX_STALL
        ), f"the loop stalled {heartbeat.max_gap:.3f}s"

        second = await context_refresh.refresh_warm_context(
            scope.owner_user_id, _MESSAGE
        )
        assert second is not None and body in second
    finally:
        await client.evict_client(scope.group_id)
        proxy.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_query_slower_than_the_budget_ends_the_read_at_the_budget_and_a_write_still_completes(
    scope_graph, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every graph query answers 2.5 s late: after a refresh's budget, and
    after the 2 s reply timeout the memory code used to have, but well inside
    the socket timeout now. The socket decides neither outcome. The refresh
    ends at its budget with no block and the loop ticking; a write through
    the same kind of client the writers use (``open_driver``: forget, the
    dream, the memory API routes) waits the reply out and completes."""
    _, scope = scope_graph
    delay = 2.5
    proxy = _DelayingProxy(
        (graphiti_config.falkordb_host, graphiti_config.falkordb_port),
        delay=delay,
        queries_only=True,
    )
    monkeypatch.setattr(graphiti_config, "falkordb_host", "127.0.0.1")
    monkeypatch.setattr(graphiti_config, "falkordb_port", proxy.port)
    monkeypatch.setattr(graphiti_config, "context_refresh_timeout", _BUDGET)
    # The episode read goes through the proxied client; the fact search
    # would call the embedding API.
    monkeypatch.setattr(context, "search_facts", AsyncMock(return_value=[]))
    try:
        # Built before the clock starts, so the read waits on the query and
        # not on the build (the build's probe is not a graph query).
        await client.get_graphiti_client(scope.group_id)
        async with _heartbeat() as heartbeat:
            started = time.perf_counter()
            block = await context_refresh.refresh_warm_context(
                scope.owner_user_id, _MESSAGE
            )
            read_took = time.perf_counter() - started

            writer = open_driver(scope)
            try:
                started = time.perf_counter()
                written = await writer.execute_query(
                    "MERGE (n:TransportProbe {name: $name}) RETURN n.name AS name",
                    name="slow write",
                )
                write_took = time.perf_counter() - started
            finally:
                await writer.close()

        assert block is None
        assert read_took < _BUDGET + _SLACK, f"the refresh took {read_took:.3f}s"
        assert (
            heartbeat.max_gap < _MAX_STALL
        ), f"the loop stalled {heartbeat.max_gap:.3f}s"
        assert written is not None
        assert written[0] == [{"name": "slow write"}]
        assert write_took >= delay - 0.1, f"the write took {write_took:.3f}s"
    finally:
        await client.evict_client(scope.group_id)
        proxy.close()
