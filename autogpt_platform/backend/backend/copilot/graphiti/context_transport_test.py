"""Warm context against a FalkorDB that is slow to answer, on real sockets.

Building a FalkorDB client sends a synchronous INFO (falkordb's cluster
probe). Built on the event loop, a server that does not answer stalls the
whole loop, deadline timers included, so a refresh overruns its budget and
every other coroutine on the loop waits with it. These run the real refresh
path with a loop heartbeat: the loop must keep ticking, the refresh must
return within its budget with no block, and the build it gave up on must
still end (its transport deadlines) or, when the server does answer, serve
the next turn from the cache.

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

from . import client, context
from .config import graphiti_config
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
        block = await context.refresh_warm_context("user-silent", _MESSAGE)
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
    """A transparent TCP proxy that holds the server's first reply back.

    Threads, not the event loop, so a stall on the loop cannot hide it.
    """

    def __init__(self, upstream: tuple[str, int], delay: float) -> None:
        self._upstream = upstream
        self._delay = delay
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
        try:
            while not self._done.is_set():
                ready, _, _ = select.select([downstream, upstream], [], [], 0.2)
                for source in ready:
                    data = source.recv(65536)
                    if not data:
                        return
                    if source is upstream:
                        self._hold_first_reply()
                        downstream.sendall(data)
                    else:
                        upstream.sendall(data)
        except OSError:
            return

    def _hold_first_reply(self) -> None:
        with self._claim:
            first, self._delayed = not self._delayed, True
        if first:
            time.sleep(self._delay)


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
            first = await context.refresh_warm_context(scope.owner_user_id, _MESSAGE)
            elapsed = time.perf_counter() - started
            build = client._get_loop_state().building[scope.group_id]
            await asyncio.wait_for(build, timeout=10)

        assert first is None
        assert elapsed < _BUDGET + _SLACK, f"the refresh took {elapsed:.3f}s"
        assert (
            heartbeat.max_gap < _MAX_STALL
        ), f"the loop stalled {heartbeat.max_gap:.3f}s"

        second = await context.refresh_warm_context(scope.owner_user_id, _MESSAGE)
        assert second is not None and body in second
    finally:
        await client.evict_client(scope.group_id)
        proxy.close()
