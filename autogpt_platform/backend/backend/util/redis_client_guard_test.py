"""Tests for the guard that fails a test leaving an async Redis Cluster client
open on its closed event loop (``redis_client_guard.py``), against the live
cluster. Skips when it is unreachable."""

import asyncio
from typing import Any

import pytest
from redis.asyncio.cluster import RedisCluster

import backend.data.redis_client as redis_client
from backend.util.cache_redis import _new_async_redis
from backend.util.testing import is_tcp_port_reachable

from .redis_client_guard import RedisClientGuard, is_open

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="local redis cluster not reachable",
)


@pytest.mark.parametrize(
    "close_client, close_loop, named",
    [
        pytest.param(False, True, True, id="left-open-on-a-closed-loop"),
        pytest.param(True, True, False, id="closed-before-its-loop"),
        pytest.param(False, False, False, id="its-loop-still-runs"),
    ],
)
def test_names_only_a_client_left_open_on_a_closed_loop(
    close_client: bool, close_loop: bool, named: bool
) -> None:
    guard = RedisClientGuard()
    guard.start()
    client = _new_async_redis()
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_ping_once(guard, client, close=close_client))
        if close_loop:
            loop.close()

        left_open = guard.stop()
    finally:
        _close(client, loop)

    assert len(left_open) == int(named)
    if named:
        assert left_open[0].startswith("Redis Cluster client first used at ")
        assert "redis_client_guard_test.py" in left_open[0]


def test_a_client_is_checked_only_in_the_test_that_used_it() -> None:
    guard = RedisClientGuard()
    client = _new_async_redis()
    loop = asyncio.new_event_loop()
    try:
        guard.start()
        loop.run_until_complete(_ping_once(guard, client, close=False))
        loop.close()
        # The test that used it is told...
        assert len(guard.stop()) == 1

        # ...and the next one, which used no client, isn't.
        guard.start()
        left_open = guard.stop()
    finally:
        _close(client, loop)

    assert left_open == []


def test_a_client_it_does_not_enforce_is_reported_not_named() -> None:
    """The session's guard enforces only for the shared cache's clients;
    ``redis_client``'s are reported instead until #17087 closes their leak."""
    guard = RedisClientGuard(enforced=lambda client: False)
    client = _new_async_redis()
    loop = asyncio.new_event_loop()
    try:
        guard.start()
        loop.run_until_complete(_ping_once(guard, client, close=False))
        loop.close()
        left_open = guard.stop()
    finally:
        _close(client, loop)

    assert left_open == []
    assert len(guard.tolerated) == 1
    assert guard.tolerated[0].startswith("Redis Cluster client first used at ")


async def _ping_once(guard: RedisClientGuard, client: RedisCluster, *, close: bool):
    execute = guard.wrap(_execute_past_the_session_guard)
    assert await execute(client, "PING")
    assert is_open(client)
    if close:
        await client.aclose()
        assert not is_open(client)


async def _execute_past_the_session_guard(
    client: RedisCluster, *args: Any, **kwargs: Any
) -> Any:
    """Sends through a node, not ``RedisCluster.execute_command``, so the
    session's own guard doesn't fail this test for the client it leaves open
    on purpose."""
    await client.initialize()
    return await client.get_default_node().execute_command(*args, **kwargs)


def _close(client: RedisCluster, loop: asyncio.AbstractEventLoop) -> None:
    """Closes a client left open on purpose, on a loop of its own if its own
    has closed, so it reports nothing when a later test collects it."""
    if not loop.is_closed():
        try:
            loop.run_until_complete(client.aclose())
        finally:
            loop.close()
        return
    cleanup = asyncio.new_event_loop()
    try:
        cleanup.run_until_complete(_forget_connections(client))
    finally:
        cleanup.close()
    assert not is_open(client)


async def _forget_connections(client: RedisCluster) -> None:
    """Each connection's transport belongs to the closed loop, so closing one
    raises; redis-py forgets the connection all the same."""
    for close in (
        client.aclose,
        client.nodes_manager.aclose,
        lambda: client.nodes_manager.aclose("startup_nodes"),
    ):
        try:
            await close()
        except RuntimeError:
            pass
