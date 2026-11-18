"""A WebSocket's shard stream must report a lost Redis stream, so the socket
is closed and the client resubscribes and refetches what it missed. Runs
against the real Redis cluster, with the stream's shard connection routed
through a TCP proxy the test breaks. Skips when no cluster is reachable."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from uuid import uuid4

import pytest

import backend.data.redis_client as redis_client
from backend.api.conn_manager import _ShardStream
from backend.api.ws_outbox import WSBacklog, WSOutbox
from backend.util.testing import TcpProxy, is_tcp_port_reachable, redis_shard_proxies

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="local redis cluster not reachable; skip shard-stream liveness",
)

_PING_AFTER_S = 0.25
_PONG_TIMEOUT_S = 1.0
_WITHIN_S = 25.0


@pytest.fixture
async def shard_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[[str], Awaitable[TcpProxy]]]:
    monkeypatch.setattr(redis_client, "PUBSUB_PING_AFTER", _PING_AFTER_S)
    monkeypatch.setattr(redis_client, "PUBSUB_PONG_TIMEOUT", _PONG_TIMEOUT_S)
    async with redis_shard_proxies() as route:
        yield route
    await redis_client.disconnect_async()


async def _open_stream(channel: str) -> tuple[_ShardStream, asyncio.Event]:
    """A stream on ``channel`` that has delivered one message, and the event
    its ``on_lost`` sets."""
    lost = asyncio.Event()
    got: list[str] = []

    async def on_lost() -> None:
        lost.set()

    async def handler(data: bytes | str | None) -> None:
        got.append(str(data))

    async def on_overflow(why: str) -> None:
        """A handler that only appends never falls behind."""

    stream = _ShardStream(on_lost, WSOutbox(on_overflow, WSBacklog()))
    await stream.add(channel, handler)
    try:
        cluster = await redis_client.get_redis_async()
        async with asyncio.timeout(_WITHIN_S):
            while not got:
                await cluster.execute_command("SPUBLISH", channel, "first")
                await asyncio.sleep(0.1)
    except BaseException:
        await stream.stop()
        raise
    return stream, lost


@pytest.mark.parametrize("breakage", ["dropped", "black-holed"])
async def test_a_stream_reports_its_lost_connection(shard_proxy, breakage: str):
    """Dropped: redis-py would reconnect and resubscribe on its own, and what
    was published in between is gone without a word. Black-holed: nothing
    errors at all, the replies just stop."""
    channel = f"ws-liveness/{breakage}-{uuid4()}"
    proxy = await shard_proxy(channel)
    stream, lost = await _open_stream(channel)
    try:
        if breakage == "dropped":
            proxy.drop()
        else:
            proxy.black_hole = True
        async with asyncio.timeout(_WITHIN_S):
            await lost.wait()
    finally:
        await stream.stop()
