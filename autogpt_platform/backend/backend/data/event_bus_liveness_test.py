"""An event-bus listener must notice when its Redis stream is lost, so its
owner can resubscribe and re-read what it missed. Runs against the real Redis
cluster, with the listener's shard connection routed through a TCP proxy the
test breaks: dropped, as a restarting shard does, or black-holed, as a
partitioned one does. Skips when no cluster is reachable."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from uuid import uuid4

import pytest
from pydantic import BaseModel

import backend.data.event_bus as event_bus
import backend.data.redis_client as redis_client
from backend.data.event_bus import AsyncRedisEventBus
from backend.data.sharded_pubsub import EventStreamLostError
from backend.util.testing import TcpProxy, is_tcp_port_reachable, redis_shard_proxies

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="local redis cluster not reachable; skip event-bus liveness",
)

# Short liveness timings, so a silent shard is noticed inside the test; the
# assertions below allow several times as long.
_PING_AFTER_S = 0.25
_PONG_TIMEOUT_S = 1.0
_WITHIN_S = 25.0


class _Event(BaseModel):
    n: int


class _Bus(AsyncRedisEventBus[_Event]):
    Model = _Event

    @property
    def event_bus_name(self) -> str:
        return "liveness_test"


@pytest.fixture
async def shard_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[Callable[[str], Awaitable[TcpProxy]]]:
    monkeypatch.setattr(redis_client, "PUBSUB_PING_AFTER", _PING_AFTER_S)
    monkeypatch.setattr(redis_client, "PUBSUB_PONG_TIMEOUT", _PONG_TIMEOUT_S)
    async with redis_shard_proxies() as route:
        yield route
    await redis_client.disconnect_async()


async def _publish_until(
    bus: _Bus, channel_key: str, n: int, received: Callable[[], bool]
) -> None:
    """Publish ``n`` until it arrives: a subscription confirms asynchronously,
    and a publish made before it lands is gone for good."""
    async with asyncio.timeout(_WITHIN_S):
        while not received():
            await bus.publish_event(_Event(n=n), channel_key)
            await asyncio.sleep(0.1)


async def test_a_listener_recovers_after_its_connection_drops(shard_proxy):
    bus = _Bus()
    channel_key = f"drop-{uuid4()}"
    proxy = await shard_proxy(f"liveness_test/{channel_key}")
    got: list[int] = []
    losses: list[Exception] = []

    async def listen_and_relisten_on_loss() -> None:
        while True:
            try:
                async for event in bus.listen_events(channel_key):
                    got.append(event.n)
            except Exception as e:
                losses.append(e)
                await asyncio.sleep(0.1)

    listener = asyncio.create_task(listen_and_relisten_on_loss())
    try:
        await _publish_until(bus, channel_key, 1, lambda: 1 in got)
        proxy.drop()
        await _publish_until(bus, channel_key, 2, lambda: 2 in got)
        # Recovered by resubscribing, not by redis-py quietly reconnecting:
        # only a reported loss tells the owner to re-read what it missed.
        assert losses and all(isinstance(e, EventStreamLostError) for e in losses)
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)


async def test_a_listener_notices_a_black_holed_shard(shard_proxy):
    bus = _Bus()
    channel_key = f"black-hole-{uuid4()}"
    proxy = await shard_proxy(f"liveness_test/{channel_key}")
    got: list[int] = []

    async def listen() -> None:
        async for event in bus.listen_events(channel_key):
            got.append(event.n)

    listener = asyncio.create_task(listen())
    try:
        await _publish_until(bus, channel_key, 1, lambda: 1 in got)
        proxy.black_hole = True
        done, _ = await asyncio.wait({listener}, timeout=_WITHIN_S)
        assert (
            listener in done
        ), f"listener still waiting {_WITHIN_S}s after its shard went silent"
        assert isinstance(listener.exception(), EventStreamLostError)
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)


async def test_a_quiet_but_healthy_shard_is_not_reported_lost(
    shard_proxy, monkeypatch: pytest.MonkeyPatch
):
    # Longer than the others, so a stalled runner can't make a healthy reply
    # late; the quiet spell below still outlasts it twice over.
    pong_timeout = 2.5
    monkeypatch.setattr(redis_client, "PUBSUB_PONG_TIMEOUT", pong_timeout)
    bus = _Bus()
    channel_key = f"quiet-{uuid4()}"
    await shard_proxy(f"liveness_test/{channel_key}")
    got: list[int] = []

    async def listen() -> None:
        async for event in bus.listen_events(channel_key):
            got.append(event.n)

    listener = asyncio.create_task(listen())
    try:
        await _publish_until(bus, channel_key, 1, lambda: 1 in got)
        # Many PING rounds with nothing published: a listener that took
        # silence itself for a lost shard would give up in this time.
        await asyncio.sleep(_PING_AFTER_S + 2 * pong_timeout)
        assert not listener.done(), listener.exception()
        await _publish_until(bus, channel_key, 2, lambda: 2 in got)
        assert not listener.done(), listener.exception()
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)


async def test_listen_with_resync_marks_each_gap_and_keeps_delivering(
    shard_proxy, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(event_bus, "RESUBSCRIBE_DELAY_S", 0.05)
    bus = _Bus()
    channel_key = f"resync-{uuid4()}"
    proxy = await shard_proxy(f"liveness_test/{channel_key}")
    seen: list[int | None] = []

    async def listen() -> None:
        async for event in bus.listen_with_resync(channel_key):
            seen.append(None if event is None else event.n)

    listener = asyncio.create_task(listen())
    try:
        await _publish_until(bus, channel_key, 1, lambda: 1 in seen)
        proxy.drop()
        await _publish_until(bus, channel_key, 2, lambda: 2 in seen)
    finally:
        listener.cancel()
        await asyncio.gather(listener, return_exceptions=True)
    # A resync mark opens each subscription, so state re-read on it can't
    # miss an event; and the gap between the two gets one too.
    assert seen[0] is None
    assert None in seen[seen.index(1) : seen.index(2)]
