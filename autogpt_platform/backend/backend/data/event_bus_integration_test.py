"""AsyncRedisEventBus.listen_events over the live 3-shard Redis cluster, with
the network to the owning shard cut through a TCP proxy. Skips when the
cluster is unreachable."""

import asyncio
from uuid import uuid4

import pytest
from pydantic import BaseModel

import backend.data.redis_client as redis_client
from backend.data.event_bus import AsyncRedisEventBus, EventStreamLostError
from backend.util.testing import (
    is_tcp_port_reachable,
    shard_proxies_routed,
    shard_subscribers,
    spublish_on_shard,
)

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="local redis cluster not reachable; skip event bus integration",
)


class _Event(BaseModel):
    marker: str


class _Bus(AsyncRedisEventBus[_Event]):
    Model = _Event

    @property
    def event_bus_name(self) -> str:
        return "event_bus_integration"


@pytest.fixture
async def shard_proxies(monkeypatch):
    async with shard_proxies_routed(monkeypatch) as proxy_for:
        yield proxy_for


async def _wait_until(predicate, timeout: float) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return False


@pytest.mark.asyncio
async def test_a_listener_notices_an_event_lost_in_a_brief_redis_blip(
    shard_proxies,
) -> None:
    """A blip short enough for redis-py to reconnect by itself loses whatever
    was published meanwhile. The listener must end with EventStreamLostError
    so its owner refetches, instead of carrying on deaf to that event."""
    bus = _Bus()
    channel_key = f"{{user-{uuid4().hex[:8]}}}/exec/{uuid4().hex[:8]}"
    full_channel = f"{bus.event_bus_name}/{channel_key}"
    proxy = await shard_proxies(full_channel)
    received: list[str] = []

    async def listen() -> None:
        async for event in bus.listen_events(channel_key):
            received.append(event.marker)

    listener = asyncio.create_task(listen())
    try:
        assert await _wait_until(
            lambda: shard_subscribers(proxy.upstream, full_channel) == 1, 10.0
        ), "listener never subscribed"

        await proxy.stop()
        assert await _wait_until(
            lambda: shard_subscribers(proxy.upstream, full_channel) == 0, 5.0
        ), "the shard never noticed the dropped subscriber"
        # Published exactly once, with nobody subscribed: Redis drops it.
        payload = bus.Message(payload=_Event(marker="completed")).model_dump_json()
        assert spublish_on_shard(proxy.upstream, full_channel, payload) == 0
        await proxy.start()

        done, _ = await asyncio.wait({listener}, timeout=20.0)
        assert done, "listener carried on deaf after losing an event"
        with pytest.raises(EventStreamLostError):
            listener.result()
        assert received == []
    finally:
        listener.cancel()
