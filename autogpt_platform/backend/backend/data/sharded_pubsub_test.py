"""ShardedSubscription's own contract. Losing a live stream (a drop, a silent
shard) is covered over a real Redis in ``event_bus_liveness_test``."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

import backend.data.redis_client as redis_client
import backend.data.sharded_pubsub as sharded_pubsub
from backend.data.sharded_pubsub import EventStreamLostError, ShardedSubscription
from backend.util.testing import is_tcp_port_reachable


class _FakePubSub:
    def __init__(self) -> None:
        self.connection: MagicMock | None = None
        self.ssubscribe = AsyncMock()
        self.aclose = AsyncMock()

    async def connect(self) -> None:
        self.connection = MagicMock()

    async def get_message(self, timeout: float | None = None) -> Any:
        await asyncio.sleep(timeout or 0)
        return None


@pytest.fixture
def fake_shard(monkeypatch: pytest.MonkeyPatch):
    pubsub = _FakePubSub()
    client = MagicMock(pubsub=MagicMock(return_value=pubsub), aclose=AsyncMock())
    monkeypatch.setattr(
        redis_client, "connect_sharded_pubsub_async", AsyncMock(return_value=client)
    )
    refresh = MagicMock()
    monkeypatch.setattr(redis_client, "refresh_slot_map", refresh)
    return pubsub, client, refresh


async def test_a_failed_subscribe_is_a_lost_stream_and_closes_its_connection(
    fake_shard,
):
    pubsub, client, refresh = fake_shard
    pubsub.ssubscribe.side_effect = RedisConnectionError("shard gone")

    with pytest.raises(EventStreamLostError):
        await ShardedSubscription.open("chan")

    client.aclose.assert_awaited_once()
    # The shard may have failed over: the next attempt reads a fresh slot map.
    refresh.assert_called_once()


async def test_an_unconfirmed_subscribe_gives_up_at_the_open_timeout(
    fake_shard, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(sharded_pubsub, "OPEN_TIMEOUT_S", 0.05)
    pubsub, client, _ = fake_shard

    async with asyncio.timeout(5):
        with pytest.raises(EventStreamLostError):
            await ShardedSubscription.open("chan")

    client.aclose.assert_awaited_once()


async def test_closing_never_raises(fake_shard):
    pubsub, client, _ = fake_shard

    async def confirm(channel: str) -> None:
        pubsub.get_message = AsyncMock(
            return_value={"type": "ssubscribe", "channel": channel, "data": 1}
        )

    pubsub.ssubscribe.side_effect = confirm
    subscription = await ShardedSubscription.open("chan")
    pubsub.aclose.side_effect = RedisConnectionError("already gone")
    client.aclose.side_effect = OSError("already gone")

    await subscription.aclose()

    pubsub.aclose.assert_awaited_once()
    client.aclose.assert_awaited_once()


def test_the_slot_map_is_refreshed_at_most_every_few_seconds(
    monkeypatch: pytest.MonkeyPatch,
):
    cluster = MagicMock()
    monkeypatch.setattr(redis_client, "get_redis", lambda: cluster)
    monkeypatch.setattr(redis_client, "_slot_map_refreshed_at", float("-inf"))

    redis_client.refresh_slot_map()
    redis_client.refresh_slot_map()

    cluster.nodes_manager.initialize.assert_called_once()


def _shard_count() -> int:
    if not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT):
        return 0
    return len(redis_client.get_redis().get_primaries())


@pytest.mark.skipif(
    _shard_count() < 2, reason="needs a Redis cluster with two or more shards"
)
async def test_a_moved_reply_points_the_next_subscription_at_the_new_owner():
    """After a reshard the cached slot map still names the old shard, whose
    reply to SSUBSCRIBE is MOVED; the next attempt must dial the new one
    instead of the stale shard for as long as the client stays cached."""
    cluster = redis_client.get_redis()
    channel = f"reshard-test/{uuid4()}"
    slot = cluster.keyslot(channel)
    owner = cluster.nodes_manager.slots_cache[slot][0]
    stale = next(n for n in cluster.get_primaries() if n.name != owner.name)
    cluster.nodes_manager.slots_cache[slot] = [stale]
    try:
        with pytest.raises(EventStreamLostError, match="moved"):
            await ShardedSubscription.open(channel)
        assert cluster.nodes_manager.slots_cache[slot][0].port == owner.port

        async with await ShardedSubscription.open(channel) as subscription:
            publisher = await redis_client.get_redis_async()
            async with asyncio.timeout(8):
                await publisher.execute_command("SPUBLISH", channel, "after-moved")
                async for message in subscription.messages():
                    if message["type"] == "smessage":
                        assert message["data"] == "after-moved"
                        break
    finally:
        cluster.nodes_manager.initialize()
        await redis_client.disconnect_async()


def test_a_moved_reply_updates_only_the_cached_slot_map():
    """``follow_moved_slot`` must not touch the network: it runs on the event
    loop, right where the MOVED reply was read."""
    cluster = MagicMock()
    error = MagicMock()
    with patch.object(redis_client, "get_redis", return_value=cluster):
        redis_client.follow_moved_slot(error)
    cluster.nodes_manager.move_slot.assert_called_once_with(error)
    cluster.nodes_manager.initialize.assert_not_called()
