"""ShardedSubscription's own contract. Losing a live stream (a drop, a silent
shard) is covered over a real Redis in ``event_bus_liveness_test``."""

import asyncio
from contextlib import aclosing
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import MovedError

import backend.data.redis_client as redis_client
import backend.data.sharded_pubsub as sharded_pubsub
from backend.data.sharded_pubsub import (
    EventStreamLostError,
    ShardedSubscription,
    _Liveness,
)
from backend.util.testing import is_tcp_port_reachable, redis_shard_proxies


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


@pytest.fixture
def confirmed_shard(fake_shard):
    """A shard that confirms the SSUBSCRIBE, as Redis does."""
    pubsub = fake_shard[0]

    async def confirm(channel: str) -> None:
        pubsub.get_message = AsyncMock(
            return_value={"type": "ssubscribe", "channel": channel, "data": 1}
        )

    pubsub.ssubscribe.side_effect = confirm
    return fake_shard


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


async def test_closing_never_raises(confirmed_shard):
    pubsub, client, _ = confirmed_shard
    subscription = await ShardedSubscription.open("chan")
    pubsub.aclose.side_effect = RedisConnectionError("already gone")
    client.aclose.side_effect = OSError("already gone")

    await subscription.aclose()

    pubsub.aclose.assert_awaited_once()
    client.aclose.assert_awaited_once()


def test_a_quiet_stream_is_pinged_and_lost_only_when_the_ping_goes_unanswered():
    live = _Liveness(now=0.0, ping_after=15.0, pong_timeout=10.0)
    assert not live.ping_due(14.9)
    assert live.ping_due(15.0)
    live.pinged(15.0)
    assert not live.ping_due(16.0)
    assert live.read_timeout(16.0) == pytest.approx(9.0)
    assert not live.lost(24.9)
    # Past the deadline, one more read first: a reply that came in while the
    # event loop was stalled can lose the race against the read's timeout.
    assert not live.lost(25.0)
    assert live.read_timeout(25.0) == sharded_pubsub._LAST_LOOK_S
    assert live.lost(26.0)


def test_any_reply_restarts_the_quiet_clock():
    live = _Liveness(now=0.0, ping_after=15.0, pong_timeout=10.0)
    live.pinged(15.0)
    live.heard(20.0)
    assert not live.lost(40.0)
    assert not live.ping_due(34.9)
    assert live.ping_due(35.0)


async def test_a_channel_no_shard_owns_is_a_lost_stream(
    fake_shard, monkeypatch: pytest.MonkeyPatch
):
    """Mid-reshard, ``resolve_shard_for_channel`` finds no owner; the caller
    must get the loss it resubscribes on, not a RuntimeError."""
    _, _, refresh = fake_shard
    monkeypatch.setattr(
        redis_client,
        "connect_sharded_pubsub_async",
        AsyncMock(side_effect=RuntimeError("No cluster node owns the keyslot")),
    )

    with pytest.raises(EventStreamLostError, match="No cluster node"):
        await ShardedSubscription.open("chan")

    refresh.assert_called_once()


async def test_a_moved_reply_is_a_lost_stream_even_if_following_it_fails(
    fake_shard, monkeypatch: pytest.MonkeyPatch
):
    pubsub, _, _ = fake_shard
    pubsub.ssubscribe.side_effect = MovedError("1234 10.0.0.9:7001")
    monkeypatch.setattr(
        redis_client, "follow_moved_slot", MagicMock(side_effect=OSError("down"))
    )

    with pytest.raises(EventStreamLostError, match="moved"):
        await ShardedSubscription.open("chan")


async def test_pong_replies_are_not_yielded_in_either_form(confirmed_shard):
    """With only sharded subscriptions Redis answers PING with the bare
    message, which redis-py would hand on as a garbled message."""
    pubsub, _, _ = confirmed_shard
    smessage = ["smessage", "chan", "event"]
    reads = iter(
        [
            sharded_pubsub._LIVENESS_PING,
            ["pong", sharded_pubsub._LIVENESS_PING],
            smessage,
        ]
    )

    async def parse_response(block: bool, timeout: float) -> Any:
        return next(reads)

    async def handle_message(response: Any) -> Any:
        return {"type": response[0], "channel": response[1], "data": response[2]}

    pubsub.parse_response = parse_response
    pubsub.handle_message = handle_message
    async with (
        await ShardedSubscription.open("chan") as subscription,
        aclosing(subscription.messages()) as messages,
    ):
        assert await anext(messages) == {
            "type": "smessage",
            "channel": "chan",
            "data": "event",
        }


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
    """Runs at collection: one short attempt, and no client left open."""
    if not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT):
        return 0
    try:
        cluster = redis_client.connect_once(timeout=2.0)
    except Exception:
        return 0
    try:
        return len(cluster.get_primaries())
    finally:
        cluster.close()


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

        async with (
            await ShardedSubscription.open(channel) as subscription,
            aclosing(subscription.messages()) as messages,
        ):
            publisher = await redis_client.get_redis_async()
            async with asyncio.timeout(8):
                await publisher.execute_command("SPUBLISH", channel, "after-moved")
                async for message in messages:
                    if message["type"] == "smessage":
                        assert message["data"] == "after-moved"
                        break
    finally:
        cluster.nodes_manager.initialize()
        await redis_client.disconnect_async()


@pytest.mark.skipif(_shard_count() < 1, reason="needs a Redis cluster")
async def test_a_dropped_connection_ends_the_stream_instead_of_resubscribing():
    """A failed SSUBSCRIBE or SUNSUBSCRIBE write drops the connection. redis-py
    would reconnect on the next read or command and resubscribe without a
    word, losing whatever was published meanwhile; the stream must end."""
    channel = f"{{reconnect-test-{uuid4()}}}/a"
    try:
        async with await ShardedSubscription.open(channel) as subscription:
            connection = subscription._pubsub.connection
            assert connection is not None
            await connection.disconnect()  # as a failed write leaves it

            with pytest.raises(EventStreamLostError):
                await subscription.add(f"{{reconnect-test-{uuid4()}}}/b")
            async with asyncio.timeout(8):
                with pytest.raises(EventStreamLostError):
                    async for _ in subscription.messages():
                        pass
            assert not connection.is_connected
    finally:
        await redis_client.disconnect_async()


@pytest.mark.skipif(_shard_count() < 1, reason="needs a Redis cluster")
async def test_a_reconnect_to_a_silent_shard_ends_the_stream():
    """A reconnect authenticates before redis-py runs its connect callbacks,
    and that read has no timeout of its own: a shard that accepts the
    connection but never answers must still end the stream, not hold it."""
    channel = f"{{silent-reconnect-{uuid4()}}}/a"
    try:
        async with redis_shard_proxies() as route:
            proxy = await route(channel)
            async with await ShardedSubscription.open(channel) as subscription:
                connection = subscription._pubsub.connection
                assert connection is not None
                # The reconnect sends AUTH first, as with REDIS_PASSWORD set.
                connection.password = "never-checked"
                proxy.black_hole = True
                await connection.disconnect()  # as a failed write leaves it

                async with asyncio.timeout(8):
                    with pytest.raises(EventStreamLostError):
                        async for _ in subscription.messages():
                            pass
    finally:
        await redis_client.disconnect_async()


def test_a_moved_reply_updates_only_the_cached_slot_map():
    """``follow_moved_slot`` re-points the slot from the MOVED reply itself,
    without a network round trip."""
    cluster = MagicMock()
    error = MagicMock()
    with patch.object(redis_client, "get_redis", return_value=cluster):
        redis_client.follow_moved_slot(error)
    cluster.nodes_manager.move_slot.assert_called_once_with(error)
    cluster.nodes_manager.initialize.assert_not_called()
