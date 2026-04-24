"""Unit tests for the cluster-only Redis client in ``redis_client``.

Patches the redis-py constructors + ``ping()`` so no real Redis is needed.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import RedisCluster

import backend.data.redis_client as redis_client


@pytest.fixture(autouse=True)
def _reset_module_caches() -> None:
    """Flush cached singletons between tests so each test sees a fresh connect."""
    redis_client.get_redis.cache_clear()
    redis_client.get_redis_pubsub.cache_clear()
    redis_client._async_clients.clear()
    redis_client._async_pubsub_clients.clear()


def test_connect_builds_redis_cluster() -> None:
    with patch.object(redis_client, "RedisCluster", autospec=True) as mock_cluster:
        mock_cluster.return_value = MagicMock(spec=RedisCluster)
        client = redis_client.connect()

    mock_cluster.assert_called_once()
    kwargs = mock_cluster.call_args.kwargs
    assert kwargs["password"] == redis_client.PASSWORD
    assert kwargs["decode_responses"] is True
    assert kwargs["socket_timeout"] == redis_client.SOCKET_TIMEOUT
    assert kwargs["socket_connect_timeout"] == redis_client.SOCKET_CONNECT_TIMEOUT
    assert kwargs["socket_keepalive"] is True
    assert kwargs["health_check_interval"] == redis_client.HEALTH_CHECK_INTERVAL
    assert kwargs["address_remap"] is redis_client._address_remap
    startup = kwargs["startup_nodes"]
    assert len(startup) == 1
    # ClusterNode resolves "localhost" → "127.0.0.1" internally; both are
    # valid representations of the configured host.
    assert startup[0].host in {redis_client.HOST, "127.0.0.1"}
    assert startup[0].port == redis_client.PORT
    client.ping.assert_called_once()


def test_address_remap_pins_host_and_preserves_port() -> None:
    """Default remap rewrites announced shard host to the configured seed."""
    with patch.object(redis_client, "USE_ANNOUNCED_ADDRESS", False):
        assert redis_client._address_remap(("any-other-host", 6380)) == (
            redis_client.HOST,
            6380,
        )


def test_address_remap_passthrough_when_use_announced_address() -> None:
    """When announced addresses resolve directly, remap leaves them alone."""
    with patch.object(redis_client, "USE_ANNOUNCED_ADDRESS", True):
        assert redis_client._address_remap(("redis-1", 17001)) == ("redis-1", 17001)


@pytest.mark.asyncio
async def test_connect_async_builds_async_redis_cluster() -> None:
    with patch.object(redis_client, "AsyncRedisCluster", autospec=True) as mock_cluster:
        fake = MagicMock(spec=AsyncRedisCluster)
        fake.ping = AsyncMock()
        mock_cluster.return_value = fake
        client = await redis_client.connect_async()

    mock_cluster.assert_called_once()
    kwargs = mock_cluster.call_args.kwargs
    assert kwargs["password"] == redis_client.PASSWORD
    assert kwargs["decode_responses"] is True
    assert kwargs["socket_timeout"] == redis_client.SOCKET_TIMEOUT
    assert kwargs["socket_connect_timeout"] == redis_client.SOCKET_CONNECT_TIMEOUT
    assert kwargs["socket_keepalive"] is True
    assert kwargs["health_check_interval"] == redis_client.HEALTH_CHECK_INTERVAL
    assert kwargs["address_remap"] is redis_client._address_remap
    startup = kwargs["startup_nodes"]
    assert len(startup) == 1
    assert startup[0].host in {redis_client.HOST, "127.0.0.1"}
    assert startup[0].port == redis_client.PORT
    client.ping.assert_awaited_once()


def test_connect_pubsub_builds_plain_redis() -> None:
    with patch.object(redis_client, "Redis", autospec=True) as mock_redis:
        mock_redis.return_value = MagicMock(spec=Redis)
        client = redis_client.connect_pubsub()

    mock_redis.assert_called_once()
    kwargs = mock_redis.call_args.kwargs
    assert kwargs["host"] == redis_client.HOST
    assert kwargs["port"] == redis_client.PORT
    assert kwargs["decode_responses"] is True
    client.ping.assert_called_once()


@pytest.mark.asyncio
async def test_connect_pubsub_async_builds_plain_async_redis() -> None:
    with patch.object(redis_client, "AsyncRedis", autospec=True) as mock_async:
        fake = MagicMock(spec=AsyncRedis)
        fake.ping = AsyncMock()
        mock_async.return_value = fake
        client = await redis_client.connect_pubsub_async()

    mock_async.assert_called_once()
    kwargs = mock_async.call_args.kwargs
    assert kwargs["host"] == redis_client.HOST
    assert kwargs["port"] == redis_client.PORT
    assert kwargs["decode_responses"] is True
    client.ping.assert_awaited_once()


def test_get_redis_caches_connect() -> None:
    with patch.object(redis_client, "connect", autospec=True) as mock_connect:
        mock_connect.return_value = MagicMock(spec=RedisCluster)
        client_a = redis_client.get_redis()
        client_b = redis_client.get_redis()

    assert client_a is client_b
    mock_connect.assert_called_once()


def test_get_redis_pubsub_caches_connect() -> None:
    with patch.object(redis_client, "connect_pubsub", autospec=True) as mock_conn:
        mock_conn.return_value = MagicMock(spec=Redis)
        a = redis_client.get_redis_pubsub()
        b = redis_client.get_redis_pubsub()

    assert a is b
    mock_conn.assert_called_once()


@pytest.mark.asyncio
async def test_get_redis_async_caches_connect() -> None:
    with patch.object(redis_client, "connect_async", autospec=True) as mock_conn:
        fake = MagicMock(spec=AsyncRedisCluster)
        mock_conn.return_value = fake
        a = await redis_client.get_redis_async()
        b = await redis_client.get_redis_async()

    assert a is b
    mock_conn.assert_called_once()


@pytest.mark.asyncio
async def test_get_redis_pubsub_async_caches_per_loop() -> None:
    """Repeated calls on the same loop reuse the same client."""
    with patch.object(redis_client, "connect_pubsub_async", autospec=True) as m:
        fake = MagicMock(spec=AsyncRedis)
        m.return_value = fake
        a = await redis_client.get_redis_pubsub_async()
        b = await redis_client.get_redis_pubsub_async()

    assert a is b
    m.assert_called_once()


def test_disconnect_closes_cached_client() -> None:
    with patch.object(redis_client, "connect", autospec=True) as mock_connect:
        fake = MagicMock(spec=RedisCluster)
        mock_connect.return_value = fake
        redis_client.get_redis()
        redis_client.disconnect()

    fake.close.assert_called_once()


def test_disconnect_closes_cached_pubsub() -> None:
    with (
        patch.object(redis_client, "connect", autospec=True) as mock_connect,
        patch.object(redis_client, "connect_pubsub", autospec=True) as mock_pubsub,
    ):
        fake = MagicMock(spec=RedisCluster)
        fake_pubsub = MagicMock(spec=Redis)
        mock_connect.return_value = fake
        mock_pubsub.return_value = fake_pubsub
        redis_client.get_redis()
        redis_client.get_redis_pubsub()
        redis_client.disconnect()

    fake.close.assert_called_once()
    fake_pubsub.close.assert_called_once()


def test_disconnect_skips_unused_pubsub() -> None:
    """Disconnect must not open a pub/sub connection just to close it."""
    with (
        patch.object(redis_client, "connect", autospec=True) as mock_connect,
        patch.object(redis_client, "connect_pubsub", autospec=True) as mock_pubsub,
    ):
        mock_connect.return_value = MagicMock(spec=RedisCluster)
        redis_client.get_redis()
        redis_client.disconnect()

    mock_pubsub.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_async_closes_cached_client() -> None:
    with patch.object(redis_client, "connect_async", autospec=True) as mock_connect:
        fake = MagicMock(spec=AsyncRedisCluster)
        fake.close = AsyncMock()
        mock_connect.return_value = fake
        await redis_client.get_redis_async()
        await redis_client.disconnect_async()

    fake.close.assert_awaited_once()
    assert redis_client._async_clients == {}


@pytest.mark.asyncio
async def test_disconnect_async_closes_cached_pubsub_client() -> None:
    with (
        patch.object(redis_client, "connect_async", autospec=True) as mock_connect,
        patch.object(
            redis_client, "connect_pubsub_async", autospec=True
        ) as mock_pubsub,
    ):
        fake_cluster = MagicMock(spec=AsyncRedisCluster)
        fake_cluster.close = AsyncMock()
        fake_pubsub = MagicMock(spec=AsyncRedis)
        fake_pubsub.close = AsyncMock()
        mock_connect.return_value = fake_cluster
        mock_pubsub.return_value = fake_pubsub
        await redis_client.get_redis_async()
        await redis_client.get_redis_pubsub_async()
        await redis_client.disconnect_async()

    fake_cluster.close.assert_awaited_once()
    fake_pubsub.close.assert_awaited_once()
    assert redis_client._async_pubsub_clients == {}


@pytest.mark.asyncio
async def test_disconnect_async_no_cached_client_is_noop() -> None:
    with patch.object(redis_client, "connect_async", autospec=True) as mock_connect:
        await redis_client.disconnect_async()
    mock_connect.assert_not_called()


@pytest.mark.asyncio
async def test_disconnect_async_closes_pubsub_without_cluster_client() -> None:
    """A pub/sub-only caller must still have its FD released on shutdown."""
    with patch.object(
        redis_client, "connect_pubsub_async", autospec=True
    ) as mock_pubsub:
        fake_pubsub = MagicMock(spec=AsyncRedis)
        fake_pubsub.close = AsyncMock()
        mock_pubsub.return_value = fake_pubsub
        await redis_client.get_redis_pubsub_async()
        await redis_client.disconnect_async()

    fake_pubsub.close.assert_awaited_once()
    assert redis_client._async_pubsub_clients == {}


# Sharded pub/sub end-to-end against the local 3-shard compose cluster.
# Skipped when no cluster is reachable so CI without docker doesn't flap.


def _has_live_cluster() -> bool:
    try:
        c = redis_client.connect()
    except Exception:  # noqa: BLE001 — any connect failure → skip the test
        return False
    try:
        c.close()
    except Exception:
        pass
    return True


@pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable; skip sharded pub/sub integration",
)
def test_sharded_pubsub_end_to_end_sync() -> None:
    """SPUBLISH → SSUBSCRIBE round-trip via the sync cluster client.

    Uses per-node ``get_message`` because redis-py 6.x's
    ``ClusterPubSub.get_sharded_message(ignore_subscribe_messages=True)``
    drops every message, not just the subscribe confirmation.
    """
    redis_client.get_redis.cache_clear()
    cluster = redis_client.get_redis()
    channel = "pr12900:sharded-pubsub:integration"
    ps = cluster.pubsub()
    try:
        ps.ssubscribe(channel)
        assert cluster.spublish(channel, "hello") >= 1

        # Exactly one node is subscribed (the keyslot owner); read from it.
        assert len(ps.node_pubsub_mapping) == 1
        node_ps = next(iter(ps.node_pubsub_mapping.values()))
        # First message is the ssubscribe confirmation, second is our payload.
        confirm = node_ps.get_message(timeout=2.0)
        assert confirm is not None and confirm["type"] == "ssubscribe"
        received = node_ps.get_message(timeout=5.0)
        assert received is not None and received["type"] == "smessage"
        assert received["data"] == "hello"
    finally:
        try:
            ps.sunsubscribe(channel)
        except Exception:
            pass
        ps.close()
        redis_client.disconnect()


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable; skip sharded pub/sub integration",
)
async def test_sharded_spublish_end_to_end_async() -> None:
    """Async cluster client routes SPUBLISH via ``execute_command``
    because redis-py 6.x has no async ``spublish()`` wrapper."""
    redis_client._async_clients.clear()
    cluster = await redis_client.get_redis_async()
    try:
        res = await cluster.execute_command(
            "SPUBLISH", "pr12900:sharded-pubsub:async", "ping"
        )
        # No subscribers — delivered count is 0, but the command must succeed
        # (i.e. not raise MOVED/ASK or routing errors).
        assert isinstance(res, int)
    finally:
        await redis_client.disconnect_async()
