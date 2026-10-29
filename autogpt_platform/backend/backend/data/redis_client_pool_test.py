"""Connection-pool sizing for the cluster clients in ``redis_client``.

redis-py 8.x caps each cluster node's pool at 100 connections by default and
raises ``MaxConnectionsError`` the moment it is full. One executor runs every
node of a graph on a single loop, so a wide fan-out needs far more than 100
concurrent commands against the user's shard.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis import Redis
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.backoff import NoBackoff
from redis.cluster import RedisCluster
from redis.exceptions import RedisError
from redis.retry import Retry

import backend.data.redis_client as redis_client
from backend.util.testing import is_tcp_port_reachable

# redis-py 8.x's per-node default; the configured cap must sit well above it.
REDIS_PY_DEFAULT_MAX_CONNECTIONS = 100

# Lock SETs plus event SPUBLISHes in flight at once, all on one hash slot, as
# a wide fan-out produces for a single user's graph execution.
FAN_OUT_COMMANDS = 500


def _has_live_cluster() -> bool:
    """A listening port is not enough: against a standalone Redis,
    ``get_redis_async`` fails cluster discovery and ``conn_retry`` keeps
    retrying it, so probe ``CLUSTER INFO`` once without retries instead."""
    if not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT):
        return False
    probe = Redis(
        host=redis_client.HOST,
        port=redis_client.PORT,
        password=redis_client.PASSWORD,
        socket_timeout=1,
        socket_connect_timeout=1,
        retry=Retry(NoBackoff(), 0),
        decode_responses=True,
    )
    try:
        info = probe.execute_command("CLUSTER INFO")
    except RedisError:
        return False
    finally:
        probe.close()
    return isinstance(info, dict) and info.get("cluster_state") == "ok"


def test_max_connections_default_is_above_redis_py_default() -> None:
    assert redis_client.REDIS_MAX_CONNECTIONS >= FAN_OUT_COMMANDS
    assert redis_client.REDIS_MAX_CONNECTIONS > REDIS_PY_DEFAULT_MAX_CONNECTIONS


def test_connect_sets_max_connections() -> None:
    with patch.object(redis_client, "RedisCluster", autospec=True) as mock_cluster:
        mock_cluster.return_value = MagicMock(spec=RedisCluster)
        redis_client.connect()

    kwargs = mock_cluster.call_args.kwargs
    assert kwargs["max_connections"] == redis_client.REDIS_MAX_CONNECTIONS


def test_connect_once_sets_max_connections() -> None:
    with patch.object(redis_client, "RedisCluster", autospec=True) as mock_cluster:
        mock_cluster.return_value = MagicMock(spec=RedisCluster)
        redis_client.connect_once(timeout=1)

    kwargs = mock_cluster.call_args.kwargs
    assert kwargs["max_connections"] == redis_client.REDIS_MAX_CONNECTIONS


@pytest.mark.asyncio
async def test_connect_async_sets_max_connections() -> None:
    with patch.object(redis_client, "AsyncRedisCluster", autospec=True) as mock_cluster:
        fake = MagicMock(spec=AsyncRedisCluster)
        fake.ping = AsyncMock()
        mock_cluster.return_value = fake
        await redis_client.connect_async()

    kwargs = mock_cluster.call_args.kwargs
    assert kwargs["max_connections"] == redis_client.REDIS_MAX_CONNECTIONS


@pytest.mark.asyncio
@pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable",
)
async def test_wide_fan_out_does_not_exhaust_async_pool() -> None:
    """Hundreds of concurrent lock and publish commands on one shard must not
    raise MaxConnectionsError."""
    await redis_client.disconnect_async()
    client = await redis_client.get_redis_async()
    prefix = "{redis-pool-fan-out-test}"
    half = FAN_OUT_COMMANDS // 2
    try:
        locks = [
            client.set(f"{prefix}:lock:{i}", "1", nx=True, px=10_000)
            for i in range(half)
        ]
        publishes = [
            client.execute_command("SPUBLISH", f"{prefix}:events", f"event-{i}")
            for i in range(half)
        ]
        results = await asyncio.gather(*locks, *publishes, return_exceptions=True)

        errors = [r for r in results if isinstance(r, BaseException)]
        assert errors == [], f"{len(errors)} commands failed, first: {errors[0]!r}"
    finally:
        await client.delete(*[f"{prefix}:lock:{i}" for i in range(half)])
        await redis_client.disconnect_async()
