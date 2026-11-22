"""Unit tests for the cluster-only Redis client in ``redis_client``.

Patches the redis-py constructors + ``ping()`` so no real Redis is needed.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.asyncio.retry import Retry as AsyncRetry
from redis.cluster import RedisCluster
from redis.exceptions import ClusterDownError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.retry import Retry

import backend.data.redis_client as redis_client
from backend.util.testing import is_tcp_port_reachable


@pytest.fixture(autouse=True)
def _reset_module_caches() -> None:
    """Flush cached singletons between tests so each test sees a fresh connect."""
    redis_client.get_redis.cache_clear()
    redis_client._async_clients.clear()
    redis_client._async_client_finalizers.clear()
    redis_client._async_client_locks.clear()


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


def test_connect_configures_retry_on_transient_errors() -> None:
    """Sync RedisCluster must carry a Retry that fires on transient errors,
    so a shard rotation blip retries internally instead of surfacing as 500."""
    with patch.object(redis_client, "RedisCluster", autospec=True) as mock_cluster:
        mock_cluster.return_value = MagicMock(spec=RedisCluster)
        redis_client.connect()

    kwargs = mock_cluster.call_args.kwargs
    retry = kwargs["retry"]
    assert isinstance(retry, Retry)
    assert retry.get_retries() == redis_client.REDIS_RETRY_ATTEMPTS
    supported = set(retry._supported_errors)
    assert RedisConnectionError in supported
    assert RedisTimeoutError in supported
    assert ClusterDownError in supported


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
async def test_connect_async_configures_retry() -> None:
    """Async RedisCluster must carry an ``AsyncRetry`` keyed on the same
    transient errors as the sync path. ``retry_on_error`` is intentionally
    omitted — redis-py 6.x ignores it for cluster operations (the cluster
    retry path uses a hardcoded {Timeout, Connection, ClusterDown} set)."""
    with patch.object(redis_client, "AsyncRedisCluster", autospec=True) as mock_cluster:
        fake = MagicMock(spec=AsyncRedisCluster)
        fake.ping = AsyncMock()
        mock_cluster.return_value = fake
        await redis_client.connect_async()

    kwargs = mock_cluster.call_args.kwargs
    retry = kwargs["retry"]
    # redis-py's async cluster uses a separate AsyncRetry class — passing the
    # sync `Retry` would type-fail at construction time.
    assert isinstance(retry, AsyncRetry)
    assert retry.get_retries() == redis_client.REDIS_RETRY_ATTEMPTS
    supported = set(retry._supported_errors)
    assert RedisConnectionError in supported
    assert RedisTimeoutError in supported
    assert ClusterDownError in supported
    # No `retry_on_error` kwarg — it's ineffective on AsyncRedisCluster.
    assert "retry_on_error" not in kwargs


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


@pytest.mark.asyncio
async def test_connect_async_disables_moved_reinitialize() -> None:
    """redis-py 8.1's default reinitialize_steps=5 closes every node's
    connections on every fifth MOVED — including sockets mid-command on this
    shared client. Slot patching alone keeps the map right, so the client
    must be built with reinitialize_steps=0."""
    with patch.object(redis_client, "AsyncRedisCluster", autospec=True) as mock_cluster:
        fake = MagicMock(spec=AsyncRedisCluster)
        fake.initialize = AsyncMock()
        fake.ping = AsyncMock()
        mock_cluster.return_value = fake
        await redis_client.connect_async()

    assert mock_cluster.call_args.kwargs["reinitialize_steps"] == 0


def test_get_redis_caches_connect() -> None:
    with patch.object(redis_client, "connect", autospec=True) as mock_connect:
        mock_connect.return_value = MagicMock(spec=RedisCluster)
        client_a = redis_client.get_redis()
        client_b = redis_client.get_redis()

    assert client_a is client_b
    mock_connect.assert_called_once()


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
async def test_get_redis_async_serializes_concurrent_connects() -> None:
    connect_started = asyncio.Event()
    allow_connect = asyncio.Event()
    fake = MagicMock(spec=AsyncRedisCluster)
    fake.aclose = AsyncMock()
    fake.nodes_manager.aclose = AsyncMock()

    async def delayed_connect() -> AsyncRedisCluster:
        connect_started.set()
        await allow_connect.wait()
        return fake

    with patch.object(
        redis_client,
        "connect_async",
        new=AsyncMock(side_effect=delayed_connect),
    ) as mock_connect:
        first = asyncio.create_task(redis_client.get_redis_async())
        second = asyncio.create_task(redis_client.get_redis_async())
        await connect_started.wait()
        await asyncio.sleep(0)

        assert mock_connect.await_count == 1

        allow_connect.set()
        first_client, second_client = await asyncio.gather(first, second)
        await redis_client.disconnect_async()

    assert first_client is fake
    assert second_client is fake
    mock_connect.assert_awaited_once()
    fake.aclose.assert_awaited_once()
    fake.nodes_manager.aclose.assert_any_await()
    fake.nodes_manager.aclose.assert_any_await("startup_nodes")


def test_get_redis_async_closes_client_when_owning_loop_shuts_down() -> None:
    fake = MagicMock(spec=AsyncRedisCluster)
    fake.aclose = AsyncMock()
    fake.nodes_manager.aclose = AsyncMock()

    with patch.object(redis_client, "connect_async", new=AsyncMock(return_value=fake)):
        loop = asyncio.new_event_loop()
        try:
            client = loop.run_until_complete(redis_client.get_redis_async())
            assert client is fake
            assert redis_client._async_clients[loop] is fake
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    fake.aclose.assert_awaited_once()
    fake.nodes_manager.aclose.assert_any_await()
    fake.nodes_manager.aclose.assert_any_await("startup_nodes")
    assert not redis_client._async_clients
    assert not redis_client._async_client_finalizers


def test_disconnect_closes_cached_client() -> None:
    with patch.object(redis_client, "connect", autospec=True) as mock_connect:
        fake = MagicMock(spec=RedisCluster)
        mock_connect.return_value = fake
        redis_client.get_redis()
        redis_client.disconnect()

    fake.close.assert_called_once()


@pytest.mark.asyncio
async def test_disconnect_async_closes_cached_client() -> None:
    with patch.object(redis_client, "connect_async", autospec=True) as mock_connect:
        fake = MagicMock(spec=AsyncRedisCluster)
        fake.aclose = AsyncMock()
        fake.nodes_manager.aclose = AsyncMock()
        mock_connect.return_value = fake
        await redis_client.get_redis_async()
        await redis_client.disconnect_async()

    fake.aclose.assert_awaited_once()
    fake.nodes_manager.aclose.assert_any_await()
    fake.nodes_manager.aclose.assert_any_await("startup_nodes")
    assert not redis_client._async_clients
    assert not redis_client._async_client_finalizers


@pytest.mark.asyncio
async def test_disconnect_async_no_cached_client_is_noop() -> None:
    with patch.object(redis_client, "connect_async", autospec=True) as mock_connect:
        await redis_client.disconnect_async()
    mock_connect.assert_not_called()


# Sharded pub/sub end-to-end against the local 3-shard compose cluster.
# Skipped when no cluster is reachable so CI without docker doesn't flap.


def _has_live_cluster() -> bool:
    return is_tcp_port_reachable(redis_client.HOST, redis_client.PORT)


@pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable; skip sharded pub/sub integration",
)
def test_sharded_pubsub_end_to_end_sync() -> None:
    """SPUBLISH → SSUBSCRIBE round-trip via the sync cluster client. Uses
    per-node `get_message` because redis-py 6.x's
    `ClusterPubSub.get_sharded_message(ignore_subscribe_messages=True)`
    drops every message, not just the subscribe confirmation."""
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


# ---------- Sharded pub/sub: unit tests with mocks ----------


def test_connect_sharded_pubsub_pins_host_and_disables_socket_timeout() -> None:
    """`socket_timeout=None` on the pubsub socket: a spurious read timeout
    forces a reconnect whose PING races with subscribe-mode."""
    with (
        patch.object(
            redis_client,
            "resolve_shard_for_channel",
            return_value=("shard-host", 7001),
        ),
        patch.object(redis_client, "Redis", autospec=True) as mock_redis,
    ):
        fake_client = MagicMock()
        mock_redis.return_value = fake_client
        client = redis_client.connect_sharded_pubsub("chan")

    mock_redis.assert_called_once()
    kwargs = mock_redis.call_args.kwargs
    # Pinned to the shard's remapped address.
    assert kwargs["host"] == "shard-host"
    assert kwargs["port"] == 7001
    # socket_timeout MUST be None for pubsub — see docstring in redis_client.py.
    assert kwargs["socket_timeout"] is None
    # Idle keepalive + health-check still intact.
    assert kwargs["socket_keepalive"] is True
    assert kwargs["health_check_interval"] == redis_client.HEALTH_CHECK_INTERVAL
    # connect() must PING before returning.
    client.ping.assert_called_once()


@pytest.mark.asyncio
async def test_connect_sharded_pubsub_async_disables_socket_timeout() -> None:
    """Async sibling of ``test_connect_sharded_pubsub_pins_host...``. Same
    invariant: socket_timeout=None."""
    with (
        patch.object(
            redis_client,
            "resolve_shard_for_channel_async",
            new_callable=AsyncMock,
            return_value=("shard-host", 7001),
        ),
        patch.object(redis_client, "AsyncRedis", autospec=True) as mock_redis,
    ):
        fake_client = MagicMock()
        fake_client.ping = AsyncMock()
        mock_redis.return_value = fake_client
        client = await redis_client.connect_sharded_pubsub_async("chan")

    kwargs = mock_redis.call_args.kwargs
    assert kwargs["host"] == "shard-host"
    assert kwargs["port"] == 7001
    assert kwargs["socket_timeout"] is None
    assert kwargs["socket_keepalive"] is True
    assert kwargs["health_check_interval"] == redis_client.HEALTH_CHECK_INTERVAL
    client.ping.assert_awaited_once()


def test_resolve_shard_for_channel_applies_address_remap() -> None:
    """The resolver must run ``_address_remap`` on the announced address so
    callers connect through the same address the cluster client uses."""
    cluster = MagicMock()
    node = MagicMock()
    node.host = "announced-host"
    node.port = 17001
    cluster.get_node_from_key.return_value = node

    with (
        patch.object(redis_client, "get_redis", return_value=cluster),
        patch.object(redis_client, "USE_ANNOUNCED_ADDRESS", False),
    ):
        host, port = redis_client.resolve_shard_for_channel("chan")

    # Remap pins the host to the seed, keeps the announced port.
    assert host == redis_client.HOST
    assert port == 17001


def test_resolve_shard_for_channel_raises_when_no_node_owns_keyslot() -> None:
    """Missing cluster node → explicit RuntimeError, not a silent None deref."""
    cluster = MagicMock()
    cluster.get_node_from_key.return_value = None

    with patch.object(redis_client, "get_redis", return_value=cluster):
        with pytest.raises(RuntimeError, match="No cluster node"):
            redis_client.resolve_shard_for_channel("chan")


def test_resolve_shard_for_channel_passthrough_with_announced_flag() -> None:
    """When REDIS_USE_ANNOUNCED_ADDRESS is on, resolver returns the announced
    address verbatim — no HOST override."""
    cluster = MagicMock()
    node = MagicMock()
    node.host = "redis-2"
    node.port = 17002
    cluster.get_node_from_key.return_value = node

    with (
        patch.object(redis_client, "get_redis", return_value=cluster),
        patch.object(redis_client, "USE_ANNOUNCED_ADDRESS", True),
    ):
        host, port = redis_client.resolve_shard_for_channel("chan")

    assert (host, port) == ("redis-2", 17002)


@pytest.mark.asyncio
async def test_resolve_shard_for_channel_async_applies_address_remap() -> None:
    """Async sibling of the remap test: the async resolver runs
    ``_address_remap`` on the announced address too."""
    cluster = MagicMock()
    cluster.initialize = AsyncMock()
    node = MagicMock()
    node.host = "announced-host"
    node.port = 17001
    cluster.get_node_from_key.return_value = node

    with (
        patch.object(redis_client, "get_redis_async", AsyncMock(return_value=cluster)),
        patch.object(redis_client, "USE_ANNOUNCED_ADDRESS", False),
    ):
        host, port = await redis_client.resolve_shard_for_channel_async("chan")

    assert host == redis_client.HOST
    assert port == 17001


@pytest.mark.asyncio
async def test_resolve_shard_for_channel_async_initializes_before_reading_map() -> None:
    """redis-py 8.1 builds the slot map lazily: the resolver must initialize
    the client before reading it, so a fresh client's empty slots_cache is
    never consulted."""
    cluster = MagicMock()
    cluster.initialize = AsyncMock()
    node = MagicMock()
    node.host = "redis-0"
    node.port = 17000
    cluster.get_node_from_key.return_value = node

    with patch.object(redis_client, "get_redis_async", AsyncMock(return_value=cluster)):
        await redis_client.resolve_shard_for_channel_async("chan")

    cluster.initialize.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_shard_for_channel_async_raises_when_no_node_owns_keyslot() -> (
    None
):
    """Missing cluster node → explicit RuntimeError, same as the sync resolver."""
    cluster = MagicMock()
    cluster.initialize = AsyncMock()
    cluster.get_node_from_key.return_value = None

    with patch.object(redis_client, "get_redis_async", AsyncMock(return_value=cluster)):
        with pytest.raises(RuntimeError, match="No cluster node"):
            await redis_client.resolve_shard_for_channel_async("chan")


@pytest.mark.asyncio
async def test_resolve_shard_for_channel_async_never_touches_the_sync_client() -> None:
    """Regression guard: the sync client's cold connect blocks the event loop,
    so the async resolve must not go near it."""
    cluster = MagicMock()
    cluster.initialize = AsyncMock()
    node = MagicMock()
    node.host = "redis-0"
    node.port = 17000
    cluster.get_node_from_key.return_value = node

    def sync_client_called():
        raise AssertionError("get_redis() must not run on the event loop")

    with (
        patch.object(redis_client, "get_redis", side_effect=sync_client_called),
        patch.object(redis_client, "get_redis_async", AsyncMock(return_value=cluster)),
        patch.object(redis_client, "USE_ANNOUNCED_ADDRESS", True),
    ):
        host, port = await redis_client.resolve_shard_for_channel_async("chan")

    assert (host, port) == ("redis-0", 17000)


@pytest.mark.asyncio
async def test_resolve_shard_for_channel_async_is_bounded() -> None:
    """Redis down and no cached client: the lookup gives up within its bound
    instead of joining the client's minutes-long connect retries."""

    async def hanging_client():
        await asyncio.Future()

    with (
        patch.object(redis_client, "get_redis_async", hanging_client),
        patch.object(redis_client, "RESOLVE_SHARD_TIMEOUT_S", 0.05),
    ):
        with pytest.raises(TimeoutError):
            await redis_client.resolve_shard_for_channel_async("chan")


@pytest.mark.asyncio
async def test_connect_async_closes_the_client_when_its_ping_is_cancelled() -> None:
    """A bounded caller's timeout cancels the PING; the half-built client must
    be closed, not left to leak its sockets."""
    client = MagicMock()
    client.initialize = AsyncMock()
    client.ping = AsyncMock(side_effect=asyncio.CancelledError())
    client.aclose = AsyncMock()
    client.nodes_manager.aclose = AsyncMock()

    with patch.object(redis_client, "AsyncRedisCluster", return_value=client):
        with pytest.raises(asyncio.CancelledError):
            await redis_client.connect_async()

    client.aclose.assert_awaited_once()
    client.nodes_manager.aclose.assert_any_await()
    client.nodes_manager.aclose.assert_any_await("startup_nodes")


@pytest.mark.asyncio
async def test_force_close_async_cluster_releases_node_pools() -> None:
    """redis-py 8.1's ``aclose()`` no-ops once a connection error flips the
    client back to uninitialized, so the node pools must be closed directly
    as well — otherwise a discarded client keeps its sockets."""
    client = MagicMock()
    client.aclose = AsyncMock()
    client.nodes_manager.aclose = AsyncMock()

    await redis_client._force_close_async_cluster(client)

    client.aclose.assert_awaited_once()
    assert client.nodes_manager.aclose.await_count == 2
    client.nodes_manager.aclose.assert_any_await()
    client.nodes_manager.aclose.assert_any_await("startup_nodes")


@pytest.mark.asyncio
async def test_get_redis_async_closes_the_client_when_caching_is_cancelled() -> None:
    """Cancelled between connect and cache: the connected client is closed
    rather than leaked."""
    client = MagicMock()
    client.aclose = AsyncMock()
    client.nodes_manager.aclose = AsyncMock()

    async def cancelled_finalizer():
        raise asyncio.CancelledError()
        yield  # pragma: no cover - makes this an async generator

    redis_client._async_clients.clear()
    try:
        with (
            patch.object(redis_client, "connect_async", AsyncMock(return_value=client)),
            patch.object(
                redis_client,
                "_close_async_client_on_loop_shutdown",
                lambda c: cancelled_finalizer(),
            ),
        ):
            with pytest.raises(asyncio.CancelledError):
                await redis_client.get_redis_async()

        client.aclose.assert_awaited_once()
        client.nodes_manager.aclose.assert_any_await("startup_nodes")
        assert list(redis_client._async_clients) == []
    finally:
        redis_client._async_clients.clear()
        redis_client._async_client_finalizers.clear()


@pytest.mark.asyncio
async def test_follow_moved_slot_async_updates_the_async_clients_map() -> None:
    """The async resolver reads the async client's map, so a MOVED reply must
    re-point that map, not the sync client's."""
    client = MagicMock()
    client.nodes_manager.move_slot = AsyncMock()
    client.nodes_manager.slots_cache = {}
    error = MagicMock()

    with patch.object(redis_client, "get_redis_async", AsyncMock(return_value=client)):
        await redis_client.follow_moved_slot_async(error)

    client.nodes_manager.move_slot.assert_awaited_once_with(error)


def _moved_cluster_setup(existing_node: bool):
    """Build a nodes_manager mock shaped like redis-py 8.1 after move_slot:
    the slot's new owner carries the redirect's announced (unremapped) host."""
    moved = MagicMock()
    moved.host, moved.port, moved.name = "redis-1", 7001, "redis-1:7001"
    nm = MagicMock()
    nm.move_slot = AsyncMock()
    nm.slots_cache = {31: [moved]}
    nm.nodes_cache = {"redis-1:7001": moved}
    nm.default_node = None
    nm.remap_host_port = lambda h, p: redis_client._address_remap((h, p))
    discovered = None
    if existing_node:
        discovered = MagicMock()
        discovered.host, discovered.port, discovered.name = (
            "127.0.0.1",
            7001,
            "127.0.0.1:7001",
        )
        nm.nodes_cache["127.0.0.1:7001"] = discovered
    client = MagicMock()
    client.nodes_manager = nm
    error = MagicMock()
    error.slot_id = 31
    return client, nm, moved, discovered, error


@pytest.mark.asyncio
async def test_follow_moved_slot_async_remaps_announced_redirect_host() -> None:
    """redis-py 8.1's move_slot stores the redirect's announced host without
    applying address_remap. When CLUSTER SLOTS already discovered the same
    shard under its remapped address, the slot must point at the discovered
    node — not at an unreachable announced duplicate."""
    client, nm, moved, discovered, error = _moved_cluster_setup(existing_node=True)

    with (
        patch.object(redis_client, "get_redis_async", AsyncMock(return_value=client)),
        patch.object(redis_client, "USE_ANNOUNCED_ADDRESS", False),
        patch.object(redis_client, "HOST", "localhost"),
    ):
        await redis_client.follow_moved_slot_async(error)

    assert nm.slots_cache[31][0] is discovered
    assert "redis-1:7001" not in nm.nodes_cache
    assert nm.nodes_cache["127.0.0.1:7001"] is discovered


@pytest.mark.asyncio
async def test_follow_moved_slot_async_rewrites_node_when_no_remap_peer() -> None:
    """When the remapped address isn't in the nodes cache yet, the moved node
    itself is rewritten in place (host, port, name, and cache key)."""
    client, nm, moved, _, error = _moved_cluster_setup(existing_node=False)

    with (
        patch.object(redis_client, "get_redis_async", AsyncMock(return_value=client)),
        patch.object(redis_client, "USE_ANNOUNCED_ADDRESS", False),
        patch.object(redis_client, "HOST", "localhost"),
    ):
        await redis_client.follow_moved_slot_async(error)

    # "localhost" normalizes to 127.0.0.1, matching how ClusterNode keys
    # the CLUSTER SLOTS-discovered nodes.
    assert moved.host == "127.0.0.1"
    assert moved.port == 7001
    assert moved.name == "127.0.0.1:7001"
    assert "redis-1:7001" not in nm.nodes_cache
    assert nm.nodes_cache["127.0.0.1:7001"] is moved


@pytest.mark.asyncio
async def test_follow_moved_slot_async_leaves_node_alone_when_remap_passthrough() -> (
    None
):
    """With REDIS_USE_ANNOUNCED_ADDRESS on, the announced address is used
    verbatim — no rewrite."""
    client, nm, moved, _, error = _moved_cluster_setup(existing_node=False)

    with (
        patch.object(redis_client, "get_redis_async", AsyncMock(return_value=client)),
        patch.object(redis_client, "USE_ANNOUNCED_ADDRESS", True),
    ):
        await redis_client.follow_moved_slot_async(error)

    assert moved.name == "redis-1:7001"
    assert nm.nodes_cache["redis-1:7001"] is moved


@pytest.mark.asyncio
async def test_refresh_slot_map_async_is_throttled_per_client() -> None:
    first, second = MagicMock(), MagicMock()
    first.nodes_manager.initialize = AsyncMock()
    second.nodes_manager.initialize = AsyncMock()

    now = 100.0
    with (
        patch.object(
            redis_client,
            "get_redis_async",
            AsyncMock(side_effect=[first, first, second, second, first]),
        ),
        patch.object(redis_client, "_async_slot_map_clock", lambda: now),
    ):
        await redis_client.refresh_slot_map_async()
        await redis_client.refresh_slot_map_async()
        await redis_client.refresh_slot_map_async()
        await redis_client.refresh_slot_map_async()

        first.nodes_manager.initialize.assert_awaited_once()
        second.nodes_manager.initialize.assert_awaited_once()

        now += redis_client._SLOT_MAP_REFRESH_EVERY_S
        await redis_client.refresh_slot_map_async()

    assert first.nodes_manager.initialize.await_count == 2
    second.nodes_manager.initialize.assert_awaited_once()


def test_health_check_interval_is_30s_default() -> None:
    """Idle PING interval must be <=30s so half-open pubsub sockets don't
    wait for the OS TCP keepalive (~2h)."""
    assert redis_client.HEALTH_CHECK_INTERVAL <= 30


def test_connect_sets_health_check_interval() -> None:
    """The cluster client must propagate health_check_interval to each node
    pool — otherwise idle cluster sockets go stale."""
    with patch.object(redis_client, "RedisCluster", autospec=True) as mock_cluster:
        mock_cluster.return_value = MagicMock(spec=RedisCluster)
        redis_client.connect()
    kwargs = mock_cluster.call_args.kwargs
    assert kwargs["health_check_interval"] == redis_client.HEALTH_CHECK_INTERVAL
    assert kwargs["health_check_interval"] > 0


# ---------- K8s same-port shard collapse regression (AUTOGPT-SERVER-8SX) ----------


def test_k8s_shard_collapse_with_announced_address_off_routes_all_to_seed() -> None:
    """In K8s every shard serves on port 6379 behind the seed service, so the
    default `_address_remap` collapses all shards to `(HOST, 6379)` — the
    AUTOGPT-SERVER-8SX bug. Fix: `REDIS_USE_ANNOUNCED_ADDRESS=true`."""
    cluster = MagicMock()
    # 3 shards, each owning a distinct hash slot, but every pod serves on
    # 6379 in K8s — exactly the production topology.
    nodes_by_channel = {
        "{ch-a}/x": MagicMock(host="redis-cluster-redis-0", port=6379),
        "{ch-b}/y": MagicMock(host="redis-cluster-redis-1", port=6379),
        "{ch-c}/z": MagicMock(host="redis-cluster-redis-2", port=6379),
    }
    cluster.get_node_from_key.side_effect = lambda c: nodes_by_channel[c]

    with (
        patch.object(redis_client, "get_redis", return_value=cluster),
        patch.object(redis_client, "USE_ANNOUNCED_ADDRESS", False),
        patch.object(redis_client, "HOST", "redis-dev-seed"),
    ):
        endpoints = {
            channel: redis_client.resolve_shard_for_channel(channel)
            for channel in nodes_by_channel
        }

    # The bug: every shard resolves to the same seed:port endpoint.
    assert len(set(endpoints.values())) == 1, (
        f"Expected the K8s shard-collapse bug, got {endpoints!r}. "
        "If this test is failing it means _address_remap behaviour changed "
        "and the AUTOGPT-SERVER-8SX regression note in this file needs review."
    )
    assert all(ep == ("redis-dev-seed", 6379) for ep in endpoints.values())


def test_k8s_shard_collapse_fixed_with_announced_address_on() -> None:
    """With `REDIS_USE_ANNOUNCED_ADDRESS=true`, each shard's announced FQDN
    passes through, so distinct slots resolve to distinct endpoints."""
    cluster = MagicMock()
    nodes_by_channel = {
        "{ch-a}/x": MagicMock(host="redis-cluster-redis-0", port=6379),
        "{ch-b}/y": MagicMock(host="redis-cluster-redis-1", port=6379),
        "{ch-c}/z": MagicMock(host="redis-cluster-redis-2", port=6379),
    }
    cluster.get_node_from_key.side_effect = lambda c: nodes_by_channel[c]

    with (
        patch.object(redis_client, "get_redis", return_value=cluster),
        patch.object(redis_client, "USE_ANNOUNCED_ADDRESS", True),
        patch.object(redis_client, "HOST", "redis-dev-seed"),
    ):
        endpoints = {
            channel: redis_client.resolve_shard_for_channel(channel)
            for channel in nodes_by_channel
        }

    # Each shard maps to a distinct endpoint — sharded pubsub can route
    # SSUBSCRIBE to the slot owner.
    assert len(set(endpoints.values())) == 3
    assert endpoints["{ch-a}/x"] == ("redis-cluster-redis-0", 6379)
    assert endpoints["{ch-b}/y"] == ("redis-cluster-redis-1", 6379)
    assert endpoints["{ch-c}/z"] == ("redis-cluster-redis-2", 6379)


def test_local_compose_remap_keeps_distinct_ports_per_shard() -> None:
    """Local docker-compose announces distinct ports per shard, so the
    `(host, port)` tuple stays distinct even with `HOST` pinned to seed."""
    cluster = MagicMock()
    nodes_by_channel = {
        "{ch-a}/x": MagicMock(host="redis-0", port=17000),
        "{ch-b}/y": MagicMock(host="redis-1", port=17001),
        "{ch-c}/z": MagicMock(host="redis-2", port=17002),
    }
    cluster.get_node_from_key.side_effect = lambda c: nodes_by_channel[c]

    with (
        patch.object(redis_client, "get_redis", return_value=cluster),
        patch.object(redis_client, "USE_ANNOUNCED_ADDRESS", False),
        patch.object(redis_client, "HOST", "localhost"),
    ):
        endpoints = {
            channel: redis_client.resolve_shard_for_channel(channel)
            for channel in nodes_by_channel
        }

    # Distinct ports → distinct endpoints even after remap pins the host.
    assert len(set(endpoints.values())) == 3
    assert endpoints["{ch-a}/x"] == ("localhost", 17000)
    assert endpoints["{ch-b}/y"] == ("localhost", 17001)
    assert endpoints["{ch-c}/z"] == ("localhost", 17002)


# ---------- Sharded pub/sub: multi-shard integration on the live cluster ----------


def _channel_owner(channel: str) -> tuple[str, int]:
    """Resolve the slot owner for ``channel`` via the live cluster client."""
    cluster = redis_client.get_redis()
    node = cluster.get_node_from_key(channel)
    assert node is not None, f"no slot owner for {channel!r}"
    return node.host, node.port


def _channels_on_distinct_shards(n: int = 3) -> list[str]:
    """Build N hash-tagged channels each mapping to a distinct shard."""
    seen: dict[tuple[str, int], str] = {}
    for tag_id in range(2000):
        chan = "{u" + str(tag_id) + "/g}/exec/e"
        owner = _channel_owner(chan)
        seen.setdefault(owner, chan)
        if len(seen) >= n:
            break
    assert len(seen) >= n, f"could only cover {len(seen)} shards"
    return list(seen.values())[:n]


@pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable; skip multi-shard integration",
)
def test_resolve_shard_for_channel_lands_on_distinct_shards() -> None:
    """3 hash-tagged channels resolve to 3 different shards (slot-distribution)."""
    redis_client.get_redis.cache_clear()
    try:
        channels = _channels_on_distinct_shards(3)
        endpoints = {ch: redis_client.resolve_shard_for_channel(ch) for ch in channels}
        # Three channels → three distinct (host, port) endpoints.
        assert len(set(endpoints.values())) == 3
    finally:
        redis_client.disconnect()


@pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable; skip multi-shard integration",
)
def test_sharded_pubsub_concurrent_subscribers_on_three_shards() -> None:
    """SSUBSCRIBE on three channels owned by three different shards, then
    SPUBLISH to each — every payload must land on its subscriber."""
    redis_client.get_redis.cache_clear()
    cluster = redis_client.get_redis()
    try:
        channels = _channels_on_distinct_shards(3)
        # Subscribe via the cluster client so redis-py's per-node pubsub
        # mapping handles the sharded routing for us.
        ps = cluster.pubsub()
        try:
            for ch in channels:
                ps.ssubscribe(ch)
            # The cluster client opens one node-pubsub per shard owner — three
            # channels on three shards must produce three distinct node clients.
            assert len(ps.node_pubsub_mapping) == 3, (
                "Expected SSUBSCRIBE on 3 channels owned by 3 distinct shards "
                f"to open 3 node-pubsubs, got {len(ps.node_pubsub_mapping)}"
            )
            # Publish to each channel and verify each reaches the right node.
            for i, ch in enumerate(channels):
                assert cluster.spublish(ch, f"payload-{i}") >= 1
            # Drain ssubscribe confirmations + smessages from every node.
            received: dict[str, str] = {}
            for node_ps in ps.node_pubsub_mapping.values():
                # First message per node is the ssubscribe confirm; subsequent
                # smessages carry the test payloads.
                for _ in range(4):  # confirm + at most 1 payload per shard
                    msg = node_ps.get_message(timeout=2.0)
                    if msg is None:
                        break
                    if msg["type"] == "smessage":
                        received[msg["channel"]] = msg["data"]
            for i, ch in enumerate(channels):
                assert ch in received, f"channel {ch!r} got no message"
                assert received[ch] == f"payload-{i}"
        finally:
            for ch in channels:
                try:
                    ps.sunsubscribe(ch)
                except Exception:
                    pass
            ps.close()
    finally:
        redis_client.disconnect()


@pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable; skip multi-shard integration",
)
def test_sharded_pubsub_idle_subscriber_survives_health_check_window() -> None:
    """An SSUBSCRIBE connection must survive an idle window longer than
    `HEALTH_CHECK_INTERVAL` — uses `+5s` to provoke at least one health check."""
    import time as _time

    redis_client.get_redis.cache_clear()
    cluster = redis_client.get_redis()
    channel = "{idle-test}/exec/e"
    client = redis_client.connect_sharded_pubsub(channel)
    ps = client.pubsub()
    try:
        ps.ssubscribe(channel)
        confirm = ps.get_message(timeout=2.0)
        assert confirm is not None and confirm["type"] == "ssubscribe"

        # Idle window — must exceed health_check_interval at least once.
        idle_seconds = redis_client.HEALTH_CHECK_INTERVAL + 5
        _time.sleep(idle_seconds)

        # After idling, publish + receive should still work.
        assert cluster.spublish(channel, "post-idle") >= 1
        msg = ps.get_message(timeout=5.0)
        assert msg is not None and msg["type"] == "smessage"
        assert msg["data"] == "post-idle"
    finally:
        try:
            ps.sunsubscribe(channel)
        except Exception:
            pass
        ps.close()
        client.close()
        redis_client.disconnect()


@pytest.mark.skipif(
    not _has_live_cluster(),
    reason="local redis cluster not reachable; skip multi-shard integration",
)
def test_sharded_pubsub_reconnect_after_forced_disconnect() -> None:
    """Subscriber reconnect after a forced disconnect — close socket, open
    a fresh one, and verify new SPUBLISH events still arrive."""
    redis_client.get_redis.cache_clear()
    cluster = redis_client.get_redis()
    channel = "{reconnect-test}/exec/e"

    # Round 1: subscribe, receive one payload, then close everything.
    client = redis_client.connect_sharded_pubsub(channel)
    ps = client.pubsub()
    try:
        ps.ssubscribe(channel)
        ps.get_message(timeout=2.0)  # ssubscribe confirmation
        assert cluster.spublish(channel, "before-restart") >= 1
        msg = ps.get_message(timeout=5.0)
        assert msg is not None and msg["data"] == "before-restart"
    finally:
        try:
            ps.sunsubscribe(channel)
        except Exception:
            pass
        ps.close()
        client.close()

    # Round 2: a fresh subscriber on the same channel — same routing,
    # different socket. This exercises the reconnect-and-resubscribe path
    # the conn_manager runs after a network blip.
    client2 = redis_client.connect_sharded_pubsub(channel)
    ps2 = client2.pubsub()
    try:
        ps2.ssubscribe(channel)
        ps2.get_message(timeout=2.0)
        assert cluster.spublish(channel, "after-restart") >= 1
        msg = ps2.get_message(timeout=5.0)
        assert msg is not None and msg["data"] == "after-restart"
    finally:
        try:
            ps2.sunsubscribe(channel)
        except Exception:
            pass
        ps2.close()
        client2.close()
        redis_client.disconnect()
