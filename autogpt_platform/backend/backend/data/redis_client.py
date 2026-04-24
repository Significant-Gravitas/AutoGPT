import asyncio
import logging
import os

from dotenv import load_dotenv
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import ClusterNode as AsyncClusterNode
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.cluster import ClusterNode, RedisCluster

from backend.util.cache import cached
from backend.util.retry import conn_retry

load_dotenv()

# Prefer the cluster env vars so the cluster-only image can co-exist with
# old-image pods still reading REDIS_HOST during a rollout.
HOST = os.getenv("REDIS_CLUSTER_HOST") or os.getenv("REDIS_HOST", "localhost")
PORT = int(os.getenv("REDIS_CLUSTER_PORT") or os.getenv("REDIS_PORT", "6379"))
PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Fail-fast on a wedged endpoint instead of blocking on no-response TCP.
SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "30"))
SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
# PING on idle sockets to detect half-open connections without waiting for
# the OS TCP keepalive (~2h default).
HEALTH_CHECK_INTERVAL = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))

# Skip the HOST-pinning remap when each shard's announced hostname resolves
# directly (e.g. compose DNS names redis-0/redis-1/redis-2).
USE_ANNOUNCED_ADDRESS = os.getenv("REDIS_USE_ANNOUNCED_ADDRESS", "").lower() in (
    "1",
    "true",
    "yes",
)

logger = logging.getLogger(__name__)

# Aliases so call-sites don't care which class this is.
RedisClient = RedisCluster
AsyncRedisClient = AsyncRedisCluster


def _address_remap(addr: tuple[str, int]) -> tuple[str, int]:
    """Rewrite the per-shard address returned by ``CLUSTER SLOTS``.

    Default: pin every shard to the seed ``HOST`` and keep its announced
    port. Works for a load-balanced seed DNS (prod) and for laptop compose
    where each shard is reachable at the seed host via a distinct
    published port. When ``REDIS_USE_ANNOUNCED_ADDRESS`` is set, pass the
    announced address through unchanged.
    """
    if USE_ANNOUNCED_ADDRESS:
        return addr
    _, port = addr
    return HOST, port


@conn_retry("Redis", "Acquiring connection")
def connect() -> RedisClient:
    c = RedisCluster(
        startup_nodes=[ClusterNode(HOST, PORT)],
        password=PASSWORD,
        decode_responses=True,
        socket_timeout=SOCKET_TIMEOUT,
        socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
        socket_keepalive=True,
        health_check_interval=HEALTH_CHECK_INTERVAL,
        address_remap=_address_remap,
    )
    c.ping()
    return c


@conn_retry("Redis", "Releasing connection")
def disconnect():
    get_redis().close()
    get_redis.cache_clear()
    # Only close pub/sub if it was already constructed — don't open a fresh
    # connection at shutdown just to tear it down.
    pubsub = _get_cached_pubsub()
    if pubsub is not None:
        try:
            pubsub.close()
        except Exception:
            logger.warning("Failed to close pub/sub Redis connection", exc_info=True)
    get_redis_pubsub.cache_clear()


def _get_cached_pubsub() -> Redis | None:
    """Return the cached pub/sub client if one exists, else ``None``."""
    if get_redis_pubsub.cache_info()["size"] == 0:
        return None
    return get_redis_pubsub()


@cached(ttl_seconds=3600)
def get_redis() -> RedisClient:
    return connect()


@conn_retry("AsyncRedis", "Acquiring connection")
async def connect_async() -> AsyncRedisClient:
    c = AsyncRedisCluster(
        startup_nodes=[AsyncClusterNode(HOST, PORT)],
        password=PASSWORD,
        decode_responses=True,
        socket_timeout=SOCKET_TIMEOUT,
        socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
        socket_keepalive=True,
        health_check_interval=HEALTH_CHECK_INTERVAL,
        address_remap=_address_remap,
    )
    await c.ping()
    return c


# One AsyncRedisCluster per event loop: the client binds to the loop it was
# first awaited on, so a module-level singleton breaks across test loops.
_async_clients: dict[int, AsyncRedisCluster] = {}
_async_pubsub_clients: dict[int, AsyncRedis] = {}


@conn_retry("AsyncRedis", "Releasing connection")
async def disconnect_async():
    loop = asyncio.get_running_loop()
    c = _async_clients.pop(id(loop), None)
    if c is not None:
        await c.close()
    # A pub/sub-only caller may never have acquired the cluster client; close
    # its FD independently rather than early-returning on a missing cluster.
    pubsub = _async_pubsub_clients.pop(id(loop), None)
    if pubsub is not None:
        await pubsub.close()


async def get_redis_async() -> AsyncRedisClient:
    loop = asyncio.get_running_loop()
    client = _async_clients.get(id(loop))
    if client is None:
        client = await connect_async()
        _async_clients[id(loop)] = client
    return client


# Pub/sub uses a plain (Async)Redis client, not the cluster client:
# - listen("*") event buses rely on PSUBSCRIBE which sharded pub/sub does
#   not support.
# - redis-py 6.x AsyncRedisCluster has no pubsub()/spublish() wrappers.
# Fire-and-forget sharded publishes (e.g. pending_messages) still route via
# the cluster client's execute_command("SPUBLISH", ...).


@conn_retry("RedisPubSub", "Acquiring connection")
def connect_pubsub(host: str = HOST, port: int = PORT) -> Redis:
    c = Redis(
        host=host,
        port=port,
        password=PASSWORD,
        decode_responses=True,
        socket_timeout=SOCKET_TIMEOUT,
        socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
        socket_keepalive=True,
        health_check_interval=HEALTH_CHECK_INTERVAL,
    )
    c.ping()
    return c


@conn_retry("AsyncRedisPubSub", "Acquiring connection")
async def connect_pubsub_async(host: str = HOST, port: int = PORT) -> AsyncRedis:
    c = AsyncRedis(
        host=host,
        port=port,
        password=PASSWORD,
        decode_responses=True,
        socket_timeout=SOCKET_TIMEOUT,
        socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
        socket_keepalive=True,
        health_check_interval=HEALTH_CHECK_INTERVAL,
    )
    await c.ping()
    return c


@cached(ttl_seconds=3600)
def get_redis_pubsub() -> Redis:
    """Return a plain ``Redis`` client dedicated to classic (non-sharded) pub/sub.

    A subscribed connection blocks on ``listen()`` and cannot be interleaved
    with regular command traffic.
    """
    return connect_pubsub()


async def get_redis_pubsub_async() -> AsyncRedis:
    """Async equivalent of :func:`get_redis_pubsub`, cached per event loop."""
    loop = asyncio.get_running_loop()
    client = _async_pubsub_clients.get(id(loop))
    if client is None:
        client = await connect_pubsub_async()
        _async_pubsub_clients[id(loop)] = client
    return client


# Sharded pub/sub routes every channel to exactly one shard (the keyslot
# owner). Subscribers must open a connection to that specific shard; a
# connection to any other shard would see no messages for the channel.
# redis-py 6.x async has no cluster pubsub wrapper, so we resolve the
# owning node manually and open a plain AsyncRedis against it.


def resolve_shard_for_channel(channel: str) -> tuple[str, int]:
    """Return the ``(host, port)`` of the shard that owns the channel's keyslot.

    Applies the configured ``_address_remap`` so callers connect through the
    same address the cluster client uses.
    """
    cluster = get_redis()
    node = cluster.get_node_from_key(channel)
    if node is None:
        raise RuntimeError(f"No cluster node owns the keyslot for channel {channel!r}")
    return _address_remap((node.host, node.port))


def connect_sharded_pubsub(channel: str) -> Redis:
    """Open a plain ``Redis`` connection pinned to the channel's owning shard."""
    host, port = resolve_shard_for_channel(channel)
    return connect_pubsub(host=host, port=port)


async def connect_sharded_pubsub_async(channel: str) -> AsyncRedis:
    """Async variant of :func:`connect_sharded_pubsub`."""
    host, port = resolve_shard_for_channel(channel)
    return await connect_pubsub_async(host=host, port=port)
