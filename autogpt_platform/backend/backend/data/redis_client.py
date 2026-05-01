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
    """Pin each shard to the seed `HOST`, keep its announced port.

    Set `REDIS_USE_ANNOUNCED_ADDRESS=true` when the announced shard FQDNs
    resolve directly (e.g. each pod has its own DNS).
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
    # Close on PING failure so retries don't leak ClusterNodes (AUTOGPT-SERVER-8T1).
    try:
        c.ping()
    except Exception:
        try:
            c.close()
        except Exception:
            pass
        raise
    return c


@conn_retry("Redis", "Releasing connection")
def disconnect():
    get_redis().close()
    get_redis.cache_clear()


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
    # Close on PING failure so retries don't leak ClusterNodes (AUTOGPT-SERVER-8V6/8V4/8V3).
    try:
        await c.ping()
    except Exception:
        try:
            await c.close()
        except Exception:
            pass
        raise
    return c


# One AsyncRedisCluster per event loop: the client binds to the loop it was
# first awaited on, so a module-level singleton breaks across test loops.
_async_clients: dict[int, AsyncRedisCluster] = {}


@conn_retry("AsyncRedis", "Releasing connection")
async def disconnect_async():
    loop = asyncio.get_running_loop()
    c = _async_clients.pop(id(loop), None)
    if c is not None:
        await c.close()


async def get_redis_async() -> AsyncRedisClient:
    loop = asyncio.get_running_loop()
    client = _async_clients.get(id(loop))
    if client is None:
        client = await connect_async()
        _async_clients[id(loop)] = client
    return client


# Sharded pub/sub only delivers on the keyslot-owning shard; subscribers
# need a plain (Async)Redis connection pinned to that node.


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


@conn_retry("RedisShardedPubSub", "Acquiring connection")
def connect_sharded_pubsub(channel: str) -> Redis:
    """Open a plain ``Redis`` connection pinned to the channel's owning shard."""
    host, port = resolve_shard_for_channel(channel)
    # socket_timeout=None: pubsub reads block indefinitely; a spurious
    # read timeout forces a reconnect whose PING races with subscribe-mode.
    c = Redis(
        host=host,
        port=port,
        password=PASSWORD,
        decode_responses=True,
        socket_timeout=None,
        socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
        socket_keepalive=True,
        health_check_interval=HEALTH_CHECK_INTERVAL,
    )
    try:
        c.ping()
    except Exception:
        try:
            c.close()
        except Exception:
            pass
        raise
    return c


@conn_retry("AsyncRedisShardedPubSub", "Acquiring connection")
async def connect_sharded_pubsub_async(channel: str) -> AsyncRedis:
    """Async variant of :func:`connect_sharded_pubsub`."""
    host, port = resolve_shard_for_channel(channel)
    # socket_timeout=None: see ``connect_sharded_pubsub``.
    c = AsyncRedis(
        host=host,
        port=port,
        password=PASSWORD,
        decode_responses=True,
        socket_timeout=None,
        socket_connect_timeout=SOCKET_CONNECT_TIMEOUT,
        socket_keepalive=True,
        health_check_interval=HEALTH_CHECK_INTERVAL,
    )
    try:
        await c.ping()
    except Exception:
        try:
            await c.close()
        except Exception:
            pass
        raise
    return c
