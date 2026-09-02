import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from weakref import ReferenceType, WeakKeyDictionary, ref

from dotenv import load_dotenv
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.asyncio.cluster import ClusterNode as AsyncClusterNode
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import ExponentialBackoff
from redis.cluster import ClusterNode, RedisCluster
from redis.exceptions import ClusterDownError
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError
from redis.retry import Retry

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

# Retry transient cluster errors internally so a rotation blip never surfaces
# as a graph-exec 500.
REDIS_RETRY_ATTEMPTS = int(os.getenv("REDIS_RETRY_ATTEMPTS", "5"))

# Errors that should trigger a per-command retry inside the redis-py client.
TRANSIENT_REDIS_ERRORS: tuple[type[RedisError], ...] = (
    RedisConnectionError,
    RedisTimeoutError,
    ClusterDownError,
)

logger = logging.getLogger(__name__)

# Aliases so call-sites don't care which class this is.
RedisClient = RedisCluster
AsyncRedisClient = AsyncRedisCluster


def _build_retry() -> Retry:
    """Per-command retry with exponential backoff (0.1s → 10s cap)."""
    return Retry(
        backoff=ExponentialBackoff(cap=10, base=0.1),
        retries=REDIS_RETRY_ATTEMPTS,
        supported_errors=TRANSIENT_REDIS_ERRORS,
    )


def _build_async_retry() -> AsyncRetry:
    """Async sibling of :func:`_build_retry` — redis-py ships separate sync
    and async ``Retry`` classes; passing the wrong one fails type checks."""
    return AsyncRetry(
        backoff=ExponentialBackoff(cap=10, base=0.1),
        retries=REDIS_RETRY_ATTEMPTS,
        supported_errors=TRANSIENT_REDIS_ERRORS,
    )


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
        # Drives both per-command retries and the cluster-level retry counter.
        retry=_build_retry(),
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
        # redis-py 6.x AsyncRedisCluster ignores `retry_on_error` — the cluster
        # retry path uses a hardcoded {Timeout, Connection, ClusterDown} set.
        # Pass `retry` only to match the sync RedisCluster call above.
        retry=_build_async_retry(),
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


# Async Redis connections bind to the loop where they are first used.
_async_clients: WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncRedisClient] = (
    WeakKeyDictionary()
)
_async_client_finalizers: WeakKeyDictionary[
    asyncio.AbstractEventLoop, AsyncGenerator[None, None]
] = WeakKeyDictionary()
_async_client_locks: WeakKeyDictionary[
    asyncio.AbstractEventLoop, ReferenceType[asyncio.Lock]
] = WeakKeyDictionary()


def _get_async_client_lock(loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
    lock_ref = _async_client_locks.get(loop)
    lock = lock_ref() if lock_ref is not None else None
    if lock is None:
        lock = asyncio.Lock()
        _async_client_locks[loop] = ref(lock)
    return lock


@conn_retry("AsyncRedis", "Releasing connection")
async def disconnect_async():
    loop = asyncio.get_running_loop()
    async with _get_async_client_lock(loop):
        client = _async_clients.pop(loop, None)
        finalizer = _async_client_finalizers.pop(loop, None)
        if finalizer is not None:
            await finalizer.aclose()
        elif client is not None:
            await client.close()


async def get_redis_async() -> AsyncRedisClient:
    loop = asyncio.get_running_loop()
    async with _get_async_client_lock(loop):
        client = _async_clients.get(loop)
        if client is None:
            client = await connect_async()
            finalizer = _close_async_client_on_loop_shutdown(client)
            await anext(finalizer)
            _async_clients[loop] = client
            _async_client_finalizers[loop] = finalizer
        return client


async def _close_async_client_on_loop_shutdown(
    client: AsyncRedisClient,
) -> AsyncGenerator[None, None]:
    """Close a loop-owned client during ``loop.shutdown_asyncgens()``."""
    try:
        yield
    finally:
        loop = asyncio.get_running_loop()
        if _async_clients.get(loop) is client:
            _async_clients.pop(loop, None)
            _async_client_finalizers.pop(loop, None)
        await client.close()


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
