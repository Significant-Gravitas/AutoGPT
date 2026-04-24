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

# Prefer REDIS_CLUSTER_HOST (the new sharded cluster) over REDIS_HOST so the
# cluster-only image can land in an environment where both services still
# exist. Old-image pods don't read the cluster vars and keep using the old
# standalone Redis via REDIS_HOST. Once the rollout is stable, the cleanup PR
# removes both the old env vars and this fallback.
HOST = os.getenv("REDIS_CLUSTER_HOST") or os.getenv("REDIS_HOST", "localhost")
PORT = int(os.getenv("REDIS_CLUSTER_PORT") or os.getenv("REDIS_PORT", "6379"))
PASSWORD = os.getenv("REDIS_PASSWORD", None)

# Default socket timeouts so a wedged Redis endpoint can't hang callers
# indefinitely — long-running code paths (cluster_lock refresh in particular)
# rely on these to fail-fast instead of blocking on no-response TCP. Override
# via env if a specific deployment needs a different budget.
#
# 30s matches the convention in ``backend.data.rabbitmq`` and leaves ~6x
# headroom over the largest ``xread(block=5000)`` wait in stream_registry.
# The connect timeout is shorter (5s) because initial connects should be
# fast; a slow connect usually means the endpoint is genuinely unreachable.
SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "30"))
SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5"))
# How often redis-py sends a PING on idle connections to detect half-open
# sockets; cheap and avoids waiting for the OS TCP keepalive (~2h default).
HEALTH_CHECK_INTERVAL = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))

# Bypass the HOST-pinning address_remap when each shard already announces a
# hostname the backend can resolve directly (e.g. compose DNS names
# ``redis-0``/``redis-1``/``redis-2``). Enable this in the compose network
# where the three shards live under distinct service names. Leave off on the
# laptop (HOST=localhost + distinct published ports reach each container) and
# in prod (HOST pins onto the load-balanced seed Service).
USE_ANNOUNCED_ADDRESS = os.getenv("REDIS_USE_ANNOUNCED_ADDRESS", "").lower() in (
    "1",
    "true",
    "yes",
)

logger = logging.getLogger(__name__)

# Aliases kept so call-sites don't care which class this is — the backend
# always talks to a Redis Cluster (1-shard locally, sharded in prod).
RedisClient = RedisCluster
AsyncRedisClient = AsyncRedisCluster


def _address_remap(addr: tuple[str, int]) -> tuple[str, int]:
    """Rewrite the address returned by ``CLUSTER SLOTS`` for each shard.

    Default: pin every shard to the configured seed ``HOST`` and keep its
    announced port. This works when the seed DNS load-balances across
    shards (prod) and when each shard is reachable at the seed host via a
    distinct published port (laptop talking to the local compose cluster,
    where each of 17000/17001/17002 maps to its container).

    When ``REDIS_USE_ANNOUNCED_ADDRESS`` is set we pass the announced
    ``(host, port)`` straight through — the caller is inside a network
    where each shard's announced hostname resolves to its own container
    (e.g. compose DNS names ``redis-0``/``redis-1``/``redis-2``).
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
    # Only close pub/sub if it was actually constructed — avoid opening a
    # fresh connection at shutdown just to tear it down.
    pubsub = _get_cached_pubsub()
    if pubsub is not None:
        try:
            pubsub.close()
        except Exception:
            logger.warning("Failed to close pub/sub Redis connection", exc_info=True)
    get_redis_pubsub.cache_clear()


def _get_cached_pubsub() -> Redis | None:
    """Return the cached pub/sub client if one exists, else ``None``.

    ``cache_info()["size"]`` tells us whether the memo has been populated
    without triggering a new call.
    """
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


# Cache one AsyncRedisCluster per event loop. `AsyncRedisCluster` binds to the
# loop it is first awaited on (unlike the sync `RedisCluster` client), so a
# simple module-level singleton breaks when tests run on multiple loops — the
# cached client's internal Tasks are attached to a dead loop and every
# subsequent call raises `RuntimeError: Event loop is closed`. Keying by
# `id(loop)` keeps the prod hot-path (one loop for the process lifetime) as
# fast as the old `@thread_cached` singleton while making test harnesses that
# spin up per-test loops safe.
_async_clients: dict[int, AsyncRedisCluster] = {}
_async_pubsub_clients: dict[int, AsyncRedis] = {}


@conn_retry("AsyncRedis", "Releasing connection")
async def disconnect_async():
    loop = asyncio.get_running_loop()
    c = _async_clients.pop(id(loop), None)
    if c is not None:
        await c.close()
    # Close the pub/sub client independently — a caller may have acquired
    # ``get_redis_pubsub_async`` without ever calling ``get_redis_async``,
    # and an early return on a missing cluster client would leak the FD.
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


# Sharded vs classic pub/sub:
#
# * For **fire-and-forget publish** (no subscriber in the backend) we prefer
#   sharded pub/sub via the cluster client's ``spublish()`` — it skips the
#   cluster-bus broadcast and only hits the one shard that owns the channel's
#   keyslot.  See ``backend.copilot.pending_messages``.
#
# * For the ``listen("*")``-style event buses (``event_bus``,
#   ``notification_bus``) we stay on classic ``PUBLISH``/``PSUBSCRIBE``
#   against a plain (Async)Redis connection because:
#     1. redis-py 6.x ``AsyncRedisCluster`` has no pub/sub support at all
#        (no ``pubsub()``, no ``spublish()``); only the sync cluster client
#        has ``ClusterPubSub``.  The backend's pub/sub call-sites are async.
#     2. Sharded pub/sub has no pattern subscribe (``PSSUBSCRIBE`` does not
#        exist), and ``ws_api.event_broadcaster`` relies on ``listen("*")``
#        fan-in to route execution/notification events to every connected
#        websocket.  Moving that off a single broadcast subscription is a
#        bigger rework than this PR covers.
#   Classic pub/sub in a Redis Cluster still works: PUBLISH is broadcast
#   across the cluster bus, so any one-node subscription receives every
#   message. Scale cost is borne by the cluster bus; acceptable at our
#   current traffic levels.


@conn_retry("RedisPubSub", "Acquiring connection")
def connect_pubsub() -> Redis:
    c = Redis(
        host=HOST,
        port=PORT,
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
async def connect_pubsub_async() -> AsyncRedis:
    c = AsyncRedis(
        host=HOST,
        port=PORT,
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
    """Return a plain ``Redis`` client dedicated to pub/sub.

    A subscribed connection blocks on ``listen()`` and cannot be interleaved
    with regular command traffic, so pub/sub gets its own connection separate
    from the cluster-aware client returned by :func:`get_redis`.
    """
    return connect_pubsub()


async def get_redis_pubsub_async() -> AsyncRedis:
    """Async equivalent of :func:`get_redis_pubsub`.

    Cached per event loop: ``AsyncRedis`` clients bind to the loop they are
    first awaited on, so a simple module-level singleton breaks across test
    loops. Keying by ``id(loop)`` keeps one client per loop for the process
    lifetime.
    """
    loop = asyncio.get_running_loop()
    client = _async_pubsub_clients.get(id(loop))
    if client is None:
        client = await connect_pubsub_async()
        _async_pubsub_clients[id(loop)] = client
    return client
