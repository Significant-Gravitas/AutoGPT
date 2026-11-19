"""The shared cache's Redis clients.

A cache answers in milliseconds or counts as absent, so every command waits at
most ``REDIS_OP_TIMEOUT_S``: a shard that stops answering costs a caller that
long, never a hang. A coroutine uses its event loop's own client
(``get_async_redis``), so the loop serves everything else while Redis answers;
the sync client (``get_redis``) blocks the thread that calls it.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Any

from redis.asyncio.cluster import ClusterNode as AsyncClusterNode
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import NoBackoff
from redis.cluster import ClusterNode, RedisCluster
from redis.retry import Retry

from backend.util.retry import conn_retry
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Far above a healthy answer (a few ms), and short enough that a caller who
# gives up and computes the value itself is still quick.
REDIS_OP_TIMEOUT_S = 1.0
# One more try after a reset or refused connection (a restart, a failover).
_RETRIES = 1


@cache
@conn_retry("Redis", "Acquiring cache connection")
def get_redis() -> RedisCluster:
    """The sync client (singleton). A command waits at most
    ``REDIS_OP_TIMEOUT_S`` per try."""
    # Local import breaks the redis_client <-> cache module cycle.
    from backend.data.redis_client import _address_remap

    c = RedisCluster(
        startup_nodes=[
            ClusterNode(settings.config.redis_host, settings.config.redis_port)
        ],
        password=settings.config.redis_password or None,
        decode_responses=False,  # Binary mode for pickle
        max_connections=50,
        socket_keepalive=True,
        socket_connect_timeout=REDIS_OP_TIMEOUT_S,
        socket_timeout=REDIS_OP_TIMEOUT_S,
        retry=Retry(NoBackoff(), _RETRIES),
        address_remap=_address_remap,
    )
    c.ping()
    return c


# An asyncio connection belongs to the loop that opened it, so each loop gets
# a client of its own. Only the loop holds it, so nothing here keeps a loop
# alive, and closing the loop closes it first, whoever does that: asyncio.run,
# pytest-asyncio, or a worker's own teardown that skips shutdown_asyncgens().
_LOOP_ATTR = "_shared_cache_redis"


async def get_async_redis() -> AsyncRedisCluster:
    """This event loop's client, with the cluster's slot map read: an
    uninitialized client sends its first command to any node, and redis-py
    follows the redirect to the owner's announced address, not remapped.
    Call it within the caller's bound, since it may read the map."""
    loop = asyncio.get_running_loop()
    client: AsyncRedisCluster | None = getattr(loop, _LOOP_ATTR, None)
    if client is None:
        client = _new_async_redis()
        setattr(loop, _LOOP_ATTR, client)
        _close_with(loop, client)
    return await client.initialize()


def _close_with(loop: asyncio.AbstractEventLoop, client: AsyncRedisCluster) -> None:
    """Make ``loop.close()`` close ``client`` first. ``loop.close()`` alone
    leaves its connections open, and nothing can close them once the loop
    is closed: collected, they report themselves to whatever loop runs."""
    close_loop = loop.close

    def close() -> None:
        if loop.is_running():
            close_loop()  # Raises: a running loop can't be closed.
            return
        if vars(loop).get("close") is close:
            delattr(loop, "close")
        if getattr(loop, _LOOP_ATTR, None) is client:
            delattr(loop, _LOOP_ATTR)
            if not loop.is_closed():
                try:
                    _run_to_completion(loop, _close(client))
                except Exception as e:
                    logger.warning(f"Could not close the cache's Redis client: {e!r}")
        close_loop()

    setattr(loop, "close", close)


def _run_to_completion(loop: asyncio.AbstractEventLoop, coroutine: Any) -> None:
    """``loop.run_until_complete``, from a thread of its own when this one
    already runs a loop (a coroutine closing a worker's loop): a thread runs
    one loop at a time."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop.run_until_complete(coroutine)
        return
    with ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(loop.run_until_complete, coroutine).result()


async def _close(client: AsyncRedisCluster) -> None:
    async with asyncio.timeout(REDIS_OP_TIMEOUT_S):
        await client.aclose()
        # aclose() skips a client that a connection error left due to read
        # the slot map again, though its connections are still open.
        await client.nodes_manager.aclose()
        await client.nodes_manager.aclose("startup_nodes")


def _new_async_redis() -> AsyncRedisCluster:
    # Local import breaks the redis_client <-> cache module cycle.
    from backend.data.redis_client import REDIS_MAX_CONNECTIONS, _address_remap

    return AsyncRedisCluster(
        startup_nodes=[
            AsyncClusterNode(settings.config.redis_host, settings.config.redis_port)
        ],
        password=settings.config.redis_password or None,
        decode_responses=False,  # Binary mode for pickle
        # Each command in flight holds a connection, and past the cap redis-py
        # refuses one rather than waits for it.
        max_connections=REDIS_MAX_CONNECTIONS,
        socket_keepalive=True,
        socket_connect_timeout=REDIS_OP_TIMEOUT_S,
        retry=AsyncRetry(NoBackoff(), _RETRIES),
        address_remap=_address_remap,
    )
