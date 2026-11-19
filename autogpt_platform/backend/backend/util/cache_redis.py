"""The shared cache's Redis clients.

A cache answers in milliseconds or counts as absent, so every command waits at
most ``REDIS_OP_TIMEOUT_S``: a shard that stops answering costs a caller that
long, never a hang. A coroutine uses its event loop's own client
(``get_async_redis``), so the loop serves everything else while Redis answers;
the sync client (``get_redis``) blocks the thread that calls it.
"""

import asyncio
from collections.abc import AsyncGenerator
from functools import cache
from weakref import WeakKeyDictionary

from redis.asyncio.cluster import ClusterNode as AsyncClusterNode
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import NoBackoff
from redis.cluster import ClusterNode, RedisCluster
from redis.retry import Retry

from backend.util.retry import conn_retry
from backend.util.settings import Settings

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
# a client of its own, closed when the loop shuts its async generators down.
_async_clients: WeakKeyDictionary[asyncio.AbstractEventLoop, AsyncRedisCluster] = (
    WeakKeyDictionary()
)
_async_client_closers: WeakKeyDictionary[
    asyncio.AbstractEventLoop, AsyncGenerator[None, None]
] = WeakKeyDictionary()


async def get_async_redis() -> AsyncRedisCluster:
    """This event loop's client. It connects on its first command, so the
    caller's bound on that command covers connecting too."""
    loop = asyncio.get_running_loop()
    client = _async_clients.get(loop)
    if client is None:
        client = _async_clients[loop] = _new_async_redis()
        closer = _async_client_closers[loop] = _close_with_loop(client)
        await anext(closer)
    return client


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


async def _close_with_loop(client: AsyncRedisCluster) -> AsyncGenerator[None, None]:
    """Waits at its ``yield`` until the loop's ``shutdown_asyncgens()``."""
    try:
        yield
    finally:
        loop = asyncio.get_running_loop()
        if _async_clients.get(loop) is client:
            _async_clients.pop(loop, None)
            _async_client_closers.pop(loop, None)
        await client.aclose()
