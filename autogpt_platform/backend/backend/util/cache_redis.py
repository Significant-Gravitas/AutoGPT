"""The shared cache's Redis clients.

A coroutine uses its event loop's own client (``get_async_redis``), so the
loop serves everything else while Redis answers. A cache answers in
milliseconds or counts as absent, so the cache gives each of its commands at
most ``REDIS_OP_TIMEOUT_S``: a shard that stops answering costs a coroutine
that long, never a hang. The sync client (``get_redis``) waits for Redis and
blocks the thread that calls it: an invalidation it sends must be applied,
not dropped.
"""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Any

from redis.asyncio.cluster import ClusterNode as AsyncClusterNode
from redis.asyncio.cluster import NodesManager as AsyncNodesManager
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.asyncio.retry import Retry as AsyncRetry
from redis.backoff import NoBackoff
from redis.cluster import ClusterNode, NodesManager, RedisCluster
from redis.exceptions import AskError, MovedError

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
    """The sync client (singleton)."""
    # Local import breaks the redis_client <-> cache module cycle.
    from backend.data.redis_client import _address_remap

    c = _RedisCluster(
        startup_nodes=[
            ClusterNode(settings.config.redis_host, settings.config.redis_port)
        ],
        password=settings.config.redis_password or None,
        decode_responses=False,  # Binary mode for pickle
        max_connections=50,
        socket_keepalive=True,
        socket_connect_timeout=5,
        retry_on_timeout=True,
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
    uninitialized client sends its first command to any node, which then
    redirects it. Call it within the caller's bound, since it may read the
    map."""
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
        try:
            if getattr(loop, _LOOP_ATTR, None) is client:
                delattr(loop, _LOOP_ATTR)
                if not loop.is_closed():
                    try:
                        _run_to_completion(loop, _close(client))
                    except Exception as e:
                        logger.warning(
                            f"Could not close the cache's Redis client: {e!r}"
                        )
        finally:
            # Even through a KeyboardInterrupt while the client closed.
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
    """Close ``client``, then each of its node pools: ``aclose()`` skips a
    client that a connection error left due to read the slot map again,
    though its connections are still open. Each attempt is bounded and made
    whatever the one before did; the first failure is raised after all."""
    failures: list[Exception] = []
    for close in (
        client.aclose,
        client.nodes_manager.aclose,
        lambda: client.nodes_manager.aclose("startup_nodes"),
    ):
        try:
            async with asyncio.timeout(REDIS_OP_TIMEOUT_S):
                await close()
        except Exception as e:
            failures.append(e)
    if failures:
        raise failures[0]


def _new_async_redis() -> AsyncRedisCluster:
    # Local import breaks the redis_client <-> cache module cycle.
    from backend.data.redis_client import REDIS_MAX_CONNECTIONS, _address_remap

    return _AsyncRedisCluster(
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
        # redis-py 8.1 closes every connection on every fifth MOVED, failing
        # the other commands in flight on them: during a resharding, a stream
        # of reads computed instead. Follow each MOVED on its own, as redis-py
        # does after 8.1. A connection error still makes it read the map again.
        reinitialize_steps=0,
    )


# redis-py 8.1 applies ``address_remap`` to the slot map it reads but not to
# the address a MOVED or ASK reply names, so a redirect would dial a shard's
# announced address: unresolvable wherever the remap is needed, as on CI's
# cluster. These clients remap both. redis-py fixed it after 8.1
# (redis/redis-py#4522, first in 8.2.0b1); drop them once on a release with it.
Remap = Callable[[str, int], tuple[str, int]]


class _NodesManager(NodesManager):
    def move_slot(self, e: AskError | MovedError) -> None:
        _remap_reply(self.remap_host_port, e)
        super().move_slot(e)


class _RedisCluster(RedisCluster):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.nodes_manager.__class__ = _NodesManager

    def get_node(
        self,
        host: str | None = None,
        port: int | None = None,
        node_name: str | None = None,
    ) -> ClusterNode | None:
        return _get_node(
            super().get_node, self.nodes_manager.remap_host_port, host, port, node_name
        )


class _AsyncNodesManager(AsyncNodesManager):
    __slots__ = ()

    async def move_slot(self, e: AskError | MovedError) -> None:
        _remap_reply(self.remap_host_port, e)
        await super().move_slot(e)


class _AsyncRedisCluster(AsyncRedisCluster):
    __slots__ = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.nodes_manager.__class__ = _AsyncNodesManager

    def get_node(
        self,
        host: str | None = None,
        port: int | None = None,
        node_name: str | None = None,
    ) -> AsyncClusterNode | None:
        return _get_node(
            super().get_node, self.nodes_manager.remap_host_port, host, port, node_name
        )


def _remap_reply(remap: Remap, e: AskError | MovedError) -> None:
    e.host, e.port = remap(e.host, e.port)
    e.node_addr = (e.host, e.port)


def _get_node(
    get_node: Callable[..., Any],
    remap: Remap,
    host: str | None,
    port: int | None,
    node_name: str | None,
) -> Any:
    """``get_node``, finding the node an ASK reply names at its remapped
    address too: redis-py looks it up by the name the reply gives. By host
    and port, since a node given as ``localhost`` is named by its IP."""
    node = get_node(host, port, node_name)
    if node is None and node_name is not None:
        announced_host, _, announced_port = node_name.rpartition(":")
        node = get_node(*remap(announced_host, int(announced_port)))
    return node
