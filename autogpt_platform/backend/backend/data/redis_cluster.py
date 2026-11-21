"""Custom RedisCluster and AsyncRedisCluster subclasses that handle address
remapping and connection pooling during slot migrations (#17074, #17090).

redis-py 8.1 applies ``address_remap`` to the slot map it reads but not to
the address a MOVED or ASK reply names, so a redirect would dial a shard's
announced address: unresolvable wherever the remap is needed, as on CI's
cluster. These clients remap both. redis-py fixed it after 8.1
(redis/redis-py#4522, first in 8.2.0b1); drop them once on a release with it.

redis-py 8.1's asyncio ``RedisCluster`` sends ``ASKING`` and the command as two
separate commands. With multiple idle connections to the target node, the command
takes a different connection from the FIFO pool, causing MOVED and TTL exhausted
during resharding migrations (#17090). ``_AsyncClusterNode`` hands out connections
last in, first out (LIFO) so the command reuses the connection ASKING just used.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from redis.asyncio.cluster import ClusterNode as AsyncClusterNode
from redis.asyncio.cluster import NodesManager as AsyncNodesManager
from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
from redis.asyncio.connection import Connection as AsyncConnection
from redis.cluster import ClusterNode, NodesManager, RedisCluster
from redis.exceptions import AskError, MovedError

Remap = Callable[[str, int], tuple[str, int]]


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


class _AsyncClusterNode(AsyncClusterNode):
    __slots__ = ()

    def release(self, connection: AsyncConnection) -> None:
        if connection.should_reconnect():
            task = asyncio.create_task(self._disconnect_and_release(connection))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return
        self._free.appendleft(connection)


class _AsyncNodesManager(AsyncNodesManager):
    __slots__ = ()

    async def initialize(self) -> None:
        await super().initialize()
        for node in self.nodes_cache.values():
            node.__class__ = _AsyncClusterNode
        for node in self.startup_nodes.values():
            node.__class__ = _AsyncClusterNode
        if self.default_node:
            self.default_node.__class__ = _AsyncClusterNode

    async def move_slot(self, e: AskError | MovedError) -> None:
        _remap_reply(self.remap_host_port, e)
        await super().move_slot(e)
        node = self.get_node(host=e.host, port=e.port)
        if node is not None:
            node.__class__ = _AsyncClusterNode

    def get_node(
        self,
        host: str | None = None,
        port: int | None = None,
        node_name: str | None = None,
    ) -> AsyncClusterNode | None:
        node = super().get_node(host=host, port=port, node_name=node_name)
        if node is not None and node.__class__ is not _AsyncClusterNode:
            node.__class__ = _AsyncClusterNode
        return node


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
        node = _get_node(
            super().get_node, self.nodes_manager.remap_host_port, host, port, node_name
        )
        if node is not None and node.__class__ is not _AsyncClusterNode:
            node.__class__ = _AsyncClusterNode
        return node
