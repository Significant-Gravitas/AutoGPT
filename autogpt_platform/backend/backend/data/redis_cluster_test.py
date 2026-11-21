"""Unit tests for the redirect-remapped and LIFO-pooled Redis cluster classes (#17074, #17090).

Tests that:
1. AsyncClusterNode implements LIFO connection release, ensuring that ASKING and
   its following command share the same connection across multiple idle connections (#17090).
2. _AsyncNodesManager and _NodesManager apply address_remap to MOVED and ASK redirects
   so announced addresses resolve on clusters where remap is active.
3. Node caches consistently upgrade nodes to _AsyncClusterNode.
"""

import asyncio
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest
from redis.asyncio.cluster import ClusterNode as AsyncClusterNode
from redis.asyncio.connection import Connection as AsyncConnection
from redis.cluster import ClusterNode
from redis.exceptions import AskError, MovedError

from backend.data.redis_cluster import (
    _AsyncClusterNode,
    _AsyncNodesManager,
    _AsyncRedisCluster,
    _NodesManager,
    _RedisCluster,
    _get_node,
    _remap_reply,
)


def test_async_cluster_node_lifo_release() -> None:
    """With multiple idle connections, LIFO release ensures the connection
    that executed ASKING is immediately popped for the subsequent command (#17090)."""
    node = AsyncClusterNode("127.0.0.1", 7001)
    node.__class__ = _AsyncClusterNode

    # Setup 8 idle connections
    mock_conns = []
    for i in range(8):
        conn = MagicMock(spec=AsyncConnection)
        conn.should_reconnect.return_value = False
        conn.id = i
        mock_conns.append(conn)

    node._connections = mock_conns[:]
    node._free = deque(mock_conns[:])

    # 1. Acquire connection for ASKING
    conn_asking = node.acquire_connection()
    assert conn_asking.id == 0
    assert len(node._free) == 7

    # 2. ASKING finishes and releases the connection
    node.release(conn_asking)
    assert len(node._free) == 8

    # 3. Acquire connection for actual command (e.g. GET)
    conn_command = node.acquire_connection()
    # In LIFO, conn_command must be the exact same connection (id 0), not id 1!
    assert conn_command is conn_asking
    assert conn_command.id == 0


def test_remap_reply_updates_host_port_and_node_addr() -> None:
    """MOVED or ASK error host/port must be remapped by the remap callback."""

    def fake_remap(host: str, port: int) -> tuple[str, int]:
        return "127.0.0.1", port

    error = AskError("1234 192.168.1.50:6379")
    error.host = "192.168.1.50"
    error.port = 6379

    _remap_reply(fake_remap, error)
    assert error.host == "127.0.0.1"
    assert error.port == 6379
    assert error.node_addr == ("127.0.0.1", 6379)


def test_get_node_finds_remapped_announced_address() -> None:
    """When a node is requested by an unresolvable announced name, get_node
    finds the node via its remapped host/port."""

    def fake_remap(host: str, port: int) -> tuple[str, int]:
        return "127.0.0.1", port

    existing_node = ClusterNode("127.0.0.1", 7001)

    def mock_get_node(host: str | None, port: int | None, node_name: str | None = None):
        if host == "127.0.0.1" and port == 7001:
            return existing_node
        return None

    # Looking up announced hostname redis-shard-1:7001 finds existing_node
    found = _get_node(mock_get_node, fake_remap, None, None, "redis-shard-1:7001")
    assert found is existing_node


@pytest.mark.asyncio
async def test_async_nodes_manager_upgrades_nodes_to_lifo() -> None:
    """Nodes added to _AsyncNodesManager are upgraded to _AsyncClusterNode."""
    remap = lambda host, port: (host, port)
    manager = _AsyncNodesManager(
        startup_nodes=[AsyncClusterNode("127.0.0.1", 7000)],
        require_full_coverage=False,
        connection_kwargs={},
        address_remap=remap,
    )

    # Initialize a test node in nodes_cache
    raw_node = AsyncClusterNode("127.0.0.1", 7001)
    manager.nodes_cache[raw_node.name] = raw_node

    # get_node must upgrade raw_node to _AsyncClusterNode
    retrieved = manager.get_node(host="127.0.0.1", port=7001)
    assert retrieved is not None
    assert isinstance(retrieved, _AsyncClusterNode)
    assert retrieved.__class__ is _AsyncClusterNode
