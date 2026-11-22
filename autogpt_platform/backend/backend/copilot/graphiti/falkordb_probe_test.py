import asyncio
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, patch

import pytest
from redis.asyncio import Redis
from redis.asyncio.cluster import RedisCluster

from . import falkordb_connect


@pytest.mark.asyncio
async def test_silent_probes_release_all_connect_workers_promptly(monkeypatch) -> None:
    monkeypatch.setattr(
        falkordb_connect.graphiti_config, "falkordb_socket_connect_timeout", 0.1
    )
    monkeypatch.setattr(
        falkordb_connect.graphiti_config, "falkordb_socket_timeout", 2.0
    )
    barrier = threading.Barrier(5, timeout=3)
    with socket.socket() as server, ThreadPoolExecutor(max_workers=4) as executor:
        server.bind(("127.0.0.1", 0))
        server.listen(16)
        port = server.getsockname()[1]
        monkeypatch.setattr(falkordb_connect, "_CONNECT_EXECUTOR", executor)

        def build():
            barrier.wait()
            return falkordb_connect.new_falkordb_client("127.0.0.1", port)

        builds = [
            asyncio.create_task(falkordb_connect.build_off_loop(build))
            for _ in range(4)
        ]
        await asyncio.to_thread(barrier.wait)
        try:
            assert (
                await asyncio.wait_for(
                    falkordb_connect.build_off_loop(lambda: "worker available"), 0.8
                )
                == "worker available"
            )
        finally:
            results = await asyncio.gather(*builds, return_exceptions=True)
        assert all(isinstance(result, Exception) for result in results)


@pytest.mark.asyncio
@pytest.mark.parametrize("cluster", [False, True])
async def test_short_probe_keeps_long_deadlines_for_commands(
    monkeypatch, cluster
) -> None:
    monkeypatch.setattr(
        falkordb_connect.graphiti_config, "falkordb_socket_connect_timeout", 0.1
    )
    monkeypatch.setattr(
        falkordb_connect.graphiti_config, "falkordb_socket_timeout", 30.0
    )
    probe_timeouts: list[float] = []

    def probe(connection: Redis) -> bool:
        probe_timeouts.append(
            connection.connection_pool.connection_kwargs["socket_timeout"]
        )
        return cluster

    with patch("falkordb.asyncio.falkordb.Is_Cluster", side_effect=probe):
        client = falkordb_connect.new_falkordb_client()
    try:
        assert probe_timeouts == [0.1]
        connection = client.connection
        if isinstance(connection, RedisCluster):
            assert connection.connection_kwargs["socket_timeout"] == 30.0
            assert connection.nodes_manager.connection_kwargs["socket_timeout"] == 30.0
            assert all(
                node.connection_kwargs["socket_timeout"] == 30.0
                for node in connection.nodes_manager.startup_nodes.values()
            )
        else:
            assert (
                connection.connection_pool.connection_kwargs["socket_timeout"] == 30.0
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_deferred_client_supports_the_upstream_graph_surface() -> None:
    real = AsyncMock()
    real.execute_command.return_value = [[], ["Query internal execution time: 0.1 ms"]]
    client = falkordb_connect.DeferredFalkorDB(lambda: real)
    graph = client.select_graph("surface_contract")

    await graph.ro_query("RETURN 1")
    real.execute_command.assert_awaited_once_with(
        "GRAPH.RO_QUERY", "surface_contract", "RETURN 1", "--compact"
    )
    real.execute_command.reset_mock()
    await graph.delete()
    real.execute_command.assert_awaited_once_with("GRAPH.DELETE", "surface_contract")
    real.list_graphs.return_value = ["surface_contract"]
    assert await client.list_graphs() == ["surface_contract"]
    await client.aclose()
    real.aclose.assert_awaited_once()
