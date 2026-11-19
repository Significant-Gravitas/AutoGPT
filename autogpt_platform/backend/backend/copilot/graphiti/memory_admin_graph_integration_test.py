import pytest
from pytest_mock import MockerFixture
from starlette.requests import Request

from backend.api.features.admin.memory_admin_routes import get_graph
from backend.copilot.graphiti.falkordb_driver import AutoGPTFalkorDriver
from backend.copilot.graphiti.scope import MemoryScope


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_inspector_edge_limit_counts_only_visible_connections(
    scope_graph: tuple[AutoGPTFalkorDriver, MemoryScope],
) -> None:
    driver, scope = scope_graph
    await driver.execute_query(
        """
        CREATE (alice:Entity {uuid: 'alice', group_id: $group_id}),
               (bob:Entity {uuid: 'bob', group_id: $group_id}),
               (carol:Entity {uuid: 'carol', group_id: $group_id}),
               (alice)-[:RELATES_TO {uuid: 'alice-carol-1', group_id: $group_id}]->(carol),
               (alice)-[:RELATES_TO {uuid: 'alice-carol-2', group_id: $group_id}]->(carol),
               (bob)-[:RELATES_TO {uuid: 'bob-alice', group_id: $group_id}]->(alice),
               (carol)-[:RELATES_TO {uuid: 'carol-bob', group_id: $group_id}]->(bob)
        """,
        group_id=scope.group_id,
    )

    graph = await get_graph(
        request=Request({"type": "http"}),
        user_id="me",
        caller_id=scope.owner_user_id,
        jwt_payload={},
        node_limit=2,
        edge_limit=1,
        include_episodes=False,
        include_communities=False,
    )

    # Any two nodes have a connection, regardless of which pair FalkorDB selects.
    visible = {node.uuid for node in graph.nodes}
    assert len(visible) == 2
    assert visible <= {"alice", "bob", "carol"}
    assert len(graph.edges) == 1
    (edge,) = graph.edges
    assert {edge.source, edge.target} == visible
    assert graph.truncated is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_memory_inspector_filters_connections_before_limiting_them(
    scope_graph: tuple[AutoGPTFalkorDriver, MemoryScope],
    mocker: MockerFixture,
) -> None:
    driver, scope = scope_graph
    await driver.execute_query(
        """
        CREATE (alice:Entity {uuid: 'alice', group_id: $group_id}),
               (bob:Entity {uuid: 'bob', group_id: $group_id}),
               (episode:Episodic {uuid: 'episode', group_id: $group_id}),
               (episode)-[:MENTIONS {uuid: 'excluded-1', group_id: $group_id}]->(alice),
               (episode)-[:MENTIONS {uuid: 'excluded-2', group_id: $group_id}]->(bob),
               (alice)-[:RELATES_TO {uuid: 'visible', group_id: $group_id}]->(bob)
        """,
        group_id=scope.group_id,
    )

    execute_query = AutoGPTFalkorDriver.execute_query
    ordered_edge_queries = 0

    async def order_edges(
        driver: AutoGPTFalkorDriver, cypher_query_: str, **params: object
    ):
        nonlocal ordered_edge_queries
        # Make the two excluded edges fill the limit if FalkorDB is asked for
        # unfiltered edges. The real database still does the matching and limit.
        limit_clause = "LIMIT $limit"
        if "RETURN e.uuid AS uuid," in cypher_query_ and limit_clause in cypher_query_:
            cypher_query_ = cypher_query_.replace(
                limit_clause, f"ORDER BY e.uuid {limit_clause}", 1
            )
            ordered_edge_queries += 1
        return await execute_query(driver, cypher_query_, **params)

    mocker.patch.object(AutoGPTFalkorDriver, "execute_query", order_edges)
    graph = await get_graph(
        request=Request({"type": "http"}),
        user_id="me",
        caller_id=scope.owner_user_id,
        jwt_payload={},
        node_limit=2,
        edge_limit=1,
        include_episodes=True,
        include_communities=False,
    )

    assert ordered_edge_queries == 1
    # Entity rows fill the node budget before the endpoint reads episodes.
    assert {node.uuid for node in graph.nodes} == {"alice", "bob"}
    assert [edge.uuid for edge in graph.edges] == ["visible"]
    assert graph.truncated is True
