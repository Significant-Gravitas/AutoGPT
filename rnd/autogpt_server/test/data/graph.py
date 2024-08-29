from uuid import UUID

import pytest
from prisma.models import User

from autogpt_server.blocks.basic import InputBlock, ValueBlock
from autogpt_server.data.graph import Graph, Link, Node
from autogpt_server.data.user import get_or_create_user
from autogpt_server.server.model import CreateGraph
from autogpt_server.util.test import SpinTestServer


async def create_test_user() -> User:
    test_user_data = {
        "sub": "ef3b97d7-1161-4eb4-92b2-10c24fb154c1",
        "email": "testuser#example.com",
        "name": "Test User",
    }
    user = await get_or_create_user(test_user_data)
    return user


@pytest.mark.asyncio(scope="session")
async def test_graph_creation(server: SpinTestServer):
    value_block = ValueBlock().id
    input_block = InputBlock().id

    test_user = await create_test_user()

    graph = Graph(
        id="test_graph",
        name="TestGraph",
        description="Test graph",
        nodes=[
            Node(id="node_1", block_id=value_block),
            Node(id="node_2", block_id=input_block),
            Node(id="node_3", block_id=value_block),
        ],
        links=[
            Link(
                source_id="node_1",
                sink_id="node_3",
                source_name="output",
                sink_name="input",
            ),
        ],
        subgraphs={"subgraph_1": ["node_2", "node_3"]},
    )
    create_graph = CreateGraph(graph=graph)

    try:
        await server.agent_server.create_graph(
            create_graph, False, user_id=test_user.id
        )
        assert False, "Should not be able to connect nodes from different subgraphs"
    except ValueError as e:
        assert "different subgraph" in str(e)

    # Change node_1 <-> node_3 link to node_1 <-> node_2 (input for subgraph_1)
    graph.links[0].sink_id = "node_2"
    created_graph = await server.agent_server.create_graph(
        create_graph, False, user_id=test_user.id
    )

    assert UUID(created_graph.id)
    assert created_graph.name == "TestGraph"

    assert len(created_graph.nodes) == 3
    assert UUID(created_graph.nodes[0].id)
    assert UUID(created_graph.nodes[1].id)
    assert UUID(created_graph.nodes[2].id)

    nodes = created_graph.nodes
    links = created_graph.links
    assert len(links) == 1
    assert {nodes[0].id, nodes[1].id} == {links[0].source_id, links[0].sink_id}

    assert len(created_graph.subgraphs) == 1
    assert len(created_graph.subgraph_map) == len(created_graph.nodes) == 3
