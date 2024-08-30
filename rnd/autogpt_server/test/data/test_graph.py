from uuid import UUID

import pytest

from autogpt_server.blocks.basic import InputBlock, StoreValueBlock
from autogpt_server.data.graph import Graph, Link, Node
from autogpt_server.data.user import DEFAULT_USER_ID, create_default_user
from autogpt_server.server.model import CreateGraph
from autogpt_server.util.test import SpinTestServer


@pytest.mark.asyncio(scope="session")
async def test_graph_creation(server: SpinTestServer):
    await create_default_user("false")

    value_block = StoreValueBlock().id
    input_block = InputBlock().id

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
        await server.agent_server.create_graph(create_graph, False, DEFAULT_USER_ID)
        assert False, "Should not be able to connect nodes from different subgraphs"
    except ValueError as e:
        assert "different subgraph" in str(e)

    # Change node_1 <-> node_3 link to node_1 <-> node_2 (input for subgraph_1)
    graph.links[0].sink_id = "node_2"
    created_graph = await server.agent_server.create_graph(
        create_graph, False, DEFAULT_USER_ID
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
    assert links[0].source_id != links[0].sink_id
    assert links[0].source_id in {nodes[0].id, nodes[1].id, nodes[2].id}
    assert links[0].sink_id in {nodes[0].id, nodes[1].id, nodes[2].id}

    assert len(created_graph.subgraphs) == 1
    assert len(created_graph.subgraph_map) == len(created_graph.nodes) == 3
