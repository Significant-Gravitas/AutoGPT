from uuid import UUID

import pytest

from backend.blocks.basic import AgentInputBlock, AgentOutputBlock, StoreValueBlock
from backend.data.graph import Graph, Link, Node
from backend.data.user import DEFAULT_USER_ID
from backend.server.model import CreateGraph
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(scope="session")
async def test_graph_creation(server: SpinTestServer):
    """
    Test the creation of a graph with nodes and links.

    This test ensures that:
    1. A graph can be successfully created with valid connections.
    2. The created graph has the correct structure and properties.

    Args:
        server (SpinTestServer): The test server instance.
    """
    value_block = StoreValueBlock().id
    input_block = AgentInputBlock().id

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
                sink_id="node_2",
                source_name="output",
                sink_name="input",
            ),
        ],
    )
    create_graph = CreateGraph(graph=graph)
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


@pytest.mark.asyncio(scope="session")
async def test_get_input_schema(server: SpinTestServer):
    """
    Test the get_input_schema method of a created graph.

    This test ensures that:
    1. A graph can be created with a single node.
    2. The input schema of the created graph is correctly generated.
    3. The input schema contains the expected input name and node id.

    Args:
        server (SpinTestServer): The test server instance.
    """
    value_block = StoreValueBlock().id
    input_block = AgentInputBlock().id
    output_block = AgentOutputBlock().id

    graph = Graph(
        name="TestInputSchema",
        description="Test input schema",
        nodes=[
            Node(
                id="node_0",
                block_id=input_block,
                input_default={"name": "in_key", "title": "Input Key"},
            ),
            Node(id="node_1", block_id=value_block),
            Node(
                id="node_2",
                block_id=output_block,
                input_default={
                    "name": "out_key",
                    "description": "This is an output key",
                },
            ),
        ],
        links=[
            Link(
                source_id="node_0",
                sink_id="node_1",
                source_name="output",
                sink_name="input",
            ),
            Link(
                source_id="node_1",
                sink_id="node_2",
                source_name="output",
                sink_name="input",
            ),
        ],
    )

    create_graph = CreateGraph(graph=graph)
    created_graph = await server.agent_server.create_graph(
        create_graph, False, DEFAULT_USER_ID
    )

    input_schema = created_graph.input_schema
    assert len(input_schema) == 1
    assert input_schema["in_key"].node_id == created_graph.nodes[0].id
    assert input_schema["in_key"].title == "Input Key"

    output_schema = created_graph.output_schema
    assert len(output_schema) == 1
    assert output_schema["out_key"].node_id == created_graph.nodes[2].id
    assert output_schema["out_key"].description == "This is an output key"
