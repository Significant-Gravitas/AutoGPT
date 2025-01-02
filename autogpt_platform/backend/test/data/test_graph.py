from typing import Any
from uuid import UUID

import pytest

from backend.blocks.basic import AgentInputBlock, AgentOutputBlock, StoreValueBlock
from backend.data.block import BlockSchema
from backend.data.graph import Graph, Link, Node
from backend.data.model import SchemaField
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
            Node(id="node_2", block_id=input_block, input_default={"name": "input"}),
            Node(id="node_3", block_id=value_block),
        ],
        links=[
            Link(
                source_id="node_1",
                sink_id="node_2",
                source_name="output",
                sink_name="name",
            ),
        ],
    )
    create_graph = CreateGraph(graph=graph)
    created_graph = await server.agent_server.test_create_graph(
        create_graph, DEFAULT_USER_ID
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
                id="node_0_a",
                block_id=input_block,
                input_default={
                    "name": "in_key_a",
                    "title": "Key A",
                    "value": "A",
                    "advanced": True,
                },
                metadata={"id": "node_0_a"},
            ),
            Node(
                id="node_0_b",
                block_id=input_block,
                input_default={"name": "in_key_b", "advanced": True},
                metadata={"id": "node_0_b"},
            ),
            Node(id="node_1", block_id=value_block, metadata={"id": "node_1"}),
            Node(
                id="node_2",
                block_id=output_block,
                input_default={
                    "name": "out_key",
                    "description": "This is an output key",
                },
                metadata={"id": "node_2"},
            ),
        ],
        links=[
            Link(
                source_id="node_0_a",
                sink_id="node_1",
                source_name="result",
                sink_name="input",
            ),
            Link(
                source_id="node_0_b",
                sink_id="node_1",
                source_name="result",
                sink_name="input",
            ),
            Link(
                source_id="node_1",
                sink_id="node_2",
                source_name="output",
                sink_name="value",
            ),
        ],
    )

    create_graph = CreateGraph(graph=graph)
    created_graph = await server.agent_server.test_create_graph(
        create_graph, DEFAULT_USER_ID
    )

    class ExpectedInputSchema(BlockSchema):
        in_key_a: Any = SchemaField(title="Key A", default="A", advanced=True)
        in_key_b: Any = SchemaField(title="in_key_b", advanced=False)

    class ExpectedOutputSchema(BlockSchema):
        out_key: Any = SchemaField(
            description="This is an output key",
            title="out_key",
            advanced=False,
        )

    input_schema = created_graph.input_schema
    input_schema["title"] = "ExpectedInputSchema"
    assert input_schema == ExpectedInputSchema.jsonschema()

    output_schema = created_graph.output_schema
    output_schema["title"] = "ExpectedOutputSchema"
    assert output_schema == ExpectedOutputSchema.jsonschema()


@pytest.mark.asyncio(scope="session")
async def test_clean_graph(server: SpinTestServer):
    """
    Test the clean_graph function that:
    1. Clears input block values
    2. Removes credentials from nodes
    """
    # Create a graph with input blocks and credentials
    graph = Graph(
        id="test_clean_graph",
        name="Test Clean Graph",
        description="Test graph cleaning",
        nodes=[
            Node(
                id="input_node",
                block_id=AgentInputBlock().id,
                input_default={
                    "name": "test_input",
                    "value": "test value",
                    "description": "Test input description",
                },
            ),
        ],
        links=[],
    )

    # Create graph and get model
    create_graph = CreateGraph(graph=graph)
    created_graph = await server.agent_server.test_create_graph(
        create_graph, DEFAULT_USER_ID
    )

    # Clean the graph
    created_graph.clean_graph()

    # # Verify input block value is cleared
    input_node = next(
        n for n in created_graph.nodes if n.block_id == AgentInputBlock().id
    )
    assert input_node.input_default["value"] == ""
