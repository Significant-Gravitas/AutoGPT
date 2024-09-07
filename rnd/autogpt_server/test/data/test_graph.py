from uuid import UUID

import pytest

from autogpt_server.blocks.basic import InputBlock, StoreValueBlock
from autogpt_server.data.graph import Graph, Link, Node
from autogpt_server.data.user import DEFAULT_USER_ID, create_default_user
from autogpt_server.server.model import CreateGraph
from autogpt_server.util.test import SpinTestServer


@pytest.mark.asyncio(scope="session")
async def test_graph_creation(server: SpinTestServer):
    """
    Test the creation of a graph with nodes and links.

    This test ensures that:
    1. Nodes from different subgraphs cannot be directly connected.
    2. A graph can be successfully created with valid connections.
    3. The created graph has the correct structure and properties.

    Args:
        server (SpinTestServer): The test server instance.
    """
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

    graph = Graph(
        name="TestInputSchema",
        description="Test input schema",
        nodes=[
            Node(id="node_1", block_id=value_block),
        ],
        links=[],
    )

    create_graph = CreateGraph(graph=graph)
    created_graph = await server.agent_server.create_graph(
        create_graph, False, DEFAULT_USER_ID
    )

    input_schema = created_graph.get_input_schema()

    assert len(input_schema) == 1

    assert input_schema[0].title == "Input"
    assert input_schema[0].node_id == created_graph.nodes[0].id


@pytest.mark.asyncio(scope="session")
async def test_get_input_schema_none_required(server: SpinTestServer):
    """
    Test the get_input_schema method when no inputs are required.

    This test ensures that:
    1. A graph can be created with a node that has a default input value.
    2. The input schema of the created graph is empty when all inputs have default values.

    Args:
        server (SpinTestServer): The test server instance.
    """
    value_block = StoreValueBlock().id

    graph = Graph(
        name="TestInputSchema",
        description="Test input schema",
        nodes=[
            Node(id="node_1", block_id=value_block, input_default={"input": "value"}),
        ],
        links=[],
    )

    create_graph = CreateGraph(graph=graph)
    created_graph = await server.agent_server.create_graph(
        create_graph, False, DEFAULT_USER_ID
    )

    input_schema = created_graph.get_input_schema()

    assert input_schema == []


@pytest.mark.asyncio(scope="session")
async def test_get_input_schema_with_linked_blocks(server: SpinTestServer):
    """
    Test the get_input_schema method with linked blocks.

    This test ensures that:
    1. A graph can be created with multiple nodes and links between them.
    2. The input schema correctly identifies required inputs for linked blocks.
    3. Inputs that are satisfied by links are not included in the input schema.

    Args:
        server (SpinTestServer): The test server instance.
    """
    value_block = StoreValueBlock().id

    graph = Graph(
        name="TestInputSchemaLinkedBlocks",
        description="Test input schema with linked blocks",
        nodes=[
            Node(id="node_1", block_id=value_block),
            Node(id="node_2", block_id=value_block),
        ],
        links=[
            Link(
                source_id="node_1",
                sink_id="node_2",
                source_name="output",
                sink_name="data",
            ),
        ],
    )

    create_graph = CreateGraph(graph=graph)
    created_graph = await server.agent_server.create_graph(
        create_graph, False, DEFAULT_USER_ID
    )

    input_schema = created_graph.get_input_schema()

    assert len(input_schema) == 2

    node_1_input = next(
        (item for item in input_schema if item.node_id == created_graph.nodes[0].id),
        None,
    )
    node_2_input = next(
        (item for item in input_schema if item.node_id == created_graph.nodes[1].id),
        None,
    )

    assert node_1_input is not None
    assert node_2_input is not None
    assert node_1_input.title == "Input"
    assert node_2_input.title == "Input"

    assert not any(
        item.title == "data" and item.node_id == created_graph.nodes[1].id
        for item in input_schema
    )
