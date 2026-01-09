import json
from typing import Any
from uuid import UUID

import fastapi.exceptions
import pytest
from pytest_snapshot.plugin import Snapshot

import backend.api.features.store.model as store
from backend.api.model import CreateGraph
from backend.blocks.basic import StoreValueBlock
from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.data.block import BlockSchema, BlockSchemaInput
from backend.data.graph import Graph, Link, Node
from backend.data.model import SchemaField
from backend.data.user import DEFAULT_USER_ID
from backend.usecases.sample import create_test_user
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_graph_creation(server: SpinTestServer, snapshot: Snapshot):
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

    # Create a serializable version of the graph for snapshot testing
    # Remove dynamic IDs to make snapshots reproducible
    graph_data = {
        "name": created_graph.name,
        "description": created_graph.description,
        "nodes_count": len(created_graph.nodes),
        "links_count": len(created_graph.links),
        "node_blocks": [node.block_id for node in created_graph.nodes],
        "link_structure": [
            {"source_name": link.source_name, "sink_name": link.sink_name}
            for link in created_graph.links
        ],
    }
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(graph_data, indent=2, sort_keys=True), "grph_struct"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_get_input_schema(server: SpinTestServer, snapshot: Snapshot):
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

    class ExpectedInputSchema(BlockSchemaInput):
        in_key_a: Any = SchemaField(title="Key A", default="A", advanced=True)
        in_key_b: Any = SchemaField(title="in_key_b", advanced=False)

    class ExpectedOutputSchema(BlockSchema):
        # Note: Graph output schemas are dynamically generated and don't inherit
        # from BlockSchemaOutput, so we use BlockSchema as the base instead
        out_key: Any = SchemaField(
            description="This is an output key",
            title="out_key",
            advanced=False,
        )

    input_schema = created_graph.input_schema
    input_schema["title"] = "ExpectedInputSchema"
    assert input_schema == ExpectedInputSchema.jsonschema()

    # Add snapshot testing for the schemas
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(input_schema, indent=2, sort_keys=True), "grph_in_schm"
    )

    output_schema = created_graph.output_schema
    output_schema["title"] = "ExpectedOutputSchema"
    assert output_schema == ExpectedOutputSchema.jsonschema()

    # Add snapshot testing for the output schema
    snapshot.snapshot_dir = "snapshots"
    snapshot.assert_match(
        json.dumps(output_schema, indent=2, sort_keys=True), "grph_out_schm"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_clean_graph(server: SpinTestServer):
    """
    Test the stripped_for_export function that:
    1. Removes sensitive/secret fields from node inputs
    2. Removes webhook information
    3. Preserves non-sensitive data including input block values
    """
    # Create a graph with input blocks containing both sensitive and normal data
    graph = Graph(
        id="test_clean_graph",
        name="Test Clean Graph",
        description="Test graph cleaning",
        nodes=[
            Node(
                block_id=AgentInputBlock().id,
                input_default={
                    "_test_id": "input_node",
                    "name": "test_input",
                    "value": "test value",  # This should be preserved
                    "description": "Test input description",
                },
            ),
            Node(
                block_id=AgentInputBlock().id,
                input_default={
                    "_test_id": "input_node_secret",
                    "name": "secret_input",
                    "value": "another value",
                    "secret": True,  # This makes the input secret
                },
            ),
            Node(
                block_id=StoreValueBlock().id,
                input_default={
                    "_test_id": "node_with_secrets",
                    "input": "normal_value",
                    "control_test_input": "should be preserved",
                    "api_key": "secret_api_key_123",  # Should be filtered
                    "password": "secret_password_456",  # Should be filtered
                    "token": "secret_token_789",  # Should be filtered
                    "credentials": {  # Should be filtered
                        "id": "fake-github-credentials-id",
                        "provider": "github",
                        "type": "api_key",
                    },
                    "anthropic_credentials": {  # Should be filtered
                        "id": "fake-anthropic-credentials-id",
                        "provider": "anthropic",
                        "type": "api_key",
                    },
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
    cleaned_graph = await server.agent_server.test_get_graph(
        created_graph.id, created_graph.version, DEFAULT_USER_ID, for_export=True
    )

    # Verify sensitive fields are removed but normal fields are preserved
    input_node = next(
        n for n in cleaned_graph.nodes if n.input_default["_test_id"] == "input_node"
    )

    # Non-sensitive fields should be preserved
    assert input_node.input_default["name"] == "test_input"
    assert input_node.input_default["value"] == "test value"  # Should be preserved now
    assert input_node.input_default["description"] == "Test input description"

    # Sensitive fields should be filtered out
    assert "api_key" not in input_node.input_default
    assert "password" not in input_node.input_default

    # Verify secret input node preserves non-sensitive fields but removes secret value
    secret_node = next(
        n
        for n in cleaned_graph.nodes
        if n.input_default["_test_id"] == "input_node_secret"
    )
    assert secret_node.input_default["name"] == "secret_input"
    assert "value" not in secret_node.input_default  # Secret default should be removed
    assert secret_node.input_default["secret"] is True

    # Verify sensitive fields are filtered from nodes with secrets
    secrets_node = next(
        n
        for n in cleaned_graph.nodes
        if n.input_default["_test_id"] == "node_with_secrets"
    )
    # Normal fields should be preserved
    assert secrets_node.input_default["input"] == "normal_value"
    assert secrets_node.input_default["control_test_input"] == "should be preserved"
    # Sensitive fields should be filtered out
    assert "api_key" not in secrets_node.input_default
    assert "password" not in secrets_node.input_default
    assert "token" not in secrets_node.input_default
    assert "credentials" not in secrets_node.input_default
    assert "anthropic_credentials" not in secrets_node.input_default

    # Verify webhook info is removed (if any nodes had it)
    for node in cleaned_graph.nodes:
        assert node.webhook_id is None
        assert node.webhook is None


@pytest.mark.asyncio(loop_scope="session")
async def test_access_store_listing_graph(server: SpinTestServer):
    """
    Test the access of a store listing graph.
    """
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

    store_submission_request = store.StoreSubmissionRequest(
        agent_id=created_graph.id,
        agent_version=created_graph.version,
        slug=created_graph.id,
        name="Test name",
        sub_heading="Test sub heading",
        video_url=None,
        image_urls=[],
        description="Test description",
        categories=[],
    )

    # First we check the graph an not be accessed by a different user
    with pytest.raises(fastapi.exceptions.HTTPException) as exc_info:
        await server.agent_server.test_get_graph(
            created_graph.id,
            created_graph.version,
            "3e53486c-cf57-477e-ba2a-cb02dc828e1b",
        )
    assert exc_info.value.status_code == 404
    assert "Graph" in str(exc_info.value.detail)

    # Now we create a store listing
    store_listing = await server.agent_server.test_create_store_listing(
        store_submission_request, DEFAULT_USER_ID
    )

    if isinstance(store_listing, fastapi.responses.JSONResponse):
        assert False, "Failed to create store listing"

    slv_id = (
        store_listing.store_listing_version_id
        if store_listing.store_listing_version_id is not None
        else None
    )

    assert slv_id is not None

    admin_user = await create_test_user(alt_user=True)
    await server.agent_server.test_review_store_listing(
        store.ReviewSubmissionRequest(
            store_listing_version_id=slv_id,
            is_approved=True,
            comments="Test comments",
        ),
        user_id=admin_user.id,
    )

    # Now we check the graph can be accessed by a user that does not own the graph
    got_graph = await server.agent_server.test_get_graph(
        created_graph.id, created_graph.version, "3e53486c-cf57-477e-ba2a-cb02dc828e1b"
    )
    assert got_graph is not None


# ============================================================================
# Tests for Optional Credentials Feature
# ============================================================================


def test_node_credentials_optional_default():
    """Test that credentials_optional defaults to False when not set in metadata."""
    node = Node(
        id="test_node",
        block_id=StoreValueBlock().id,
        input_default={},
        metadata={},
    )
    assert node.credentials_optional is False


def test_node_credentials_optional_true():
    """Test that credentials_optional returns True when explicitly set."""
    node = Node(
        id="test_node",
        block_id=StoreValueBlock().id,
        input_default={},
        metadata={"credentials_optional": True},
    )
    assert node.credentials_optional is True


def test_node_credentials_optional_false():
    """Test that credentials_optional returns False when explicitly set to False."""
    node = Node(
        id="test_node",
        block_id=StoreValueBlock().id,
        input_default={},
        metadata={"credentials_optional": False},
    )
    assert node.credentials_optional is False


def test_node_credentials_optional_with_other_metadata():
    """Test that credentials_optional works correctly with other metadata present."""
    node = Node(
        id="test_node",
        block_id=StoreValueBlock().id,
        input_default={},
        metadata={
            "position": {"x": 100, "y": 200},
            "customized_name": "My Custom Node",
            "credentials_optional": True,
        },
    )
    assert node.credentials_optional is True
    assert node.metadata["position"] == {"x": 100, "y": 200}
    assert node.metadata["customized_name"] == "My Custom Node"
