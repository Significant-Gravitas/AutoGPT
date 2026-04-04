import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import fastapi.exceptions
import prisma
import pytest
from pytest_snapshot.plugin import Snapshot

import backend.api.features.store.model as store
from backend.api.model import CreateGraph
from backend.blocks._base import BlockSchema, BlockSchemaInput
from backend.blocks.basic import StoreValueBlock
from backend.blocks.io import AgentInputBlock, AgentOutputBlock
from backend.data.graph import (
    Graph,
    Link,
    Node,
    get_graph,
    validate_graph_execution_permissions,
)
from backend.data.model import SchemaField
from backend.data.user import DEFAULT_USER_ID
from backend.usecases.sample import create_test_user
from backend.util.exceptions import GraphNotAccessibleError, GraphNotInLibraryError
from backend.util.test import SpinTestServer


@pytest.fixture(scope="session", autouse=True)
def mock_embedding_functions():
    """Mock embedding functions for all tests to avoid database/API dependencies."""
    with patch(
        "backend.api.features.store.db.ensure_embedding",
        new_callable=AsyncMock,
        return_value=True,
    ):
        yield


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
                    "api_key": "secret_api_key_123",  # Should be filtered # pragma: allowlist secret # noqa
                    "password": "secret_password_456",  # Should be filtered # pragma: allowlist secret # noqa
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

    # Ensure the default user has a Profile (required for store submissions)
    existing_profile = await prisma.models.Profile.prisma().find_first(
        where={"userId": DEFAULT_USER_ID}
    )
    if not existing_profile:
        await prisma.models.Profile.prisma().create(
            data=prisma.types.ProfileCreateInput(
                userId=DEFAULT_USER_ID,
                name="Default User",
                username=f"default-user-{DEFAULT_USER_ID[:8]}",
                description="Default test user profile",
                links=[],
            )
        )

    store_submission_request = store.StoreSubmissionRequest(
        graph_id=created_graph.id,
        graph_version=created_graph.version,
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
        store_listing.listing_version_id
        if store_listing.listing_version_id is not None
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


# ============================================================================
# Tests for MCP Credential Deduplication
# ============================================================================


def test_mcp_credential_combine_different_servers():
    """Two MCP credential fields with different server URLs should produce
    separate entries when combined (not merged into one)."""
    from backend.data.model import CredentialsFieldInfo, CredentialsType
    from backend.integrations.providers import ProviderName

    oauth2_types: frozenset[CredentialsType] = frozenset(["oauth2"])

    field_sentry = CredentialsFieldInfo(
        credentials_provider=frozenset([ProviderName.MCP]),
        credentials_types=oauth2_types,
        credentials_scopes=None,
        discriminator="server_url",
        discriminator_values={"https://mcp.sentry.dev/mcp"},
    )
    field_linear = CredentialsFieldInfo(
        credentials_provider=frozenset([ProviderName.MCP]),
        credentials_types=oauth2_types,
        credentials_scopes=None,
        discriminator="server_url",
        discriminator_values={"https://mcp.linear.app/mcp"},
    )

    combined = CredentialsFieldInfo.combine(
        (field_sentry, ("node-sentry", "credentials")),
        (field_linear, ("node-linear", "credentials")),
    )

    # Should produce 2 separate credential entries
    assert len(combined) == 2, (
        f"Expected 2 credential entries for 2 MCP blocks with different servers, "
        f"got {len(combined)}: {list(combined.keys())}"
    )

    # Each entry should contain the server hostname in its key
    keys = list(combined.keys())
    assert any(
        "mcp.sentry.dev" in k for k in keys
    ), f"Expected 'mcp.sentry.dev' in one key, got {keys}"
    assert any(
        "mcp.linear.app" in k for k in keys
    ), f"Expected 'mcp.linear.app' in one key, got {keys}"


def test_mcp_credential_combine_same_server():
    """Two MCP credential fields with the same server URL should be combined
    into one credential entry."""
    from backend.data.model import CredentialsFieldInfo, CredentialsType
    from backend.integrations.providers import ProviderName

    oauth2_types: frozenset[CredentialsType] = frozenset(["oauth2"])

    field_a = CredentialsFieldInfo(
        credentials_provider=frozenset([ProviderName.MCP]),
        credentials_types=oauth2_types,
        credentials_scopes=None,
        discriminator="server_url",
        discriminator_values={"https://mcp.sentry.dev/mcp"},
    )
    field_b = CredentialsFieldInfo(
        credentials_provider=frozenset([ProviderName.MCP]),
        credentials_types=oauth2_types,
        credentials_scopes=None,
        discriminator="server_url",
        discriminator_values={"https://mcp.sentry.dev/mcp"},
    )

    combined = CredentialsFieldInfo.combine(
        (field_a, ("node-a", "credentials")),
        (field_b, ("node-b", "credentials")),
    )

    # Should produce 1 credential entry (same server URL)
    assert len(combined) == 1, (
        f"Expected 1 credential entry for 2 MCP blocks with same server, "
        f"got {len(combined)}: {list(combined.keys())}"
    )


def test_mcp_credential_combine_no_discriminator_values():
    """MCP credential fields without discriminator_values should be merged
    into a single entry (backwards compat for blocks without server_url set)."""
    from backend.data.model import CredentialsFieldInfo, CredentialsType
    from backend.integrations.providers import ProviderName

    oauth2_types: frozenset[CredentialsType] = frozenset(["oauth2"])

    field_a = CredentialsFieldInfo(
        credentials_provider=frozenset([ProviderName.MCP]),
        credentials_types=oauth2_types,
        credentials_scopes=None,
        discriminator="server_url",
    )
    field_b = CredentialsFieldInfo(
        credentials_provider=frozenset([ProviderName.MCP]),
        credentials_types=oauth2_types,
        credentials_scopes=None,
        discriminator="server_url",
    )

    combined = CredentialsFieldInfo.combine(
        (field_a, ("node-a", "credentials")),
        (field_b, ("node-b", "credentials")),
    )

    # Should produce 1 entry (no URL differentiation)
    assert len(combined) == 1, (
        f"Expected 1 credential entry for MCP blocks without discriminator_values, "
        f"got {len(combined)}: {list(combined.keys())}"
    )


# --------------- get_graph access-control truth table --------------- #
#
# Full matrix of access scenarios for get_graph() and get_graph_as_admin().
# Access priority: ownership > marketplace APPROVED > library membership.
# Library is version-specific. get_graph_as_admin bypasses everything.
#
# | User     | Owns? | Marketplace | Library          | Version | Result  | Test
# |----------|-------|-------------|------------------|---------|---------|-----
# | regular  | yes   | any         | any              | v1      | ACCESS  | test_get_graph_library_not_queried_when_owned
# | regular  | no    | APPROVED    | any              | v1      | ACCESS  | test_get_graph_non_owner_approved_marketplace_agent
# | regular  | no    | not listed  | active, same ver | v1      | ACCESS  | test_get_graph_library_member_can_access_unpublished
# | regular  | no    | not listed  | active, diff ver | v2      | DENIED  | test_get_graph_library_wrong_version_denied
# | regular  | no    | not listed  | deleted          | v1      | DENIED  | test_get_graph_deleted_library_agent_denied
# | regular  | no    | not listed  | archived         | v1      | DENIED  | test_get_graph_archived_library_agent_denied
# | regular  | no    | not listed  | not present      | v1      | DENIED  | test_get_graph_non_owner_pending_not_in_library_denied
# | regular  | no    | PENDING     | active v1        | v2      | DENIED  | test_library_v1_does_not_grant_access_to_pending_v2
# | regular  | no    | not listed  | null AgentGraph  | v1      | DENIED  | test_get_graph_library_with_null_agent_graph_denied
# | anon     | no    | not listed  | -                | v1      | DENIED  | test_get_graph_library_fallback_not_used_for_anonymous
# | anon     | no    | APPROVED    | -                | v1      | ACCESS  | test_get_graph_anonymous_approved_marketplace_access
# | admin*   | no    | PENDING     | -                | v2      | ACCESS  | test_admin_can_access_pending_v2_via_get_graph_as_admin
#
# Efficiency (no unnecessary queries):
# | regular  | yes   | -           | -                | v1      | no mkt/lib | test_get_graph_library_not_queried_when_owned
# | regular  | no    | APPROVED    | -                | v1      | no lib     | test_get_graph_library_not_queried_when_marketplace_approved
#
# * = via get_graph_as_admin (admin-only routes)


def _make_mock_db_graph(user_id: str = "owner-user-id") -> MagicMock:
    graph = MagicMock()
    graph.userId = user_id
    graph.id = "graph-id"
    graph.version = 1
    graph.Nodes = []
    return graph


@pytest.mark.asyncio
async def test_get_graph_non_owner_approved_marketplace_agent() -> None:
    """A non-owner should be able to access a graph that has an APPROVED
    marketplace listing.  This is the normal marketplace download flow."""
    owner_id = "owner-user-id"
    requester_id = "different-user-id"
    graph_id = "graph-id"
    mock_graph = _make_mock_db_graph(owner_id)
    mock_graph_model = MagicMock(name="GraphModel")

    mock_listing = MagicMock()
    mock_listing.AgentGraph = mock_graph

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch(
            "backend.data.graph.GraphModel.from_db",
            return_value=mock_graph_model,
        ),
    ):
        # First lookup (owned graph) returns None — requester != owner
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=None)
        # Marketplace fallback finds an APPROVED listing
        mock_slv_prisma.return_value.find_first = AsyncMock(return_value=mock_listing)

        result = await get_graph(
            graph_id=graph_id,
            version=1,
            user_id=requester_id,
        )

    assert result is not None, "Non-owner should access APPROVED marketplace agent"


@pytest.mark.asyncio
async def test_get_graph_non_owner_pending_not_in_library_denied() -> None:
    """A non-owner with no library membership and no APPROVED marketplace
    listing must be denied access."""
    requester_id = "different-user-id"
    graph_id = "graph-id"

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
    ):
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_slv_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=None)

        result = await get_graph(
            graph_id=graph_id,
            version=1,
            user_id=requester_id,
        )

    assert (
        result is None
    ), "User without ownership, marketplace, or library access must be denied"


# --------------- Library membership grants graph access --------------- #
# "You added it, you keep it" — product decision from SECRT-2167.


@pytest.mark.asyncio
async def test_get_graph_library_member_can_access_unpublished() -> None:
    """A user who has the agent in their library should be able to access it
    even if it's no longer published in the marketplace."""
    requester_id = "library-user-id"
    graph_id = "graph-id"
    mock_graph = _make_mock_db_graph("original-creator-id")
    mock_graph_model = MagicMock(name="GraphModel")

    mock_library_agent = MagicMock()
    mock_library_agent.AgentGraph = mock_graph

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.GraphModel.from_db",
            return_value=mock_graph_model,
        ),
    ):
        # Not owned
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=None)
        # Not in marketplace (unpublished)
        mock_slv_prisma.return_value.find_first = AsyncMock(return_value=None)
        # But IS in user's library
        mock_lib_prisma.return_value.find_first = AsyncMock(
            return_value=mock_library_agent
        )

        result = await get_graph(
            graph_id=graph_id,
            version=1,
            user_id=requester_id,
        )

    assert result is mock_graph_model, "Library member should access unpublished agent"

    # Verify library query filters on non-deleted, non-archived
    lib_call = mock_lib_prisma.return_value.find_first
    lib_call.assert_awaited_once()
    assert lib_call.await_args is not None
    lib_where = lib_call.await_args.kwargs["where"]
    assert lib_where["userId"] == requester_id
    assert lib_where["agentGraphId"] == graph_id
    assert lib_where["isDeleted"] is False
    assert lib_where["isArchived"] is False


@pytest.mark.asyncio
async def test_get_graph_deleted_library_agent_denied() -> None:
    """If the user soft-deleted the agent from their library, they should
    NOT get access via the library fallback."""
    requester_id = "library-user-id"
    graph_id = "graph-id"

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
    ):
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_slv_prisma.return_value.find_first = AsyncMock(return_value=None)
        # Library query returns None because isDeleted=False filter excludes it
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=None)

        result = await get_graph(
            graph_id=graph_id,
            version=1,
            user_id=requester_id,
        )

    assert result is None, "Deleted library agent should not grant graph access"


@pytest.mark.asyncio
async def test_get_graph_anonymous_approved_marketplace_access() -> None:
    """Anonymous users (user_id=None) should still access APPROVED marketplace
    agents — the marketplace fallback doesn't require authentication."""
    graph_id = "graph-id"
    mock_graph = _make_mock_db_graph("creator-id")
    mock_graph_model = MagicMock(name="GraphModel")

    mock_listing = MagicMock()
    mock_listing.AgentGraph = mock_graph

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch(
            "backend.data.graph.GraphModel.from_db",
            return_value=mock_graph_model,
        ),
    ):
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_slv_prisma.return_value.find_first = AsyncMock(return_value=mock_listing)

        result = await get_graph(
            graph_id=graph_id,
            version=1,
            user_id=None,
        )

    assert (
        result is mock_graph_model
    ), "Anonymous user should access APPROVED marketplace agent"


@pytest.mark.asyncio
async def test_get_graph_library_fallback_not_used_for_anonymous() -> None:
    """Anonymous requests (user_id=None) must not trigger the library
    fallback — there's no user to check library membership for."""
    graph_id = "graph-id"

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
    ):
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_slv_prisma.return_value.find_first = AsyncMock(return_value=None)

        result = await get_graph(
            graph_id=graph_id,
            version=1,
            user_id=None,
        )

    assert result is None
    # Library should never be queried for anonymous users
    mock_lib_prisma.return_value.find_first.assert_not_called()


@pytest.mark.asyncio
async def test_get_graph_library_not_queried_when_owned() -> None:
    """If the user owns the graph, the library fallback should NOT be
    triggered — ownership is sufficient."""
    owner_id = "owner-user-id"
    graph_id = "graph-id"
    mock_graph = _make_mock_db_graph(owner_id)
    mock_graph_model = MagicMock(name="GraphModel")

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.GraphModel.from_db",
            return_value=mock_graph_model,
        ),
    ):
        # User owns the graph — first lookup succeeds
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=mock_graph)

        result = await get_graph(
            graph_id=graph_id,
            version=1,
            user_id=owner_id,
        )

    assert result is mock_graph_model
    # Neither marketplace nor library should be queried
    mock_slv_prisma.return_value.find_first.assert_not_called()
    mock_lib_prisma.return_value.find_first.assert_not_called()


@pytest.mark.asyncio
async def test_get_graph_library_not_queried_when_marketplace_approved() -> None:
    """If the graph is APPROVED in the marketplace, the library fallback
    should NOT be triggered — marketplace access is sufficient."""
    requester_id = "different-user-id"
    graph_id = "graph-id"
    mock_graph = _make_mock_db_graph("original-creator-id")
    mock_graph_model = MagicMock(name="GraphModel")

    mock_listing = MagicMock()
    mock_listing.AgentGraph = mock_graph

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.GraphModel.from_db",
            return_value=mock_graph_model,
        ),
    ):
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_slv_prisma.return_value.find_first = AsyncMock(return_value=mock_listing)

        result = await get_graph(
            graph_id=graph_id,
            version=1,
            user_id=requester_id,
        )

    assert result is mock_graph_model
    # Library should not be queried — marketplace was sufficient
    mock_lib_prisma.return_value.find_first.assert_not_called()


@pytest.mark.asyncio
async def test_get_graph_archived_library_agent_denied() -> None:
    """If the user archived the agent in their library, they should
    NOT get access via the library fallback."""
    requester_id = "library-user-id"
    graph_id = "graph-id"

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
    ):
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_slv_prisma.return_value.find_first = AsyncMock(return_value=None)
        # Library query returns None because isArchived=False filter excludes it
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=None)

        result = await get_graph(
            graph_id=graph_id,
            version=1,
            user_id=requester_id,
        )

    assert result is None, "Archived library agent should not grant graph access"


@pytest.mark.asyncio
async def test_get_graph_library_with_null_agent_graph_denied() -> None:
    """If LibraryAgent exists but its AgentGraph relation is None
    (data integrity issue), access must be denied, not crash."""
    requester_id = "library-user-id"
    graph_id = "graph-id"

    mock_library_agent = MagicMock()
    mock_library_agent.AgentGraph = None  # broken relation

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
    ):
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_slv_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_lib_prisma.return_value.find_first = AsyncMock(
            return_value=mock_library_agent
        )

        result = await get_graph(
            graph_id=graph_id,
            version=1,
            user_id=requester_id,
        )

    assert (
        result is None
    ), "Library agent with missing graph relation should not grant access"


@pytest.mark.asyncio
async def test_get_graph_library_wrong_version_denied() -> None:
    """Having version 1 in your library must NOT grant access to version 2."""
    requester_id = "library-user-id"
    graph_id = "graph-id"

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
    ):
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=None)
        mock_slv_prisma.return_value.find_first = AsyncMock(return_value=None)
        # Library has version 1 but we're requesting version 2 —
        # the where clause includes agentGraphVersion so this returns None
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=None)

        result = await get_graph(
            graph_id=graph_id,
            version=2,
            user_id=requester_id,
        )

    assert (
        result is None
    ), "Library agent for version 1 must not grant access to version 2"
    # Verify version was included in the library query
    lib_call = mock_lib_prisma.return_value.find_first
    lib_call.assert_called_once()
    lib_where = lib_call.call_args.kwargs["where"]
    assert lib_where["agentGraphVersion"] == 2


@pytest.mark.asyncio
async def test_library_v1_does_not_grant_access_to_pending_v2() -> None:
    """A regular user has v1 in their library. v2 is pending (not approved).
    They must NOT get access to v2 — library membership is version-specific."""
    requester_id = "regular-user-id"
    graph_id = "graph-id"

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch(
            "backend.data.graph.StoreListingVersion.prisma",
        ) as mock_slv_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
    ):
        # Not owned
        mock_ag_prisma.return_value.find_first = AsyncMock(return_value=None)
        # v2 is not APPROVED in marketplace
        mock_slv_prisma.return_value.find_first = AsyncMock(return_value=None)
        # Library has v1 but not v2 — version filter excludes it
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=None)

        result = await get_graph(
            graph_id=graph_id,
            version=2,
            user_id=requester_id,
        )

    assert result is None, "Regular user with v1 in library must not access pending v2"


@pytest.mark.asyncio
async def test_admin_can_access_pending_v2_via_get_graph_as_admin() -> None:
    """An admin can access v2 (pending) via get_graph_as_admin even though
    only v1 is approved. get_graph_as_admin bypasses all access checks."""
    from backend.data.graph import get_graph_as_admin

    admin_id = "admin-user-id"
    mock_graph = _make_mock_db_graph("creator-user-id")
    mock_graph.version = 2
    mock_graph_model = MagicMock(name="GraphModel")

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_prisma,
        patch(
            "backend.data.graph.GraphModel.from_db",
            return_value=mock_graph_model,
        ),
    ):
        mock_prisma.return_value.find_first = AsyncMock(return_value=mock_graph)

        result = await get_graph_as_admin(
            graph_id="graph-id",
            version=2,
            user_id=admin_id,
            for_export=False,
        )

    assert (
        result is mock_graph_model
    ), "Admin must access pending v2 via get_graph_as_admin"


# --------------- execution permission truth table --------------- #
#
# validate_graph_execution_permissions() has two gates:
# 1. Accessible graph: owner OR exact-version library entry OR marketplace-published
# 2. Runnable graph: exact-version library entry OR owner fallback to any live
#    library entry for the graph OR sub-graph exception
#
# Desired owner behavior differs from non-owners:
# owners should be allowed to run a new version when some non-archived/non-deleted
# version of that graph is still in their library. Non-owners stay
# version-specific.
#
# | User     | Owns? | Marketplace | Library state                | is_sub_graph | Result   | Test
# |----------|-------|-------------|------------------------------|--------------|----------|-----
# | regular  | no    | no          | exact version present        | false        | ALLOW    | test_validate_graph_execution_permissions_library_member_same_version_allowed
# | owner    | yes   | no          | exact version present        | false        | ALLOW    | test_validate_graph_execution_permissions_owner_same_version_in_library_allowed
# | owner    | yes   | no          | previous version present     | false        | ALLOW    | test_validate_graph_execution_permissions_owner_previous_library_version_allowed
# | owner    | yes   | no          | none present                 | false        | DENY lib | test_validate_graph_execution_permissions_owner_without_library_denied
# | owner    | yes   | no          | only archived/deleted older  | false        | DENY lib | test_validate_graph_execution_permissions_owner_previous_archived_library_version_denied
# | regular  | no    | yes         | none present                 | false        | DENY lib | test_validate_graph_execution_permissions_marketplace_graph_not_in_library_denied
# | admin    | no    | no          | none present                 | false        | DENY acc | test_validate_graph_execution_permissions_admin_without_library_or_marketplace_denied
# | regular  | no    | yes         | none present                 | true         | ALLOW    | test_validate_graph_execution_permissions_marketplace_sub_graph_without_library_allowed
# | regular  | no    | no          | none present                 | true         | DENY acc | test_validate_graph_execution_permissions_unpublished_sub_graph_without_library_denied
# | regular  | no    | no          | wrong version only           | false        | DENY acc | test_validate_graph_execution_permissions_library_wrong_version_denied


@pytest.mark.asyncio
async def test_validate_graph_execution_permissions_library_member_same_version_allowed() -> (
    None
):
    requester_id = "library-user-id"
    graph_id = "graph-id"
    graph_version = 2
    mock_graph = MagicMock(userId="creator-user-id")

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.is_graph_published_in_marketplace",
            new_callable=AsyncMock,
            return_value=False,
        ) as mock_is_published,
    ):
        mock_ag_prisma.return_value.find_unique = AsyncMock(return_value=mock_graph)
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=MagicMock())

        await validate_graph_execution_permissions(
            user_id=requester_id,
            graph_id=graph_id,
            graph_version=graph_version,
        )

    mock_is_published.assert_not_awaited()
    lib_where = mock_lib_prisma.return_value.find_first.call_args.kwargs["where"]
    assert lib_where["agentGraphVersion"] == graph_version


@pytest.mark.asyncio
async def test_validate_graph_execution_permissions_owner_same_version_in_library_allowed() -> (
    None
):
    requester_id = "owner-user-id"
    graph_id = "graph-id"
    graph_version = 2
    mock_graph = MagicMock(userId=requester_id)

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.is_graph_published_in_marketplace",
            new_callable=AsyncMock,
            return_value=False,
        ) as mock_is_published,
    ):
        mock_ag_prisma.return_value.find_unique = AsyncMock(return_value=mock_graph)
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=MagicMock())

        await validate_graph_execution_permissions(
            user_id=requester_id,
            graph_id=graph_id,
            graph_version=graph_version,
        )

    mock_is_published.assert_not_awaited()
    lib_where = mock_lib_prisma.return_value.find_first.call_args.kwargs["where"]
    assert lib_where["agentGraphVersion"] == graph_version


@pytest.mark.asyncio
async def test_validate_graph_execution_permissions_owner_previous_library_version_allowed() -> (
    None
):
    requester_id = "owner-user-id"
    graph_id = "graph-id"
    graph_version = 2
    mock_graph = MagicMock(userId=requester_id)

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.is_graph_published_in_marketplace",
            new_callable=AsyncMock,
            return_value=False,
        ) as mock_is_published,
    ):
        mock_ag_prisma.return_value.find_unique = AsyncMock(return_value=mock_graph)
        mock_lib_prisma.return_value.find_first = AsyncMock(
            side_effect=[None, MagicMock(name="PriorVersionLibraryAgent")]
        )

        await validate_graph_execution_permissions(
            user_id=requester_id,
            graph_id=graph_id,
            graph_version=graph_version,
        )

    mock_is_published.assert_not_awaited()
    assert mock_lib_prisma.return_value.find_first.await_count == 2
    first_where = mock_lib_prisma.return_value.find_first.await_args_list[0].kwargs[
        "where"
    ]
    second_where = mock_lib_prisma.return_value.find_first.await_args_list[1].kwargs[
        "where"
    ]
    assert first_where["agentGraphVersion"] == graph_version
    assert "agentGraphVersion" not in second_where


@pytest.mark.asyncio
async def test_validate_graph_execution_permissions_owner_without_library_denied() -> (
    None
):
    requester_id = "owner-user-id"
    graph_id = "graph-id"
    graph_version = 2
    mock_graph = MagicMock(userId=requester_id)

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.is_graph_published_in_marketplace",
            new_callable=AsyncMock,
            return_value=False,
        ) as mock_is_published,
    ):
        mock_ag_prisma.return_value.find_unique = AsyncMock(return_value=mock_graph)
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=None)

        with pytest.raises(GraphNotInLibraryError):
            await validate_graph_execution_permissions(
                user_id=requester_id,
                graph_id=graph_id,
                graph_version=graph_version,
            )

    mock_is_published.assert_not_awaited()
    assert mock_lib_prisma.return_value.find_first.await_count == 2
    first_where = mock_lib_prisma.return_value.find_first.await_args_list[0].kwargs[
        "where"
    ]
    second_where = mock_lib_prisma.return_value.find_first.await_args_list[1].kwargs[
        "where"
    ]
    assert first_where["agentGraphVersion"] == graph_version
    assert second_where == {
        "userId": requester_id,
        "agentGraphId": graph_id,
        "isDeleted": False,
        "isArchived": False,
    }


@pytest.mark.asyncio
async def test_validate_graph_execution_permissions_owner_previous_archived_library_version_denied() -> (
    None
):
    requester_id = "owner-user-id"
    graph_id = "graph-id"
    graph_version = 2
    mock_graph = MagicMock(userId=requester_id)

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.is_graph_published_in_marketplace",
            new_callable=AsyncMock,
            return_value=False,
        ) as mock_is_published,
    ):
        mock_ag_prisma.return_value.find_unique = AsyncMock(return_value=mock_graph)
        mock_lib_prisma.return_value.find_first = AsyncMock(side_effect=[None, None])

        with pytest.raises(GraphNotInLibraryError):
            await validate_graph_execution_permissions(
                user_id=requester_id,
                graph_id=graph_id,
                graph_version=graph_version,
            )

    mock_is_published.assert_not_awaited()
    assert mock_lib_prisma.return_value.find_first.await_count == 2
    first_where = mock_lib_prisma.return_value.find_first.await_args_list[0].kwargs[
        "where"
    ]
    second_where = mock_lib_prisma.return_value.find_first.await_args_list[1].kwargs[
        "where"
    ]
    assert first_where["agentGraphVersion"] == graph_version
    assert second_where == {
        "userId": requester_id,
        "agentGraphId": graph_id,
        "isDeleted": False,
        "isArchived": False,
    }


@pytest.mark.asyncio
async def test_validate_graph_execution_permissions_marketplace_graph_not_in_library_denied() -> (
    None
):
    requester_id = "marketplace-user-id"
    graph_id = "graph-id"
    graph_version = 2
    mock_graph = MagicMock(userId="creator-user-id")

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.is_graph_published_in_marketplace",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_is_published,
    ):
        mock_ag_prisma.return_value.find_unique = AsyncMock(return_value=mock_graph)
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=None)

        with pytest.raises(GraphNotInLibraryError):
            await validate_graph_execution_permissions(
                user_id=requester_id,
                graph_id=graph_id,
                graph_version=graph_version,
            )

    mock_is_published.assert_awaited_once_with(graph_id, graph_version)


@pytest.mark.asyncio
async def test_validate_graph_execution_permissions_admin_without_library_or_marketplace_denied() -> (
    None
):
    requester_id = "admin-user-id"
    graph_id = "graph-id"
    graph_version = 2
    mock_graph = MagicMock(userId="creator-user-id")

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.is_graph_published_in_marketplace",
            new_callable=AsyncMock,
            return_value=False,
        ) as mock_is_published,
    ):
        mock_ag_prisma.return_value.find_unique = AsyncMock(return_value=mock_graph)
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=None)

        with pytest.raises(GraphNotAccessibleError):
            await validate_graph_execution_permissions(
                user_id=requester_id,
                graph_id=graph_id,
                graph_version=graph_version,
            )

    mock_is_published.assert_awaited_once_with(graph_id, graph_version)


@pytest.mark.asyncio
async def test_validate_graph_execution_permissions_unpublished_sub_graph_without_library_denied() -> (
    None
):
    requester_id = "marketplace-user-id"
    graph_id = "graph-id"
    graph_version = 2
    mock_graph = MagicMock(userId="creator-user-id")

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.is_graph_published_in_marketplace",
            new_callable=AsyncMock,
            return_value=False,
        ) as mock_is_published,
    ):
        mock_ag_prisma.return_value.find_unique = AsyncMock(return_value=mock_graph)
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=None)

        with pytest.raises(GraphNotAccessibleError):
            await validate_graph_execution_permissions(
                user_id=requester_id,
                graph_id=graph_id,
                graph_version=graph_version,
                is_sub_graph=True,
            )

    mock_is_published.assert_awaited_once_with(graph_id, graph_version)


@pytest.mark.asyncio
async def test_validate_graph_execution_permissions_marketplace_sub_graph_without_library_allowed() -> (
    None
):
    requester_id = "marketplace-user-id"
    graph_id = "graph-id"
    graph_version = 2
    mock_graph = MagicMock(userId="creator-user-id")

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.is_graph_published_in_marketplace",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_is_published,
    ):
        mock_ag_prisma.return_value.find_unique = AsyncMock(return_value=mock_graph)
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=None)

        await validate_graph_execution_permissions(
            user_id=requester_id,
            graph_id=graph_id,
            graph_version=graph_version,
            is_sub_graph=True,
        )

    mock_is_published.assert_awaited_once_with(graph_id, graph_version)


@pytest.mark.asyncio
async def test_validate_graph_execution_permissions_library_wrong_version_denied() -> (
    None
):
    requester_id = "library-user-id"
    graph_id = "graph-id"
    graph_version = 2
    mock_graph = MagicMock(userId="creator-user-id")

    with (
        patch("backend.data.graph.AgentGraph.prisma") as mock_ag_prisma,
        patch("backend.data.graph.LibraryAgent.prisma") as mock_lib_prisma,
        patch(
            "backend.data.graph.is_graph_published_in_marketplace",
            new_callable=AsyncMock,
            return_value=False,
        ) as mock_is_published,
    ):
        mock_ag_prisma.return_value.find_unique = AsyncMock(return_value=mock_graph)
        mock_lib_prisma.return_value.find_first = AsyncMock(return_value=None)

        with pytest.raises(GraphNotAccessibleError):
            await validate_graph_execution_permissions(
                user_id=requester_id,
                graph_id=graph_id,
                graph_version=graph_version,
            )

    mock_is_published.assert_awaited_once_with(graph_id, graph_version)
    lib_where = mock_lib_prisma.return_value.find_first.call_args.kwargs["where"]
    assert lib_where["agentGraphVersion"] == graph_version
