from unittest.mock import MagicMock

import pytest

from backend.blocks.system.library_operations import (
    AddToLibraryFromStoreBlock,
    LibraryAgent,
)
from backend.blocks.system.store_operations import (
    GetStoreAgentDetailsBlock,
    SearchAgentsResponse,
    SearchStoreAgentsBlock,
    StoreAgentDetails,
    StoreAgentDict,
)


@pytest.mark.asyncio
async def test_add_to_library_from_store_block_success(mocker):
    """Test successful addition of agent from store to library."""
    block = AddToLibraryFromStoreBlock()

    # Mock the library agent response
    mock_library_agent = MagicMock()
    mock_library_agent.id = "lib-agent-123"
    mock_library_agent.graph_id = "graph-456"
    mock_library_agent.graph_version = 1
    mock_library_agent.name = "Test Agent"

    mocker.patch.object(
        block,
        "_add_to_library",
        return_value=LibraryAgent(
            library_agent_id="lib-agent-123",
            agent_id="graph-456",
            agent_version=1,
            agent_name="Test Agent",
        ),
    )

    input_data = block.Input(
        store_listing_version_id="store-listing-v1", agent_name="Custom Agent Name"
    )

    outputs = {}
    async for name, value in block.run(input_data, user_id="test-user"):
        outputs[name] = value

    assert outputs["success"] is True
    assert outputs["library_agent_id"] == "lib-agent-123"
    assert outputs["agent_id"] == "graph-456"
    assert outputs["agent_version"] == 1
    assert outputs["agent_name"] == "Test Agent"
    assert outputs["message"] == "Agent successfully added to library"


@pytest.mark.asyncio
async def test_get_store_agent_details_block_success(mocker):
    """Test successful retrieval of store agent details."""
    block = GetStoreAgentDetailsBlock()

    mocker.patch.object(
        block,
        "_get_agent_details",
        return_value=StoreAgentDetails(
            found=True,
            store_listing_version_id="version-123",
            agent_name="Test Agent",
            description="A test agent for testing",
            creator="Test Creator",
            categories=["productivity", "automation"],
            runs=100,
            rating=4.5,
        ),
    )

    input_data = block.Input(creator="Test Creator", slug="test-slug")
    outputs = {}
    async for name, value in block.run(input_data):
        outputs[name] = value

    assert outputs["found"] is True
    assert outputs["store_listing_version_id"] == "version-123"
    assert outputs["agent_name"] == "Test Agent"
    assert outputs["description"] == "A test agent for testing"
    assert outputs["creator"] == "Test Creator"
    assert outputs["categories"] == ["productivity", "automation"]
    assert outputs["runs"] == 100
    assert outputs["rating"] == 4.5


@pytest.mark.asyncio
async def test_search_store_agents_block(mocker):
    """Test searching for store agents."""
    block = SearchStoreAgentsBlock()

    mocker.patch.object(
        block,
        "_search_agents",
        return_value=SearchAgentsResponse(
            agents=[
                StoreAgentDict(
                    slug="creator1/agent1",
                    name="Agent One",
                    description="First test agent",
                    creator="Creator 1",
                    rating=4.8,
                    runs=500,
                ),
                StoreAgentDict(
                    slug="creator2/agent2",
                    name="Agent Two",
                    description="Second test agent",
                    creator="Creator 2",
                    rating=4.2,
                    runs=200,
                ),
            ],
            total_count=2,
        ),
    )

    input_data = block.Input(
        query="test", category="productivity", sort_by="rating", limit=10
    )

    outputs = {}
    async for name, value in block.run(input_data):
        outputs[name] = value

    assert len(outputs["agents"]) == 2
    assert outputs["total_count"] == 2
    assert outputs["agents"][0]["name"] == "Agent One"
    assert outputs["agents"][0]["rating"] == 4.8


@pytest.mark.asyncio
async def test_search_store_agents_block_empty_results(mocker):
    """Test searching with no results."""
    block = SearchStoreAgentsBlock()

    mocker.patch.object(
        block,
        "_search_agents",
        return_value=SearchAgentsResponse(agents=[], total_count=0),
    )

    input_data = block.Input(query="nonexistent", limit=10)

    outputs = {}
    async for name, value in block.run(input_data):
        outputs[name] = value

    assert outputs["agents"] == []
    assert outputs["total_count"] == 0
