import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.library import model as library_model
from backend.api.features.store.db import StoreAgentsSortOptions
from backend.blocks.system.library_operations import (
    AddToLibraryFromStoreBlock,
    LibraryAgent,
    ListLibraryAgentsBlock,
)
from backend.blocks.system.store_operations import (
    GetStoreAgentDetailsBlock,
    SearchAgentsResponse,
    SearchStoreAgentsBlock,
    StoreAgentDetails,
    StoreAgentDict,
)
from backend.util.models import Pagination


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
        query="test",
        category="productivity",
        sort_by=StoreAgentsSortOptions.RATING,  # type: ignore[reportArgumentType]
        limit=10,
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


def _real_library_agent(creator_name: str) -> library_model.LibraryAgent:
    now = datetime.datetime.now(datetime.timezone.utc)
    return library_model.LibraryAgent(
        id="lib-agent-1",
        graph_id="graph-1",
        graph_version=3,
        image_url=None,
        creator_name=creator_name,
        creator_image_url="",
        status=library_model.LibraryAgentStatus.COMPLETED,
        created_at=now,
        updated_at=now,
        name="Library Agent",
        description="An agent in the library",
        input_schema={},
        output_schema={},
        credentials_input_schema=None,
        has_external_trigger=False,
        has_human_in_the_loop=False,
        has_sensitive_action=False,
        new_output=False,
        can_access_graph=True,
        is_latest_version=True,
        is_favorite=False,
    )


async def _run_list_library_agents_block(mocker, creator_name: str) -> dict:
    """Run ListLibraryAgentsBlock end to end on a real library model, mocking
    only the database client (not the block's own mapping)."""
    db_client = MagicMock()
    db_client.list_library_agents = AsyncMock(
        return_value=library_model.LibraryAgentResponse(
            agents=[_real_library_agent(creator_name)],
            pagination=Pagination(
                total_items=1, total_pages=1, current_page=1, page_size=10
            ),
        )
    )
    mocker.patch(
        "backend.blocks.system.library_operations.get_database_manager_async_client",
        return_value=db_client,
    )

    block = ListLibraryAgentsBlock()
    outputs = {}
    async for name, value in block.run(
        block.Input(search_query=None, limit=10, page=1), user_id="test-user"
    ):
        outputs[name] = value

    db_client.list_library_agents.assert_awaited_once_with(
        user_id="test-user", search_term=None, page=1, page_size=10
    )
    return outputs


@pytest.mark.asyncio
async def test_list_library_agents_block_maps_real_library_model(mocker):
    """The block must read the real library model fields (#15666)."""
    creator = "Zoë O'Brien 🌙「東京」"
    outputs = await _run_list_library_agents_block(mocker, creator)

    expected = LibraryAgent(
        library_agent_id="lib-agent-1",
        agent_id="graph-1",
        agent_version=3,
        agent_name="Library Agent",
        description="An agent in the library",
        creator=creator,
        is_archived=False,
        categories=[],
    )
    assert outputs["agents"] == [expected]
    assert outputs["agent"] == expected
    assert outputs["agent"].creator == creator
    assert outputs["total_count"] == 1
    assert outputs["page"] == 1
    assert outputs["total_pages"] == 1


@pytest.mark.asyncio
async def test_list_library_agents_block_unknown_creator_is_empty(mocker):
    """The library model's "Unknown" placeholder is exposed as empty."""
    outputs = await _run_list_library_agents_block(mocker, "Unknown")

    assert outputs["agent"].creator == ""
    assert outputs["agents"][0].creator == ""
    assert outputs["agent"].categories == []
    assert outputs["agent"].is_archived is False
