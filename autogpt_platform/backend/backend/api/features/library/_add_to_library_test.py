from unittest.mock import AsyncMock, MagicMock, patch

import prisma.errors
import pytest

from ._add_to_library import _get_marketplace_metadata, add_graph_to_library

MARKETPLACE = {
    "name": "Marketplace Title",
    "description": "Marketplace description",
    "imageUrl": "https://cdn.example.com/agent.png",
}


@pytest.mark.asyncio
async def test_add_graph_to_library_create_new_agent() -> None:
    """When no matching LibraryAgent exists, create inserts a new one."""
    graph_model = MagicMock(id="graph-id", version=2, nodes=[])
    created_agent = MagicMock(name="CreatedLibraryAgent")
    converted_agent = MagicMock(name="ConvertedLibraryAgent")

    with (
        patch(
            "backend.api.features.library._add_to_library.prisma.models.LibraryAgent.prisma"
        ) as mock_prisma,
        patch(
            "backend.api.features.library._add_to_library.library_model.LibraryAgent.from_db",
            return_value=converted_agent,
        ) as mock_from_db,
        patch(
            "backend.api.features.library._add_to_library._fetch_schedule_info",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "backend.api.features.library._add_to_library._get_marketplace_metadata",
            new=AsyncMock(return_value=dict(MARKETPLACE)),
        ),
    ):
        mock_prisma.return_value.create = AsyncMock(return_value=created_agent)

        result = await add_graph_to_library("slv-id", graph_model, "user-id")

    assert result is converted_agent
    mock_from_db.assert_called_once_with(created_agent, schedule_info={})
    # Verify create was called with correct data
    create_call = mock_prisma.return_value.create.call_args
    create_data = create_call.kwargs["data"]
    assert create_data["User"] == {"connect": {"id": "user-id"}}
    assert create_data["AgentGraph"] == {
        "connect": {"graphVersionId": {"id": "graph-id", "version": 2}}
    }
    assert create_data["isCreatedByUser"] is False
    assert create_data["useGraphIsActiveVersion"] is False
    # Marketplace metadata is snapshotted onto the new LibraryAgent
    assert create_data["name"] == "Marketplace Title"
    assert create_data["description"] == "Marketplace description"
    assert create_data["imageUrl"] == "https://cdn.example.com/agent.png"


@pytest.mark.asyncio
async def test_add_graph_to_library_unique_violation_updates_existing() -> None:
    """UniqueViolationError on create falls back to update."""
    graph_model = MagicMock(id="graph-id", version=2, nodes=[])
    updated_agent = MagicMock(name="UpdatedLibraryAgent")
    converted_agent = MagicMock(name="ConvertedLibraryAgent")

    with (
        patch(
            "backend.api.features.library._add_to_library.prisma.models.LibraryAgent.prisma"
        ) as mock_prisma,
        patch(
            "backend.api.features.library._add_to_library.library_model.LibraryAgent.from_db",
            return_value=converted_agent,
        ) as mock_from_db,
        patch(
            "backend.api.features.library._add_to_library._fetch_schedule_info",
            new=AsyncMock(return_value={}),
        ),
        patch(
            "backend.api.features.library._add_to_library._get_marketplace_metadata",
            new=AsyncMock(return_value=dict(MARKETPLACE)),
        ),
    ):
        mock_prisma.return_value.create = AsyncMock(
            side_effect=prisma.errors.UniqueViolationError(
                MagicMock(), message="unique constraint"
            )
        )
        mock_prisma.return_value.update = AsyncMock(return_value=updated_agent)

        result = await add_graph_to_library("slv-id", graph_model, "user-id")

    assert result is converted_agent
    mock_from_db.assert_called_once_with(updated_agent, schedule_info={})
    # Verify update was called with correct where and data
    update_call = mock_prisma.return_value.update.call_args
    assert update_call.kwargs["where"] == {
        "userId_agentGraphId_agentGraphVersion": {
            "userId": "user-id",
            "agentGraphId": "graph-id",
            "agentGraphVersion": 2,
        }
    }
    update_data = update_call.kwargs["data"]
    assert update_data["isDeleted"] is False
    assert update_data["isArchived"] is False
    # Restoring a soft-deleted agent refreshes the marketplace snapshot too
    assert update_data["name"] == "Marketplace Title"
    assert update_data["description"] == "Marketplace description"
    assert update_data["imageUrl"] == "https://cdn.example.com/agent.png"


@pytest.mark.asyncio
async def test_get_marketplace_metadata_returns_first_image() -> None:
    """Pulls name/description and the first image URL from the listing version."""
    slv = MagicMock(
        description="Marketplace description",
        imageUrls=["https://cdn.example.com/a.png", "https://cdn.example.com/b.png"],
    )
    slv.name = "Marketplace Title"  # `name` is reserved in the MagicMock ctor

    with patch(
        "backend.api.features.library._add_to_library.prisma.models.StoreListingVersion.prisma"
    ) as mock_prisma:
        mock_prisma.return_value.find_unique = AsyncMock(return_value=slv)

        result = await _get_marketplace_metadata("slv-id")

    assert result == {
        "name": "Marketplace Title",
        "description": "Marketplace description",
        "imageUrl": "https://cdn.example.com/a.png",
    }


@pytest.mark.asyncio
async def test_get_marketplace_metadata_missing_listing_returns_nulls() -> None:
    """A missing listing version yields all-null metadata (graph values win)."""
    with patch(
        "backend.api.features.library._add_to_library.prisma.models.StoreListingVersion.prisma"
    ) as mock_prisma:
        mock_prisma.return_value.find_unique = AsyncMock(return_value=None)

        result = await _get_marketplace_metadata("missing")

    assert result == {"name": None, "description": None, "imageUrl": None}
