from unittest.mock import AsyncMock, MagicMock, patch

import prisma.errors
import pytest

from ._add_to_library import _marketplace_metadata, add_graph_to_library


def _make_slv(
    *,
    name: str = "Marketplace Title",
    description: str = "Marketplace description",
    image_urls: list[str] | None = None,
) -> MagicMock:
    slv = MagicMock(
        id="slv-id",
        description=description,
        imageUrls=(
            ["https://cdn.example.com/agent.png"] if image_urls is None else image_urls
        ),
    )
    slv.name = name  # `name` is reserved in the MagicMock ctor
    return slv


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
    ):
        mock_prisma.return_value.create = AsyncMock(return_value=created_agent)

        result = await add_graph_to_library(graph_model, "user-id", _make_slv())

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
    ):
        mock_prisma.return_value.create = AsyncMock(
            side_effect=prisma.errors.UniqueViolationError(
                MagicMock(), message="unique constraint"
            )
        )
        mock_prisma.return_value.update = AsyncMock(return_value=updated_agent)

        result = await add_graph_to_library(graph_model, "user-id", _make_slv())

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


def test_marketplace_metadata_returns_first_image() -> None:
    """Pulls name/description and the first image URL from the listing version."""
    slv = _make_slv(
        image_urls=[
            "https://cdn.example.com/a.png",
            "https://cdn.example.com/b.png",
        ]
    )

    assert _marketplace_metadata(slv) == {
        "name": "Marketplace Title",
        "description": "Marketplace description",
        "imageUrl": "https://cdn.example.com/a.png",
    }


def test_marketplace_metadata_without_images_yields_null_image() -> None:
    """A listing with no images snapshots a null imageUrl (graph image wins)."""
    slv = _make_slv(image_urls=[])

    assert _marketplace_metadata(slv) == {
        "name": "Marketplace Title",
        "description": "Marketplace description",
        "imageUrl": None,
    }
