from unittest.mock import AsyncMock, MagicMock, patch

import prisma.errors
import pytest

from ._add_to_library import add_graph_to_library


@pytest.mark.asyncio
async def test_add_graph_to_library_upsert_creates_new_agent() -> None:
    """When no matching LibraryAgent exists, upsert creates a new one."""
    graph_model = MagicMock(id="graph-id", version=2, nodes=[])
    upserted_agent = MagicMock(name="UpsertedLibraryAgent")
    converted_agent = MagicMock(name="ConvertedLibraryAgent")

    with (
        patch(
            "backend.api.features.library._add_to_library.prisma.models.LibraryAgent.prisma"
        ) as mock_prisma,
        patch(
            "backend.api.features.library._add_to_library.library_model.LibraryAgent.from_db",
            return_value=converted_agent,
        ) as mock_from_db,
    ):
        mock_prisma.return_value.upsert = AsyncMock(return_value=upserted_agent)

        result = await add_graph_to_library("slv-id", graph_model, "user-id")

    assert result is converted_agent
    mock_from_db.assert_called_once_with(upserted_agent)
    # Verify upsert was called with correct where clause
    upsert_call = mock_prisma.return_value.upsert.call_args
    assert upsert_call.kwargs["where"] == {
        "userId_agentGraphId_agentGraphVersion": {
            "userId": "user-id",
            "agentGraphId": "graph-id",
            "agentGraphVersion": 2,
        }
    }
    # Verify create data has the right structure
    create_data = upsert_call.kwargs["data"]["create"]
    assert create_data["User"] == {"connect": {"id": "user-id"}}
    assert create_data["AgentGraph"] == {
        "connect": {"graphVersionId": {"id": "graph-id", "version": 2}}
    }
    assert create_data["isCreatedByUser"] is False
    assert create_data["useGraphIsActiveVersion"] is False
    # Verify update data restores soft-deleted/archived
    update_data = upsert_call.kwargs["data"]["update"]
    assert update_data["isDeleted"] is False
    assert update_data["isArchived"] is False


@pytest.mark.asyncio
async def test_add_graph_to_library_upsert_restores_archived_agent() -> None:
    """When an archived LibraryAgent exists, upsert restores it."""
    graph_model = MagicMock(id="graph-id", version=2, nodes=[])
    # The upsert will hit the update branch for an existing (archived) agent
    restored_agent = MagicMock(name="RestoredLibraryAgent")
    converted_agent = MagicMock(name="ConvertedLibraryAgent")

    with (
        patch(
            "backend.api.features.library._add_to_library.prisma.models.LibraryAgent.prisma"
        ) as mock_prisma,
        patch(
            "backend.api.features.library._add_to_library.library_model.LibraryAgent.from_db",
            return_value=converted_agent,
        ),
    ):
        mock_prisma.return_value.upsert = AsyncMock(return_value=restored_agent)

        result = await add_graph_to_library("slv-id", graph_model, "user-id")

    assert result is converted_agent


@pytest.mark.asyncio
async def test_add_graph_to_library_unique_violation_fallback() -> None:
    """UniqueViolationError falls back to get_library_agent_by_graph_id."""
    graph_model = MagicMock(id="graph-id", version=2, nodes=[])
    existing_agent = MagicMock(name="ExistingLibraryAgent")

    with (
        patch(
            "backend.api.features.library._add_to_library.prisma.models.LibraryAgent.prisma"
        ) as mock_prisma,
        patch(
            "backend.api.features.library._add_to_library.get_library_agent_by_graph_id",
            new=AsyncMock(return_value=existing_agent),
        ) as mock_get,
    ):
        mock_prisma.return_value.upsert = AsyncMock(
            side_effect=prisma.errors.UniqueViolationError(
                MagicMock(), message="unique constraint"
            )
        )

        result = await add_graph_to_library("slv-id", graph_model, "user-id")

    assert result is existing_agent
    mock_get.assert_awaited_once_with("user-id", "graph-id", 2)


@pytest.mark.asyncio
async def test_add_graph_to_library_unique_violation_reraises_when_not_found() -> None:
    """UniqueViolationError re-raises if fallback lookup also fails."""
    graph_model = MagicMock(id="graph-id", version=2, nodes=[])

    with (
        patch(
            "backend.api.features.library._add_to_library.prisma.models.LibraryAgent.prisma"
        ) as mock_prisma,
        patch(
            "backend.api.features.library._add_to_library.get_library_agent_by_graph_id",
            new=AsyncMock(return_value=None),
        ),
    ):
        mock_prisma.return_value.upsert = AsyncMock(
            side_effect=prisma.errors.UniqueViolationError(
                MagicMock(), message="unique constraint"
            )
        )

        with pytest.raises(prisma.errors.UniqueViolationError):
            await add_graph_to_library("slv-id", graph_model, "user-id")
