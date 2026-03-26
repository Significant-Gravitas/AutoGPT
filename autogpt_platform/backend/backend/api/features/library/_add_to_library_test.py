from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ._add_to_library import add_graph_to_library


@pytest.mark.asyncio
async def test_add_graph_to_library_restores_archived_agent() -> None:
    graph_model = MagicMock(id="graph-id", version=2)
    archived_agent = MagicMock(id="library-agent-id", isDeleted=False, isArchived=True)
    restored_agent = MagicMock(name="LibraryAgentModel")

    with (
        patch(
            "backend.api.features.library._add_to_library.get_library_agent_by_graph_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "backend.api.features.library._add_to_library.prisma.models.LibraryAgent.prisma"
        ) as mock_prisma,
        patch(
            "backend.api.features.library._add_to_library.update_library_agent",
            new=AsyncMock(return_value=restored_agent),
        ) as mock_update,
    ):
        mock_prisma.return_value.find_unique = AsyncMock(return_value=archived_agent)

        result = await add_graph_to_library("slv-id", graph_model, "user-id")

    assert result is restored_agent
    mock_update.assert_awaited_once_with(
        "library-agent-id",
        "user-id",
        is_deleted=False,
        is_archived=False,
    )
    mock_prisma.return_value.create.assert_not_called()


@pytest.mark.asyncio
async def test_add_graph_to_library_restores_deleted_agent() -> None:
    graph_model = MagicMock(id="graph-id", version=2)
    deleted_agent = MagicMock(id="library-agent-id", isDeleted=True, isArchived=False)
    restored_agent = MagicMock(name="LibraryAgentModel")

    with (
        patch(
            "backend.api.features.library._add_to_library.get_library_agent_by_graph_id",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "backend.api.features.library._add_to_library.prisma.models.LibraryAgent.prisma"
        ) as mock_prisma,
        patch(
            "backend.api.features.library._add_to_library.update_library_agent",
            new=AsyncMock(return_value=restored_agent),
        ) as mock_update,
    ):
        mock_prisma.return_value.find_unique = AsyncMock(return_value=deleted_agent)

        result = await add_graph_to_library("slv-id", graph_model, "user-id")

    assert result is restored_agent
    mock_update.assert_awaited_once_with(
        "library-agent-id",
        "user-id",
        is_deleted=False,
        is_archived=False,
    )
    mock_prisma.return_value.create.assert_not_called()
