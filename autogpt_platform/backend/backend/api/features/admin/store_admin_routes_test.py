from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data.graph import get_graph_as_admin

# Shared constants
ADMIN_USER_ID = "admin-user-id"
CREATOR_USER_ID = "other-creator-id"
GRAPH_ID = "test-graph-id"
GRAPH_VERSION = 3


def _make_mock_graph(user_id: str = CREATOR_USER_ID) -> MagicMock:
    graph = MagicMock()
    graph.userId = user_id
    graph.id = GRAPH_ID
    graph.version = GRAPH_VERSION
    graph.Nodes = []
    return graph


@pytest.mark.asyncio
async def test_admin_can_access_pending_agent_not_owned() -> None:
    """Admin must be able to access a graph they don't own even if it's not
    APPROVED in the marketplace. This is the core use case: reviewing a
    submitted-but-pending agent from the admin dashboard."""
    mock_graph = _make_mock_graph()
    mock_graph_model = MagicMock(name="GraphModel")

    with (
        patch(
            "backend.data.graph.AgentGraph.prisma",
        ) as mock_prisma,
        patch(
            "backend.data.graph.is_graph_published_in_marketplace",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "backend.data.graph.GraphModel.from_db",
            return_value=mock_graph_model,
        ),
    ):
        mock_prisma.return_value.find_first = AsyncMock(return_value=mock_graph)

        result = await get_graph_as_admin(
            graph_id=GRAPH_ID,
            version=GRAPH_VERSION,
            user_id=ADMIN_USER_ID,
            for_export=False,
        )

    assert (
        result is not None
    ), "Admin should be able to access a pending agent they don't own"
    assert result is mock_graph_model


@pytest.mark.asyncio
async def test_admin_download_pending_agent_with_subagents() -> None:
    """Admin export (for_export=True) of a pending agent must include
    sub-graphs. This exercises the full export code path that the Download
    button uses."""
    mock_graph = _make_mock_graph()
    mock_sub_graph = MagicMock(name="SubGraph")
    mock_graph_model = MagicMock(name="GraphModel")

    with (
        patch(
            "backend.data.graph.AgentGraph.prisma",
        ) as mock_prisma,
        patch(
            "backend.data.graph.is_graph_published_in_marketplace",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch(
            "backend.data.graph.get_sub_graphs",
            new_callable=AsyncMock,
            return_value=[mock_sub_graph],
        ) as mock_get_sub,
        patch(
            "backend.data.graph.GraphModel.from_db",
            return_value=mock_graph_model,
        ) as mock_from_db,
    ):
        mock_prisma.return_value.find_first = AsyncMock(return_value=mock_graph)

        result = await get_graph_as_admin(
            graph_id=GRAPH_ID,
            version=GRAPH_VERSION,
            user_id=ADMIN_USER_ID,
            for_export=True,
        )

    assert result is not None, "Admin export of pending agent must succeed"
    mock_get_sub.assert_awaited_once_with(mock_graph)
    mock_from_db.assert_called_once_with(
        graph=mock_graph,
        sub_graphs=[mock_sub_graph],
        for_export=True,
    )
