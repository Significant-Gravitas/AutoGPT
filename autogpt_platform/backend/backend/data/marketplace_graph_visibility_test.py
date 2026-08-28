from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import SubmissionStatus

from backend.data import graph


@pytest.mark.asyncio
async def test_marketplace_graph_fallback_requires_live_public_version(mocker) -> None:
    store_client = MagicMock(find_first=AsyncMock(return_value=None))
    mocker.patch.object(graph.StoreListingVersion, "prisma", return_value=store_client)

    assert await graph.get_graph("graph-1", version=3, user_id=None) is None

    store_client.find_first.assert_awaited_once_with(
        where={
            "agentGraphId": "graph-1",
            "submissionStatus": SubmissionStatus.APPROVED,
            "isDeleted": False,
            "isAvailable": True,
            "StoreListing": {"is": {"isDeleted": False}},
            "agentGraphVersion": 3,
        },
        order={"agentGraphVersion": "desc"},
        include={"AgentGraph": {"include": graph.AGENT_GRAPH_INCLUDE}},
    )


@pytest.mark.asyncio
async def test_marketplace_execution_permission_requires_live_public_version(
    mocker,
) -> None:
    store_client = MagicMock(find_first=AsyncMock(return_value=None))
    mocker.patch.object(graph.StoreListingVersion, "prisma", return_value=store_client)

    assert await graph.is_graph_published_in_marketplace("graph-1", 3) is False

    store_client.find_first.assert_awaited_once_with(
        where={
            "agentGraphId": "graph-1",
            "agentGraphVersion": 3,
            "submissionStatus": SubmissionStatus.APPROVED,
            "isDeleted": False,
            "isAvailable": True,
            "StoreListing": {"is": {"isDeleted": False}},
        }
    )
