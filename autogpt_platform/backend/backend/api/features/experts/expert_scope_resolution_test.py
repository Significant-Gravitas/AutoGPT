from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.experts import experts_db


@pytest.mark.asyncio
async def test_resolve_expert_for_graph_filters_exact_library_scope(mocker) -> None:
    client = MagicMock(find_many=AsyncMock(return_value=[]))
    mocker.patch(
        "prisma.models.ExpertWorkflow.prisma",
        return_value=client,
    )

    assert (
        await experts_db.resolve_expert_for_graph(
            "user-1",
            "graph-1",
            organization_id="org-1",
            team_id="team-b",
            enforce_scope=True,
        )
        is None
    )

    library_filter = client.find_many.await_args.kwargs["where"]["LibraryAgent"]["is"]
    assert library_filter == {
        "userId": "user-1",
        "agentGraphId": "graph-1",
        "isDeleted": False,
        "organizationId": "org-1",
        "teamId": "team-b",
    }
