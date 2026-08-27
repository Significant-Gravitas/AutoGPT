from datetime import datetime
from unittest.mock import AsyncMock

import prisma.enums
import prisma.models
import pytest

from backend.data.graph import GraphMeta, get_graph_metadata


@pytest.mark.asyncio
async def test_get_graph_metadata_returns_persisted_tenancy(mocker):
    row = prisma.models.AgentGraph(
        id="g-org",
        version=1,
        name="Org Graph",
        description="",
        userId="u1",
        isActive=True,
        createdAt=datetime.now(),
        visibility=prisma.enums.ResourceVisibility.PRIVATE,
        organizationId="org-1",
        teamId="team-1",
    )
    query = AsyncMock()
    query.find_first.return_value = row
    mocker.patch(
        "backend.data.graph.AgentGraph.prisma", autospec=True, return_value=query
    )

    metadata = await get_graph_metadata("g-org")

    assert isinstance(metadata, GraphMeta)
    assert metadata.organization_id == "org-1"
    assert metadata.team_id == "team-1"
