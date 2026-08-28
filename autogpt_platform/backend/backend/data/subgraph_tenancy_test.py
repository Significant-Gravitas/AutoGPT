from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.blocks.agent import AgentExecutorBlock
from backend.data import graph as graph_db


@pytest.mark.parametrize("team_id", [None, "team-b"])
@pytest.mark.asyncio
async def test_subgraph_discovery_requires_exact_parent_scope(mocker, team_id) -> None:
    node = MagicMock(
        AgentBlock=MagicMock(id=AgentExecutorBlock().id),
        constantInput={"graph_id": "child-graph", "graph_version": 4},
    )
    parent = MagicMock(
        id="parent-graph",
        userId="owner-1",
        organizationId="org-1",
        teamId=team_id,
        Nodes=[node],
    )
    client = MagicMock(find_many=AsyncMock(return_value=[]))
    mocker.patch.object(graph_db.AgentGraph, "prisma", return_value=client)

    assert await graph_db.get_sub_graphs(parent) == []

    client.find_many.assert_awaited_once_with(
        where={
            "OR": [
                {
                    "id": "child-graph",
                    "version": 4,
                    "userId": "owner-1",
                    "organizationId": "org-1",
                    "teamId": team_id,
                }
            ]
        },
        include=graph_db.AGENT_GRAPH_INCLUDE,
    )
