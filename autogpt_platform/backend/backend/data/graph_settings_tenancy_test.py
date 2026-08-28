from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data import graph


@pytest.mark.parametrize(
    ("organization_id", "team_id"),
    [("org-1", None), ("org-1", "team-a")],
)
@pytest.mark.asyncio
async def test_graph_settings_use_exact_execution_scope(
    mocker, organization_id: str, team_id: str | None
) -> None:
    client = MagicMock(
        find_first=AsyncMock(
            return_value=MagicMock(
                settings={
                    "human_in_the_loop_safe_mode": False,
                    "sensitive_action_safe_mode": True,
                }
            )
        )
    )
    mocker.patch.object(graph.LibraryAgent, "prisma", return_value=client)

    settings = await graph.get_graph_settings(
        user_id="user-1",
        graph_id="graph-1",
        graph_version=3,
        organization_id=organization_id,
        team_id=team_id,
    )

    client.find_first.assert_awaited_once_with(
        where={
            "userId": "user-1",
            "agentGraphId": "graph-1",
            "agentGraphVersion": 3,
            "organizationId": organization_id,
            "teamId": team_id,
            "isDeleted": False,
            "isArchived": False,
        }
    )
    assert settings.human_in_the_loop_safe_mode is False
    assert settings.sensitive_action_safe_mode is True
