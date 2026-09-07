from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data import graph


@pytest.mark.asyncio
@pytest.mark.parametrize("team_restriction", [None, "team-a"])
async def test_owner_previous_version_fallback_preserves_visible_team_scope(
    mocker, team_restriction
):
    graph_client = MagicMock(
        find_unique=AsyncMock(
            return_value=MagicMock(
                userId="owner", organizationId="org", teamId="team-a"
            )
        )
    )
    library_client = MagicMock(find_first=AsyncMock(side_effect=[None, MagicMock()]))
    mocker.patch.object(graph.AgentGraph, "prisma", return_value=graph_client)
    mocker.patch.object(graph.LibraryAgent, "prisma", return_value=library_client)
    mocker.patch.object(graph, "get_user_team_ids", AsyncMock(return_value=["team-a"]))
    mocker.patch.object(graph, "resolve_graph_grants", AsyncMock(return_value=[]))

    await graph.validate_graph_execution_permissions(
        "owner",
        "graph",
        2,
        organization_id="org",
        team_id_restriction=team_restriction,
    )

    fallback = library_client.find_first.await_args_list[1].kwargs["where"]
    assert fallback["userId"] == "owner"
    assert "agentGraphVersion" not in fallback
    assert fallback.get("teamId", "unrestricted") is not None
    scope = fallback.get("AND", [fallback])[0]
    if team_restriction:
        assert scope["organizationId"] == "org"
        assert scope["teamId"] == team_restriction
    else:
        assert {"organizationId": "org", "teamId": {"in": ["team-a"]}} in scope["OR"]
