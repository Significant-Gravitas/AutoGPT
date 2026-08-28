from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.library import db
from backend.api.features.library._schedule_info import _fetch_schedule_info
from backend.data.includes import library_agent_include


def _agent(team_id: str | None):
    return SimpleNamespace(
        agentGraphId="graph-1",
        agentGraphVersion=3,
        organizationId="org-1",
        teamId=team_id,
    )


@pytest.mark.asyncio
async def test_execution_counts_are_grouped_by_exact_agent_scope(mocker) -> None:
    client = MagicMock(
        group_by=AsyncMock(
            return_value=[
                {
                    "agentGraphId": "graph-1",
                    "agentGraphVersion": 3,
                    "organizationId": "org-1",
                    "teamId": "team-b",
                    "_count": {"_all": 2},
                }
            ]
        )
    )
    mocker.patch("prisma.models.AgentGraphExecution.prisma", return_value=client)

    result = await db._fetch_execution_counts(
        "user-1", [_agent(None), _agent("team-b")]
    )

    assert result == {("graph-1", 3, "org-1", "team-b"): 2}
    where = client.group_by.await_args.kwargs["where"]
    assert where["OR"] == [
        {
            "agentGraphId": "graph-1",
            "agentGraphVersion": 3,
            "organizationId": "org-1",
            "teamId": None,
        },
        {
            "agentGraphId": "graph-1",
            "agentGraphVersion": 3,
            "organizationId": "org-1",
            "teamId": "team-b",
        },
    ]


@pytest.mark.asyncio
async def test_schedule_info_is_keyed_by_exact_agent_scope(mocker) -> None:
    scheduler = MagicMock(
        get_graph_execution_schedules=AsyncMock(
            return_value=[
                SimpleNamespace(
                    graph_id="graph-1",
                    graph_version=3,
                    organization_id="org-1",
                    team_id="team-b",
                    next_run_time="2026-08-28T12:00:00+00:00",
                )
            ]
        )
    )
    mocker.patch(
        "backend.api.features.library._schedule_info.get_scheduler_client",
        return_value=scheduler,
    )

    result = await _fetch_schedule_info("user-1", exact_scope=True)

    assert result == {("graph-1", 3, "org-1", "team-b"): "2026-08-28T12:00:00+00:00"}


def test_library_agent_execution_include_is_exact_scope() -> None:
    include = library_agent_include("user-1", execution_scope=("org-1", None))

    assert include["AgentGraph"]["include"]["Executions"]["where"] == {
        "userId": "user-1",
        "organizationId": "org-1",
        "teamId": None,
    }
