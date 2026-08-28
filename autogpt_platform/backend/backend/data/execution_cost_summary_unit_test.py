from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from backend.data.execution_cost_summary import _fetch_by_agent, _fetch_top_runs


PARAMS = (
    "user-1",
    datetime(2026, 8, 1, tzinfo=timezone.utc),
    datetime(2026, 8, 31, tzinfo=timezone.utc),
    "org-1",
    None,
)


@pytest.mark.asyncio
async def test_agent_cost_rollups_keep_same_graph_in_each_team_distinct():
    rows = [
        {
            "graph_id": "shared-graph",
            "organization_id": "org-1",
            "team_id": "team-a",
            "cost_cents": 120,
            "run_count": 2,
        },
        {
            "graph_id": "shared-graph",
            "organization_id": "org-1",
            "team_id": "team-b",
            "cost_cents": 80,
            "run_count": 1,
        },
    ]

    query = AsyncMock(return_value=rows)
    with patch("backend.data.execution_cost_summary.query_raw_with_schema", query):
        rollups = await _fetch_by_agent(PARAMS)

    sql = query.await_args.args[0]
    assert 'GROUP BY "agentGraphId", "organizationId", "teamId"' in sql
    assert [(row.graph_id, row.organization_id, row.team_id) for row in rollups] == [
        ("shared-graph", "org-1", "team-a"),
        ("shared-graph", "org-1", "team-b"),
    ]


@pytest.mark.asyncio
async def test_top_runs_include_the_execution_tenant_scope():
    started_at = datetime(2026, 8, 27, tzinfo=timezone.utc)
    query = AsyncMock(
        return_value=[
            {
                "execution_id": "run-1",
                "graph_id": "shared-graph",
                "organization_id": "org-1",
                "team_id": "team-b",
                "cost_cents": 80,
                "started_at": started_at,
                "status": "COMPLETED",
                "duration_seconds": 2.5,
                "node_error_count": 0,
            }
        ]
    )
    with patch("backend.data.execution_cost_summary.query_raw_with_schema", query):
        runs = await _fetch_top_runs(PARAMS, 10)

    assert len(runs) == 1
    assert runs[0].organization_id == "org-1"
    assert runs[0].team_id == "team-b"
