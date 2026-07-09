"""Org/team visibility where-clause tests for execution reads."""

from unittest.mock import AsyncMock

import prisma.models
import pytest

from backend.data.execution import (
    get_graph_execution,
    get_graph_execution_meta,
    get_graph_executions_paginated,
)

VISIBILITY_AND = [
    {
        "OR": [
            {
                "userId": "u-1",
                "OR": [{"organizationId": "org-1"}, {"organizationId": None}],
            },
            {"organizationId": "org-1", "teamId": None},
            {"organizationId": "org-1", "teamId": {"in": ["team-a"]}},
        ]
    }
]


@pytest.fixture
def mock_exec_client(mocker):
    client = AsyncMock()
    client.find_first.return_value = None
    client.find_many.return_value = []
    client.count.return_value = 0
    mocker.patch.object(
        prisma.models.AgentGraphExecution, "prisma", return_value=client
    )
    mocker.patch(
        "backend.data.execution.get_user_team_ids",
        AsyncMock(return_value=["team-a"]),
    )
    return client


@pytest.mark.asyncio
async def test_execution_meta_org_visibility(mock_exec_client):
    """A member can fetch run details the list endpoints show them."""
    await get_graph_execution_meta("u-1", "exec-1", organization_id="org-1")

    where = mock_exec_client.find_first.call_args.kwargs["where"]
    assert "userId" not in where
    assert where["AND"] == VISIBILITY_AND


@pytest.mark.asyncio
async def test_execution_meta_without_org_strict_ownership(mock_exec_client):
    await get_graph_execution_meta("u-1", "exec-1")

    where = mock_exec_client.find_first.call_args.kwargs["where"]
    assert where["userId"] == "u-1"
    assert "AND" not in where


@pytest.mark.asyncio
async def test_execution_get_org_visibility(mock_exec_client):
    await get_graph_execution("u-1", "exec-1", organization_id="org-1")

    where = mock_exec_client.find_first.call_args.kwargs["where"]
    assert "userId" not in where
    assert where["AND"] == VISIBILITY_AND


@pytest.mark.asyncio
async def test_executions_paginated_org_visibility_coexists_with_status_or(
    mock_exec_client,
):
    """The visibility predicate nests under AND so the statuses OR-clause
    can't clobber (or be clobbered by) it."""
    from backend.data.execution import ExecutionStatus

    await get_graph_executions_paginated(
        user_id="u-1",
        organization_id="org-1",
        statuses=[ExecutionStatus.RUNNING],
    )

    where = mock_exec_client.find_many.call_args.kwargs["where"]
    assert where["AND"] == VISIBILITY_AND
    assert where["OR"] == [{"executionStatus": ExecutionStatus.RUNNING}]
    assert "userId" not in where
