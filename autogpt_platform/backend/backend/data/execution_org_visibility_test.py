"""Org/team visibility where-clause tests for execution reads."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import prisma.models
import pytest
from prisma.enums import ResourceVisibility

from backend.data.execution import (
    _create_graph_execution_locked,
    get_graph_execution,
    get_graph_execution_meta,
    get_graph_executions_paginated,
)

VISIBILITY_AND = [
    {
        "OR": [
            {
                "userId": "u-1",
                "organizationId": None,
            },
            {"organizationId": "org-1", "teamId": None},
            {"organizationId": "org-1", "teamId": {"in": ["team-a"]}},
        ]
    },
    {"OR": [{"expertId": None}, {"userId": "u-1"}]},
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


@pytest.mark.asyncio
async def test_paginated_list_excludes_legacy_shared_expert_runs(mock_exec_client):
    await get_graph_executions_paginated(
        user_id="u-1",
        organization_id="org-1",
    )

    count_where = mock_exec_client.count.call_args.kwargs["where"]
    list_where = mock_exec_client.find_many.call_args.kwargs["where"]
    expected_owner_guard = {"OR": [{"expertId": None}, {"userId": "u-1"}]}
    assert expected_owner_guard in count_where["AND"]
    assert expected_owner_guard in list_where["AND"]


@pytest.mark.asyncio
async def test_meta_excludes_legacy_shared_expert_runs(mock_exec_client):
    await get_graph_execution_meta("u-1", "exec-1", organization_id="org-1")

    where = mock_exec_client.find_first.call_args.kwargs["where"]
    assert {"OR": [{"expertId": None}, {"userId": "u-1"}]} in where["AND"]


@pytest.mark.asyncio
async def test_detail_excludes_legacy_shared_expert_runs(mock_exec_client):
    await get_graph_execution("u-1", "exec-1", organization_id="org-1")

    where = mock_exec_client.find_first.call_args.kwargs["where"]
    assert {"OR": [{"expertId": None}, {"userId": "u-1"}]} in where["AND"]


@pytest.mark.asyncio
async def test_create_execution_rejects_non_private_expert_before_write(mocker):
    expert_client = AsyncMock()
    expert_client.find_first.return_value = None
    execution_client = AsyncMock()
    mocker.patch.object(prisma.models.Expert, "prisma", return_value=expert_client)
    mocker.patch.object(
        prisma.models.AgentGraphExecution,
        "prisma",
        return_value=execution_client,
    )

    with pytest.raises(ValueError, match="Expert #shared-expert is unavailable"):
        await _create_graph_execution_locked(
            graph_id="graph-1",
            graph_version=1,
            starting_nodes_input=[],
            inputs={},
            user_id="owner",
            expert_id="shared-expert",
        )

    expert_client.find_first.assert_awaited_once_with(
        where={
            "id": "shared-expert",
            "ownerUserId": "owner",
            "isTemplate": False,
            "isArchived": False,
            "visibility": ResourceVisibility.PRIVATE,
        }
    )
    execution_client.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_execution_rejects_cross_scope_preset_before_write(mocker):
    preset_client = AsyncMock()
    preset_client.find_first.return_value = None
    execution_client = AsyncMock()
    mocker.patch.object(prisma.models.AgentPreset, "prisma", return_value=preset_client)
    mocker.patch.object(
        prisma.models.AgentGraphExecution,
        "prisma",
        return_value=execution_client,
    )

    with pytest.raises(ValueError, match="Preset #preset-team-b is unavailable"):
        await _create_graph_execution_locked(
            graph_id="graph-1",
            graph_version=3,
            starting_nodes_input=[],
            inputs={},
            user_id="user-1",
            preset_id="preset-team-b",
            organization_id="org-1",
            team_id="team-a",
        )

    preset_client.find_first.assert_awaited_once_with(
        where={
            "id": "preset-team-b",
            "userId": "user-1",
            "agentGraphId": "graph-1",
            "agentGraphVersion": 3,
            "organizationId": "org-1",
            "teamId": "team-a",
            "isDeleted": False,
        }
    )
    execution_client.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_create_execution_updates_last_run_only_in_exact_scope(mocker):
    created_at = datetime.now(timezone.utc)
    persisted = mocker.MagicMock(createdAt=created_at)
    execution_client = AsyncMock()
    execution_client.create.return_value = persisted
    library_client = AsyncMock()
    mocker.patch.object(
        prisma.models.AgentGraphExecution,
        "prisma",
        return_value=execution_client,
    )
    mocker.patch.object(
        prisma.models.LibraryAgent,
        "prisma",
        return_value=library_client,
    )
    converted = mocker.MagicMock()
    mocker.patch(
        "backend.data.execution.GraphExecutionWithNodes.from_db",
        return_value=converted,
    )

    result = await _create_graph_execution_locked(
        graph_id="graph-1",
        graph_version=3,
        starting_nodes_input=[],
        inputs={},
        user_id="user-1",
        organization_id="org-1",
        team_id="team-b",
    )

    assert result is converted
    library_client.update_many.assert_awaited_once_with(
        where={
            "agentGraphId": "graph-1",
            "agentGraphVersion": 3,
            "userId": "user-1",
            "organizationId": "org-1",
            "teamId": "team-b",
            "OR": [
                {"lastRunAt": None},
                {"lastRunAt": {"lt": created_at}},
            ],
        },
        data={"lastRunAt": created_at},
    )
