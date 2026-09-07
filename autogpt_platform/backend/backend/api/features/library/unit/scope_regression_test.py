from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import prisma.enums
import prisma.models
import pytest

from backend.api.features.library import db, model
from backend.api.features.library._schedule_info import _fetch_schedule_info


@pytest.mark.asyncio
@pytest.mark.parametrize("team_id", [None, "team-1"])
async def test_exact_graph_lookup_selects_callers_installation(mocker, team_id):
    own = MagicMock(
        id="own-installation",
        userId="caller",
        agentGraphId="graph-1",
        organizationId="org-1",
        teamId=team_id,
    )
    teammate = MagicMock(
        id="teammate-installation",
        userId="teammate",
        agentGraphId="graph-1",
        organizationId="org-1",
        teamId=team_id,
    )

    async def find_first(*, where, include):
        if "id" in where:
            return own if where["id"] == own.id else teammate
        return own if where.get("userId") == "caller" else teammate

    client = MagicMock(find_first=AsyncMock(side_effect=find_first))
    mocker.patch("prisma.models.LibraryAgent.prisma", return_value=client)
    mocker.patch.object(db.graph_db, "get_sub_graphs", AsyncMock(return_value=[]))
    mocker.patch.object(db, "_fetch_schedule_info", AsyncMock(return_value={}))
    mocker.patch.object(
        model.LibraryAgent,
        "from_db",
        side_effect=lambda agent, **kwargs: SimpleNamespace(id=agent.id),
    )

    result = await db.get_library_agent_by_graph_id(
        "caller",
        "graph-1",
        organization_id="org-1",
        team_id_restriction=team_id,
        exact_scope=True,
    )

    assert result is not None
    assert result.id == "own-installation"


@pytest.mark.asyncio
async def test_library_keeps_schedules_from_older_graph_versions_in_own_scope(mocker):
    scheduler = MagicMock(
        get_graph_execution_schedules=AsyncMock(
            return_value=[
                _schedule("org-1", "team-1", 1, "2026-09-08T10:00:00+00:00"),
                _schedule("org-1", "team-1", 2, "2026-09-08T11:00:00+00:00"),
                _schedule("org-1", "team-2", 1, "2026-09-08T08:00:00+00:00"),
                _schedule("org-2", "team-1", 1, "2026-09-08T07:00:00+00:00"),
            ]
        )
    )
    mocker.patch(
        "backend.api.features.library._schedule_info.get_scheduler_client",
        return_value=scheduler,
    )
    schedules = await _fetch_schedule_info("caller", exact_scope=True)

    scheduled = model.LibraryAgent.from_db(
        _library_agent("org-1", "team-1", version=3), schedule_info=schedules
    )
    unscheduled = model.LibraryAgent.from_db(
        _library_agent("org-1", None, version=3), schedule_info=schedules
    )

    assert scheduled.is_scheduled
    assert scheduled.next_scheduled_run == "2026-09-08T10:00:00+00:00"
    assert not unscheduled.is_scheduled
    assert unscheduled.next_scheduled_run is None
    scheduler.get_graph_execution_schedules.assert_awaited_once_with(
        graph_id=None, user_id="caller"
    )


def _schedule(organization_id, team_id, version, next_run):
    return SimpleNamespace(
        graph_id="graph-1",
        graph_version=version,
        organization_id=organization_id,
        team_id=team_id,
        next_run_time=next_run,
    )


def _library_agent(organization_id, team_id, version):
    now = datetime.now(timezone.utc)
    return prisma.models.LibraryAgent.model_validate(
        dict(
            id="own-installation",
            scopeKey="org-1:team-1",
            userId="caller",
            agentGraphId="graph-1",
            agentGraphVersion=version,
            settings="{}",
            isCreatedByUser=True,
            isDeleted=False,
            isArchived=False,
            isHidden=False,
            createdAt=now,
            updatedAt=now,
            isFavorite=False,
            useGraphIsActiveVersion=True,
            organizationId=organization_id,
            teamId=team_id,
            visibility=prisma.enums.ResourceVisibility.PRIVATE,
            AgentGraph=prisma.models.AgentGraph(
                id="graph-1",
                version=version,
                name="Agent",
                description="Description",
                userId="caller",
                isActive=True,
                createdAt=now,
                visibility=prisma.enums.ResourceVisibility.PRIVATE,
            ),
        )
    )
