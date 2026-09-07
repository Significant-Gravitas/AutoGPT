from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data import activity_event


@pytest.mark.asyncio
async def test_activity_events_restrict_to_selected_workspace(mocker):
    rows = AsyncMock(return_value=[])
    mocker.patch.object(
        activity_event.prisma.models.ActivityEvent,
        "prisma",
        return_value=MagicMock(find_many=rows),
    )
    await activity_event.list_activity_events(
        "user-1",
        datetime.now(timezone.utc),
        organization_id="org-1",
        team_id="team-1",
        team_ids=["team-1"],
    )
    where = rows.await_args.kwargs["where"]
    assert where["userId"] == "user-1"
    assert where["organizationId"] == "org-1"
    assert where["teamId"] == "team-1"


@pytest.mark.asyncio
async def test_org_events_exclude_inaccessible_teams(mocker):
    rows = AsyncMock(return_value=[])
    mocker.patch.object(
        activity_event.prisma.models.ActivityEvent,
        "prisma",
        return_value=MagicMock(find_many=rows),
    )
    await activity_event.list_activity_events(
        "user-1",
        datetime.now(timezone.utc),
        organization_id="org-1",
        team_ids=["team-visible"],
    )
    where = rows.await_args.kwargs["where"]
    assert where["organizationId"] == "org-1"
    assert where["OR"] == [
        {"teamId": None},
        {"teamId": {"in": ["team-visible"]}},
    ]


@pytest.mark.asyncio
async def test_org_events_without_team_memberships_only_show_org_home(mocker):
    rows = AsyncMock(return_value=[])
    mocker.patch.object(
        activity_event.prisma.models.ActivityEvent,
        "prisma",
        return_value=MagicMock(find_many=rows),
    )
    await activity_event.list_activity_events(
        "user-1",
        datetime.now(timezone.utc),
        organization_id="org-1",
        team_ids=[],
    )
    assert rows.await_args.kwargs["where"]["teamId"] is None
