"""The workspace routes check a join policy before any database call."""

from unittest.mock import AsyncMock

import pytest
from autogpt_libs.auth.models import RequestContext
from prisma.enums import TeamJoinPolicy
from pytest_mock import MockerFixture

from backend.api.features.orgs import team_routes
from backend.api.features.orgs.team_model import CreateTeamRequest, UpdateTeamRequest

ORG_ID = "org-1"
WS_ID = "ws-1"


def _owner_ctx() -> RequestContext:
    return RequestContext(
        user_id="user-1",
        org_id=ORG_ID,
        team_id=WS_ID,
        is_org_owner=True,
        is_org_admin=False,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


@pytest.mark.asyncio
async def test_create_refuses_an_unknown_join_policy_before_writing(
    mocker: MockerFixture,
):
    create = mocker.patch.object(team_routes.team_db, "create_team", AsyncMock())

    with pytest.raises(ValueError):
        await team_routes.create_team(
            org_id=ORG_ID,
            request=CreateTeamRequest(name="Dev", join_policy="SECRET"),
            ctx=_owner_ctx(),
        )
    create.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_refuses_an_unknown_join_policy_before_writing(
    mocker: MockerFixture,
):
    mocker.patch.object(team_routes.team_db, "get_team", AsyncMock())
    update = mocker.patch.object(team_routes.team_db, "update_team", AsyncMock())

    with pytest.raises(ValueError):
        await team_routes.update_team(
            org_id=ORG_ID,
            ws_id=WS_ID,
            request=UpdateTeamRequest(join_policy="SECRET"),
            ctx=_owner_ctx(),
        )
    update.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_writes_only_the_fields_the_request_set(mocker: MockerFixture):
    mocker.patch.object(team_routes.team_db, "get_team", AsyncMock())
    update = mocker.patch.object(team_routes.team_db, "update_team", AsyncMock())

    await team_routes.update_team(
        org_id=ORG_ID,
        ws_id=WS_ID,
        request=UpdateTeamRequest(join_policy="PRIVATE"),
        ctx=_owner_ctx(),
    )

    update.assert_awaited_once_with(WS_ID, {"joinPolicy": TeamJoinPolicy.PRIVATE})
