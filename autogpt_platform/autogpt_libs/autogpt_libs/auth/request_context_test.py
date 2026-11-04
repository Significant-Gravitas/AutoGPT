"""A personal organization's request acts in its default team.

Its rows live there (the database moves any written at org-home), so a request
that names no team must read and write there too.
"""

import sys
import types
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException, Request
from pytest_mock import MockerFixture

from autogpt_libs.auth.dependencies import get_request_context

USER = "user-1"
PAYLOAD = {"sub": USER, "email": "a@b.c"}


def _request(headers: dict | None = None) -> Mock:
    request = Mock(spec=Request)
    request.headers = headers or {}
    request.method = "GET"
    request.url = "http://test/api/library/agents"
    return request


@pytest.fixture
def prisma(mocker: MockerFixture) -> Mock:
    """The deferred ``backend.data.db`` import, with an active org member."""
    db_mod = types.ModuleType("backend.data.db")
    mocker.patch.object(db_mod, "prisma", Mock(), create=True)
    db_mod.prisma.orgmember.find_first = AsyncMock(return_value=Mock(orgId="org-1"))
    db_mod.prisma.teammember.find_first = AsyncMock(
        return_value=Mock(teamId="team-default", isAdmin=True, isBillingManager=False)
    )
    mocker.patch.dict(
        sys.modules,
        {
            "backend": types.ModuleType("backend"),
            "backend.data": types.ModuleType("backend.data"),
            "backend.data.db": db_mod,
        },
    )
    return db_mod.prisma


def _org_member(prisma: Mock, *, personal: bool) -> None:
    member = Mock(status="ACTIVE", isOwner=True, isAdmin=True, isBillingManager=False)
    member.Org = Mock(deletedAt=None, isPersonal=personal)
    prisma.orgmember.find_unique = AsyncMock(return_value=member)


@pytest.mark.asyncio
async def test_a_personal_request_without_a_team_acts_in_the_default_team(
    prisma: Mock,
):
    _org_member(prisma, personal=True)

    ctx = await get_request_context(_request(), PAYLOAD)

    assert (ctx.org_id, ctx.team_id) == ("org-1", "team-default")
    assert ctx.is_team_admin and not ctx.is_team_billing_manager
    prisma.teammember.find_first.assert_awaited_once_with(
        where={
            "userId": USER,
            "status": "ACTIVE",
            "Team": {"orgId": "org-1", "isDefault": True, "archivedAt": None},
        },
        order={"createdAt": "asc"},
    )


@pytest.mark.asyncio
async def test_an_explicit_personal_org_header_gets_the_default_team_too(
    prisma: Mock,
):
    _org_member(prisma, personal=True)

    ctx = await get_request_context(_request({"X-Org-Id": "org-1"}), PAYLOAD)

    assert ctx.team_id == "team-default"


@pytest.mark.asyncio
async def test_a_team_organization_request_without_a_team_stays_at_org_home(
    prisma: Mock,
):
    _org_member(prisma, personal=False)

    ctx = await get_request_context(_request({"X-Org-Id": "org-1"}), PAYLOAD)

    assert ctx.team_id is None
    assert not ctx.is_team_admin
    prisma.teammember.find_first.assert_not_awaited()


@pytest.mark.asyncio
async def test_without_a_live_default_membership_the_request_stays_at_org_home(
    prisma: Mock,
):
    """The database refuses writes there, so nothing is widened silently."""
    _org_member(prisma, personal=True)
    prisma.teammember.find_first = AsyncMock(return_value=None)

    ctx = await get_request_context(_request(), PAYLOAD)

    assert ctx.team_id is None
    assert not ctx.is_team_admin


@pytest.mark.asyncio
async def test_an_explicit_team_is_still_validated_rather_than_replaced(
    prisma: Mock,
):
    _org_member(prisma, personal=True)
    prisma.teammember.find_unique = AsyncMock(return_value=None)

    with pytest.raises(HTTPException) as refused:
        await get_request_context(_request({"X-Team-Id": "team-other"}), PAYLOAD)

    assert refused.value.status_code == 403
    prisma.teammember.find_first.assert_not_awaited()
