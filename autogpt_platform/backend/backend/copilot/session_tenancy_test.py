"""Unit tests for per-turn session org/team membership re-verification.

Prisma is mocked at the boundary the helper uses
(``backend.copilot.session_tenancy.prisma``); the tests exercise the real
``verify_session_org_membership`` branching in isolation.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import OrgMemberStatus

from backend.copilot import session_tenancy
from backend.copilot.session_tenancy import (
    SessionOrgMembershipRevoked,
    verify_session_org_membership,
)


def _patch_prisma(mocker, *, org_member, team_member):
    prisma = MagicMock()
    prisma.orgmember.find_unique = AsyncMock(return_value=org_member)
    prisma.teammember.find_unique = AsyncMock(return_value=team_member)
    mocker.patch.object(session_tenancy, "prisma", prisma)
    return prisma


def _member(status) -> MagicMock:
    return MagicMock(status=status)


@pytest.mark.asyncio
async def test_missing_org_member_raises(mocker):
    """No OrgMember row for the session's org → revoked."""
    _patch_prisma(mocker, org_member=None, team_member=None)
    with pytest.raises(SessionOrgMembershipRevoked) as exc:
        await verify_session_org_membership(
            user_id="u1", organization_id="org-1", team_id=None
        )
    assert exc.value.organization_id == "org-1"


@pytest.mark.asyncio
async def test_suspended_org_member_raises(mocker):
    """A SUSPENDED (non-ACTIVE) OrgMember → revoked."""
    _patch_prisma(
        mocker,
        org_member=_member(OrgMemberStatus.SUSPENDED),
        team_member=None,
    )
    with pytest.raises(SessionOrgMembershipRevoked):
        await verify_session_org_membership(
            user_id="u1", organization_id="org-1", team_id=None
        )


@pytest.mark.asyncio
async def test_active_org_no_team_skips_team_lookup(mocker):
    """ACTIVE org membership, no team on the session → returns None and never
    runs the team lookup (only one indexed read)."""
    prisma = _patch_prisma(
        mocker,
        org_member=_member(OrgMemberStatus.ACTIVE),
        team_member=None,
    )
    result = await verify_session_org_membership(
        user_id="u1", organization_id="org-1", team_id=None
    )
    assert result is None
    prisma.teammember.find_unique.assert_not_awaited()


@pytest.mark.asyncio
async def test_active_org_and_active_team_returns_team(mocker):
    """Both memberships ACTIVE → the team is preserved."""
    _patch_prisma(
        mocker,
        org_member=_member(OrgMemberStatus.ACTIVE),
        team_member=_member(OrgMemberStatus.ACTIVE),
    )
    result = await verify_session_org_membership(
        user_id="u1", organization_id="org-1", team_id="team-1"
    )
    assert result == "team-1"


@pytest.mark.asyncio
async def test_active_org_missing_team_member_strips_team(mocker):
    """ACTIVE org but no TeamMember row → team stripped to org-home (None)."""
    _patch_prisma(
        mocker,
        org_member=_member(OrgMemberStatus.ACTIVE),
        team_member=None,
    )
    result = await verify_session_org_membership(
        user_id="u1", organization_id="org-1", team_id="team-1"
    )
    assert result is None


@pytest.mark.asyncio
async def test_active_org_suspended_team_member_strips_team(mocker):
    """ACTIVE org but SUSPENDED team membership → team stripped, no raise
    (team removal is routine; only org removal is access revocation)."""
    _patch_prisma(
        mocker,
        org_member=_member(OrgMemberStatus.ACTIVE),
        team_member=_member(OrgMemberStatus.SUSPENDED),
    )
    result = await verify_session_org_membership(
        user_id="u1", organization_id="org-1", team_id="team-1"
    )
    assert result is None
