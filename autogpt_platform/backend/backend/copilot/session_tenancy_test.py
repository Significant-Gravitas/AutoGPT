"""Unit tests for per-turn session org/team membership re-verification.

The policy under test spans two modules — ``session_tenancy``'s hard-gate /
soft-strip branching and ``orgs.db.get_session_tenancy_membership``'s
ACTIVE + ``Org.deletedAt`` + ``Team.orgId`` predicates — so the tests drive
the real chain and stub only the two indexed reads.  The accessor is pointed
at the real orgs DB module so ``db.is_connected()`` (false under pytest)
cannot silently swap in an RPC client.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import OrgMemberStatus

from backend.api.features.orgs import db as orgs_db_module
from backend.copilot import session_tenancy
from backend.copilot.session_tenancy import (
    SessionOrgMembershipRevoked,
    resolve_session_tenancy,
)


def _patch_membership_reads(mocker, *, org_member, team_member):
    prisma = MagicMock()
    prisma.orgmember.find_unique = AsyncMock(return_value=org_member)
    prisma.teammember.find_unique = AsyncMock(return_value=team_member)
    mocker.patch.object(orgs_db_module, "prisma", prisma)
    mocker.patch.object(session_tenancy, "orgs_db", return_value=orgs_db_module)
    return prisma


def _org_member(
    status=OrgMemberStatus.ACTIVE, *, deleted_at: datetime | None = None
) -> MagicMock:
    """An OrgMember row with its ``Org`` relation loaded."""
    return MagicMock(status=status, Org=MagicMock(deletedAt=deleted_at))


def _team_member(status=OrgMemberStatus.ACTIVE, *, team_org_id="org-1") -> MagicMock:
    """A TeamMember row with its ``Team`` relation loaded."""
    return MagicMock(status=status, Team=MagicMock(orgId=team_org_id))


@pytest.mark.asyncio
async def test_missing_org_member_raises(mocker):
    """No OrgMember row for the session's org → revoked."""
    _patch_membership_reads(mocker, org_member=None, team_member=None)
    with pytest.raises(SessionOrgMembershipRevoked) as exc:
        await resolve_session_tenancy(
            user_id="u1", organization_id="org-1", team_id=None
        )
    assert exc.value.organization_id == "org-1"


@pytest.mark.asyncio
async def test_suspended_org_member_raises(mocker):
    """A SUSPENDED (non-ACTIVE) OrgMember → revoked."""
    _patch_membership_reads(
        mocker,
        org_member=_org_member(OrgMemberStatus.SUSPENDED),
        team_member=None,
    )
    with pytest.raises(SessionOrgMembershipRevoked):
        await resolve_session_tenancy(
            user_id="u1", organization_id="org-1", team_id=None
        )


@pytest.mark.asyncio
async def test_soft_deleted_org_raises(mocker):
    """``delete_org`` only sets ``Organization.deletedAt`` and leaves member
    rows ACTIVE — a session anchored to a deleted org must still be revoked,
    matching the 403 ``get_request_context`` raises for the header org."""
    _patch_membership_reads(
        mocker,
        org_member=_org_member(deleted_at=datetime.now(timezone.utc)),
        team_member=None,
    )
    with pytest.raises(SessionOrgMembershipRevoked):
        await resolve_session_tenancy(
            user_id="u1", organization_id="org-1", team_id=None
        )


@pytest.mark.asyncio
async def test_missing_org_relation_raises(mocker):
    """Defensive: an OrgMember whose ``Org`` relation is absent (dangling row)
    must fail closed rather than pass the gate."""
    _patch_membership_reads(
        mocker,
        org_member=MagicMock(status=OrgMemberStatus.ACTIVE, Org=None),
        team_member=None,
    )
    with pytest.raises(SessionOrgMembershipRevoked):
        await resolve_session_tenancy(
            user_id="u1", organization_id="org-1", team_id=None
        )


@pytest.mark.asyncio
async def test_active_org_no_team_skips_team_lookup(mocker):
    """ACTIVE org membership, no team on the session → returns None and never
    runs the team lookup (only one indexed read)."""
    prisma = _patch_membership_reads(
        mocker,
        org_member=_org_member(),
        team_member=None,
    )
    result = await resolve_session_tenancy(
        user_id="u1", organization_id="org-1", team_id=None
    )
    assert result is None
    prisma.teammember.find_unique.assert_not_awaited()
    # Pin the unique-lookup payload: a wrong key or swapped id would still
    # return the mocked row, so assert the where-clause and the Org include.
    prisma.orgmember.find_unique.assert_awaited_once_with(
        where={"orgId_userId": {"orgId": "org-1", "userId": "u1"}},
        include={"Org": True},
    )


@pytest.mark.asyncio
async def test_active_org_and_active_team_returns_team(mocker):
    """Both memberships ACTIVE and the team belongs to the org → preserved."""
    prisma = _patch_membership_reads(
        mocker,
        org_member=_org_member(),
        team_member=_team_member(),
    )
    result = await resolve_session_tenancy(
        user_id="u1", organization_id="org-1", team_id="team-1"
    )
    assert result == "team-1"
    prisma.teammember.find_unique.assert_awaited_once_with(
        where={"teamId_userId": {"teamId": "team-1", "userId": "u1"}},
        include={"Team": True},
    )


@pytest.mark.asyncio
async def test_active_org_missing_team_member_strips_team(mocker):
    """ACTIVE org but no TeamMember row → team stripped to org-home (None)."""
    _patch_membership_reads(
        mocker,
        org_member=_org_member(),
        team_member=None,
    )
    result = await resolve_session_tenancy(
        user_id="u1", organization_id="org-1", team_id="team-1"
    )
    assert result is None


@pytest.mark.asyncio
async def test_active_org_suspended_team_member_strips_team(mocker):
    """ACTIVE org but SUSPENDED team membership → team stripped, no raise
    (team removal is routine; only org removal is access revocation)."""
    _patch_membership_reads(
        mocker,
        org_member=_org_member(),
        team_member=_team_member(OrgMemberStatus.SUSPENDED),
    )
    result = await resolve_session_tenancy(
        user_id="u1", organization_id="org-1", team_id="team-1"
    )
    assert result is None


@pytest.mark.asyncio
async def test_team_outside_session_org_strips_team(mocker):
    """An ACTIVE TeamMember whose team no longer belongs to the session's org
    must not be honoured — otherwise the turn would be attributed to a team
    under a different org.  ``get_request_context`` strips this the same way."""
    _patch_membership_reads(
        mocker,
        org_member=_org_member(),
        team_member=_team_member(team_org_id="some-other-org"),
    )
    result = await resolve_session_tenancy(
        user_id="u1", organization_id="org-1", team_id="team-1"
    )
    assert result is None


@pytest.mark.asyncio
async def test_revoked_org_skips_team_lookup(mocker):
    """A failed org gate short-circuits: no point spending the second read."""
    prisma = _patch_membership_reads(
        mocker,
        org_member=None,
        team_member=_team_member(),
    )
    with pytest.raises(SessionOrgMembershipRevoked):
        await resolve_session_tenancy(
            user_id="u1", organization_id="org-1", team_id="team-1"
        )
    prisma.teammember.find_unique.assert_not_awaited()
