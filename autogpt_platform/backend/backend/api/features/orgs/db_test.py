"""Tests for ``resolve_member_tenancy``: which org/team something a user
creates or runs keeps from the one it would inherit."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.orgs import db as orgs_db

USER = "user-1"


def _org_member(*, status: str = "ACTIVE", deleted: bool = False):
    return SimpleNamespace(
        status=status,
        Org=SimpleNamespace(deletedAt="2026-10-01" if deleted else None),
    )


def _team_member(*, status: str = "ACTIVE", org_id: str = "org-1"):
    return SimpleNamespace(status=status, Team=SimpleNamespace(orgId=org_id))


@pytest.fixture
def db(mocker):
    client = MagicMock()
    client.orgmember.find_unique = AsyncMock(return_value=None)
    client.teammember.find_unique = AsyncMock(return_value=None)
    mocker.patch.object(orgs_db, "prisma", client)
    return client


@pytest.fixture(autouse=True)
def role_refusal(mocker):
    """The user's role covers workflows unless a test says it doesn't; which
    roles do is `resource_role_refusal`'s own test's business."""
    return mocker.patch.object(
        orgs_db, "resource_role_refusal", new=AsyncMock(return_value=None)
    )


@pytest.fixture
def default_tenancy(mocker):
    return mocker.patch.object(
        orgs_db,
        "resolve_default_tenancy",
        new=AsyncMock(return_value=("own-org", "own-team")),
    )


@pytest.mark.asyncio
async def test_a_member_keeps_the_org_and_their_team(db, default_tenancy, role_refusal):
    db.orgmember.find_unique.return_value = _org_member()
    db.teammember.find_unique.return_value = _team_member()

    tenancy = await orgs_db.resolve_member_tenancy(USER, "org-1", "team-1")

    assert tenancy == ("org-1", "team-1")
    role_refusal.assert_awaited_once_with(USER, "org-1", "team-1")
    db.orgmember.find_unique.assert_awaited_once_with(
        where={"orgId_userId": {"orgId": "org-1", "userId": USER}},
        include={"Org": True},
    )
    db.teammember.find_unique.assert_awaited_once_with(
        where={"teamId_userId": {"teamId": "team-1", "userId": USER}},
        include={"Team": True},
    )
    default_tenancy.assert_not_awaited()


@pytest.mark.asyncio
async def test_someone_elses_org_is_replaced_by_the_users_own(db, default_tenancy):
    """A user who isn't a member of the org gets their own org and team."""
    tenancy = await orgs_db.resolve_member_tenancy(USER, "other-org", "other-team")

    assert tenancy == ("own-org", "own-team")
    default_tenancy.assert_awaited_once_with(USER)
    db.teammember.find_unique.assert_not_awaited()


@pytest.mark.parametrize(
    "membership",
    [
        _org_member(status="INVITED"),
        _org_member(status="SUSPENDED"),
        _org_member(status="REMOVED"),
        _org_member(deleted=True),
    ],
    ids=["invited", "suspended", "removed", "org-deleted"],
)
@pytest.mark.asyncio
async def test_a_membership_that_isnt_active_counts_as_none(
    db, default_tenancy, membership
):
    """Even with the team's membership still in place: it counts only with
    the org's."""
    db.orgmember.find_unique.return_value = membership
    db.teammember.find_unique.return_value = _team_member()

    assert await orgs_db.resolve_member_tenancy(USER, "org-1", "team-1") == (
        "own-org",
        "own-team",
    )


@pytest.mark.parametrize(
    "team_membership",
    [None, _team_member(status="REMOVED"), _team_member(org_id="other-org")],
    ids=["not-on-team", "removed-from-team", "team-of-another-org"],
)
@pytest.mark.asyncio
async def test_a_team_the_user_isnt_on_is_replaced_by_the_users_own(
    db, default_tenancy, team_membership
):
    """Everyone in an org sees its home, so something made for a team doesn't
    move there when the user isn't on that team."""
    db.orgmember.find_unique.return_value = _org_member()
    db.teammember.find_unique.return_value = team_membership

    assert await orgs_db.resolve_member_tenancy(USER, "org-1", "team-1") == (
        "own-org",
        "own-team",
    )
    default_tenancy.assert_awaited_once_with(USER)


@pytest.mark.parametrize("team_id", ["team-1", None], ids=["team", "org-home"])
@pytest.mark.asyncio
async def test_a_role_that_covers_only_billing_counts_as_no_membership(
    db, default_tenancy, role_refusal, team_id
):
    """It covers spend, not creating or running workflows, so a run, preset
    or trigger doesn't land in that org or team, however active the
    membership is."""
    db.orgmember.find_unique.return_value = _org_member()
    db.teammember.find_unique.return_value = _team_member()
    role_refusal.return_value = "Your role here covers its billing"

    assert await orgs_db.resolve_member_tenancy(USER, "org-1", team_id) == (
        "own-org",
        "own-team",
    )
    role_refusal.assert_awaited_once_with(USER, "org-1", team_id)
    default_tenancy.assert_awaited_once_with(USER)


@pytest.mark.asyncio
async def test_the_membership_check_reads_no_role(db, default_tenancy):
    """Roles are decided once, by `resource_role_refusal`. The membership check
    behind it answers for membership alone, as its other callers expect: an
    active billing manager is an active member."""
    db.orgmember.find_unique.return_value = SimpleNamespace(
        **vars(_org_member()), isOwner=False, isAdmin=False, isBillingManager=True
    )
    db.teammember.find_unique.return_value = SimpleNamespace(
        **vars(_team_member()), isAdmin=False, isBillingManager=True
    )

    assert await orgs_db.resolve_member_tenancy(USER, "org-1", "team-1") == (
        "org-1",
        "team-1",
    )
    default_tenancy.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_member_given_no_team_keeps_the_org_home(db, default_tenancy):
    db.orgmember.find_unique.return_value = _org_member()

    assert await orgs_db.resolve_member_tenancy(USER, "org-1", None) == (
        "org-1",
        None,
    )
    db.teammember.find_unique.assert_not_awaited()


@pytest.mark.asyncio
async def test_nothing_to_inherit_stays_untagged(db, default_tenancy, role_refusal):
    assert await orgs_db.resolve_member_tenancy(USER, None, None) == (None, None)
    db.orgmember.find_unique.assert_not_awaited()
    role_refusal.assert_not_awaited()
    default_tenancy.assert_not_awaited()
