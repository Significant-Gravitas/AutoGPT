"""Tests for the shared org/team visibility predicate."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from autogpt_libs.auth.permissions import OrgAction, TeamAction

from backend.data import tenancy
from backend.data.tenancy import (
    BILLING_ONLY_ORG_ROLE,
    BILLING_ONLY_TEAM_ROLE,
    get_user_team_ids,
    organization_filter,
    require_role_where_resource_lives,
    resource_role_refusal,
    visibility_filter,
    writable_organization_ids,
)
from backend.util.exceptions import NotAuthorizedError


def test_no_org_degrades_to_personal_ownership():
    assert visibility_filter("u1", None, []) == {"userId": "u1"}


def test_org_filter_covers_own_orghome_and_team_rows():
    where = visibility_filter("u1", "org-1", ["team-a", "team-b"])
    assert where == {
        "OR": [
            {
                "userId": "u1",
                "OR": [{"organizationId": "org-1"}, {"organizationId": None}],
            },
            {"organizationId": "org-1", "teamId": None},
            {"organizationId": "org-1", "teamId": {"in": ["team-a", "team-b"]}},
        ]
    }


def test_org_filter_without_teams_omits_team_clause():
    where = visibility_filter("u1", "org-1", [])
    assert where == {
        "OR": [
            {
                "userId": "u1",
                "OR": [{"organizationId": "org-1"}, {"organizationId": None}],
            },
            {"organizationId": "org-1", "teamId": None},
        ]
    }


def test_custom_field_names():
    where = visibility_filter(
        "u1",
        "org-1",
        [],
        user_field="owningUserId",
        org_field="owningOrgId",
    )
    assert where["OR"][0] == {
        "owningUserId": "u1",
        "OR": [{"owningOrgId": "org-1"}, {"owningOrgId": None}],
    }


def _org_member(*, owner=False, admin=False, billing=False) -> SimpleNamespace:
    return SimpleNamespace(isOwner=owner, isAdmin=admin, isBillingManager=billing)


def _team_member(team_id: str, *, admin=False, billing=False) -> SimpleNamespace:
    return SimpleNamespace(teamId=team_id, isAdmin=admin, isBillingManager=billing)


@pytest.fixture
def members(mocker):
    """The org and team membership rows the lookups below read."""
    client = MagicMock()
    client.orgmember.find_first = AsyncMock(return_value=_org_member())
    client.teammember.find_many = AsyncMock(return_value=[])
    client.teammember.find_first = AsyncMock(return_value=None)
    mocker.patch.object(tenancy, "prisma", client)
    return client


@pytest.mark.asyncio
async def test_get_user_team_ids_filters_active_org_memberships(members):
    members.teammember.find_many.return_value = [_team_member("team-a")]

    result = await get_user_team_ids("u1", "org-1")

    assert result == ["team-a"]
    where = members.teammember.find_many.call_args.kwargs["where"]
    assert where["userId"] == "u1"
    assert where["status"] == "ACTIVE"
    assert where["Team"] == {"is": {"orgId": "org-1"}}
    assert members.orgmember.find_first.call_args.kwargs["where"] == {
        "userId": "u1",
        "orgId": "org-1",
        "status": "ACTIVE",
    }


@pytest.mark.asyncio
async def test_get_user_team_ids_leaves_out_teams_where_the_user_only_manages_billing(
    members,
):
    members.teammember.find_many.return_value = [
        _team_member("member-of"),
        _team_member("billing-only", billing=True),
        _team_member("admin-and-billing", admin=True, billing=True),
        _team_member("admin-of", admin=True),
    ]

    assert await get_user_team_ids("u1", "org-1") == [
        "member-of",
        "admin-and-billing",
        "admin-of",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "org_member, teams",
    [
        (None, []),
        (_org_member(billing=True), []),
        (_org_member(owner=True, billing=True), ["team-a"]),
        (_org_member(admin=True, billing=True), ["team-a"]),
        (_org_member(), ["team-a"]),
    ],
    ids=[
        "not-an-active-member",
        "org-billing-manager",
        "owner-and-billing-manager",
        "admin-and-billing-manager",
        "plain-member",
    ],
)
async def test_get_user_team_ids_follows_the_org_role(members, org_member, teams):
    """An org billing manager sees no team's rows, whatever teams they're on."""
    members.orgmember.find_first.return_value = org_member
    members.teammember.find_many.return_value = [_team_member("team-a")]

    assert await get_user_team_ids("u1", "org-1") == teams


async def _require(team_id: str | None = "team-1", org_id: str | None = "org-1"):
    await require_role_where_resource_lives(
        "u1", org_id, team_id, OrgAction.SHARE_RESOURCES, TeamAction.VIEW_EXECUTIONS
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "org_member, team_member",
    [
        (_org_member(), _team_member("team-1")),
        (_org_member(), _team_member("team-1", admin=True, billing=True)),
        (_org_member(owner=True, billing=True), _team_member("team-1")),
        (_org_member(), None),
        (None, None),
    ],
    ids=[
        "team-member",
        "team-admin-and-billing",
        "owner-and-billing",
        "no-longer-on-the-team",
        "no-longer-in-the-org",
    ],
)
async def test_a_role_that_holds_the_action_or_no_membership_is_not_refused(
    members, org_member, team_member
):
    """Without a membership there the route's ownership checks decide, as before."""
    members.orgmember.find_first.return_value = org_member
    members.teammember.find_first.return_value = team_member

    await _require()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "org_member, team_member, refused_in",
    [
        (_org_member(billing=True), None, "organization"),
        (_org_member(billing=True), _team_member("team-1"), "organization"),
        (_org_member(), _team_member("team-1", billing=True), "workspace"),
        (_org_member(owner=True), _team_member("team-1", billing=True), "workspace"),
    ],
    ids=[
        "org-billing-manager",
        "org-billing-manager-on-the-team",
        "team-billing-manager",
        "owner-who-only-manages-the-teams-billing",
    ],
)
async def test_a_billing_manager_role_where_the_resource_lives_is_refused(
    members, org_member, team_member, refused_in
):
    members.orgmember.find_first.return_value = org_member
    members.teammember.find_first.return_value = team_member

    with pytest.raises(NotAuthorizedError, match=refused_in):
        await _require()


@pytest.mark.asyncio
async def test_the_team_lookup_is_confined_to_the_resources_org(members):
    await _require()

    assert members.teammember.find_first.call_args.kwargs["where"] == {
        "userId": "u1",
        "teamId": "team-1",
        "status": "ACTIVE",
        "Team": {"is": {"orgId": "org-1"}},
    }


@pytest.mark.asyncio
async def test_an_untagged_resource_needs_no_lookup(members):
    await _require(team_id=None, org_id=None)

    members.orgmember.find_first.assert_not_awaited()
    members.teammember.find_first.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_org_home_resource_checks_only_the_org_role(members):
    members.orgmember.find_first.return_value = _org_member(billing=True)

    with pytest.raises(NotAuthorizedError, match="organization"):
        await _require(team_id=None)
    members.teammember.find_first.assert_not_awaited()


TEAM_REFUSAL = BILLING_ONLY_TEAM_ROLE.format(team_id="team-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "org_member, team_member, team_id, refusal",
    [
        (_org_member(billing=True), None, None, BILLING_ONLY_ORG_ROLE),
        (
            _org_member(billing=True),
            _team_member("team-1"),
            "team-1",
            BILLING_ONLY_ORG_ROLE,
        ),
        (_org_member(), _team_member("team-1", billing=True), "team-1", TEAM_REFUSAL),
        (
            _org_member(owner=True),
            _team_member("team-1", billing=True),
            "team-1",
            TEAM_REFUSAL,
        ),
    ],
    ids=[
        "org-billing-manager-in-org-home",
        "org-billing-manager-on-the-team",
        "team-billing-manager",
        "owner-who-only-manages-the-teams-billing",
    ],
)
async def test_a_role_that_covers_only_billing_is_refused_the_resources(
    members, org_member, team_member, team_id, refusal
):
    members.orgmember.find_first.return_value = org_member
    members.teammember.find_first.return_value = team_member

    assert await resource_role_refusal("u1", "org-1", team_id) == refusal


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "org_member, team_member",
    [
        (_org_member(), _team_member("team-1")),
        (_org_member(owner=True, billing=True), _team_member("team-1")),
        (_org_member(admin=True, billing=True), _team_member("team-1")),
        (_org_member(), _team_member("team-1", admin=True, billing=True)),
        (_org_member(), None),
        (None, None),
    ],
    ids=[
        "plain-member",
        "owner-and-billing",
        "admin-and-billing",
        "team-admin-and-billing",
        "not-on-the-team",
        "not-in-the-org",
    ],
)
async def test_a_role_that_covers_workflows_or_no_membership_is_not_refused(
    members, org_member, team_member
):
    """Membership itself is checked where the org and team are resolved."""
    members.orgmember.find_first.return_value = org_member
    members.teammember.find_first.return_value = team_member

    assert await resource_role_refusal("u1", "org-1", "team-1") is None


@pytest.mark.asyncio
async def test_the_role_comes_from_active_memberships_within_the_org(members):
    await resource_role_refusal("u1", "org-1", "team-1")

    assert members.orgmember.find_first.call_args.kwargs["where"] == {
        "userId": "u1",
        "orgId": "org-1",
        "status": "ACTIVE",
    }
    assert members.teammember.find_first.call_args.kwargs["where"] == {
        "userId": "u1",
        "teamId": "team-1",
        "status": "ACTIVE",
        "Team": {"is": {"orgId": "org-1"}},
    }


@pytest.mark.asyncio
async def test_org_home_reads_no_team_membership(members):
    assert await resource_role_refusal("u1", "org-1", None) is None

    members.teammember.find_first.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "personal, writable", [(True, {"org-1", None}), (False, {"org-1"})]
)
async def test_untagged_rows_are_writable_only_from_the_personal_org(
    mocker, personal: bool, writable: set
):
    """They predate org tagging, and the backfill gives them the personal org."""
    mock_prisma = MagicMock()
    mock_prisma.orgmember.find_first = AsyncMock(
        return_value=MagicMock() if personal else None
    )
    mocker.patch.object(tenancy, "prisma", mock_prisma)

    assert await writable_organization_ids("u1", "org-1") == writable
    where = mock_prisma.orgmember.find_first.call_args.kwargs["where"]
    assert (where["userId"], where["orgId"]) == ("u1", "org-1")
    assert where["Org"] == {"isPersonal": True, "deletedAt": None}


def test_organization_filter_matches_untagged_rows_only_when_asked():
    assert organization_filter({"org-1"}) == {
        "OR": [{"organizationId": {"in": ["org-1"]}}]
    }
    assert organization_filter({"org-1", None}) == {
        "OR": [{"organizationId": {"in": ["org-1"]}}, {"organizationId": None}]
    }
