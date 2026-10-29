"""Tests for the shared org/team visibility predicate."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data import tenancy
from backend.data.tenancy import (
    get_user_team_ids,
    organization_filter,
    visibility_filter,
    writable_organization_ids,
)


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


@pytest.mark.asyncio
async def test_get_user_team_ids_filters_active_org_memberships(mocker):
    m1 = MagicMock()
    m1.teamId = "team-a"
    mock_prisma = MagicMock()
    mock_prisma.teammember.find_many = AsyncMock(return_value=[m1])
    mocker.patch.object(tenancy, "prisma", mock_prisma)

    result = await get_user_team_ids("u1", "org-1")

    assert result == ["team-a"]
    where = mock_prisma.teammember.find_many.call_args.kwargs["where"]
    assert where["userId"] == "u1"
    assert where["status"] == "ACTIVE"
    assert where["Team"] == {"is": {"orgId": "org-1"}}


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
