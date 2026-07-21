"""Tests for the shared org/team visibility predicate."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data import tenancy
from backend.data.tenancy import get_user_team_ids, visibility_filter


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
