"""How many orgs and workspaces one account can create is bounded.

Bounded at the create, because `GET /api/orgs` and the workspace listing have
no page to truncate: a `take` there would hide rows from their owner.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.orgs import db as orgs_db
from backend.api.features.orgs import team_db
from backend.api.features.orgs.db import MAX_ORGS_PER_USER
from backend.api.features.orgs.team_db import MAX_TEAMS_PER_ORG


@pytest.fixture
def orgs_client(mocker):
    client = MagicMock()
    client.organization.find_unique = AsyncMock(return_value=None)
    client.organizationalias.find_unique = AsyncMock(return_value=None)
    client.orgmember.count = AsyncMock(return_value=0)
    mocker.patch.object(orgs_db, "prisma", client)
    return client


@pytest.fixture
def teams_client(mocker):
    client = MagicMock()
    client.team.count = AsyncMock(return_value=0)
    client.team.create = AsyncMock()
    mocker.patch.object(team_db, "prisma", client)
    return client


@pytest.mark.asyncio
async def test_an_account_at_the_org_bound_cannot_create_another(orgs_client):
    orgs_client.orgmember.count.return_value = MAX_ORGS_PER_USER

    with pytest.raises(ValueError, match="at most 50 organizations"):
        await orgs_db.create_org("a name", "a-slug", "user-1")
    orgs_client.organization.find_unique.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_account_below_the_org_bound_gets_past_the_check(orgs_client):
    orgs_client.orgmember.count.return_value = MAX_ORGS_PER_USER - 1
    orgs_client.organization.find_unique.return_value = object()

    # The next check is the slug one, which is how this test tells "past the
    # count check" from "refused by it".
    with pytest.raises(ValueError, match="already in use"):
        await orgs_db.create_org("a name", "a-slug", "user-1")


@pytest.mark.asyncio
async def test_an_org_at_the_workspace_bound_cannot_create_another(teams_client):
    teams_client.team.count.return_value = MAX_TEAMS_PER_ORG

    with pytest.raises(ValueError, match="at most 500 workspaces"):
        await team_db.create_team("org-1", "a name", "user-1")
    teams_client.team.create.assert_not_awaited()
