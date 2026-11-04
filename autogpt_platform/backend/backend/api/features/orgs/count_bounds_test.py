"""How many orgs an account belongs to, and workspaces an org holds, is bounded.

Bounded at the write, because `GET /api/orgs` and the workspace listing have
no page to truncate: a `take` there would hide rows from their owner. The
concurrent cases are in `org_count_bounds_db_test.py`, on a real database.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.api.features.orgs import db as orgs_db
from backend.api.features.orgs import team_db
from backend.api.features.orgs.db import MAX_ORGS_PER_USER
from backend.api.features.orgs.team_db import MAX_TEAMS_PER_ORG
from backend.data import org_migration


class _Written(Exception):
    pass


@pytest.fixture
def tx(mocker):
    tx = MagicMock()
    tx.execute_raw = AsyncMock()
    tx.orgmember.count = AsyncMock(return_value=0)
    tx.orgmember.create = AsyncMock(side_effect=_Written)
    tx.organization.create = AsyncMock(side_effect=_Written)
    tx.team.count = AsyncMock(return_value=0)
    tx.team.create = AsyncMock(side_effect=_Written)

    @asynccontextmanager
    async def transaction():
        yield tx

    mocker.patch.object(orgs_db, "transaction", transaction)
    mocker.patch.object(team_db, "transaction", transaction)
    client = MagicMock()
    client.organization.find_unique = AsyncMock(return_value=None)
    client.organization.find_first = AsyncMock(return_value=None)
    client.organizationalias.find_unique = AsyncMock(return_value=None)
    mocker.patch.object(orgs_db, "prisma", client)
    return tx


@pytest.mark.asyncio
async def test_an_account_at_the_org_bound_cannot_create_another(tx):
    tx.orgmember.count.return_value = MAX_ORGS_PER_USER

    with pytest.raises(ValueError, match="at most 50 organizations"):
        await orgs_db.create_org("a name", "a-slug", "user-1")
    tx.organization.create.assert_not_awaited()
    assert tx.orgmember.count.await_args.kwargs["where"] == {
        "userId": "user-1",
        "status": "ACTIVE",
        "Org": {"deletedAt": None},
    }


@pytest.mark.asyncio
async def test_an_account_below_the_org_bound_creates_one(tx):
    tx.orgmember.count.return_value = MAX_ORGS_PER_USER - 1

    with pytest.raises(_Written):
        await orgs_db.create_org("a name", "a-slug", "user-1")


@pytest.mark.asyncio
async def test_an_account_at_the_org_bound_cannot_join_another(tx):
    """An accepted invitation adds the membership through `add_org_member`."""
    tx.orgmember.count.return_value = MAX_ORGS_PER_USER

    with pytest.raises(ValueError, match="at most 50 organizations"):
        await orgs_db.add_org_member("org-1", "user-1")
    tx.orgmember.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_account_below_the_org_bound_joins_one(tx):
    tx.orgmember.count.return_value = MAX_ORGS_PER_USER - 1

    with pytest.raises(_Written):
        await orgs_db.add_org_member("org-1", "user-1")


@pytest.mark.asyncio
async def test_an_org_at_the_workspace_bound_cannot_create_another(tx):
    tx.team.count.return_value = MAX_TEAMS_PER_ORG

    with pytest.raises(ValueError, match="at most 500 workspaces"):
        await team_db.create_team("org-1", "a name", "user-1")
    tx.team.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_org_below_the_workspace_bound_creates_one(tx):
    tx.team.count.return_value = MAX_TEAMS_PER_ORG - 1

    with pytest.raises(_Written):
        await team_db.create_team("org-1", "a name", "user-1")
    tx.team.count.assert_awaited_once_with(where={"orgId": "org-1", "archivedAt": None})


@pytest.fixture
def converting(tx, mocker):
    """`convert_personal_org` reaching the real `create_personal_org`."""
    tx.organization.update_many = AsyncMock(side_effect=_Written)

    @asynccontextmanager
    async def transaction():
        yield tx

    mocker.patch.object(org_migration, "transaction", transaction)
    mocker.patch.object(
        org_migration, "_resolve_unique_slug", AsyncMock(return_value="a-slug")
    )
    orgs_db.prisma.organization.find_unique = AsyncMock(
        return_value=MagicMock(isPersonal=True, slug="a-slug")
    )
    orgs_db.prisma.user.find_unique = AsyncMock(return_value=None)
    return tx


@pytest.mark.asyncio
async def test_an_account_at_the_org_bound_cannot_convert_its_personal_org(converting):
    """Converting adds a new personal org, so it is a membership like any other."""
    converting.orgmember.count.return_value = MAX_ORGS_PER_USER

    with pytest.raises(ValueError, match="at most 50 organizations"):
        await orgs_db.convert_personal_org("org-1", "user-1")
    converting.organization.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_account_below_the_org_bound_converts_its_personal_org(converting):
    converting.orgmember.count.return_value = MAX_ORGS_PER_USER - 1

    with pytest.raises(_Written):
        await orgs_db.convert_personal_org("org-1", "user-1")
