"""An admin who isn't the owner invites and keeps plain members only: the
owner alone hands out admin and billing manager."""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from autogpt_libs.auth.models import RequestContext

from backend.api.features.orgs import db as org_db
from backend.api.features.orgs import invitation_routes
from backend.api.features.orgs.model import CreateInvitationRequest
from backend.util.exceptions import NotAuthorizedError

ADMIN = RequestContext(
    user_id="admin",
    org_id="org",
    team_id=None,
    is_org_owner=False,
    is_org_admin=True,
    is_org_billing_manager=False,
    is_team_admin=True,
    is_team_billing_manager=False,
    seat_status="ACTIVE",
)


def _client() -> MagicMock:
    client = MagicMock()
    client.execute_raw = AsyncMock(return_value=1)
    client.user.find_first = AsyncMock(return_value=None)
    client.user.find_unique = AsyncMock(return_value=None)
    client.organization.find_unique = AsyncMock(
        return_value=MagicMock(isPersonal=False)
    )
    client.orgmember.find_unique = AsyncMock(return_value=None)
    client.orginvitation.find_first = AsyncMock(return_value=None)
    client.orginvitation.create = AsyncMock(
        return_value=MagicMock(
            id="inv",
            email="new@example.com",
            token="tok",
            isAdmin=False,
            isBillingManager=False,
            expiresAt=datetime.now(timezone.utc) + timedelta(days=7),
            createdAt=datetime.now(timezone.utc),
            teamIds=[],
        )
    )
    return client


def _fake_transaction(client: MagicMock):
    @asynccontextmanager
    async def transaction():
        yield client

    return transaction


@pytest.fixture
def as_admin(mocker) -> MagicMock:
    """The actor is an admin, not the owner, under every live lock."""
    client = _client()
    for module in (invitation_routes, org_db):
        mocker.patch.object(module, "prisma", client)
        mocker.patch.object(module, "transaction", _fake_transaction(client))
        mocker.patch.object(
            module, "lock_live_org_permission_scope", AsyncMock(return_value=ADMIN)
        )
    mocker.patch.object(org_db, "execute_raw_with_schema", AsyncMock(return_value=1))
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role", [{"is_admin": True}, {"is_billing_manager": True}], ids=["admin", "billing"]
)
async def test_an_admin_cannot_invite_someone_into_a_role(as_admin, role):
    with pytest.raises(NotAuthorizedError, match="owner"):
        await invitation_routes._create_invitation_locked(
            "org", CreateInvitationRequest(email="new@example.com", **role), "admin"
        )
    as_admin.orginvitation.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_admin_can_invite_a_plain_member(as_admin):
    created = await invitation_routes._create_invitation_locked(
        "org", CreateInvitationRequest(email="new@example.com"), "admin"
    )
    assert created.token == "tok"
    as_admin.orginvitation.create.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [{"is_admin": True}, {"is_billing_manager": True}],
    ids=["promote", "billing"],
)
async def test_an_admin_cannot_change_a_members_role(as_admin, change):
    as_admin.orgmember.find_unique = AsyncMock(
        return_value=MagicMock(isOwner=False, isAdmin=False, isBillingManager=False)
    )
    as_admin.orgmember.update = AsyncMock()
    with pytest.raises(NotAuthorizedError, match="owner"):
        await org_db.update_org_member(
            org_id="org",
            user_id="member",
            is_admin=change.get("is_admin"),
            is_billing_manager=change.get("is_billing_manager"),
            requesting_user_id="admin",
        )
    as_admin.orgmember.update.assert_not_awaited()
