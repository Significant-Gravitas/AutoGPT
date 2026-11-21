"""The billing-manager role is given or taken only by a holder of MANAGE_BILLING."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth import get_request_context
from autogpt_libs.auth.models import RequestContext

from . import db as orgs_db
from . import invitation_routes, routes
from .model import OrgMemberResponse

ORG_ID = "org-1"
CALLER_ID = "caller-1"
MEMBER_ID = "member-1"
NOW = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _ctx(*, owner=False, admin=False, billing=False) -> RequestContext:
    return RequestContext(
        user_id=CALLER_ID,
        org_id=ORG_ID,
        team_id=None,
        is_org_owner=owner,
        is_org_admin=admin,
        is_org_billing_manager=billing,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


ADMIN = _ctx(admin=True)
OWNER = _ctx(owner=True, admin=True)
BILLING_ADMIN = _ctx(admin=True, billing=True)


@pytest.fixture
def app(mocker):
    app = fastapi.FastAPI()
    app.include_router(routes.router, prefix="/orgs")
    app.include_router(
        invitation_routes.org_router, prefix="/orgs/{org_id}/invitations"
    )
    mocker.patch.object(
        orgs_db,
        "update_org_member",
        AsyncMock(
            return_value=OrgMemberResponse(
                id="m-1",
                user_id=MEMBER_ID,
                email="member@example.com",
                name=None,
                is_owner=False,
                is_admin=False,
                is_billing_manager=True,
                joined_at=NOW,
            )
        ),
    )
    client = MagicMock()
    client.orginvitation.create = AsyncMock(side_effect=_stored_invitation)
    mocker.patch.object(invitation_routes, "prisma", client)
    return app


def _client(app: fastapi.FastAPI, ctx: RequestContext) -> fastapi.testclient.TestClient:
    app.dependency_overrides[get_request_context] = lambda: ctx
    return fastapi.testclient.TestClient(app)


@pytest.mark.parametrize(
    "uid, body",
    [
        (CALLER_ID, {"is_billing_manager": True}),
        (MEMBER_ID, {"is_billing_manager": True}),
        (MEMBER_ID, {"is_billing_manager": False}),
    ],
    ids=["own-row", "grant", "revoke"],
)
def test_an_admin_without_billing_cannot_change_the_role(app, uid, body):
    response = _client(app, ADMIN).patch(f"/orgs/{ORG_ID}/members/{uid}", json=body)

    assert response.status_code == 403
    assert response.json()["detail"] == "Missing org permission: MANAGE_BILLING"
    orgs_db.update_org_member.assert_not_awaited()


@pytest.mark.parametrize(
    "ctx, body",
    [
        (ADMIN, {"is_admin": True}),
        (OWNER, {"is_billing_manager": True}),
        (BILLING_ADMIN, {"is_billing_manager": True}),
    ],
    ids=["admin-sets-admin", "owner", "admin-holding-billing"],
)
def test_a_role_change_the_caller_may_make_goes_through(app, ctx, body):
    response = _client(app, ctx).patch(f"/orgs/{ORG_ID}/members/{MEMBER_ID}", json=body)

    assert response.status_code == 200
    orgs_db.update_org_member.assert_awaited_once()


@pytest.mark.parametrize(
    "ctx, billing, status",
    [
        (ADMIN, True, 403),
        (ADMIN, False, 200),
        (OWNER, True, 200),
        (BILLING_ADMIN, True, 200),
    ],
    ids=["admin-billing", "admin-plain", "owner-billing", "billing-admin-billing"],
)
def test_an_invitation_carries_the_role_only_from_a_billing_holder(
    app, ctx, billing, status
):
    response = _client(app, ctx).post(
        f"/orgs/{ORG_ID}/invitations",
        json={"email": "new@example.com", "is_billing_manager": billing},
    )

    assert response.status_code == status
    assert invitation_routes.prisma.orginvitation.create.await_count == (
        1 if status == 200 else 0
    )


@pytest.mark.parametrize(
    "inviter, granted",
    [
        (MagicMock(isOwner=True, isBillingManager=False), True),
        (MagicMock(isOwner=False, isBillingManager=True), True),
        (MagicMock(isOwner=False, isBillingManager=False), False),
        (None, False),
    ],
    ids=["owner", "billing-manager", "plain-admin", "inviter-gone"],
)
@pytest.mark.asyncio
async def test_accepting_grants_the_role_only_while_the_inviter_holds_billing(
    mocker, inviter, granted
):
    """An invitation made before its inviter lost MANAGE_BILLING, or made by
    someone who never held it, joins the member without the role."""
    invitation = _stored_invitation(
        data={
            "orgId": ORG_ID,
            "email": "new@example.com",
            "isAdmin": False,
            "isBillingManager": True,
            "expiresAt": NOW + timedelta(days=3650),
            "invitedByUserId": CALLER_ID,
            "teamIds": [],
        }
    )
    invitation.acceptedAt = invitation.revokedAt = None
    invitation.Org = MagicMock(deletedAt=None)
    client = MagicMock()
    client.orginvitation.find_unique = AsyncMock(return_value=invitation)
    client.orginvitation.update = AsyncMock()
    client.user.find_unique = AsyncMock(return_value=MagicMock(email="new@example.com"))
    client.orgmember.find_first = AsyncMock(return_value=inviter)
    mocker.patch.object(invitation_routes, "prisma", client)
    add = mocker.patch.object(orgs_db, "add_org_member", AsyncMock())

    await invitation_routes.accept_invitation("tok-1", user_id=MEMBER_ID)

    assert add.await_args.kwargs["is_billing_manager"] is granted
    if inviter is not None:
        assert client.orgmember.find_first.await_args.kwargs["where"] == {
            "orgId": ORG_ID,
            "userId": CALLER_ID,
            "status": "ACTIVE",
        }


def _stored_invitation(data: dict) -> MagicMock:
    inv = MagicMock(
        id="inv-1",
        token="tok-1",
        email=data["email"],
        isAdmin=data["isAdmin"],
        isBillingManager=data["isBillingManager"],
        expiresAt=data["expiresAt"],
        createdAt=NOW,
        teamIds=data["teamIds"],
        orgId=data["orgId"],
        invitedByUserId=data["invitedByUserId"],
    )
    return inv
