"""HTTP-level tests for the org invitation resend endpoint.

Kept separate from routes_test.py so this dev-based feature branch shares no
test-file surface with the in-flight org-UI stack (rollup conflict hygiene).
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.models import RequestContext

from backend.api.features.orgs.invitation_routes import INVITATION_TTL_DAYS

USER_ID = "user-owner-1"
OTHER_USER_ID = "user-member-2"
ORG_ID = "org-aaa"
FIXED_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _owner_ctx() -> RequestContext:
    return RequestContext(
        user_id=USER_ID,
        org_id=ORG_ID,
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


def _member_ctx() -> RequestContext:
    return RequestContext(
        user_id=OTHER_USER_ID,
        org_id=ORG_ID,
        team_id=None,
        is_org_owner=False,
        is_org_admin=False,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


class TestInvitationResend:
    """Tests for the org-scoped invitation resend endpoint."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.api.features.orgs.invitation_routes.prisma", self.prisma)

    def _make_invitation(self, **overrides):
        inv = MagicMock()
        inv.id = "inv-1"
        inv.token = "tok-old"
        inv.email = "alice@example.com"
        inv.orgId = ORG_ID
        inv.isAdmin = False
        inv.isBillingManager = False
        inv.acceptedAt = None
        inv.revokedAt = None
        inv.expiresAt = datetime.now(timezone.utc) + timedelta(days=3)
        inv.createdAt = FIXED_NOW
        inv.invitedByUserId = USER_ID
        inv.teamIds = []
        for key, value in overrides.items():
            setattr(inv, key, value)
        return inv

    def _client(self, ctx):
        from autogpt_libs.auth.dependencies import get_request_context

        from backend.api.features.orgs.invitation_routes import org_router

        app = fastapi.FastAPI()
        app.include_router(org_router, prefix="/orgs/{org_id}/invitations")
        app.dependency_overrides[get_request_context] = lambda: ctx
        return fastapi.testclient.TestClient(app, raise_server_exceptions=False)

    def test_resend_pending_rotates_token_and_extends_expiry(self):
        invitation = self._make_invitation()
        refreshed = self._make_invitation(token="tok-new")
        self.prisma.orginvitation.find_unique = AsyncMock(return_value=invitation)
        self.prisma.orginvitation.update = AsyncMock(return_value=refreshed)

        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 200
        update_data = self.prisma.orginvitation.update.call_args[1]["data"]
        assert update_data["token"] != "tok-old"
        assert update_data["tokenHash"] is None
        assert update_data["expiresAt"] > datetime.now(timezone.utc) + timedelta(
            days=INVITATION_TTL_DAYS - 1
        )

    def test_resend_expired_pending_invitation_succeeds(self):
        """Expired-but-unaccepted is the primary resend use case."""
        invitation = self._make_invitation(
            expiresAt=datetime.now(timezone.utc) - timedelta(days=1)
        )
        self.prisma.orginvitation.find_unique = AsyncMock(return_value=invitation)
        self.prisma.orginvitation.update = AsyncMock(
            return_value=self._make_invitation(token="tok-new")
        )

        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 200

    def test_resend_accepted_invitation_rejected(self):
        invitation = self._make_invitation(acceptedAt=FIXED_NOW)
        self.prisma.orginvitation.find_unique = AsyncMock(return_value=invitation)
        self.prisma.orginvitation.update = AsyncMock()

        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 400
        assert "accepted" in resp.json()["detail"].lower()
        self.prisma.orginvitation.update.assert_not_called()

    def test_resend_revoked_invitation_rejected(self):
        invitation = self._make_invitation(revokedAt=FIXED_NOW)
        self.prisma.orginvitation.find_unique = AsyncMock(return_value=invitation)
        self.prisma.orginvitation.update = AsyncMock()

        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 400
        assert "revoked" in resp.json()["detail"].lower()
        self.prisma.orginvitation.update.assert_not_called()

    def test_resend_invitation_from_other_org_not_found(self):
        invitation = self._make_invitation(orgId="other-org")
        self.prisma.orginvitation.find_unique = AsyncMock(return_value=invitation)
        self.prisma.orginvitation.update = AsyncMock()

        # NotFoundError has no handler in the bare test app (same pattern as
        # TestInvitationAcceptance): surfaces as 500 with a safe client.
        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 500
        self.prisma.orginvitation.update.assert_not_called()

    def test_resend_requires_member_management_permission(self):
        resp = self._client(_member_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 403
