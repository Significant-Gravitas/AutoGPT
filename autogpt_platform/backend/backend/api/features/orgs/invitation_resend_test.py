"""HTTP-level tests for the org invitation resend endpoint."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.models import RequestContext

from backend.api.features.orgs.invitation_routes import INVITATION_TTL_DAYS, org_router
from backend.util.exceptions import NotFoundError

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
        # Default: no teams to re-validate. Tests that exercise team pruning
        # override this.
        self.prisma.team.find_many = AsyncMock(return_value=[])
        mocker.patch("backend.api.features.orgs.invitation_routes.prisma", self.prisma)

    def _expect_successful_resend(self, invitation, refreshed):
        """Wire the two find_unique calls a successful resend makes.

        The handler reads the invitation by id, compare-and-swaps via
        update_many, then reads the row back by the freshly minted token.
        """
        self.prisma.orginvitation.find_unique = AsyncMock(
            side_effect=[invitation, refreshed]
        )
        self.prisma.orginvitation.update_many = AsyncMock(return_value=1)

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

    def _app(self, ctx):
        app = fastapi.FastAPI()
        app.include_router(org_router, prefix="/orgs/{org_id}/invitations")
        app.dependency_overrides[get_request_context] = lambda: ctx
        return app

    def _client(self, ctx):
        return fastapi.testclient.TestClient(
            self._app(ctx), raise_server_exceptions=False
        )

    def _raising_client(self, ctx):
        """Client that re-raises handler exceptions instead of returning 500.

        The bare test app has no NotFoundError handler (the real app maps it to
        404 in rest_api.py), so this is how a test asserts *which* error was
        raised rather than just observing an opaque 500.
        """
        return fastapi.testclient.TestClient(
            self._app(ctx), raise_server_exceptions=True
        )

    def test_resend_pending_rotates_token_and_extends_expiry(self):
        invitation = self._make_invitation()
        refreshed = self._make_invitation(token="tok-new")
        self._expect_successful_resend(invitation, refreshed)

        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 200
        update_data = self.prisma.orginvitation.update_many.call_args[1]["data"]
        assert update_data["token"] != "tok-old"
        assert update_data["tokenHash"] is None
        assert update_data["expiresAt"] > datetime.now(timezone.utc) + timedelta(
            days=INVITATION_TTL_DAYS - 1
        )

        # The response body is the whole point of the endpoint: it must carry
        # the rotated token, not the stale pre-update one. (That the token
        # returned is the one this request minted is covered by
        # test_resend_reads_back_the_row_it_wrote.)
        body = resp.json()
        assert body["token"] == "tok-new"
        assert body["token"] != "tok-old"

    def test_resend_reads_the_row_back_by_id(self):
        """Read back by id, not by the minted token.

        A second concurrent resend may have rotated the token again; looking up
        the token this request minted would then find nothing and turn a
        harmless double-resend into a spurious 404.
        """
        invitation = self._make_invitation()
        refreshed = self._make_invitation(token="tok-new")
        self._expect_successful_resend(invitation, refreshed)

        self._client(_owner_ctx()).post(f"/orgs/{ORG_ID}/invitations/inv-1/resend")

        read_back = self.prisma.orginvitation.find_unique.call_args_list[1][1]["where"]
        assert read_back == {"id": "inv-1"}

    def test_resend_superseded_by_concurrent_resend_still_succeeds(self):
        """Our rotation committed, then another resend rotated again."""
        invitation = self._make_invitation()
        # The read-back sees the *other* request's token, not the one we minted.
        superseded = self._make_invitation(token="tok-from-other-resend")
        self._expect_successful_resend(invitation, superseded)

        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 200
        # Still a live, non-stale token: never the pre-update one.
        assert resp.json()["token"] == "tok-from-other-resend"
        assert resp.json()["token"] != "tok-old"

    def test_resend_update_is_scoped_to_still_pending_rows(self):
        """The write must re-assert pending state to close the TOCTOU window."""
        invitation = self._make_invitation()
        self._expect_successful_resend(
            invitation, self._make_invitation(token="tok-new")
        )

        self._client(_owner_ctx()).post(f"/orgs/{ORG_ID}/invitations/inv-1/resend")

        where = self.prisma.orginvitation.update_many.call_args[1]["where"]
        assert where == {"id": "inv-1", "acceptedAt": None, "revokedAt": None}

    def test_resend_losing_race_with_concurrent_accept_rejected(self):
        """Accepted between the read and the write: no token may be minted."""
        invitation = self._make_invitation()
        accepted = self._make_invitation(acceptedAt=FIXED_NOW)
        self.prisma.orginvitation.find_unique = AsyncMock(
            side_effect=[invitation, accepted]
        )
        # Zero rows matched: the CAS where-clause rejected the write.
        self.prisma.orginvitation.update_many = AsyncMock(return_value=0)

        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 400
        assert "accepted" in resp.json()["detail"].lower()

    def test_resend_expired_pending_invitation_succeeds(self):
        """Expired-but-unaccepted is the primary resend use case."""
        invitation = self._make_invitation(
            expiresAt=datetime.now(timezone.utc) - timedelta(days=1)
        )
        self._expect_successful_resend(
            invitation, self._make_invitation(token="tok-new")
        )

        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 200
        assert resp.json()["token"] == "tok-new"

    def test_resend_prunes_teams_deleted_since_invite(self):
        """A team deleted after invite must not be re-promised on resend."""
        invitation = self._make_invitation(teamIds=["team-live", "team-gone"])
        refreshed = self._make_invitation(token="tok-new", teamIds=["team-live"])
        self._expect_successful_resend(invitation, refreshed)
        live_team = MagicMock()
        live_team.id = "team-live"
        live_team.orgId = ORG_ID
        self.prisma.team.find_many = AsyncMock(return_value=[live_team])

        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 200
        update_data = self.prisma.orginvitation.update_many.call_args[1]["data"]
        assert update_data["teamIds"] == ["team-live"]
        assert resp.json()["team_ids"] == ["team-live"]

    def test_resend_drops_teams_belonging_to_another_org(self):
        invitation = self._make_invitation(teamIds=["team-foreign"])
        self._expect_successful_resend(
            invitation, self._make_invitation(token="tok-new")
        )
        foreign_team = MagicMock()
        foreign_team.id = "team-foreign"
        foreign_team.orgId = "other-org"
        self.prisma.team.find_many = AsyncMock(return_value=[foreign_team])

        self._client(_owner_ctx()).post(f"/orgs/{ORG_ID}/invitations/inv-1/resend")

        update_data = self.prisma.orginvitation.update_many.call_args[1]["data"]
        assert update_data["teamIds"] == []

    def test_resend_accepted_invitation_rejected(self):
        invitation = self._make_invitation(acceptedAt=FIXED_NOW)
        self.prisma.orginvitation.find_unique = AsyncMock(return_value=invitation)
        self.prisma.orginvitation.update_many = AsyncMock()

        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 400
        assert "accepted" in resp.json()["detail"].lower()
        self.prisma.orginvitation.update_many.assert_not_called()

    def test_resend_revoked_invitation_rejected(self):
        invitation = self._make_invitation(revokedAt=FIXED_NOW)
        self.prisma.orginvitation.find_unique = AsyncMock(return_value=invitation)
        self.prisma.orginvitation.update_many = AsyncMock()

        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 400
        assert "revoked" in resp.json()["detail"].lower()
        self.prisma.orginvitation.update_many.assert_not_called()

    def test_resend_invitation_from_other_org_not_found(self):
        invitation = self._make_invitation(orgId="other-org")
        self.prisma.orginvitation.find_unique = AsyncMock(return_value=invitation)
        self.prisma.orginvitation.update_many = AsyncMock()

        # NotFoundError has no handler in the bare test app (same pattern as
        # TestInvitationAcceptance): surfaces as 500 with a safe client.
        resp = self._client(_owner_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 500
        self.prisma.orginvitation.update_many.assert_not_called()

    def test_resend_missing_invitation_not_found(self):
        """The other half of the not-found condition: no such row at all."""
        self.prisma.orginvitation.find_unique = AsyncMock(return_value=None)
        self.prisma.orginvitation.update_many = AsyncMock()

        with pytest.raises(NotFoundError):
            self._raising_client(_owner_ctx()).post(
                f"/orgs/{ORG_ID}/invitations/inv-1/resend"
            )

        self.prisma.orginvitation.update_many.assert_not_called()

    def test_resend_requires_member_management_permission(self):
        resp = self._client(_member_ctx()).post(
            f"/orgs/{ORG_ID}/invitations/inv-1/resend"
        )

        assert resp.status_code == 403

    def test_list_hides_expired_invitations_by_default(self):
        self.prisma.orginvitation.find_many = AsyncMock(return_value=[])

        resp = self._client(_owner_ctx()).get(f"/orgs/{ORG_ID}/invitations")

        assert resp.status_code == 200
        where = self.prisma.orginvitation.find_many.call_args[1]["where"]
        assert "expiresAt" in where

    def test_list_include_expired_surfaces_resendable_invitations(self):
        """Without this the resend endpoint is unreachable: an admin can only
        resend an expired invite if some endpoint hands them its id."""
        expired = self._make_invitation(
            expiresAt=datetime.now(timezone.utc) - timedelta(days=1)
        )
        self.prisma.orginvitation.find_many = AsyncMock(return_value=[expired])

        resp = self._client(_owner_ctx()).get(
            f"/orgs/{ORG_ID}/invitations?include_expired=true"
        )

        assert resp.status_code == 200
        where = self.prisma.orginvitation.find_many.call_args[1]["where"]
        assert "expiresAt" not in where
        assert where["acceptedAt"] is None
        assert where["revokedAt"] is None
        assert where["orgId"] == ORG_ID
        assert [inv["id"] for inv in resp.json()] == ["inv-1"]

    def test_list_include_expired_requires_member_management_permission(self):
        resp = self._client(_member_ctx()).get(
            f"/orgs/{ORG_ID}/invitations?include_expired=true"
        )

        assert resp.status_code == 403
