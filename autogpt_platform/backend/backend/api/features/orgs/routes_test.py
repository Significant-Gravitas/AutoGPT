"""Comprehensive tests for org/workspace backend.

Covers org CRUD, workspace CRUD, invitations, credits, and migration edge
cases. Tests are organized by domain and mock at the Prisma boundary so the
actual logic in db.py / routes.py / workspace_db.py is exercised.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest

from autogpt_libs.auth.models import RequestContext
from backend.util.exceptions import InsufficientBalanceError, NotFoundError

# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

USER_ID = "user-owner-1"
OTHER_USER_ID = "user-member-2"
ORG_ID = "org-aaa"
WS_ID = "ws-default"
FIXED_NOW = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_org(
    *,
    id=ORG_ID,
    name="Acme",
    slug="acme",
    description=None,
    avatarUrl=None,
    isPersonal=False,
    createdAt=FIXED_NOW,
):
    m = MagicMock()
    m.id = id
    m.name = name
    m.slug = slug
    m.description = description
    m.avatarUrl = avatarUrl
    m.isPersonal = isPersonal
    m.createdAt = createdAt
    return m


def _make_member(
    *,
    id="mem-1",
    orgId=ORG_ID,
    userId=USER_ID,
    isOwner=False,
    isAdmin=False,
    isBillingManager=False,
    status="ACTIVE",
    joinedAt=FIXED_NOW,
    user_email="test@example.com",
    user_name="Test",
):
    m = MagicMock()
    m.id = id
    m.orgId = orgId
    m.userId = userId
    m.isOwner = isOwner
    m.isAdmin = isAdmin
    m.isBillingManager = isBillingManager
    m.status = status
    m.joinedAt = joinedAt
    user_mock = MagicMock(email=user_email)
    user_mock.name = user_name
    m.User = user_mock
    m.Org = _make_org(id=orgId)
    return m


def _make_workspace(
    *,
    id=WS_ID,
    name="Default",
    slug=None,
    description=None,
    isDefault=True,
    joinPolicy="OPEN",
    orgId=ORG_ID,
    createdAt=FIXED_NOW,
    archivedAt=None,
):
    m = MagicMock()
    m.id = id
    m.name = name
    m.slug = slug
    m.description = description
    m.isDefault = isDefault
    m.joinPolicy = joinPolicy
    m.orgId = orgId
    m.createdAt = createdAt
    m.archivedAt = archivedAt
    return m


def _make_ws_member(
    *,
    id="wm-1",
    workspaceId=WS_ID,
    userId=USER_ID,
    isAdmin=False,
    isBillingManager=False,
    status="ACTIVE",
    joinedAt=FIXED_NOW,
    user_email="test@example.com",
    user_name="Test",
):
    m = MagicMock()
    m.id = id
    m.workspaceId = workspaceId
    m.userId = userId
    m.isAdmin = isAdmin
    m.isBillingManager = isBillingManager
    m.status = status
    m.joinedAt = joinedAt
    user_mock = MagicMock(email=user_email)
    user_mock.name = user_name
    m.User = user_mock
    m.Workspace = _make_workspace(id=workspaceId)
    return m


def _owner_ctx(
    org_id=ORG_ID, user_id=USER_ID, workspace_id=None
) -> RequestContext:
    return RequestContext(
        user_id=user_id,
        org_id=org_id,
        workspace_id=workspace_id,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_workspace_admin=True,
        is_workspace_billing_manager=False,
        seat_status="ACTIVE",
    )


def _member_ctx(
    org_id=ORG_ID, user_id=OTHER_USER_ID, workspace_id=None
) -> RequestContext:
    return RequestContext(
        user_id=user_id,
        org_id=org_id,
        workspace_id=workspace_id,
        is_org_owner=False,
        is_org_admin=False,
        is_org_billing_manager=False,
        is_workspace_admin=False,
        is_workspace_billing_manager=False,
        seat_status="ACTIVE",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ORG CRUD  (db.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrgDbCreateOrg:
    """Tests for backend.api.features.orgs.db.create_org."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        self.prisma.organization.find_unique = AsyncMock(return_value=None)
        self.prisma.organizationalias.find_unique = AsyncMock(return_value=None)
        self.prisma.organization.create = AsyncMock(return_value=_make_org())
        self.prisma.orgmember.create = AsyncMock()
        self.prisma.orgworkspace.create = AsyncMock(
            return_value=_make_workspace()
        )
        self.prisma.orgworkspacemember.create = AsyncMock()
        self.prisma.organizationprofile.create = AsyncMock()
        self.prisma.organizationseatassignment.create = AsyncMock()
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_create_org_success(self):
        from backend.api.features.orgs.db import create_org

        result = await create_org(
            name="Acme", slug="acme", user_id=USER_ID, description="A corp"
        )

        # Verify returned dict shape
        assert result["id"] == ORG_ID
        assert result["name"] == "Acme"
        assert result["slug"] == "acme"
        assert result["isPersonal"] is False
        assert result["memberCount"] == 1
        assert result["createdAt"] == FIXED_NOW

        # Org created with isPersonal=False
        org_create_data = self.prisma.organization.create.call_args[1]["data"]
        assert org_create_data["isPersonal"] is False
        assert org_create_data["slug"] == "acme"

        # Owner membership was created
        member_data = self.prisma.orgmember.create.call_args[1]["data"]
        assert member_data["isOwner"] is True
        assert member_data["isAdmin"] is True
        assert member_data["userId"] == USER_ID

        # Default workspace was created
        ws_data = self.prisma.orgworkspace.create.call_args[1]["data"]
        assert ws_data["name"] == "Default"
        assert ws_data["isDefault"] is True
        assert ws_data["joinPolicy"] == "OPEN"

        # User added to default workspace
        wsm_data = self.prisma.orgworkspacemember.create.call_args[1]["data"]
        assert wsm_data["isAdmin"] is True
        assert wsm_data["userId"] == USER_ID

        # Profile was created
        self.prisma.organizationprofile.create.assert_called_once()
        profile_data = self.prisma.organizationprofile.create.call_args[1][
            "data"
        ]
        assert profile_data["username"] == "acme"
        assert profile_data["displayName"] == "Acme"

        # Seat assigned
        self.prisma.organizationseatassignment.create.assert_called_once()
        seat_data = self.prisma.organizationseatassignment.create.call_args[1][
            "data"
        ]
        assert seat_data["seatType"] == "FREE"
        assert seat_data["userId"] == USER_ID

    @pytest.mark.asyncio
    async def test_create_org_duplicate_slug_raises_value_error(self):
        from backend.api.features.orgs.db import create_org

        self.prisma.organization.find_unique = AsyncMock(
            return_value=_make_org(slug="taken")
        )

        with pytest.raises(ValueError, match="already in use"):
            await create_org(name="Dup", slug="taken", user_id=USER_ID)

        # Org should NOT have been created
        self.prisma.organization.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_org_slug_taken_by_alias_raises_value_error(self):
        from backend.api.features.orgs.db import create_org

        self.prisma.organizationalias.find_unique = AsyncMock(
            return_value=MagicMock(aliasSlug="aliased")
        )

        with pytest.raises(ValueError, match="already in use as an alias"):
            await create_org(name="Dup", slug="aliased", user_id=USER_ID)


class TestOrgDbUpdateOrg:
    """Tests for backend.api.features.orgs.db.update_org."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        self.old_org = _make_org(slug="old-slug")
        self.updated_org = _make_org(slug="new-slug")

        self.prisma.organization.find_unique = AsyncMock(
            return_value=self.old_org
        )
        self.prisma.organizationalias.find_unique = AsyncMock(
            return_value=None
        )
        self.prisma.organizationalias.create = AsyncMock()
        self.prisma.organization.update = AsyncMock()
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_update_org_creates_rename_alias_on_slug_change(self):
        from backend.api.features.orgs.db import update_org

        # After update, get_org is called which does find_unique again
        self.prisma.organization.find_unique = AsyncMock(
            side_effect=[
                None,  # slug uniqueness check (new slug not taken)
                self.old_org,  # fetch old org for alias creation
                self.updated_org,  # get_org after update
            ]
        )

        await update_org(ORG_ID, {"slug": "new-slug"})

        # Alias should have been created for the old slug
        self.prisma.organizationalias.create.assert_called_once()
        alias_data = self.prisma.organizationalias.create.call_args[1]["data"]
        assert alias_data["aliasSlug"] == "old-slug"
        assert alias_data["aliasType"] == "RENAME"
        assert alias_data["organizationId"] == ORG_ID

    @pytest.mark.asyncio
    async def test_update_org_slug_collision_raises_value_error(self):
        from backend.api.features.orgs.db import update_org

        other_org = _make_org(id="other-org", slug="new-slug")
        self.prisma.organization.find_unique = AsyncMock(
            return_value=other_org
        )

        with pytest.raises(ValueError, match="already in use"):
            await update_org(ORG_ID, {"slug": "new-slug"})

    @pytest.mark.asyncio
    async def test_update_org_no_op_when_all_none(self):
        from backend.api.features.orgs.db import update_org

        # When all values are None, update_org returns get_org result
        self.prisma.organization.find_unique = AsyncMock(
            return_value=self.old_org
        )

        result = await update_org(ORG_ID, {"name": None, "slug": None})

        # No update call should have happened
        self.prisma.organization.update.assert_not_called()
        assert result["id"] == ORG_ID


class TestOrgDbDeleteOrg:
    """Tests for backend.api.features.orgs.db.delete_org."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_delete_personal_org_raises_value_error(self):
        from backend.api.features.orgs.db import delete_org

        self.prisma.organization.find_unique = AsyncMock(
            return_value=_make_org(isPersonal=True)
        )

        with pytest.raises(ValueError, match="Cannot delete a personal"):
            await delete_org(ORG_ID)

        self.prisma.organization.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_non_personal_org_success(self):
        from backend.api.features.orgs.db import delete_org

        self.prisma.organization.find_unique = AsyncMock(
            return_value=_make_org(isPersonal=False)
        )
        self.prisma.organization.delete = AsyncMock()

        await delete_org(ORG_ID)

        self.prisma.organization.delete.assert_called_once_with(
            where={"id": ORG_ID}
        )

    @pytest.mark.asyncio
    async def test_delete_org_not_found_raises(self):
        from backend.api.features.orgs.db import delete_org

        self.prisma.organization.find_unique = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await delete_org("nonexistent")


class TestOrgDbConvertOrg:
    """Tests for backend.api.features.orgs.db.convert_personal_org."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        self.prisma.organization.update = AsyncMock()
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_convert_personal_to_team_org(self):
        from backend.api.features.orgs.db import convert_personal_org

        personal_org = _make_org(isPersonal=True)
        converted_org = _make_org(isPersonal=False)

        self.prisma.organization.find_unique = AsyncMock(
            side_effect=[personal_org, converted_org]
        )

        await convert_personal_org(ORG_ID)

        self.prisma.organization.update.assert_called_once_with(
            where={"id": ORG_ID},
            data={"isPersonal": False},
        )

    @pytest.mark.asyncio
    async def test_convert_already_team_org_raises(self):
        from backend.api.features.orgs.db import convert_personal_org

        self.prisma.organization.find_unique = AsyncMock(
            return_value=_make_org(isPersonal=False)
        )

        with pytest.raises(ValueError, match="already a team org"):
            await convert_personal_org(ORG_ID)


class TestOrgDbMembers:
    """Tests for member management in db.py."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_add_member_auto_joins_default_workspace(self):
        from backend.api.features.orgs.db import add_org_member

        new_member = _make_member(
            userId=OTHER_USER_ID, user_email="bob@example.com"
        )
        self.prisma.orgmember.create = AsyncMock(return_value=new_member)
        default_ws = _make_workspace(id="ws-default", isDefault=True)
        self.prisma.orgworkspace.find_first = AsyncMock(return_value=default_ws)
        self.prisma.orgworkspacemember.create = AsyncMock()

        result = await add_org_member(
            org_id=ORG_ID,
            user_id=OTHER_USER_ID,
            is_admin=False,
            invited_by=USER_ID,
        )

        # Org member created
        assert result["userId"] == OTHER_USER_ID
        assert result["email"] == "bob@example.com"

        # Workspace member auto-created for default workspace
        self.prisma.orgworkspacemember.create.assert_called_once()
        wsm_data = self.prisma.orgworkspacemember.create.call_args[1]["data"]
        assert wsm_data["workspaceId"] == "ws-default"
        assert wsm_data["userId"] == OTHER_USER_ID

    @pytest.mark.asyncio
    async def test_add_member_no_default_workspace_skips_ws_join(self):
        from backend.api.features.orgs.db import add_org_member

        new_member = _make_member(userId=OTHER_USER_ID)
        self.prisma.orgmember.create = AsyncMock(return_value=new_member)
        self.prisma.orgworkspace.find_first = AsyncMock(return_value=None)
        self.prisma.orgworkspacemember.create = AsyncMock()

        await add_org_member(org_id=ORG_ID, user_id=OTHER_USER_ID)

        # No workspace member should have been created
        self.prisma.orgworkspacemember.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_member_cascades_workspace_memberships(self):
        from backend.api.features.orgs.db import remove_org_member

        member = _make_member(userId=OTHER_USER_ID, isOwner=False)
        self.prisma.orgmember.find_unique = AsyncMock(return_value=member)

        ws1 = _make_workspace(id="ws-1")
        ws2 = _make_workspace(id="ws-2")
        self.prisma.orgworkspace.find_many = AsyncMock(return_value=[ws1, ws2])
        self.prisma.orgworkspacemember.delete_many = AsyncMock()
        self.prisma.orgmember.delete = AsyncMock()

        await remove_org_member(ORG_ID, OTHER_USER_ID)

        # Should delete workspace memberships for each workspace
        assert self.prisma.orgworkspacemember.delete_many.call_count == 2
        calls = self.prisma.orgworkspacemember.delete_many.call_args_list
        ws_ids = [c[1]["where"]["workspaceId"] for c in calls]
        assert set(ws_ids) == {"ws-1", "ws-2"}

        # Org membership deleted
        self.prisma.orgmember.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_owner_raises_value_error(self):
        from backend.api.features.orgs.db import remove_org_member

        owner = _make_member(userId=USER_ID, isOwner=True)
        self.prisma.orgmember.find_unique = AsyncMock(return_value=owner)

        with pytest.raises(ValueError, match="Cannot remove the org owner"):
            await remove_org_member(ORG_ID, USER_ID)

    @pytest.mark.asyncio
    async def test_remove_nonexistent_member_raises_not_found(self):
        from backend.api.features.orgs.db import remove_org_member

        self.prisma.orgmember.find_unique = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await remove_org_member(ORG_ID, "ghost-user")

    @pytest.mark.asyncio
    async def test_update_owner_role_raises_value_error(self):
        from backend.api.features.orgs.db import update_org_member

        owner = _make_member(userId=USER_ID, isOwner=True)
        self.prisma.orgmember.find_unique = AsyncMock(return_value=owner)

        with pytest.raises(
            ValueError, match="Cannot change the owner's role"
        ):
            await update_org_member(ORG_ID, USER_ID, is_admin=False, is_billing_manager=None)

    @pytest.mark.asyncio
    async def test_update_member_not_found_raises(self):
        from backend.api.features.orgs.db import update_org_member

        self.prisma.orgmember.find_unique = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await update_org_member(ORG_ID, "ghost", is_admin=True, is_billing_manager=None)


class TestOrgDbTransferOwnership:
    """Tests for transfer_ownership in db.py."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        self.prisma.execute_raw = AsyncMock(return_value=2)
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_transfer_ownership_atomic(self):
        from backend.api.features.orgs.db import transfer_ownership

        current_owner = _make_member(userId=USER_ID, isOwner=True)
        new_owner = _make_member(userId=OTHER_USER_ID, isOwner=False)

        self.prisma.orgmember.find_unique = AsyncMock(
            side_effect=[current_owner, new_owner]
        )

        await transfer_ownership(ORG_ID, USER_ID, OTHER_USER_ID)

        # Raw SQL was executed for atomic transfer
        self.prisma.execute_raw.assert_called_once()
        raw_call_args = self.prisma.execute_raw.call_args
        sql = raw_call_args[0][0]
        assert "isOwner" in sql
        # Positional params: current_owner_id, new_owner_id, org_id
        assert raw_call_args[0][1] == USER_ID
        assert raw_call_args[0][2] == OTHER_USER_ID
        assert raw_call_args[0][3] == ORG_ID

    @pytest.mark.asyncio
    async def test_transfer_to_non_member_raises(self):
        from backend.api.features.orgs.db import transfer_ownership

        current_owner = _make_member(userId=USER_ID, isOwner=True)
        self.prisma.orgmember.find_unique = AsyncMock(
            side_effect=[current_owner, None]
        )

        with pytest.raises(NotFoundError, match="not a member"):
            await transfer_ownership(ORG_ID, USER_ID, "stranger")

    @pytest.mark.asyncio
    async def test_transfer_from_non_owner_raises(self):
        from backend.api.features.orgs.db import transfer_ownership

        not_owner = _make_member(userId=USER_ID, isOwner=False)
        self.prisma.orgmember.find_unique = AsyncMock(
            side_effect=[not_owner]
        )

        with pytest.raises(ValueError, match="not the org owner"):
            await transfer_ownership(ORG_ID, USER_ID, OTHER_USER_ID)

    @pytest.mark.asyncio
    async def test_transfer_when_current_not_found_raises(self):
        from backend.api.features.orgs.db import transfer_ownership

        self.prisma.orgmember.find_unique = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not the org owner"):
            await transfer_ownership(ORG_ID, "no-one", OTHER_USER_ID)


class TestOrgDbAliases:
    """Tests for alias operations in db.py."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_create_alias_duplicate_raises(self):
        from backend.api.features.orgs.db import create_org_alias

        self.prisma.organization.find_unique = AsyncMock(return_value=None)
        self.prisma.organizationalias.find_unique = AsyncMock(
            return_value=MagicMock(aliasSlug="existing-alias")
        )

        with pytest.raises(ValueError, match="already used as an alias"):
            await create_org_alias(ORG_ID, "existing-alias", USER_ID)

    @pytest.mark.asyncio
    async def test_create_alias_slug_taken_by_org_raises(self):
        from backend.api.features.orgs.db import create_org_alias

        self.prisma.organization.find_unique = AsyncMock(
            return_value=_make_org(slug="org-slug")
        )

        with pytest.raises(ValueError, match="already used by an organization"):
            await create_org_alias(ORG_ID, "org-slug", USER_ID)

    @pytest.mark.asyncio
    async def test_create_alias_success(self):
        from backend.api.features.orgs.db import create_org_alias

        self.prisma.organization.find_unique = AsyncMock(return_value=None)
        self.prisma.organizationalias.find_unique = AsyncMock(
            return_value=None
        )
        alias_mock = MagicMock()
        alias_mock.id = "alias-1"
        alias_mock.aliasSlug = "my-alias"
        alias_mock.aliasType = "MANUAL"
        alias_mock.createdAt = FIXED_NOW
        self.prisma.organizationalias.create = AsyncMock(
            return_value=alias_mock
        )

        result = await create_org_alias(ORG_ID, "my-alias", USER_ID)
        assert result["aliasSlug"] == "my-alias"
        assert result["aliasType"] == "MANUAL"

    @pytest.mark.asyncio
    async def test_list_aliases_filters_removed(self):
        from backend.api.features.orgs.db import list_org_aliases

        a1 = MagicMock(
            id="a1", aliasSlug="old-name", aliasType="RENAME", createdAt=FIXED_NOW
        )
        self.prisma.organizationalias.find_many = AsyncMock(return_value=[a1])

        result = await list_org_aliases(ORG_ID)

        assert len(result) == 1
        assert result[0]["aliasSlug"] == "old-name"

        # Verify the query filters by removedAt=None
        find_call = self.prisma.organizationalias.find_many.call_args
        assert find_call[1]["where"]["removedAt"] is None


# ═══════════════════════════════════════════════════════════════════════════════
# 1b. ORG ROUTES  (routes.py) — HTTP-level tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrgRoutes:
    """HTTP-level tests for org routes including security checks."""

    @pytest.fixture(autouse=True)
    def _setup(self, mocker, mock_jwt_user):
        from backend.api.features.orgs.routes import router as org_router

        self.app = fastapi.FastAPI()
        self.app.include_router(org_router, prefix="/orgs")

        from autogpt_libs.auth.jwt_utils import get_jwt_payload

        self.app.dependency_overrides[get_jwt_payload] = mock_jwt_user[
            "get_jwt_payload"
        ]
        self._user_id = mock_jwt_user["user_id"]

        # Mock the db module
        self.mock_db = mocker.patch(
            "backend.api.features.orgs.routes.org_db"
        )

        self.client = fastapi.testclient.TestClient(self.app)
        yield
        self.app.dependency_overrides.clear()

    def test_create_org_returns_200(self):
        self.mock_db.create_org = AsyncMock(
            return_value={
                "id": ORG_ID,
                "name": "Acme",
                "slug": "acme",
                "avatarUrl": None,
                "description": None,
                "isPersonal": False,
                "memberCount": 1,
                "createdAt": FIXED_NOW.isoformat(),
            }
        )

        resp = self.client.post(
            "/orgs",
            json={"name": "Acme", "slug": "acme"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["slug"] == "acme"
        assert data["isPersonal"] is False

    def test_write_route_rejects_mismatched_org_id(self):
        """SECURITY: _verify_org_path blocks when ctx.org_id != path org_id."""
        from backend.api.features.orgs.routes import _verify_org_path

        ctx = _owner_ctx(org_id="org-A", user_id=self._user_id)
        with pytest.raises(fastapi.HTTPException) as exc_info:
            _verify_org_path(ctx, "org-B")

        assert exc_info.value.status_code == 403
        assert "Not a member" in exc_info.value.detail

    def test_write_route_accepts_matching_org_id(self):
        """_verify_org_path passes silently when ctx.org_id matches."""
        from backend.api.features.orgs.routes import _verify_org_path

        ctx = _owner_ctx(org_id="org-A", user_id=self._user_id)
        # Should not raise
        _verify_org_path(ctx, "org-A")

    def test_get_org_mismatched_ctx_returns_403(self):
        """GET /orgs/{org_id} also checks ctx.org_id matches via inline check."""
        from backend.api.features.orgs.routes import get_org

        wrong_ctx = _member_ctx(org_id="org-X")

        # The route handler directly checks ctx.org_id != org_id
        with pytest.raises(fastapi.HTTPException) as exc_info:
            # Call the async handler synchronously using run
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                get_org(org_id="org-Y", ctx=wrong_ctx)
            )

        assert exc_info.value.status_code == 403


# ═══════════════════════════════════════════════════════════════════════════════
# 2. WORKSPACE CRUD (workspace_db.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestWorkspaceDbCreate:
    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch(
            "backend.api.features.orgs.workspace_db.prisma", self.prisma
        )

    @pytest.mark.asyncio
    async def test_create_workspace_creator_becomes_admin(self):
        from backend.api.features.orgs.workspace_db import create_workspace

        ws = _make_workspace(id="ws-new", isDefault=False)
        self.prisma.orgworkspace.create = AsyncMock(return_value=ws)
        self.prisma.orgworkspacemember.create = AsyncMock()

        result = await create_workspace(
            org_id=ORG_ID, name="Dev", user_id=USER_ID
        )

        assert result["name"] == "Default"  # from mock
        assert result["memberCount"] == 1

        # Verify creator is marked as admin
        wsm_data = self.prisma.orgworkspacemember.create.call_args[1]["data"]
        assert wsm_data["isAdmin"] is True
        assert wsm_data["userId"] == USER_ID
        assert wsm_data["status"] == "ACTIVE"


class TestWorkspaceDbJoinLeave:
    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch(
            "backend.api.features.orgs.workspace_db.prisma", self.prisma
        )

    @pytest.mark.asyncio
    async def test_join_open_workspace_success(self):
        from backend.api.features.orgs.workspace_db import join_workspace

        open_ws = _make_workspace(joinPolicy="OPEN")
        self.prisma.orgworkspace.find_unique = AsyncMock(return_value=open_ws)
        self.prisma.orgworkspacemember.find_unique = AsyncMock(
            return_value=None
        )
        self.prisma.orgworkspacemember.create = AsyncMock()

        result = await join_workspace(WS_ID, OTHER_USER_ID, ORG_ID)

        assert result["id"] == WS_ID
        self.prisma.orgworkspacemember.create.assert_called_once()
        create_data = self.prisma.orgworkspacemember.create.call_args[1]["data"]
        assert create_data["userId"] == OTHER_USER_ID

    @pytest.mark.asyncio
    async def test_join_open_workspace_already_member_is_idempotent(self):
        from backend.api.features.orgs.workspace_db import join_workspace

        open_ws = _make_workspace(joinPolicy="OPEN")
        self.prisma.orgworkspace.find_unique = AsyncMock(return_value=open_ws)
        self.prisma.orgworkspacemember.find_unique = AsyncMock(
            return_value=_make_ws_member()
        )

        await join_workspace(WS_ID, USER_ID, ORG_ID)

        # Should return workspace without creating a duplicate member
        self.prisma.orgworkspacemember.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_join_private_workspace_raises(self):
        from backend.api.features.orgs.workspace_db import join_workspace

        private_ws = _make_workspace(joinPolicy="PRIVATE")
        self.prisma.orgworkspace.find_unique = AsyncMock(
            return_value=private_ws
        )

        with pytest.raises(ValueError, match="Cannot self-join a PRIVATE"):
            await join_workspace(WS_ID, OTHER_USER_ID, ORG_ID)

    @pytest.mark.asyncio
    async def test_join_workspace_wrong_org_raises(self):
        from backend.api.features.orgs.workspace_db import join_workspace

        ws = _make_workspace(orgId="other-org")
        self.prisma.orgworkspace.find_unique = AsyncMock(return_value=ws)

        with pytest.raises(ValueError, match="does not belong"):
            await join_workspace(WS_ID, USER_ID, ORG_ID)

    @pytest.mark.asyncio
    async def test_leave_default_workspace_raises(self):
        from backend.api.features.orgs.workspace_db import leave_workspace

        default_ws = _make_workspace(isDefault=True)
        self.prisma.orgworkspace.find_unique = AsyncMock(
            return_value=default_ws
        )

        with pytest.raises(ValueError, match="Cannot leave the default"):
            await leave_workspace(WS_ID, USER_ID)

    @pytest.mark.asyncio
    async def test_leave_non_default_workspace_success(self):
        from backend.api.features.orgs.workspace_db import leave_workspace

        ws = _make_workspace(isDefault=False)
        self.prisma.orgworkspace.find_unique = AsyncMock(return_value=ws)
        self.prisma.orgworkspacemember.delete_many = AsyncMock()

        await leave_workspace(WS_ID, USER_ID)

        self.prisma.orgworkspacemember.delete_many.assert_called_once()


class TestWorkspaceDbDelete:
    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch(
            "backend.api.features.orgs.workspace_db.prisma", self.prisma
        )

    @pytest.mark.asyncio
    async def test_delete_default_workspace_raises(self):
        from backend.api.features.orgs.workspace_db import delete_workspace

        default_ws = _make_workspace(isDefault=True)
        self.prisma.orgworkspace.find_unique = AsyncMock(
            return_value=default_ws
        )

        with pytest.raises(ValueError, match="Cannot delete the default"):
            await delete_workspace(WS_ID)

    @pytest.mark.asyncio
    async def test_delete_non_default_workspace_success(self):
        from backend.api.features.orgs.workspace_db import delete_workspace

        ws = _make_workspace(isDefault=False)
        self.prisma.orgworkspace.find_unique = AsyncMock(return_value=ws)
        self.prisma.orgworkspace.delete = AsyncMock()

        await delete_workspace(WS_ID)

        self.prisma.orgworkspace.delete.assert_called_once_with(
            where={"id": WS_ID}
        )

    @pytest.mark.asyncio
    async def test_delete_workspace_not_found_raises(self):
        from backend.api.features.orgs.workspace_db import delete_workspace

        self.prisma.orgworkspace.find_unique = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError):
            await delete_workspace("nonexistent-ws")


class TestWorkspaceDbGetWorkspace:
    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch(
            "backend.api.features.orgs.workspace_db.prisma", self.prisma
        )

    @pytest.mark.asyncio
    async def test_get_workspace_wrong_org_raises(self):
        from backend.api.features.orgs.workspace_db import get_workspace

        ws = _make_workspace(orgId="org-real")
        self.prisma.orgworkspace.find_unique = AsyncMock(return_value=ws)

        with pytest.raises(NotFoundError, match="not found in org"):
            await get_workspace(WS_ID, expected_org_id="org-wrong")

    @pytest.mark.asyncio
    async def test_get_workspace_correct_org_success(self):
        from backend.api.features.orgs.workspace_db import get_workspace

        ws = _make_workspace(orgId=ORG_ID)
        self.prisma.orgworkspace.find_unique = AsyncMock(return_value=ws)

        result = await get_workspace(WS_ID, expected_org_id=ORG_ID)
        assert result["id"] == WS_ID
        assert result["orgId"] == ORG_ID

    @pytest.mark.asyncio
    async def test_get_workspace_no_org_check_success(self):
        from backend.api.features.orgs.workspace_db import get_workspace

        ws = _make_workspace(orgId="any-org")
        self.prisma.orgworkspace.find_unique = AsyncMock(return_value=ws)

        result = await get_workspace(WS_ID)
        assert result["id"] == WS_ID


class TestWorkspaceDbMembers:
    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch(
            "backend.api.features.orgs.workspace_db.prisma", self.prisma
        )

    @pytest.mark.asyncio
    async def test_add_workspace_member_requires_org_membership(self):
        from backend.api.features.orgs.workspace_db import (
            add_workspace_member,
        )

        self.prisma.orgmember.find_unique = AsyncMock(return_value=None)

        with pytest.raises(ValueError, match="not a member of the organization"):
            await add_workspace_member(
                ws_id=WS_ID,
                user_id="outsider",
                org_id=ORG_ID,
            )

    @pytest.mark.asyncio
    async def test_add_workspace_member_success(self):
        from backend.api.features.orgs.workspace_db import (
            add_workspace_member,
        )

        org_mem = _make_member(userId=OTHER_USER_ID)
        self.prisma.orgmember.find_unique = AsyncMock(return_value=org_mem)
        ws_mem = _make_ws_member(
            userId=OTHER_USER_ID,
            isAdmin=True,
            user_email="bob@example.com",
        )
        self.prisma.orgworkspacemember.create = AsyncMock(return_value=ws_mem)

        result = await add_workspace_member(
            ws_id=WS_ID,
            user_id=OTHER_USER_ID,
            org_id=ORG_ID,
            is_admin=True,
        )

        assert result["userId"] == OTHER_USER_ID
        assert result["isAdmin"] is True

    @pytest.mark.asyncio
    async def test_list_workspace_members_returns_active_only(self):
        from backend.api.features.orgs.workspace_db import (
            list_workspace_members,
        )

        m1 = _make_ws_member(
            userId="u1", user_email="a@example.com", user_name="A"
        )
        m2 = _make_ws_member(
            userId="u2", user_email="b@example.com", user_name="B"
        )
        self.prisma.orgworkspacemember.find_many = AsyncMock(
            return_value=[m1, m2]
        )

        result = await list_workspace_members(WS_ID)

        assert len(result) == 2
        # Verify the query filters for active status
        find_call = self.prisma.orgworkspacemember.find_many.call_args
        assert find_call[1]["where"]["status"] == "ACTIVE"


# ═══════════════════════════════════════════════════════════════════════════════
# 2b. WORKSPACE ROUTES — HTTP-level tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestWorkspaceRoutes:
    """Route-level security tests for workspace endpoints.

    Since the route handlers use inline `ctx.org_id != org_id` checks,
    we test the handlers directly rather than fighting with nested
    FastAPI Security dependency overrides.
    """

    def test_list_workspaces_wrong_org_returns_403(self):
        """list_workspaces raises 403 when ctx.org_id != path org_id."""
        from backend.api.features.orgs.workspace_routes import list_workspaces

        ctx = _member_ctx(org_id="org-X")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                list_workspaces(org_id="org-Y", ctx=ctx)
            )

        assert exc_info.value.status_code == 403

    def test_list_workspace_members_requires_org_membership(self, mocker):
        """list_members raises 403 when ctx.org_id != path org_id."""
        from backend.api.features.orgs.workspace_routes import list_members

        ctx = _member_ctx(org_id="org-mine")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                list_members(
                    org_id="org-different", ws_id=WS_ID, ctx=ctx
                )
            )

        assert exc_info.value.status_code == 403

    def test_get_workspace_wrong_org_returns_403(self):
        """get_workspace raises 403 when ctx.org_id != path org_id."""
        from backend.api.features.orgs.workspace_routes import get_workspace

        ctx = _member_ctx(org_id="org-A")

        with pytest.raises(fastapi.HTTPException) as exc_info:
            import asyncio

            asyncio.get_event_loop().run_until_complete(
                get_workspace(org_id="org-B", ws_id=WS_ID, ctx=ctx)
            )

        assert exc_info.value.status_code == 403


# ═══════════════════════════════════════════════════════════════════════════════
# 3. INVITATION (invitation_routes.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestInvitationAcceptance:
    """Tests for the token-based invitation acceptance flow."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch(
            "backend.api.features.orgs.invitation_routes.prisma", self.prisma
        )
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    def _make_invitation(
        self,
        *,
        token="tok-abc",
        email="alice@example.com",
        orgId=ORG_ID,
        acceptedAt=None,
        revokedAt=None,
        expiresAt=None,
        isAdmin=False,
        isBillingManager=False,
        workspaceIds=None,
    ):
        inv = MagicMock()
        inv.id = "inv-1"
        inv.token = token
        inv.email = email
        inv.orgId = orgId
        inv.isAdmin = isAdmin
        inv.isBillingManager = isBillingManager
        inv.acceptedAt = acceptedAt
        inv.revokedAt = revokedAt
        inv.expiresAt = expiresAt or (
            datetime.now(timezone.utc) + timedelta(days=3)
        )
        inv.createdAt = FIXED_NOW
        inv.invitedByUserId = USER_ID
        inv.workspaceIds = workspaceIds or []
        return inv

    @pytest.fixture
    def _app_and_client(self, mock_jwt_user):
        from backend.api.features.orgs.invitation_routes import router

        app = fastapi.FastAPI()
        app.include_router(router, prefix="/invitations")

        from autogpt_libs.auth.jwt_utils import get_jwt_payload

        app.dependency_overrides[get_jwt_payload] = mock_jwt_user[
            "get_jwt_payload"
        ]
        client = fastapi.testclient.TestClient(app)
        yield app, client
        app.dependency_overrides.clear()

    def test_accept_invitation_success(self, _app_and_client, test_user_id):
        app, client = _app_and_client

        invitation = self._make_invitation(email="test@example.com")
        self.prisma.orginvitation.find_unique = AsyncMock(
            return_value=invitation
        )
        accepting_user = MagicMock(
            id=test_user_id, email="test@example.com"
        )
        self.prisma.user.find_unique = AsyncMock(return_value=accepting_user)

        # Mock add_org_member chain
        new_member = _make_member(userId=test_user_id)
        self.prisma.orgmember.create = AsyncMock(return_value=new_member)
        self.prisma.orgworkspace.find_first = AsyncMock(return_value=None)
        self.prisma.orginvitation.update = AsyncMock()

        resp = client.post("/invitations/tok-abc/accept")

        assert resp.status_code == 200
        data = resp.json()
        assert data["orgId"] == ORG_ID
        assert "accepted" in data["message"].lower()

    def test_accept_expired_invitation_raises(self, _app_and_client):
        _, client = _app_and_client

        expired = self._make_invitation(
            email="test@example.com",
            expiresAt=datetime.now(timezone.utc) - timedelta(days=1),
        )
        self.prisma.orginvitation.find_unique = AsyncMock(return_value=expired)
        user = MagicMock(id="test-user-id", email="test@example.com")
        self.prisma.user.find_unique = AsyncMock(return_value=user)

        resp = client.post("/invitations/tok-expired/accept")

        assert resp.status_code == 400
        assert "expired" in resp.json()["detail"].lower()

    def test_accept_revoked_invitation_raises(self, _app_and_client):
        _, client = _app_and_client

        revoked = self._make_invitation(
            email="test@example.com",
            revokedAt=FIXED_NOW,
        )
        self.prisma.orginvitation.find_unique = AsyncMock(return_value=revoked)

        resp = client.post("/invitations/tok-revoked/accept")

        assert resp.status_code == 400
        assert "revoked" in resp.json()["detail"].lower()

    def test_accept_already_accepted_raises(self, _app_and_client):
        _, client = _app_and_client

        already = self._make_invitation(
            email="test@example.com",
            acceptedAt=FIXED_NOW,
        )
        self.prisma.orginvitation.find_unique = AsyncMock(return_value=already)

        resp = client.post("/invitations/tok-used/accept")

        assert resp.status_code == 400
        assert "already accepted" in resp.json()["detail"].lower()

    def test_accept_wrong_email_raises_403(self, _app_and_client, test_user_id):
        """Bob cannot accept an invitation sent to alice@example.com."""
        _, client = _app_and_client

        alice_inv = self._make_invitation(email="alice@example.com")
        self.prisma.orginvitation.find_unique = AsyncMock(
            return_value=alice_inv
        )
        bob = MagicMock(id=test_user_id, email="test@example.com")
        self.prisma.user.find_unique = AsyncMock(return_value=bob)

        resp = client.post("/invitations/tok-abc/accept")

        assert resp.status_code == 403
        assert "different email" in resp.json()["detail"].lower()

    def test_accept_invitation_not_found_raises_error(
        self, _app_and_client
    ):
        """When invitation is not found, NotFoundError is raised.  TestClient
        propagates unhandled exceptions, so we catch it directly."""
        app, _ = _app_and_client

        self.prisma.orginvitation.find_unique = AsyncMock(return_value=None)

        # Use raise_server_exceptions=False to get the 500 response
        safe_client = fastapi.testclient.TestClient(
            app, raise_server_exceptions=False
        )
        resp = safe_client.post("/invitations/nonexistent-token/accept")
        assert resp.status_code == 500

    def test_decline_invitation(self, _app_and_client):
        _, client = _app_and_client

        invitation = self._make_invitation()
        self.prisma.orginvitation.find_unique = AsyncMock(
            return_value=invitation
        )
        self.prisma.orginvitation.update = AsyncMock()

        resp = client.post("/invitations/tok-abc/decline")

        assert resp.status_code == 204
        # Should have set revokedAt
        self.prisma.orginvitation.update.assert_called_once()
        update_data = self.prisma.orginvitation.update.call_args[1]["data"]
        assert "revokedAt" in update_data


class TestInvitationListPending:
    """Tests for listing pending invitations for the current user."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch(
            "backend.api.features.orgs.invitation_routes.prisma", self.prisma
        )

    @pytest.fixture
    def _app_and_client(self, mock_jwt_user):
        from backend.api.features.orgs.invitation_routes import router

        app = fastapi.FastAPI()
        app.include_router(router, prefix="/invitations")

        from autogpt_libs.auth.jwt_utils import get_jwt_payload

        app.dependency_overrides[get_jwt_payload] = mock_jwt_user[
            "get_jwt_payload"
        ]
        client = fastapi.testclient.TestClient(app)
        yield app, client
        app.dependency_overrides.clear()

    def test_list_pending_for_user(self, _app_and_client):
        _, client = _app_and_client

        user = MagicMock(email="test@example.com")
        self.prisma.user.find_unique = AsyncMock(return_value=user)

        inv = MagicMock()
        inv.id = "inv-1"
        inv.email = "test@example.com"
        inv.isAdmin = False
        inv.isBillingManager = False
        inv.token = "tok-1"
        inv.expiresAt = datetime.now(timezone.utc) + timedelta(days=5)
        inv.createdAt = FIXED_NOW
        inv.workspaceIds = []
        self.prisma.orginvitation.find_many = AsyncMock(return_value=[inv])

        resp = client.get("/invitations/pending")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["email"] == "test@example.com"

    def test_list_pending_no_user_returns_empty(self, _app_and_client):
        _, client = _app_and_client

        self.prisma.user.find_unique = AsyncMock(return_value=None)

        resp = client.get("/invitations/pending")

        assert resp.status_code == 200
        assert resp.json() == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CREDITS (org_credit.py)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrgCreditsSpend:
    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.data.org_credit.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_spend_credits_success(self):
        from backend.data.org_credit import spend_org_credits

        # Atomic deduct succeeds (1 row affected)
        self.prisma.execute_raw = AsyncMock(return_value=1)
        # Balance after spend
        balance_row = MagicMock(balance=90)
        self.prisma.orgbalance.find_unique = AsyncMock(
            return_value=balance_row
        )
        self.prisma.orgcredittransaction.create = AsyncMock()

        remaining = await spend_org_credits(
            org_id=ORG_ID, user_id=USER_ID, amount=10
        )

        assert remaining == 90

        # Transaction recorded with negative amount
        tx_data = self.prisma.orgcredittransaction.create.call_args[1]["data"]
        assert tx_data["amount"] == -10
        assert tx_data["orgId"] == ORG_ID
        assert tx_data["runningBalance"] == 90

    @pytest.mark.asyncio
    async def test_spend_credits_insufficient_balance_raises(self):
        from backend.data.org_credit import spend_org_credits

        # Atomic deduct fails (0 rows affected)
        self.prisma.execute_raw = AsyncMock(return_value=0)
        balance_row = MagicMock(balance=5)
        self.prisma.orgbalance.find_unique = AsyncMock(
            return_value=balance_row
        )

        with pytest.raises(InsufficientBalanceError) as exc_info:
            await spend_org_credits(
                org_id=ORG_ID, user_id=USER_ID, amount=100
            )

        assert exc_info.value.balance == 5
        assert exc_info.value.amount == 100

    @pytest.mark.asyncio
    async def test_spend_zero_raises_value_error(self):
        from backend.data.org_credit import spend_org_credits

        with pytest.raises(ValueError, match="must be positive"):
            await spend_org_credits(
                org_id=ORG_ID, user_id=USER_ID, amount=0
            )

    @pytest.mark.asyncio
    async def test_spend_negative_raises_value_error(self):
        from backend.data.org_credit import spend_org_credits

        with pytest.raises(ValueError, match="must be positive"):
            await spend_org_credits(
                org_id=ORG_ID, user_id=USER_ID, amount=-5
            )

    @pytest.mark.asyncio
    async def test_spend_credits_with_metadata_and_workspace(self):
        from backend.data.org_credit import spend_org_credits

        self.prisma.execute_raw = AsyncMock(return_value=1)
        self.prisma.orgbalance.find_unique = AsyncMock(
            return_value=MagicMock(balance=50)
        )
        self.prisma.orgcredittransaction.create = AsyncMock()

        await spend_org_credits(
            org_id=ORG_ID,
            user_id=USER_ID,
            amount=10,
            workspace_id="ws-1",
            metadata={"reason": "block execution"},
        )

        tx_data = self.prisma.orgcredittransaction.create.call_args[1]["data"]
        assert tx_data["workspaceId"] == "ws-1"
        assert tx_data["metadata"] is not None

    @pytest.mark.asyncio
    async def test_spend_credits_no_balance_row_returns_zero_in_error(self):
        from backend.data.org_credit import spend_org_credits

        self.prisma.execute_raw = AsyncMock(return_value=0)
        # No balance row at all
        self.prisma.orgbalance.find_unique = AsyncMock(return_value=None)

        with pytest.raises(InsufficientBalanceError) as exc_info:
            await spend_org_credits(
                org_id=ORG_ID, user_id=USER_ID, amount=1
            )

        assert exc_info.value.balance == 0


class TestOrgCreditsTopUp:
    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.data.org_credit.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_top_up_creates_balance_if_not_exists(self):
        from backend.data.org_credit import top_up_org_credits

        self.prisma.execute_raw = AsyncMock(return_value=1)
        # Balance after top-up
        self.prisma.orgbalance.find_unique = AsyncMock(
            return_value=MagicMock(balance=100)
        )
        self.prisma.orgcredittransaction.create = AsyncMock()

        new_balance = await top_up_org_credits(
            org_id=ORG_ID, amount=100, user_id=USER_ID
        )

        assert new_balance == 100

        # Verify raw SQL uses INSERT ... ON CONFLICT (upsert)
        raw_sql = self.prisma.execute_raw.call_args[0][0]
        assert "INSERT" in raw_sql
        assert "ON CONFLICT" in raw_sql

        # Transaction recorded with positive amount
        tx_data = self.prisma.orgcredittransaction.create.call_args[1]["data"]
        assert tx_data["amount"] == 100

    @pytest.mark.asyncio
    async def test_top_up_zero_raises_value_error(self):
        from backend.data.org_credit import top_up_org_credits

        with pytest.raises(ValueError, match="must be positive"):
            await top_up_org_credits(org_id=ORG_ID, amount=0)

    @pytest.mark.asyncio
    async def test_top_up_negative_raises_value_error(self):
        from backend.data.org_credit import top_up_org_credits

        with pytest.raises(ValueError, match="must be positive"):
            await top_up_org_credits(org_id=ORG_ID, amount=-10)

    @pytest.mark.asyncio
    async def test_top_up_without_user_id_omits_field(self):
        from backend.data.org_credit import top_up_org_credits

        self.prisma.execute_raw = AsyncMock(return_value=1)
        self.prisma.orgbalance.find_unique = AsyncMock(
            return_value=MagicMock(balance=50)
        )
        self.prisma.orgcredittransaction.create = AsyncMock()

        await top_up_org_credits(org_id=ORG_ID, amount=50)

        tx_data = self.prisma.orgcredittransaction.create.call_args[1]["data"]
        assert "initiatedByUserId" not in tx_data


class TestOrgCreditsGet:
    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.data.org_credit.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_get_credits_returns_balance(self):
        from backend.data.org_credit import get_org_credits

        self.prisma.orgbalance.find_unique = AsyncMock(
            return_value=MagicMock(balance=42)
        )

        result = await get_org_credits(ORG_ID)
        assert result == 42

    @pytest.mark.asyncio
    async def test_get_credits_no_row_returns_zero(self):
        from backend.data.org_credit import get_org_credits

        self.prisma.orgbalance.find_unique = AsyncMock(return_value=None)

        result = await get_org_credits(ORG_ID)
        assert result == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MIGRATION (org_migration.py) — additional edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestMigrationSlugEdgeCases:
    """Additional edge cases beyond what org_migration_test.py covers."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        self.prisma.organization.find_unique = AsyncMock(return_value=None)
        self.prisma.organizationalias.find_unique = AsyncMock(
            return_value=None
        )
        self.prisma.organization.create = AsyncMock(
            return_value=MagicMock(id="org-new")
        )
        self.prisma.orgmember.create = AsyncMock()
        self.prisma.orgworkspace.create = AsyncMock(
            return_value=MagicMock(id="ws-new")
        )
        self.prisma.orgworkspacemember.create = AsyncMock()
        self.prisma.organizationprofile.create = AsyncMock()
        self.prisma.organizationseatassignment.create = AsyncMock()
        self.prisma.query_raw = AsyncMock(return_value=[])
        self.prisma.execute_raw = AsyncMock(return_value=0)
        mocker.patch("backend.data.org_migration.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_user_with_name_but_no_profile_uses_name_slug(self):
        from backend.data.org_migration import create_orgs_for_existing_users

        self.prisma.query_raw = AsyncMock(
            return_value=[
                {
                    "id": "user-3",
                    "email": "charlie@example.com",
                    "name": "Charlie Brown",
                    "stripeCustomerId": None,
                    "topUpConfig": None,
                    "profile_username": None,
                    "profile_name": None,
                    "profile_description": None,
                    "profile_avatar_url": None,
                    "profile_links": None,
                },
            ]
        )

        result = await create_orgs_for_existing_users()
        assert result == 1

        create_data = self.prisma.organization.create.call_args[1]["data"]
        # "Charlie Brown" -> "charlie-brown"
        assert create_data["slug"] == "charlie-brown"
        # display_name falls through: profile_name(None) -> user_name("Charlie Brown")
        assert create_data["name"] == "Charlie Brown"

    @pytest.mark.asyncio
    async def test_empty_email_uses_user_id_slug(self):
        from backend.data.org_migration import create_orgs_for_existing_users

        self.prisma.query_raw = AsyncMock(
            return_value=[
                {
                    "id": "abcd1234-rest-of-uuid",
                    "email": "",
                    "name": None,
                    "stripeCustomerId": None,
                    "topUpConfig": None,
                    "profile_username": None,
                    "profile_name": None,
                    "profile_description": None,
                    "profile_avatar_url": None,
                    "profile_links": None,
                },
            ]
        )

        result = await create_orgs_for_existing_users()
        assert result == 1

        create_data = self.prisma.organization.create.call_args[1]["data"]
        # Empty email local part -> user-{id[:8]}
        assert create_data["slug"] == "user-abcd1234"

    @pytest.mark.asyncio
    async def test_resolve_unique_slug_exhaustion_raises_runtime_error(self):
        """When all 10000 suffixes are taken, RuntimeError should be raised."""
        from backend.data.org_migration import _resolve_unique_slug

        # Every slug is "taken"
        self.prisma.organization.find_unique = AsyncMock(
            return_value=MagicMock(id="taken")
        )

        with pytest.raises(RuntimeError, match="10000 attempts"):
            await _resolve_unique_slug("occupied")


class TestMigrationSanitizeSlugAdditional:
    """Additional _sanitize_slug edge cases."""

    def test_mixed_case_and_numbers(self):
        from backend.data.org_migration import _sanitize_slug

        assert _sanitize_slug("MyOrg123") == "myorg123"

    def test_leading_hyphens_stripped(self):
        from backend.data.org_migration import _sanitize_slug

        assert _sanitize_slug("---hello") == "hello"

    def test_consecutive_spaces_become_single_hyphen(self):
        from backend.data.org_migration import _sanitize_slug

        assert _sanitize_slug("a   b") == "a-b"

    def test_tab_and_newline_treated_as_special(self):
        from backend.data.org_migration import _sanitize_slug

        assert _sanitize_slug("hello\tworld\n") == "hello-world"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ADDITIONAL DB EDGE CASES
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrgDbGetOrg:
    """Edge cases for get_org."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_get_org_not_found_raises(self):
        from backend.api.features.orgs.db import get_org

        self.prisma.organization.find_unique = AsyncMock(return_value=None)

        with pytest.raises(NotFoundError, match="not found"):
            await get_org("nonexistent")

    @pytest.mark.asyncio
    async def test_get_org_returns_member_count(self):
        from backend.api.features.orgs.db import get_org

        org = _make_org()
        org.Members_count = 5
        self.prisma.organization.find_unique = AsyncMock(return_value=org)

        result = await get_org(ORG_ID)
        assert result["memberCount"] == 5

    @pytest.mark.asyncio
    async def test_get_org_without_members_count_attr_returns_zero(self):
        from backend.api.features.orgs.db import get_org

        org = _make_org()
        # MagicMock will have Members_count as an attribute automatically,
        # so we need to delete it to simulate the missing attribute
        del org.Members_count
        self.prisma.organization.find_unique = AsyncMock(return_value=org)

        result = await get_org(ORG_ID)
        assert result["memberCount"] == 0


class TestOrgDbListUserOrgs:
    """Tests for list_user_orgs."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_list_user_orgs_empty(self):
        from backend.api.features.orgs.db import list_user_orgs

        self.prisma.orgmember.find_many = AsyncMock(return_value=[])

        result = await list_user_orgs(USER_ID)
        assert result == []

    @pytest.mark.asyncio
    async def test_list_user_orgs_skips_null_org(self):
        from backend.api.features.orgs.db import list_user_orgs

        mem_with_null_org = MagicMock()
        mem_with_null_org.Org = None
        self.prisma.orgmember.find_many = AsyncMock(
            return_value=[mem_with_null_org]
        )

        result = await list_user_orgs(USER_ID)
        assert result == []

    @pytest.mark.asyncio
    async def test_list_user_orgs_returns_multiple(self):
        from backend.api.features.orgs.db import list_user_orgs

        m1 = _make_member(orgId="org-1")
        m1.Org = _make_org(id="org-1", name="Org 1", slug="org-1")
        m2 = _make_member(orgId="org-2")
        m2.Org = _make_org(id="org-2", name="Org 2", slug="org-2")
        self.prisma.orgmember.find_many = AsyncMock(return_value=[m1, m2])

        result = await list_user_orgs(USER_ID)
        assert len(result) == 2
        assert result[0]["id"] == "org-1"
        assert result[1]["id"] == "org-2"


class TestOrgDbListMembers:
    """Tests for list_org_members."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_list_members_returns_correct_fields(self):
        from backend.api.features.orgs.db import list_org_members

        m = _make_member(
            userId="u1",
            isOwner=True,
            isAdmin=True,
            isBillingManager=False,
            user_email="owner@example.com",
            user_name="Owner",
        )
        self.prisma.orgmember.find_many = AsyncMock(return_value=[m])

        result = await list_org_members(ORG_ID)

        assert len(result) == 1
        assert result[0]["userId"] == "u1"
        assert result[0]["isOwner"] is True
        assert result[0]["email"] == "owner@example.com"
        assert result[0]["name"] == "Owner"

    @pytest.mark.asyncio
    async def test_list_members_with_null_user_returns_empty_email(self):
        from backend.api.features.orgs.db import list_org_members

        m = _make_member(userId="u1")
        m.User = None
        self.prisma.orgmember.find_many = AsyncMock(return_value=[m])

        result = await list_org_members(ORG_ID)

        assert result[0]["email"] == ""
        assert result[0]["name"] is None


class TestOrgDbUpdateMemberNoop:
    """Test update_org_member no-op when no fields to update."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_update_member_no_changes_skips_db_update(self):
        from backend.api.features.orgs.db import update_org_member

        member = _make_member(userId=OTHER_USER_ID, isOwner=False)
        self.prisma.orgmember.find_unique = AsyncMock(return_value=member)
        self.prisma.orgmember.update = AsyncMock()

        # Both None => no update_data => skip prisma update
        list_member = _make_member(userId=OTHER_USER_ID)
        self.prisma.orgmember.find_many = AsyncMock(return_value=[list_member])

        result = await update_org_member(
            ORG_ID, OTHER_USER_ID, is_admin=None, is_billing_manager=None
        )

        self.prisma.orgmember.update.assert_not_called()
        assert result["userId"] == OTHER_USER_ID


class TestOrgCreditsTransactionHistory:
    """Tests for get_org_transaction_history."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.data.org_credit.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_transaction_history_returns_formatted(self):
        from backend.data.org_credit import get_org_transaction_history

        tx = MagicMock()
        tx.transactionKey = "tx-1"
        tx.createdAt = FIXED_NOW
        tx.amount = -10
        tx.type = "USAGE"
        tx.runningBalance = 90
        tx.initiatedByUserId = USER_ID
        tx.workspaceId = None
        tx.metadata = None
        self.prisma.orgcredittransaction.find_many = AsyncMock(
            return_value=[tx]
        )

        result = await get_org_transaction_history(ORG_ID)

        assert len(result) == 1
        assert result[0]["transactionKey"] == "tx-1"
        assert result[0]["amount"] == -10

    @pytest.mark.asyncio
    async def test_transaction_history_respects_limit_and_offset(self):
        from backend.data.org_credit import get_org_transaction_history

        self.prisma.orgcredittransaction.find_many = AsyncMock(
            return_value=[]
        )

        await get_org_transaction_history(ORG_ID, limit=10, offset=5)

        call_kwargs = self.prisma.orgcredittransaction.find_many.call_args[1]
        assert call_kwargs["take"] == 10
        assert call_kwargs["skip"] == 5


class TestOrgCreditsSeatInfo:
    """Tests for seat management."""

    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.data.org_credit.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_get_seat_info_counts_active_and_inactive(self):
        from backend.data.org_credit import get_seat_info

        s1 = MagicMock(userId="u1", seatType="FREE", status="ACTIVE", createdAt=FIXED_NOW)
        s2 = MagicMock(userId="u2", seatType="FREE", status="INACTIVE", createdAt=FIXED_NOW)
        s3 = MagicMock(userId="u3", seatType="PRO", status="ACTIVE", createdAt=FIXED_NOW)
        self.prisma.organizationseatassignment.find_many = AsyncMock(
            return_value=[s1, s2, s3]
        )

        result = await get_seat_info(ORG_ID)

        assert result["total"] == 3
        assert result["active"] == 2
        assert result["inactive"] == 1
        assert len(result["seats"]) == 3
