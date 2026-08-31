"""Tests for org shared-memory governance.

Two surfaces:
  1. The holdBuffer settings toggle riding on PATCH /orgs/{org_id}
     (model default-true read, db read-modify-write preserving siblings,
     permission gating).
  2. The held-memory review queue (list / approve / reject) under
     /orgs/{org_id}/memory, mocked at the graphiti-driver + prisma boundary.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.models import RequestContext

from backend.copilot.graphiti.client import (
    derive_group_id,
    derive_org_group_id,
    derive_team_group_id,
)
from backend.util.exceptions import NotFoundError

ORG_ID = "org-aaa"
OTHER_ORG_ID = "org-bbb"
TEAM_ID = "team-1"
USER_ID = "user-owner-1"
MEM_ID = "edge-uuid-123"

ORG_GROUP = derive_org_group_id(ORG_ID)
TEAM_GROUP = derive_team_group_id(TEAM_ID)


# ──────────────────────────────────────────────────────────────────────────────
# Context helpers
# ──────────────────────────────────────────────────────────────────────────────


def _owner_ctx(org_id=ORG_ID, user_id=USER_ID) -> RequestContext:
    return RequestContext(
        user_id=user_id,
        org_id=org_id,
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


def _member_ctx(org_id=ORG_ID, user_id="user-plain-2") -> RequestContext:
    return RequestContext(
        user_id=user_id,
        org_id=org_id,
        team_id=None,
        is_org_owner=False,
        is_org_admin=False,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


def _make_org(*, settings, id=ORG_ID):
    m = MagicMock()
    m.id = id
    m.name = "Acme"
    m.slug = "acme"
    m.description = None
    m.avatarUrl = None
    m.isPersonal = False
    m.createdAt = datetime(2025, 6, 1, tzinfo=timezone.utc)
    m.deletedAt = None
    m.settings = settings
    return m


def _make_team(*, id=TEAM_ID, name="Engineering", orgId=ORG_ID):
    m = MagicMock()
    m.id = id
    m.name = name
    m.orgId = orgId
    return m


def _driver_with_rows(*row_batches):
    """AsyncMock FalkorDB driver returning one (rows, None, None) per call."""
    driver = AsyncMock()
    driver.execute_query.side_effect = [(rb, None, None) for rb in row_batches]
    driver.close = AsyncMock()
    return driver


# ══════════════════════════════════════════════════════════════════════════════
# 1. holdBuffer settings toggle
# ══════════════════════════════════════════════════════════════════════════════


class TestHoldBufferModel:
    def test_defaults_true_when_settings_absent(self):
        from backend.api.features.orgs.model import OrgResponse

        resp = OrgResponse.from_db(_make_org(settings={}))
        assert resp.memory_hold_buffer is True

    def test_defaults_true_when_settings_is_empty_json_string(self):
        from backend.api.features.orgs.model import OrgResponse

        resp = OrgResponse.from_db(_make_org(settings="{}"))
        assert resp.memory_hold_buffer is True

    def test_reads_false_from_nested_key(self):
        from backend.api.features.orgs.model import OrgResponse

        resp = OrgResponse.from_db(
            _make_org(settings={"memory": {"holdBuffer": False}})
        )
        assert resp.memory_hold_buffer is False

    def test_reads_false_from_json_string_form(self):
        from backend.api.features.orgs.model import OrgResponse

        resp = OrgResponse.from_db(
            _make_org(settings='{"memory": {"holdBuffer": false}}')
        )
        assert resp.memory_hold_buffer is False


class TestHoldBufferUpdateDb:
    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        self.prisma.organizationprofile.update = AsyncMock()
        mocker.patch("backend.api.features.orgs.db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_write_preserves_sibling_settings_keys(self):
        from backend.api.features.orgs.db import update_org
        from backend.api.features.orgs.model import UpdateOrgData

        # Existing settings carry an unrelated key that must survive the write.
        current = _make_org(
            settings={"branding": {"color": "blue"}, "memory": {"other": 1}}
        )
        self.prisma.organization.find_unique = AsyncMock(return_value=current)
        self.prisma.organization.update = AsyncMock()

        await update_org(ORG_ID, UpdateOrgData(memory_hold_buffer=False))

        written = self.prisma.organization.update.call_args[1]["data"]["settings"]
        # prisma.Json wraps the dict; read the payload back.
        payload = written.data
        assert payload["branding"] == {"color": "blue"}
        assert payload["memory"]["other"] == 1
        assert payload["memory"]["holdBuffer"] is False

    @pytest.mark.asyncio
    async def test_write_true_sets_hold_buffer_on(self):
        from backend.api.features.orgs.db import update_org
        from backend.api.features.orgs.model import UpdateOrgData

        self.prisma.organization.find_unique = AsyncMock(
            return_value=_make_org(settings={})
        )
        self.prisma.organization.update = AsyncMock()

        await update_org(ORG_ID, UpdateOrgData(memory_hold_buffer=True))

        payload = self.prisma.organization.update.call_args[1]["data"]["settings"].data
        assert payload["memory"]["holdBuffer"] is True

    @pytest.mark.asyncio
    async def test_none_leaves_settings_untouched(self):
        from backend.api.features.orgs.db import update_org
        from backend.api.features.orgs.model import UpdateOrgData

        self.prisma.organization.find_unique = AsyncMock(
            return_value=_make_org(settings={})
        )
        self.prisma.organization.update = AsyncMock()

        # Only a name change — no memory_hold_buffer.
        await update_org(ORG_ID, UpdateOrgData(name="Renamed"))

        data = self.prisma.organization.update.call_args[1]["data"]
        assert "settings" not in data


class TestHoldBufferPermissionGate:
    """PATCH /orgs/{org_id} carrying memory_hold_buffer is RENAME_ORG-gated."""

    @pytest.fixture(autouse=True)
    def _app(self, mocker):
        from backend.api.features.orgs.routes import router

        self.app = fastapi.FastAPI()
        self.app.include_router(router, prefix="/orgs")
        self.mock_db = mocker.patch("backend.api.features.orgs.routes.org_db")
        self.client = fastapi.testclient.TestClient(self.app)
        yield
        self.app.dependency_overrides.clear()

    def test_plain_member_cannot_toggle(self):
        self.app.dependency_overrides[get_request_context] = lambda: _member_ctx()
        resp = self.client.patch(f"/orgs/{ORG_ID}", json={"memory_hold_buffer": False})
        assert resp.status_code == 403

    def test_owner_can_toggle(self):
        from backend.api.features.orgs.model import OrgResponse

        self.app.dependency_overrides[get_request_context] = lambda: _owner_ctx()
        self.mock_db.update_org = AsyncMock(
            return_value=OrgResponse.from_db(
                _make_org(settings={"memory": {"holdBuffer": False}})
            )
        )
        resp = self.client.patch(f"/orgs/{ORG_ID}", json={"memory_hold_buffer": False})
        assert resp.status_code == 200
        assert resp.json()["memory_hold_buffer"] is False


# ══════════════════════════════════════════════════════════════════════════════
# 2. Held-memory review queue (list / approve / reject)
# ══════════════════════════════════════════════════════════════════════════════


class TestListHeld:
    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.api.features.orgs.memory_db.prisma", self.prisma)

    @pytest.mark.asyncio
    async def test_returns_org_and_team_tentative_with_tier_labels(self, mocker):
        from backend.api.features.orgs import memory_db

        self.prisma.team.find_many = AsyncMock(return_value=[_make_team()])

        org_rows = [
            {
                "uuid": "org-edge",
                "name": "policy",
                "fact": "Company uses Postgres",
                "source_kind": "user_asserted",
                "provenance": "session:s1#msg:4",
                "created_at": "2025-06-02T00:00:00Z",
            }
        ]
        team_rows = [
            {
                "uuid": "team-edge",
                "name": "stack",
                "fact": "Team ships on Fridays",
                "source_kind": "assistant_derived",
                "provenance": "session:s2#msg:9",
                "created_at": "2025-06-03T00:00:00Z",
            }
        ]
        drivers = [_driver_with_rows(org_rows), _driver_with_rows(team_rows)]
        open_driver = mocker.patch(
            "backend.api.features.orgs.memory_db._open_driver",
            side_effect=drivers,
        )

        result = await memory_db.list_held_memories(ORG_ID, limit=50)

        # Only THIS org's shared groups were opened — never a personal group.
        opened = [c.args[0] for c in open_driver.call_args_list]
        assert opened == [ORG_GROUP, TEAM_GROUP]
        assert derive_group_id(USER_ID) not in opened

        by_id = {h.id: h for h in result.items}
        assert by_id["org-edge"].tier == "org"
        assert by_id["org-edge"].team_id is None
        assert by_id["team-edge"].tier == "team"
        assert by_id["team-edge"].team_id == TEAM_ID
        assert by_id["team-edge"].team_name == "Engineering"
        # Newest first across tiers.
        assert result.items[0].id == "team-edge"

    @pytest.mark.asyncio
    async def test_only_scans_this_orgs_teams(self, mocker):
        from backend.api.features.orgs import memory_db

        # find_many is filtered by orgId — other orgs' teams never appear.
        self.prisma.team.find_many = AsyncMock(return_value=[])
        open_driver = mocker.patch(
            "backend.api.features.orgs.memory_db._open_driver",
            side_effect=[_driver_with_rows([])],
        )

        await memory_db.list_held_memories(ORG_ID, limit=50)

        assert self.prisma.team.find_many.call_args[1]["where"] == {"orgId": ORG_ID}
        opened = [c.args[0] for c in open_driver.call_args_list]
        assert opened == [ORG_GROUP]
        assert derive_org_group_id(OTHER_ORG_ID) not in opened


class TestApproveReject:
    @pytest.fixture(autouse=True)
    def _mock_prisma(self, mocker):
        self.prisma = MagicMock()
        mocker.patch("backend.api.features.orgs.memory_db.prisma", self.prisma)
        # No teams → only the org group is in scope, simplifying driver order.
        self.prisma.team.find_many = AsyncMock(return_value=[])

    @pytest.mark.asyncio
    async def test_approve_flips_status_via_ratification_path(self, mocker):
        from backend.api.features.orgs import memory_db

        locate_driver = _driver_with_rows([{"uuid": MEM_ID}])  # found tentative
        promote_driver = _driver_with_rows([{"uuid": MEM_ID}])  # promote touched row
        mocker.patch(
            "backend.api.features.orgs.memory_db._open_driver",
            side_effect=[locate_driver, promote_driver],
        )

        result = await memory_db.approve_held_memory(ORG_ID, MEM_ID, USER_ID)

        assert result.applied is True
        assert result.action == "approve"
        assert result.tier == "org"
        # The promote reused the ratification status-flip Cypher.
        promote_query = promote_driver.execute_query.call_args.args[0]
        assert "SET e.status = 'active'" in promote_query
        assert "ratified_at" in promote_query

    @pytest.mark.asyncio
    async def test_reject_retracts_via_supersede_path(self, mocker):
        from backend.api.features.orgs import memory_db

        locate_driver = _driver_with_rows([{"uuid": MEM_ID}])
        retract_driver = _driver_with_rows([{"uuid": MEM_ID}])
        mocker.patch(
            "backend.api.features.orgs.memory_db._open_driver",
            side_effect=[locate_driver, retract_driver],
        )

        result = await memory_db.reject_held_memory(ORG_ID, MEM_ID, USER_ID)

        assert result.applied is True
        assert result.action == "reject"
        call = retract_driver.execute_query.call_args
        query = call.args[0]
        assert "SET e.expired_at" in query
        assert "e.status = $new_status" in query
        # Tenant-scoped defense-in-depth + audit reason were passed through.
        assert call.kwargs["new_status"] == "superseded"
        assert call.kwargs["group_id"] == ORG_GROUP

    @pytest.mark.asyncio
    async def test_approve_cross_org_memory_id_404s(self, mocker):
        from backend.api.features.orgs import memory_db

        # Edge lives in no group of this org → locate returns nothing.
        mocker.patch(
            "backend.api.features.orgs.memory_db._open_driver",
            side_effect=[_driver_with_rows([])],
        )

        with pytest.raises(NotFoundError):
            await memory_db.approve_held_memory(ORG_ID, "foreign-edge", USER_ID)

    @pytest.mark.asyncio
    async def test_reject_cross_org_memory_id_404s(self, mocker):
        from backend.api.features.orgs import memory_db

        mocker.patch(
            "backend.api.features.orgs.memory_db._open_driver",
            side_effect=[_driver_with_rows([])],
        )

        with pytest.raises(NotFoundError):
            await memory_db.reject_held_memory(ORG_ID, "foreign-edge", USER_ID)


class TestHeldRoutePermissions:
    """All three endpoints are MANAGE_MEMBERS-gated (org admins only)."""

    @pytest.fixture(autouse=True)
    def _app(self, mocker):
        from backend.api.features.orgs.memory_routes import router

        self.app = fastapi.FastAPI()
        self.app.include_router(router, prefix="/orgs/{org_id}/memory")
        self.mock_db = mocker.patch("backend.api.features.orgs.memory_routes.memory_db")
        self.client = fastapi.testclient.TestClient(self.app)
        yield
        self.app.dependency_overrides.clear()

    def test_non_admin_cannot_list(self):
        self.app.dependency_overrides[get_request_context] = lambda: _member_ctx()
        resp = self.client.get(f"/orgs/{ORG_ID}/memory/held")
        assert resp.status_code == 403

    def test_non_admin_cannot_approve(self):
        self.app.dependency_overrides[get_request_context] = lambda: _member_ctx()
        resp = self.client.post(f"/orgs/{ORG_ID}/memory/held/{MEM_ID}/approve")
        assert resp.status_code == 403

    def test_non_admin_cannot_reject(self):
        self.app.dependency_overrides[get_request_context] = lambda: _member_ctx()
        resp = self.client.post(f"/orgs/{ORG_ID}/memory/held/{MEM_ID}/reject")
        assert resp.status_code == 403

    def test_admin_of_other_org_cannot_reach_this_orgs_queue(self):
        # Owner, but of a DIFFERENT org than the path — _verify_org_path blocks.
        self.app.dependency_overrides[get_request_context] = lambda: _owner_ctx(
            org_id=OTHER_ORG_ID
        )
        resp = self.client.get(f"/orgs/{ORG_ID}/memory/held")
        assert resp.status_code == 403

    def test_admin_can_list(self):
        from backend.api.features.orgs.memory_model import HeldMemoryListResponse

        self.app.dependency_overrides[get_request_context] = lambda: _owner_ctx()
        self.mock_db.list_held_memories = AsyncMock(
            return_value=HeldMemoryListResponse(org_id=ORG_ID, items=[])
        )
        resp = self.client.get(f"/orgs/{ORG_ID}/memory/held")
        assert resp.status_code == 200
        assert resp.json() == {"org_id": ORG_ID, "items": []}
