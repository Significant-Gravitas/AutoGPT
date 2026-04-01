"""
Exhaustive tests for org-level and workspace-level permission checks.

Every OrgAction x role combination and every WorkspaceAction x role
combination is covered.  These are pure-function tests -- no mocking,
no database, no I/O.
"""

import pytest

from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import (
    OrgAction,
    WorkspaceAction,
    check_org_permission,
    check_workspace_permission,
)

# ---------------------------------------------------------------------------
# Helpers to build RequestContext fixtures for each role
# ---------------------------------------------------------------------------


def _make_ctx(
    *,
    is_org_owner: bool = False,
    is_org_admin: bool = False,
    is_org_billing_manager: bool = False,
    is_workspace_admin: bool = False,
    is_workspace_billing_manager: bool = False,
    seat_status: str = "ACTIVE",
    workspace_id: str | None = None,
) -> RequestContext:
    return RequestContext(
        user_id="user-1",
        org_id="org-1",
        workspace_id=workspace_id,
        is_org_owner=is_org_owner,
        is_org_admin=is_org_admin,
        is_org_billing_manager=is_org_billing_manager,
        is_workspace_admin=is_workspace_admin,
        is_workspace_billing_manager=is_workspace_billing_manager,
        seat_status=seat_status,
    )


# Convenience contexts for org-level roles
ORG_OWNER = _make_ctx(is_org_owner=True)
ORG_ADMIN = _make_ctx(is_org_admin=True)
ORG_BILLING_MANAGER = _make_ctx(is_org_billing_manager=True)
ORG_MEMBER = _make_ctx()  # ACTIVE seat, no special role flags


# Convenience contexts for workspace-level roles (workspace_id is set)
WS_ADMIN = _make_ctx(workspace_id="ws-1", is_workspace_admin=True)
WS_BILLING_MANAGER = _make_ctx(workspace_id="ws-1", is_workspace_billing_manager=True)
WS_MEMBER = _make_ctx(workspace_id="ws-1")  # regular workspace member


# ---------------------------------------------------------------------------
# Org permission matrix
# ---------------------------------------------------------------------------

# Expected outcomes per (action, role).  True = allowed.
_ORG_EXPECTED: dict[OrgAction, dict[str, bool]] = {
    OrgAction.DELETE_ORG: {
        "owner": True,
        "admin": False,
        "billing_manager": False,
        "member": False,
    },
    OrgAction.RENAME_ORG: {
        "owner": True,
        "admin": True,
        "billing_manager": False,
        "member": False,
    },
    OrgAction.MANAGE_MEMBERS: {
        "owner": True,
        "admin": True,
        "billing_manager": False,
        "member": False,
    },
    OrgAction.MANAGE_WORKSPACES: {
        "owner": True,
        "admin": True,
        "billing_manager": False,
        "member": False,
    },
    OrgAction.CREATE_WORKSPACES: {
        "owner": True,
        "admin": True,
        "billing_manager": True,
        "member": False,
    },
    OrgAction.MANAGE_BILLING: {
        "owner": True,
        "admin": False,
        "billing_manager": True,
        "member": False,
    },
    OrgAction.PUBLISH_TO_STORE: {
        "owner": True,
        "admin": True,
        "billing_manager": False,
        "member": True,
    },
    OrgAction.TRANSFER_RESOURCES: {
        "owner": True,
        "admin": True,
        "billing_manager": False,
        "member": False,
    },
    OrgAction.VIEW_ORG: {
        "owner": True,
        "admin": True,
        "billing_manager": True,
        "member": True,
    },
    OrgAction.CREATE_RESOURCES: {
        "owner": True,
        "admin": True,
        "billing_manager": False,
        "member": True,
    },
    OrgAction.SHARE_RESOURCES: {
        "owner": True,
        "admin": True,
        "billing_manager": False,
        "member": True,
    },
}

_ORG_ROLE_CTX = {
    "owner": ORG_OWNER,
    "admin": ORG_ADMIN,
    "billing_manager": ORG_BILLING_MANAGER,
    "member": ORG_MEMBER,
}


class TestOrgPermissions:
    """Exhaustive org action x role matrix."""

    @pytest.mark.parametrize(
        "action",
        list(OrgAction),
        ids=[a.value for a in OrgAction],
    )
    @pytest.mark.parametrize(
        "role",
        ["owner", "admin", "billing_manager", "member"],
    )
    def test_org_permission_matrix(self, action: OrgAction, role: str):
        ctx = _ORG_ROLE_CTX[role]
        expected = _ORG_EXPECTED[action][role]
        result = check_org_permission(ctx, action)
        assert result is expected, (
            f"OrgAction.{action.value} for role={role}: "
            f"expected {expected}, got {result}"
        )

    def test_member_role_always_present_regardless_of_seat(self):
        """The 'member' role is implicit for all org members.
        Seat-gating is enforced at the endpoint level, not in permission checks."""
        ctx = _make_ctx(seat_status="INACTIVE")
        # VIEW_ORG is allowed for members regardless of seat status
        assert check_org_permission(ctx, OrgAction.VIEW_ORG) is True

    def test_owner_with_inactive_seat_retains_all_owner_permissions(self):
        """Owner flag is independent of seat_status."""
        ctx = _make_ctx(is_org_owner=True, seat_status="INACTIVE")
        assert check_org_permission(ctx, OrgAction.DELETE_ORG) is True
        assert check_org_permission(ctx, OrgAction.PUBLISH_TO_STORE) is True


# ---------------------------------------------------------------------------
# Workspace permission matrix
# ---------------------------------------------------------------------------

_WS_EXPECTED: dict[WorkspaceAction, dict[str, bool]] = {
    WorkspaceAction.MANAGE_MEMBERS: {
        "ws_admin": True,
        "ws_billing_manager": False,
        "ws_member": False,
    },
    WorkspaceAction.MANAGE_SETTINGS: {
        "ws_admin": True,
        "ws_billing_manager": False,
        "ws_member": False,
    },
    WorkspaceAction.MANAGE_CREDENTIALS: {
        "ws_admin": True,
        "ws_billing_manager": False,
        "ws_member": False,
    },
    WorkspaceAction.VIEW_SPEND: {
        "ws_admin": True,
        "ws_billing_manager": True,
        "ws_member": False,
    },
    WorkspaceAction.CREATE_AGENTS: {
        "ws_admin": True,
        "ws_billing_manager": False,
        "ws_member": True,
    },
    WorkspaceAction.USE_CREDENTIALS: {
        "ws_admin": True,
        "ws_billing_manager": False,
        "ws_member": True,
    },
    WorkspaceAction.VIEW_EXECUTIONS: {
        "ws_admin": True,
        "ws_billing_manager": False,
        "ws_member": True,
    },
    WorkspaceAction.DELETE_AGENTS: {
        "ws_admin": True,
        "ws_billing_manager": False,
        "ws_member": False,
    },
}

_WS_ROLE_CTX = {
    "ws_admin": WS_ADMIN,
    "ws_billing_manager": WS_BILLING_MANAGER,
    "ws_member": WS_MEMBER,
}


class TestWorkspacePermissions:
    """Exhaustive workspace action x role matrix."""

    @pytest.mark.parametrize(
        "action",
        list(WorkspaceAction),
        ids=[a.value for a in WorkspaceAction],
    )
    @pytest.mark.parametrize(
        "role",
        ["ws_admin", "ws_billing_manager", "ws_member"],
    )
    def test_workspace_permission_matrix(self, action: WorkspaceAction, role: str):
        ctx = _WS_ROLE_CTX[role]
        expected = _WS_EXPECTED[action][role]
        result = check_workspace_permission(ctx, action)
        assert result is expected, (
            f"WorkspaceAction.{action.value} for role={role}: "
            f"expected {expected}, got {result}"
        )

    def test_no_workspace_context_denies_all(self):
        """Without a workspace_id, all workspace actions are denied."""
        ctx = _make_ctx(is_workspace_admin=True)  # no workspace_id
        for action in WorkspaceAction:
            assert check_workspace_permission(ctx, action) is False

    def test_ws_billing_manager_is_not_ws_member(self):
        """A workspace billing manager should NOT get ws_member permissions."""
        ctx = WS_BILLING_MANAGER
        assert check_workspace_permission(ctx, WorkspaceAction.CREATE_AGENTS) is False
        assert check_workspace_permission(ctx, WorkspaceAction.VIEW_SPEND) is True
