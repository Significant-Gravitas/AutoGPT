"""
Exhaustive tests for org-level and workspace-level permission checks.

Every OrgAction x role combination and every TeamAction x role
combination is covered.  These are pure-function tests -- no mocking,
no database, no I/O.
"""

import pytest

from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import (
    OrgAction,
    TeamAction,
    check_org_permission,
    check_team_permission,
)

# ---------------------------------------------------------------------------
# Helpers to build RequestContext fixtures for each role
# ---------------------------------------------------------------------------


def _make_ctx(
    *,
    is_org_owner: bool = False,
    is_org_admin: bool = False,
    is_org_billing_manager: bool = False,
    is_team_admin: bool = False,
    is_team_billing_manager: bool = False,
    seat_status: str = "ACTIVE",
    team_id: str | None = None,
) -> RequestContext:
    return RequestContext(
        user_id="user-1",
        org_id="org-1",
        team_id=team_id,
        is_org_owner=is_org_owner,
        is_org_admin=is_org_admin,
        is_org_billing_manager=is_org_billing_manager,
        is_team_admin=is_team_admin,
        is_team_billing_manager=is_team_billing_manager,
        seat_status=seat_status,
    )


# Convenience contexts for org-level roles
ORG_OWNER = _make_ctx(is_org_owner=True)
ORG_ADMIN = _make_ctx(is_org_admin=True)
ORG_BILLING_MANAGER = _make_ctx(is_org_billing_manager=True)
ORG_MEMBER = _make_ctx()  # ACTIVE seat, no special role flags


# Convenience contexts for workspace-level roles (team_id is set)
TEAM_ADMIN = _make_ctx(team_id="ws-1", is_team_admin=True)
TEAM_BILLING_MGR = _make_ctx(team_id="ws-1", is_team_billing_manager=True)
TEAM_MEMBER = _make_ctx(team_id="ws-1")  # regular workspace member


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

_TEAM_EXPECTED: dict[TeamAction, dict[str, bool]] = {
    TeamAction.MANAGE_MEMBERS: {
        "team_admin": True,
        "team_billing_manager": False,
        "team_member": False,
    },
    TeamAction.MANAGE_SETTINGS: {
        "team_admin": True,
        "team_billing_manager": False,
        "team_member": False,
    },
    TeamAction.MANAGE_CREDENTIALS: {
        "team_admin": True,
        "team_billing_manager": False,
        "team_member": False,
    },
    TeamAction.VIEW_SPEND: {
        "team_admin": True,
        "team_billing_manager": True,
        "team_member": False,
    },
    TeamAction.CREATE_AGENTS: {
        "team_admin": True,
        "team_billing_manager": False,
        "team_member": True,
    },
    TeamAction.USE_CREDENTIALS: {
        "team_admin": True,
        "team_billing_manager": False,
        "team_member": True,
    },
    TeamAction.VIEW_EXECUTIONS: {
        "team_admin": True,
        "team_billing_manager": False,
        "team_member": True,
    },
    TeamAction.DELETE_AGENTS: {
        "team_admin": True,
        "team_billing_manager": False,
        "team_member": False,
    },
}

_TEAM_ROLE_CTX = {
    "team_admin": TEAM_ADMIN,
    "team_billing_manager": TEAM_BILLING_MGR,
    "team_member": TEAM_MEMBER,
}


class TestTeamPermissions:
    """Exhaustive workspace action x role matrix."""

    @pytest.mark.parametrize(
        "action",
        list(TeamAction),
        ids=[a.value for a in TeamAction],
    )
    @pytest.mark.parametrize(
        "role",
        ["team_admin", "team_billing_manager", "team_member"],
    )
    def test_team_permission_matrix(self, action: TeamAction, role: str):
        ctx = _TEAM_ROLE_CTX[role]
        expected = _TEAM_EXPECTED[action][role]
        result = check_team_permission(ctx, action)
        assert result is expected, (
            f"TeamAction.{action.value} for role={role}: "
            f"expected {expected}, got {result}"
        )

    def test_no_team_context_denies_all(self):
        """Without a team_id, all workspace actions are denied."""
        ctx = _make_ctx(is_team_admin=True)  # no team_id
        for action in TeamAction:
            assert check_team_permission(ctx, action) is False

    def test_team_billing_manager_is_not_team_member(self):
        """A workspace billing manager should NOT get team_member permissions."""
        ctx = TEAM_BILLING_MGR
        assert check_team_permission(ctx, TeamAction.CREATE_AGENTS) is False
        assert check_team_permission(ctx, TeamAction.VIEW_SPEND) is True
