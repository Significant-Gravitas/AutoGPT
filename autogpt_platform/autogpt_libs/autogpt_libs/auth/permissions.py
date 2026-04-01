"""
Role-based permission checks for org-level and workspace-level actions.

Permission maps are pure data; check functions are pure functions operating
on a RequestContext -- no I/O, no database access.
"""

from enum import Enum

from .models import RequestContext


class OrgAction(str, Enum):
    DELETE_ORG = "DELETE_ORG"
    RENAME_ORG = "RENAME_ORG"
    MANAGE_MEMBERS = "MANAGE_MEMBERS"
    MANAGE_WORKSPACES = "MANAGE_WORKSPACES"
    CREATE_WORKSPACES = "CREATE_WORKSPACES"
    MANAGE_BILLING = "MANAGE_BILLING"
    PUBLISH_TO_STORE = "PUBLISH_TO_STORE"
    TRANSFER_RESOURCES = "TRANSFER_RESOURCES"
    VIEW_ORG = "VIEW_ORG"
    CREATE_RESOURCES = "CREATE_RESOURCES"
    SHARE_RESOURCES = "SHARE_RESOURCES"


class WorkspaceAction(str, Enum):
    MANAGE_MEMBERS = "MANAGE_MEMBERS"
    MANAGE_SETTINGS = "MANAGE_SETTINGS"
    MANAGE_CREDENTIALS = "MANAGE_CREDENTIALS"
    VIEW_SPEND = "VIEW_SPEND"
    CREATE_AGENTS = "CREATE_AGENTS"
    USE_CREDENTIALS = "USE_CREDENTIALS"
    VIEW_EXECUTIONS = "VIEW_EXECUTIONS"
    DELETE_AGENTS = "DELETE_AGENTS"


# Org permission map: action -> set of roles that are allowed.
# Roles checked via RequestContext boolean flags.
# "member" means any active org member (no special role flag required).
_ORG_PERMISSIONS: dict[OrgAction, set[str]] = {
    OrgAction.DELETE_ORG: {"owner"},
    OrgAction.RENAME_ORG: {"owner", "admin"},
    OrgAction.MANAGE_MEMBERS: {"owner", "admin"},
    OrgAction.MANAGE_WORKSPACES: {"owner", "admin"},
    OrgAction.CREATE_WORKSPACES: {"owner", "admin", "billing_manager"},
    OrgAction.MANAGE_BILLING: {"owner", "billing_manager"},
    OrgAction.PUBLISH_TO_STORE: {"owner", "admin", "member"},
    OrgAction.TRANSFER_RESOURCES: {"owner", "admin"},
    OrgAction.VIEW_ORG: {"owner", "admin", "billing_manager", "member"},
    OrgAction.CREATE_RESOURCES: {"owner", "admin", "member"},
    OrgAction.SHARE_RESOURCES: {"owner", "admin", "member"},
}

# Workspace permission map: action -> set of roles that are allowed.
# "ws_member" means any workspace member with no special workspace role.
_WORKSPACE_PERMISSIONS: dict[WorkspaceAction, set[str]] = {
    WorkspaceAction.MANAGE_MEMBERS: {"ws_admin"},
    WorkspaceAction.MANAGE_SETTINGS: {"ws_admin"},
    WorkspaceAction.MANAGE_CREDENTIALS: {"ws_admin"},
    WorkspaceAction.VIEW_SPEND: {"ws_admin", "ws_billing_manager"},
    WorkspaceAction.CREATE_AGENTS: {"ws_admin", "ws_member"},
    WorkspaceAction.USE_CREDENTIALS: {"ws_admin", "ws_member"},
    WorkspaceAction.VIEW_EXECUTIONS: {"ws_admin", "ws_member"},
    WorkspaceAction.DELETE_AGENTS: {"ws_admin"},
}


def _get_org_roles(ctx: RequestContext) -> set[str]:
    """Derive the set of org-level role tags from a RequestContext."""
    roles: set[str] = set()
    if ctx.is_org_owner:
        roles.add("owner")
    if ctx.is_org_admin:
        roles.add("admin")
    if ctx.is_org_billing_manager:
        roles.add("billing_manager")
    # A plain member (no owner/admin/billing_manager flags) gets "member".
    # Owner and admin also get "member" since they can do everything a member can.
    # Billing managers do NOT get "member" — they only get finance-related actions.
    if ctx.is_org_owner or ctx.is_org_admin:
        roles.add("member")
    elif not ctx.is_org_billing_manager:
        # Plain member with no special role flags
        roles.add("member")
    return roles


def _get_workspace_roles(ctx: RequestContext) -> set[str]:
    """Derive the set of workspace-level role tags from a RequestContext."""
    roles: set[str] = set()
    if ctx.is_workspace_admin:
        roles.add("ws_admin")
    if ctx.is_workspace_billing_manager:
        roles.add("ws_billing_manager")
    # Regular workspace members (not admin, not billing_manager) get ws_member.
    # WS admins also get ws_member (they can do everything a member can).
    if ctx.workspace_id is not None:
        if ctx.is_workspace_admin or (not ctx.is_workspace_billing_manager):
            roles.add("ws_member")
    return roles


def check_org_permission(ctx: RequestContext, action: OrgAction) -> bool:
    """Return True if the RequestContext grants the given org-level action."""
    allowed_roles = _ORG_PERMISSIONS.get(action, set())
    return bool(_get_org_roles(ctx) & allowed_roles)


def check_workspace_permission(ctx: RequestContext, action: WorkspaceAction) -> bool:
    """Return True if the RequestContext grants the given workspace-level action."""
    if ctx.workspace_id is None:
        return False
    allowed_roles = _WORKSPACE_PERMISSIONS.get(action, set())
    return bool(_get_workspace_roles(ctx) & allowed_roles)
