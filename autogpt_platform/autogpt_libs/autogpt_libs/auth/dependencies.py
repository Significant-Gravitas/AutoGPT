"""
FastAPI dependency functions for JWT-based authentication and authorization.

These are the high-level dependency functions used in route definitions.
"""

import logging

import fastapi
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .jwt_utils import get_jwt_payload, verify_user
from .models import RequestContext, User
from .permissions import (
    OrgAction,
    WorkspaceAction,
    check_org_permission,
    check_workspace_permission,
)

optional_bearer = HTTPBearer(auto_error=False)

# Header name for admin impersonation
IMPERSONATION_HEADER_NAME = "X-Act-As-User-Id"

logger = logging.getLogger(__name__)


def get_optional_user_id(
    credentials: HTTPAuthorizationCredentials | None = fastapi.Security(
        optional_bearer
    ),
) -> str | None:
    """
    Attempts to extract the user ID ("sub" claim) from a Bearer JWT if provided.

    This dependency allows for both authenticated and anonymous access. If a valid bearer token is
    supplied, it parses the JWT and extracts the user ID. If the token is missing or invalid, it returns None,
    treating the request as anonymous.

    Args:
        credentials: Optional HTTPAuthorizationCredentials object from FastAPI Security dependency.

    Returns:
        The user ID (str) extracted from the JWT "sub" claim, or None if no valid token is present.
    """
    if not credentials:
        return None

    try:
        # Parse JWT token to get user ID
        from autogpt_libs.auth.jwt_utils import parse_jwt_token

        payload = parse_jwt_token(credentials.credentials)
        return payload.get("sub")
    except Exception as e:
        logger.debug(f"Auth token validation failed (anonymous access): {e}")
        return None


async def requires_user(jwt_payload: dict = fastapi.Security(get_jwt_payload)) -> User:
    """
    FastAPI dependency that requires a valid authenticated user.

    Raises:
        HTTPException: 401 for authentication failures
    """
    return verify_user(jwt_payload, admin_only=False)


async def requires_admin_user(
    jwt_payload: dict = fastapi.Security(get_jwt_payload),
) -> User:
    """
    FastAPI dependency that requires a valid admin user.

    Raises:
        HTTPException: 401 for authentication failures, 403 for insufficient permissions
    """
    return verify_user(jwt_payload, admin_only=True)


async def get_user_id(
    request: fastapi.Request, jwt_payload: dict = fastapi.Security(get_jwt_payload)
) -> str:
    """
    FastAPI dependency that returns the ID of the authenticated user.

    Supports admin impersonation via X-Act-As-User-Id header:
    - If the header is present and user is admin, returns the impersonated user ID
    - Otherwise returns the authenticated user's own ID
    - Logs all impersonation actions for audit trail

    Raises:
        HTTPException: 401 for authentication failures or missing user ID
        HTTPException: 403 if non-admin tries to use impersonation
    """
    # Get the authenticated user's ID from JWT
    user_id = jwt_payload.get("sub")
    if not user_id:
        raise fastapi.HTTPException(
            status_code=401, detail="User ID not found in token"
        )

    # Check for admin impersonation header
    impersonate_header = request.headers.get(IMPERSONATION_HEADER_NAME, "").strip()
    if impersonate_header:
        # Verify the authenticated user is an admin
        authenticated_user = verify_user(jwt_payload, admin_only=False)
        if authenticated_user.role != "admin":
            raise fastapi.HTTPException(
                status_code=403, detail="Only admin users can impersonate other users"
            )

        # Log the impersonation for audit trail
        logger.info(
            f"Admin impersonation: {authenticated_user.user_id} ({authenticated_user.email}) "
            f"acting as user {impersonate_header} for requesting {request.method} {request.url}"
        )

        return impersonate_header

    return user_id


# ---------------------------------------------------------------------------
# Org / Workspace context resolution
# ---------------------------------------------------------------------------

ORG_HEADER_NAME = "X-Org-Id"
WORKSPACE_HEADER_NAME = "X-Workspace-Id"


async def get_request_context(
    request: fastapi.Request,
    jwt_payload: dict = fastapi.Security(get_jwt_payload),
) -> RequestContext:
    """
    FastAPI dependency that resolves the full org/workspace context for a request.

    Resolution order:
      1. Extract user_id from JWT (supports admin impersonation via X-Act-As-User-Id).
      2. Read X-Org-Id header; fall back to the user's personal org; fail if none.
      3. Validate that the user has an ACTIVE OrgMember row for that org.
      4. Read X-Workspace-Id header (optional). If set, validate that the
         workspace belongs to the org AND the user has an OrgWorkspaceMember
         row. On failure, silently fall back to None (org-home).
      5. Populate all role flags and return a RequestContext.
    """
    from backend.data.db import prisma  # deferred -- only needed at runtime

    # --- 1. user_id (reuse existing impersonation logic) ----------------------
    user_id = jwt_payload.get("sub")
    if not user_id:
        raise fastapi.HTTPException(
            status_code=401, detail="User ID not found in token"
        )

    impersonate_header = request.headers.get(IMPERSONATION_HEADER_NAME, "").strip()
    if impersonate_header:
        authenticated_user = verify_user(jwt_payload, admin_only=False)
        if authenticated_user.role != "admin":
            raise fastapi.HTTPException(
                status_code=403,
                detail="Only admin users can impersonate other users",
            )
        logger.info(
            f"Admin impersonation: {authenticated_user.user_id} ({authenticated_user.email}) "
            f"acting as user {impersonate_header} for requesting {request.method} {request.url}"
        )
        user_id = impersonate_header

    # --- 2. org_id ------------------------------------------------------------
    org_id = request.headers.get(ORG_HEADER_NAME, "").strip() or None

    if org_id is None:
        # Fall back to the user's personal org (an org where the user is the
        # sole owner, typically created at sign-up).
        personal_org = await prisma.orgmember.find_first(
            where={
                "userId": user_id,
                "isOwner": True,
                "Org": {"isPersonal": True},
            },
            order={"createdAt": "asc"},
        )
        if personal_org is None:
            raise fastapi.HTTPException(
                status_code=400,
                detail="No org specified and user has no personal org",
            )
        org_id = personal_org.orgId

    # --- 3. validate OrgMember ------------------------------------------------
    org_member = await prisma.orgmember.find_unique(
        where={
            "orgId_userId": {"orgId": org_id, "userId": user_id},
        },
    )
    if org_member is None or org_member.status != "ACTIVE":
        raise fastapi.HTTPException(
            status_code=403,
            detail="User is not an active member of this organization",
        )

    is_org_owner = org_member.isOwner
    is_org_admin = org_member.isAdmin
    is_org_billing_manager = org_member.isBillingManager
    seat_status = "ACTIVE"  # validated above; seat assignment checked separately

    # --- 4. workspace_id (optional) -------------------------------------------
    workspace_id: str | None = (
        request.headers.get(WORKSPACE_HEADER_NAME, "").strip() or None
    )
    is_workspace_admin = False
    is_workspace_billing_manager = False

    if workspace_id is not None:
        # Validate workspace belongs to org AND user has a membership row
        ws_member = await prisma.orgworkspacemember.find_unique(
            where={
                "workspaceId_userId": {
                    "workspaceId": workspace_id,
                    "userId": user_id,
                },
            },
            include={"Workspace": True},
        )
        if (
            ws_member is None
            or ws_member.Workspace is None
            or ws_member.Workspace.orgId != org_id
        ):
            logger.debug(
                "Workspace %s not valid for user %s in org %s; falling back to org-home",
                workspace_id,
                user_id,
                org_id,
            )
            workspace_id = None
        else:
            is_workspace_admin = ws_member.isAdmin
            is_workspace_billing_manager = ws_member.isBillingManager

    # --- 5. build context -----------------------------------------------------
    return RequestContext(
        user_id=user_id,
        org_id=org_id,
        workspace_id=workspace_id,
        is_org_owner=is_org_owner,
        is_org_admin=is_org_admin,
        is_org_billing_manager=is_org_billing_manager,
        is_workspace_admin=is_workspace_admin,
        is_workspace_billing_manager=is_workspace_billing_manager,
        seat_status=seat_status,
    )


def requires_org_permission(
    *actions: OrgAction,
):
    """
    Factory that returns a FastAPI dependency enforcing one or more org-level
    permissions.  The request is allowed if the user holds **all** listed actions.

    Usage::

        @router.delete("/org/{org_id}")
        async def delete_org(
            ctx: RequestContext = Security(requires_org_permission(OrgAction.DELETE_ORG)),
        ):
            ...
    """

    async def _dependency(
        ctx: RequestContext = fastapi.Security(get_request_context),
    ) -> RequestContext:
        for action in actions:
            if not check_org_permission(ctx, action):
                raise fastapi.HTTPException(
                    status_code=403,
                    detail=f"Missing org permission: {action.value}",
                )
        return ctx

    return _dependency


def requires_workspace_permission(
    *actions: WorkspaceAction,
):
    """
    Factory that returns a FastAPI dependency enforcing one or more
    workspace-level permissions.  The user must be in a workspace context
    (workspace_id is set) and hold **all** listed actions.

    Usage::

        @router.post("/workspace/{ws_id}/agents")
        async def create_agent(
            ctx: RequestContext = Security(
                requires_workspace_permission(WorkspaceAction.CREATE_AGENTS)
            ),
        ):
            ...
    """

    async def _dependency(
        ctx: RequestContext = fastapi.Security(get_request_context),
    ) -> RequestContext:
        if ctx.workspace_id is None:
            raise fastapi.HTTPException(
                status_code=400,
                detail="Workspace context required for this action",
            )
        for action in actions:
            if not check_workspace_permission(ctx, action):
                raise fastapi.HTTPException(
                    status_code=403,
                    detail=f"Missing workspace permission: {action.value}",
                )
        return ctx

    return _dependency
