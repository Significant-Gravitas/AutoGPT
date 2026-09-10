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
    TeamAction,
    check_org_permission,
    check_team_permission,
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
TEAM_HEADER_NAME = "X-Team-Id"


async def _ensure_platform_user(user_id: str, jwt_payload: dict) -> None:
    """Provision the platform ``User`` row for a valid token that has none.

    The auth provider and the platform keep separate user tables, bridged only
    by the client calling ``POST /api/v1/auth/user`` after sign-in. Better Auth
    issues a session the moment the auth identity is created, so a request can
    legitimately arrive before that call has run — or when it never ran,
    because the OAuth flow lost the redirect that would have made it.

    The org bootstrap below cannot create an org for a user it cannot find, and
    nothing the client does later re-provisions, so such an account 400s on
    every org-scoped endpoint forever rather than transiently. Provisioning
    here from our own verified claims makes that self-healing.

    Best-effort once provisioning is attempted: a failure leaves the caller to
    run the org bootstrap and surface the same 400 as before, so this can only
    improve the outcome. (The probe below is deliberately outside that guard —
    if the database is unreachable the request has no context to resolve
    anyway, and the very next query would raise regardless.)
    """
    from backend.data.db import prisma  # deferred -- only needed at runtime

    # Only ever provision from claims that describe the user being resolved.
    # Under admin impersonation ``user_id`` is the target while the JWT
    # describes the admin, and provisioning from it would create the account
    # under the admin's email.
    if user_id != jwt_payload.get("sub"):
        logger.debug("Not provisioning %s from an impersonator's claims", user_id)
        return
    if not jwt_payload.get("email"):
        # Nothing to provision with, so this account stays broken. Say so —
        # a silent return here is the exact failure mode this function exists
        # to end: bricked and invisible.
        logger.warning(f"Cannot provision user {user_id}: token carries no email claim")
        return

    if await prisma.user.find_unique(where={"id": user_id}) is not None:
        return

    from backend.data.user import get_or_create_user_with_status  # deferred

    try:
        # The uncached entry point: `get_or_create_user` memoizes for 5min
        # across the whole process, so a row deleted (or healed) out from
        # under a live cache entry would be masked for the rest of its TTL.
        result = await get_or_create_user_with_status(jwt_payload)
    except Exception:
        # A first page load fans out ~20 requests that all miss the probe
        # above, so all but one lose the create race and surface it as a
        # DatabaseError. Report only a failure that actually left the user
        # unprovisioned — otherwise one successful heal buries its own signal
        # under ~19 tracebacks claiming it failed.
        if await prisma.user.find_unique(where={"id": user_id}) is not None:
            # The row exists, so the account is not stranded — but this is not
            # necessarily a clean lost race: we may have created the row and
            # then failed inside ensure_personal_org. Keep the traceback at
            # WARNING, which LoggingIntegration attaches as a breadcrumb
            # without raising a Sentry event, so the detail survives without
            # the ~19-per-heal noise that reporting it as an error caused.
            logger.warning(
                f"Provisioning for user {user_id} raised, but the platform row "
                "exists; the caller's org bootstrap will finish or report",
                exc_info=True,
            )
            return
        logger.error(f"On-demand provisioning failed for user {user_id}", exc_info=True)
        return

    if not result.was_created:
        # Returning without raising is not proof we created anything: a
        # concurrent request can land its row between our probe and the
        # get-or-create's own lookup, which then simply reads it back. That
        # request reports the breach; this one would only double-count it.
        return

    # ERROR, not WARNING: LoggingIntegration reports this to Sentry, and a
    # token with no platform user is an invariant breach worth seeing. Gated
    # on `was_created` so it fires exactly once for the account that was
    # missing rather than once per request in the fan-out above.
    logger.error(
        f"Provisioned a missing platform User row for {user_id} on first touch"
    )


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
      4. Read X-Team-Id header (optional). If set, validate that the
         workspace belongs to the org AND the user has an TeamMember
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
                "Org": {"isPersonal": True, "deletedAt": None},
            },
            order={"createdAt": "asc"},
        )
        if personal_org is not None:
            org_id = personal_org.orgId
        else:
            # Self-heal: bootstrap the personal org. Sign-up and the startup
            # backfill cover normal accounts, but users created outside
            # get_or_create_user (e.g. seeded test accounts, direct DB
            # inserts) would otherwise 400 on every request forever.
            from backend.api.features.orgs.db import get_user_default_team  # deferred

            # Two things can be missing here, and only one of them used to be
            # recoverable: the personal org, or the platform User row the org
            # would hang off. Provision the user first so the bootstrap below
            # has something to work with.
            await _ensure_platform_user(user_id, jwt_payload)

            org_id, _ = await get_user_default_team(user_id)
            if org_id is None:
                # Deliberately WARNING, not ERROR. This state is permanent
                # until something heals it, and the client re-enters it on
                # every org-scoped request, so a Sentry event here fires at a
                # cadence the browser controls — thousands per broken account.
                # The bounded signals carry this instead: _ensure_platform_user
                # reports the failed provision once per request with a
                # traceback, and reports a successful heal once per account.
                logger.warning(
                    f"User {user_id} has no personal org and bootstrap "
                    "failed — account in inconsistent state"
                )
                raise fastapi.HTTPException(
                    status_code=400,
                    detail=(
                        "No organization context available. Your account may "
                        "be in an inconsistent state — please contact support."
                    ),
                )

    # --- 3. validate OrgMember ------------------------------------------------
    org_member = await prisma.orgmember.find_unique(
        where={
            "orgId_userId": {"orgId": org_id, "userId": user_id},
        },
        include={"Org": True},
    )
    if org_member is None or org_member.status != "ACTIVE":
        raise fastapi.HTTPException(
            status_code=403,
            detail="User is not an active member of this organization",
        )
    # A soft-deleted org must not remain usable as request context —
    # delete_org keeps the row (deletedAt set) but memberships with it
    # no longer grant access.
    if org_member.Org is not None and org_member.Org.deletedAt is not None:
        raise fastapi.HTTPException(
            status_code=403,
            detail="This organization has been deleted",
        )

    is_org_owner = org_member.isOwner
    is_org_admin = org_member.isAdmin
    is_org_billing_manager = org_member.isBillingManager
    seat_status = "ACTIVE"  # validated above; seat assignment checked separately

    # --- 4. team_id (optional) -------------------------------------------
    team_id: str | None = request.headers.get(TEAM_HEADER_NAME, "").strip() or None
    is_team_admin = False
    is_team_billing_manager = False

    if team_id is not None:
        # Validate workspace belongs to org AND user has a membership row
        ws_member = await prisma.teammember.find_unique(
            where={
                "teamId_userId": {
                    "teamId": team_id,
                    "userId": user_id,
                },
            },
            include={"Team": True},
        )
        if (
            ws_member is None
            or ws_member.status != "ACTIVE"
            or ws_member.Team is None
            or ws_member.Team.orgId != org_id
        ):
            logger.debug(
                "Workspace %s not valid for user %s in org %s; falling back to org-home",
                team_id,
                user_id,
                org_id,
            )
            team_id = None
        else:
            is_team_admin = ws_member.isAdmin
            is_team_billing_manager = ws_member.isBillingManager

    # --- 5. build context -----------------------------------------------------
    return RequestContext(
        user_id=user_id,
        org_id=org_id,
        team_id=team_id,
        is_org_owner=is_org_owner,
        is_org_admin=is_org_admin,
        is_org_billing_manager=is_org_billing_manager,
        is_team_admin=is_team_admin,
        is_team_billing_manager=is_team_billing_manager,
        seat_status=seat_status,
    )


def requires_org_permission(
    *actions: OrgAction,
):
    """Factory returning a FastAPI dependency that enforces org-level permissions.

    The request is allowed only if the user holds **all** listed actions.

    Example::

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


def requires_team_permission(
    *actions: TeamAction,
):
    """Factory returning a FastAPI dependency that enforces workspace-level permissions.

    The user must be in a workspace context (team_id is set) and
    hold **all** listed actions.

    Example::

        @router.post("/workspace/{ws_id}/agents")
        async def create_agent(
            ctx: RequestContext = Security(
                requires_team_permission(TeamAction.CREATE_AGENTS)
            ),
        ):
            ...
    """

    async def _dependency(
        ctx: RequestContext = fastapi.Security(get_request_context),
    ) -> RequestContext:
        if ctx.team_id is None:
            raise fastapi.HTTPException(
                status_code=400,
                detail="Workspace context required for this action",
            )
        for action in actions:
            if not check_team_permission(ctx, action):
                raise fastapi.HTTPException(
                    status_code=403,
                    detail=f"Missing workspace permission: {action.value}",
                )
        return ctx

    return _dependency
