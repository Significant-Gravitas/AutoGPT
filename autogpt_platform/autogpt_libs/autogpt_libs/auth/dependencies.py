"""
FastAPI dependency functions for JWT-based authentication and authorization.

These are the high-level dependency functions used in route definitions.
"""

import logging

import fastapi

from .jwt_utils import get_jwt_payload, verify_user
from .models import User

logger = logging.getLogger(__name__)

# Header name for admin impersonation
IMPERSONATION_HEADER_NAME = "X-Act-As-User-Id"


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
            f"acting as user {impersonate_header}"
        )

        return impersonate_header

    return user_id
