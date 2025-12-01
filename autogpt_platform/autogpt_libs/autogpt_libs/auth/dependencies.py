"""
FastAPI dependency functions for JWT-based authentication and authorization.

These are the high-level dependency functions used in route definitions.
"""

import logging

import fastapi
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .jwt_utils import get_jwt_payload, verify_user
from .models import User

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
