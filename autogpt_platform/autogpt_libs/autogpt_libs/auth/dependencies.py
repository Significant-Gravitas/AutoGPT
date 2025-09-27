"""
FastAPI dependency functions for JWT-based authentication and authorization.

These are the high-level dependency functions used in route definitions.
"""

import fastapi

from .jwt_utils import get_jwt_payload, verify_user
from .models import User


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


async def get_user_id(jwt_payload: dict = fastapi.Security(get_jwt_payload)) -> str:
    """
    FastAPI dependency that returns the ID of the authenticated user.

    Raises:
        HTTPException: 401 for authentication failures or missing user ID
    """
    user_id = jwt_payload.get("sub")
    if not user_id:
        raise fastapi.HTTPException(
            status_code=401, detail="User ID not found in token"
        )
    return user_id
