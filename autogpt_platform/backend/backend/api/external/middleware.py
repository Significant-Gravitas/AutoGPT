from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prisma.enums import APIKeyPermission

from backend.data.auth.api_key import APIKeyInfo, validate_api_key
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.auth.oauth import (
    InvalidClientError,
    InvalidTokenError,
    OAuthAccessTokenInfo,
    validate_access_token,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


async def require_api_key(api_key: str | None = Security(api_key_header)) -> APIKeyInfo:
    """Middleware for API key authentication only"""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing API key"
        )

    api_key_obj = await validate_api_key(api_key)

    if not api_key_obj:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    return api_key_obj


async def require_access_token(
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_auth),
) -> OAuthAccessTokenInfo:
    """Middleware for OAuth access token authentication only"""
    if bearer is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    try:
        token_info, _ = await validate_access_token(bearer.credentials)
    except (InvalidClientError, InvalidTokenError) as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    return token_info


async def require_auth(
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_auth),
) -> APIAuthorizationInfo:
    """
    Unified authentication middleware supporting both API keys and OAuth tokens.

    Supports two authentication methods, which are checked in order:
    1. X-API-Key header (existing API key authentication)
    2. Authorization: Bearer <token> header (OAuth access token)

    Returns:
        APIAuthorizationInfo: base class of both APIKeyInfo and OAuthAccessTokenInfo.
    """
    # Try API key first
    if api_key is not None:
        api_key_info = await validate_api_key(api_key)
        if api_key_info:
            return api_key_info
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    # Try OAuth bearer token
    if bearer is not None:
        try:
            token_info, _ = await validate_access_token(bearer.credentials)
            return token_info
        except (InvalidClientError, InvalidTokenError) as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    # No credentials provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing authentication. Provide API key or access token.",
    )


def require_permission(permission: APIKeyPermission):
    """
    Dependency function for checking specific permissions
    (works with API keys and OAuth tokens)
    """

    async def check_permission(
        auth: APIAuthorizationInfo = Security(require_auth),
    ) -> APIAuthorizationInfo:
        if permission not in auth.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission: {permission.value}",
            )
        return auth

    return check_permission
