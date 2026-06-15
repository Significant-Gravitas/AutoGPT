from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prisma.enums import APIKeyPermission

from backend.data.auth.api_key import validate_api_key
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.auth.oauth import (
    InvalidClientError,
    InvalidTokenError,
    validate_access_token,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


async def resolve_auth_info(
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_auth),
) -> APIAuthorizationInfo | None:
    """
    Resolve authentication from API key or Bearer token headers.

    Returns the auth info if valid credentials are provided, or None if no
    credentials are present. Raises HTTPException on *invalid* credentials.
    """
    if api_key is not None:
        api_key_info = await validate_api_key(api_key)
        if api_key_info:
            return api_key_info
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    if bearer is not None:
        try:
            token_info, _ = await validate_access_token(bearer.credentials)
            return token_info
        except (InvalidClientError, InvalidTokenError) as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    return None


async def require_auth(
    auth: APIAuthorizationInfo | None = Security(resolve_auth_info),
) -> APIAuthorizationInfo:
    """
    Unified authentication dependency that requires valid credentials.

    Depends on `resolve_auth_info` (which accepts API key or Bearer token)
    and rejects requests with no credentials.
    """
    if auth is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication. Provide API key or access token.",
        )
    return auth


def require_permission(*permissions: APIKeyPermission):
    """
    Dependency function for checking required permissions.
    All listed permissions must be present.
    (works with API keys and OAuth tokens)
    """

    async def check_permissions(
        auth: APIAuthorizationInfo = Security(
            require_auth, scopes=[p.value for p in permissions]
        ),
    ) -> APIAuthorizationInfo:
        missing = [p for p in permissions if p not in auth.scopes]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission(s): "
                f"{', '.join(p.value for p in missing)}",
            )
        return auth

    return check_permissions


def add_auth_responses_to_openapi(app: FastAPI) -> None:
    """
    Add 401 responses to all endpoints secured with `require_auth`,
    `require_api_key`, or `require_access_token` middleware.
    """
    from autogpt_libs.auth.helpers import add_auth_responses_to_openapi

    add_auth_responses_to_openapi(
        app, [api_key_header.scheme_name, bearer_auth.scheme_name]
    )
