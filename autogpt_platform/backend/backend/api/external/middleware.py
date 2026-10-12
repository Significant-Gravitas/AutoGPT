from typing import Optional

from fastapi import FastAPI, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prisma.enums import APIKeyPermission
from pydantic import BaseModel, ConfigDict
from starlette.types import Scope

from backend.data.auth.api_key import validate_api_key
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.auth.oauth import (
    InvalidClientError,
    InvalidTokenError,
    is_access_token,
    validate_access_token,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)

_REQUEST_AUTH = "external_api_auth"


class VerifiedCredential(BaseModel):
    """What this request's credential verified as: who, or rejected."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    credential: Optional[str]
    auth: Optional[APIAuthorizationInfo] = None
    rejection: Optional[HTTPException] = None


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
        # MCP clients can only send credentials as a Bearer token, so an API key
        # arrives here too; without this the request resolves to anonymous and
        # gets the per-IP rate limit. Order matches v2's MCP TokenVerifier.
        if not is_access_token(bearer.credentials) and (
            api_key_info := await validate_api_key(bearer.credentials)
        ):
            return api_key_info

        try:
            token_info, _ = await validate_access_token(bearer.credentials)
            return token_info
        except (InvalidClientError, InvalidTokenError) as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    return None


async def resolve_request_auth(
    scope: Scope,
    api_key: str | None,
    bearer: HTTPAuthorizationCredentials | None,
) -> APIAuthorizationInfo | None:
    """`resolve_auth_info`, verified at most once per request.

    v2's rate limiter and route dependency both need the caller, and an API key
    costs a Scrypt hash per check. A rejection is remembered too, so an invalid
    key costs one hash, not two; an error that isn't a rejection (the database
    unreachable) is not, so the route's own check tries again. The result is
    kept with the credential it was for, and only reused for that credential.
    """
    credential = (
        api_key if api_key is not None else bearer.credentials if bearer else None
    )
    state = scope.setdefault("state", {})
    verified = state.get(_REQUEST_AUTH)
    if not isinstance(verified, VerifiedCredential) or (
        verified.credential != credential
    ):
        try:
            verified = VerifiedCredential(
                credential=credential,
                auth=await resolve_auth_info(api_key=api_key, bearer=bearer),
            )
        except HTTPException as rejection:
            verified = VerifiedCredential(credential=credential, rejection=rejection)
        state[_REQUEST_AUTH] = verified
    if verified.rejection is not None:
        raise HTTPException(
            status_code=verified.rejection.status_code,
            detail=verified.rejection.detail,
            headers=verified.rejection.headers,
        )
    return verified.auth


def verified_credential(scope: Scope) -> Optional[VerifiedCredential]:
    """What this request's credential already verified as, if it was checked."""
    verified = scope.get("state", {}).get(_REQUEST_AUTH)
    return verified if isinstance(verified, VerifiedCredential) else None


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
