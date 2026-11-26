from collections.abc import Awaitable, Callable
from typing import Annotated, Optional

from autogpt_libs.auth import reject_closed_account
from autogpt_libs.auth.dependencies import ACCOUNT_BANNED_DETAIL
from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prisma.enums import APIKeyPermission
from pydantic import BaseModel, ConfigDict
from starlette.types import Scope

from backend.api.external.auth_rate_limit import (
    APIKeyValidationBudget,
    enforce_api_key_validate_rate_limit,
    refund_api_key_validate_rate_limit,
)
from backend.api.features.orgs.db import get_user_default_team, is_active_member
from backend.data.auth.api_key import validate_api_key
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.auth.oauth import (
    InvalidClientError,
    InvalidTokenError,
    is_access_token,
    validate_access_token,
)
from backend.data.tenancy import resource_role_refusal
from backend.data.user import is_user_banned
from backend.util.cache import cached

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)

_REQUEST_AUTH = "external_api_auth"

FORMER_MEMBER_DETAIL = (
    "These credentials belong to an organization or team you are not an active "
    "member of"
)
NO_ORGANIZATION_DETAIL = (
    "No organization context available. Your account may be in an inconsistent "
    "state — please contact support."
)

# How long a membership answer is reused. Nearly every key names an
# organization, and the rate limiter and the route each check the caller, so
# without it every request would read the membership twice. Removing a member
# revokes their keys there at once; the cache only delays the other ways a
# membership ends, such as a deleted organization, by this much.
MEMBERSHIP_CACHE_SECONDS = 10


class VerifiedCredential(BaseModel):
    """What this request's credential verified as: who, or rejected."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    credential: Optional[str]
    auth: Optional[APIAuthorizationInfo] = None
    rejection: Optional[HTTPException] = None


# Every path rejects a closed account's key or token with a 401: deletion
# revokes every key and grant, but one issued while the account was closing
# would otherwise keep working until the account's data is erased.
async def resolve_auth_info(
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_auth),
    *,
    before_api_key_verify: Callable[[], Awaitable[None]] | None = None,
) -> APIAuthorizationInfo | None:
    """
    Resolve authentication from API key or Bearer token headers.

    Returns the auth info if valid credentials are provided, or None if no
    credentials are present. Raises HTTPException on *invalid* credentials.
    """
    key = _plaintext_api_key(api_key, bearer)
    if key is not None:
        api_key_info = (
            await validate_api_key(key, before_verify=before_api_key_verify)
            if before_api_key_verify is not None
            else await validate_api_key(key)
        )
        if api_key_info:
            await reject_closed_account(api_key_info.user_id)
            await _refuse_banned_owner(api_key_info)
            return api_key_info
        if api_key is not None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
            )

    if bearer is not None:
        try:
            token_info, _ = await validate_access_token(bearer.credentials)
        except (InvalidClientError, InvalidTokenError) as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
        await reject_closed_account(token_info.user_id)
        await _refuse_banned_owner(token_info)
        return token_info

    return None


async def resolve_request_auth(
    scope: Scope,
    api_key: str | None,
    bearer: HTTPAuthorizationCredentials | None,
    *,
    before_api_key_verify: Callable[[], Awaitable[None]] | None = None,
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
                auth=await resolve_auth_info(
                    api_key=api_key,
                    bearer=bearer,
                    before_api_key_verify=before_api_key_verify,
                ),
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
    request: Request,
    api_key: Annotated[str | None, Security(api_key_header)] = None,
    bearer: Annotated[
        HTTPAuthorizationCredentials | None, Security(bearer_auth)
    ] = None,
) -> APIAuthorizationInfo:
    """The request's verified caller; 401 when it sent no credentials.

    Resolved once per request: v1's rate limiter and its routes ask for it under
    different scopes, which FastAPI caches apart. Verifying an API key first
    reserves from v1's pre-Scrypt budget, which a valid key gets back. The
    membership check below runs on every call, stored result or not, so
    credentials whose owner has left the organization or team they were issued
    in are refused, and like a closed or banned account's key they keep their
    reservation. Every v1 route reaches this, directly or through
    `require_permission`.
    """
    key = _plaintext_api_key(api_key, bearer)
    budget: APIKeyValidationBudget | None = None

    async def reserve() -> None:
        nonlocal budget
        if key is not None:
            budget = await enforce_api_key_validate_rate_limit(request, key)

    auth = await resolve_request_auth(
        request.scope, api_key=api_key, bearer=bearer, before_api_key_verify=reserve
    )
    if auth is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication. Provide API key or access token.",
        )
    await _refuse_former_member(auth)
    await refund_api_key_validate_rate_limit(budget)
    return auth


def _plaintext_api_key(
    api_key: str | None, bearer: HTTPAuthorizationCredentials | None
) -> str | None:
    """Prefer an explicit API key while supporting Bearer-only MCP clients."""
    if api_key is not None:
        return api_key
    if bearer is not None and not is_access_token(bearer.credentials):
        return bearer.credentials
    return None


async def _refuse_banned_owner(auth: APIAuthorizationInfo) -> None:
    if await owner_is_banned(auth):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=ACCOUNT_BANNED_DETAIL
        )


async def owner_is_banned(auth: APIAuthorizationInfo) -> bool:
    """Whether the account a credential belongs to is banned; for verifiers that
    resolve credentials outside `resolve_auth_info`, such as v2's MCP server."""
    return await is_user_banned(auth.user_id)


async def _refuse_former_member(auth: APIAuthorizationInfo) -> None:
    """Refuse a credential issued in an organization or team its owner has left.

    v1 acts in the organization and team stored on the credential, so it is
    good only while its owner is an active member of both, as v2 checks too.
    It is refused, never moved to the owner's personal organization, which it
    wasn't issued for. A credential that names no organization (a key made
    before organizations existed) acts in that personal organization, so there
    is nothing to check; one that names a team but no organization is refused.
    """
    if auth.organization_id is None and auth.team_id_restriction is None:
        return
    if auth.organization_id is None or not await holds_membership(
        auth.user_id, auth.organization_id, auth.team_id_restriction
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=FORMER_MEMBER_DETAIL
        )


@cached(maxsize=1000, ttl_seconds=MEMBERSHIP_CACHE_SECONDS)
async def holds_membership(
    user_id: str, organization_id: str, team_id: str | None
) -> bool:
    """`is_active_member`, reused for `MEMBERSHIP_CACHE_SECONDS`."""
    return await is_active_member(user_id, organization_id, team_id)


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


def require_resource_permission(*permissions: APIKeyPermission):
    """`require_permission`, for routes that create, run or read workflows and
    runs in the credential's organization and team.

    The web app holds these to the caller's role there: a membership whose only
    role is billing manager covers spend, not workflows and runs. A credential
    that names no organization acts in its owner's personal organization, which
    its owner owns, so the route gets it bound to that organization's home.
    """
    check_permissions = require_permission(*permissions)

    async def check_resource_role(
        auth: APIAuthorizationInfo = Security(check_permissions),
    ) -> APIAuthorizationInfo:
        if not auth.organization_id:
            return await _in_personal_organization(auth)
        if reason := await resource_role_refusal(
            auth.user_id, auth.organization_id, auth.team_id_restriction
        ):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=reason)
        return auth

    return check_resource_role


async def _in_personal_organization(
    auth: APIAuthorizationInfo,
) -> APIAuthorizationInfo:
    organization_id, _ = await get_user_default_team(auth.user_id)
    if organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=NO_ORGANIZATION_DETAIL
        )
    return auth.model_copy(update={"organization_id": organization_id})


def add_auth_responses_to_openapi(app: FastAPI) -> None:
    """
    Add 401 responses to all endpoints secured with `require_auth`,
    `require_api_key`, or `require_access_token` middleware.
    """
    from autogpt_libs.auth.helpers import add_auth_responses_to_openapi

    add_auth_responses_to_openapi(
        app, [api_key_header.scheme_name, bearer_auth.scheme_name]
    )
