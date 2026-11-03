from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional, cast

from autogpt_libs.auth.dependencies import ACCOUNT_BANNED_DETAIL
from autogpt_libs.auth.permissions import OrgAction, TeamAction
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.params import Depends as DependsParameter
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.models import APIKey as PrismaAPIKey
from prisma.models import OAuthAccessToken as PrismaOAuthAccessToken
from prisma.models import OAuthApplication as PrismaOAuthApplication
from pydantic import BaseModel, ConfigDict
from starlette.types import Scope

from backend.api.features.orgs import db as org_db
from backend.api.live_auth import live_dependency
from backend.api.org_rollout import require_personal_scope_without_collaboration
from backend.data.auth.api_key import APIKeyInfo, validate_api_key
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.auth.oauth import (
    InvalidClientError,
    InvalidTokenError,
    OAuthAccessTokenInfo,
    OAuthApplicationInfo,
    is_access_token,
    validate_access_token,
)
from backend.data.db import prisma
from backend.data.tenancy import (
    ResourceAccess,
    has_live_resource_access,
    has_live_resource_permission,
    live_request_transaction,
)
from backend.data.user import is_user_banned

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)

_REQUEST_AUTH = "external_api_auth"


def _auth_error(detail: str, status_code: int = status.HTTP_401_UNAUTHORIZED):
    return HTTPException(status_code=status_code, detail=detail)


@asynccontextmanager
async def _live_authorization_principal(
    auth: APIAuthorizationInfo,
    permissions: tuple[APIKeyPermission, ...],
) -> AsyncIterator[None]:
    async with live_request_transaction(prisma) as tx:
        if isinstance(auth, APIKeyInfo):
            locked = await tx.query_raw(
                'SELECT "id" FROM "APIKey" WHERE "id" = $1 FOR SHARE',
                auth.id,
            )
            if not locked:
                raise _auth_error("API key no longer exists")
            key = await PrismaAPIKey.prisma(tx).find_unique(where={"id": auth.id})
            if key is None or key.status != APIKeyStatus.ACTIVE:
                raise _auth_error("API key is no longer active")
            if key.userId != auth.user_id:
                raise _auth_error("API key owner changed")
            live_permissions = {APIKeyPermission(value) for value in key.permissions}
            missing = [
                permission
                for permission in permissions
                if permission not in live_permissions
            ]
            if missing:
                raise _auth_error(
                    "API key permissions changed",
                    status.HTTP_403_FORBIDDEN,
                )
        elif isinstance(auth, OAuthAccessTokenInfo):
            app_locked = await tx.query_raw(
                'SELECT "id" FROM "OAuthApplication" WHERE "id" = $1 FOR SHARE',
                auth.application_id,
            )
            if not app_locked:
                raise _auth_error("OAuth application no longer exists")
            token_locked = await tx.query_raw(
                'SELECT "id" FROM "OAuthAccessToken" WHERE "id" = $1 FOR SHARE',
                auth.id,
            )
            if not token_locked:
                raise _auth_error("OAuth access token no longer exists")
            application = await PrismaOAuthApplication.prisma(tx).find_unique(
                where={"id": auth.application_id}
            )
            token = await PrismaOAuthAccessToken.prisma(tx).find_unique(
                where={"id": auth.id}
            )
            if application is None or not application.isActive:
                raise _auth_error("OAuth application is no longer active")
            if (
                token is None
                or token.applicationId != auth.application_id
                or token.userId != auth.user_id
                or token.revokedAt is not None
                or token.expiresAt <= datetime.now(timezone.utc)
            ):
                raise _auth_error("OAuth access token is no longer active")
            live_token_permissions = {APIKeyPermission(value) for value in token.scopes}
            live_app_permissions = {
                APIKeyPermission(value) for value in application.scopes
            }
            missing = [
                permission
                for permission in permissions
                if permission not in live_token_permissions
                or permission not in live_app_permissions
            ]
            if missing:
                raise _auth_error(
                    "OAuth permissions changed",
                    status.HTTP_403_FORBIDDEN,
                )
        else:
            raise _auth_error("Unsupported authorization principal")
        yield


async def _scope_api_key(api_key: APIAuthorizationInfo) -> APIAuthorizationInfo:
    if api_key.organization_id is not None:
        await require_personal_scope_without_collaboration(
            api_key.user_id, api_key.organization_id, api_key.team_id_restriction
        )
        return api_key
    if api_key.team_id_restriction is not None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key has an invalid workspace scope",
        )
    organization_id, team_id = await org_db.get_user_default_team(api_key.user_id)
    if organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key has no active organization scope",
        )
    return api_key.model_copy(
        update={
            "organization_id": organization_id,
            "team_id_restriction": team_id,
        }
    )


async def _scope_oauth_token(
    token: APIAuthorizationInfo, application: OAuthApplicationInfo
) -> APIAuthorizationInfo:
    organization_id = application.organization_id
    team_id = application.team_id_restriction
    if organization_id is None:
        if team_id is not None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="OAuth application has an invalid workspace scope",
            )
        organization_id, team_id = await org_db.get_user_default_team(token.user_id)
    if organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="OAuth token has no active organization scope",
        )
    if application.organization_id is not None:
        await require_personal_scope_without_collaboration(
            token.user_id, organization_id, team_id
        )
    return token.model_copy(
        update={
            "organization_id": organization_id,
            "team_id_restriction": team_id,
        }
    )


class VerifiedCredential(BaseModel):
    """What this request's credential verified as: who, or rejected."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    credential: Optional[str]
    auth: Optional[APIAuthorizationInfo] = None
    rejection: Optional[HTTPException] = None


class _VerifiedPrincipal(BaseModel):
    """A verified credential, with the OAuth application a token was issued to."""

    auth: APIAuthorizationInfo
    application: OAuthApplicationInfo | None = None


async def resolve_auth_info(
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_auth),
) -> APIAuthorizationInfo | None:
    """
    Resolve authentication from API key or Bearer token headers.

    Returns the auth info if valid credentials are provided, or None if no
    credentials are present. Raises HTTPException on *invalid* credentials.
    The credential is not pinned to a tenant here: v1 does that in
    `require_auth`, v2 in its own tenancy resolution.
    """
    principal = await _verify_principal(api_key, bearer)
    return principal.auth if principal is not None else None


async def _verify_principal(
    api_key: str | None, bearer: HTTPAuthorizationCredentials | None
) -> _VerifiedPrincipal | None:
    if api_key is not None:
        api_key_info = await validate_api_key(api_key)
        if api_key_info:
            await _refuse_banned_owner(api_key_info)
            return _VerifiedPrincipal(auth=api_key_info)
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
            await _refuse_banned_owner(api_key_info)
            return _VerifiedPrincipal(auth=api_key_info)

        try:
            token_info, application = await validate_access_token(bearer.credentials)
        except (InvalidClientError, InvalidTokenError) as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
        await _refuse_banned_owner(token_info)
        return _VerifiedPrincipal(auth=token_info, application=application)

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
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_auth),
) -> APIAuthorizationInfo:
    """
    Unified authentication dependency that requires valid credentials.

    Accepts an API key (`X-API-Key`, or sent as a Bearer token) or an OAuth
    access token, rejects requests with no credentials, and pins the
    credential to the organization and team it acts in: an OAuth token to its
    application's scope, a credential minted before organizations to its
    owner's personal organization. An explicitly shared scope must pass the org
    rollout gate.
    """
    principal = await _verify_principal(api_key, bearer)
    if principal is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication. Provide API key or access token.",
        )
    if principal.application is not None:
        return await _scope_oauth_token(principal.auth, principal.application)
    return await _scope_api_key(principal.auth)


async def _refuse_banned_owner(auth: APIAuthorizationInfo) -> None:
    if await owner_is_banned(auth):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail=ACCOUNT_BANNED_DETAIL
        )


async def owner_is_banned(auth: APIAuthorizationInfo) -> bool:
    """Whether the account a credential belongs to is banned; for verifiers that
    resolve credentials outside `resolve_auth_info`, such as v2's MCP server."""
    return await is_user_banned(auth.user_id)


def permission_dependency(*permissions: APIKeyPermission) -> DependsParameter:
    """
    Dependency function for checking required permissions.
    All listed permissions must be present.
    (works with API keys and OAuth tokens)
    """

    async def check_permissions(
        auth: APIAuthorizationInfo = Security(
            require_auth, scopes=[p.value for p in permissions]
        ),
    ) -> AsyncIterator[APIAuthorizationInfo]:
        missing = [p for p in permissions if p not in auth.scopes]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permission(s): "
                f"{', '.join(p.value for p in missing)}",
            )
        async with _live_authorization_principal(auth, permissions):
            integration_admin = any(
                permission
                in {
                    APIKeyPermission.MANAGE_INTEGRATIONS,
                    APIKeyPermission.DELETE_INTEGRATIONS,
                }
                for permission in permissions
            )
            if integration_admin:
                has_access = await has_live_resource_permission(
                    auth.user_id,
                    auth.organization_id,
                    auth.team_id_restriction,
                    OrgAction.MANAGE_CREDENTIALS,
                    TeamAction.MANAGE_CREDENTIALS,
                )
            elif (access := _resource_access_for_permissions(permissions)) is not None:
                has_access = await has_live_resource_access(
                    auth.user_id,
                    auth.organization_id,
                    auth.team_id_restriction,
                    access,
                )
            else:
                has_access = True
            if auth.organization_id is None or not has_access:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="The principal no longer has access to this resource scope",
                )
            yield auth

    return live_dependency(check_permissions)


def require_permission(*permissions: APIKeyPermission) -> APIAuthorizationInfo:
    return cast(APIAuthorizationInfo, permission_dependency(*permissions))


def _resource_access_for_permissions(
    permissions: tuple[APIKeyPermission, ...],
) -> ResourceAccess | None:
    if any(
        permission
        in {
            APIKeyPermission.EXECUTE_GRAPH,
            APIKeyPermission.EXECUTE_BLOCK,
            APIKeyPermission.USE_TOOLS,
        }
        for permission in permissions
    ):
        return "execute"
    if any(
        permission
        in {
            APIKeyPermission.WRITE_GRAPH,
            APIKeyPermission.WRITE_LIBRARY,
        }
        for permission in permissions
    ):
        return "create"
    if any(
        permission
        in {
            APIKeyPermission.READ_GRAPH,
            APIKeyPermission.READ_INTEGRATIONS,
        }
        for permission in permissions
    ):
        return "view"
    return None


def add_auth_responses_to_openapi(app: FastAPI) -> None:
    """
    Add 401 responses to all endpoints secured with `require_auth` (v1) or
    v2's tenancy dependencies.
    """
    from autogpt_libs.auth.helpers import add_auth_responses_to_openapi

    add_auth_responses_to_openapi(
        app, [api_key_header.scheme_name, bearer_auth.scheme_name]
    )
