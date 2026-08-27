from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import cast

from autogpt_libs.auth.permissions import OrgAction, TeamAction
from fastapi import HTTPException, Security, status
from fastapi.params import Depends as DependsParameter
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.models import APIKey as PrismaAPIKey
from prisma.models import OAuthAccessToken as PrismaOAuthAccessToken
from prisma.models import OAuthApplication as PrismaOAuthApplication

from backend.api.live_auth import live_dependency
from backend.data.auth.api_key import APIKeyInfo, validate_api_key
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.auth.oauth import (
    InvalidClientError,
    InvalidTokenError,
    OAuthAccessTokenInfo,
    OAuthApplicationInfo,
    validate_access_token,
)
from backend.data.db import prisma
from backend.data.tenancy import (
    ResourceAccess,
    has_live_resource_access,
    has_live_resource_permission,
    live_request_transaction,
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_auth = HTTPBearer(auto_error=False)


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


async def _scope_api_key(api_key: APIKeyInfo) -> APIKeyInfo:
    if api_key.organization_id is not None:
        return api_key
    if api_key.team_id_restriction is not None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key has an invalid workspace scope",
        )
    from backend.api.features.orgs.db import get_user_default_team

    organization_id, team_id = await get_user_default_team(api_key.user_id)
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
    token: OAuthAccessTokenInfo, application: OAuthApplicationInfo
) -> OAuthAccessTokenInfo:
    organization_id = application.organization_id
    team_id = application.team_id_restriction
    if organization_id is None:
        if team_id is not None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="OAuth application has an invalid workspace scope",
            )
        from backend.api.features.orgs.db import get_user_default_team

        organization_id, team_id = await get_user_default_team(token.user_id)
    if organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="OAuth token has no active organization scope",
        )
    return token.model_copy(
        update={
            "organization_id": organization_id,
            "team_id_restriction": team_id,
        }
    )


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

    return await _scope_api_key(api_key_obj)


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
        token_info, application = await validate_access_token(bearer.credentials)
    except (InvalidClientError, InvalidTokenError) as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    return await _scope_oauth_token(token_info, application)


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
            return await _scope_api_key(api_key_info)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    # Try OAuth bearer token
    if bearer is not None:
        try:
            token_info, application = await validate_access_token(bearer.credentials)
            return await _scope_oauth_token(token_info, application)
        except (InvalidClientError, InvalidTokenError) as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))

    # No credentials provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing authentication. Provide API key or access token.",
    )


def permission_dependency(*permissions: APIKeyPermission) -> DependsParameter:
    """
    Dependency function for checking required permissions.
    All listed permissions must be present.
    (works with API keys and OAuth tokens)
    """

    async def check_permissions(
        auth: APIAuthorizationInfo = Security(require_auth),
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
