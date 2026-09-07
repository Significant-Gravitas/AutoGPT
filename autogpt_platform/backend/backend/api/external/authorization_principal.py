"""Short principal checks and cancellable authorization for external requests.

Revocation is checked every 0.5 seconds, with a 0.5 second validation deadline:
ongoing async work is cancelled within about one second of revocation or loss of
database validation. A final check protects the response between polls. No row
lock or request transaction is held while the handler performs external I/O.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from prisma import Prisma
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.models import APIKey as PrismaAPIKey
from prisma.models import OAuthAccessToken as PrismaOAuthAccessToken
from prisma.models import OAuthApplication as PrismaOAuthApplication
from pydantic import BaseModel, ConfigDict

from backend.data.auth.api_key import APIKeyInfo
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.auth.oauth import OAuthAccessTokenInfo
from backend.data.db import prisma
from backend.data.tenancy import live_request_transaction

PRINCIPAL_POLL_INTERVAL_SECONDS = 0.5
PRINCIPAL_VALIDATION_TIMEOUT_SECONDS = 0.5


class _PrincipalScope(BaseModel):
    model_config = ConfigDict(frozen=True)

    organization_id: str | None
    team_id: str | None
    owner_id: str
    owner_type: str | None = None


@asynccontextmanager
async def live_authorization_principal(
    auth: APIAuthorizationInfo,
    permissions: tuple[APIKeyPermission, ...],
) -> AsyncIterator[None]:
    scope = await _validate_principal(auth, permissions)
    owner = asyncio.current_task()
    if owner is None:
        raise RuntimeError("Authorization requires an active request task")
    lost_authorization: HTTPException | None = None

    async def monitor() -> None:
        nonlocal lost_authorization
        try:
            while True:
                await asyncio.sleep(PRINCIPAL_POLL_INTERVAL_SECONDS)
                await _validate_principal(auth, permissions, scope)
        except HTTPException as exc:
            lost_authorization = exc
            owner.cancel()

    monitor_task = asyncio.create_task(monitor())
    try:
        try:
            yield
        finally:
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)
    except asyncio.CancelledError:
        if lost_authorization is None or owner.uncancel():
            raise
        raise lost_authorization from None
    if lost_authorization is not None:
        owner.uncancel()
        raise lost_authorization
    await _validate_principal(auth, permissions, scope)


async def _validate_principal(
    auth: APIAuthorizationInfo,
    permissions: tuple[APIKeyPermission, ...],
    original_scope: _PrincipalScope | None = None,
) -> _PrincipalScope:
    try:
        async with asyncio.timeout(PRINCIPAL_VALIDATION_TIMEOUT_SECONDS):
            deadline = timedelta(seconds=PRINCIPAL_VALIDATION_TIMEOUT_SECONDS)
            async with live_request_transaction(prisma, timeout=deadline) as tx:
                scope = await _read_principal_scope(tx, auth, permissions)
                if original_scope is not None and scope != original_scope:
                    raise _auth_error("Authorization principal scope changed", 403)
                if scope.organization_id is not None and (
                    scope.organization_id != auth.organization_id
                    or scope.team_id != auth.team_id_restriction
                ):
                    raise _auth_error("Authorization principal scope changed", 403)
                if scope.organization_id is None and scope.team_id is not None:
                    raise _auth_error("Authorization principal scope is invalid", 403)
                return scope
    except HTTPException:
        raise
    except Exception:
        raise _auth_error("Authorization could not be validated", 503) from None


async def _read_principal_scope(
    tx: Prisma,
    auth: APIAuthorizationInfo,
    permissions: tuple[APIKeyPermission, ...],
) -> _PrincipalScope:
    if isinstance(auth, APIKeyInfo):
        return await _read_api_key_scope(tx, auth, permissions)
    if isinstance(auth, OAuthAccessTokenInfo):
        return await _read_oauth_scope(tx, auth, permissions)
    raise _auth_error("Unsupported authorization principal")


async def _read_api_key_scope(
    tx: Prisma,
    auth: APIKeyInfo,
    permissions: tuple[APIKeyPermission, ...],
) -> _PrincipalScope:
    locked = await tx.query_raw(
        'SELECT "id" FROM "APIKey" WHERE "id" = $1 FOR SHARE', auth.id
    )
    if not locked:
        raise _auth_error("API key no longer exists")
    key = await PrismaAPIKey.prisma(tx).find_unique(where={"id": auth.id})
    if key is None or key.status != APIKeyStatus.ACTIVE or key.revokedAt is not None:
        raise _auth_error("API key is no longer active")
    if key.userId != auth.user_id or key.ownerType != auth.owner_type:
        raise _auth_error("API key owner changed")
    if not set(permissions).issubset(key.permissions):
        raise _auth_error("API key permissions changed", 403)
    return _PrincipalScope(
        organization_id=key.organizationId,
        team_id=key.teamIdRestriction,
        owner_id=key.userId,
        owner_type=key.ownerType,
    )


async def _read_oauth_scope(
    tx: Prisma,
    auth: OAuthAccessTokenInfo,
    permissions: tuple[APIKeyPermission, ...],
) -> _PrincipalScope:
    app_locked = await tx.query_raw(
        'SELECT "id" FROM "OAuthApplication" WHERE "id" = $1 FOR SHARE',
        auth.application_id,
    )
    token_locked = await tx.query_raw(
        'SELECT "id" FROM "OAuthAccessToken" WHERE "id" = $1 FOR SHARE', auth.id
    )
    if not app_locked or not token_locked:
        raise _auth_error("OAuth authorization no longer exists")
    application = await PrismaOAuthApplication.prisma(tx).find_unique(
        where={"id": auth.application_id}
    )
    token = await PrismaOAuthAccessToken.prisma(tx).find_unique(where={"id": auth.id})
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
    if not set(permissions).issubset(token.scopes) or not set(permissions).issubset(
        application.scopes
    ):
        raise _auth_error("OAuth permissions changed", 403)
    return _PrincipalScope(
        organization_id=application.organizationId,
        team_id=application.teamIdRestriction,
        owner_id=application.ownerId,
        owner_type=application.ownerType,
    )


def _auth_error(detail: str, status_code: int = status.HTTP_401_UNAUTHORIZED):
    return HTTPException(status_code=status_code, detail=detail)
