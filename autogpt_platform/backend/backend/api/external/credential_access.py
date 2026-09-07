from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from autogpt_libs.auth.permissions import OrgAction, TeamAction
from fastapi import HTTPException

from backend.data.auth.base import APIAuthorizationInfo
from backend.data.db_accessors import (
    LiveResourceAccessRevoked,
    LiveResourceLeaseGuard,
    live_resource_lease,
)
from backend.data.tenancy import live_resource_permission_barrier


@asynccontextmanager
async def live_credential_management_lease(
    auth: APIAuthorizationInfo,
) -> AsyncIterator[LiveResourceLeaseGuard]:
    """Pin membership on the dedicated lease pool while credential I/O runs.

    The view lease pins the same membership/role mutation locks as the short
    management permission check, so that check remains valid after its request
    transaction exits. Callers run network work through the yielded guard.
    """
    try:
        async with live_resource_lease(
            auth.user_id, auth.organization_id, auth.team_id_restriction, "view"
        ) as lease:
            if not lease:
                raise HTTPException(403, "Credential management access was revoked")
            async with live_resource_permission_barrier(
                auth.user_id,
                auth.organization_id,
                auth.team_id_restriction,
                OrgAction.MANAGE_CREDENTIALS,
                TeamAction.MANAGE_CREDENTIALS,
            ) as allowed:
                if not allowed:
                    raise HTTPException(403, "Credential management access was revoked")
            yield lease
    except LiveResourceAccessRevoked:
        raise HTTPException(403, "Credential management access was revoked") from None
