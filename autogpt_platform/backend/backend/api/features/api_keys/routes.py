from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from autogpt_libs.auth import get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction, TeamAction
from fastapi import APIRouter, HTTPException, Security

from backend.api.features.api_keys.model import (
    CreateAPIKeyRequest,
    CreateAPIKeyResponse,
    UpdatePermissionsRequest,
)
from backend.api.live_auth import (
    requires_live_resource_permission as requires_resource_permission,
)
from backend.api.live_auth import resolve_write_team_id
from backend.api.org_rollout import require_personal_scope
from backend.data.auth import api_key as api_key_db
from backend.data.tenancy import get_user_team_ids, live_resource_permission_barrier

router = APIRouter(dependencies=[Security(requires_user)])


@asynccontextmanager
async def _live_api_key_target(
    key_id: str,
    user_id: str,
    ctx: RequestContext,
) -> AsyncIterator[api_key_db.APIKeyInfo]:
    candidate = await api_key_db.get_api_key_by_id(
        key_id,
        user_id,
        organization_id=ctx.org_id or None,
        team_id_restriction=ctx.team_id,
        exact_scope=ctx.team_id is not None,
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="API key not found")
    target_team_id = candidate.team_id_restriction
    async with live_resource_permission_barrier(
        user_id,
        ctx.org_id,
        target_team_id,
        OrgAction.MANAGE_CREDENTIALS,
        TeamAction.MANAGE_CREDENTIALS,
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
        current = await api_key_db.get_api_key_by_id(
            key_id,
            user_id,
            organization_id=ctx.org_id or None,
            team_id_restriction=target_team_id,
            exact_scope=True,
        )
        if current is None or current.team_id_restriction != target_team_id:
            raise HTTPException(status_code=404, detail="API key not found")
        yield current


@router.post("", summary="Create new API key")
async def create_api_key(
    request: CreateAPIKeyRequest,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.MANAGE_CREDENTIALS, TeamAction.MANAGE_CREDENTIALS
        ),
    ],
) -> CreateAPIKeyResponse:
    """Create a new API key in your personal organization's default team.

    Keys act there only for now; one for any other organization or team is
    refused with a 403.
    """
    if (
        ctx.team_id is not None
        and request.team_id is not None
        and request.team_id != ctx.team_id
    ):
        raise HTTPException(
            status_code=400,
            detail="The requested team must match the active workspace context.",
        )
    requested_team_id = await resolve_write_team_id(
        user_id,
        ctx.org_id,
        request.team_id if request.team_id is not None else ctx.team_id,
    )
    organization_id, team_id_restriction = await require_personal_scope(
        user_id, ctx.org_id, requested_team_id
    )
    async with live_resource_permission_barrier(
        user_id,
        organization_id,
        team_id_restriction,
        OrgAction.MANAGE_CREDENTIALS,
        TeamAction.MANAGE_CREDENTIALS,
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
        api_key_info, plain_text_key = await api_key_db.create_api_key(
            name=request.name,
            user_id=user_id,
            permissions=request.permissions,
            description=request.description,
            organization_id=organization_id,
            owner_type="USER",
            team_id_restriction=team_id_restriction,
        )
    return CreateAPIKeyResponse(api_key=api_key_info, plain_text_key=plain_text_key)


@router.get("", summary="List user API keys")
async def get_api_keys(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.MANAGE_CREDENTIALS, TeamAction.MANAGE_CREDENTIALS
        ),
    ],
) -> list[api_key_db.APIKeyInfo]:
    """List all API keys for the user"""
    if ctx.team_id is not None:
        return await api_key_db.list_user_api_keys(
            user_id,
            organization_id=ctx.org_id or None,
            team_id_restriction=ctx.team_id,
            exact_scope=True,
        )
    return await api_key_db.list_user_api_keys(
        user_id,
        organization_id=ctx.org_id or None,
        team_ids=(
            [ctx.team_id]
            if ctx.team_id is not None
            else await get_user_team_ids(user_id, ctx.org_id) if ctx.org_id else []
        ),
    )


@router.get("/{key_id}", summary="Get specific API key")
async def get_api_key(
    key_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.MANAGE_CREDENTIALS, TeamAction.MANAGE_CREDENTIALS
        ),
    ],
) -> api_key_db.APIKeyInfo:
    """Get a specific API key"""
    async with _live_api_key_target(key_id, user_id, ctx) as api_key:
        return api_key


@router.delete("/{key_id}", summary="Revoke API key")
async def delete_api_key(
    key_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.MANAGE_CREDENTIALS, TeamAction.MANAGE_CREDENTIALS
        ),
    ],
) -> api_key_db.APIKeyInfo:
    """Revoke an API key"""
    async with _live_api_key_target(key_id, user_id, ctx):
        return await api_key_db.revoke_api_key(
            key_id,
            user_id,
            organization_id=ctx.org_id or None,
            team_id_restriction=ctx.team_id,
            exact_scope=True,
        )


@router.post("/{key_id}/suspend", summary="Suspend API key")
async def suspend_key(
    key_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.MANAGE_CREDENTIALS, TeamAction.MANAGE_CREDENTIALS
        ),
    ],
) -> api_key_db.APIKeyInfo:
    """Suspend an API key"""
    async with _live_api_key_target(key_id, user_id, ctx):
        return await api_key_db.suspend_api_key(
            key_id,
            user_id,
            organization_id=ctx.org_id or None,
            team_id_restriction=ctx.team_id,
            exact_scope=True,
        )


@router.put("/{key_id}/permissions", summary="Update key permissions")
async def update_permissions(
    key_id: str,
    request: UpdatePermissionsRequest,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.MANAGE_CREDENTIALS, TeamAction.MANAGE_CREDENTIALS
        ),
    ],
) -> api_key_db.APIKeyInfo:
    """Update API key permissions"""
    async with _live_api_key_target(key_id, user_id, ctx):
        return await api_key_db.update_api_key_permissions(
            key_id,
            user_id,
            request.permissions,
            organization_id=ctx.org_id or None,
            team_id_restriction=ctx.team_id,
            exact_scope=True,
        )
