from typing import Annotated

from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from fastapi import APIRouter, HTTPException, Security

from backend.api.features.api_keys.model import (
    CreateAPIKeyRequest,
    CreateAPIKeyResponse,
    UpdatePermissionsRequest,
)
from backend.data.auth import api_key as api_key_db
from backend.data.tenancy import get_user_team_ids

router = APIRouter(dependencies=[Security(requires_user)])


@router.post("", summary="Create new API key")
async def create_api_key(
    request: CreateAPIKeyRequest,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> CreateAPIKeyResponse:
    """Create a new API key"""
    api_key_info, plain_text_key = await api_key_db.create_api_key(
        name=request.name,
        user_id=user_id,
        permissions=request.permissions,
        description=request.description,
        organization_id=ctx.org_id,
    )
    return CreateAPIKeyResponse(api_key=api_key_info, plain_text_key=plain_text_key)


@router.get("", summary="List user API keys")
async def get_api_keys(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> list[api_key_db.APIKeyInfo]:
    """List all API keys for the user"""
    team_ids = await get_user_team_ids(user_id, ctx.org_id) if ctx.org_id else []
    return await api_key_db.list_user_api_keys(
        user_id, organization_id=ctx.org_id or None, team_ids=team_ids
    )


@router.get("/{key_id}", summary="Get specific API key")
async def get_api_key(
    key_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> api_key_db.APIKeyInfo:
    """Get a specific API key"""
    api_key = await api_key_db.get_api_key_by_id(
        key_id, user_id, organization_id=ctx.org_id or None
    )
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    return api_key


@router.delete("/{key_id}", summary="Revoke API key")
async def delete_api_key(
    key_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> api_key_db.APIKeyInfo:
    """Revoke an API key"""
    return await api_key_db.revoke_api_key(
        key_id, user_id, organization_id=ctx.org_id or None
    )


@router.post("/{key_id}/suspend", summary="Suspend API key")
async def suspend_key(
    key_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> api_key_db.APIKeyInfo:
    """Suspend an API key"""
    return await api_key_db.suspend_api_key(
        key_id, user_id, organization_id=ctx.org_id or None
    )


@router.put("/{key_id}/permissions", summary="Update key permissions")
async def update_permissions(
    key_id: str,
    request: UpdatePermissionsRequest,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> api_key_db.APIKeyInfo:
    """Update API key permissions"""
    return await api_key_db.update_api_key_permissions(
        key_id, user_id, request.permissions, organization_id=ctx.org_id or None
    )
