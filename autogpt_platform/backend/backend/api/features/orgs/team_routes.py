"""Workspace management API routes (nested under /api/orgs/{org_id}/workspaces)."""

from typing import Annotated

from autogpt_libs.auth import (
    get_request_context,
    requires_org_permission,
    requires_team_permission,
)
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction, TeamAction
from fastapi import APIRouter, HTTPException, Security

from . import team_db as team_db
from .team_model import (
    AddTeamMemberRequest,
    CreateTeamRequest,
    TeamMemberResponse,
    TeamResponse,
    UpdateTeamMemberRequest,
    UpdateTeamRequest,
)

router = APIRouter()


@router.post(
    "",
    summary="Create workspace",
    tags=["orgs", "workspaces"],
)
async def create_team(
    org_id: str,
    request: CreateTeamRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.CREATE_WORKSPACES)),
    ],
) -> TeamResponse:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    return await team_db.create_team(
        org_id=org_id,
        name=request.name,
        user_id=ctx.user_id,
        description=request.description,
        join_policy=request.join_policy,
    )


@router.get(
    "",
    summary="List workspaces",
    tags=["orgs", "workspaces"],
)
async def list_teams(
    org_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> list[TeamResponse]:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    return await team_db.list_teams(org_id, ctx.user_id)


@router.get(
    "/{ws_id}",
    summary="Get workspace details",
    tags=["orgs", "workspaces"],
)
async def get_team(
    org_id: str,
    ws_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> TeamResponse:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    return await team_db.get_team(ws_id, expected_org_id=org_id)


@router.patch(
    "/{ws_id}",
    summary="Update workspace",
    tags=["orgs", "workspaces"],
)
async def update_team(
    org_id: str,
    ws_id: str,
    request: UpdateTeamRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_team_permission(TeamAction.MANAGE_SETTINGS)),
    ],
) -> TeamResponse:
    if ctx.team_id != ws_id:
        raise HTTPException(403, detail="Team context does not match the target team")
    # Verify workspace belongs to org (ctx validates workspace membership)
    await team_db.get_team(ws_id, expected_org_id=org_id)
    return await team_db.update_team(
        ws_id,
        {
            "name": request.name,
            "description": request.description,
            "joinPolicy": request.join_policy,
        },
    )


@router.delete(
    "/{ws_id}",
    summary="Delete workspace",
    tags=["orgs", "workspaces"],
    status_code=204,
)
async def delete_team(
    org_id: str,
    ws_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_WORKSPACES)),
    ],
) -> None:
    await team_db.get_team(ws_id, expected_org_id=org_id)
    await team_db.delete_team(ws_id)


@router.post(
    "/{ws_id}/join",
    summary="Self-join open workspace",
    tags=["orgs", "workspaces"],
)
async def join_team(
    org_id: str,
    ws_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> TeamResponse:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    return await team_db.join_team(ws_id, ctx.user_id, org_id)


@router.post(
    "/{ws_id}/leave",
    summary="Leave workspace",
    tags=["orgs", "workspaces"],
    status_code=204,
)
async def leave_team(
    org_id: str,
    ws_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> None:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    await team_db.leave_team(ws_id, ctx.user_id)


# --- Members ---


@router.get(
    "/{ws_id}/members",
    summary="List workspace members",
    tags=["orgs", "workspaces"],
)
async def list_members(
    org_id: str,
    ws_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> list[TeamMemberResponse]:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    await team_db.get_team(ws_id, expected_org_id=org_id)
    return await team_db.list_team_members(ws_id)


@router.post(
    "/{ws_id}/members",
    summary="Add member to workspace",
    tags=["orgs", "workspaces"],
)
async def add_member(
    org_id: str,
    ws_id: str,
    request: AddTeamMemberRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_team_permission(TeamAction.MANAGE_MEMBERS)),
    ],
) -> TeamMemberResponse:
    if ctx.team_id != ws_id:
        raise HTTPException(403, detail="Team context does not match the target team")
    return await team_db.add_team_member(
        ws_id=ws_id,
        user_id=request.user_id,
        org_id=org_id,
        is_admin=request.is_admin,
        is_billing_manager=request.is_billing_manager,
        invited_by=ctx.user_id,
    )


@router.patch(
    "/{ws_id}/members/{uid}",
    summary="Update workspace member role",
    tags=["orgs", "workspaces"],
)
async def update_member(
    org_id: str,
    ws_id: str,
    uid: str,
    request: UpdateTeamMemberRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_team_permission(TeamAction.MANAGE_MEMBERS)),
    ],
) -> TeamMemberResponse:
    if ctx.team_id != ws_id:
        raise HTTPException(403, detail="Team context does not match the target team")
    return await team_db.update_team_member(
        ws_id=ws_id,
        user_id=uid,
        is_admin=request.is_admin,
        is_billing_manager=request.is_billing_manager,
    )


@router.delete(
    "/{ws_id}/members/{uid}",
    summary="Remove member from workspace",
    tags=["orgs", "workspaces"],
    status_code=204,
)
async def remove_member(
    org_id: str,
    ws_id: str,
    uid: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_team_permission(TeamAction.MANAGE_MEMBERS)),
    ],
) -> None:
    if ctx.team_id != ws_id:
        raise HTTPException(403, detail="Team context does not match the target team")
    await team_db.remove_team_member(ws_id, uid)
