"""Workspace management API routes (nested under /api/orgs/{org_id}/workspaces)."""

import functools
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated, ParamSpec, TypeVar

from autogpt_libs.auth import get_request_context, requires_org_permission
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction, check_org_permission
from fastapi import APIRouter, HTTPException, Security

from backend.data.tenancy import live_org_context_barrier
from backend.util.exceptions import NotAuthorizedError, NotFoundError

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

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _rejects_as_400(
    handler: Callable[_P, Awaitable[_R]],
) -> Callable[_P, Awaitable[_R]]:
    """Surface team_db's user-triggerable rejections as HTTP 400s.

    team_db raises ValueError for rejections the caller can fix (changing the
    default team's join policy, deleting/leaving the default team, removing the
    last admin, adding a non-org-member, self-joining a non-OPEN team). Without
    translation these reach the client as 500s. Applied only to the mutating
    handlers whose db calls raise ValueError, so a genuine bug elsewhere still
    surfaces as a 500 rather than being masked as a 400.

    NotFoundError/NotAuthorizedError subclass ValueError but carry their own
    404/403 mappings, so they are re-raised untouched rather than flattened.
    """

    @functools.wraps(handler)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        try:
            return await handler(*args, **kwargs)
        except (NotFoundError, NotAuthorizedError):
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return wrapper


@asynccontextmanager
async def _live_org_view(
    ctx: RequestContext,
    org_id: str,
    action: OrgAction = OrgAction.VIEW_RESOURCES,
) -> AsyncIterator[RequestContext]:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    async with live_org_context_barrier(ctx.user_id, org_id) as live_ctx:
        if live_ctx is None or not check_org_permission(live_ctx, action):
            raise HTTPException(403, detail="Organization access was revoked")
        yield live_ctx


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
        require_live_permission=True,
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
    async with _live_org_view(ctx, org_id, OrgAction.VIEW_ORG) as live_ctx:
        return await team_db.list_teams(
            org_id,
            ctx.user_id,
            can_manage_workspaces=check_org_permission(
                live_ctx, OrgAction.MANAGE_WORKSPACES
            ),
        )


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
    async with _live_org_view(ctx, org_id) as live_ctx:
        return await team_db.get_team_for_viewer(
            ws_id,
            org_id,
            ctx.user_id,
            can_manage_workspaces=check_org_permission(
                live_ctx, OrgAction.MANAGE_WORKSPACES
            ),
        )


@router.patch(
    "/{ws_id}",
    summary="Update workspace",
    tags=["orgs", "workspaces"],
)
@_rejects_as_400
async def update_team(
    org_id: str,
    ws_id: str,
    request: UpdateTeamRequest,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> TeamResponse:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    return await team_db.update_team(
        ws_id,
        {
            "name": request.name,
            "description": request.description,
            "joinPolicy": request.join_policy,
        },
        org_id=org_id,
        actor_user_id=ctx.user_id,
    )


@router.delete(
    "/{ws_id}",
    summary="Delete workspace",
    tags=["orgs", "workspaces"],
    status_code=204,
)
@_rejects_as_400
async def delete_team(
    org_id: str,
    ws_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_WORKSPACES)),
    ],
) -> None:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    await team_db.delete_team(
        ws_id,
        org_id=org_id,
        actor_user_id=ctx.user_id,
    )


@router.post(
    "/{ws_id}/join",
    summary="Self-join open workspace",
    tags=["orgs", "workspaces"],
)
@_rejects_as_400
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
@_rejects_as_400
async def leave_team(
    org_id: str,
    ws_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> None:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    await team_db.leave_team(ws_id, ctx.user_id, org_id)


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
    async with _live_org_view(ctx, org_id) as live_ctx:
        team = await team_db.get_team_for_viewer(
            ws_id,
            org_id,
            ctx.user_id,
            can_manage_workspaces=check_org_permission(
                live_ctx, OrgAction.MANAGE_WORKSPACES
            ),
        )
        if team.join_policy != "OPEN" and not team.is_member:
            raise HTTPException(403, detail="Join this workspace to view its members")
        return await team_db.list_team_members(ws_id)


@router.post(
    "/{ws_id}/members",
    summary="Add member to workspace",
    tags=["orgs", "workspaces"],
)
@_rejects_as_400
async def add_member(
    org_id: str,
    ws_id: str,
    request: AddTeamMemberRequest,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> TeamMemberResponse:
    if request.user_id == ctx.user_id:
        raise HTTPException(400, detail="Use the workspace join action to add yourself")
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
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
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> TeamMemberResponse:
    if uid == ctx.user_id:
        raise HTTPException(400, detail="You cannot change your own workspace role")
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    return await team_db.update_team_member(
        ws_id=ws_id,
        user_id=uid,
        is_admin=request.is_admin,
        is_billing_manager=request.is_billing_manager,
        org_id=org_id,
        requesting_user_id=ctx.user_id,
    )


@router.delete(
    "/{ws_id}/members/{uid}",
    summary="Remove member from workspace",
    tags=["orgs", "workspaces"],
    status_code=204,
)
@_rejects_as_400
async def remove_member(
    org_id: str,
    ws_id: str,
    uid: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> None:
    if uid == ctx.user_id:
        raise HTTPException(400, detail="Use the workspace leave action")
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    await team_db.remove_team_member(
        ws_id,
        uid,
        org_id=org_id,
        requesting_user_id=ctx.user_id,
    )
