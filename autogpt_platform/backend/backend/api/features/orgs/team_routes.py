"""Workspace management API routes (nested under /api/orgs/{org_id}/workspaces)."""

import functools
from collections.abc import Awaitable, Callable
from typing import Annotated, ParamSpec, TypeVar

from autogpt_libs.auth import get_request_context, requires_org_permission
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction, check_org_permission
from fastapi import APIRouter, HTTPException, Security

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


async def _authorize_team_management(
    ctx: RequestContext, org_id: str, ws_id: str
) -> None:
    """Authorize a management action against the target team from the URL path.

    Independent of the caller's active team (X-Team-Id): allowed when the caller
    administers the target team directly, or holds org-level MANAGE_WORKSPACES
    over the org that owns it. The target team must belong to the path org.
    """
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    await team_db.get_team(ws_id, expected_org_id=org_id)
    if check_org_permission(ctx, OrgAction.MANAGE_WORKSPACES):
        return
    if await team_db.is_team_admin(ws_id, ctx.user_id):
        return
    raise HTTPException(
        403, detail="Must be a team admin or org admin to manage this team"
    )


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
@_rejects_as_400
async def update_team(
    org_id: str,
    ws_id: str,
    request: UpdateTeamRequest,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> TeamResponse:
    await _authorize_team_management(ctx, org_id, ws_id)
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
@_rejects_as_400
async def delete_team(
    org_id: str,
    ws_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_WORKSPACES)),
    ],
) -> None:
    # Deliberate asymmetry with _authorize_team_management: team admins may
    # rename a team and manage its members, but deletion is destructive and
    # stays org-admin-only (MANAGE_WORKSPACES).
    #
    # MANAGE_WORKSPACES only proves the caller administers their *active*
    # org (ctx.org_id); without this guard an admin of org A could delete
    # a workspace in org B (the permission passes, and get_team only checks
    # the workspace is in the path org). Mirror every sibling route.
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    await team_db.get_team(ws_id, expected_org_id=org_id)
    await team_db.delete_team(ws_id)


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
@_rejects_as_400
async def add_member(
    org_id: str,
    ws_id: str,
    request: AddTeamMemberRequest,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> TeamMemberResponse:
    await _authorize_team_management(ctx, org_id, ws_id)
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
    await _authorize_team_management(ctx, org_id, ws_id)
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
@_rejects_as_400
async def remove_member(
    org_id: str,
    ws_id: str,
    uid: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> None:
    await _authorize_team_management(ctx, org_id, ws_id)
    await team_db.remove_team_member(ws_id, uid)
