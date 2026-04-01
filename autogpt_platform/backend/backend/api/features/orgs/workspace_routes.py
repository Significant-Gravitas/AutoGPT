"""Workspace management API routes (nested under /api/orgs/{org_id}/workspaces)."""

from typing import Annotated

from autogpt_libs.auth import (
    get_request_context,
    requires_org_permission,
    requires_workspace_permission,
)
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction, WorkspaceAction
from fastapi import APIRouter, HTTPException, Security

from . import workspace_db as ws_db
from .workspace_model import (
    AddWorkspaceMemberRequest,
    CreateWorkspaceRequest,
    UpdateWorkspaceMemberRequest,
    UpdateWorkspaceRequest,
    WorkspaceMemberResponse,
    WorkspaceResponse,
)

router = APIRouter()


@router.post(
    "",
    summary="Create workspace",
    tags=["orgs", "workspaces"],
)
async def create_workspace(
    org_id: str,
    request: CreateWorkspaceRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.CREATE_WORKSPACES)),
    ],
) -> WorkspaceResponse:
    result = await ws_db.create_workspace(
        org_id=org_id,
        name=request.name,
        user_id=ctx.user_id,
        description=request.description,
        join_policy=request.joinPolicy,
    )
    return WorkspaceResponse(**result)


@router.get(
    "",
    summary="List workspaces",
    tags=["orgs", "workspaces"],
)
async def list_workspaces(
    org_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> list[WorkspaceResponse]:
    if ctx.org_id != org_id:
        raise HTTPException(403, detail="Not a member of this organization")
    results = await ws_db.list_workspaces(org_id, ctx.user_id)
    return [WorkspaceResponse(**r) for r in results]


@router.get(
    "/{ws_id}",
    summary="Get workspace details",
    tags=["orgs", "workspaces"],
)
async def get_workspace(
    org_id: str,
    ws_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> WorkspaceResponse:
    result = await ws_db.get_workspace(ws_id)
    return WorkspaceResponse(**result)


@router.patch(
    "/{ws_id}",
    summary="Update workspace",
    tags=["orgs", "workspaces"],
)
async def update_workspace(
    org_id: str,
    ws_id: str,
    request: UpdateWorkspaceRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_workspace_permission(WorkspaceAction.MANAGE_SETTINGS)),
    ],
) -> WorkspaceResponse:
    result = await ws_db.update_workspace(
        ws_id,
        {
            "name": request.name,
            "description": request.description,
            "joinPolicy": request.joinPolicy,
        },
    )
    return WorkspaceResponse(**result)


@router.delete(
    "/{ws_id}",
    summary="Delete workspace",
    tags=["orgs", "workspaces"],
    status_code=204,
)
async def delete_workspace(
    org_id: str,
    ws_id: str,
    ctx: Annotated[
        RequestContext,
        Security(requires_org_permission(OrgAction.MANAGE_WORKSPACES)),
    ],
) -> None:
    await ws_db.delete_workspace(ws_id)


@router.post(
    "/{ws_id}/join",
    summary="Self-join open workspace",
    tags=["orgs", "workspaces"],
)
async def join_workspace(
    org_id: str,
    ws_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> WorkspaceResponse:
    result = await ws_db.join_workspace(ws_id, ctx.user_id, org_id)
    return WorkspaceResponse(**result)


@router.post(
    "/{ws_id}/leave",
    summary="Leave workspace",
    tags=["orgs", "workspaces"],
    status_code=204,
)
async def leave_workspace(
    org_id: str,
    ws_id: str,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> None:
    await ws_db.leave_workspace(ws_id, ctx.user_id)


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
) -> list[WorkspaceMemberResponse]:
    results = await ws_db.list_workspace_members(ws_id)
    return [WorkspaceMemberResponse(**r) for r in results]


@router.post(
    "/{ws_id}/members",
    summary="Add member to workspace",
    tags=["orgs", "workspaces"],
)
async def add_member(
    org_id: str,
    ws_id: str,
    request: AddWorkspaceMemberRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_workspace_permission(WorkspaceAction.MANAGE_MEMBERS)),
    ],
) -> WorkspaceMemberResponse:
    result = await ws_db.add_workspace_member(
        ws_id=ws_id,
        user_id=request.userId,
        org_id=org_id,
        is_admin=request.isAdmin,
        is_billing_manager=request.isBillingManager,
        invited_by=ctx.user_id,
    )
    return WorkspaceMemberResponse(**result)


@router.patch(
    "/{ws_id}/members/{uid}",
    summary="Update workspace member role",
    tags=["orgs", "workspaces"],
)
async def update_member(
    org_id: str,
    ws_id: str,
    uid: str,
    request: UpdateWorkspaceMemberRequest,
    ctx: Annotated[
        RequestContext,
        Security(requires_workspace_permission(WorkspaceAction.MANAGE_MEMBERS)),
    ],
) -> WorkspaceMemberResponse:
    result = await ws_db.update_workspace_member(
        ws_id=ws_id,
        user_id=uid,
        is_admin=request.isAdmin,
        is_billing_manager=request.isBillingManager,
    )
    return WorkspaceMemberResponse(**result)


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
        Security(requires_workspace_permission(WorkspaceAction.MANAGE_MEMBERS)),
    ],
) -> None:
    await ws_db.remove_workspace_member(ws_id, uid)
