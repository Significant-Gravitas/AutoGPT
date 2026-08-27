"""
Workspace folder API routes.

Folders are a DB-level organizational layer over workspace files; storage paths
are unaffected. Routes spell out their full ``/folders`` paths (matching the
sibling workspace file routes) and are included into the workspace router, which
the app mounts under ``/api/workspace`` — giving ``/api/workspace/folders``.
"""

from typing import Annotated, Awaitable, Callable, TypeVar

import fastapi
from autogpt_libs.auth.dependencies import get_request_context, requires_user
from autogpt_libs.auth.models import RequestContext
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.data.db_accessors import LiveResourceAccessRevoked, live_resource_lease
from backend.data.tenancy import ResourceAccess
from backend.data.workspace import WorkspaceFile, get_or_create_workspace
from backend.data.workspace_folder import (
    WorkspaceFolder,
    bulk_move_files_to_folder,
    create_folder,
    delete_folder,
    list_folders,
    update_folder,
)

router = fastapi.APIRouter(
    dependencies=[fastapi.Security(requires_user)],
)
T = TypeVar("T")


async def _run_scoped(
    ctx: RequestContext,
    access: ResourceAccess,
    action: Callable[[], Awaitable[T]],
) -> T:
    try:
        async with live_resource_lease(
            ctx.user_id, ctx.org_id, ctx.team_id, access
        ) as guard:
            if not guard:
                raise LiveResourceAccessRevoked("workspace_access_revoked")
            return await guard.run(action())
    except LiveResourceAccessRevoked as exc:
        raise fastapi.HTTPException(status_code=404, detail="Folder not found") from exc


class WorkspaceFolderCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    icon: str | None = None


class WorkspaceFolderUpdateRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    icon: str | None = None


class BulkMoveFilesRequest(BaseModel):
    file_ids: list[str]
    # None = move to root. Reject "" so it can't bypass the truthiness-based
    # ownership check and hit a foreign-key error (returns 422 instead).
    folder_id: str | None = Field(None, min_length=1)


class WorkspaceFolderListResponse(BaseModel):
    folders: list[WorkspaceFolder]


@router.get(
    "/folders",
    summary="List workspace folders",
    operation_id="listWorkspaceFolders",
)
async def list_workspace_folders(
    ctx: Annotated[RequestContext, fastapi.Security(get_request_context)],
) -> WorkspaceFolderListResponse:
    workspace = await get_or_create_workspace(ctx.user_id)
    folders = await _run_scoped(
        ctx,
        "view",
        lambda: list_folders(workspace.id, ctx.org_id, ctx.team_id),
    )
    return WorkspaceFolderListResponse(folders=folders)


@router.post(
    "/folders",
    summary="Create workspace folder",
    operation_id="createWorkspaceFolder",
    status_code=fastapi.status.HTTP_201_CREATED,
    responses={409: {"description": "A folder with this name already exists"}},
)
async def create_workspace_folder(
    ctx: Annotated[RequestContext, fastapi.Security(get_request_context)],
    payload: WorkspaceFolderCreateRequest,
) -> WorkspaceFolder:
    workspace = await get_or_create_workspace(ctx.user_id)
    return await _run_scoped(
        ctx,
        "create",
        lambda: create_folder(
            workspace_id=workspace.id,
            name=payload.name,
            icon=payload.icon,
            organization_id=ctx.org_id,
            team_id=ctx.team_id,
        ),
    )


@router.patch(
    "/folders/{folder_id}",
    summary="Update workspace folder",
    operation_id="updateWorkspaceFolder",
    responses={
        404: {"description": "Folder not found"},
        409: {"description": "A folder with this name already exists"},
    },
)
async def update_workspace_folder(
    ctx: Annotated[RequestContext, fastapi.Security(get_request_context)],
    folder_id: str,
    payload: WorkspaceFolderUpdateRequest,
) -> WorkspaceFolder:
    workspace = await get_or_create_workspace(ctx.user_id)
    return await _run_scoped(
        ctx,
        "create",
        lambda: update_folder(
            folder_id=folder_id,
            workspace_id=workspace.id,
            name=payload.name,
            icon=payload.icon,
            organization_id=ctx.org_id,
            team_id=ctx.team_id,
        ),
    )


@router.delete(
    "/folders/{folder_id}",
    summary="Delete workspace folder",
    operation_id="deleteWorkspaceFolder",
    status_code=fastapi.status.HTTP_204_NO_CONTENT,
    responses={404: {"description": "Folder not found"}},
)
async def delete_workspace_folder(
    ctx: Annotated[RequestContext, fastapi.Security(get_request_context)],
    folder_id: str,
) -> Response:
    workspace = await get_or_create_workspace(ctx.user_id)
    await _run_scoped(
        ctx,
        "create",
        lambda: delete_folder(
            folder_id=folder_id,
            workspace_id=workspace.id,
            organization_id=ctx.org_id,
            team_id=ctx.team_id,
        ),
    )
    return Response(status_code=fastapi.status.HTTP_204_NO_CONTENT)


@router.post(
    "/folders/files/bulk-move",
    summary="Move workspace files to a folder",
    operation_id="bulkMoveWorkspaceFiles",
)
async def bulk_move_workspace_files(
    ctx: Annotated[RequestContext, fastapi.Security(get_request_context)],
    payload: BulkMoveFilesRequest,
) -> list[WorkspaceFile]:
    workspace = await get_or_create_workspace(ctx.user_id)
    return await _run_scoped(
        ctx,
        "create",
        lambda: bulk_move_files_to_folder(
            workspace_id=workspace.id,
            file_ids=payload.file_ids,
            folder_id=payload.folder_id,
            organization_id=ctx.org_id,
            team_id=ctx.team_id,
        ),
    )
