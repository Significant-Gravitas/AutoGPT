"""
Workspace folder API routes.

Folders are a DB-level organizational layer over workspace files; storage paths
are unaffected. Routes spell out their full ``/folders`` paths (matching the
sibling workspace file routes) and are included into the workspace router, which
the app mounts under ``/api/workspace`` — giving ``/api/workspace/folders``.
"""

from typing import Annotated

import fastapi
from autogpt_libs.auth.dependencies import get_user_id, requires_user
from fastapi.responses import Response
from pydantic import BaseModel, Field

from backend.data.workspace import WorkspaceFile, get_or_create_workspace
from backend.data.workspace_folder import (
    UNCHANGED,
    WorkspaceFolder,
    apply_folder_update,
    bulk_move_files_to_folder,
    create_folder,
    delete_folder,
    list_workspace_folders,
)

router = fastapi.APIRouter(
    dependencies=[fastapi.Security(requires_user)],
)


class WorkspaceFolderCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    icon: str | None = None
    # None = create at the workspace root. Reject "" so it cannot bypass the
    # truthiness-based ownership check (422 instead of a foreign-key error).
    parent_id: str | None = Field(None, min_length=1)


class WorkspaceFolderUpdateRequest(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    icon: str | None = None
    # Present-but-null means "move to the root", absent means "leave where it
    # is", which a plain default cannot express — the handler reads
    # ``model_fields_set`` to tell them apart.
    parent_id: str | None = Field(None, min_length=1)


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
async def list_workspace_folders_route(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
) -> WorkspaceFolderListResponse:
    """Every folder in the workspace, flat; nesting is read off ``parent_id``.

    ``file_count`` counts the files directly in each folder, not its subtree.
    """
    workspace = await get_or_create_workspace(user_id)
    folders = await list_workspace_folders(workspace.id)
    return WorkspaceFolderListResponse(folders=folders)


@router.post(
    "/folders",
    summary="Create workspace folder",
    operation_id="createWorkspaceFolder",
    status_code=fastapi.status.HTTP_201_CREATED,
    responses={
        404: {"description": "Parent folder not found"},
        409: {"description": "A folder with this name already exists"},
    },
)
async def create_workspace_folder(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    payload: WorkspaceFolderCreateRequest,
) -> WorkspaceFolder:
    workspace = await get_or_create_workspace(user_id)
    return await create_folder(
        workspace_id=workspace.id,
        name=payload.name,
        icon=payload.icon,
        parent_id=payload.parent_id,
    )


@router.patch(
    "/folders/{folder_id}",
    summary="Update workspace folder",
    operation_id="updateWorkspaceFolder",
    responses={
        400: {"description": "A folder cannot be moved into its own subtree"},
        404: {"description": "Folder or destination not found"},
        409: {"description": "A folder with this name already exists"},
    },
)
async def update_workspace_folder(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    folder_id: str,
    payload: WorkspaceFolderUpdateRequest,
) -> WorkspaceFolder:
    """Rename a folder, change its icon, and/or move it.

    Sending ``parent_id: null`` moves the folder to the workspace root;
    leaving the field out keeps it where it is. Move and rename are applied
    together or not at all, and the new name is checked for a clash against
    the destination.
    """
    workspace = await get_or_create_workspace(user_id)
    return await apply_folder_update(
        folder_id=folder_id,
        workspace_id=workspace.id,
        parent_id=(
            payload.parent_id if "parent_id" in payload.model_fields_set else UNCHANGED
        ),
        name=payload.name,
        icon=payload.icon,
    )


@router.delete(
    "/folders/{folder_id}",
    summary="Delete workspace folder",
    operation_id="deleteWorkspaceFolder",
    status_code=fastapi.status.HTTP_204_NO_CONTENT,
    responses={404: {"description": "Folder not found"}},
)
async def delete_workspace_folder(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    folder_id: str,
) -> Response:
    """Delete a folder and its subfolders; their files return to the root."""
    workspace = await get_or_create_workspace(user_id)
    await delete_folder(folder_id=folder_id, workspace_id=workspace.id)
    return Response(status_code=fastapi.status.HTTP_204_NO_CONTENT)


@router.post(
    "/folders/files/bulk-move",
    summary="Move workspace files to a folder",
    operation_id="bulkMoveWorkspaceFiles",
)
async def bulk_move_workspace_files(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    payload: BulkMoveFilesRequest,
) -> list[WorkspaceFile]:
    workspace = await get_or_create_workspace(user_id)
    return await bulk_move_files_to_folder(
        workspace_id=workspace.id,
        file_ids=payload.file_ids,
        folder_id=payload.folder_id,
    )
