"""
V2 External API - Files Endpoints

Provides file upload, download, listing, metadata, and deletion functionality.
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, Query, Security, UploadFile
from fastapi.responses import Response
from prisma.enums import APIKeyPermission
from starlette import status

from backend.api.features.workspace.routes import create_file_download_response
from backend.api.features.workspace.service import store_workspace_upload
from backend.data.workspace import (
    count_workspace_files,
    get_workspace,
    get_workspace_file,
    list_workspace_files,
)
from backend.util.workspace import WorkspaceManager

from .models import UploadWorkspaceFileResponse, WorkspaceFileInfo
from .pagination import Page, PageRequest, page_request
from .rate_limit import enforce, file_upload_limiter
from .tenancy import TenantContext, require_permission

logger = logging.getLogger(__name__)

file_workspace_router = APIRouter(tags=["files"])


# ============================================================================
# Endpoints
# ============================================================================


@file_workspace_router.get(
    path="",
    summary="List workspace files",
    operation_id="listWorkspaceFiles",
)
async def list_files(
    page: PageRequest = Depends(page_request),
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_FILES)),
) -> Page[WorkspaceFileInfo]:
    """List files in the user's workspace."""
    workspace = await get_workspace(auth.user_id)
    if workspace is None:
        return page.paged([], total_count=0)

    total_count = await count_workspace_files(workspace.id)
    files = await list_workspace_files(
        workspace_id=workspace.id,
        limit=page.limit,
        offset=(page.page - 1) * page.limit,
    )

    return page.paged(
        [
            WorkspaceFileInfo(
                id=f.id,
                name=f.name,
                path=f.path,
                mime_type=f.mime_type,
                size_bytes=f.size_bytes,
                created_at=f.created_at,
                updated_at=f.updated_at,
            )
            for f in files
        ],
        total_count=total_count,
    )


@file_workspace_router.get(
    path="/{file_id}",
    summary="Get workspace file metadata",
    operation_id="getWorkspaceFileInfo",
)
async def get_file(
    file_id: str,
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_FILES)),
) -> WorkspaceFileInfo:
    """Get metadata for a specific file in the user's workspace."""
    workspace = await get_workspace(auth.user_id)
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    file = await get_workspace_file(file_id, workspace.id)
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File #{file_id} not found",
        )

    return WorkspaceFileInfo(
        id=file.id,
        name=file.name,
        path=file.path,
        mime_type=file.mime_type,
        size_bytes=file.size_bytes,
        created_at=file.created_at,
        updated_at=file.updated_at,
    )


@file_workspace_router.delete(
    path="/{file_id}",
    summary="Delete file from workspace",
    operation_id="deleteWorkspaceFile",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_file(
    file_id: str,
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_FILES)),
) -> None:
    """Delete a file from the user's workspace, with its stored content."""
    workspace = await get_workspace(auth.user_id)
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    # The manager also removes the stored blob and its search entry, as the
    # internal route does; soft-deleting the row alone freed no storage.
    deleted = await WorkspaceManager(auth.user_id, workspace.id).delete_file(file_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File #{file_id} not found",
        )


@file_workspace_router.post(
    path="/upload",
    summary="Upload file to workspace",
    operation_id="uploadWorkspaceFile",
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    response: Response,
    file: UploadFile = File(...),
    overwrite: bool = Query(
        default=False, description="Replace an existing file with the same path"
    ),
    auth: TenantContext = Security(require_permission(APIKeyPermission.WRITE_FILES)),
) -> UploadWorkspaceFileResponse:
    """
    Upload a file to the user's workspace.

    The file is listed by `GET /files` and can be passed to agent file inputs
    as the returned `file_uri`. Uploads are virus-scanned before storage and
    count against the account's storage quota.

    **Rate limit:** 20 requests per 5 minutes per user.
    """
    await enforce(file_upload_limiter, auth.user_id, response)

    workspace_file = await store_workspace_upload(
        auth.user_id, file, overwrite=overwrite
    )

    return UploadWorkspaceFileResponse(
        id=workspace_file.id,
        name=workspace_file.name,
        path=workspace_file.path,
        mime_type=workspace_file.mime_type,
        size_bytes=workspace_file.size_bytes,
        created_at=workspace_file.created_at,
        updated_at=workspace_file.updated_at,
        file_uri=f"workspace://{workspace_file.id}#{workspace_file.mime_type}",
    )


# ============================================================================
# Endpoints - Download
# ============================================================================


@file_workspace_router.get(
    path="/{file_id}/download",
    summary="Download file from workspace",
    operation_id="getWorkspaceFileDownload",
)
async def download_file(
    file_id: str,
    auth: TenantContext = Security(require_permission(APIKeyPermission.READ_FILES)),
) -> Response:
    """Download a file from the user's workspace."""
    workspace = await get_workspace(auth.user_id)
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    file = await get_workspace_file(file_id, workspace.id)
    if file is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File #{file_id} not found",
        )

    return await create_file_download_response(file)
