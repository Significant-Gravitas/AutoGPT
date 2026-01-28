"""
Workspace API routes for managing user file storage.
"""

import base64
import logging
from typing import Annotated, Optional

import fastapi
from autogpt_libs.auth.dependencies import get_user_id, requires_user
from fastapi import File, Query, UploadFile
from fastapi.responses import Response
from prisma.enums import WorkspaceFileSource

from backend.data.workspace import (
    count_workspace_files,
    get_or_create_workspace,
    get_workspace,
    get_workspace_file,
    get_workspace_file_by_path,
)
from backend.util.virus_scanner import scan_content_safe
from backend.util.workspace import MAX_FILE_SIZE_BYTES, WorkspaceManager
from backend.util.workspace_storage import get_workspace_storage

from .models import (
    DeleteFileResponse,
    DownloadUrlResponse,
    UploadFileResponse,
    WorkspaceFileInfo,
    WorkspaceFileListResponse,
    WorkspaceInfo,
    WriteFileRequest,
)

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(
    dependencies=[fastapi.Security(requires_user)],
)


def _file_to_info(file) -> WorkspaceFileInfo:
    """Convert database file record to API response model."""
    return WorkspaceFileInfo(
        id=file.id,
        name=file.name,
        path=file.path,
        mime_type=file.mimeType,
        size_bytes=file.sizeBytes,
        checksum=file.checksum,
        source=file.source,
        source_exec_id=file.sourceExecId,
        source_session_id=file.sourceSessionId,
        created_at=file.createdAt,
        updated_at=file.updatedAt,
        metadata=file.metadata if file.metadata else {},
    )


async def _create_file_download_response(file) -> Response:
    """
    Create a download response for a workspace file.

    Handles both local storage (direct streaming) and GCS (signed URL redirect
    with fallback to streaming).
    """
    storage = await get_workspace_storage()

    # For local storage, stream the file directly
    if file.storagePath.startswith("local://"):
        content = await storage.retrieve(file.storagePath)
        return Response(
            content=content,
            media_type=file.mimeType,
            headers={
                "Content-Disposition": f'attachment; filename="{file.name}"',
                "Content-Length": str(len(content)),
            },
        )

    # For GCS, try to redirect to signed URL, fall back to streaming
    try:
        url = await storage.get_download_url(file.storagePath, expires_in=300)
        # If we got back an API path (fallback), stream directly instead
        if url.startswith("/api/"):
            content = await storage.retrieve(file.storagePath)
            return Response(
                content=content,
                media_type=file.mimeType,
                headers={
                    "Content-Disposition": f'attachment; filename="{file.name}"',
                    "Content-Length": str(len(content)),
                },
            )
        return fastapi.responses.RedirectResponse(url=url, status_code=302)
    except Exception:
        # Fall back to streaming directly from GCS
        content = await storage.retrieve(file.storagePath)
        return Response(
            content=content,
            media_type=file.mimeType,
            headers={
                "Content-Disposition": f'attachment; filename="{file.name}"',
                "Content-Length": str(len(content)),
            },
        )


@router.get(
    "",
    summary="Get workspace info",
    response_model=WorkspaceInfo,
)
async def get_workspace_info(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
) -> WorkspaceInfo:
    """
    Get the current user's workspace information.
    Creates workspace if it doesn't exist.
    """
    workspace = await get_or_create_workspace(user_id)
    file_count = await count_workspace_files(workspace.id)

    return WorkspaceInfo(
        id=workspace.id,
        user_id=workspace.userId,
        created_at=workspace.createdAt,
        updated_at=workspace.updatedAt,
        file_count=file_count,
    )


@router.post(
    "/files",
    summary="Upload file to workspace",
    response_model=UploadFileResponse,
)
async def upload_file(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file: UploadFile = File(...),
    path: Annotated[Optional[str], Query()] = None,
    overwrite: Annotated[bool, Query()] = False,
) -> UploadFileResponse:
    """
    Upload a file to the user's workspace.

    - **file**: The file to upload (max 100MB)
    - **path**: Optional virtual path (defaults to "/{filename}")
    - **overwrite**: Whether to overwrite existing file at path
    """
    workspace = await get_or_create_workspace(user_id)
    manager = WorkspaceManager(user_id, workspace.id)

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise fastapi.HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // (1024*1024)}MB",
        )

    # Virus scan
    filename = file.filename or "uploaded_file"
    await scan_content_safe(content, filename=filename)

    # Write file to workspace
    try:
        workspace_file = await manager.write_file(
            content=content,
            filename=filename,
            path=path,
            mime_type=file.content_type,
            source=WorkspaceFileSource.UPLOAD,
            overwrite=overwrite,
        )
    except ValueError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))

    return UploadFileResponse(
        file=_file_to_info(workspace_file),
        message="File uploaded successfully",
    )


@router.post(
    "/files/write",
    summary="Write file content directly",
    response_model=UploadFileResponse,
)
async def write_file_content(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    request: WriteFileRequest,
) -> UploadFileResponse:
    """
    Write file content directly to workspace (for programmatic access).

    - **filename**: Name for the file
    - **content_base64**: Base64-encoded file content
    - **path**: Optional virtual path (defaults to "/{filename}")
    - **mime_type**: Optional MIME type (auto-detected if not provided)
    - **overwrite**: Whether to overwrite existing file at path
    """
    workspace = await get_or_create_workspace(user_id)
    manager = WorkspaceManager(user_id, workspace.id)

    # Decode content
    try:
        content = base64.b64decode(request.content_base64)
    except Exception:
        raise fastapi.HTTPException(
            status_code=400, detail="Invalid base64-encoded content"
        )

    # Check file size
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise fastapi.HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // (1024*1024)}MB",
        )

    # Virus scan
    await scan_content_safe(content, filename=request.filename)

    # Write file to workspace
    try:
        workspace_file = await manager.write_file(
            content=content,
            filename=request.filename,
            path=request.path,
            mime_type=request.mime_type,
            source=WorkspaceFileSource.UPLOAD,
            overwrite=request.overwrite,
        )
    except ValueError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))

    return UploadFileResponse(
        file=_file_to_info(workspace_file),
        message="File written successfully",
    )


@router.get(
    "/files",
    summary="List workspace files",
    response_model=WorkspaceFileListResponse,
)
async def list_files(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    path: Annotated[Optional[str], Query(description="Path prefix filter")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> WorkspaceFileListResponse:
    """
    List files in the user's workspace.

    - **path**: Optional path prefix to filter results
    - **limit**: Maximum number of files to return (1-100)
    - **offset**: Number of files to skip
    """
    workspace = await get_workspace(user_id)
    if workspace is None:
        return WorkspaceFileListResponse(
            files=[],
            total_count=0,
            path_filter=path,
        )

    manager = WorkspaceManager(user_id, workspace.id)
    files = await manager.list_files(path=path, limit=limit, offset=offset)
    total = await manager.get_file_count(path=path)

    return WorkspaceFileListResponse(
        files=[_file_to_info(f) for f in files],
        total_count=total,
        path_filter=path,
    )


@router.get(
    "/files/{file_id}",
    summary="Get file info by ID",
    response_model=WorkspaceFileInfo,
)
async def get_file_info(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file_id: str,
) -> WorkspaceFileInfo:
    """
    Get file metadata by file ID.
    """
    workspace = await get_workspace(user_id)
    if workspace is None:
        raise fastapi.HTTPException(status_code=404, detail="Workspace not found")

    file = await get_workspace_file(file_id, workspace.id)
    if file is None:
        raise fastapi.HTTPException(status_code=404, detail="File not found")

    return _file_to_info(file)


@router.get(
    "/files/{file_id}/download",
    summary="Download file by ID",
)
async def download_file(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file_id: str,
) -> Response:
    """
    Download a file by its ID.

    Returns the file content directly or redirects to a signed URL for GCS.
    """
    workspace = await get_workspace(user_id)
    if workspace is None:
        raise fastapi.HTTPException(status_code=404, detail="Workspace not found")

    file = await get_workspace_file(file_id, workspace.id)
    if file is None:
        raise fastapi.HTTPException(status_code=404, detail="File not found")

    return await _create_file_download_response(file)


@router.get(
    "/files/{file_id}/url",
    summary="Get download URL",
    response_model=DownloadUrlResponse,
)
async def get_download_url(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file_id: str,
    expires_in: Annotated[int, Query(ge=60, le=86400)] = 3600,
) -> DownloadUrlResponse:
    """
    Get a download URL for a file.

    - **expires_in**: URL expiration time in seconds (60-86400, default 3600)
    """
    workspace = await get_workspace(user_id)
    if workspace is None:
        raise fastapi.HTTPException(status_code=404, detail="Workspace not found")

    manager = WorkspaceManager(user_id, workspace.id)

    try:
        url = await manager.get_download_url(file_id, expires_in)
    except FileNotFoundError:
        raise fastapi.HTTPException(status_code=404, detail="File not found")

    return DownloadUrlResponse(
        url=url,
        expires_in_seconds=expires_in,
    )


@router.delete(
    "/files/{file_id}",
    summary="Delete file by ID",
    response_model=DeleteFileResponse,
)
async def delete_file(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file_id: str,
) -> DeleteFileResponse:
    """
    Delete a file from the workspace (soft-delete).
    """
    workspace = await get_workspace(user_id)
    if workspace is None:
        raise fastapi.HTTPException(status_code=404, detail="Workspace not found")

    manager = WorkspaceManager(user_id, workspace.id)
    success = await manager.delete_file(file_id)

    if not success:
        raise fastapi.HTTPException(status_code=404, detail="File not found")

    return DeleteFileResponse(
        success=True,
        file_id=file_id,
        message="File deleted successfully",
    )


# By-path endpoints


@router.get(
    "/files/by-path",
    summary="Get file info by path",
    response_model=WorkspaceFileInfo,
)
async def get_file_by_path(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    path: Annotated[str, Query(description="Virtual file path")],
) -> WorkspaceFileInfo:
    """
    Get file metadata by virtual path.
    """
    workspace = await get_workspace(user_id)
    if workspace is None:
        raise fastapi.HTTPException(status_code=404, detail="Workspace not found")

    file = await get_workspace_file_by_path(workspace.id, path)
    if file is None:
        raise fastapi.HTTPException(status_code=404, detail="File not found")

    return _file_to_info(file)


@router.get(
    "/files/by-path/download",
    summary="Download file by path",
)
async def download_file_by_path(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    path: Annotated[str, Query(description="Virtual file path")],
) -> Response:
    """
    Download a file by its virtual path.
    """
    workspace = await get_workspace(user_id)
    if workspace is None:
        raise fastapi.HTTPException(status_code=404, detail="Workspace not found")

    file = await get_workspace_file_by_path(workspace.id, path)
    if file is None:
        raise fastapi.HTTPException(status_code=404, detail="File not found")

    return await _create_file_download_response(file)


@router.delete(
    "/files/by-path",
    summary="Delete file by path",
    response_model=DeleteFileResponse,
)
async def delete_file_by_path(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    path: Annotated[str, Query(description="Virtual file path")],
) -> DeleteFileResponse:
    """
    Delete a file by its virtual path (soft-delete).
    """
    workspace = await get_workspace(user_id)
    if workspace is None:
        raise fastapi.HTTPException(status_code=404, detail="Workspace not found")

    file = await get_workspace_file_by_path(workspace.id, path)
    if file is None:
        raise fastapi.HTTPException(status_code=404, detail="File not found")

    manager = WorkspaceManager(user_id, workspace.id)
    success = await manager.delete_file(file.id)

    return DeleteFileResponse(
        success=success,
        file_id=file.id,
        message="File deleted successfully" if success else "Failed to delete file",
    )
