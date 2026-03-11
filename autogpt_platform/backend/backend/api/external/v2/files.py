"""
V2 External API - Files Endpoints

Provides file upload, download, listing, metadata, and deletion functionality.
"""

import base64
import logging
import re
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, Query, Security, UploadFile
from fastapi.responses import RedirectResponse, Response
from prisma.enums import APIKeyPermission
from starlette import status

from backend.api.external.middleware import require_permission
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.workspace import (
    count_workspace_files,
    get_workspace,
    get_workspace_file,
    list_workspace_files,
    soft_delete_workspace_file,
)
from backend.util.cloud_storage import get_cloud_storage_handler
from backend.util.settings import Settings
from backend.util.virus_scanner import scan_content_safe
from backend.util.workspace_storage import get_workspace_storage

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from .models import (
    UploadWorkspaceFileResponse,
    WorkspaceFileInfo,
    WorkspaceFileListResponse,
)
from .rate_limit import file_upload_limiter

logger = logging.getLogger(__name__)
settings = Settings()

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
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.DOWNLOAD_FILES)
    ),
) -> WorkspaceFileListResponse:
    """List files in the user's workspace."""
    workspace = await get_workspace(auth.user_id)
    if workspace is None:
        return WorkspaceFileListResponse(
            files=[], page=page, page_size=page_size, total_count=0, total_pages=0
        )

    total_count = await count_workspace_files(workspace.id)
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 0
    offset = (page - 1) * page_size

    files = await list_workspace_files(
        workspace_id=workspace.id,
        limit=page_size,
        offset=offset,
    )

    return WorkspaceFileListResponse(
        files=[
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
        page=page,
        page_size=page_size,
        total_count=total_count,
        total_pages=total_pages,
    )


@file_workspace_router.get(
    path="/{file_id}",
    summary="Get workspace file metadata",
    operation_id="getWorkspaceFileInfo",
)
async def get_file(
    file_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.DOWNLOAD_FILES)
    ),
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
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.UPLOAD_FILES)
    ),
) -> None:
    """Soft-delete a file from the user's workspace."""
    workspace = await get_workspace(auth.user_id)
    if workspace is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workspace not found",
        )

    result = await soft_delete_workspace_file(file_id, workspace.id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File #{file_id} not found",
        )


def _create_file_size_error(size_bytes: int, max_size_mb: int) -> HTTPException:
    """Create standardized file size error response."""
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            f"File size ({size_bytes} bytes) exceeds "
            f"the maximum allowed size of {max_size_mb}MB"
        ),
    )


@file_workspace_router.post(
    path="/upload",
    summary="Upload file to workspace",
    operation_id="uploadWorkspaceFile",
)
async def upload_file(
    file: UploadFile = File(...),
    expiration_hours: int = Query(
        default=24, ge=1, le=48, description="Hours until file expires (1-48)"
    ),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.UPLOAD_FILES)
    ),
) -> UploadWorkspaceFileResponse:
    """
    Upload a file to cloud storage for use with agents.

    Returns a `file_uri` that can be passed to agent graph/node file inputs.
    Uploaded files are virus-scanned before storage.
    """
    file_upload_limiter.check(auth.user_id)

    # Check file size limit
    max_size_mb = settings.config.upload_file_size_limit_mb
    max_size_bytes = max_size_mb * 1024 * 1024

    # Try to get file size from headers first
    if hasattr(file, "size") and file.size is not None and file.size > max_size_bytes:
        raise _create_file_size_error(file.size, max_size_mb)

    # Read file content
    content = await file.read()
    content_size = len(content)

    # Double-check file size after reading
    if content_size > max_size_bytes:
        raise _create_file_size_error(content_size, max_size_mb)

    # Extract file info
    file_name = file.filename or "uploaded_file"
    content_type = file.content_type or "application/octet-stream"

    # Virus scan the content
    await scan_content_safe(content, filename=file_name)

    # Check if cloud storage is configured
    cloud_storage = await get_cloud_storage_handler()
    if not cloud_storage.config.gcs_bucket_name:
        # Fallback to base64 data URI when GCS is not configured
        base64_content = base64.b64encode(content).decode("utf-8")
        data_uri = f"data:{content_type};base64,{base64_content}"

        return UploadWorkspaceFileResponse(
            file_uri=data_uri,
            file_name=file_name,
            size=content_size,
            content_type=content_type,
            expires_in_hours=expiration_hours,
        )

    # Store in cloud storage
    storage_path = await cloud_storage.store_file(
        content=content,
        filename=file_name,
        expiration_hours=expiration_hours,
        user_id=auth.user_id,
    )

    return UploadWorkspaceFileResponse(
        file_uri=storage_path,
        file_name=file_name,
        size=content_size,
        content_type=content_type,
        expires_in_hours=expiration_hours,
    )


# ============================================================================
# Endpoints - Download
# ============================================================================


def _sanitize_filename_for_header(filename: str) -> str:
    """Sanitize filename for Content-Disposition header."""
    sanitized = re.sub(r"[\r\n\x00]", "", filename)
    sanitized = sanitized.replace('"', '\\"')
    try:
        sanitized.encode("ascii")
        return f'attachment; filename="{sanitized}"'
    except UnicodeEncodeError:
        encoded = quote(sanitized, safe="")
        return f"attachment; filename*=UTF-8''{encoded}"


@file_workspace_router.get(
    path="/{file_id}/download",
    summary="Download file from workspace",
    operation_id="getWorkspaceFileDownload",
)
async def download_file(
    file_id: str,
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.DOWNLOAD_FILES)
    ),
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

    storage = await get_workspace_storage()

    # For local storage, stream directly
    if file.storage_path.startswith("local://"):
        content = await storage.retrieve(file.storage_path)
        return Response(
            content=content,
            media_type=file.mime_type,
            headers={
                "Content-Disposition": _sanitize_filename_for_header(file.name),
                "Content-Length": str(len(content)),
            },
        )

    # For cloud storage, try signed URL redirect, fall back to streaming
    try:
        url = await storage.get_download_url(file.storage_path, expires_in=300)
        if url.startswith("/api/"):
            content = await storage.retrieve(file.storage_path)
            return Response(
                content=content,
                media_type=file.mime_type,
                headers={
                    "Content-Disposition": _sanitize_filename_for_header(file.name),
                    "Content-Length": str(len(content)),
                },
            )
        return RedirectResponse(url=url, status_code=302)
    except Exception:
        logger.error(
            f"Failed to get download URL for file {file.id}, falling back to stream",
            exc_info=True,
        )
        content = await storage.retrieve(file.storage_path)
        return Response(
            content=content,
            media_type=file.mime_type,
            headers={
                "Content-Disposition": _sanitize_filename_for_header(file.name),
                "Content-Length": str(len(content)),
            },
        )
