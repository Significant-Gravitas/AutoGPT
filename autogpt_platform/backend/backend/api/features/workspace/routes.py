"""
Workspace API routes for managing user file storage.
"""

import logging
import os
import re
from typing import Annotated
from urllib.parse import quote

import fastapi
from autogpt_libs.auth.dependencies import get_user_id, requires_user
from fastapi import Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from backend.data.workspace import (
    WorkspaceFile,
    count_workspace_files,
    get_or_create_workspace,
    get_workspace,
    get_workspace_file,
    get_workspace_total_size,
    soft_delete_workspace_file,
)
from backend.util.settings import Config
from backend.util.virus_scanner import scan_content_safe
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_storage import get_workspace_storage

# Allowed file extensions for upload, grouped by category.
_ALLOWED_EXTENSIONS: set[str] = {
    # Documents
    ".pdf",
    ".doc",
    ".docx",
    ".txt",
    ".rtf",
    ".odt",
    # Spreadsheets
    ".csv",
    ".xls",
    ".xlsx",
    ".ods",
    # Images
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".bmp",
    ".ico",
    # Code / config
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".css",
    ".md",
    ".sh",
    ".bat",
    # Archives
    ".zip",
    ".tar",
    ".gz",
    ".7z",
    ".rar",
    # Audio / Video
    ".mp3",
    ".wav",
    ".ogg",
    ".mp4",
    ".webm",
    ".mov",
    ".avi",
}


def _sanitize_filename_for_header(filename: str) -> str:
    """
    Sanitize filename for Content-Disposition header to prevent header injection.

    Removes/replaces characters that could break the header or inject new headers.
    Uses RFC5987 encoding for non-ASCII characters.
    """
    # Remove CR, LF, and null bytes (header injection prevention)
    sanitized = re.sub(r"[\r\n\x00]", "", filename)
    # Escape quotes
    sanitized = sanitized.replace('"', '\\"')
    # For non-ASCII, use RFC5987 filename* parameter
    # Check if filename has non-ASCII characters
    try:
        sanitized.encode("ascii")
        return f'attachment; filename="{sanitized}"'
    except UnicodeEncodeError:
        # Use RFC5987 encoding for UTF-8 filenames
        encoded = quote(sanitized, safe="")
        return f"attachment; filename*=UTF-8''{encoded}"


logger = logging.getLogger(__name__)

router = fastapi.APIRouter(
    dependencies=[fastapi.Security(requires_user)],
)


def _create_streaming_response(content: bytes, file: WorkspaceFile) -> Response:
    """Create a streaming response for file content."""
    return Response(
        content=content,
        media_type=file.mime_type,
        headers={
            "Content-Disposition": _sanitize_filename_for_header(file.name),
            "Content-Length": str(len(content)),
        },
    )


async def _create_file_download_response(file: WorkspaceFile) -> Response:
    """
    Create a download response for a workspace file.

    Handles both local storage (direct streaming) and GCS (signed URL redirect
    with fallback to streaming).
    """
    storage = await get_workspace_storage()

    # For local storage, stream the file directly
    if file.storage_path.startswith("local://"):
        content = await storage.retrieve(file.storage_path)
        return _create_streaming_response(content, file)

    # For GCS, try to redirect to signed URL, fall back to streaming
    try:
        url = await storage.get_download_url(file.storage_path, expires_in=300)
        # If we got back an API path (fallback), stream directly instead
        if url.startswith("/api/"):
            content = await storage.retrieve(file.storage_path)
            return _create_streaming_response(content, file)
        return fastapi.responses.RedirectResponse(url=url, status_code=302)
    except Exception as e:
        # Log the signed URL failure with context
        logger.error(
            f"Failed to get signed URL for file {file.id} "
            f"(storagePath={file.storage_path}): {e}",
            exc_info=True,
        )
        # Fall back to streaming directly from GCS
        try:
            content = await storage.retrieve(file.storage_path)
            return _create_streaming_response(content, file)
        except Exception as fallback_error:
            logger.error(
                f"Fallback streaming also failed for file {file.id} "
                f"(storagePath={file.storage_path}): {fallback_error}",
                exc_info=True,
            )
            raise


class UploadFileResponse(BaseModel):
    file_id: str
    name: str
    path: str
    mime_type: str
    size_bytes: int


class StorageUsageResponse(BaseModel):
    used_bytes: int
    limit_bytes: int
    used_percent: float
    file_count: int


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


@router.post(
    "/files/upload",
    summary="Upload file to workspace",
)
async def upload_file(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file: UploadFile,
    session_id: str | None = Query(default=None),
) -> UploadFileResponse:
    """
    Upload a file to the user's workspace.

    Files are stored in session-scoped paths when session_id is provided,
    so the agent's session-scoped tools can discover them automatically.
    """
    config = Config()

    # Sanitize filename — strip any directory components
    filename = os.path.basename(file.filename or "upload") or "upload"

    # Validate file extension against allowlist
    _, ext = os.path.splitext(filename)
    if ext.lower() not in _ALLOWED_EXTENSIONS:
        raise fastapi.HTTPException(
            status_code=415,
            detail=f"File type '{ext}' is not supported",
        )

    # Read file content with early abort on size limit
    max_file_bytes = config.max_file_size_mb * 1024 * 1024
    chunks: list[bytes] = []
    total_size = 0
    while chunk := await file.read(64 * 1024):  # 64KB chunks
        total_size += len(chunk)
        if total_size > max_file_bytes:
            raise fastapi.HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {config.max_file_size_mb} MB",
            )
        chunks.append(chunk)
    content = b"".join(chunks)

    # Get or create workspace
    workspace = await get_or_create_workspace(user_id)

    # Pre-write storage cap check (soft check — final enforcement is post-write)
    storage_limit_bytes = config.max_workspace_storage_mb * 1024 * 1024
    current_usage = await get_workspace_total_size(workspace.id)
    if current_usage + len(content) > storage_limit_bytes:
        used_percent = (current_usage / storage_limit_bytes) * 100
        raise fastapi.HTTPException(
            status_code=413,
            detail={
                "message": "Storage limit exceeded",
                "used_bytes": current_usage,
                "limit_bytes": storage_limit_bytes,
                "used_percent": round(used_percent, 1),
            },
        )

    # Warn at 80% usage
    usage_ratio = (current_usage + len(content)) / storage_limit_bytes
    if usage_ratio >= 0.8:
        logger.warning(
            f"User {user_id} workspace storage at {usage_ratio * 100:.1f}% "
            f"({current_usage + len(content)} / {storage_limit_bytes} bytes)"
        )

    # Virus scan
    await scan_content_safe(content, filename=filename)

    # Write file via WorkspaceManager
    manager = WorkspaceManager(user_id, workspace.id, session_id)
    workspace_file = await manager.write_file(content, filename)

    # Post-write storage check — eliminates TOCTOU race on the quota.
    # If a concurrent upload pushed us over the limit, undo this write.
    new_total = await get_workspace_total_size(workspace.id)
    if new_total > storage_limit_bytes:
        await soft_delete_workspace_file(workspace_file.id, workspace.id)
        raise fastapi.HTTPException(
            status_code=413,
            detail={
                "message": "Storage limit exceeded (concurrent upload)",
                "used_bytes": new_total,
                "limit_bytes": storage_limit_bytes,
            },
        )

    return UploadFileResponse(
        file_id=workspace_file.id,
        name=workspace_file.name,
        path=workspace_file.path,
        mime_type=workspace_file.mime_type,
        size_bytes=workspace_file.size_bytes,
    )


@router.get(
    "/storage/usage",
    summary="Get workspace storage usage",
)
async def get_storage_usage(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
) -> StorageUsageResponse:
    """
    Get storage usage information for the user's workspace.
    """
    config = Config()
    workspace = await get_or_create_workspace(user_id)

    used_bytes = await get_workspace_total_size(workspace.id)
    file_count = await count_workspace_files(workspace.id)
    limit_bytes = config.max_workspace_storage_mb * 1024 * 1024

    return StorageUsageResponse(
        used_bytes=used_bytes,
        limit_bytes=limit_bytes,
        used_percent=round((used_bytes / limit_bytes) * 100, 1) if limit_bytes else 0,
        file_count=file_count,
    )
