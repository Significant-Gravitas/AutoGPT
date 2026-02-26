"""
Workspace API routes for managing user file storage.

Provides endpoints for uploading, downloading, listing, and deleting files
in a user's workspace.  Uploads are session-scoped by default and support
per-user storage quotas.
"""

import logging
import re
from typing import Annotated
from urllib.parse import quote

import fastapi
import pydantic
from autogpt_libs.auth.dependencies import get_user_id, requires_user
from fastapi.responses import Response

from backend.data.workspace import WorkspaceFile, get_workspace, get_workspace_file
from backend.util.settings import Config
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_storage import get_workspace_storage

logger = logging.getLogger(__name__)

# ---------- Allowed MIME types for user uploads ----------
# Phase 1: text, PDF, CSV/spreadsheets, images.  Video/audio is future.
ALLOWED_UPLOAD_MIME_TYPES: set[str] = {
    # Text
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/html",
    "text/xml",
    "application/json",
    "application/xml",
    # PDF
    "application/pdf",
    # Spreadsheets
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-excel",  # .xls
    # Documents
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/msword",  # .doc
    # Images
    "image/png",
    "image/jpeg",
    "image/gif",
    "image/webp",
    "image/svg+xml",
}

# Extensions allowed when MIME is generic application/octet-stream
_EXTENSION_ALLOWLIST: set[str] = {
    ".txt", ".md", ".csv", ".json", ".xml", ".html",
    ".pdf",
    ".xlsx", ".xls", ".docx", ".doc",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
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


def _is_allowed_upload(content_type: str | None, filename: str) -> bool:
    """Check whether a file's MIME type or extension is permitted for upload."""
    if content_type and content_type in ALLOWED_UPLOAD_MIME_TYPES:
        return True
    # Fall back to extension check (handles octet-stream or missing MIME)
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in _EXTENSION_ALLOWLIST


# ---------- Response models ----------

class FileInfoResponse(pydantic.BaseModel):
    file_id: str
    name: str
    path: str
    mime_type: str
    size_bytes: int
    created_at: str


class UploadFileResponse(pydantic.BaseModel):
    file_id: str
    name: str
    path: str
    mime_type: str
    size_bytes: int


class ListFilesResponse(pydantic.BaseModel):
    files: list[FileInfoResponse]
    total_count: int


class StorageUsageResponse(pydantic.BaseModel):
    used_bytes: int
    quota_bytes: int
    used_pct: float
    file_count: int


# ---------- Router ----------

router = fastapi.APIRouter(
    dependencies=[fastapi.Security(requires_user)],
)


# ---------- Helpers ----------

async def _get_manager(user_id: str, session_id: str | None = None) -> WorkspaceManager:
    """Build a WorkspaceManager, creating the workspace if needed."""
    from backend.data.db_accessors import workspace_db

    workspace = await workspace_db().get_or_create_workspace(user_id)
    return WorkspaceManager(user_id, workspace.id, session_id)


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


# ---------- Endpoints ----------


@router.post(
    "/files/upload",
    summary="Upload a file to the workspace",
    status_code=201,
)
async def upload_file(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    session_id: str = fastapi.Form(..., description="Chat session ID to scope this upload"),
    file: fastapi.UploadFile = fastapi.File(...),
) -> UploadFileResponse:
    """
    Upload a file to the user's workspace, scoped to a chat session.

    The file is stored under ``/sessions/{session_id}/uploads/{filename}``
    and is immediately accessible to the CoPilot agent for that session.
    Files persist across runs within the same thread.

    Enforces per-user storage quotas and file-type restrictions.
    """
    filename = file.filename or "untitled"

    # Validate file type
    if not _is_allowed_upload(file.content_type, filename):
        raise fastapi.HTTPException(
            status_code=415,
            detail=(
                f"File type not supported: {file.content_type or 'unknown'}. "
                "Allowed types: text, PDF, CSV/spreadsheets, images."
            ),
        )

    # Read content (enforcing max size during read)
    config = Config()
    max_size = config.max_file_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise fastapi.HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). "
            f"Maximum size is {config.max_file_size_mb} MB.",
        )

    manager = await _get_manager(user_id, session_id)

    # Quota check
    within_quota, current_usage, quota_bytes = await manager.check_quota(len(content))
    if not within_quota:
        logger.warning(
            f"User {user_id} hit storage quota: "
            f"{current_usage / 1024 / 1024:.1f} MB / {quota_bytes / 1024 / 1024:.1f} MB"
        )
        raise fastapi.HTTPException(
            status_code=413,
            detail=(
                f"Storage quota exceeded. Using {current_usage / 1024 / 1024:.1f} MB "
                f"of {quota_bytes / 1024 / 1024:.1f} MB. "
                "Please delete some files and try again."
            ),
        )

    # Write file via WorkspaceManager (handles storage, DB, virus scan, checksums)
    # Path: /sessions/{session_id}/uploads/{filename}
    upload_path = f"/uploads/{filename}"
    try:
        workspace_file = await manager.write_file(
            content=content,
            filename=filename,
            path=upload_path,
            mime_type=file.content_type,
            overwrite=True,
        )
    except ValueError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e))

    logger.info(
        f"User {user_id} uploaded {filename} ({len(content)} bytes) "
        f"to session {session_id}"
    )

    return UploadFileResponse(
        file_id=workspace_file.id,
        name=workspace_file.name,
        path=workspace_file.path,
        mime_type=workspace_file.mime_type,
        size_bytes=workspace_file.size_bytes,
    )


@router.get(
    "/files",
    summary="List files in workspace",
)
async def list_files(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    session_id: str | None = fastapi.Query(
        default=None, description="Scope to files in this session"
    ),
    limit: int = fastapi.Query(default=50, ge=1, le=100),
    offset: int = fastapi.Query(default=0, ge=0),
) -> ListFilesResponse:
    """
    List files in the user's workspace.

    If ``session_id`` is provided, only files in that session are returned.
    Otherwise, all workspace files are listed.
    """
    manager = await _get_manager(user_id, session_id)
    files = await manager.list_files(
        limit=limit,
        offset=offset,
        include_all_sessions=session_id is None,
    )
    total = await manager.get_file_count(include_all_sessions=session_id is None)

    return ListFilesResponse(
        files=[
            FileInfoResponse(
                file_id=f.id,
                name=f.name,
                path=f.path,
                mime_type=f.mime_type,
                size_bytes=f.size_bytes,
                created_at=f.created_at.isoformat(),
            )
            for f in files
        ],
        total_count=total,
    )


@router.get(
    "/usage",
    summary="Get workspace storage usage and quota",
)
async def get_usage(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
) -> StorageUsageResponse:
    """
    Get the user's workspace storage usage vs their quota.

    Useful for showing remaining capacity in the UI and triggering
    alerts when approaching the limit.
    """
    manager = await _get_manager(user_id)
    current_usage = await manager.get_total_usage_bytes()
    config = Config()
    quota_bytes = config.user_storage_quota_mb * 1024 * 1024
    file_count = await manager.get_file_count(include_all_sessions=True)

    pct = (current_usage / quota_bytes * 100) if quota_bytes > 0 else 0.0

    if pct >= 90:
        logger.warning(
            f"User {user_id} storage usage at {pct:.1f}%: "
            f"{current_usage / 1024 / 1024:.1f} MB / {quota_bytes / 1024 / 1024:.1f} MB"
        )

    return StorageUsageResponse(
        used_bytes=current_usage,
        quota_bytes=quota_bytes,
        used_pct=round(pct, 2),
        file_count=file_count,
    )


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


@router.delete(
    "/files/{file_id}",
    summary="Delete a file from workspace",
    status_code=204,
)
async def delete_file(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file_id: str,
) -> Response:
    """
    Soft-delete a file from the user's workspace.

    The storage is reclaimed and the file is no longer accessible.
    """
    manager = await _get_manager(user_id)
    deleted = await manager.delete_file(file_id)
    if not deleted:
        raise fastapi.HTTPException(status_code=404, detail="File not found")

    logger.info(f"User {user_id} deleted file {file_id}")
    return Response(status_code=204)
