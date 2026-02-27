"""
Workspace API routes for managing user file storage.
"""

import logging
import re
import uuid
from typing import Annotated
from urllib.parse import quote

import fastapi
import pydantic
from autogpt_libs.auth.dependencies import get_user_id, requires_user
from fastapi import File, UploadFile
from fastapi.responses import Response

from backend.data.workspace import (
    WorkspaceFile,
    get_or_create_workspace,
    get_workspace,
    get_workspace_file,
)
from backend.util.workspace import WorkspaceManager
from backend.util.workspace_storage import get_workspace_storage


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
    summary="Delete a workspace file",
)
async def delete_workspace_file(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file_id: str,
) -> dict[str, bool]:
    """
    Soft-delete a workspace file and remove it from storage.

    Used when a user clears a file input in the builder.
    """
    workspace = await get_workspace(user_id)
    if workspace is None:
        raise fastapi.HTTPException(status_code=404, detail="Workspace not found")

    manager = WorkspaceManager(user_id, workspace.id)
    deleted = await manager.delete_file(file_id)
    if not deleted:
        raise fastapi.HTTPException(status_code=404, detail="File not found")

    return {"deleted": True}


class WorkspaceUploadResponse(pydantic.BaseModel):
    file_uri: str
    file_name: str
    size: int
    content_type: str


@router.post(
    "/files/upload",
    summary="Upload a file to the workspace",
)
async def upload_workspace_file(
    user_id: Annotated[str, fastapi.Security(get_user_id)],
    file: UploadFile = File(...),
) -> WorkspaceUploadResponse:
    """
    Upload a file to the user's workspace for use in builder graph inputs.

    Returns a workspace:// URI that can be stored in graph node inputs
    instead of embedding full file content as base64.
    """
    workspace = await get_or_create_workspace(user_id)
    content = await file.read()

    filename = file.filename or "upload"
    content_type = file.content_type or "application/octet-stream"
    file_id = str(uuid.uuid4())
    path = f"/builder-uploads/{file_id}/{filename}"

    manager = WorkspaceManager(user_id, workspace.id)
    workspace_file = await manager.write_file(
        content=content,
        filename=filename,
        path=path,
        mime_type=content_type,
    )

    return WorkspaceUploadResponse(
        file_uri=f"workspace://{workspace_file.id}",
        file_name=workspace_file.name,
        size=workspace_file.size_bytes,
        content_type=workspace_file.mime_type,
    )
