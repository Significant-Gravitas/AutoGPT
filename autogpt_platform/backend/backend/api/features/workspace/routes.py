"""
Workspace API routes for managing user file storage.
"""

import logging
import re
from typing import Annotated
from urllib.parse import quote

import fastapi
from autogpt_libs.auth.dependencies import get_user_id, requires_user
from fastapi.responses import Response

from backend.data.workspace import get_workspace, get_workspace_file
from backend.util.workspace_storage import get_workspace_storage


def _sanitize_filename_for_header(filename: str) -> str:
    """
    Sanitize filename for Content-Disposition header to prevent header injection.

    Removes/replaces characters that could break the header or inject new headers.
    Uses RFC5987 encoding for non-ASCII characters.
    """
    # Remove CR, LF, and null bytes (header injection prevention)
    sanitized = re.sub(r'[\r\n\x00]', '', filename)
    # Escape quotes
    sanitized = sanitized.replace('"', '\\"')
    # For non-ASCII, use RFC5987 filename* parameter
    # Check if filename has non-ASCII characters
    try:
        sanitized.encode('ascii')
        return f'attachment; filename="{sanitized}"'
    except UnicodeEncodeError:
        # Use RFC5987 encoding for UTF-8 filenames
        encoded = quote(filename, safe='')
        return f"attachment; filename*=UTF-8''{encoded}"

logger = logging.getLogger(__name__)

router = fastapi.APIRouter(
    dependencies=[fastapi.Security(requires_user)],
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
                "Content-Disposition": _sanitize_filename_for_header(file.name),
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
                    "Content-Disposition": _sanitize_filename_for_header(file.name),
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
                "Content-Disposition": _sanitize_filename_for_header(file.name),
                "Content-Length": str(len(content)),
            },
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
