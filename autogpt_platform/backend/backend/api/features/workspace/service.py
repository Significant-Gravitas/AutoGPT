"""Workspace writes shared by the internal API and the v2 external API.

Both surfaces must apply the same virus scan, per-file size cap and storage
quota; keeping one implementation is what stops them diverging.
"""

import logging
import os

import fastapi
from fastapi import UploadFile

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.data.workspace import WorkspaceFile, get_or_create_workspace
from backend.util.settings import Config
from backend.util.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


async def store_workspace_upload(
    user_id: str,
    file: UploadFile,
    *,
    session_id: str | None = None,
    overwrite: bool = False,
) -> WorkspaceFile:
    """Write an uploaded file into the user's persistent workspace.

    Raises `fastapi.HTTPException` with the status each failure earns: 400 for
    a virus, 409 for a name conflict, 413 for size or quota, 500 for a scanner
    outage or a storage failure, whose details stay out of the response.
    """
    filename = _sanitized_filename(file)
    content = await read_within_size_limit(file, Config().max_file_size_mb)

    workspace = await get_or_create_workspace(user_id)
    manager = WorkspaceManager(user_id, workspace.id, session_id)
    try:
        return await manager.write_file(
            content, filename, overwrite=overwrite, metadata={"origin": "user-upload"}
        )
    except VirusDetectedError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e)) from e
    except VirusScanError as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e)) from e
    except OSError as e:
        logger.exception("Failed to store uploaded workspace file")
        raise fastapi.HTTPException(
            status_code=500, detail="Unable to store file. Please try again."
        ) from e
    except ValueError as e:
        # write_file raises ValueError for path-conflict, size-limit, and
        # storage-quota cases; map each to its correct HTTP status.
        message = str(e)
        if message.startswith(("File too large", "Storage limit exceeded")):
            raise fastapi.HTTPException(status_code=413, detail=message) from e
        raise fastapi.HTTPException(status_code=409, detail=message) from e


def _sanitized_filename(file: UploadFile) -> str:
    """Basename only — a path in the filename must not steer the write."""
    return os.path.basename(file.filename or "upload") or "upload"


async def read_within_size_limit(file: UploadFile, max_size_mb: int) -> bytes:
    """Read the upload, refusing it with a 413 as soon as it passes the cap, or
    before reading at all when the multipart parser already counted its size."""
    max_bytes = max_size_mb * 1024 * 1024
    too_large = fastapi.HTTPException(
        status_code=413, detail=f"File exceeds maximum size of {max_size_mb} MB"
    )
    if file.size is not None and file.size > max_bytes:
        raise too_large
    chunks: list[bytes] = []
    total_size = 0
    while chunk := await file.read(64 * 1024):
        total_size += len(chunk)
        if total_size > max_bytes:
            raise too_large
        chunks.append(chunk)
    return b"".join(chunks)
