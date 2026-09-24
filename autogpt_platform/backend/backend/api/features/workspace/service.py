"""Workspace writes shared by the internal API and the v2 external API.

Both surfaces must apply the same virus scan, per-file size cap and storage
quota; keeping one implementation is what stops them diverging.
"""

import logging
import os

import fastapi
from fastapi import UploadFile

from backend.api.features.store.exceptions import VirusDetectedError, VirusScanError
from backend.copilot.rate_limit import get_workspace_storage_limit_bytes
from backend.data.workspace import (
    WorkspaceFile,
    get_or_create_workspace,
    get_workspace_total_size,
)
from backend.util.settings import Config
from backend.util.workspace import WorkspaceManager, format_bytes

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
    outage.
    """
    filename = _sanitized_filename(file)
    content = await _read_within_size_limit(file)

    workspace = await get_or_create_workspace(user_id)
    manager = WorkspaceManager(user_id, workspace.id, session_id)
    try:
        workspace_file = await manager.write_file(
            content, filename, overwrite=overwrite, metadata={"origin": "user-upload"}
        )
    except VirusDetectedError as e:
        raise fastapi.HTTPException(status_code=400, detail=str(e)) from e
    except VirusScanError as e:
        raise fastapi.HTTPException(status_code=500, detail=str(e)) from e
    except ValueError as e:
        # write_file raises ValueError for path-conflict, size-limit, and
        # storage-quota cases; map each to its correct HTTP status.
        message = str(e)
        if message.startswith(("File too large", "Storage limit exceeded")):
            raise fastapi.HTTPException(status_code=413, detail=message) from e
        raise fastapi.HTTPException(status_code=409, detail=message) from e

    await _undo_if_over_quota(manager, workspace_file, user_id, workspace.id)
    return workspace_file


def _sanitized_filename(file: UploadFile) -> str:
    """Basename only — a path in the filename must not steer the write."""
    return os.path.basename(file.filename or "upload") or "upload"


async def _read_within_size_limit(file: UploadFile) -> bytes:
    """Read the upload, aborting as soon as it passes the per-file cap."""
    max_file_size_mb = Config().max_file_size_mb
    max_file_bytes = max_file_size_mb * 1024 * 1024
    chunks: list[bytes] = []
    total_size = 0
    while chunk := await file.read(64 * 1024):
        total_size += len(chunk)
        if total_size > max_file_bytes:
            raise fastapi.HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {max_file_size_mb} MB",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _undo_if_over_quota(
    manager: WorkspaceManager,
    workspace_file: WorkspaceFile,
    user_id: str,
    workspace_id: str,
) -> None:
    """Post-write quota check — eliminates the TOCTOU race a pre-check leaves."""
    storage_limit_bytes = await get_workspace_storage_limit_bytes(user_id)
    new_total = await get_workspace_total_size(workspace_id)
    if not storage_limit_bytes or new_total <= storage_limit_bytes:
        return

    try:
        # Route through WorkspaceManager so the storage backend blob is
        # removed too — soft_delete_workspace_file alone leaks the blob.
        await manager.delete_file(workspace_file.id)
    except Exception as e:
        logger.warning(
            f"Failed to delete over-quota file {workspace_file.id} "
            f"in workspace {workspace_id}: {e}"
        )
    raise fastapi.HTTPException(
        status_code=413,
        detail=(
            f"Storage limit exceeded. "
            f"You've used {format_bytes(new_total)} of your "
            f"{format_bytes(storage_limit_bytes)} quota. "
            f"Delete some files or upgrade your plan for more storage."
        ),
    )
