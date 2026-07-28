"""
Server-side move and copy for workspace files.

Both operations keep file bytes on the server: a move is a pure metadata
rewrite (``storagePath`` is untouched), and a copy delegates to the storage
backend's native ``copy`` primitive rather than streaming content through this
process. Split out of :mod:`backend.util.workspace` to keep that module
focused on the read/write/delete lifecycle.
"""

import asyncio
import logging
import os
import uuid
from typing import TYPE_CHECKING, Optional

from prisma.errors import UniqueViolationError

from backend.copilot.rate_limit import get_workspace_storage_limit_bytes
from backend.data.db_accessors import workspace_db, workspace_folder_db
from backend.data.workspace import WorkspaceFile
from backend.util.exceptions import NotFoundError
from backend.util.workspace_storage import get_workspace_storage

if TYPE_CHECKING:
    from backend.util.workspace import WorkspaceManager

logger = logging.getLogger(__name__)


async def move_file(
    manager: "WorkspaceManager",
    file_id: str,
    new_path: str,
    folder_id: Optional[str] = None,
    overwrite: bool = False,
) -> WorkspaceFile:
    """
    Relocate a workspace file to ``new_path`` without moving its bytes.

    Args:
        manager: The owning WorkspaceManager (supplies workspace/session scope)
        file_id: ID of the file to move
        new_path: Destination virtual path, resolved against the session scope
        folder_id: Folder membership for the moved file. ``None`` keeps the
            file's current folder.
        overwrite: Replace an existing file at the destination

    Returns:
        The updated WorkspaceFile

    Raises:
        FileNotFoundError: If the source file does not exist
        ValueError: If the destination is occupied and ``overwrite`` is False
    """
    db = workspace_db()
    source = await db.get_workspace_file(file_id, manager.workspace_id)
    if source is None:
        raise FileNotFoundError(f"File not found: {file_id}")

    await _validate_folder(manager, folder_id)

    resolved_path = manager._resolve_path(new_path)
    target_folder_id = folder_id if folder_id is not None else source.folder_id

    if resolved_path == source.path and target_folder_id == source.folder_id:
        return source

    await _clear_destination(manager, resolved_path, file_id, overwrite)

    try:
        moved = await db.update_workspace_file_location(
            file_id=file_id,
            workspace_id=manager.workspace_id,
            name=_basename(resolved_path),
            path=resolved_path,
            folder_id=target_folder_id,
        )
    except UniqueViolationError:
        raise ValueError(
            f"File already exists at path: {resolved_path} (concurrent write conflict)"
        ) from None

    if moved is None:
        raise FileNotFoundError(f"File not found: {file_id}")

    logger.info(
        f"Moved workspace file {file_id} from {source.path} to {resolved_path} "
        f"in workspace {manager.workspace_id}"
    )
    _reindex(manager, moved)
    return moved


async def copy_file(
    manager: "WorkspaceManager",
    file_id: str,
    new_path: str,
    folder_id: Optional[str] = None,
    overwrite: bool = False,
) -> WorkspaceFile:
    """
    Duplicate a workspace file to ``new_path`` using a server-side byte copy.

    Args:
        manager: The owning WorkspaceManager (supplies workspace/session scope)
        file_id: ID of the file to copy
        new_path: Destination virtual path, resolved against the session scope
        folder_id: Folder membership for the copy. ``None`` inherits the
            source file's folder.
        overwrite: Replace an existing file at the destination

    Returns:
        The newly created WorkspaceFile

    Raises:
        FileNotFoundError: If the source file does not exist
        ValueError: If the destination is occupied and ``overwrite`` is False,
            or if the copy would exceed the user's storage quota

    The content is not re-scanned for viruses: these exact bytes were already
    scanned by ``WorkspaceManager.write_file`` when they first entered the
    workspace, and a copy cannot change them.
    """
    db = workspace_db()
    source = await db.get_workspace_file(file_id, manager.workspace_id)
    if source is None:
        raise FileNotFoundError(f"File not found: {file_id}")

    await _validate_folder(manager, folder_id)

    resolved_path = manager._resolve_path(new_path)
    if resolved_path == source.path:
        raise ValueError(
            f"Cannot copy a file onto itself: {resolved_path}. "
            f"Provide a different new_path."
        )

    await _check_quota(manager, source.size_bytes, resolved_path, overwrite)
    await _clear_destination(manager, resolved_path, file_id, overwrite)

    new_name = _basename(resolved_path)
    new_file_id = str(uuid.uuid4())
    storage = await get_workspace_storage()
    storage_path = await storage.copy(
        source.storage_path,
        manager.workspace_id,
        new_file_id,
        new_name,
    )

    try:
        copied = await db.create_workspace_file(
            workspace_id=manager.workspace_id,
            file_id=new_file_id,
            name=new_name,
            path=resolved_path,
            storage_path=storage_path,
            mime_type=source.mime_type,
            size_bytes=source.size_bytes,
            checksum=source.checksum,
            metadata=source.metadata,
            folder_id=folder_id if folder_id is not None else source.folder_id,
        )
    except UniqueViolationError:
        await _discard_blob(storage_path)
        raise ValueError(
            f"File already exists at path: {resolved_path} (concurrent write conflict)"
        ) from None
    except Exception:
        await _discard_blob(storage_path)
        raise

    logger.info(
        f"Copied workspace file {file_id} to {copied.id} at {resolved_path} "
        f"in workspace {manager.workspace_id}, size={source.size_bytes} bytes"
    )
    _reindex(manager, copied)
    return copied


def _basename(path: str) -> str:
    """Filename component of a virtual path, falling back to the whole path."""
    return os.path.basename(path.rstrip("/")) or path


async def _validate_folder(
    manager: "WorkspaceManager", folder_id: Optional[str]
) -> None:
    """Reject a caller-supplied ``folder_id`` that isn't in this workspace.

    Mirrors the ownership check the folder tools already do, so a foreign or
    stale folder id can't silently mis-file the transferred file (and surfaces
    a clean message instead of a raw foreign-key error).
    """
    if folder_id is None:
        return
    try:
        await workspace_folder_db().get_workspace_folder(
            folder_id, manager.workspace_id
        )
    except NotFoundError:
        raise ValueError(
            f"Workspace folder not found: {folder_id}. "
            f"Use create_workspace_folder to create it first, or omit folder_id."
        ) from None


async def _clear_destination(
    manager: "WorkspaceManager",
    resolved_path: str,
    source_file_id: str,
    overwrite: bool,
) -> None:
    """Ensure ``resolved_path`` is free, deleting the occupant if allowed."""
    db = workspace_db()
    existing = await db.get_workspace_file_by_path(manager.workspace_id, resolved_path)
    if existing is None or existing.id == source_file_id:
        return
    if not overwrite:
        raise ValueError(
            f"File already exists at path: {resolved_path}. "
            f"Pass overwrite=true to replace it, or choose a different new_path."
        )
    await manager.delete_file(existing.id)


async def _check_quota(
    manager: "WorkspaceManager",
    added_bytes: int,
    resolved_path: str,
    overwrite: bool,
) -> None:
    """Reject a copy that would push the user past their storage quota."""
    db = workspace_db()
    storage_limit, current_usage = await asyncio.gather(
        get_workspace_storage_limit_bytes(manager.user_id),
        db.get_workspace_total_size(manager.workspace_id),
    )
    if overwrite:
        existing = await db.get_workspace_file_by_path(
            manager.workspace_id, resolved_path
        )
        if existing is not None:
            current_usage = max(0, current_usage - existing.size_bytes)

    if storage_limit > 0 and current_usage + added_bytes > storage_limit:
        from backend.util.workspace import format_bytes

        raise ValueError(
            f"Storage limit exceeded. "
            f"You've used {format_bytes(current_usage)} of your "
            f"{format_bytes(storage_limit)} quota. "
            f"Delete some files or upgrade your plan for more storage."
        )


async def _discard_blob(storage_path: str) -> None:
    """Best-effort cleanup of a blob whose DB record was never created."""
    try:
        storage = await get_workspace_storage()
        await storage.delete(storage_path)
    except Exception as e:
        logger.warning(f"Failed to clean up orphaned storage file: {e}")


def _reindex(manager: "WorkspaceManager", file: WorkspaceFile) -> None:
    """Refresh the hybrid-search entry, which is keyed on name and path."""
    try:
        from backend.api.features.workspace.embeddings import (
            schedule_workspace_file_embedding,
        )

        schedule_workspace_file_embedding(
            file_id=file.id,
            user_id=manager.user_id,
            name=file.name,
            path=file.path,
        )
    except Exception as e:
        # Search quality is never worth failing a move/copy over.
        logger.warning(f"Failed to schedule file embedding for {file.id}: {e}")
