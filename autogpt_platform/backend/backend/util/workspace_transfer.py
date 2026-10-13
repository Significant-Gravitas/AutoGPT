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
import re
import uuid
from typing import TYPE_CHECKING, Optional

from prisma.errors import UniqueViolationError

from backend.copilot.rate_limit import get_workspace_storage_limit_bytes
from backend.data.db_accessors import workspace_db, workspace_folder_db
from backend.data.workspace import WorkspaceFile
from backend.data.workspace_scope import WorkspaceAccessDeniedError
from backend.util.exceptions import NotFoundError
from backend.util.workspace_storage import get_workspace_storage

if TYPE_CHECKING:
    from backend.util.workspace import WorkspaceManager

logger = logging.getLogger(__name__)

# Skill packages are published through ``publish_workspace_skill_file``, which
# enforces the per-owner cap and ownership; a raw path rewrite or byte copy into
# or out of one would bypass both, so neither end of a transfer may sit there.
_SKILLS_TREE = re.compile(r"^(/skills|/experts/[^/]+/skills)(/|$)", re.IGNORECASE)
_SKILLS_TREE_ERROR = (
    "Skill packages cannot be moved or copied as files; use tool:store_skill / "
    "delete_skill to manage them."
)
WORKSPACE_FOLDERS_OWNER_ONLY = (
    "Workspace folders belong to the account owner, so an expert cannot change "
    "which folder a file is in. Open a chat with Otto to organise folders."
)


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
        ValueError: If the destination is occupied and ``overwrite`` is False,
            or either end lies in a skill package
        WorkspaceAccessDeniedError: If the manager's scope may not write the
            source or the destination, or (for an expert) sets a folder
    """
    source, resolved_path = await _resolve_ends(manager, file_id, new_path)
    # A move takes the file away from where it was, so it needs write there.
    manager._authorize_file(source, write=True)
    await _validate_folder(manager, folder_id)

    db = workspace_db()
    target_folder_id = folder_id if folder_id is not None else source.folder_id

    if resolved_path == source.path and target_folder_id == source.folder_id:
        return source

    occupant = await _occupant(manager, resolved_path, file_id, overwrite)

    try:
        # The occupant is retired in the same transaction as the move, so a
        # failed move leaves it in place.
        moved = await db.update_workspace_file_location(
            file_id=file_id,
            workspace_id=manager.workspace_id,
            name=_basename(resolved_path),
            path=resolved_path,
            folder_id=target_folder_id,
            replace_file_id=occupant.id if occupant else None,
        )
    except UniqueViolationError:
        raise ValueError(
            f"File already exists at path: {resolved_path} (concurrent write conflict)"
        ) from None
    except NotFoundError:
        # The folder went away between the check above and the write.
        raise ValueError(f"Workspace folder not found: {target_folder_id}") from None

    if moved is None:
        raise FileNotFoundError(f"File not found: {file_id}")
    await _dispose(manager, occupant)

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
            source file's folder (see below for an expert's copy).
        overwrite: Replace an existing file at the destination

    Returns:
        The newly created WorkspaceFile

    Raises:
        FileNotFoundError: If the source file does not exist
        ValueError: If the destination is occupied and ``overwrite`` is False,
            either end lies in a skill package, or the copy would exceed the
            user's storage quota
        WorkspaceAccessDeniedError: If the manager's scope may not read the
            source or write the destination, or (for an expert) sets a folder

    The content is not re-scanned for viruses: these exact bytes were already
    scanned by ``WorkspaceManager.write_file`` when they first entered the
    workspace, and a copy cannot change them.

    An expert's copy lands at the workspace root rather than in the source's
    folder: folders are the owner's, and the copy is the expert's file.
    """
    source, resolved_path = await _resolve_ends(manager, file_id, new_path)
    manager._authorize_file(source)
    await _validate_folder(manager, folder_id)

    db = workspace_db()
    if folder_id is None and manager.scope is None:
        folder_id = source.folder_id
    if resolved_path == source.path:
        raise ValueError(
            f"Cannot copy a file onto itself: {resolved_path}. "
            f"Provide a different new_path."
        )

    occupant = await _occupant(manager, resolved_path, file_id, overwrite)
    await _check_quota(manager, source.size_bytes, occupant)

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
            folder_id=folder_id,
            # Retired in the same transaction as the insert; its blob goes only
            # after that commits, so a failed copy loses nothing.
            replace_file_id=occupant.id if occupant else None,
        )
    except UniqueViolationError:
        await _discard_blob(storage_path)
        raise ValueError(
            f"File already exists at path: {resolved_path} (concurrent write conflict)"
        ) from None
    except Exception:
        await _discard_blob(storage_path)
        raise
    await _dispose(manager, occupant)

    logger.info(
        f"Copied workspace file {file_id} to {copied.id} at {resolved_path} "
        f"in workspace {manager.workspace_id}, size={source.size_bytes} bytes"
    )
    _reindex(manager, copied)
    return copied


async def _resolve_ends(
    manager: "WorkspaceManager", file_id: str, new_path: str
) -> tuple[WorkspaceFile, str]:
    """The source file and the resolved destination, both checked.

    The destination is authorised for write here; the source's grant depends
    on the operation (a move writes it, a copy only reads it), so the caller
    checks that one.
    """
    source = await workspace_db().get_workspace_file(file_id, manager.workspace_id)
    if source is None:
        raise FileNotFoundError(f"File not found: {file_id}")
    resolved_path = manager._resolve_path(new_path)
    if _SKILLS_TREE.match(source.path) or _SKILLS_TREE.match(resolved_path):
        raise ValueError(_SKILLS_TREE_ERROR)
    manager._authorize_path(resolved_path, write=True)
    return source, resolved_path


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
    if manager.scope is not None:
        raise WorkspaceAccessDeniedError(WORKSPACE_FOLDERS_OWNER_ONLY)
    try:
        await workspace_folder_db().get_workspace_folder(
            folder_id, manager.workspace_id
        )
    except NotFoundError:
        raise ValueError(
            f"Workspace folder not found: {folder_id}. "
            "Use tool:create_workspace_folder to create it first, or omit folder_id."
        ) from None


async def _occupant(
    manager: "WorkspaceManager",
    resolved_path: str,
    source_file_id: str,
    overwrite: bool,
) -> Optional[WorkspaceFile]:
    """The other live file at ``resolved_path``, refused unless overwriting.

    Nothing is deleted here: the caller retires it in the same transaction
    that writes the replacement, then :func:`_dispose` removes its blob.
    """
    existing = await workspace_db().get_workspace_file_by_path(
        manager.workspace_id, resolved_path
    )
    if existing is None or existing.id == source_file_id:
        return None
    if not overwrite:
        raise ValueError(
            f"File already exists at path: {resolved_path}. "
            f"Pass overwrite=true to replace it, or choose a different new_path."
        )
    return existing


async def _dispose(
    manager: "WorkspaceManager", occupant: Optional[WorkspaceFile]
) -> None:
    """Best-effort cleanup of a replaced file once its row is retired."""
    if occupant is None:
        return
    await _discard_blob(occupant.storage_path)
    try:
        from backend.api.features.workspace.embeddings import (
            delete_workspace_file_embedding,
        )

        await delete_workspace_file_embedding(
            file_id=occupant.id, user_id=manager.user_id
        )
    except Exception as e:
        logger.warning(f"Failed to delete file embedding for {occupant.id}: {e}")


async def _check_quota(
    manager: "WorkspaceManager",
    added_bytes: int,
    occupant: Optional[WorkspaceFile],
) -> None:
    """Reject a copy that would push the user past their storage quota.

    A file the copy replaces is not counted, so overwriting a file with one
    the same size or smaller is not refused near the cap.
    """
    storage_limit, current_usage = await asyncio.gather(
        get_workspace_storage_limit_bytes(manager.user_id),
        workspace_db().get_workspace_total_size(manager.workspace_id),
    )
    if occupant is not None:
        current_usage = max(0, current_usage - occupant.size_bytes)

    if storage_limit > 0 and current_usage + added_bytes > storage_limit:
        # Lazy: backend.util.workspace imports this module at the top.
        from backend.util.workspace import format_bytes

        raise ValueError(
            f"Storage limit exceeded. "
            f"You've used {format_bytes(current_usage)} of your "
            f"{format_bytes(storage_limit)} quota. "
            f"Delete some files or upgrade your plan for more storage."
        )


async def _discard_blob(storage_path: str) -> None:
    """Best-effort delete of a blob no live record points at."""
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
