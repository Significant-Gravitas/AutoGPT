"""
Database CRUD operations for User Workspace.

This module provides functions for managing user workspaces and workspace files.
"""

import logging
import re
from datetime import datetime, timezone
from typing import Optional

import pydantic
from prisma.errors import UniqueViolationError
from prisma.models import UserWorkspace, UserWorkspaceFile
from prisma.types import UserWorkspaceFileWhereInput

from backend.util.json import SafeJson

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
)

logger = logging.getLogger(__name__)


class Workspace(pydantic.BaseModel):
    """Pydantic model for UserWorkspace, safe for RPC transport."""

    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    @staticmethod
    def from_db(workspace: "UserWorkspace") -> "Workspace":
        return Workspace(
            id=workspace.id,
            user_id=workspace.userId,
            created_at=workspace.createdAt,
            updated_at=workspace.updatedAt,
        )


class WorkspaceFile(pydantic.BaseModel):
    """Pydantic model for UserWorkspaceFile, safe for RPC transport."""

    id: str
    workspace_id: str
    created_at: datetime
    updated_at: datetime
    name: str
    path: str
    storage_path: str
    mime_type: str
    size_bytes: int
    checksum: Optional[str] = None
    is_deleted: bool = False
    deleted_at: Optional[datetime] = None
    metadata: dict = pydantic.Field(default_factory=dict)

    @staticmethod
    def from_db(file: "UserWorkspaceFile") -> "WorkspaceFile":
        return WorkspaceFile(
            id=file.id,
            workspace_id=file.workspaceId,
            created_at=file.createdAt,
            updated_at=file.updatedAt,
            name=file.name,
            path=file.path,
            storage_path=file.storagePath,
            mime_type=file.mimeType,
            size_bytes=file.sizeBytes,
            checksum=file.checksum,
            is_deleted=file.isDeleted,
            deleted_at=file.deletedAt,
            metadata=file.metadata if isinstance(file.metadata, dict) else {},
        )


async def get_or_create_workspace(user_id: str) -> Workspace:
    """
    Get user's workspace, creating one if it doesn't exist.

    Uses upsert to atomically handle concurrent creation attempts.

    Args:
        user_id: The user's ID

    Returns:
        Workspace instance
    """
    try:
        workspace = await UserWorkspace.prisma().upsert(
            where={"userId": user_id},
            data={
                "create": {"userId": user_id},
                "update": {},  # No-op update; workspace already exists
            },
        )
    except UniqueViolationError:
        # Defense-in-depth: should not happen with upsert, but handle gracefully
        workspace = await UserWorkspace.prisma().find_unique(where={"userId": user_id})
        if workspace is None:
            raise

    return Workspace.from_db(workspace)


async def get_workspace(user_id: str) -> Optional[Workspace]:
    """
    Get user's workspace if it exists.

    Args:
        user_id: The user's ID

    Returns:
        Workspace instance or None
    """
    workspace = await UserWorkspace.prisma().find_unique(where={"userId": user_id})
    return Workspace.from_db(workspace) if workspace else None


async def create_workspace_file(
    workspace_id: str,
    file_id: str,
    name: str,
    path: str,
    storage_path: str,
    mime_type: str,
    size_bytes: int,
    checksum: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> WorkspaceFile:
    """
    Create a new workspace file record.

    Raises ``UniqueViolationError`` if a record with the same
    ``(workspaceId, path)`` already exists.  The caller
    (``WorkspaceManager._persist_db_record``) relies on this to trigger
    its delete-old-file-then-retry flow, which cleans up the old storage
    blob before re-creating the DB record.  Using ``upsert`` here would
    silently overwrite ``storagePath`` and orphan the old blob in storage.

    Args:
        workspace_id: The workspace ID
        file_id: The file ID (same as used in storage path for consistency)
        name: User-visible filename
        path: Virtual path (e.g., "/documents/report.pdf")
        storage_path: Actual storage path (GCS or local)
        mime_type: MIME type of the file
        size_bytes: File size in bytes
        checksum: Optional SHA256 checksum
        metadata: Optional additional metadata

    Returns:
        Created WorkspaceFile instance
    """
    # Normalize path to start with /
    if not path.startswith("/"):
        path = f"/{path}"

    file = await UserWorkspaceFile.prisma().create(
        data={
            "id": file_id,
            "workspaceId": workspace_id,
            "name": name,
            "path": path,
            "storagePath": storage_path,
            "mimeType": mime_type,
            "sizeBytes": size_bytes,
            "checksum": checksum,
            "metadata": SafeJson(metadata or {}),
        }
    )

    logger.info(
        f"Created workspace file {file.id} at path {path} "
        f"in workspace {workspace_id}"
    )
    return WorkspaceFile.from_db(file)


async def get_workspace_file(
    file_id: str,
    workspace_id: str,
) -> Optional[WorkspaceFile]:
    """
    Get a workspace file by ID.

    Args:
        file_id: The file ID
        workspace_id: Workspace ID for scoping (required)

    Returns:
        WorkspaceFile instance or None
    """
    where_clause: UserWorkspaceFileWhereInput = {
        "id": file_id,
        "isDeleted": False,
        "workspaceId": workspace_id,
    }

    file = await UserWorkspaceFile.prisma().find_first(where=where_clause)
    return WorkspaceFile.from_db(file) if file else None


async def get_workspace_file_by_path(
    workspace_id: str,
    path: str,
) -> Optional[WorkspaceFile]:
    """
    Get a workspace file by its virtual path.

    Args:
        workspace_id: The workspace ID
        path: Virtual path

    Returns:
        WorkspaceFile instance or None
    """
    # Normalize path
    if not path.startswith("/"):
        path = f"/{path}"

    file = await UserWorkspaceFile.prisma().find_first(
        where={
            "workspaceId": workspace_id,
            "path": path,
            "isDeleted": False,
        }
    )
    return WorkspaceFile.from_db(file) if file else None


async def list_workspace_files(
    workspace_id: str,
    path_prefix: Optional[str] = None,
    include_deleted: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[WorkspaceFile]:
    """
    List files in a workspace.

    Args:
        workspace_id: The workspace ID
        path_prefix: Optional path prefix to filter (e.g., "/documents/")
        include_deleted: Whether to include soft-deleted files
        limit: Maximum number of files to return
        offset: Number of files to skip

    Returns:
        List of WorkspaceFile instances
    """
    where_clause: UserWorkspaceFileWhereInput = {"workspaceId": workspace_id}

    if not include_deleted:
        where_clause["isDeleted"] = False

    if path_prefix:
        # Normalize prefix
        if not path_prefix.startswith("/"):
            path_prefix = f"/{path_prefix}"
        where_clause["path"] = {"startswith": path_prefix}

    files = await UserWorkspaceFile.prisma().find_many(
        where=where_clause,
        order={"createdAt": "desc"},
        take=limit,
        skip=offset,
    )
    return [WorkspaceFile.from_db(f) for f in files]


async def count_workspace_files(
    workspace_id: str,
    path_prefix: Optional[str] = None,
    include_deleted: bool = False,
) -> int:
    """
    Count files in a workspace.

    Args:
        workspace_id: The workspace ID
        path_prefix: Optional path prefix to filter (e.g., "/sessions/abc123/")
        include_deleted: Whether to include soft-deleted files

    Returns:
        Number of files
    """
    where_clause: UserWorkspaceFileWhereInput = {"workspaceId": workspace_id}
    if not include_deleted:
        where_clause["isDeleted"] = False

    if path_prefix:
        # Normalize prefix
        if not path_prefix.startswith("/"):
            path_prefix = f"/{path_prefix}"
        where_clause["path"] = {"startswith": path_prefix}

    return await UserWorkspaceFile.prisma().count(where=where_clause)


async def soft_delete_workspace_file(
    file_id: str,
    workspace_id: str,
) -> Optional[WorkspaceFile]:
    """
    Soft-delete a workspace file.

    The path is modified to include a deletion timestamp to free up the original
    path for new files while preserving the record for potential recovery.

    Args:
        file_id: The file ID
        workspace_id: Workspace ID for scoping (required)

    Returns:
        Updated WorkspaceFile instance or None if not found
    """
    # First verify the file exists and belongs to workspace
    file = await get_workspace_file(file_id, workspace_id)
    if file is None:
        return None

    deleted_at = datetime.now(timezone.utc)
    # Modify path to free up the unique constraint for new files at original path
    # Format: {original_path}__deleted__{timestamp}
    deleted_path = f"{file.path}__deleted__{int(deleted_at.timestamp())}"

    updated = await UserWorkspaceFile.prisma().update(
        where={"id": file_id},
        data={
            "isDeleted": True,
            "deletedAt": deleted_at,
            "path": deleted_path,
        },
    )

    logger.info(f"Soft-deleted workspace file {file_id}")
    return WorkspaceFile.from_db(updated) if updated else None


async def resolve_workspace_files(
    user_id: str,
    file_ids: list[str],
) -> list[UserWorkspaceFile]:
    """Return workspace-scoped file records for the given IDs.

    Filters out non-UUID entries, then queries only IDs that belong to the
    caller's workspace and are not soft-deleted.  Safe to call with
    untrusted input — invalid IDs and cross-user IDs are silently dropped.
    """
    valid_ids = [fid for fid in file_ids if _UUID_RE.fullmatch(fid)]
    if not valid_ids:
        return []
    workspace = await get_or_create_workspace(user_id)
    return await UserWorkspaceFile.prisma().find_many(
        where={
            "id": {"in": valid_ids},
            "workspaceId": workspace.id,
            "isDeleted": False,
        }
    )


def build_files_block(files: list[UserWorkspaceFile]) -> str:
    """Return a formatted ``[Attached files]`` block for injection into a message.

    Returns an empty string when *files* is empty so callers can do a simple
    ``message += build_files_block(files)`` without an extra ``if`` check.
    """
    if not files:
        return ""
    lines = [
        f"- {f.name} ({f.mimeType}, {round(f.sizeBytes / 1024, 1)} KB), file_id={f.id}"
        for f in files
    ]
    return (
        "\n\n[Attached files]\n"
        + "\n".join(lines)
        + "\nUse read_workspace_file with the file_id to access file contents."
    )


async def get_workspace_total_size(workspace_id: str) -> int:
    """
    Get the total size of all files in a workspace.

    Queries Prisma directly (skipping Pydantic model conversion) and only
    fetches the ``sizeBytes`` column to minimise data transfer.

    Args:
        workspace_id: The workspace ID

    Returns:
        Total size in bytes
    """
    files = await UserWorkspaceFile.prisma().find_many(
        where={"workspaceId": workspace_id, "isDeleted": False},
    )
    return sum(f.sizeBytes for f in files)
