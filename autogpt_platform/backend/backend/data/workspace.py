"""
Database CRUD operations for User Workspace.

This module provides functions for managing user workspaces and workspace files.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from prisma.models import UserWorkspace, UserWorkspaceFile
from prisma.types import UserWorkspaceFileWhereInput

from backend.util.json import SafeJson

logger = logging.getLogger(__name__)


async def get_or_create_workspace(user_id: str) -> UserWorkspace:
    """
    Get user's workspace, creating one if it doesn't exist.

    Uses upsert to handle race conditions when multiple concurrent requests
    attempt to create a workspace for the same user.

    Args:
        user_id: The user's ID

    Returns:
        UserWorkspace instance
    """
    workspace = await UserWorkspace.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id},
            "update": {},  # No updates needed if exists
        },
    )

    return workspace


async def get_workspace(user_id: str) -> Optional[UserWorkspace]:
    """
    Get user's workspace if it exists.

    Args:
        user_id: The user's ID

    Returns:
        UserWorkspace instance or None
    """
    return await UserWorkspace.prisma().find_unique(where={"userId": user_id})


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
) -> UserWorkspaceFile:
    """
    Create a new workspace file record.

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
        Created UserWorkspaceFile instance
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
    return file


async def get_workspace_file(
    file_id: str,
    workspace_id: Optional[str] = None,
) -> Optional[UserWorkspaceFile]:
    """
    Get a workspace file by ID.

    Args:
        file_id: The file ID
        workspace_id: Optional workspace ID for validation

    Returns:
        UserWorkspaceFile instance or None
    """
    where_clause: dict = {"id": file_id, "isDeleted": False}
    if workspace_id:
        where_clause["workspaceId"] = workspace_id

    return await UserWorkspaceFile.prisma().find_first(where=where_clause)


async def get_workspace_file_by_path(
    workspace_id: str,
    path: str,
) -> Optional[UserWorkspaceFile]:
    """
    Get a workspace file by its virtual path.

    Args:
        workspace_id: The workspace ID
        path: Virtual path

    Returns:
        UserWorkspaceFile instance or None
    """
    # Normalize path
    if not path.startswith("/"):
        path = f"/{path}"

    return await UserWorkspaceFile.prisma().find_first(
        where={
            "workspaceId": workspace_id,
            "path": path,
            "isDeleted": False,
        }
    )


async def list_workspace_files(
    workspace_id: str,
    path_prefix: Optional[str] = None,
    include_deleted: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
) -> list[UserWorkspaceFile]:
    """
    List files in a workspace.

    Args:
        workspace_id: The workspace ID
        path_prefix: Optional path prefix to filter (e.g., "/documents/")
        include_deleted: Whether to include soft-deleted files
        limit: Maximum number of files to return
        offset: Number of files to skip

    Returns:
        List of UserWorkspaceFile instances
    """
    where_clause: UserWorkspaceFileWhereInput = {"workspaceId": workspace_id}

    if not include_deleted:
        where_clause["isDeleted"] = False

    if path_prefix:
        # Normalize prefix
        if not path_prefix.startswith("/"):
            path_prefix = f"/{path_prefix}"
        where_clause["path"] = {"startswith": path_prefix}

    return await UserWorkspaceFile.prisma().find_many(
        where=where_clause,
        order={"createdAt": "desc"},
        take=limit,
        skip=offset,
    )


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
    where_clause: dict = {"workspaceId": workspace_id}
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
    workspace_id: Optional[str] = None,
) -> Optional[UserWorkspaceFile]:
    """
    Soft-delete a workspace file.

    The path is modified to include a deletion timestamp to free up the original
    path for new files while preserving the record for potential recovery.

    Args:
        file_id: The file ID
        workspace_id: Optional workspace ID for validation

    Returns:
        Updated UserWorkspaceFile instance or None if not found
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
    return updated


async def get_workspace_total_size(workspace_id: str) -> int:
    """
    Get the total size of all files in a workspace.

    Args:
        workspace_id: The workspace ID

    Returns:
        Total size in bytes
    """
    files = await list_workspace_files(workspace_id)
    return sum(file.sizeBytes for file in files)
