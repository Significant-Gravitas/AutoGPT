"""
Database CRUD operations for User Workspace Folders.

Folders are a purely DB-level organizational layer on top of workspace files —
storage paths are unaffected. A file's membership is tracked by its ``folderId``
(null = root). Mirrors the Library folder implementation.
"""

import logging
from datetime import datetime
from typing import Optional

import pydantic
from prisma.errors import UniqueViolationError
from prisma.models import UserWorkspaceFile, UserWorkspaceFolder

from backend.api.features.library.exceptions import FolderAlreadyExistsError
from backend.data.db import transaction
from backend.data.workspace import WorkspaceFile
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class WorkspaceFolder(pydantic.BaseModel):
    """Pydantic model for UserWorkspaceFolder, safe for RPC transport."""

    id: str
    workspace_id: str
    name: str
    icon: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    file_count: int = 0

    @staticmethod
    def from_db(
        folder: "UserWorkspaceFolder", file_count: int = 0
    ) -> "WorkspaceFolder":
        return WorkspaceFolder(
            id=folder.id,
            workspace_id=folder.workspaceId,
            name=folder.name,
            icon=folder.icon,
            created_at=folder.createdAt,
            updated_at=folder.updatedAt,
            file_count=file_count,
        )


def _file_count(folder: "UserWorkspaceFolder") -> int:
    # Prisma Python has no _count include; the Files relation is loaded with an
    # isDeleted=False filter so len() yields the live file count.
    return len(folder.Files) if folder.Files else 0


async def _get_folder_record(
    folder_id: str,
    workspace_id: str,
) -> "UserWorkspaceFolder":
    """Fetch a workspace-scoped folder record or raise NotFoundError."""
    folder = await UserWorkspaceFolder.prisma().find_first(
        where={
            "id": folder_id,
            "workspaceId": workspace_id,
            "isDeleted": False,
        },
        include={"Files": {"where": {"isDeleted": False}}},
    )
    if not folder:
        raise NotFoundError(f"Folder #{folder_id} not found")
    return folder


async def list_folders(workspace_id: str) -> list[WorkspaceFolder]:
    """List non-deleted folders for a workspace (flat; v1 has no nesting)."""
    folders = await UserWorkspaceFolder.prisma().find_many(
        where={"workspaceId": workspace_id, "isDeleted": False},
        order={"name": "asc"},
        include={"Files": {"where": {"isDeleted": False}}},
    )
    return [WorkspaceFolder.from_db(f, file_count=_file_count(f)) for f in folders]


async def get_folder(folder_id: str, workspace_id: str) -> WorkspaceFolder:
    """Get a single folder by ID, scoped to the workspace."""
    folder = await _get_folder_record(folder_id, workspace_id)
    return WorkspaceFolder.from_db(folder, file_count=_file_count(folder))


async def _root_name_taken(
    workspace_id: str,
    name: str,
    exclude_folder_id: Optional[str] = None,
) -> bool:
    """Whether a live root-level folder with this name already exists.

    The DB unique constraint ``@@unique([workspaceId, parentId, name])`` does
    not prevent duplicates at root: v1 folders always have ``parentId = NULL``
    and Postgres treats NULLs as distinct, so the constraint never fires for
    root folders. Guard the v1 (root) case explicitly here.
    """
    where: dict = {
        "workspaceId": workspace_id,
        "name": name,
        "parentId": None,
        "isDeleted": False,
    }
    if exclude_folder_id is not None:
        where["id"] = {"not": exclude_folder_id}
    return await UserWorkspaceFolder.prisma().find_first(where=where) is not None


async def create_folder(
    workspace_id: str,
    name: str,
    icon: Optional[str] = None,
) -> WorkspaceFolder:
    """Create a new root-level folder for the workspace."""
    if await _root_name_taken(workspace_id, name):
        raise FolderAlreadyExistsError("A folder with this name already exists")

    create_data: dict = {
        "name": name,
        "Workspace": {"connect": {"id": workspace_id}},
    }
    if icon is not None:
        create_data["icon"] = icon

    try:
        folder = await UserWorkspaceFolder.prisma().create(data=create_data)
    except UniqueViolationError:
        raise FolderAlreadyExistsError("A folder with this name already exists")

    logger.info(f"Created workspace folder {folder.id} in workspace {workspace_id}")
    return WorkspaceFolder.from_db(folder)


async def update_folder(
    folder_id: str,
    workspace_id: str,
    name: Optional[str] = None,
    icon: Optional[str] = None,
) -> WorkspaceFolder:
    """Update a folder's name/icon."""
    # update() uses where={"id": ...} without workspaceId — verify ownership first.
    await _get_folder_record(folder_id, workspace_id)

    if name is not None and await _root_name_taken(
        workspace_id, name, exclude_folder_id=folder_id
    ):
        raise FolderAlreadyExistsError("A folder with this name already exists")

    update_data: dict = {}
    if name is not None:
        update_data["name"] = name
    if icon is not None:
        update_data["icon"] = icon

    if not update_data:
        return await get_folder(folder_id, workspace_id)

    # update_many (not update) so the write itself is guarded by isDeleted: a
    # folder soft-deleted concurrently after the ownership check above must not
    # be silently updated (and reported as a 200).
    try:
        updated_count = await UserWorkspaceFolder.prisma().update_many(
            where={"id": folder_id, "isDeleted": False},
            data=update_data,
        )
    except UniqueViolationError:
        raise FolderAlreadyExistsError("A folder with this name already exists")

    if updated_count == 0:
        raise NotFoundError(f"Folder #{folder_id} not found")

    # Re-read without an isDeleted filter so a delete racing in *after* a
    # successful update doesn't turn it into a spurious 404.
    refreshed = await UserWorkspaceFolder.prisma().find_first(
        where={"id": folder_id},
        include={"Files": {"where": {"isDeleted": False}}},
    )
    if refreshed is None:
        raise NotFoundError(f"Folder #{folder_id} not found")
    return WorkspaceFolder.from_db(refreshed, file_count=_file_count(refreshed))


async def delete_folder(folder_id: str, workspace_id: str) -> None:
    """
    Soft-delete a folder and return its files to root.

    Files are reparented to root (``folderId = null``) in the same transaction
    so they remain visible and are never orphaned behind a hidden folder.
    """
    await _get_folder_record(folder_id, workspace_id)

    async with transaction() as tx:
        await UserWorkspaceFile.prisma(tx).update_many(
            where={"folderId": folder_id, "workspaceId": workspace_id},
            data={"folderId": None},
        )
        await UserWorkspaceFolder.prisma(tx).update(
            where={"id": folder_id},
            data={"isDeleted": True},
        )

    logger.info(f"Soft-deleted workspace folder {folder_id}; files moved to root")


async def bulk_move_files_to_folder(
    workspace_id: str,
    file_ids: list[str],
    folder_id: Optional[str],
) -> list[WorkspaceFile]:
    """
    Move multiple files into a folder (or to root when ``folder_id`` is None).

    Only files belonging to this workspace are updated; the target folder must
    belong to the workspace too. Cross-workspace IDs are silently dropped.
    """
    # folderId is set directly; the FK only checks existence, not ownership.
    if folder_id:
        await _get_folder_record(folder_id, workspace_id)

    if not file_ids:
        return []

    await UserWorkspaceFile.prisma().update_many(
        where={
            "id": {"in": file_ids},
            "workspaceId": workspace_id,
            "isDeleted": False,
        },
        data={"folderId": folder_id},
    )

    files = await UserWorkspaceFile.prisma().find_many(
        where={
            "id": {"in": file_ids},
            "workspaceId": workspace_id,
            "isDeleted": False,
        },
    )
    return [WorkspaceFile.from_db(f) for f in files]
