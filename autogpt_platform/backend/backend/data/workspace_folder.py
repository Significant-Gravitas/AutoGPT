"""
Database CRUD operations for User Workspace Folders.

Folders are a purely DB-level organizational layer on top of workspace files —
storage paths are unaffected. A file's membership is tracked by its ``folderId``
(null = root) and a folder's by its ``parentId`` (null = root), so folders nest.
Mirrors the Library folder implementation.
"""

import logging
from datetime import datetime
from typing import Optional

import pydantic
from prisma import Prisma
from prisma.errors import UniqueViolationError
from prisma.models import UserWorkspaceFile, UserWorkspaceFolder

from backend.api.features.library.exceptions import (
    FolderAlreadyExistsError,
    FolderValidationError,
)
from backend.data.db import transaction
from backend.data.workspace import WorkspaceFile, get_or_create_workspace
from backend.data.workspace_scope import resolve_expert_workspace_scope
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)


class _Unchanged:
    """``parent_id`` absent from a PATCH: leave the folder where it is."""


UNCHANGED = _Unchanged()
ParentChange = Optional[str] | _Unchanged


class WorkspaceFolder(pydantic.BaseModel):
    """Pydantic model for UserWorkspaceFolder, safe for RPC transport."""

    id: str
    workspace_id: str
    name: str
    icon: Optional[str] = None
    parent_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    # Files directly in this folder; a subfolder's files are counted on it.
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
            parent_id=folder.parentId,
            created_at=folder.createdAt,
            updated_at=folder.updatedAt,
            file_count=file_count,
        )


async def _file_count(workspace_id: str, folder_id: str) -> int:
    """Live (non-deleted) file count for a single folder."""
    return await UserWorkspaceFile.prisma().count(
        where={
            "folderId": folder_id,
            "workspaceId": workspace_id,
            "isDeleted": False,
        },
    )


async def _file_counts(workspace_id: str, folder_ids: list[str]) -> dict[str, int]:
    """Live file counts per folder in a single batched query.

    Counts in SQL instead of hydrating every file row just to ``len()`` it, so
    listing folders stays cheap regardless of how many files a workspace holds.
    """
    if not folder_ids:
        return {}
    rows = await UserWorkspaceFile.prisma().group_by(
        by=["folderId"],
        where={
            "workspaceId": workspace_id,
            "folderId": {"in": folder_ids},
            "isDeleted": False,
        },
        count=True,
    )
    return {
        row["folderId"]: int((row.get("_count") or {}).get("_all") or 0)
        for row in rows
        if row.get("folderId")
    }


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
    )
    if not folder:
        raise NotFoundError(f"Folder #{folder_id} not found")
    return folder


async def list_workspace_folders(workspace_id: str) -> list[WorkspaceFolder]:
    """Every non-deleted folder in a workspace, flat.

    Named for the workspace because the RPC surface the copilot tools reach is
    one flat namespace that already carries the Library's ``list_folders``.
    The tree is not materialised here: a caller builds it from ``parent_id``,
    which costs one query however deep the nesting goes.
    """
    folders = await UserWorkspaceFolder.prisma().find_many(
        where={"workspaceId": workspace_id, "isDeleted": False},
        order={"name": "asc"},
    )
    counts = await _file_counts(workspace_id, [f.id for f in folders])
    return [WorkspaceFolder.from_db(f, file_count=counts.get(f.id, 0)) for f in folders]


async def get_folder(folder_id: str, workspace_id: str) -> WorkspaceFolder:
    """Get a single folder by ID, scoped to the workspace."""
    folder = await _get_folder_record(folder_id, workspace_id)
    count = await _file_count(workspace_id, folder_id)
    return WorkspaceFolder.from_db(folder, file_count=count)


async def create_folder(
    workspace_id: str,
    name: str,
    icon: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> WorkspaceFolder:
    """Create a folder, under *parent_id* or at the workspace root.

    A parent outside this workspace, or already deleted, is a
    :class:`NotFoundError` rather than a foreign-key error: ``parentId`` is
    written directly and the FK only checks that the row exists.

    Creating under a parent takes the workspace's hierarchy lock, because a
    check against a live parent and an insert under it must not straddle a
    delete of that parent's subtree.
    """
    create_data: dict = {
        "name": name,
        "Workspace": {"connect": {"id": workspace_id}},
    }
    if icon is not None:
        create_data["icon"] = icon
    if parent_id is not None:
        create_data["Parent"] = {"connect": {"id": parent_id}}

    async with transaction() as tx:
        if parent_id is not None:
            await _lock_workspace_hierarchy(tx, workspace_id)
            await _get_folder_record(parent_id, workspace_id)
        if await _name_taken(workspace_id, name, parent_id):
            raise FolderAlreadyExistsError("A folder with this name already exists")

        try:
            folder = await UserWorkspaceFolder.prisma(tx).create(data=create_data)
        except UniqueViolationError:
            raise FolderAlreadyExistsError("A folder with this name already exists")

    logger.info(f"Created workspace folder {folder.id} in workspace {workspace_id}")
    return WorkspaceFolder.from_db(folder)


async def _name_taken(
    workspace_id: str,
    name: str,
    parent_id: Optional[str],
    exclude_folder_id: Optional[str] = None,
) -> bool:
    """Whether a live sibling under *parent_id* already carries this name.

    The database enforces the same rule through two partial unique indexes —
    one per level, because Postgres treats every NULL ``parentId`` as distinct
    — so this check exists to answer 409 rather than a unique violation, and
    the indexes are the race-safe backstop.
    """
    where: dict = {
        "workspaceId": workspace_id,
        "name": name,
        "parentId": parent_id,
        "isDeleted": False,
    }
    if exclude_folder_id is not None:
        where["id"] = {"not": exclude_folder_id}
    return await UserWorkspaceFolder.prisma().find_first(where=where) is not None


async def update_folder(
    folder_id: str,
    workspace_id: str,
    name: Optional[str] = None,
    icon: Optional[str] = None,
) -> WorkspaceFolder:
    """Rename a folder and/or change its icon, leaving it where it is."""
    return await apply_folder_update(folder_id, workspace_id, name=name, icon=icon)


async def move_folder(
    folder_id: str,
    workspace_id: str,
    parent_id: Optional[str],
) -> WorkspaceFolder:
    """Move a folder under *parent_id*, or to the workspace root when None."""
    return await apply_folder_update(folder_id, workspace_id, parent_id=parent_id)


async def apply_folder_update(
    folder_id: str,
    workspace_id: str,
    parent_id: ParentChange = UNCHANGED,
    name: Optional[str] = None,
    icon: Optional[str] = None,
) -> WorkspaceFolder:
    """Move and/or rename a folder as a single write.

    One PATCH may do both, and half of it must not survive a refusal of the
    other, so the destination check, the name check and the write share one
    transaction and one statement. The name is checked against the folder's
    destination rather than where it came from, since that is where it has to
    be unique.

    A folder cannot move into its own subtree, which would detach it from the
    root. A workspace's moves are serialized, because two of them checking at
    once would each pass and then make the other's folder its parent.

    *parent_id* left at ``UNCHANGED`` keeps the folder where it is; ``None``
    moves it to the workspace root.
    """
    moving = not isinstance(parent_id, _Unchanged)
    async with transaction() as tx:
        if moving or name is not None:
            await _lock_workspace_hierarchy(tx, workspace_id)

        folder = await _get_folder_record(folder_id, workspace_id)
        destination = parent_id if moving else folder.parentId

        if moving and destination is not None:
            await _get_folder_record(destination, workspace_id)
            if folder_id in await _ancestor_ids(workspace_id, destination):
                raise FolderValidationError(
                    "A folder cannot be moved into itself or one of its subfolders"
                )

        if (moving or name is not None) and await _name_taken(
            workspace_id, folder.name if name is None else name, destination, folder_id
        ):
            raise FolderAlreadyExistsError("A folder with this name already exists")

        data: dict = {}
        if moving:
            data["parentId"] = destination
        if name is not None:
            data["name"] = name
        if icon is not None:
            data["icon"] = icon

        if data:
            # update_many (not update) so the write itself is guarded by
            # isDeleted: a folder soft-deleted concurrently after the ownership
            # check must not be updated and reported as a 200.
            try:
                updated_count = await UserWorkspaceFolder.prisma(tx).update_many(
                    where={"id": folder_id, "isDeleted": False},
                    data=data,
                )
            except UniqueViolationError:
                raise FolderAlreadyExistsError("A folder with this name already exists")
            if updated_count == 0:
                raise NotFoundError(f"Folder #{folder_id} not found")

    if data:
        logger.info(f"Updated workspace folder {folder_id}: {sorted(data)}")
    return await get_folder(folder_id, workspace_id)


async def _lock_workspace_hierarchy(tx: Prisma, workspace_id: str) -> None:
    """Hold the workspace's folder-hierarchy lock until the transaction ends.

    Moves, parented creates and subtree deletes all check the tree and then
    write it, so they serialize against each other or one can act on a shape
    another has already changed.

    execute_raw, not query_raw: pg_advisory_xact_lock returns void, which
    Prisma cannot deserialize as a result column.
    """
    await tx.execute_raw(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))", workspace_id
    )


async def _ancestor_ids(workspace_id: str, folder_id: str) -> list[str]:
    """*folder_id* and every live folder above it, nearest first.

    Stops on a repeat so a cycle already in the data cannot hang the walk.
    """
    by_id = {
        f.id: f
        for f in await UserWorkspaceFolder.prisma().find_many(
            where={"workspaceId": workspace_id, "isDeleted": False},
        )
    }
    chain: list[str] = []
    seen: set[str] = set()
    current: Optional[str] = folder_id
    while current is not None and current in by_id and current not in seen:
        seen.add(current)
        chain.append(current)
        current = by_id[current].parentId
    return chain


async def delete_folder(folder_id: str, workspace_id: str) -> None:
    """
    Soft-delete a folder and its subfolders, returning every file to root.

    Files are reparented to root (``folderId = null``) rather than deleted —
    the semantics the single-folder delete already had — so nothing is
    orphaned behind a hidden folder, however deep it sat.

    The subtree is resolved under the workspace's hierarchy lock, so a move or
    a create cannot add a folder to it after the snapshot and leave that folder
    live under a deleted parent, reachable from no root-anchored listing.
    """
    async with transaction() as tx:
        await _lock_workspace_hierarchy(tx, workspace_id)
        await _get_folder_record(folder_id, workspace_id)
        doomed = await _subtree_ids(workspace_id, folder_id)

        await UserWorkspaceFile.prisma(tx).update_many(
            where={"folderId": {"in": doomed}, "workspaceId": workspace_id},
            data={"folderId": None},
        )
        # update_many with an isDeleted guard so a concurrent delete that slips
        # past the ownership check above is a no-op rather than re-deleting a
        # logically-deleted row (TOCTOU-safe, idempotent).
        await UserWorkspaceFolder.prisma(tx).update_many(
            where={"id": {"in": doomed}, "isDeleted": False},
            data={"isDeleted": True},
        )

    logger.info(
        f"Soft-deleted workspace folder {folder_id} and {len(doomed) - 1} "
        "subfolder(s); files moved to root"
    )


async def _subtree_ids(workspace_id: str, folder_id: str) -> list[str]:
    """*folder_id* and every live folder beneath it.

    One query plus a walk in Python: a user's folder list is small, and a
    recursive CTE would mean raw SQL for no measurable gain. Uncapped, unlike
    the model-facing ``workspace_files._folder_subtree``, because a delete has
    to reach the whole subtree.
    """
    children: dict[Optional[str], list[str]] = {}
    for f in await UserWorkspaceFolder.prisma().find_many(
        where={"workspaceId": workspace_id, "isDeleted": False},
    ):
        children.setdefault(f.parentId, []).append(f.id)

    subtree: list[str] = []
    queue = [folder_id]
    seen = {folder_id}
    while queue:
        current = queue.pop()
        subtree.append(current)
        for child in children.get(current, []):
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return subtree


async def resolve_attachable_workspace_folders(
    user_id: str,
    folder_ids: list[str],
    *,
    expert_id: str | None,
) -> list[WorkspaceFolder]:
    """Return the caller's own live folders among *folder_ids*.

    Folders are user-level, so an expert session attaches one only if its
    scope reads the owner's files; what it may read inside is still decided
    per file. Unknown and cross-user IDs are silently dropped, as
    ``resolve_workspace_files`` does.
    """
    if not folder_ids:
        return []
    if expert_id is not None:
        scope = await resolve_expert_workspace_scope(user_id, expert_id)
        if not scope.reads_user_files:
            return []
    workspace = await get_or_create_workspace(user_id)
    folders = await UserWorkspaceFolder.prisma().find_many(
        where={
            "id": {"in": folder_ids},
            "workspaceId": workspace.id,
            "isDeleted": False,
        },
    )
    counts = await _file_counts(workspace.id, [f.id for f in folders])
    return [WorkspaceFolder.from_db(f, file_count=counts.get(f.id, 0)) for f in folders]


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

    # Move and read back in one transaction so the returned list reflects
    # exactly what this call moved, even under concurrent writes.
    scope: dict = {
        "id": {"in": file_ids},
        "workspaceId": workspace_id,
        "isDeleted": False,
    }
    async with transaction() as tx:
        await UserWorkspaceFile.prisma(tx).update_many(
            where=scope,
            data={"folderId": folder_id},
        )
        files = await UserWorkspaceFile.prisma(tx).find_many(where=scope)
    return [WorkspaceFile.from_db(f) for f in files]
