"""
Database CRUD operations for User Workspace.

This module provides functions for managing user workspaces and workspace files.
"""

import logging
import posixpath
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import pydantic
from prisma.errors import UniqueViolationError
from prisma.models import UserWorkspace, UserWorkspaceFile
from prisma.types import UserWorkspaceFileWhereInput

from backend.data.skill_capacity import skill_owner_folder
from backend.data.workspace_scope import (
    SHARED_ROOTS,
    WorkspaceAccessDeniedError,
    resolve_expert_workspace_scope,
)
from backend.util.json import SafeJson

if TYPE_CHECKING:
    # Imported for typing only: workspace_folder imports this module.
    from backend.data.workspace_folder import WorkspaceFolder

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
    folder_id: Optional[str] = None
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
            folder_id=file.folderId,
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


async def get_workspace_file_by_id(
    file_id: str,
) -> Optional[WorkspaceFile]:
    """
    Get a workspace file by ID without workspace scoping.

    Only use this when access has already been validated through another
    mechanism (e.g. SharedExecutionFile allowlist). For user-facing
    endpoints, use get_workspace_file() which enforces workspace scoping.
    """
    file = await UserWorkspaceFile.prisma().find_first(
        where={"id": file_id, "isDeleted": False}
    )
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
    path_not_starts_with: Optional[str] = None,
    include_deleted: bool = False,
    limit: Optional[int] = None,
    offset: int = 0,
    name_contains: Optional[str] = None,
    metadata_equals: Optional[dict] = None,
    metadata_not_equals: Optional[dict] = None,
    folder_id: Optional[str] = None,
    folder_ids: Optional[list[str]] = None,
    root_only: bool = False,
    allowed_path_prefixes: Optional[list[str]] = None,
    include_user_files: bool = False,
) -> list[WorkspaceFile]:
    """
    List files in a workspace.

    Args:
        workspace_id: The workspace ID
        path_prefix: Optional path prefix to filter (e.g., "/documents/")
        path_not_starts_with: Optional path prefix to *exclude* from results.
            Generic path filter; the Artifacts origin filter is handled
            separately via ``metadata_equals``/``metadata_not_equals``.
        include_deleted: Whether to include soft-deleted files
        limit: Maximum number of files to return
        offset: Number of files to skip
        name_contains: Case-insensitive substring filter applied to
            ``name``. Used by /search/global so files are findable by
            literal name match without waiting on async embedding.
        metadata_equals: Match files whose ``metadata`` JSON equals this
            object exactly. Used by the Artifacts page "Uploaded" filter
            (``{"origin": "user-upload"}``).
        metadata_not_equals: Match files whose ``metadata`` JSON does *not*
            equal this object. Used by the "Generated" filter. ``metadata``
            is never SQL NULL (column default ``{}``), so whole-object
            inequality is null-safe and covers untagged/legacy files.
        folder_id: If set, only return files in this folder.
        folder_ids: If set, only return files in any of these folders — a
            folder and its descendants, for a recursive listing. An empty
            list matches nothing. Ignored when ``folder_id`` is set.
        root_only: If True, only return root-level files (folderId IS NULL).
            Ignored when either folder filter is set.
        allowed_path_prefixes: When set, only files whose path starts with
            one of these prefixes are returned (ANDed with ``path_prefix``).
            An empty list matches nothing. Used to apply an expert session's
            resolved scope inside the query so pagination stays correct.
        include_user_files: Widen ``allowed_path_prefixes`` with the owner's
            own files (see :func:`_scope_branches`). Ignored when the caller
            passes no prefixes, which already reaches the whole workspace.

    Returns:
        List of WorkspaceFile instances
    """
    scope_branches = _scope_branches(allowed_path_prefixes, include_user_files)
    if scope_branches is not None and not scope_branches:
        return []
    where_clause: UserWorkspaceFileWhereInput = {"workspaceId": workspace_id}

    if not include_deleted:
        where_clause["isDeleted"] = False

    if folder_id:
        where_clause["folderId"] = folder_id
    elif folder_ids is not None:
        if not folder_ids:
            return []
        where_clause["folderId"] = {"in": folder_ids}
    elif root_only:
        where_clause["folderId"] = None

    if path_prefix:
        # Normalize prefix
        if not path_prefix.startswith("/"):
            path_prefix = f"/{path_prefix}"
        where_clause["path"] = {"startswith": path_prefix}

    not_clause: list[UserWorkspaceFileWhereInput] = []
    if path_not_starts_with:
        if not path_not_starts_with.startswith("/"):
            path_not_starts_with = f"/{path_not_starts_with}"
        not_clause.append({"path": {"startswith": path_not_starts_with}})

    if metadata_equals is not None:
        where_clause["metadata"] = {"equals": SafeJson(metadata_equals)}

    if metadata_not_equals is not None:
        not_clause.append({"metadata": {"equals": SafeJson(metadata_not_equals)}})

    if not_clause:
        where_clause["NOT"] = not_clause

    if name_contains:
        where_clause["name"] = {"contains": name_contains, "mode": "insensitive"}

    if scope_branches:
        where_clause["AND"] = [{"OR": scope_branches}]

    files = await UserWorkspaceFile.prisma().find_many(
        where=where_clause,
        order={"createdAt": "desc"},
        take=limit,
        skip=offset,
    )
    return [WorkspaceFile.from_db(f) for f in files]


def _scope_branches(
    allowed_path_prefixes: Optional[list[str]],
    include_user_files: bool,
) -> Optional[list[UserWorkspaceFileWhereInput]]:
    """The OR-branches a scoped listing may match, or None when unscoped.

    Nested under ``AND`` by the callers so the scope filter can never replace
    another ``OR`` branch added later; scope prefixes are always rooted. An
    empty list means the scope reaches nothing, which callers answer with an
    empty result rather than an unfiltered query.
    """
    if allowed_path_prefixes is None:
        return None
    branches: list[UserWorkspaceFileWhereInput] = [
        {"path": {"startswith": prefix}} for prefix in allowed_path_prefixes
    ]
    if include_user_files:
        # The owner's own uploads: every path outside the managed roots, which
        # is the SQL form of ``workspace_scope.is_user_file_path``.
        branches.append(
            {"NOT": [{"path": {"startswith": root}} for root in SHARED_ROOTS]}
        )
    return branches


async def count_workspace_files(
    workspace_id: str,
    path_prefix: Optional[str] = None,
    include_deleted: bool = False,
    folder_id: Optional[str] = None,
    folder_ids: Optional[list[str]] = None,
    root_only: bool = False,
    allowed_path_prefixes: Optional[list[str]] = None,
    include_user_files: bool = False,
) -> int:
    """
    Count files in a workspace.

    Args:
        workspace_id: The workspace ID
        path_prefix: Optional path prefix to filter (e.g., "/sessions/abc123/")
        include_deleted: Whether to include soft-deleted files
        folder_id: See :func:`list_workspace_files`.
        folder_ids: See :func:`list_workspace_files`.
        root_only: See :func:`list_workspace_files`.
        allowed_path_prefixes: See :func:`list_workspace_files`.
        include_user_files: See :func:`list_workspace_files`.

    Returns:
        Number of files
    """
    scope_branches = _scope_branches(allowed_path_prefixes, include_user_files)
    if scope_branches is not None and not scope_branches:
        return 0
    where_clause: UserWorkspaceFileWhereInput = {"workspaceId": workspace_id}
    if not include_deleted:
        where_clause["isDeleted"] = False

    if folder_id:
        where_clause["folderId"] = folder_id
    elif folder_ids is not None:
        if not folder_ids:
            return 0
        where_clause["folderId"] = {"in": folder_ids}
    elif root_only:
        where_clause["folderId"] = None

    if path_prefix:
        # Normalize prefix
        if not path_prefix.startswith("/"):
            path_prefix = f"/{path_prefix}"
        where_clause["path"] = {"startswith": path_prefix}

    if scope_branches:
        where_clause["AND"] = [{"OR": scope_branches}]

    return await UserWorkspaceFile.prisma().count(where=where_clause)


async def rename_workspace_file(
    file_id: str,
    workspace_id: str,
    name: str,
) -> Optional[WorkspaceFile]:
    """Rename a file in place: its virtual path keeps its folder and gets the
    new name; the storage blob is untouched.

    Raises ``UniqueViolationError`` when another file already lives at the
    resulting path, so the caller can answer with a conflict.

    Returns the updated file, or None when it does not exist in the workspace.
    """
    file = await get_workspace_file(file_id, workspace_id)
    if file is None:
        return None
    folder = skill_owner_folder(
        posixpath.join(posixpath.dirname(file.path), "SKILL.md")
    )
    if folder is not None:
        from backend.data.workspace_skill import rename_workspace_skill_file

        return await rename_workspace_skill_file(file_id, workspace_id, name, folder)
    new_path = posixpath.join(posixpath.dirname(file.path), name)
    updated = await UserWorkspaceFile.prisma().update(
        where={"id": file_id},
        data={"name": name, "path": new_path},
    )
    if updated is None:
        return None
    logger.info(f"Renamed workspace file {file_id} to {new_path}")
    return WorkspaceFile.from_db(updated)


async def soft_delete_workspace_file(
    file_id: str,
    workspace_id: str,
) -> Optional[WorkspaceFile]:
    """
    Soft-delete a workspace file.

    The row keeps the path it was deleted at: the unique index over
    ``(workspaceId, path)`` covers live rows only, so the path is free for the
    next write and the record still says where the file was.

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

    updated = await UserWorkspaceFile.prisma().update(
        where={"id": file_id},
        data={
            "isDeleted": True,
            "deletedAt": datetime.now(timezone.utc),
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


async def resolve_attachable_workspace_files(
    user_id: str,
    file_ids: list[str],
    *,
    session_id: str,
    expert_id: str | None,
) -> list[UserWorkspaceFile]:
    """Resolve attachment IDs for a message sent in ``session_id``.

    Personal Otto sessions may attach any file in the owner's workspace.
    Expert sessions are confined to the expert's resolved scope — their own
    conversations plus the owner's own files — so only another expert's
    conversation file is refused, with a ``WorkspaceAccessDeniedError`` naming
    the files so the caller can surface a clear error instead of leaking the
    file's metadata into the turn.

    Runs in the API server with direct DB access, so the resolver is called
    in-process; code in the executor must go through ``workspace_db()``.
    """
    files = await resolve_workspace_files(user_id, file_ids)
    if expert_id is None or not files:
        return files
    scope = await resolve_expert_workspace_scope(user_id, expert_id)
    scope = scope.with_session(session_id)
    denied = [f.name for f in files if not scope.allows_path(f.path)]
    if denied:
        raise WorkspaceAccessDeniedError(
            "These files belong to another expert's conversations and cannot "
            f"be attached here: {', '.join(denied)}."
        )
    return files


def build_files_block(
    files: list[UserWorkspaceFile],
    folders: Optional[list["WorkspaceFolder"]] = None,
) -> str:
    """Return a formatted ``[Attached files]`` block for injection into a message.

    An attached folder is named rather than expanded: the model opens it with
    ``list_workspace_files``, so attaching one costs a line whatever it holds.

    Returns an empty string when nothing is attached so callers can do a simple
    ``message += build_files_block(files)`` without an extra ``if`` check.
    """
    if not files and not folders:
        return ""
    lines = [
        f"- {f.name} ({f.mimeType}, {round(f.sizeBytes / 1024, 1)} KB), file_id={f.id}"
        for f in files
    ]
    lines += [
        f"- {d.name} (folder, {d.file_count} file(s) directly inside), folder_id={d.id}"
        for d in folders or []
    ]
    hints = []
    if files:
        hints.append(
            "Use read_workspace_file with the file_id to access file contents."
        )
    if folders:
        hints.append(
            "Use list_workspace_files with the folder_id to see what is in a folder."
        )
    return "\n\n[Attached files]\n" + "\n".join(lines) + "\n" + "\n".join(hints)


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
