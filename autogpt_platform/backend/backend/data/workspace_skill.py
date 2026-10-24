import posixpath
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from prisma import Prisma
from prisma.models import UserWorkspaceFile
from pydantic import BaseModel, Field

from backend.data import db
from backend.data.db import transaction
from backend.data.skill_capacity import (
    MAX_SKILLS_PER_EXPERT,
    SKILL_ORIGIN_LABELS,
    SKILL_ORIGIN_MARKETPLACE,
    SKILL_ORIGIN_METADATA_KEY,
    SKILL_ORIGIN_USER,
    SkillLimitError,
    SkillOwnedError,
    skill_origin,
    skill_owner_folder,
)
from backend.data.workspace import WorkspaceFile
from backend.data.workspace_internal import is_internal_workspace_file
from backend.util.json import SafeJson


class WorkspaceSkillFile(BaseModel):
    """A package file other than SKILL.md, already stored at ``storage_path``."""

    file_id: str
    name: str
    path: str
    storage_path: str
    mime_type: str
    size_bytes: int
    checksum: str | None = None
    metadata: dict | None = None


class WorkspaceSkillWrite(BaseModel):
    workspace_id: str
    file_id: str
    name: str
    path: str
    storage_path: str
    mime_type: str
    size_bytes: int
    overwrite: bool
    checksum: str | None = None
    metadata: dict | None = None
    # The rest of the package, swapped in with the root: files that changed,
    # and paths the package no longer has.
    files: list[WorkspaceSkillFile] = Field(default_factory=list)
    removed_paths: list[str] = Field(default_factory=list)


class RetiredSkillFile(BaseModel):
    file_id: str
    storage_path: str


class WorkspaceSkillPublication(BaseModel):
    status: Literal["stored", "capacity", "exists", "owned"]
    file: WorkspaceFile | None = None
    replaced_storage_path: str | None = None
    replaced_file_id: str | None = None
    retired_files: list[RetiredSkillFile] = Field(default_factory=list)


def publication_refusal(
    publication: WorkspaceSkillPublication, write: WorkspaceSkillWrite
) -> Exception | None:
    """The error a refused publication raises; ``None`` once it is stored."""
    if publication.status == "capacity":
        origin = skill_origin(write.metadata) or SKILL_ORIGIN_USER
        return SkillLimitError(
            f"Skill limit reached ({MAX_SKILLS_PER_EXPERT} {SKILL_ORIGIN_LABELS[origin]} "
            "skills). Delete an unused skill first."
        )
    if publication.status == "owned":
        return SkillOwnedError(
            f"'{write.path.rsplit('/', 2)[1]}' is one of the owner's own skills; "
            "rename or delete it before installing a skill by that name."
        )
    if publication.status == "exists":
        return ValueError(f"File already exists at path: {write.path}")
    return None


async def publish_workspace_skill_file(
    write: WorkspaceSkillWrite,
) -> WorkspaceSkillPublication:
    """Publish a skill root, and with it any package files, as one change.

    Every blob is stored before this runs, so readers see either the whole
    previous package or the whole new one: a refused or failed publication
    leaves every row as it was.
    """
    folder = skill_owner_folder(write.path)
    if folder is None:
        raise ValueError("Skill publication requires a canonical SKILL.md path")
    _require_package_paths(write)
    async with _publication_transaction(write.workspace_id, folder) as tx:
        # Live rows only: since paths are unique among live files, a path holds
        # at most one, and retired roots left behind at it are history rather
        # than something to retire again.
        existing = await tx.userworkspacefile.find_first(
            where={
                "workspaceId": write.workspace_id,
                "path": write.path,
                "isDeleted": False,
            }
        )
        if existing and not write.overwrite:
            return WorkspaceSkillPublication(status="exists")
        origin = skill_origin(write.metadata) or SKILL_ORIGIN_USER
        existing_origin = skill_origin(
            existing.metadata
            if existing and isinstance(existing.metadata, dict)
            else None
        )
        active = existing is not None
        if (
            active
            and origin == SKILL_ORIGIN_MARKETPLACE
            and existing_origin == SKILL_ORIGIN_USER
        ):
            return WorkspaceSkillPublication(status="owned")
        if _takes_a_slot(existing, origin) and (
            await _root_count(tx, write.workspace_id, folder, origin)
            >= MAX_SKILLS_PER_EXPERT
        ):
            return WorkspaceSkillPublication(status="capacity")
        retired = await _swap_package_files(tx, write)
        record = (
            await _replace_root(tx, existing.id, write)
            if existing
            else await _create_root(tx, write)
        )
        return WorkspaceSkillPublication(
            status="stored",
            file=WorkspaceFile.from_db(record),
            replaced_storage_path=existing.storagePath if existing else None,
            replaced_file_id=existing.id if existing else None,
            retired_files=retired,
        )


async def skill_root_has_room(workspace_id: str, path: str, origin: str) -> bool:
    """Whether a root at ``path`` would fit the owner's budget for ``origin``:
    the capacity decision :func:`publish_workspace_skill_file` makes, read
    without taking its lock, so a concurrent write can still win the slot."""
    folder = skill_owner_folder(path)
    if folder is None:
        raise ValueError("Skill capacity is counted for canonical SKILL.md paths")
    existing = await UserWorkspaceFile.prisma().find_first(
        where={"workspaceId": workspace_id, "path": path, "isDeleted": False}
    )
    if not _takes_a_slot(existing, origin):
        return True
    return (
        await _root_count(db.prisma, workspace_id, folder, origin)
        < MAX_SKILLS_PER_EXPERT
    )


def _takes_a_slot(existing: UserWorkspaceFile | None, origin: str) -> bool:
    """A new root, or one changing budgets, needs a free slot in its own."""
    if existing is None:
        return True
    existing_origin = skill_origin(
        existing.metadata if isinstance(existing.metadata, dict) else None
    )
    return (existing_origin or SKILL_ORIGIN_USER) != origin


def _require_package_paths(write: WorkspaceSkillWrite) -> None:
    package = posixpath.dirname(write.path) + "/"
    paths = [f.path for f in write.files] + write.removed_paths
    if any(
        not path.startswith(package) or path == write.path or ".." in path.split("/")
        for path in paths
    ):
        raise ValueError("Skill package files must sit inside the package folder")
    if len(set(paths)) != len(paths):
        raise ValueError("Skill package paths must be distinct")


async def _swap_package_files(
    tx: Prisma, write: WorkspaceSkillWrite
) -> list[RetiredSkillFile]:
    """Retire each live row the package replaces or drops, then add the new
    ones. Paths are unique among live rows only, so a retired row frees its
    path for the replacement in the same transaction."""
    retired: list[RetiredSkillFile] = []
    for path in [f.path for f in write.files] + write.removed_paths:
        live = await tx.userworkspacefile.find_first(
            where={"workspaceId": write.workspace_id, "path": path, "isDeleted": False}
        )
        if live is None:
            continue
        await tx.userworkspacefile.update(
            where={"id": live.id},
            data={"isDeleted": True, "deletedAt": datetime.now(timezone.utc)},
        )
        retired.append(RetiredSkillFile(file_id=live.id, storage_path=live.storagePath))
    for file in write.files:
        await tx.userworkspacefile.create(
            data={
                "id": file.file_id,
                "workspaceId": write.workspace_id,
                "name": file.name,
                "path": file.path,
                "storagePath": file.storage_path,
                "mimeType": file.mime_type,
                "sizeBytes": file.size_bytes,
                "checksum": file.checksum,
                "metadata": SafeJson(file.metadata or {}),
                "isInternal": is_internal_workspace_file(
                    name=file.name, path=file.path, metadata=file.metadata
                ),
            }
        )
    return retired


@asynccontextmanager
async def _publication_transaction(
    workspace_id: str, folder: str
) -> AsyncIterator[Prisma]:
    async with transaction() as tx:
        await tx.execute_raw("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"skill-capacity:{workspace_id}:{folder}",
        )
        yield tx


async def rename_workspace_skill_file(
    file_id: str, workspace_id: str, name: str, folder: str
) -> WorkspaceFile | None:
    async with _publication_transaction(workspace_id, folder) as tx:
        file = await tx.userworkspacefile.find_first(
            where={"id": file_id, "workspaceId": workspace_id, "isDeleted": False}
        )
        if file is None:
            return None
        new_path = posixpath.join(posixpath.dirname(file.path), name)
        if skill_owner_folder(new_path) and not skill_owner_folder(file.path):
            origin = (
                skill_origin(file.metadata if isinstance(file.metadata, dict) else None)
                or SKILL_ORIGIN_USER
            )
            if (
                await _root_count(tx, workspace_id, folder, origin)
                >= MAX_SKILLS_PER_EXPERT
            ):
                raise SkillLimitError(
                    f"Skill limit reached ({MAX_SKILLS_PER_EXPERT} {SKILL_ORIGIN_LABELS[origin]} "
                    "skills). Delete an unused skill first."
                )
        updated = await tx.userworkspacefile.update(
            where={"id": file_id}, data={"name": name, "path": new_path}
        )
        return WorkspaceFile.from_db(updated) if updated else None


async def _root_count(tx: Prisma, workspace_id: str, folder: str, origin: str) -> int:
    rows = await tx.query_raw(
        'SELECT COUNT(*)::int AS count FROM "UserWorkspaceFile" '
        'WHERE "workspaceId" = $1 AND NOT "isDeleted" AND path ~ $2 '
        "AND CASE WHEN metadata->>$3 = $4 THEN $4 ELSE $5 END = $6",
        workspace_id,
        rf"^{re.escape(folder)}/[^/]+/SKILL\.md$",
        SKILL_ORIGIN_METADATA_KEY,
        SKILL_ORIGIN_MARKETPLACE,
        SKILL_ORIGIN_USER,
        origin,
    )
    return rows[0]["count"]


async def _create_root(tx: Prisma, write: WorkspaceSkillWrite) -> UserWorkspaceFile:
    return await tx.userworkspacefile.create(
        data={
            "id": write.file_id,
            "workspaceId": write.workspace_id,
            "name": write.name,
            "path": write.path,
            "storagePath": write.storage_path,
            "mimeType": write.mime_type,
            "sizeBytes": write.size_bytes,
            "checksum": write.checksum,
            "metadata": SafeJson(write.metadata or {}),
            "isInternal": is_internal_workspace_file(
                name=write.name, path=write.path, metadata=write.metadata
            ),
        }
    )


async def _replace_root(
    tx: Prisma, file_id: str, write: WorkspaceSkillWrite
) -> UserWorkspaceFile:
    # No rename: the retired row leaves the live-only unique index the moment
    # it is marked deleted, so the replacement takes the path in the same
    # transaction and the retired row keeps the path it was retired at.
    await tx.userworkspacefile.update(
        where={"id": file_id},
        data={"isDeleted": True, "deletedAt": datetime.now(timezone.utc)},
    )
    return await _create_root(tx, write)
