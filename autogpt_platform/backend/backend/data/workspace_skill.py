import posixpath
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Literal

from prisma import Prisma
from prisma.models import UserWorkspaceFile
from pydantic import BaseModel

from backend.data.db import transaction
from backend.data.skill_capacity import (
    MAX_SKILLS_PER_EXPERT,
    SKILL_ORIGIN_LABELS,
    SKILL_ORIGIN_MARKETPLACE,
    SKILL_ORIGIN_METADATA_KEY,
    SKILL_ORIGIN_USER,
    SkillLimitError,
    skill_origin,
    skill_owner_folder,
)
from backend.data.workspace import WorkspaceFile
from backend.util.json import SafeJson


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


class WorkspaceSkillPublication(BaseModel):
    status: Literal["stored", "capacity", "exists", "owned"]
    file: WorkspaceFile | None = None
    replaced_storage_path: str | None = None
    replaced_file_id: str | None = None


async def publish_workspace_skill_file(
    write: WorkspaceSkillWrite,
) -> WorkspaceSkillPublication:
    folder = skill_owner_folder(write.path)
    if folder is None:
        raise ValueError("Skill publication requires a canonical SKILL.md path")
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
        if (not active or (existing_origin or SKILL_ORIGIN_USER) != origin) and (
            await _root_count(tx, write.workspace_id, folder, origin)
            >= MAX_SKILLS_PER_EXPERT
        ):
            return WorkspaceSkillPublication(status="capacity")
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
        )


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
