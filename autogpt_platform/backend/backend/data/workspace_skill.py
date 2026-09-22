import re
from datetime import datetime, timezone
from typing import Literal

from prisma import Prisma
from prisma.models import UserWorkspaceFile
from pydantic import BaseModel

from backend.data.db import transaction
from backend.data.skill_capacity import MAX_SKILLS_PER_EXPERT, skill_owner_folder
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
    status: Literal["stored", "capacity", "exists"]
    file: WorkspaceFile | None = None
    replaced_storage_path: str | None = None


async def publish_workspace_skill_file(
    write: WorkspaceSkillWrite,
) -> WorkspaceSkillPublication:
    folder = skill_owner_folder(write.path)
    if folder is None:
        raise ValueError("Skill publication requires a canonical SKILL.md path")
    async with transaction() as tx:
        await tx.execute_raw("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        await tx.execute_raw(
            "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
            f"skill-capacity:{write.workspace_id}:{folder}",
        )
        existing = await tx.userworkspacefile.find_unique(
            where={
                "workspaceId_path": {
                    "workspaceId": write.workspace_id,
                    "path": write.path,
                }
            }
        )
        if existing and not existing.isDeleted and not write.overwrite:
            return WorkspaceSkillPublication(status="exists")
        if existing is None or existing.isDeleted:
            if (
                await _root_count(tx, write.workspace_id, folder)
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
        )


async def _root_count(tx: Prisma, workspace_id: str, folder: str) -> int:
    rows = await tx.query_raw(
        'SELECT COUNT(*)::int AS count FROM "UserWorkspaceFile" '
        'WHERE "workspaceId" = $1 AND NOT "isDeleted" AND path ~ $2',
        workspace_id,
        rf"^{re.escape(folder)}/[^/]+/SKILL\.md$",
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
    await tx.userworkspacefile.update(
        where={"id": file_id},
        data={
            "isDeleted": True,
            "deletedAt": datetime.now(timezone.utc),
            "path": f"{write.path}__deleted__{file_id}",
        },
    )
    return await _create_root(tx, write)
