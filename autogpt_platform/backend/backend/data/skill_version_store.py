"""Where a skill version's package files live, and how long versions stay.

A version keeps a manifest (paths, hashes, sizes, modes); the bytes are in
``SkillFileContent``, one row per user and SHA-256, so a file unchanged across
saves, skills or paths is stored once. Every query names the user, and one
user's writes and cleanups take that user's lock, so a cleanup never deletes
bytes a version being written is about to list. Versions written before the
manifest keep inline copies in ``packageFiles`` until
:func:`move_inline_package_files` moves them.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import prisma
import prisma.models
from prisma import Prisma
from pydantic import BaseModel, TypeAdapter

from backend.data.db import (
    bounded_transaction,
    execute_raw_with_schema,
    query_raw_with_schema,
    transaction,
)
from backend.data.skill_package import file_sha256
from backend.data.skill_version_files import SkillVersionFile, parse_version_files
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)

MAX_VERSIONS_PER_SKILL = 50
# Versions a decision, a publication or a review is still working on.
_OPEN_STATES = ["pending_write", "candidate", "needs_decision", "paused"]
_MOVE_PAGE = 50
# Versions moved at once; each holds its package in memory while it moves.
_MOVE_CONCURRENCY = 4
_MOVE_STATEMENT_TIMEOUT_MS = 20_000


class SkillVersionFileRef(BaseModel):
    relative_path: str
    sha256: str
    size_bytes: int
    is_executable: bool = False


class StoredPackage(BaseModel):
    manifest: list[SkillVersionFileRef]
    # How many of the files were new to this user.
    stored: int


class SkillVersionFilesMissingError(Exception):
    """A version lists a file its owner's store does not hold."""


async def store_version_files(
    tx: Prisma, user_id: str, files: list[SkillVersionFile]
) -> StoredPackage:
    """Store the files *user_id* does not hold yet and return the manifest.

    Call inside the transaction that writes the version row, so the cleanup,
    which takes the same lock, sees the row that lists them.
    """
    contents = [f.content for f in files]
    manifest = [
        SkillVersionFileRef(
            relative_path=f.relative_path,
            sha256=file_sha256(content),
            size_bytes=len(content),
            is_executable=f.is_executable,
        )
        for f, content in zip(files, contents)
    ]
    if not files:
        return StoredPackage(manifest=manifest, stored=0)
    await _lock_user_files(tx, user_id)
    by_hash = {ref.sha256: content for ref, content in zip(manifest, contents)}
    held = await query_raw_with_schema(
        'SELECT "sha256" FROM {schema_prefix}"SkillFileContent" '
        'WHERE "userId" = $1 AND "sha256" = ANY($2::text[])',
        user_id,
        list(by_hash),
        client=tx,
    )
    missing = set(by_hash) - {row["sha256"] for row in held}
    if missing:
        await tx.skillfilecontent.create_many(
            data=[
                {
                    "userId": user_id,
                    "sha256": sha256,
                    "content": prisma.Base64.encode(by_hash[sha256]),
                }
                for sha256 in sorted(missing)
            ],
            skip_duplicates=True,
        )
    return StoredPackage(manifest=manifest, stored=len(missing))


def manifest_json(package: StoredPackage) -> SafeJson:
    return SafeJson([ref.model_dump() for ref in package.manifest])


async def read_version_files(
    row: prisma.models.SkillVersion,
) -> list[SkillVersionFile] | None:
    """The version's package as stored, ``None`` for a Markdown-only one."""
    if row.packageManifest is None:
        return parse_version_files(row.packageFiles)
    refs = TypeAdapter(list[SkillVersionFileRef]).validate_python(row.packageManifest)
    if not refs:
        return []
    blobs = await prisma.models.SkillFileContent.prisma().find_many(
        where={"userId": row.userId, "sha256": {"in": list({r.sha256 for r in refs})}}
    )
    content = {blob.sha256: blob.content.decode() for blob in blobs}
    if any(ref.sha256 not in content for ref in refs):
        raise SkillVersionFilesMissingError(f"version {row.id} lists a missing file")
    return [
        SkillVersionFile.from_content(
            ref.relative_path, content[ref.sha256], ref.is_executable
        )
        for ref in refs
    ]


async def prune_version_history(user_id: str, head_id: str) -> int:
    """Delete the skill's versions past its newest ``MAX_VERSIONS_PER_SKILL``
    that nothing still points at, then the files only they listed, and say
    how many versions went. Never raises: the next save prunes again."""
    try:
        async with transaction() as tx:
            await _lock_user_files(tx, user_id)
            deleted = await query_raw_with_schema(
                _PRUNE_VERSIONS,
                user_id,
                head_id,
                MAX_VERSIONS_PER_SKILL,
                _OPEN_STATES,
                client=tx,
            )
            hashes = {
                ref["sha256"]
                for row in deleted
                for ref in _json_refs(row["packageManifest"])
                if isinstance(ref.get("sha256"), str)
            }
            if hashes:
                await _delete_unlisted_files(tx, user_id, sorted(hashes))
    except Exception:
        logger.warning(f"Could not prune skill history {head_id}", exc_info=True)
        return 0
    return len(deleted)


async def move_inline_package_files() -> int:
    """Move every legacy version's inline package into the store, a version
    per transaction, and say how many moved.

    Runs at start-up under the caller's deadline: a version moved stays
    moved, the next start goes on from the rest, and one that fails is
    logged and left for the next start rather than retried here.
    """
    moved = 0
    after = ""
    limit = asyncio.Semaphore(_MOVE_CONCURRENCY)

    async def move(row: dict[str, Any]) -> int:
        async with limit:
            try:
                return await _move_one(row["id"], row["userId"])
            except Exception:
                logger.warning(
                    f"Could not move the package of skill version {row['id']}",
                    exc_info=True,
                )
                return 0

    while True:
        page = await query_raw_with_schema(
            'SELECT "id", "userId" FROM {schema_prefix}"SkillVersion" '
            'WHERE "packageFiles" IS NOT NULL AND "id" > $1 ORDER BY "id" LIMIT $2',
            after,
            _MOVE_PAGE,
        )
        if not page:
            return moved
        moved += sum(await asyncio.gather(*(move(row) for row in page)))
        after = page[-1]["id"]


_PRUNE_VERSIONS = """
WITH ranked AS (
    SELECT "id", "state", "baseVersionId", "restoredFromVersionId",
           row_number() OVER (ORDER BY "version" DESC) AS rn
      FROM {schema_prefix}"SkillVersion"
     WHERE "userId" = $1 AND "headId" = $2
), kept AS (
    SELECT * FROM ranked
     WHERE rn <= $3
        OR "state" = ANY($4::text[])
        OR "id" = (SELECT "currentVersionId" FROM {schema_prefix}"SkillHead"
                    WHERE "id" = $2 AND "userId" = $1)
), pointed AS (
    SELECT "baseVersionId" AS "id" FROM kept WHERE "state" = ANY($4::text[])
    UNION
    SELECT "restoredFromVersionId" FROM kept
)
DELETE FROM {schema_prefix}"SkillVersion" v
 USING ranked r
 WHERE v."id" = r."id" AND v."userId" = $1 AND v."headId" = $2
   AND r."id" NOT IN (SELECT "id" FROM kept)
   AND r."id" NOT IN (SELECT "id" FROM pointed WHERE "id" IS NOT NULL)
RETURNING v."packageManifest"
"""


async def _move_one(version_id: str, user_id: str) -> int:
    async with bounded_transaction(_MOVE_STATEMENT_TIMEOUT_MS) as tx:
        await _lock_user_files(tx, user_id)
        rows = await query_raw_with_schema(
            'SELECT "packageFiles" FROM {schema_prefix}"SkillVersion" '
            'WHERE "id" = $1 AND "userId" = $2 AND "packageFiles" IS NOT NULL '
            "FOR UPDATE",
            version_id,
            user_id,
            client=tx,
        )
        if not rows:
            return 0
        files = parse_version_files(rows[0]["packageFiles"])
        manifest = None
        if files is not None:
            package = await store_version_files(tx, user_id, files)
            manifest = json.dumps([ref.model_dump() for ref in package.manifest])
        await execute_raw_with_schema(
            'UPDATE {schema_prefix}"SkillVersion" '
            'SET "packageManifest" = $3::jsonb, "packageFiles" = NULL '
            'WHERE "id" = $1 AND "userId" = $2',
            version_id,
            user_id,
            manifest,
            client=tx,
        )
    return 1


async def _delete_unlisted_files(tx: Prisma, user_id: str, hashes: list[str]) -> None:
    await execute_raw_with_schema(
        'DELETE FROM {schema_prefix}"SkillFileContent" c '
        'WHERE c."userId" = $1 AND c."sha256" = ANY($2::text[]) '
        "AND NOT EXISTS ("
        'SELECT 1 FROM {schema_prefix}"SkillVersion" v WHERE v."userId" = $1 '
        'AND v."packageManifest" @> jsonb_build_array('
        "jsonb_build_object('sha256', c.\"sha256\")))",
        user_id,
        hashes,
        client=tx,
    )


async def _lock_user_files(tx: Prisma, user_id: str) -> None:
    # execute_raw, not query_raw: pg_advisory_xact_lock returns void.
    await tx.execute_raw(
        "SELECT pg_advisory_xact_lock(hashtextextended('skill-files:' || $1, 0))",
        user_id,
    )


def _json_refs(value: Any) -> list[dict[str, Any]]:
    return (
        [ref for ref in value if isinstance(ref, dict)]
        if isinstance(value, list)
        else []
    )
