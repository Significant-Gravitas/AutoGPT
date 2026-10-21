"""Complete dependency invalidation without loading skill package snapshots."""

import json

from pydantic import BaseModel

from backend.data.db import query_raw_with_schema
from backend.data.skill_learning import require_owned_source


class RevokedVersion(BaseModel):
    id: str
    head_id: str
    owner_key: str
    expert_id: str | None
    skill_name: str


async def invalidate_source_versions(
    user_id: str, source_id: str, reason: str
) -> list[RevokedVersion]:
    await require_owned_source(user_id, source_id)
    return await query_raw_with_schema(
        """WITH RECURSIVE affected AS (
            SELECT "id" FROM {schema_prefix}"SkillVersion"
            WHERE "userId" = $1 AND "sources" @> $2::jsonb
            UNION
            SELECT child."id" FROM {schema_prefix}"SkillVersion" child
            JOIN affected parent ON
                COALESCE(child."restoredFromVersionId", child."baseVersionId") = parent."id"
            WHERE child."userId" = $1
        )
        UPDATE {schema_prefix}"SkillVersion" v
        SET "state" = 'invalidated', "stateReason" = $3
        WHERE v."userId" = $1 AND v."id" IN (SELECT "id" FROM affected)
        RETURNING v."id", v."headId" AS head_id, v."ownerKey" AS owner_key,
                  v."expertId" AS expert_id, v."skillName" AS skill_name""",
        user_id,
        json.dumps([{"source_id": source_id}]),
        reason[:1000],
        model=RevokedVersion,
    )


async def latest_independent_version_id(user_id: str, head_id: str) -> str | None:
    rows = await query_raw_with_schema(
        """SELECT "id" FROM {schema_prefix}"SkillVersion"
        WHERE "userId" = $1 AND "headId" = $2 AND "state" = 'ready'
              AND "content" <> ''
        ORDER BY "version" DESC LIMIT 1""",
        user_id,
        head_id,
    )
    return str(rows[0]["id"]) if rows else None
