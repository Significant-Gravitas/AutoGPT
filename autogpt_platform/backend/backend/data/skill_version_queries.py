"""Version lists exclude package snapshots; exact-version reads load them."""

from typing import get_args

import prisma.models
import prisma.types

from backend.data.db import query_raw_with_schema

_SUMMARY_COLUMNS = ", ".join(
    f'"{field}"'
    for field in get_args(prisma.types.SkillVersionScalarFieldKeys)
    if field != "packageFiles"
)


async def list_version_summaries(
    user_id: str,
    *,
    owner_key: str | None = None,
    skill_name: str | None = None,
    origin: str | None = None,
    states: list[str] | None = None,
    limit: int = 50,
) -> list[prisma.models.SkillVersion]:
    conditions = ['"userId" = $1']
    parameters: list[object] = [user_id]
    for column, value in (
        ("ownerKey", owner_key),
        ("skillName", skill_name),
        ("origin", origin),
    ):
        if value is not None:
            parameters.append(value)
            conditions.append(f'"{column}" = ${len(parameters)}')
    if states is not None:
        parameters.append(states)
        conditions.append(f'"state" = ANY(${len(parameters)}::text[])')
    parameters.append(limit)
    order = "version" if skill_name is not None else "createdAt"
    return await query_raw_with_schema(
        f'SELECT {_SUMMARY_COLUMNS} FROM {{schema_prefix}}"SkillVersion" '
        f'WHERE {" AND ".join(conditions)} ORDER BY "{order}" DESC '
        f"LIMIT ${len(parameters)}",
        *parameters,
        model=prisma.models.SkillVersion,
    )
