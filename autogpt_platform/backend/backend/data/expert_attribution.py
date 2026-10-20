from typing import LiteralString, cast

from prisma import Prisma

from backend.data.db import get_database_schema


async def resolve_attributable_expert(
    client: Prisma,
    user_id: str,
    expert_id: str | None,
    *,
    lock_for_update: bool = False,
) -> str | None:
    """Resolve an active, hired expert owned by ``user_id``.

    Durable expert-attributed writes call this inside their transaction with
    ``lock_for_update=True``. The row lock serializes the write against expert
    archival, closing the validation-to-persistence race.
    """
    if not expert_id:
        return None

    schema = get_database_schema()
    schema_prefix = f'"{schema}".' if schema != "public" else ""
    lock_clause = " FOR UPDATE" if lock_for_update else ""
    query = cast(
        LiteralString,
        f"""
        SELECT "id"
        FROM {schema_prefix}"Expert"
        WHERE "id" = $1
          AND "ownerUserId" = $2
          AND "isTemplate" = FALSE
          AND "isArchived" = FALSE
        {lock_clause}
        """,
    )
    rows = await client.query_raw(
        query,
        expert_id,
        user_id,
    )
    return str(rows[0]["id"]) if rows else None
