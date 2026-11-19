"""Which of a user's tasks watch a webhook.

A task keeps what it watches in its metadata (``TaskWatch``), so the lookup
reads that JSON. Every query is scoped to the webhook's owner. Kept apart
from the rest of the watch code so the webhook managers can ask whether a
webhook is still in use without importing the tasks.
"""

import prisma.models

from backend.data.db import query_raw_with_schema

_WATCHING = """
    SELECT t."id" AS id
      FROM {schema_prefix}"Task" t
     WHERE t."userId" = $1
       AND t."isArchived" = false
       AND t."status"::text IN ('TODO', 'IN_PROGRESS', 'NEEDS_INPUT')
       AND t."metadata" -> 'schedule' -> 'repeat' -> 'watch' ->> 'webhook_id' = $2
"""


async def watching_task_rows(user_id: str, webhook_id: str) -> list[prisma.models.Task]:
    """The user's open tasks that watch ``webhook_id``."""
    rows = await query_raw_with_schema(_WATCHING, user_id, webhook_id)
    ids = [row["id"] for row in rows]
    if not ids:
        return []
    return await prisma.models.Task.prisma().find_many(
        where={"id": {"in": ids}, "userId": user_id}
    )


async def is_webhook_watched(user_id: str, webhook_id: str) -> bool:
    """Whether any open task of the user's still watches ``webhook_id``."""
    return bool(await query_raw_with_schema(_WATCHING, user_id, webhook_id))
