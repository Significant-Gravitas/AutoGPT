"""The durable record of a dream pass: one ``DreamPass`` row per pass.

``backend/copilot/dream/store.py`` writes the row as a pass moves, on both
routes, through ``db_accessors.dream_db()``: the scheduler and the batch
executor keep no Prisma connection, so their writes cross the DatabaseManager
RPC. The row's shapes are ``dream_pass_models.py``. Every transition goes
through one statement (``dream_pass_update.py``) that only ever moves the row
forward and merges its JSON columns in the database.

A pass that has reached a terminal status (complete, errored, cancelled,
expired, skipped) is final: no update writes it, the way
``job_status.mark_complete`` never rewrites a finished job, but for the
cleanup after it marking itself finished.

Three queries work across users: the reaper lists open passes whose lease
lapsed (``list_expired_dream_passes``, on the status and lease expiry index)
and closed passes whose cleanup is due and has not finished
(``list_dream_pass_cleanups``, on the status and cleanup index), and the
retention job deletes closed passes past their retention
(``delete_old_dream_passes``). None is exposed to a user.
"""

from datetime import datetime

import prisma.models
from prisma.types import DreamPassWhereInput

from backend.data.db import execute_raw_with_schema
from backend.data.dream_pass_models import (
    CLOSED_STATUSES,
    OPEN_STATUSES,
    DreamPassDraft,
    DreamPassRecord,
    DreamPassUpdate,
)
from backend.data.dream_pass_update import TRANSITION_SQL, transition_args

# One batch of the retention delete: at most $3 closed passes created before
# $2 whose cleanup is not still pending. DELETE takes no LIMIT, so the batch's
# ids are picked in a subquery.
RETENTION_SQL = """
DELETE FROM {schema_prefix}"DreamPass" WHERE "id" IN (
    SELECT "id" FROM {schema_prefix}"DreamPass"
    WHERE NOT ("status"::text = ANY($1::text[]))
        AND "cleanupPendingAt" IS NULL
        AND "createdAt" < $2::timestamptz AT TIME ZONE 'UTC'
    LIMIT $3::int
)
"""


async def create_dream_pass(draft: DreamPassDraft) -> DreamPassRecord:
    row = await prisma.models.DreamPass.prisma().create(
        data={
            "id": draft.id,
            "userId": draft.user_id,
            "expertId": draft.expert_id,
            "scopeKey": draft.scope_key,
            "route": draft.route,
            "trigger": draft.trigger,
            "status": draft.status,
            "phase": draft.phase,
            "startedAt": draft.started_at,
            "leaseToken": draft.lease_token,
            "leaseExpiresAt": draft.lease_expires_at,
        }
    )
    return DreamPassRecord.from_db(row)


async def update_dream_pass(pass_id: str, update: DreamPassUpdate) -> bool:
    """Apply one transition to the pass's row, in a single statement.

    ``False`` when there is no such row, it has reached a terminal status
    (it is open, for an update marked ``closed_row``), or it fails the
    update's owner or not-updated-since condition. ``True``
    means the row was written, not that every field moved: a status or phase
    behind the row's, or a batch for a phase the row has left, is dropped in
    the statement.
    """
    written = await execute_raw_with_schema(
        TRANSITION_SQL, *transition_args(pass_id, update)
    )
    return written > 0


async def get_dream_pass(pass_id: str) -> DreamPassRecord | None:
    row = await prisma.models.DreamPass.prisma().find_unique(where={"id": pass_id})
    return DreamPassRecord.from_db(row) if row else None


async def get_dream_pass_for_user(pass_id: str, user_id: str) -> DreamPassRecord | None:
    """The pass's row when *user_id* owns it. ``None`` for another user's pass,
    the same as for a missing one, so a caller cannot probe for pass ids."""
    row = await prisma.models.DreamPass.prisma().find_first(
        where={"id": pass_id, "userId": user_id}
    )
    return DreamPassRecord.from_db(row) if row else None


async def list_open_dream_passes(
    scope_key: str, limit: int | None = None
) -> list[DreamPassRecord]:
    """The scope's passes that have not reached a terminal status, newest
    first; at most *limit* of them when given."""
    rows = await prisma.models.DreamPass.prisma().find_many(
        where={"scopeKey": scope_key, "status": {"in": list(OPEN_STATUSES)}},
        order={"createdAt": "desc"},
        take=limit,
    )
    return [DreamPassRecord.from_db(row) for row in rows]


async def list_dream_passes(
    user_id: str, limit: int = 20, open_only: bool = False
) -> list[DreamPassRecord]:
    """The user's passes, across every scope, newest first; only those still
    open when *open_only*."""
    where: DreamPassWhereInput = {"userId": user_id}
    if open_only:
        where["status"] = {"in": list(OPEN_STATUSES)}
    rows = await prisma.models.DreamPass.prisma().find_many(
        where=where, order={"createdAt": "desc"}, take=limit
    )
    return [DreamPassRecord.from_db(row) for row in rows]


async def list_expired_dream_passes(
    expired_before: datetime, limit: int = 100
) -> list[DreamPassRecord]:
    """Open passes, of every user, whose lease lapsed before
    *expired_before*: the oldest lapse first, at most *limit*. A pass with
    no lease is not listed."""
    rows = await prisma.models.DreamPass.prisma().find_many(
        where={
            "status": {"in": list(OPEN_STATUSES)},
            "leaseExpiresAt": {"lt": expired_before},
        },
        order={"leaseExpiresAt": "asc"},
        take=limit,
    )
    return [DreamPassRecord.from_db(row) for row in rows]


async def list_dream_pass_cleanups(
    due_before: datetime, limit: int = 100
) -> list[DreamPassRecord]:
    """Closed passes, of every user, marked for a cleanup that has not
    finished and is due: marked before *due_before*, or holding no lease or
    one that lapsed before it. The longest pending first, at most *limit*."""
    rows = await prisma.models.DreamPass.prisma().find_many(
        where={
            "status": {"in": list(CLOSED_STATUSES)},
            "cleanupPendingAt": {"not": None},
            "OR": [
                {"cleanupPendingAt": {"lte": due_before}},
                {"leaseExpiresAt": None},
                {"leaseExpiresAt": {"lte": due_before}},
            ],
        },
        order={"cleanupPendingAt": "asc"},
        take=limit,
    )
    return [DreamPassRecord.from_db(row) for row in rows]


async def delete_old_dream_passes(created_before: datetime, limit: int = 1000) -> int:
    """Delete at most *limit* closed passes, of every user, created before
    *created_before*, and say how many went. An open pass is never deleted,
    however old, nor one whose cleanup the reaper has yet to finish."""
    return await execute_raw_with_schema(
        RETENTION_SQL,
        [status.value for status in OPEN_STATUSES],
        created_before,
        limit,
    )
