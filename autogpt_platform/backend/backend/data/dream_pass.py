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
cleanup after it marking itself finished, and a hard forget marking it and
blanking the dream's text in it (``erase_dream_passes``). Its read set
(``readUuids``) only ever grows, and outlives its input bundle.

Three queries work across users: the reaper lists open passes whose lease
lapsed (``list_expired_dream_passes``, on the status and lease expiry index)
and closed passes whose cleanup is due and has not finished
(``list_dream_pass_cleanups``, on the status and cleanup index), and the
retention job deletes closed passes past their retention
(``delete_old_dream_passes``). None is exposed to a user.
"""

from datetime import datetime

import prisma.models
from prisma.types import DreamPassCreateInput, DreamPassWhereInput

from backend.copilot.dream.pass_errors import PASS_ERROR_CODES, PASS_ERROR_STAGES
from backend.data.db import (
    bounded_transaction,
    execute_raw_with_schema,
    query_raw_with_schema,
)
from backend.data.dream_pass_models import (
    CLOSED_STATUSES,
    DREAM_TEXT_FIELDS,
    ERASED_TEXT,
    OPEN_STATUSES,
    UNKNOWN_READS,
    DreamPassDraft,
    DreamPassRecord,
    DreamPassUpdate,
)
from backend.data.dream_pass_update import TRANSITION_SQL, transition_args
from backend.util.json import SafeJson

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

# A hard forget's marks, for one user's scope ($1, $2), in one statement
# over every pass of the scope: no page or limit can leave one out. A pass is
# marked when it read what the forget reached (its read set, matched against
# the reached uuids, $4, by a hash join), may have read anything (its read
# set holds UNKNOWN_READS, $10: a pass from before read sets were recorded),
# may still be writing (it is open, $5, or closed with its cleanup pending: a
# stop from outside closes the row while the pass may still be applying), or
# wrote what the forget erased ($3, the passes its erased dream writes name:
# a pass's own write is not in its read set). A pass marked by an earlier
# forget is returned again, so a forget repeated after its narratives' blank
# failed finishes it, but not written again: it was blanked when marked
# closed, or as it closed. One marked now that is closed as this statement
# writes it (its status read from the row being written, so one closing
# meanwhile is not missed) is blanked now; an open one when it closes
# (``dream_pass_update.py``). Every pass of the scope keeps its error only
# when it is one of the fixed codes ($8, $9), marked or not. Returns the ids
# marked, now or before.
ERASE_SQL = """
WITH reached AS (
    SELECT DISTINCT uuid FROM unnest($4::text[]) AS reached(uuid)
),
read_reached AS (
    SELECT DISTINCT pass."id"
    FROM {schema_prefix}"DreamPass" AS pass,
        unnest(pass."readUuids") AS read(uuid)
    WHERE pass."userId" = $1 AND pass."scopeKey" = $2
        AND read.uuid IN (SELECT uuid FROM reached)
),
target AS (
    SELECT "id",
        ("erasedAt" IS NOT NULL
            OR "status"::text = ANY($5::text[])
            OR "cleanupPendingAt" IS NOT NULL
            OR "id" IN (SELECT unnest($3::text[]))
            OR "id" IN (SELECT "id" FROM read_reached)
            OR $10::text = ANY("readUuids")) AS reached
    FROM {schema_prefix}"DreamPass"
    WHERE "userId" = $1 AND "scopeKey" = $2
),
written AS (
    UPDATE {schema_prefix}"DreamPass" AS pass SET
        "erasedAt" = CASE WHEN target.reached
            THEN COALESCE(pass."erasedAt", CURRENT_TIMESTAMP AT TIME ZONE 'UTC')
            ELSE pass."erasedAt" END,
        "phaseOutputs" = CASE
            WHEN target.reached AND pass."erasedAt" IS NULL
                AND NOT (pass."status"::text = ANY($5::text[]))
            THEN {schema_prefix}"dream_text_blanked"(
                pass."phaseOutputs", $6::text[], $7::text)
            ELSE pass."phaseOutputs" END,
        "operations" = CASE
            WHEN target.reached AND pass."erasedAt" IS NULL
                AND NOT (pass."status"::text = ANY($5::text[]))
            THEN {schema_prefix}"dream_text_blanked"(
                pass."operations", $6::text[], $7::text)
            ELSE pass."operations" END,
        "error" = {schema_prefix}"dream_error_blanked"(
            pass."error", $8::text[], $9::text[], $7::text),
        "updatedAt" = CURRENT_TIMESTAMP AT TIME ZONE 'UTC'
    FROM target
    WHERE pass."id" = target."id"
        AND ((target.reached AND pass."erasedAt" IS NULL)
            OR pass."error" IS DISTINCT FROM {schema_prefix}"dream_error_blanked"(
                pass."error", $8::text[], $9::text[], $7::text))
    RETURNING pass."id"
)
SELECT "id" FROM target WHERE reached
"""

# The erase's own deadline in the database, just under the store's for the
# call (``copilot/dream/store.RECORD_WRITE_TIMEOUT_SECONDS``): the statement
# is cancelled there before the caller stops waiting for it, and the forget
# reports a cleanup error to repeat.
ERASE_STATEMENT_TIMEOUT_MS = 9_000


async def create_dream_pass(draft: DreamPassDraft) -> DreamPassRecord:
    data: DreamPassCreateInput = {
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
        # Nothing read yet: the column's default is UNKNOWN_READS.
        "readUuids": [],
    }
    if draft.cost_context is not None:
        data["costContext"] = SafeJson(draft.cost_context)
    row = await prisma.models.DreamPass.prisma().create(data=data)
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
    expired_before: datetime,
    limit: int = 100,
    after_lapse: datetime | None = None,
    after_id: str | None = None,
) -> list[DreamPassRecord]:
    """Open passes, of every user, whose lease lapsed before
    *expired_before*, in lapse order (the lapse, then the id), at most
    *limit*: the first ones, or, given *after_lapse* and *after_id*, those
    after that point (keyset pagination, so the reaper can go on past rows it
    tried). A pass with no lease is not listed."""
    where: DreamPassWhereInput = {
        "status": {"in": list(OPEN_STATUSES)},
        "leaseExpiresAt": {"lt": expired_before},
    }
    if after_lapse is not None and after_id is not None:
        where["OR"] = [
            {"leaseExpiresAt": {"gt": after_lapse}},
            {"leaseExpiresAt": {"equals": after_lapse}, "id": {"gt": after_id}},
        ]
    rows = await prisma.models.DreamPass.prisma().find_many(
        where=where,
        order=[{"leaseExpiresAt": "asc"}, {"id": "asc"}],
        take=limit,
    )
    return [DreamPassRecord.from_db(row) for row in rows]


async def list_dream_pass_cleanups(
    due_before: datetime,
    limit: int = 100,
    marked_through: datetime | None = None,
    after_mark: datetime | None = None,
    after_id: str | None = None,
) -> list[DreamPassRecord]:
    """Closed passes, of every user, marked for a cleanup that has not
    finished and is due: marked before *due_before*, or holding no lease or
    one that lapsed before it. In mark order (the mark, then the id), at
    most *limit*: given *marked_through*, only those marked by then, and,
    given *after_mark* and *after_id*, only those after that point (keyset
    pagination, so the reaper can go on past cleanups it tried)."""
    where: list[DreamPassWhereInput] = [
        {
            "OR": [
                {"cleanupPendingAt": {"lte": due_before}},
                {"leaseExpiresAt": None},
                {"leaseExpiresAt": {"lte": due_before}},
            ]
        }
    ]
    if marked_through is not None:
        where.append({"cleanupPendingAt": {"lte": marked_through}})
    if after_mark is not None and after_id is not None:
        where.append(
            {
                "OR": [
                    {"cleanupPendingAt": {"gt": after_mark}},
                    {
                        "cleanupPendingAt": {"equals": after_mark},
                        "id": {"gt": after_id},
                    },
                ]
            }
        )
    rows = await prisma.models.DreamPass.prisma().find_many(
        where={
            "status": {"in": list(CLOSED_STATUSES)},
            "cleanupPendingAt": {"not": None},
            "AND": where,
        },
        order=[{"cleanupPendingAt": "asc"}, {"id": "asc"}],
        take=limit,
    )
    return [DreamPassRecord.from_db(row) for row in rows]


async def erase_dream_passes(
    user_id: str, scope_key: str, *, reached: list[str], named: list[str]
) -> list[str]:
    """Mark, for *user_id*'s hard forget, every pass of the scope that read
    what it reached (*reached*, uuids) or may have read anything (from before
    read sets were recorded), may still be writing, or wrote what it erased
    (*named*, pass ids), blanking those already closed, and keep every
    error of the scope only as a fixed code (``ERASE_SQL``,
    ``copilot/dream/forget_copies.py``). The ids of the passes marked, now or
    by an earlier forget, the open ones included; repeating it writes nothing
    more. Raises when the statement runs past ``ERASE_STATEMENT_TIMEOUT_MS``
    or a lock it waits on outlasts it."""
    async with bounded_transaction(ERASE_STATEMENT_TIMEOUT_MS) as tx:
        rows = await query_raw_with_schema(
            ERASE_SQL,
            user_id,
            scope_key,
            named,
            reached,
            [status.value for status in OPEN_STATUSES],
            list(DREAM_TEXT_FIELDS),
            ERASED_TEXT,
            list(PASS_ERROR_STAGES),
            list(PASS_ERROR_CODES),
            UNKNOWN_READS,
            client=tx,
        )
    return [row["id"] for row in rows]


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
