"""The durable record of a dream pass: one ``DreamPass`` row per pass.

``backend/copilot/dream/store.py`` writes the row as a pass moves, on both
routes, through ``db_accessors.dream_db()``: the scheduler and the batch
executor keep no Prisma connection, so their writes cross the DatabaseManager
RPC. The row's shapes are ``dream_pass_models.py``. Every transition goes
through one statement (``dream_pass_update.py``) that only ever moves the row
forward and merges its JSON columns in the database.

A pass that has reached a terminal status (complete, errored, cancelled,
expired, skipped) is final: no update writes it, the way
``job_status.mark_complete`` never rewrites a finished job.
"""

import prisma.models

from backend.data.db import execute_raw_with_schema
from backend.data.dream_pass_models import (
    OPEN_STATUSES,
    DreamPassDraft,
    DreamPassRecord,
    DreamPassUpdate,
)
from backend.data.dream_pass_update import TRANSITION_SQL, transition_args


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
        }
    )
    return DreamPassRecord.from_db(row)


async def update_dream_pass(pass_id: str, update: DreamPassUpdate) -> bool:
    """Apply one transition to the pass's row, in a single statement.

    ``False`` when there is no such row, it has reached a terminal status, or
    it fails the update's owner or not-updated-since condition. ``True``
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


async def list_open_dream_passes(scope_key: str) -> list[DreamPassRecord]:
    """The scope's passes that have not reached a terminal status, oldest first."""
    rows = await prisma.models.DreamPass.prisma().find_many(
        where={"scopeKey": scope_key, "status": {"in": list(OPEN_STATUSES)}},
        order={"createdAt": "asc"},
    )
    return [DreamPassRecord.from_db(row) for row in rows]


async def list_dream_passes(user_id: str, limit: int = 20) -> list[DreamPassRecord]:
    """The user's passes, across every scope, newest first."""
    rows = await prisma.models.DreamPass.prisma().find_many(
        where={"userId": user_id}, order={"createdAt": "desc"}, take=limit
    )
    return [DreamPassRecord.from_db(row) for row in rows]
