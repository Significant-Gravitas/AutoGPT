"""Delegated threads with their status, filtered and counted in SQL.

A thread's status is not a column: it is read from the turn state, a parked
question, and the thread's last message (a reply, an error marker, or a stop).
Deriving it in the query lets a status filter, a page and the counts all see
the same rows, instead of filtering a page that was already cut.
"""

from datetime import datetime
from typing import Literal

from prisma.models import ChatSession as PrismaChatSession
from pydantic import BaseModel

from backend.copilot.constants import (
    COPILOT_ERROR_PREFIX,
    COPILOT_RETRYABLE_ERROR_PREFIX,
)
from backend.copilot.db import _PENDING_QUESTION_SESSION_COLUMNS as SESSION_COLUMNS
from backend.copilot.delegation_db import delegated_session_sql
from backend.copilot.model import ChatSessionInfo
from backend.copilot.stream_registry import CANCELLED_MESSAGE
from backend.copilot.turn_pause import PAUSE_KEY
from backend.copilot.turn_queue import QUEUED_TURN_REFUSED_KEY, TURN_CANCELLED
from backend.data import db

ThreadStatus = Literal[
    "queued", "running", "needs_input", "completed", "failed", "cancelled"
]


class DelegationFilter(BaseModel):
    expert_id: str | None = None
    parent_session_id: str | None = None


class DelegatedThread(BaseModel):
    session: ChatSessionInfo
    status: ThreadStatus


# $1 user, $2 expert, $3 parent, $4/$5 error prefixes, $6 the stop marker,
# $7 the queue's refusal flag, $8 its note for a turn stopped while queued,
# $9 the pause marker's key.
_THREADS = (
    'WITH d AS (SELECT s."id", s."createdAt", CASE '
    "WHEN s.\"chatStatus\" = 'running' THEN 'running' "
    "WHEN s.\"chatStatus\" = 'queued' THEN 'queued' "
    "WHEN jsonb_typeof(s.\"metadata\" -> 'pending_question') = 'object' "
    "THEN 'needs_input' "
    # Held at its hand-off cap and stopped there.
    "WHEN (s.\"metadata\" ->> 'delegation_cap_stopped')::boolean THEN 'failed' "
    # Idle, yet the turn never answered: it died before a reply or a marker.
    "WHEN last.\"role\" IS DISTINCT FROM 'assistant' THEN 'failed' "
    # Never started: the queue's note closes it, stopped or refused.
    "WHEN last.\"metadata\" -> $7::text = 'true'::jsonb "
    "THEN CASE WHEN last.\"content\" = $8 THEN 'cancelled' ELSE 'failed' END "
    # Landed on a limit part-way and picked up from the thread: stopped. A
    # turn refused before it ran never started.
    'WHEN last."metadata" -> $9::text IS NOT NULL '
    "THEN CASE WHEN last.\"metadata\" -> $9::text ->> 'started' = 'false' "
    "THEN 'failed' ELSE 'cancelled' END "
    "WHEN position($4 in COALESCE(last.\"content\", '')) = 1 "
    "OR position($5 in COALESCE(last.\"content\", '')) = 1 "
    'THEN CASE WHEN position($6 in last."content") > 0 '
    "THEN 'cancelled' ELSE 'failed' END "
    "ELSE 'completed' END AS status "
    'FROM {schema_prefix}"ChatSession" s LEFT JOIN LATERAL ('
    'SELECT m."role", m."content", m."metadata" '
    'FROM {schema_prefix}"ChatMessage" m '
    'WHERE m."sessionId" = s."id" ORDER BY m."sequence" DESC LIMIT 1'
    ") last ON true "
    'WHERE s."userId" = $1 AND '
    + delegated_session_sql("s.")
    + ' AND ($2::text IS NULL OR s."expertId" = $2) '
    "AND ($3::text IS NULL OR s.\"metadata\" ->> 'delegated_by_session_id' = $3)) "
)


async def list_delegated_threads(
    user_id: str,
    filters: DelegationFilter,
    *,
    statuses: list[ThreadStatus] | None = None,
    limit: int = 50,
) -> list[DelegatedThread]:
    """The newest *limit* delegated threads matching *filters*, in any of
    *statuses* (every status when None)."""
    rows = await db.query_raw_with_schema(
        _THREADS + 'SELECT "id", status FROM d WHERE ($10::text[] IS NULL OR '
        'status = ANY($10::text[])) ORDER BY "createdAt" DESC LIMIT $11',
        *_params(user_id, filters),
        statuses,
        limit,
    )
    found: dict[str, ThreadStatus] = {str(r["id"]): r["status"] for r in rows}
    sessions = await _sessions(user_id, list(found))
    return [
        DelegatedThread(session=sessions[sid], status=state)
        for sid, state in found.items()
        if sid in sessions
    ]


async def count_threads_started_since(
    user_id: str, filters: DelegationFilter, since: datetime
) -> int:
    """How many delegated threads matching *filters* started at or after
    *since*."""
    rows = await db.query_raw_with_schema(
        _THREADS + "SELECT COUNT(*) AS n FROM d "
        # The column holds UTC without a zone; the driver binds timestamptz.
        "WHERE \"createdAt\" >= ($10::timestamptz AT TIME ZONE 'UTC')",
        *_params(user_id, filters),
        since,
    )
    return int(rows[0]["n"]) if rows else 0


async def count_delegated_threads(
    user_id: str, filters: DelegationFilter
) -> dict[ThreadStatus, int]:
    """How many delegated threads matching *filters* are in each status."""
    rows = await db.query_raw_with_schema(
        _THREADS + "SELECT status, COUNT(*) AS n FROM d GROUP BY status",
        *_params(user_id, filters),
    )
    return {row["status"]: int(row["n"]) for row in rows}


def _params(user_id: str, filters: DelegationFilter) -> tuple[str | None, ...]:
    return (
        user_id,
        filters.expert_id,
        filters.parent_session_id,
        COPILOT_ERROR_PREFIX,
        COPILOT_RETRYABLE_ERROR_PREFIX,
        CANCELLED_MESSAGE,
        QUEUED_TURN_REFUSED_KEY,
        TURN_CANCELLED,
        PAUSE_KEY,
    )


async def _sessions(user_id: str, ids: list[str]) -> dict[str, ChatSessionInfo]:
    if not ids:
        return {}
    rows = await db.query_raw_with_schema(
        f'SELECT {SESSION_COLUMNS} FROM {{schema_prefix}}"ChatSession" '
        'WHERE "userId" = $1 AND "id" = ANY($2::text[])',
        user_id,
        ids,
        model=PrismaChatSession,
    )
    return {row.id: ChatSessionInfo.from_db(row) for row in rows}
