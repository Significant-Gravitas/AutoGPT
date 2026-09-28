"""Reads behind the delegation lists: Otto's Delegations tab, an expert's Work
tab, and the Home rows for hand-offs.

Batched per page of threads (one query per fact, never one per thread) and
scoped to ``user_id`` in every query.
"""

from datetime import UTC, datetime
from typing import Any

from prisma.enums import ReviewStatus
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import PendingHumanReview
from pydantic import BaseModel, field_validator

from backend.copilot.constants import (
    COPILOT_NODE_EXEC_ID_SEPARATOR,
    COPILOT_NODE_PREFIX,
)
from backend.copilot.db import _PENDING_QUESTION_SESSION_COLUMNS as SESSION_COLUMNS
from backend.copilot.delegation_db import delegated_session_sql
from backend.copilot.model import ChatSessionInfo
from backend.data import db

# ``gate.review.node_id_for("delegate_to_expert")`` plus the separator, built
# here: importing the gate would load the tool registry into every reader.
HELD_HANDOFF_PREFIX = (
    f"{COPILOT_NODE_PREFIX}gate-delegate_to_expert{COPILOT_NODE_EXEC_ID_SEPARATOR}"
)


class ThreadMessages(BaseModel):
    first_user_content: str | None = None
    last_role: str | None = None
    last_content: str | None = None
    last_at: datetime | None = None

    @field_validator("last_at")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        # The column is UTC without a zone, and a raw read returns it naive.
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value


class HeldHandoff(BaseModel):
    review_id: str
    parent_session_id: str | None
    payload: dict[str, Any]
    created_at: datetime


async def list_delegated_sessions(
    user_id: str,
    *,
    expert_id: str | None = None,
    parent_session_id: str | None = None,
    limit: int = 50,
    question_only: bool = False,
) -> list[ChatSessionInfo]:
    """The user's delegated threads, newest first."""
    rows = await db.query_raw_with_schema(
        f'SELECT {SESSION_COLUMNS} FROM {{schema_prefix}}"ChatSession" '
        'WHERE "userId" = $1 AND '
        + delegated_session_sql()
        + ' AND ($2::text IS NULL OR "expertId" = $2) '
        "AND ($3::text IS NULL OR \"metadata\" ->> 'delegated_by_session_id' = $3) "
        "AND (NOT $4::boolean OR "
        "jsonb_typeof(\"metadata\" -> 'pending_question') = 'object') "
        'ORDER BY "createdAt" DESC LIMIT $5',
        user_id,
        expert_id,
        parent_session_id,
        question_only,
        limit,
        model=PrismaChatSession,
    )
    return [ChatSessionInfo.from_db(row) for row in rows]


async def get_thread_messages(
    user_id: str, session_ids: list[str]
) -> dict[str, ThreadMessages]:
    """Each thread's first user message (the brief) and its last message."""
    if not session_ids:
        return {}
    first = await _edge_messages(user_id, session_ids, "ASC", "AND m.\"role\" = 'user'")
    last = await _edge_messages(user_id, session_ids, "DESC", "")
    return {
        sid: ThreadMessages(
            first_user_content=first.get(sid, {}).get("content"),
            last_role=last.get(sid, {}).get("role"),
            last_content=last.get(sid, {}).get("content"),
            last_at=last.get(sid, {}).get("created_at"),
        )
        for sid in session_ids
    }


async def count_session_files(user_id: str, session_ids: list[str]) -> dict[str, int]:
    """How many live files each thread wrote to the user's workspace."""
    if not session_ids:
        return {}
    rows = await db.query_raw_with_schema(
        "SELECT split_part(f.\"path\", '/', 3) AS session_id, COUNT(*) AS files "
        'FROM {schema_prefix}"UserWorkspaceFile" f '
        'JOIN {schema_prefix}"UserWorkspace" w ON w."id" = f."workspaceId" '
        'WHERE w."userId" = $1 AND NOT f."isDeleted" '
        "AND f.\"metadata\" ->> 'origin' = 'agent-created' "
        "AND split_part(f.\"path\", '/', 3) = ANY($2::text[]) "
        "GROUP BY 1",
        user_id,
        session_ids,
    )
    return {str(row["session_id"]): int(row["files"]) for row in rows}


async def list_held_handoffs(
    user_id: str, parent_session_id: str | None = None
) -> list[HeldHandoff]:
    """Hand-offs waiting on the user's approval card, newest first."""
    rows = await PendingHumanReview.prisma().find_many(
        where={
            "userId": user_id,
            "status": ReviewStatus.WAITING,
            "nodeExecId": {"startswith": HELD_HANDOFF_PREFIX},
            **({"chatSessionId": parent_session_id} if parent_session_id else {}),
        },
        order={"createdAt": "desc"},
        take=50,
    )
    return [
        HeldHandoff(
            review_id=row.nodeExecId,
            parent_session_id=row.chatSessionId,
            payload=row.payload if isinstance(row.payload, dict) else {},
            created_at=row.createdAt,
        )
        for row in rows
    ]


async def _edge_messages(
    user_id: str, session_ids: list[str], order: str, role_filter: str
) -> dict[str, dict[str, Any]]:
    rows = await db.query_raw_with_schema(
        'SELECT DISTINCT ON (m."sessionId") m."sessionId" AS session_id, '
        'm."role" AS role, m."content" AS content, m."createdAt" AS created_at '
        'FROM {schema_prefix}"ChatMessage" m '
        'JOIN {schema_prefix}"ChatSession" s ON s."id" = m."sessionId" '
        'WHERE s."userId" = $1 AND m."sessionId" = ANY($2::text[]) '
        f'{role_filter} ORDER BY m."sessionId", m."sequence" {order}',
        user_id,
        session_ids,
    )
    return {str(row["session_id"]): row for row in rows}
