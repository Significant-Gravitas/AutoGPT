"""Task board reads and writes: the user's columns, card placement, and the
per-card facts the board shows (started, needs you, last line, files)."""

import logging
from datetime import UTC, datetime
from typing import Any, Literal

from prisma import Prisma
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import UserBoard
from pydantic import BaseModel

from backend.data import db
from backend.data.expert_attribution import resolve_attributable_expert
from backend.util.json import SafeJson

from .board import (
    BACKLOG,
    DEFAULT_COLUMNS,
    DONE,
    UNPLACED_WINDOW,
    BoardColumn,
    first_sentence,
    validate_columns,
)
from .db import exclude_dream_sessions_sql
from .model import ChatSessionInfo

logger = logging.getLogger(__name__)

BOARD_SESSION_LIMIT = 500
OLDER_SESSION_LIMIT = 50
POSITION_GAP = 1024

# Started = the user has written in it. Hidden rows (an expert kickoff, a
# reassign brief) are the system talking, not the user.
_STARTED_SQL = (
    "EXISTS ("
    'SELECT 1 FROM {schema_prefix}"ChatMessage" m '
    'WHERE m."sessionId" = cs."id" AND m."role" = \'user\' '
    "AND COALESCE(m.\"metadata\"->>'hidden', 'false') <> 'true'"
    ")"
)

_EXTRAS_SQL = (
    'SELECT cs."id", '
    'cs."boardColumn" AS "board_column", '
    'cs."boardPosition" AS "board_position", '
    f'{_STARTED_SQL} AS "started", '
    "("
    "jsonb_typeof(cs.\"metadata\"->'pending_question') = 'object' "
    "OR EXISTS ("
    'SELECT 1 FROM {schema_prefix}"PendingHumanReview" r '
    'WHERE r."chatSessionId" = cs."id" AND r."status" = \'WAITING\''
    ")"
    ') AS "needs_you", '
    "("
    'SELECT m."content" FROM {schema_prefix}"ChatMessage" m '
    'WHERE m."sessionId" = cs."id" AND m."role" = \'assistant\' '
    'AND m."content" IS NOT NULL AND m."content" <> \'\' '
    'ORDER BY m."sequence" DESC LIMIT 1'
    ') AS "last_assistant", '
    "("
    'SELECT COUNT(*)::int FROM {schema_prefix}"UserWorkspaceFile" f '
    'JOIN {schema_prefix}"UserWorkspace" w ON w."id" = f."workspaceId" '
    'WHERE w."userId" = cs."userId" AND f."isDeleted" = FALSE '
    "AND f.\"path\" LIKE '/sessions/' || cs.\"id\" || '/%'"
    ') AS "file_count" '
    'FROM {schema_prefix}"ChatSession" cs '
    'WHERE cs."userId" = $1 AND cs."id" = ANY($2::text[])'
)

# Done sorts last so the row cap drops finished cards before active ones.
_BOARD_ORDER = (
    f"CASE WHEN \"boardColumn\" = '{DONE}' THEN 1 ELSE 0 END, "
    '"boardColumn" ASC NULLS LAST, "boardPosition" ASC NULLS LAST, '
    '"updatedAt" DESC, "id" DESC'
)

# Session timestamps are UTC wall-clock ``timestamp`` columns; the driver
# binds a datetime as ``timestamptz``.
_CUTOFF_SQL = "($2::timestamptz AT TIME ZONE 'UTC')"


class BoardExtras(BaseModel):
    board_column: str | None = None
    board_position: float | None = None
    started: bool = False
    needs_you: bool = False
    last_line: str | None = None
    file_count: int = 0


class BoardSessionRow(BaseModel):
    session: ChatSessionInfo
    board_column: str | None
    board_position: float | None
    started: bool
    needs_you: bool
    last_line: str | None
    file_count: int


async def get_board_columns(user_id: str) -> list[BoardColumn]:
    row = await UserBoard.prisma().find_unique(where={"userId": user_id})
    if row is None:
        return _default_columns()
    try:
        return validate_columns([BoardColumn.model_validate(c) for c in row.columns])
    except (TypeError, ValueError) as e:
        logger.warning(f"Unreadable board columns for user {user_id}: {e}")
        return _default_columns()


async def save_board_columns(
    user_id: str, columns: list[BoardColumn]
) -> list[BoardColumn]:
    """Store the user's columns. Cards in a column that is gone go to Backlog."""
    validated = validate_columns(columns)
    stored = SafeJson([c.model_dump() for c in validated])
    async with db.transaction() as tx:
        await UserBoard.prisma(tx).upsert(
            where={"userId": user_id},
            data={
                "create": {"userId": user_id, "columns": stored},
                "update": {"columns": stored},
            },
        )
        await db.execute_raw_with_schema(
            'UPDATE {schema_prefix}"ChatSession" SET "boardColumn" = $3 '
            'WHERE "userId" = $1 AND "boardColumn" IS NOT NULL '
            'AND NOT ("boardColumn" = ANY($2::text[]))',
            user_id,
            [c.key for c in validated],
            BACKLOG,
            client=tx,
        )
    return validated


async def list_board_sessions(
    user_id: str,
    *,
    organization_id: str | None,
    include_older: bool = False,
    now: datetime | None = None,
) -> list[BoardSessionRow]:
    """The user's cards: every placed session, plus unplaced ones touched in
    the last week. ``include_older`` adds the newest older unplaced sessions,
    which the board shows in Done."""
    cutoff = (now or datetime.now(UTC)) - UNPLACED_WINDOW
    base, params = _board_sessions_query(user_id, organization_id, cutoff)
    rows = await db.query_raw_with_schema(
        base
        + f' AND ("boardColumn" IS NOT NULL OR "updatedAt" > {_CUTOFF_SQL})'
        + f" ORDER BY {_BOARD_ORDER} LIMIT {BOARD_SESSION_LIMIT}",
        *params,
        model=PrismaChatSession,
    )
    if include_older:
        rows += await db.query_raw_with_schema(
            base
            + f' AND "boardColumn" IS NULL AND "updatedAt" <= {_CUTOFF_SQL}'
            + f' ORDER BY "updatedAt" DESC, "id" DESC LIMIT {OLDER_SESSION_LIMIT}',
            *params,
            model=PrismaChatSession,
        )
    extras = await get_board_session_extras(user_id, [row.id for row in rows])
    return [_board_session_row(row, extras.get(row.id, BoardExtras())) for row in rows]


async def get_board_session_extras(
    user_id: str, session_ids: list[str]
) -> dict[str, BoardExtras]:
    """Card facts for *user_id*'s sessions in *session_ids*, in one round-trip.
    Sessions that are not the user's are left out."""
    if not session_ids:
        return {}
    rows = await db.query_raw_with_schema(_EXTRAS_SQL, user_id, session_ids)
    return {
        row["id"]: BoardExtras(
            board_column=row["board_column"],
            board_position=row["board_position"],
            started=row["started"],
            needs_you=row["needs_you"],
            last_line=first_sentence(row["last_assistant"]),
            file_count=row["file_count"],
        )
        for row in rows
    }


async def top_position(user_id: str, column: str) -> float:
    """A position above every card in *column*, or 0.0 when it is empty."""
    rows = await db.query_raw_with_schema(
        "SELECT " + _top_position_sql("$1", "$2") + ' AS "top"',
        user_id,
        column,
    )
    return float(rows[0]["top"]) if rows else 0.0


async def set_session_board(
    session_id: str,
    user_id: str,
    *,
    column: str | None,
    position: float | None,
) -> bool:
    """Place a card; ``None`` leaves that field as it is.

    Raw SQL so ``updatedAt`` stays put: moving a card is not activity in the
    chat. False when the session is not the user's.
    """
    params: list[Any] = [session_id, user_id]
    assignments = []
    if column is not None:
        params.append(column)
        assignments.append(f'"boardColumn" = ${len(params)}::text')
    if position is not None:
        params.append(position)
        assignments.append(f'"boardPosition" = ${len(params)}::double precision')
    result = await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatSession" SET '
        + ", ".join(assignments or ['"boardColumn" = "boardColumn"'])
        + ' WHERE "id" = $1 AND "userId" = $2',
        *params,
    )
    return result > 0


async def session_is_started(session_id: str) -> bool:
    """Not scoped to a user: callers must check ownership first."""
    rows = await db.query_raw_with_schema(
        f'SELECT {_STARTED_SQL} AS "started" '
        'FROM {schema_prefix}"ChatSession" cs WHERE cs."id" = $1',
        session_id,
    )
    return bool(rows and rows[0]["started"])


async def set_session_expert_if_unstarted(
    session_id: str,
    user_id: str,
    expert_id: str | None,
    *,
    organization_id: str | None,
    team_id: str | None,
) -> Literal["ok", "not_found", "started", "expert_not_found"]:
    """Change who works a card, only while the user has not written in it.

    Locks the expert like ``create_chat_session`` so it cannot be archived in
    between, and the session so a first message cannot land in between.
    """
    async with db.transaction() as tx:
        if expert_id is not None:
            resolved = await resolve_attributable_expert(
                tx, user_id, expert_id, lock_for_update=True
            )
            if resolved is None:
                return "expert_not_found"
        state = await _lock_unstarted_session(tx, session_id, user_id)
        if state != "ok":
            return state
        await db.execute_raw_with_schema(
            'UPDATE {schema_prefix}"ChatSession" SET "expertId" = $3::text, '
            '"organizationId" = $4::text, "teamId" = $5::text '
            'WHERE "id" = $1 AND "userId" = $2',
            session_id,
            user_id,
            expert_id,
            organization_id,
            team_id,
            client=tx,
        )
    return "ok"


async def mark_session_reassigned(
    session_id: str,
    user_id: str,
    *,
    to_session_id: str,
    to_expert_id: str | None,
) -> None:
    """Record where the task went and move the old card to the top of Done.

    Merges only the two keys into metadata, like ``set_session_pending_question``,
    so a concurrent write to another key survives.
    """
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatSession" SET "metadata" = '
        "COALESCE(\"metadata\", '{{}}'::jsonb) || "
        "jsonb_build_object("
        "'reassigned_to_session_id', $3::text, "
        "'reassigned_to_expert_id', $4::text"
        '), "boardColumn" = $5::text, "boardPosition" = '
        + _top_position_sql("$2", "$5::text")
        + ' WHERE "id" = $1 AND "userId" = $2',
        session_id,
        user_id,
        to_session_id,
        to_expert_id,
        DONE,
    )


async def _lock_unstarted_session(
    tx: Prisma, session_id: str, user_id: str
) -> Literal["ok", "not_found", "started"]:
    """Lock the session, then check it is unstarted.

    Two statements on purpose: under READ COMMITTED a check inside the
    locking statement reads that statement's snapshot, so a first message
    committed while it waited for the lock would go unseen. ``FOR UPDATE``
    also blocks the key-share lock a new ``ChatMessage`` row takes on its
    session, so no message can land after the check.
    """
    locked = await db.query_raw_with_schema(
        'SELECT 1 FROM {schema_prefix}"ChatSession" '
        'WHERE "id" = $1 AND "userId" = $2 FOR UPDATE',
        session_id,
        user_id,
        client=tx,
    )
    if not locked:
        return "not_found"
    rows = await db.query_raw_with_schema(
        f'SELECT ({_STARTED_SQL} OR cs."chatStatus" <> \'idle\') AS "started" '
        'FROM {schema_prefix}"ChatSession" cs WHERE cs."id" = $1',
        session_id,
        client=tx,
    )
    return "started" if rows[0]["started"] else "ok"


def _default_columns() -> list[BoardColumn]:
    return [c.model_copy() for c in DEFAULT_COLUMNS]


def _board_sessions_query(
    user_id: str, organization_id: str | None, cutoff: datetime
) -> tuple[str, list[Any]]:
    """The board's session filter; ``$2`` is the unplaced cutoff."""
    params: list[Any] = [user_id, cutoff]
    conditions = [
        '"userId" = $1',
        exclude_dream_sessions_sql(),
        "(\"metadata\"->>'builder_graph_id') IS NULL",
    ]
    if organization_id is not None:
        params.append(organization_id)
        # Same carve-out as get_user_chat_sessions: expert sessions belong to
        # the owner's personal org and stay visible under any active org.
        conditions.append(
            f'("organizationId" = ${len(params)} OR "organizationId" IS NULL'
            ' OR "expertId" IS NOT NULL)'
        )
    query = 'SELECT * FROM {schema_prefix}"ChatSession" WHERE ' + " AND ".join(
        conditions
    )
    return query, params


def _board_session_row(row: PrismaChatSession, extra: BoardExtras) -> BoardSessionRow:
    return BoardSessionRow(
        session=ChatSessionInfo.from_db(row),
        board_column=row.boardColumn,
        board_position=row.boardPosition,
        started=extra.started,
        needs_you=extra.needs_you,
        last_line=extra.last_line,
        file_count=extra.file_count,
    )


def _top_position_sql(user_param: str, column_param: str) -> str:
    return (
        f'(SELECT COALESCE(MIN(b."boardPosition") - {POSITION_GAP}, 0) '
        'FROM {schema_prefix}"ChatSession" b '
        f'WHERE b."userId" = {user_param} AND b."boardColumn" = {column_param})'
    )
