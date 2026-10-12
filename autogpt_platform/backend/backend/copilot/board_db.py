"""Task board reads and writes: the user's boards and their columns, card
placement, and the per-card facts the board shows (started, needs you, last
line, files)."""

import enum
import logging
from datetime import UTC, datetime
from typing import Any, Literal

from prisma import Prisma
from prisma.errors import RawQueryError
from prisma.models import Board
from prisma.models import ChatSession as PrismaChatSession
from prisma.types import BoardUpdateInput
from pydantic import BaseModel

from backend.data import db
from backend.data.expert_attribution import resolve_attributable_expert
from backend.util.json import SafeJson, dumps

from .board import (
    BACKLOG,
    DEFAULT_BOARD_NAME,
    DEFAULT_COLUMNS,
    DONE,
    MAX_BOARDS,
    REASSIGN_BRIEF_KIND,
    UNPLACED_WINDOW,
    BoardColumn,
    first_sentence,
    validate_board_name,
    validate_columns,
)
from .db import exclude_dream_sessions_sql
from .model import ChatSessionInfo

logger = logging.getLogger(__name__)

BOARD_SESSION_LIMIT = 500
OLDER_SESSION_LIMIT = 50
POSITION_GAP = 1024
DEFAULT_BOARD_REF = "default"
_FOREIGN_KEY_VIOLATION = "23503"

# Started = the user has written in it. A hidden row (an expert kickoff) is
# the system talking, not the user; a reassign brief carries a task the user
# already started elsewhere.
_STARTED_SQL = (
    "EXISTS ("
    'SELECT 1 FROM {schema_prefix}"ChatMessage" m '
    'WHERE m."sessionId" = cs."id" AND m."role" = \'user\' '
    "AND (COALESCE(m.\"metadata\"->>'hidden', 'false') <> 'true' "
    f"OR m.\"metadata\"->>'kind' = '{REASSIGN_BRIEF_KIND}')"
    ")"
)

# Something the chat would show: an expert's greeting after a hidden kickoff
# counts, the kickoff itself does not.
_HAS_MESSAGES_SQL = (
    "EXISTS ("
    'SELECT 1 FROM {schema_prefix}"ChatMessage" m '
    'WHERE m."sessionId" = cs."id" AND (m."role" = \'assistant\' '
    "OR (m.\"role\" = 'user' "
    "AND COALESCE(m.\"metadata\"->>'hidden', 'false') <> 'true'))"
    ")"
)

_EXTRAS_SQL = (
    'SELECT cs."id", '
    'cs."boardId" AS "board_id", '
    'cs."boardColumn" AS "board_column", '
    'cs."boardPosition" AS "board_position", '
    f'{_STARTED_SQL} AS "started", '
    f'{_HAS_MESSAGES_SQL} AS "has_messages", '
    "("
    # A missing key makes jsonb_typeof NULL, and NULL OR FALSE stays NULL.
    "COALESCE(jsonb_typeof(cs.\"metadata\"->'pending_question') = 'object', FALSE) "
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


class BoardInfo(BaseModel):
    id: str
    name: str
    is_default: bool
    position: float
    columns: list[BoardColumn]


class BoardNotFoundError(LookupError):
    """No board with that id belongs to the user."""


class BoardLimitError(ValueError):
    """The user already has as many boards as allowed."""


class DefaultBoardError(ValueError):
    """The default board cannot be renamed or deleted."""


class _Unset(enum.Enum):
    UNSET = "UNSET"


UNSET = _Unset.UNSET


class BoardExtras(BaseModel):
    board_id: str | None = None
    board_column: str | None = None
    board_position: float | None = None
    started: bool = False
    has_messages: bool = False
    needs_you: bool = False
    last_line: str | None = None
    file_count: int = 0


class BoardSessionRow(BaseModel):
    session: ChatSessionInfo
    board_id: str | None
    board_column: str | None
    board_position: float | None
    started: bool
    has_messages: bool
    needs_you: bool
    last_line: str | None
    file_count: int


async def get_or_create_default_board(user_id: str) -> BoardInfo:
    """The user's Home board, made on first use. The partial unique index
    keeps it to one per user when two requests make it at once."""
    row = await _find_default_board(user_id)
    if row is None:
        await db.execute_raw_with_schema(
            'INSERT INTO {schema_prefix}"Board" '
            '("id", "updatedAt", "userId", "name", "isDefault", "position", "columns") '
            "VALUES (gen_random_uuid()::text, CURRENT_TIMESTAMP, $1, $2, TRUE, 0, "
            "$3::jsonb) "
            'ON CONFLICT ("userId") WHERE "isDefault" DO NOTHING',
            user_id,
            DEFAULT_BOARD_NAME,
            dumps([c.model_dump() for c in DEFAULT_COLUMNS]),
        )
        row = await _find_default_board(user_id)
    if row is None:
        raise RuntimeError(f"Default board for user {user_id} was not created")
    return _board_info(row)


async def list_boards(user_id: str) -> list[BoardInfo]:
    """Home first, then the user's other boards in order."""
    default = await get_or_create_default_board(user_id)
    rows = await Board.prisma().find_many(
        where={"userId": user_id, "isDefault": False},
        order=[{"position": "asc"}, {"createdAt": "asc"}],
    )
    return [default, *(_board_info(row) for row in rows)]


async def resolve_board(user_id: str, board_ref: str) -> BoardInfo:
    """``"default"`` is the user's Home board; anything else is a board id."""
    if board_ref == DEFAULT_BOARD_REF:
        return await get_or_create_default_board(user_id)
    row = await Board.prisma().find_first(where={"id": board_ref, "userId": user_id})
    if row is None:
        raise BoardNotFoundError(f"Board {board_ref} not found")
    return _board_info(row)


async def create_board(user_id: str, name: str) -> BoardInfo:
    """A new board with the built-in columns, last in the user's order."""
    clean = validate_board_name(name)
    default = await get_or_create_default_board(user_id)
    async with db.transaction() as tx:
        # Locking Home queues a user's creates, so two cannot both pass the cap.
        await db.query_raw_with_schema(
            'SELECT 1 FROM {schema_prefix}"Board" WHERE "id" = $1 FOR UPDATE',
            default.id,
            client=tx,
        )
        [stats] = await db.query_raw_with_schema(
            'SELECT COUNT(*)::int AS "count", COALESCE(MAX("position"), 0) AS "last" '
            'FROM {schema_prefix}"Board" WHERE "userId" = $1',
            user_id,
            client=tx,
        )
        if stats["count"] >= MAX_BOARDS:
            raise BoardLimitError(f"You can have up to {MAX_BOARDS} projects")
        row = await Board.prisma(tx).create(
            data={
                "userId": user_id,
                "name": clean,
                "position": float(stats["last"]) + POSITION_GAP,
                "columns": SafeJson([c.model_dump() for c in DEFAULT_COLUMNS]),
            }
        )
    return _board_info(row)


async def update_board(
    user_id: str,
    board_id: str,
    *,
    name: str | None = None,
    position: float | None = None,
) -> BoardInfo:
    """Rename or reorder a board; ``None`` leaves that field as it is."""
    board = await resolve_board(user_id, board_id)
    data: BoardUpdateInput = {}
    if name is not None:
        if board.is_default:
            raise DefaultBoardError(f"{DEFAULT_BOARD_NAME} can't be renamed")
        data["name"] = validate_board_name(name)
    if position is not None:
        data["position"] = position
    if not data:
        return board
    row = await Board.prisma().update(where={"id": board.id}, data=data)
    if row is None:
        raise BoardNotFoundError(f"Board {board_id} not found")
    return _board_info(row)


async def delete_board(user_id: str, board_id: str) -> None:
    """Delete a board. Its cards go on top of Home's Backlog, in the order
    they had on the board."""
    board = await resolve_board(user_id, board_id)
    if board.is_default:
        raise DefaultBoardError(f"{DEFAULT_BOARD_NAME} can't be deleted")
    async with db.transaction() as tx:
        # The lock holds off a card moving onto the board until it is gone.
        locked = await db.query_raw_with_schema(
            'SELECT 1 FROM {schema_prefix}"Board" '
            'WHERE "id" = $1 AND "userId" = $2 FOR UPDATE',
            board.id,
            user_id,
            client=tx,
        )
        if not locked:
            raise BoardNotFoundError(f"Board {board_id} not found")
        await db.execute_raw_with_schema(
            "WITH home_top AS (SELECT "
            + _top_position_sql("$1", _on_board("b", None), "$3")
            + ' AS "position"), moved AS ('
            'SELECT "id", '
            'ROW_NUMBER() OVER (ORDER BY array_position($4::text[], "boardColumn") '
            'NULLS LAST, "boardPosition" ASC NULLS LAST, "updatedAt" DESC, "id" DESC'
            ') AS "rank", COUNT(*) OVER () AS "total" '
            'FROM {schema_prefix}"ChatSession" '
            'WHERE "userId" = $1 AND "boardId" = $2'
            ") "
            'UPDATE {schema_prefix}"ChatSession" s SET "boardId" = NULL, '
            '"boardColumn" = $3, "boardPosition" = home_top."position" - '
            f'{POSITION_GAP} * (moved."total" - moved."rank") '
            'FROM moved, home_top WHERE s."id" = moved."id"',
            user_id,
            board.id,
            BACKLOG,
            [c.key for c in board.columns],
            client=tx,
        )
        await db.execute_raw_with_schema(
            'DELETE FROM {schema_prefix}"Board" WHERE "id" = $1 AND "userId" = $2',
            board.id,
            user_id,
            client=tx,
        )


async def save_board_columns(
    user_id: str, board: BoardInfo, columns: list[BoardColumn]
) -> list[BoardColumn]:
    """Store a board's columns. Its cards in a column that is gone go to
    Backlog."""
    validated = validate_columns(columns)
    stored = SafeJson([c.model_dump() for c in validated])
    params: list[Any] = [user_id, [c.key for c in validated], BACKLOG]
    board_param = None
    if not board.is_default:
        params.append(board.id)
        board_param = "$4"
    async with db.transaction() as tx:
        updated = await Board.prisma(tx).update_many(
            where={"id": board.id, "userId": user_id}, data={"columns": stored}
        )
        if not updated:
            raise BoardNotFoundError(f"Board {board.id} not found")
        await db.execute_raw_with_schema(
            'UPDATE {schema_prefix}"ChatSession" cs SET "boardColumn" = $3 '
            f'WHERE cs."userId" = $1 AND {_on_board("cs", board_param)} '
            'AND cs."boardColumn" IS NOT NULL '
            'AND NOT (cs."boardColumn" = ANY($2::text[]))',
            *params,
            client=tx,
        )
    return validated


async def list_board_sessions(
    user_id: str,
    board: BoardInfo,
    *,
    organization_id: str | None,
    include_older: bool = False,
    now: datetime | None = None,
) -> list[BoardSessionRow]:
    """A board's cards. On Home: every placed session, plus unplaced ones
    touched in the last week; ``include_older`` adds the newest older
    unplaced sessions, which the board shows in Done. Other boards hold only
    the cards placed on them."""
    board_id = None if board.is_default else board.id
    base, params = _board_sessions_query(user_id, organization_id, board_id)
    if board_id is not None:
        rows = await db.query_raw_with_schema(
            base + f" ORDER BY {_BOARD_ORDER} LIMIT {BOARD_SESSION_LIMIT}",
            *params,
            model=PrismaChatSession,
        )
    else:
        params.append((now or datetime.now(UTC)) - UNPLACED_WINDOW)
        # Session timestamps are UTC wall-clock ``timestamp`` columns; the
        # driver binds a datetime as ``timestamptz``.
        cutoff = f"(${len(params)}::timestamptz AT TIME ZONE 'UTC')"
        rows = await db.query_raw_with_schema(
            base
            + f' AND ("boardColumn" IS NOT NULL OR "updatedAt" > {cutoff})'
            + f" ORDER BY {_BOARD_ORDER} LIMIT {BOARD_SESSION_LIMIT}",
            *params,
            model=PrismaChatSession,
        )
        if include_older:
            rows += await db.query_raw_with_schema(
                base
                + f' AND "boardColumn" IS NULL AND "updatedAt" <= {cutoff}'
                + ' ORDER BY "updatedAt" DESC, "id" DESC'
                + f" LIMIT {OLDER_SESSION_LIMIT}",
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
            board_id=row["board_id"],
            board_column=row["board_column"],
            board_position=row["board_position"],
            started=row["started"],
            has_messages=row["has_messages"],
            needs_you=row["needs_you"],
            last_line=first_sentence(row["last_assistant"]),
            file_count=row["file_count"],
        )
        for row in rows
    }


async def top_position(user_id: str, board_id: str | None, column: str) -> float:
    """A position above every card in *column* of a board (``None`` = Home),
    or 0.0 when it is empty."""
    params: list[Any] = [user_id, column]
    board_param = None
    if board_id is not None:
        params.append(board_id)
        board_param = "$3"
    rows = await db.query_raw_with_schema(
        "SELECT "
        + _top_position_sql("$1", _on_board("b", board_param), "$2")
        + ' AS "top"',
        *params,
    )
    return float(rows[0]["top"]) if rows else 0.0


async def set_session_board(
    session_id: str,
    user_id: str,
    *,
    column: str | None,
    position: float | None,
    board_id: str | None | _Unset = UNSET,
) -> bool:
    """Place a card; ``None`` leaves column or position as it is. *board_id*
    moves the card to that board (``None`` = Home); ``UNSET`` keeps it.

    Raw SQL so ``updatedAt`` stays put: moving a card is not activity in the
    chat. False when the session, or the board, is not the user's;
    ``BoardNotFoundError`` when the board is deleted while the card moves.
    """
    params: list[Any] = [session_id, user_id]
    assignments = []
    conditions = ['"id" = $1', '"userId" = $2']
    if column is not None:
        params.append(column)
        assignments.append(f'"boardColumn" = ${len(params)}::text')
    if position is not None:
        params.append(position)
        assignments.append(f'"boardPosition" = ${len(params)}::double precision')
    if board_id is None:
        assignments.append('"boardId" = NULL')
    elif board_id is not UNSET:
        params.append(board_id)
        assignments.append(f'"boardId" = ${len(params)}::text')
        conditions.append(
            'EXISTS (SELECT 1 FROM {schema_prefix}"Board" bd '
            f'WHERE bd."id" = ${len(params)}::text AND bd."userId" = $2)'
        )
    try:
        result = await db.execute_raw_with_schema(
            'UPDATE {schema_prefix}"ChatSession" SET '
            + ", ".join(assignments or ['"boardColumn" = "boardColumn"'])
            + " WHERE "
            + " AND ".join(conditions),
            *params,
        )
    except RawQueryError as e:
        if (e.meta or {}).get("code") != _FOREIGN_KEY_VIOLATION:
            raise
        raise BoardNotFoundError(f"Board {board_id} not found") from e
    return result > 0


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
    """Record where the task went and move the old card to the top of Done on
    its board.

    Merges only the two keys into metadata, like ``set_session_pending_question``,
    so a concurrent write to another key survives.
    """
    await db.execute_raw_with_schema(
        'UPDATE {schema_prefix}"ChatSession" cs SET "metadata" = '
        "COALESCE(cs.\"metadata\", '{{}}'::jsonb) || "
        "jsonb_build_object("
        "'reassigned_to_session_id', $3::text, "
        "'reassigned_to_expert_id', $4::text"
        '), "boardColumn" = $5::text, "boardPosition" = '
        # Done on the card's own board.
        + _top_position_sql(
            "$2", 'b."boardId" IS NOT DISTINCT FROM cs."boardId"', "$5::text"
        )
        + ' WHERE cs."id" = $1 AND cs."userId" = $2',
        session_id,
        user_id,
        to_session_id,
        to_expert_id,
        DONE,
    )


async def list_session_files(user_id: str, session_id: str) -> list[tuple[str, str]]:
    """``(name, file_id)`` of the live files *user_id* keeps under a session."""
    rows = await db.query_raw_with_schema(
        'SELECT f."name", f."id" FROM {schema_prefix}"UserWorkspaceFile" f '
        'JOIN {schema_prefix}"UserWorkspace" w ON w."id" = f."workspaceId" '
        'WHERE w."userId" = $1 AND f."isDeleted" = FALSE '
        "AND f.\"path\" LIKE '/sessions/' || $2::text || '/%' "
        'ORDER BY f."createdAt" ASC, f."id" ASC',
        user_id,
        session_id,
    )
    return [(row["name"], row["id"]) for row in rows]


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


async def _find_default_board(user_id: str) -> Board | None:
    return await Board.prisma().find_first(where={"userId": user_id, "isDefault": True})


def _board_info(row: Board) -> BoardInfo:
    return BoardInfo(
        id=row.id,
        name=row.name,
        is_default=row.isDefault,
        position=row.position,
        columns=_board_columns(row),
    )


def _board_columns(row: Board) -> list[BoardColumn]:
    try:
        return validate_columns([BoardColumn.model_validate(c) for c in row.columns])
    except (TypeError, ValueError) as e:
        logger.warning(f"Unreadable columns on board {row.id}: {e}")
        return _default_columns()


def _on_board(alias: str, board_param: str | None) -> str:
    """Two shapes, not ``IS NOT DISTINCT FROM``, so the board index is used."""
    if board_param is None:
        return f'{alias}."boardId" IS NULL'
    return f'{alias}."boardId" = {board_param}'


def _board_sessions_query(
    user_id: str, organization_id: str | None, board_id: str | None
) -> tuple[str, list[Any]]:
    """The board's session filter; ``board_id`` ``None`` is Home."""
    params: list[Any] = [user_id]
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
    if board_id is None:
        conditions.append('"boardId" IS NULL')
    else:
        params.append(board_id)
        conditions.append(f'"boardId" = ${len(params)}')
    query = 'SELECT * FROM {schema_prefix}"ChatSession" WHERE ' + " AND ".join(
        conditions
    )
    return query, params


def _board_session_row(row: PrismaChatSession, extra: BoardExtras) -> BoardSessionRow:
    return BoardSessionRow(
        session=ChatSessionInfo.from_db(row),
        board_id=row.boardId,
        board_column=row.boardColumn,
        board_position=row.boardPosition,
        started=extra.started,
        has_messages=extra.has_messages,
        needs_you=extra.needs_you,
        last_line=extra.last_line,
        file_count=extra.file_count,
    )


def _top_position_sql(user_param: str, board_condition: str, column_param: str) -> str:
    return (
        f'(SELECT COALESCE(MIN(b."boardPosition") - {POSITION_GAP}, 0) '
        'FROM {schema_prefix}"ChatSession" b '
        f'WHERE b."userId" = {user_param} AND {board_condition} '
        f'AND b."boardColumn" = {column_param})'
    )
