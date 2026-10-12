"""Task board routes: the user's boards, their columns, and placing,
moving, assigning and starting cards. Served under ``/api/chat`` next to
the session routes."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Query, Response, Security
from pydantic import BaseModel, Field

from backend.api.features.chat.routes import (
    SessionSummaryResponse,
    _validate_and_get_session,
    _validate_and_get_writable_session,
    _validate_session_expert_writable_by_user,
    session_summaries,
)
from backend.api.features.chat.server_turn import (
    dispatch_server_turn,
    start_board_card,
    turn_in_flight,
)
from backend.api.features.experts import experts_db
from backend.api.features.experts.models import Expert
from backend.copilot.board import (
    BACKLOG,
    DONE,
    IN_PROGRESS,
    BoardColumn,
    BoardColumnsError,
    BoardNameError,
)
from backend.copilot.board_db import (
    DEFAULT_BOARD_REF,
    UNSET,
    BoardExtras,
    BoardInfo,
    BoardLimitError,
    BoardNotFoundError,
    DefaultBoardError,
    create_board,
    delete_board,
    get_board_session_extras,
    list_board_sessions,
    list_boards,
    list_session_files,
    mark_session_reassigned,
    resolve_board,
    save_board_columns,
    set_session_board,
    set_session_expert_if_unstarted,
    top_position,
    update_board,
)
from backend.copilot.board_reassign import build_reassign_brief, reassign_brief_metadata
from backend.copilot.constants import AUTOPILOT_NAME
from backend.copilot.expert_kickoff import scoped_client_message_id
from backend.copilot.model import (
    CHAT_STATUS_IDLE,
    create_chat_session,
    delete_chat_session,
    get_chat_session,
    invalidate_session_cache,
    update_session_title,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


class BoardSummary(BaseModel):
    id: str
    name: str
    is_default: bool
    position: float


class BoardListResponse(BaseModel):
    boards: list[BoardSummary]


class CreateBoardRequest(BaseModel):
    name: str


class UpdateBoardRequest(BaseModel):
    """A new name or place; ``None`` leaves that field as it is."""

    name: str | None = None
    position: float | None = Field(default=None, allow_inf_nan=False)


class BoardResponse(BaseModel):
    board: BoardSummary
    columns: list[BoardColumn]
    sessions: list[SessionSummaryResponse]


class UpdateBoardColumnsRequest(BaseModel):
    columns: list[BoardColumn]


class BoardColumnsResponse(BaseModel):
    columns: list[BoardColumn]


class UpdateSessionBoardRequest(BaseModel):
    """Where a card goes. ``None`` leaves that field as it is; ``board_id``
    is ``"default"`` or a board id."""

    column: str | None = Field(default=None, max_length=64)
    position: float | None = Field(default=None, allow_inf_nan=False)
    board_id: str | None = Field(default=None, min_length=1, max_length=128)


class UpdateSessionExpertRequest(BaseModel):
    """Who works a card; ``None`` is Otto."""

    expert_id: str | None = Field(max_length=128)


class StartSessionRequest(BaseModel):
    """The first message; ``None`` sends the card's title."""

    message: str | None = Field(default=None, max_length=10000)


class ReassignSessionRequest(BaseModel):
    """Who takes the task over; ``None`` is Otto."""

    expert_id: str | None = Field(max_length=128)


class ReassignSessionResponse(BaseModel):
    old_session: SessionSummaryResponse
    new_session: SessionSummaryResponse


@router.get(
    "/boards",
    summary="List boards",
    dependencies=[Security(auth.requires_user)],
)
async def get_boards(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> BoardListResponse:
    """The user's boards: the default board first, then the rest in order."""
    return BoardListResponse(
        boards=[_board_summary(board) for board in await list_boards(user_id)]
    )


@router.post(
    "/boards",
    summary="Create board",
    dependencies=[Security(auth.requires_user)],
    responses={422: {"description": "Bad name, or too many projects"}},
)
async def post_board(
    request: CreateBoardRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> BoardSummary:
    """A new board with the built-in columns, last in the user's order."""
    with _board_errors():
        return _board_summary(await create_board(user_id, request.name))


@router.patch(
    "/boards/{board_id}",
    summary="Update board",
    dependencies=[Security(auth.requires_user)],
    responses={
        404: {"description": "Project not found"},
        409: {"description": "The default board cannot be renamed"},
        422: {"description": "Bad name"},
    },
)
async def patch_board(
    board_id: str,
    request: UpdateBoardRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> BoardSummary:
    """Rename or reorder a board."""
    with _board_errors():
        board = await update_board(
            user_id, board_id, name=request.name, position=request.position
        )
    return _board_summary(board)


@router.delete(
    "/boards/{board_id}",
    summary="Delete board",
    status_code=204,
    dependencies=[Security(auth.requires_user)],
    responses={
        404: {"description": "Project not found"},
        409: {"description": "The default board cannot be deleted"},
    },
)
async def remove_board(
    board_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> Response:
    """Delete a board. Its cards go on top of the default board's Backlog."""
    with _board_errors():
        await delete_board(user_id, board_id)
    return Response(status_code=204)


@router.get(
    "/boards/{board_id}",
    summary="Get board",
    dependencies=[Security(auth.requires_user)],
    responses={404: {"description": "Project not found"}},
)
async def get_board(
    board_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
    ctx: Annotated[auth.RequestContext, Security(auth.get_request_context)],
    include_older: bool = Query(
        default=False,
        description="On the default board, also return unplaced sessions "
        "older than a week.",
    ),
) -> BoardResponse:
    """A board's columns and cards. The default board (``default``) also
    shows unplaced sessions touched in the last week."""
    board = await _resolve_board(user_id, board_id)
    rows = await list_board_sessions(
        user_id, board, organization_id=ctx.org_id, include_older=include_older
    )
    extras = {
        row.session.session_id: BoardExtras(**row.model_dump(exclude={"session"}))
        for row in rows
    }
    return BoardResponse(
        board=_board_summary(board),
        columns=board.columns,
        sessions=await session_summaries(
            user_id, [row.session for row in rows], extras
        ),
    )


@router.put(
    "/boards/{board_id}/columns",
    summary="Update board columns",
    dependencies=[Security(auth.requires_user)],
    responses={
        404: {"description": "Project not found"},
        422: {"description": "The columns cannot be stored"},
    },
)
async def update_board_columns(
    board_id: str,
    request: UpdateBoardColumnsRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> BoardColumnsResponse:
    """Replace a board's columns. Its cards in a removed column go to
    Backlog."""
    with _board_errors():
        board = await resolve_board(user_id, board_id)
        columns = await save_board_columns(user_id, board, request.columns)
    return BoardColumnsResponse(columns=columns)


@router.patch(
    "/sessions/{session_id}/board",
    summary="Update session board",
    dependencies=[Security(auth.requires_user)],
    responses={
        404: {"description": "Session or project not found"},
        422: {"description": "Unknown board column"},
    },
)
async def update_session_board(
    session_id: str,
    request: UpdateSessionBoardRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
    ctx: Annotated[auth.RequestContext, Security(auth.get_request_context)],
) -> SessionSummaryResponse:
    """Move a card, within its board or onto ``board_id``. A card moved to
    another board keeps its column if that board has it, else goes to
    Backlog, and goes on top. A card that has not started yet starts with
    its title as the first message when it moves to any column but Backlog
    or Done."""
    session = await _validate_and_get_session(session_id, user_id)
    extras = await _session_extras(user_id, session_id)
    column = request.column
    position = request.position
    moving = request.board_id is not None
    target = UNSET
    if moving or column is not None:
        board = await _resolve_board(
            user_id, request.board_id or extras.board_id or DEFAULT_BOARD_REF
        )
        keys = {c.key for c in board.columns}
        if column is None:
            column = extras.board_column if extras.board_column in keys else BACKLOG
        elif column not in keys:
            raise HTTPException(status_code=422, detail="Unknown board column")
        board_id = None if board.is_default else board.id
        if moving:
            target = board_id
        if position is None:
            position = await top_position(user_id, board_id, column)

    starts = (
        column is not None
        and column not in (BACKLOG, DONE)
        and bool((session.title or "").strip())
        and session.chat_status == CHAT_STATUS_IDLE
        and not extras.started
    )
    if starts:
        await _validate_session_expert_writable_by_user(session, user_id)
        # Start first: a refused turn leaves the card where it was.
        await start_board_card(session, user_id, org_id=ctx.org_id, team_id=ctx.team_id)

    with _board_errors():
        placed = await set_session_board(
            session_id, user_id, column=column, position=position, board_id=target
        )
    if not placed:
        if moving:
            # The session was checked above: its new board is gone.
            raise _board_not_found()
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found or access denied",
        )
    return await _fresh_summary(session_id, user_id)


@router.patch(
    "/sessions/{session_id}/expert",
    summary="Update session expert",
    dependencies=[Security(auth.requires_user)],
    responses={
        404: {"description": "Session or expert not found"},
        409: {"description": "The task has already started"},
        503: {"description": "The expert's workspace is still being set up"},
    },
)
async def update_session_expert(
    session_id: str,
    request: UpdateSessionExpertRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
    ctx: Annotated[auth.RequestContext, Security(auth.get_request_context)],
) -> SessionSummaryResponse:
    """Change who works a card, only until the user has written in it."""
    expert_id = request.expert_id
    if expert_id is None:
        organization_id, team_id = ctx.org_id, ctx.team_id
    else:
        await _available_expert(user_id, expert_id)
        try:
            organization_id, team_id = await experts_db.resolve_private_expert_tenancy(
                user_id, expert_id
            )
        except experts_db.ExpertNotFoundError as e:
            raise _expert_not_found() from e
        except experts_db.ExpertPrivateTenancyNotFoundError as e:
            raise _expert_workspace_pending() from e

    result = await set_session_expert_if_unstarted(
        session_id,
        user_id,
        expert_id,
        organization_id=organization_id,
        team_id=team_id,
    )
    if result == "started":
        raise HTTPException(
            status_code=409,
            detail="Assignee can only change before the task starts",
        )
    if result == "expert_not_found":
        raise _expert_not_found()
    if result == "not_found":
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found or access denied",
        )
    # Turns read the session's expert from the cache.
    await invalidate_session_cache(session_id)
    return await _fresh_summary(session_id, user_id)


@router.post(
    "/sessions/{session_id}/start",
    summary="Start session",
    dependencies=[Security(auth.requires_user)],
    responses={
        404: {"description": "Session not found or access denied"},
        409: {"description": "The task has already started or is working"},
        422: {"description": "Nothing to send: no message and no title"},
    },
)
async def start_session(
    session_id: str,
    request: StartSessionRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
    ctx: Annotated[auth.RequestContext, Security(auth.get_request_context)],
) -> SessionSummaryResponse:
    """Start a card: send *message* (or the title) and move it to the top of
    In progress. The title stays as it is."""
    session = await _validate_and_get_writable_session(session_id, user_id)
    extras = await _session_extras(user_id, session_id)
    if session.chat_status != CHAT_STATUS_IDLE or extras.started:
        raise HTTPException(status_code=409, detail="The task has already started")
    # Start first: a refused turn leaves the card where it was.
    await start_board_card(
        session,
        user_id,
        org_id=ctx.org_id,
        team_id=ctx.team_id,
        message=request.message,
    )
    await set_session_board(
        session_id,
        user_id,
        column=IN_PROGRESS,
        position=await top_position(user_id, extras.board_id, IN_PROGRESS),
    )
    return await _fresh_summary(session_id, user_id)


@router.post(
    "/sessions/{session_id}/reassign",
    summary="Reassign session",
    dependencies=[Security(auth.requires_user)],
    responses={
        404: {"description": "Session or expert not found"},
        409: {"description": "The task is working, not started, or reassigned"},
        422: {"description": "The task is already assigned to them"},
        503: {"description": "The expert's workspace is still being set up"},
    },
)
async def reassign_session(
    session_id: str,
    request: ReassignSessionRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
    ctx: Annotated[auth.RequestContext, Security(auth.get_request_context)],
) -> ReassignSessionResponse:
    """Hand a started task to another assignee in a new chat.

    The new chat opens with a hidden brief of the old one and can read its
    files; the old card moves to Done.
    """
    old = await _validate_and_get_writable_session(session_id, user_id)
    if old.metadata.reassigned_to_session_id:
        raise HTTPException(
            status_code=409, detail="This task has already been reassigned"
        )
    if old.chat_status != CHAT_STATUS_IDLE or await turn_in_flight(session_id):
        raise HTTPException(
            status_code=409, detail="Reassign the task once it stops working"
        )
    old_extras = await _session_extras(user_id, session_id)
    if not old_extras.started:
        raise HTTPException(
            status_code=409,
            detail="The task has not started; change its assignee instead",
        )
    expert_id = request.expert_id
    if expert_id == old.expert_id:
        raise HTTPException(
            status_code=422, detail="The task is already assigned to them"
        )
    from_name = await _assignee_name(user_id, old.expert_id)
    to_name = AUTOPILOT_NAME
    if expert_id is not None:
        to_name = (await _available_expert(user_id, expert_id)).name

    try:
        new = await create_chat_session(
            user_id,
            dry_run=old.metadata.dry_run,
            organization_id=ctx.org_id,
            team_id=ctx.team_id,
            llm_auth_provider=old.metadata.llm_auth_provider,
            llm_credential_id=old.metadata.llm_credential_id,
            expert_id=expert_id,
            kind="reassigned",
            # Only the session checked above: its files become readable there.
            reassigned_from_session_id=old.session_id,
            reassigned_from_expert_id=old.expert_id,
        )
    except experts_db.ExpertNotFoundError as e:
        raise _expert_not_found() from e
    except experts_db.ExpertPrivateTenancyNotFoundError as e:
        raise _expert_workspace_pending() from e
    new_id = new.session_id

    try:
        if old.title:
            await update_session_title(new_id, user_id, old.title)
        # The new card goes on the old card's board.
        with _board_errors():
            placed = await set_session_board(
                new_id,
                user_id,
                column=IN_PROGRESS,
                position=await top_position(user_id, old_extras.board_id, IN_PROGRESS),
                board_id=old_extras.board_id,
            )
        if not placed:
            raise _board_not_found()
        history = await get_chat_session(session_id, user_id)
        brief = build_reassign_brief(
            from_name=from_name,
            to_name=to_name,
            messages=history.messages if history else [],
            files=await list_session_files(user_id, session_id),
        )
        await dispatch_server_turn(
            new,
            user_id,
            org_id=ctx.org_id,
            team_id=ctx.team_id,
            message=brief,
            message_id=scoped_client_message_id(user_id, new_id, "reassign-brief"),
            message_metadata=reassign_brief_metadata(expert_id),
        )
    except Exception:
        # A refused turn (paywall, limits) must not leave a half-made card.
        try:
            await delete_chat_session(new_id, user_id)
        except Exception:
            logger.exception(f"Failed to delete half-made reassign session {new_id}")
        raise

    await mark_session_reassigned(
        session_id, user_id, to_session_id=new_id, to_expert_id=expert_id
    )
    await invalidate_session_cache(session_id)
    return ReassignSessionResponse(
        old_session=await _fresh_summary(session_id, user_id),
        new_session=await _fresh_summary(new_id, user_id),
    )


def _board_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Project not found")


@contextmanager
def _board_errors() -> Iterator[None]:
    """The board layer's refusals as HTTP errors."""
    try:
        yield
    except BoardNotFoundError as e:
        raise _board_not_found() from e
    except DefaultBoardError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except (BoardNameError, BoardLimitError, BoardColumnsError) as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


async def _resolve_board(user_id: str, board_ref: str) -> BoardInfo:
    with _board_errors():
        return await resolve_board(user_id, board_ref)


def _board_summary(board: BoardInfo) -> BoardSummary:
    return BoardSummary(
        id=board.id,
        name=board.name,
        is_default=board.is_default,
        position=board.position,
    )


def _expert_not_found() -> HTTPException:
    return HTTPException(status_code=404, detail="Expert not found")


def _expert_workspace_pending() -> HTTPException:
    return HTTPException(
        status_code=503,
        detail="Your expert workspace is still being set up. Try again shortly.",
    )


async def _available_expert(user_id: str, expert_id: str) -> Expert:
    """The expert, unless it is gone, archived or paused (404)."""
    expert = await experts_db.get_expert(user_id, expert_id, include_workflows=False)
    if expert is None or expert.is_archived or expert.schedules_paused_at is not None:
        raise _expert_not_found()
    return expert


async def _assignee_name(user_id: str, expert_id: str | None) -> str:
    if expert_id is None:
        return AUTOPILOT_NAME
    expert = await experts_db.get_expert(user_id, expert_id, include_workflows=False)
    return expert.name if expert else "another expert"


async def _session_extras(user_id: str, session_id: str) -> BoardExtras:
    extras = await get_board_session_extras(user_id, [session_id])
    return extras.get(session_id) or BoardExtras()


async def _fresh_summary(session_id: str, user_id: str) -> SessionSummaryResponse:
    session = await _validate_and_get_session(session_id, user_id)
    extras = await get_board_session_extras(user_id, [session_id])
    [summary] = await session_summaries(user_id, [session], extras)
    return summary
