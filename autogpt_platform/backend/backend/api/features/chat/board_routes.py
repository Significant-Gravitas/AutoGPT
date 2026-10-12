"""Task board routes: the board, its columns, and placing, assigning and
starting cards. Served under ``/api/chat`` next to the session routes."""

from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Query, Security
from pydantic import BaseModel, Field

from backend.api.features.chat.routes import (
    SessionSummaryResponse,
    _validate_and_get_session,
    _validate_and_get_writable_session,
    _validate_session_expert_writable_by_user,
    processing_session_ids,
    session_summary,
)
from backend.api.features.chat.server_turn import start_board_card
from backend.api.features.experts import experts_db
from backend.copilot.board import BACKLOG, IN_PROGRESS, BoardColumn, BoardColumnsError
from backend.copilot.board_db import (
    BoardExtras,
    BoardSessionRow,
    get_board_columns,
    get_board_session_extras,
    list_board_sessions,
    save_board_columns,
    set_session_board,
    set_session_expert_if_unstarted,
    top_position,
)
from backend.copilot.constants import AUTOPILOT_NAME
from backend.copilot.model import invalidate_session_cache

router = APIRouter(tags=["chat"])


class BoardResponse(BaseModel):
    columns: list[BoardColumn]
    sessions: list[SessionSummaryResponse]


class UpdateBoardColumnsRequest(BaseModel):
    columns: list[BoardColumn]


class BoardColumnsResponse(BaseModel):
    columns: list[BoardColumn]


class UpdateSessionBoardRequest(BaseModel):
    """Where a card goes. ``None`` leaves that field as it is."""

    column: str | None = Field(default=None, max_length=64)
    position: float | None = Field(default=None, allow_inf_nan=False)


class UpdateSessionExpertRequest(BaseModel):
    """Who works a card; ``None`` is Otto."""

    expert_id: str | None = Field(max_length=128)


class StartSessionRequest(BaseModel):
    """The first message; ``None`` sends the card's title."""

    message: str | None = Field(default=None, max_length=10000)


@router.get(
    "/board",
    summary="Get board",
    dependencies=[Security(auth.requires_user)],
)
async def get_board(
    user_id: Annotated[str, Security(auth.get_user_id)],
    ctx: Annotated[auth.RequestContext, Security(auth.get_request_context)],
    include_older: bool = Query(
        default=False,
        description="Also return unplaced sessions older than a week.",
    ),
) -> BoardResponse:
    """The user's columns and cards: every placed session, plus unplaced ones
    touched in the last week."""
    columns = await get_board_columns(user_id)
    rows = await list_board_sessions(
        user_id, organization_id=ctx.org_id, include_older=include_older
    )
    processing = await processing_session_ids([row.session.session_id for row in rows])
    expert_names = await _reassigned_expert_names(user_id, rows)
    return BoardResponse(
        columns=columns,
        sessions=[
            session_summary(
                row.session,
                is_processing=row.session.session_id in processing,
                extras=_row_extras(row, expert_names),
            )
            for row in rows
        ],
    )


@router.put(
    "/board/columns",
    summary="Update board columns",
    dependencies=[Security(auth.requires_user)],
    responses={422: {"description": "The columns cannot be stored"}},
)
async def update_board_columns(
    request: UpdateBoardColumnsRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> BoardColumnsResponse:
    """Replace the user's columns. Cards in a removed column go to Backlog."""
    try:
        columns = await save_board_columns(user_id, request.columns)
    except BoardColumnsError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return BoardColumnsResponse(columns=columns)


@router.patch(
    "/sessions/{session_id}/board",
    summary="Update session board",
    dependencies=[Security(auth.requires_user)],
    responses={
        404: {"description": "Session not found or access denied"},
        422: {"description": "Unknown board column"},
    },
)
async def update_session_board(
    session_id: str,
    request: UpdateSessionBoardRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
    ctx: Annotated[auth.RequestContext, Security(auth.get_request_context)],
) -> SessionSummaryResponse:
    """Move a card. A card leaving Backlog that has not started yet starts
    with its title as the first message."""
    session = await _validate_and_get_session(session_id, user_id)
    column = request.column
    position = request.position
    if column is not None:
        if column not in {c.key for c in await get_board_columns(user_id)}:
            raise HTTPException(status_code=422, detail="Unknown board column")
        if position is None:
            position = await top_position(user_id, column)

    starts = (
        column is not None
        and column != BACKLOG
        and bool((session.title or "").strip())
        and not (await _session_extras(user_id, session_id)).started
    )
    if starts:
        await _validate_session_expert_writable_by_user(session, user_id)

    if not await set_session_board(
        session_id, user_id, column=column, position=position
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found or access denied",
        )
    if starts:
        await start_board_card(session, user_id, org_id=ctx.org_id, team_id=ctx.team_id)
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
        expert = await experts_db.get_expert(
            user_id, expert_id, include_workflows=False
        )
        if (
            expert is None
            or expert.is_archived
            or expert.schedules_paused_at is not None
        ):
            raise HTTPException(status_code=404, detail="Expert not found")
        try:
            organization_id, team_id = await experts_db.resolve_private_expert_tenancy(
                user_id, expert_id
            )
        except experts_db.ExpertNotFoundError as e:
            raise HTTPException(status_code=404, detail="Expert not found") from e
        except experts_db.ExpertPrivateTenancyNotFoundError as e:
            raise HTTPException(
                status_code=503,
                detail="Your expert workspace is still being set up. Try again shortly.",
            ) from e

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
        raise HTTPException(status_code=404, detail="Expert not found")
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
        409: {"description": "The task has already started"},
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
    if (await _session_extras(user_id, session_id)).started:
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
        position=await top_position(user_id, IN_PROGRESS),
    )
    return await _fresh_summary(session_id, user_id)


async def _session_extras(user_id: str, session_id: str) -> BoardExtras:
    extras = await get_board_session_extras(user_id, [session_id])
    return extras.get(session_id) or BoardExtras()


async def _fresh_summary(session_id: str, user_id: str) -> SessionSummaryResponse:
    session = await _validate_and_get_session(session_id, user_id)
    processing = await processing_session_ids([session_id])
    return session_summary(
        session,
        is_processing=session_id in processing,
        extras=await _session_extras(user_id, session_id),
    )


async def _reassigned_expert_names(
    user_id: str, rows: list[BoardSessionRow]
) -> dict[str, str]:
    if not any(
        row.session.metadata.reassigned_to_session_id
        and row.session.metadata.reassigned_to_expert_id
        for row in rows
    ):
        return {}
    experts = await experts_db.list_experts(user_id, with_metrics=False)
    return {expert.id: expert.name for expert in experts}


def _row_extras(row: BoardSessionRow, expert_names: dict[str, str]) -> BoardExtras:
    last_line = row.last_line
    metadata = row.session.metadata
    if metadata.reassigned_to_session_id:
        expert_id = metadata.reassigned_to_expert_id
        name = AUTOPILOT_NAME if expert_id is None else expert_names.get(expert_id)
        # An expert fired since then has no name to show.
        last_line = f"Reassigned to {name}" if name else "Reassigned"
    return BoardExtras(
        board_column=row.board_column,
        board_position=row.board_position,
        started=row.started,
        needs_you=row.needs_you,
        last_line=last_line,
        file_count=row.file_count,
    )
