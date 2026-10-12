"""Task board routes: the board, its columns, and placing, assigning and
starting cards. Served under ``/api/chat`` next to the session routes."""

import logging
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
from backend.api.features.chat.server_turn import dispatch_server_turn, start_board_card
from backend.api.features.experts import experts_db
from backend.copilot.board import BACKLOG, IN_PROGRESS, BoardColumn, BoardColumnsError
from backend.copilot.board_db import (
    BoardExtras,
    BoardSessionRow,
    get_board_columns,
    get_board_session_extras,
    list_board_sessions,
    list_session_files,
    mark_session_reassigned,
    save_board_columns,
    set_session_board,
    set_session_expert_if_unstarted,
    top_position,
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
from backend.copilot.pending_message_helpers import (
    StreamRegistryUnavailable,
    is_turn_in_flight,
)

logger = logging.getLogger(__name__)

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


class ReassignSessionRequest(BaseModel):
    """Who takes the task over; ``None`` is Otto."""

    expert_id: str | None = Field(max_length=128)


class ReassignSessionResponse(BaseModel):
    old_session: SessionSummaryResponse
    new_session: SessionSummaryResponse


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
        and session.chat_status == CHAT_STATUS_IDLE
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
    if (
        session.chat_status != CHAT_STATUS_IDLE
        or (await _session_extras(user_id, session_id)).started
    ):
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
    if old.chat_status != CHAT_STATUS_IDLE or await _turn_in_flight(session_id):
        raise HTTPException(
            status_code=409, detail="Reassign the task once it stops working"
        )
    if not (await _session_extras(user_id, session_id)).started:
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
        expert = await experts_db.get_expert(
            user_id, expert_id, include_workflows=False
        )
        if (
            expert is None
            or expert.is_archived
            or expert.schedules_paused_at is not None
        ):
            raise HTTPException(status_code=404, detail="Expert not found")
        to_name = expert.name

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
        raise HTTPException(status_code=404, detail="Expert not found") from e
    except experts_db.ExpertPrivateTenancyNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Your expert workspace is still being set up. Try again shortly.",
        ) from e
    new_id = new.session_id

    try:
        if old.title:
            await update_session_title(new_id, user_id, old.title)
        await set_session_board(
            new_id,
            user_id,
            column=IN_PROGRESS,
            position=await top_position(user_id, IN_PROGRESS),
        )
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
    old_summary = await _fresh_summary(session_id, user_id)
    return ReassignSessionResponse(
        old_session=old_summary.model_copy(
            update={"last_line": f"Reassigned to {to_name}"}
        ),
        new_session=await _fresh_summary(new_id, user_id),
    )


async def _turn_in_flight(session_id: str) -> bool:
    try:
        return await is_turn_in_flight(session_id)
    except StreamRegistryUnavailable as e:
        raise HTTPException(
            status_code=503,
            detail="Chat service degraded, retry shortly",
            headers={"Retry-After": "30"},
        ) from e


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
