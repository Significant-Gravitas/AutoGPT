"""Resume a paused task, or send a written message, when the budget resets.

Served under ``/api/chat`` next to the other session routes. When to fire, and
whether to still fire then, live in ``backend.copilot.at_reset``; the job is a
scheduled copilot turn, at most one of each kind per chat.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, Depends, HTTPException, Response, Security
from pydantic import BaseModel, Field, model_validator

from backend.api.features.chat.routes import (
    _validate_and_get_session,
    _validate_and_get_writable_session,
)
from backend.copilot.at_reset import (
    AtResetKind,
    NoResetAhead,
    at_reset_schedule_id,
    next_reset_run_at,
)
from backend.copilot.at_reset_lock import at_reset_job_lock
from backend.copilot.continuation import RESUMED_AT_RESET_NOTE, as_model_prompt
from backend.copilot.model import ChatSessionInfo
from backend.copilot.rate_limit import RateLimitUnavailable, enforce_payment_paywall
from backend.executor.scheduler import CopilotTurnJobInfo, SchedulerClient
from backend.util.clients import get_scheduler_client

router = APIRouter(tags=["chat"])

# The same cap as a message sent now (``StreamChatRequest``), for one
# message and for messages joined while waiting for the reset.
_MAX_MESSAGE_CHARS = 64_000

# What a resume job keeps as its message, for the Scheduled page: the note it
# fires carries a marker that page would show raw. It always fires the note.
_RESUME_DESCRIPTION = as_model_prompt(RESUMED_AT_RESET_NOTE)


class AtResetRequest(BaseModel):
    """A turn to run when the budget resets."""

    kind: AtResetKind = Field(
        description='"resume" picks the paused task back up; "send" sends '
        "``message`` as the user's.",
    )
    message: str | None = Field(
        default=None,
        max_length=_MAX_MESSAGE_CHARS,
        description="What to send. Required for send, ignored for resume.",
    )

    @model_validator(mode="after")
    def send_has_a_message(self) -> "AtResetRequest":
        if self.kind == "send" and not (self.message and self.message.strip()):
            raise ValueError("A send needs a message")
        return self


class AtResetTurn(BaseModel):
    """A turn set to run when the budget resets."""

    kind: AtResetKind
    run_at: datetime
    # What a send will send; None for a resume.
    message: str | None = None


class AtResetTurnsResponse(BaseModel):
    """The chat's turns set for the budget reset, at most one of each kind."""

    jobs: list[AtResetTurn]


@router.post(
    "/sessions/{session_id}/at-reset",
    summary="Schedule turn at budget reset",
    dependencies=[Security(auth.requires_user), Depends(enforce_payment_paywall)],
    responses={
        404: {"description": "Session not found, or not writable"},
        409: {
            "description": '"not_out_of_budget": nothing to wait for, so carry '
            'on now. "no_reset": the allowance never refills (a trial).'
        },
        413: {
            "description": '"message_too_long": the message waiting plus '
            "this one would be longer than a single message may be"
        },
        503: {"description": "Usage could not be read; retry shortly"},
    },
)
async def schedule_at_reset(
    session_id: str,
    request: AtResetRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
    ctx: Annotated[auth.RequestContext, Security(auth.get_request_context)],
) -> AtResetTurn:
    """Resume the chat's paused task, or send a message, when the budget resets.

    It fires just after the exhausted window that refills last. A chat has
    at most one of each: a resume set again replaces the earlier one, and a
    send joins the message already waiting, so nothing written meanwhile is
    lost. Sending a message in the chat cancels a pending resume.
    """
    session = await _validate_and_get_writable_session(session_id, user_id)
    run_at = await _run_at(user_id, session, request.kind)
    job_id = at_reset_schedule_id(request.kind, session_id)
    async with _holding(job_id):
        message = (
            await _joined_with_waiting_send(user_id, session_id, request.message)
            if request.kind == "send" and request.message
            else None
        )
        await _scheduler().add_copilot_turn_schedule(
            user_id=user_id,
            session_id=session_id,
            message=message or _RESUME_DESCRIPTION,
            run_at=run_at,
            # The budget windows reset in UTC.
            user_timezone="UTC",
            # Anchored on the session, as for a message sent now.
            organization_id=session.organization_id or ctx.org_id,
            team_id=session.team_id if session.organization_id else ctx.team_id,
            expert_id=session.expert_id,
            purpose=request.kind,
            schedule_id=job_id,
        )
    return AtResetTurn(kind=request.kind, run_at=run_at, message=message)


@router.get(
    "/sessions/{session_id}/at-reset",
    summary="List turns at budget reset",
    dependencies=[Security(auth.requires_user)],
    responses={404: {"description": "Session not found or access denied"}},
)
async def list_at_reset(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> AtResetTurnsResponse:
    """The chat's pending resume and send at the budget reset."""
    await _validate_and_get_session(session_id, user_id)
    turns = [_as_turn(job) for job in await _chat_turns(user_id, session_id)]
    return AtResetTurnsResponse(jobs=[t for t in turns if t is not None])


@router.delete(
    "/sessions/{session_id}/at-reset/{kind}",
    summary="Cancel turn at budget reset",
    dependencies=[Security(auth.requires_user)],
    status_code=204,
    responses={404: {"description": "Session not found or access denied"}},
)
async def cancel_at_reset(
    session_id: str,
    kind: AtResetKind,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> Response:
    """Cancel the chat's resume or send at the budget reset; one that isn't
    set is already cancelled."""
    await _validate_and_get_session(session_id, user_id)
    async with _holding(at_reset_schedule_id(kind, session_id)):
        await _scheduler().cancel_at_reset_turn(
            user_id=user_id, session_id=session_id, kind=kind
        )
    return Response(status_code=204)


async def _run_at(
    user_id: str, session: ChatSessionInfo, kind: AtResetKind
) -> datetime:
    """When the turn fires, or the 409 or 503 that says why it can't be set."""
    if session.metadata.llm_auth_provider != "platform":
        # Only the platform route spends this budget, so the chat isn't
        # waiting on it and can carry on now.
        raise HTTPException(status_code=409, detail="not_out_of_budget")
    try:
        return await next_reset_run_at(user_id, session.session_id, kind)
    except NoResetAhead as e:
        raise HTTPException(status_code=409, detail=e.reason) from e
    except RateLimitUnavailable as e:
        raise _service_degraded() from e


@asynccontextmanager
async def _holding(job_id: str) -> AsyncIterator[None]:
    """Hold the chat's job from reading it to setting it, on any replica.

    Two tabs, or the scheduler putting the job back after it ran, would
    otherwise each write over the other, and a send written meanwhile would
    be lost. A 503 rather than going on without it.
    """
    async with at_reset_job_lock(job_id) as locked:
        if not locked:
            raise _service_degraded()
        yield


def _service_degraded() -> HTTPException:
    """Redis, behind the budget and the job's lock, can't answer right now."""
    return HTTPException(
        status_code=503,
        detail="Rate limit service degraded, retry shortly",
        headers={"Retry-After": "30"},
    )


async def _joined_with_waiting_send(user_id: str, session_id: str, message: str) -> str:
    """The message to schedule: this one after any already waiting."""
    waiting = next(
        (
            job.message
            for job in await _chat_turns(user_id, session_id)
            if job.purpose == "send" and job.message
        ),
        None,
    )
    if waiting is None:
        return message
    joined = f"{waiting}\n\n{message}"
    if len(joined) > _MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=413, detail="message_too_long")
    return joined


async def _chat_turns(user_id: str, session_id: str) -> list[CopilotTurnJobInfo]:
    schedules = await _scheduler().get_execution_schedules(
        user_id=user_id, session_id=session_id, kind="copilot_turn"
    )
    return [s for s in schedules if isinstance(s, CopilotTurnJobInfo)]


def _scheduler() -> SchedulerClient:
    # Every call here is safe to repeat: a read, a job set again under the
    # chat's own id, a cancel. A chat lists its turns each time it opens, so a
    # call lost on a connection the scheduler had just closed failed the load.
    return get_scheduler_client(should_retry=True)


def _as_turn(job: CopilotTurnJobInfo) -> AtResetTurn | None:
    if job.purpose == "followup" or job.run_at is None:
        return None
    message = job.message if job.purpose == "send" else None
    return AtResetTurn(kind=job.purpose, run_at=job.run_at, message=message)
