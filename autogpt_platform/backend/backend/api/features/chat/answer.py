"""Answer a chat without opening its stream.

The chat page sends a message through ``POST /sessions/{id}/stream``, which
holds an SSE connection open for the turn it starts. A delegated expert's
question is answered from *another* chat — Otto's, or its Work panel — which
has no reason to stream the expert's thread. This route posts the user's
message into the thread and returns at once; the reply is read the way the
delegation is already read, by polling the thread.
"""

import logging

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Security
from pydantic import BaseModel, Field, field_validator

from backend.api.features.experts import experts_db
from backend.copilot import turn_queue
from backend.copilot.active_turns import (
    ConcurrentTurnLimitError,
    get_inflight_turn_limit,
    inflight_turn_limit_message,
)
from backend.copilot.config import ChatConfig
from backend.copilot.db import clear_session_pending_question
from backend.copilot.executor.utils import schedule_chat_turn
from backend.copilot.model import (
    ChatSessionInfo,
    get_chat_session_metadata,
    invalidate_session_cache,
)
from backend.copilot.pending_message_helpers import (
    StreamRegistryUnavailable,
    is_turn_in_flight,
    queue_pending_for_http,
)
from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    check_rate_limit,
    enforce_payment_paywall,
    get_global_rate_limits,
)
from backend.copilot.session_permissions import resolve_session_permissions
from backend.integrations.codex.access import enforce_codex_access_http
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)
config = ChatConfig()

router = APIRouter(tags=["chat"])

_MAX_ANSWER_CHARS = 32_000


class AnswerSessionRequest(BaseModel):
    message: str = Field(min_length=1, max_length=_MAX_ANSWER_CHARS)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must not be blank")
        return value


class AnswerSessionResponse(BaseModel):
    session_id: str
    queued: bool = Field(
        description=(
            "True when the message waits behind work already in flight: the "
            "thread's running turn takes it at its next step, or the turn is "
            "queued behind the user's running-turn cap. False when a new turn "
            "started with it."
        )
    )


@router.post(
    "/sessions/{session_id}/messages",
    operation_id="answer_session",
    response_model=AnswerSessionResponse,
    responses={
        404: {"description": "Session not found or access denied"},
        429: {"description": "Rate limit or concurrent-turn cap exceeded"},
        503: {"description": "Chat service degraded; retry shortly"},
    },
)
async def answer_session(
    session_id: str,
    request: AnswerSessionRequest,
    user_id: str = Security(auth.get_user_id),
) -> AnswerSessionResponse:
    """Post a user message into a session and start its turn, without a stream.

    Used to answer a delegated expert's question from the chat that
    delegated it. A running turn takes the message into its pending buffer
    instead, the same as a follow-up typed mid-turn.
    """
    session = await _writable_session(session_id, user_id)
    await _admit(session, user_id)
    if await _in_flight(session_id):
        await queue_pending_for_http(
            session_id=session_id,
            user_id=user_id,
            message=request.message,
            context=None,
            file_ids=None,
            folder_ids=None,
            expert_id=session.expert_id,
        )
        queued = True
    else:
        queued = await _start_turn(session, user_id, request.message)
    await _resolve_question(session_id, user_id)
    return AnswerSessionResponse(session_id=session_id, queued=queued)


async def _writable_session(session_id: str, user_id: str) -> ChatSessionInfo:
    session = await get_chat_session_metadata(session_id, user_id)
    if session is None:
        raise NotFoundError(f"Session {session_id} not found.")
    if session.expert_id is not None and not await experts_db.owns_active_expert(
        user_id, session.expert_id
    ):
        raise HTTPException(status_code=404, detail="Expert not found")
    return session


async def _admit(session: ChatSessionInfo, user_id: str) -> None:
    """The same spend gates the stream route applies before a turn."""
    provider = session.metadata.llm_auth_provider
    if provider == "codex":
        await enforce_codex_access_http(user_id)
    if provider != "platform":
        return
    await enforce_payment_paywall(user_id)
    try:
        daily, weekly, _ = await get_global_rate_limits(
            user_id,
            config.daily_cost_limit_microdollars,
            config.weekly_cost_limit_microdollars,
        )
        await check_rate_limit(
            user_id=user_id, daily_cost_limit=daily, weekly_cost_limit=weekly
        )
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e)) from e
    except RateLimitUnavailable as e:
        raise HTTPException(
            status_code=503,
            detail="Rate limit service degraded, retry shortly",
            headers={"Retry-After": "30"},
        ) from e


async def _in_flight(session_id: str) -> bool:
    try:
        return await is_turn_in_flight(session_id)
    except StreamRegistryUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="Chat service degraded, retry shortly",
            headers={"Retry-After": "30"},
        ) from exc


async def _start_turn(session: ChatSessionInfo, user_id: str, message: str) -> bool:
    """Start a turn with *message*; True when it had to wait in the queue."""
    route = session.metadata
    permissions = resolve_session_permissions(session)
    try:
        await schedule_chat_turn(
            session_id=session.session_id,
            user_id=user_id,
            message=message,
            is_user_message=True,
            expert_id=session.expert_id,
            session_origin=route.origin,
            organization_id=session.organization_id,
            team_id=session.team_id,
            llm_auth_provider=route.llm_auth_provider,
            llm_credential_id=route.llm_credential_id,
            permissions=permissions,
        )
        return False
    except ConcurrentTurnLimitError:
        cap = get_inflight_turn_limit()
        try:
            await turn_queue.try_enqueue_turn(
                user_id=user_id,
                inflight_cap=cap,
                session_id=session.session_id,
                message=message,
                llm_auth_provider=route.llm_auth_provider,
                llm_credential_id=route.llm_credential_id,
                permissions=(
                    permissions.model_dump(exclude_none=True) if permissions else None
                ),
            )
        except turn_queue.InflightCapExceeded:
            raise HTTPException(
                status_code=429, detail=inflight_turn_limit_message(cap)
            )
        return True


async def _resolve_question(session_id: str, user_id: str) -> None:
    """The question is answered the moment the reply lands, not when the turn
    reaches it: a delegator polling in between must stop saying "needs you".
    Best effort, like the turn's own clear."""
    try:
        await clear_session_pending_question(session_id, user_id)
        await invalidate_session_cache(session_id)
    except Exception:
        logger.warning(
            f"Could not clear the pending question of {session_id}", exc_info=True
        )
