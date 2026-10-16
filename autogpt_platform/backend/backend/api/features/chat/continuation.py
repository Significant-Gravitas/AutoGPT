"""Keep going: pick a paused task back up in the same chat, now.

The Paused card posts here after a step check-in, or when a task reached its
size limit while the account still has budget. The continuation runs as a
turn whose message is a system note (see ``backend.copilot.continuation``),
so the chat shows what happened without a user bubble. It is admitted like
any message the user sends: the same paywall, the same allowance check, and
one turn per chat at a time. It carries on the paused turn's work, so it
starts with that turn's taint, read off the pause marker.
"""

import logging

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Request, Security
from pydantic import BaseModel, Field

from backend.api.features.chat.routes import (
    _validate_and_get_writable_session,
    enforce_model_entitlement,
    enforce_platform_allowance,
)
from backend.copilot import stream_registry
from backend.copilot.active_turns import ConcurrentTurnLimitError
from backend.copilot.at_reset import cancel_resume_at_reset_in_background
from backend.copilot.config import CopilotLLMModel
from backend.copilot.continuation import KEEP_GOING_NOTE
from backend.copilot.executor.utils import schedule_chat_turn
from backend.copilot.llm_attribution import is_impersonated_request
from backend.copilot.model import ChatSessionInfo, get_chat_session
from backend.copilot.pending_message_helpers import (
    StreamRegistryUnavailable,
    is_turn_in_flight,
)
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.copilot.response_model import StreamStatus
from backend.copilot.session_permissions import resolve_session_permissions
from backend.copilot.turn_pause import continues_tainted
from backend.integrations.codex.access import enforce_codex_access_http

logger = logging.getLogger(__name__)

router = APIRouter()


class ContinueSessionRequest(BaseModel):
    model: CopilotLLMModel | None = Field(
        default=None,
        description="Model tier the continuation runs on, as for a message.",
    )


class ContinueSessionResponse(BaseModel):
    turn_id: str = Field(description="The continuation turn; attach to its stream.")


@router.post(
    "/sessions/{session_id}/continue",
    status_code=202,
    responses={
        402: {"description": "Subscription required (NO_TIER user, paywall on)"},
        404: {"description": "Session not found or access denied"},
        409: {
            "description": "A turn is already running in this chat "
            "(turn_in_progress), or the chat runs on a connection that can't "
            "continue a task (unsupported_connection)"
        },
        429: {"description": "Usage limit or concurrent-turn limit reached"},
        503: {"description": "Chat service degraded; retry after Retry-After"},
    },
)
async def continue_session(
    session_id: str,
    http_request: Request,
    request: ContinueSessionRequest | None = None,
    user_id: str = Security(auth.get_user_id),
    ctx: auth.RequestContext = Security(auth.get_request_context),
) -> ContinueSessionResponse:
    """Continue the chat's task from the checkpoint it paused at."""
    session = await _validate_and_get_writable_session(session_id, user_id)
    model = request.model if request is not None else None
    await enforce_model_entitlement(model, session, user_id)
    await _admit(session, user_id)
    await _refuse_if_busy(session_id)
    tainted = await _continues_tainted(session_id, user_id)
    try:
        # Every note is the same text, so the trailing-text dedupe would refuse
        # a Keep going after a continuation that wrote only assistant rows, or
        # nothing. The fresh slot is what stops a second turn on this chat.
        turn_id = await schedule_chat_turn(
            session_id=session_id,
            user_id=user_id,
            message=KEEP_GOING_NOTE,
            require_fresh_slot=True,
            dedupe_content=False,
            is_user_message=False,
            expert_id=session.expert_id,
            session_origin=session.metadata.origin,
            session_source_platform=session.metadata.source_platform,
            organization_id=session.organization_id or ctx.org_id,
            team_id=session.team_id if session.organization_id else ctx.team_id,
            model=model,
            llm_auth_provider=session.metadata.llm_auth_provider,
            llm_credential_id=session.metadata.llm_credential_id,
            permissions=resolve_session_permissions(session),
            impersonated=is_impersonated_request(http_request),
            tainted=tainted,
        )
    except ConcurrentTurnLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    if turn_id is None:
        # Another turn took the chat after the busy check above: a message
        # sent at the same moment, or the first click of a double click.
        raise HTTPException(status_code=409, detail="turn_in_progress")
    # Picked up now, so a pending "Resume at reset" would repeat it.
    cancel_resume_at_reset_in_background(session_id, user_id)
    await _announce(turn_id, session_id)
    return ContinueSessionResponse(turn_id=turn_id)


async def _admit(session: ChatSessionInfo, user_id: str) -> None:
    route = session.metadata.llm_auth_provider
    if route == "platform":
        await enforce_payment_paywall(user_id)
        await enforce_platform_allowance(user_id)
    elif route == "codex":
        await enforce_codex_access_http(user_id)
    elif route == "microsoft_365_copilot":
        # Microsoft 365 Copilot answers plain messages: it has none of
        # AutoPilot's tools and never saw the task, so it has nothing to pick
        # up, and the note would reach Microsoft as the user's message.
        raise HTTPException(status_code=409, detail="unsupported_connection")


async def _continues_tainted(session_id: str, user_id: str) -> bool:
    """Whether the turn that paused ran tainted, as its marker recorded.

    Read once the chat is idle, so the paused turn has saved its rows.
    """
    session = await get_chat_session(session_id, user_id)
    return session is not None and continues_tainted(session.messages)


async def _refuse_if_busy(session_id: str) -> None:
    try:
        busy = await is_turn_in_flight(session_id)
    except StreamRegistryUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="Chat service degraded, retry shortly",
            headers={"Retry-After": "30"},
        ) from exc
    if busy:
        raise HTTPException(status_code=409, detail="turn_in_progress")


async def _announce(turn_id: str, session_id: str) -> None:
    """Give a subscriber something to show while the turn waits for pickup."""
    try:
        await stream_registry.publish_chunk(
            turn_id,
            StreamStatus(message="Picking the task back up…"),
            session_id=session_id,
        )
    except Exception:
        logger.warning("[CONTINUE] Failed to publish initial status", exc_info=True)
