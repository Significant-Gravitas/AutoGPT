"""Turns the server starts on the user's behalf, with no stream to answer.

A board card starts when it leaves Backlog, whether or not its chat is open;
the panel resumes the running stream when it opens.
"""

import logging
import time
from typing import Any

from fastapi import HTTPException

from backend.copilot import stream_registry, turn_queue
from backend.copilot.active_turns import (
    ConcurrentTurnLimitError,
    get_inflight_turn_limit,
    inflight_turn_limit_message,
)
from backend.copilot.config import ChatConfig
from backend.copilot.executor.utils import schedule_chat_turn
from backend.copilot.expert_kickoff import scoped_client_message_id
from backend.copilot.model import ChatSessionInfo
from backend.copilot.provider_failure import ProviderFailure, ProviderFailureKind
from backend.copilot.rate_limit import (
    RateLimitExceeded,
    RateLimitUnavailable,
    check_rate_limit,
    enforce_payment_paywall,
    get_global_rate_limits,
)
from backend.copilot.response_model import StreamStatus
from backend.copilot.session_permissions import resolve_session_permissions
from backend.integrations.codex.access import enforce_codex_access_http

logger = logging.getLogger(__name__)

config = ChatConfig()


def board_start_message_id(user_id: str, session_id: str) -> str:
    """One id per card, so a second drag or Start cannot send it twice."""
    return scoped_client_message_id(user_id, session_id, "board-start")


async def start_board_card(
    session: ChatSessionInfo,
    user_id: str,
    *,
    org_id: str | None,
    team_id: str | None,
    message: str | None = None,
) -> None:
    """Send a card's first message: *message* if given, else its title."""
    text = (message or "").strip() or (session.title or "").strip()
    if not text:
        raise HTTPException(
            status_code=422, detail="A task needs a title or a message to start"
        )
    await dispatch_server_turn(
        session,
        user_id,
        org_id=org_id,
        team_id=team_id,
        message=text,
        message_id=board_start_message_id(user_id, session.session_id),
    )


async def dispatch_server_turn(
    session: ChatSessionInfo,
    user_id: str,
    *,
    org_id: str | None,
    team_id: str | None,
    message: str,
    message_id: str | None,
    message_metadata: dict[str, Any] | None = None,
) -> str | None:
    """Start a user turn with the stream route's checks, without its stream.

    Returns the new turn id, or None when *message_id* was already saved or
    the user's running cap queued the turn for later.
    """
    request_arrival_at = time.time()
    session_id = session.session_id
    llm_auth_provider = session.metadata.llm_auth_provider
    if llm_auth_provider == "platform":
        await enforce_payment_paywall(user_id)
        await _check_platform_rate_limit(user_id)
    elif llm_auth_provider == "codex":
        await enforce_codex_access_http(user_id)

    permissions = resolve_session_permissions(session)
    try:
        turn_id = await schedule_chat_turn(
            session_id=session_id,
            user_id=user_id,
            message=message,
            message_id=message_id,
            message_metadata=message_metadata,
            is_user_message=True,
            expert_id=session.expert_id,
            session_origin=session.metadata.origin,
            session_source_platform=session.metadata.source_platform,
            organization_id=session.organization_id or org_id,
            team_id=session.team_id if session.organization_id else team_id,
            llm_auth_provider=llm_auth_provider,
            llm_credential_id=session.metadata.llm_credential_id,
            permissions=permissions,
            request_arrival_at=request_arrival_at,
        )
    except ConcurrentTurnLimitError:
        inflight_cap = get_inflight_turn_limit()
        try:
            await turn_queue.try_enqueue_turn(
                user_id=user_id,
                inflight_cap=inflight_cap,
                session_id=session_id,
                message=message,
                message_id=message_id,
                message_metadata=message_metadata,
                is_user_message=True,
                llm_auth_provider=llm_auth_provider,
                llm_credential_id=session.metadata.llm_credential_id,
                permissions=(
                    permissions.model_dump(exclude_none=True) if permissions else None
                ),
                request_arrival_at=request_arrival_at,
            )
        except turn_queue.InflightCapExceeded:
            raise HTTPException(
                status_code=429,
                detail=inflight_turn_limit_message(inflight_cap),
            )
        logger.info(
            f"[SERVER_TURN] Queued turn for session={session_id} "
            f"(running cap reached; inflight cap={inflight_cap})"
        )
        return None

    if turn_id is None:
        logger.info(f"[SERVER_TURN] Duplicate message for session {session_id}")
        return None
    try:
        await stream_registry.publish_chunk(
            turn_id,
            StreamStatus(message="Message received…"),
            session_id=session_id,
        )
    except Exception:
        logger.warning(
            "[SERVER_TURN] Failed to publish initial status chunk", exc_info=True
        )
    return turn_id


async def _check_platform_rate_limit(user_id: str) -> None:
    # Twin of the rate-limit block in routes.stream_chat_post; keep them in step.
    try:
        daily_limit, weekly_limit, _ = await get_global_rate_limits(
            user_id,
            config.daily_cost_limit_microdollars,
            config.weekly_cost_limit_microdollars,
        )
        await check_rate_limit(
            user_id=user_id,
            daily_cost_limit=daily_limit,
            weekly_cost_limit=weekly_limit,
        )
    except RateLimitExceeded as e:
        failure = ProviderFailure(
            kind=ProviderFailureKind.USAGE_LIMIT,
            message=str(e),
            auth_provider="platform",
            resets_at=int(e.resets_at.timestamp()),
        )
        raise HTTPException(status_code=429, detail=failure.as_part()) from e
    except RateLimitUnavailable as e:
        raise HTTPException(
            status_code=503,
            detail="Rate limit service degraded, retry shortly",
            headers={"Retry-After": "30"},
        ) from e
