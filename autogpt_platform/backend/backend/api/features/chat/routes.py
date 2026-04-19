"""Chat API routes for chat session management and streaming via SSE."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import uuid4

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Query, Response, Security
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.copilot import service as chat_service
from backend.copilot import stream_registry
from backend.copilot.config import ChatConfig, CopilotLlmModel, CopilotMode
from backend.copilot.db import get_chat_messages_paginated
from backend.copilot.executor.utils import enqueue_cancel_task, enqueue_copilot_turn
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    ChatSessionMetadata,
    append_and_save_message,
    create_chat_session,
    delete_chat_session,
    get_chat_session,
    get_user_sessions,
    update_session_title,
)
from backend.copilot.pending_message_helpers import (
    QueuePendingMessageResponse,
    is_turn_in_flight,
    queue_pending_for_http,
)
from backend.copilot.pending_messages import peek_pending_messages
from backend.copilot.rate_limit import (
    CoPilotUsageStatus,
    RateLimitExceeded,
    acquire_reset_lock,
    check_rate_limit,
    get_daily_reset_count,
    get_global_rate_limits,
    get_usage_status,
    increment_daily_reset_count,
    release_reset_lock,
    reset_daily_usage,
)
from backend.copilot.response_model import StreamError, StreamFinish, StreamHeartbeat
from backend.copilot.service import strip_injected_context_for_display
from backend.copilot.tools.e2b_sandbox import kill_sandbox
from backend.copilot.tools.models import (
    AgentDetailsResponse,
    AgentOutputResponse,
    AgentPreviewResponse,
    AgentSavedResponse,
    AgentsFoundResponse,
    BlockDetailsResponse,
    BlockListResponse,
    BlockOutputResponse,
    ClarificationNeededResponse,
    DocPageResponse,
    DocSearchResultsResponse,
    ErrorResponse,
    ExecutionStartedResponse,
    InputValidationErrorResponse,
    MCPToolOutputResponse,
    MCPToolsDiscoveredResponse,
    MemoryForgetCandidatesResponse,
    MemoryForgetConfirmResponse,
    MemorySearchResponse,
    MemoryStoreResponse,
    NeedLoginResponse,
    NoResultsResponse,
    SetupRequirementsResponse,
    SuggestedGoalResponse,
    UnderstandingUpdatedResponse,
)
from backend.copilot.tracking import track_user_message
from backend.data.credit import UsageTransactionMetadata, get_user_credit_model
from backend.data.redis_client import get_redis_async
from backend.data.understanding import get_business_understanding
from backend.data.workspace import build_files_block, resolve_workspace_files
from backend.util.exceptions import InsufficientBalanceError, NotFoundError
from backend.util.settings import Settings

settings = Settings()

logger = logging.getLogger(__name__)

config = ChatConfig()


async def _validate_and_get_session(
    session_id: str,
    user_id: str | None,
) -> ChatSession:
    """Validate session exists and belongs to user."""
    session = await get_chat_session(session_id, user_id)
    if not session:
        raise NotFoundError(f"Session {session_id} not found.")
    return session


router = APIRouter(
    tags=["chat"],
)


def _strip_injected_context(message: dict) -> dict:
    """Hide server-injected context blocks from the API response.

    Returns a **shallow copy** of *message* with all server-injected XML
    blocks removed from ``content`` (if applicable).  The original dict is
    never mutated, so callers can safely pass live session dicts without
    risking side-effects.

    Handles all three injected block types — ``<memory_context>``,
    ``<env_context>``, and ``<user_context>`` — regardless of the order they
    appear at the start of the message.  Only ``user``-role messages with
    string content are touched; assistant / multimodal blocks pass through
    unchanged.
    """
    if message.get("role") == "user" and isinstance(message.get("content"), str):
        result = message.copy()
        result["content"] = strip_injected_context_for_display(message["content"])
        return result
    return message


# ========== Request/Response Models ==========


class StreamChatRequest(BaseModel):
    """Request model for streaming chat with optional context."""

    message: str
    is_user_message: bool = True
    context: dict[str, str] | None = None  # {url: str, content: str}
    file_ids: list[str] | None = Field(
        default=None, max_length=20
    )  # Workspace file IDs attached to this message
    mode: CopilotMode | None = Field(
        default=None,
        description="Autopilot mode: 'fast' for baseline LLM, 'extended_thinking' for Claude Agent SDK. "
        "If None, uses the server default (extended_thinking).",
    )
    model: CopilotLlmModel | None = Field(
        default=None,
        description="Model tier: 'standard' for the default model, 'advanced' for the highest-capability model. "
        "If None, the server applies per-user LD targeting then falls back to config.",
    )


class PeekPendingMessagesResponse(BaseModel):
    """Response for the pending-message peek (GET) endpoint.

    Returns a read-only view of the pending buffer — messages are NOT
    consumed.  The frontend uses this to restore the queued-message
    indicator after a page refresh and to decide when to clear it once
    a turn has ended.
    """

    messages: list[str]
    count: int


class CreateSessionRequest(BaseModel):
    """Request model for creating a new chat session.

    ``dry_run`` is a **top-level** field — do not nest it inside ``metadata``.
    Extra/unknown fields are rejected (422) to prevent silent mis-use.
    """

    model_config = ConfigDict(extra="forbid")

    dry_run: bool = False


class CreateSessionResponse(BaseModel):
    """Response model containing information on a newly created chat session."""

    id: str
    created_at: str
    user_id: str | None
    metadata: ChatSessionMetadata = ChatSessionMetadata()


class ActiveStreamInfo(BaseModel):
    """Information about an active stream for reconnection."""

    turn_id: str
    last_message_id: str  # Redis Stream message ID for resumption


class SessionDetailResponse(BaseModel):
    """Response model providing complete details for a chat session, including messages."""

    id: str
    created_at: str
    updated_at: str
    user_id: str | None
    messages: list[dict]
    active_stream: ActiveStreamInfo | None = None  # Present if stream is still active
    has_more_messages: bool = False
    oldest_sequence: int | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    metadata: ChatSessionMetadata = ChatSessionMetadata()


class SessionSummaryResponse(BaseModel):
    """Response model for a session summary (without messages)."""

    id: str
    created_at: str
    updated_at: str
    title: str | None = None
    is_processing: bool


class ListSessionsResponse(BaseModel):
    """Response model for listing chat sessions."""

    sessions: list[SessionSummaryResponse]
    total: int


class CancelSessionResponse(BaseModel):
    """Response model for the cancel session endpoint."""

    cancelled: bool
    reason: str | None = None


class UpdateSessionTitleRequest(BaseModel):
    """Request model for updating a session's title."""

    title: str

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Title must not be blank")
        return stripped


# ========== Routes ==========


@router.get(
    "/sessions",
    dependencies=[Security(auth.requires_user)],
)
async def list_sessions(
    user_id: Annotated[str, Security(auth.get_user_id)],
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> ListSessionsResponse:
    """
    List chat sessions for the authenticated user.

    Returns a paginated list of chat sessions belonging to the current user,
    ordered by most recently updated.

    Args:
        user_id: The authenticated user's ID.
        limit: Maximum number of sessions to return (1-100).
        offset: Number of sessions to skip for pagination.

    Returns:
        ListSessionsResponse: List of session summaries and total count.
    """
    sessions, total_count = await get_user_sessions(user_id, limit, offset)

    # Batch-check Redis for active stream status on each session
    processing_set: set[str] = set()
    if sessions:
        try:
            redis = await get_redis_async()
            pipe = redis.pipeline(transaction=False)
            for session in sessions:
                pipe.hget(
                    f"{config.session_meta_prefix}{session.session_id}",
                    "status",
                )
            statuses = await pipe.execute()
            processing_set = {
                session.session_id
                for session, st in zip(sessions, statuses)
                if st == "running"
            }
        except Exception:
            logger.warning(
                "Failed to fetch processing status from Redis; defaulting to empty"
            )

    return ListSessionsResponse(
        sessions=[
            SessionSummaryResponse(
                id=session.session_id,
                created_at=session.started_at.isoformat(),
                updated_at=session.updated_at.isoformat(),
                title=session.title,
                is_processing=session.session_id in processing_set,
            )
            for session in sessions
        ],
        total=total_count,
    )


@router.post(
    "/sessions",
)
async def create_session(
    user_id: Annotated[str, Security(auth.get_user_id)],
    request: CreateSessionRequest | None = None,
) -> CreateSessionResponse:
    """
    Create a new chat session.

    Initiates a new chat session for the authenticated user.

    Args:
        user_id: The authenticated user ID parsed from the JWT (required).
        request: Optional request body. When provided, ``dry_run=True``
            forces run_block and run_agent calls to use dry-run simulation.

    Returns:
        CreateSessionResponse: Details of the created session.

    """
    dry_run = request.dry_run if request else False

    logger.info(
        f"Creating session with user_id: "
        f"...{user_id[-8:] if len(user_id) > 8 else '<redacted>'}"
        f"{', dry_run=True' if dry_run else ''}"
    )

    session = await create_chat_session(user_id, dry_run=dry_run)

    return CreateSessionResponse(
        id=session.session_id,
        created_at=session.started_at.isoformat(),
        user_id=session.user_id,
        metadata=session.metadata,
    )


@router.delete(
    "/sessions/{session_id}",
    dependencies=[Security(auth.requires_user)],
    status_code=204,
    responses={404: {"description": "Session not found or access denied"}},
)
async def delete_session(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> Response:
    """
    Delete a chat session.

    Permanently removes a chat session and all its messages.
    Only the owner can delete their sessions.

    Args:
        session_id: The session ID to delete.
        user_id: The authenticated user's ID.

    Returns:
        204 No Content on success.

    Raises:
        HTTPException: 404 if session not found or not owned by user.
    """
    deleted = await delete_chat_session(session_id, user_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found or access denied",
        )

    # Best-effort cleanup of the E2B sandbox (if any).
    # sandbox_id is in Redis; kill_sandbox() fetches it from there.
    e2b_cfg = ChatConfig()
    if e2b_cfg.e2b_active:
        assert e2b_cfg.e2b_api_key  # guaranteed by e2b_active check
        try:
            await kill_sandbox(session_id, e2b_cfg.e2b_api_key)
        except Exception:
            logger.warning(
                "[E2B] Failed to kill sandbox for session %s", session_id[:12]
            )

    return Response(status_code=204)


@router.delete(
    "/sessions/{session_id}/stream",
    dependencies=[Security(auth.requires_user)],
    status_code=204,
)
async def disconnect_session_stream(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> Response:
    """Disconnect all active SSE listeners for a session.

    Called by the frontend when the user switches away from a chat so the
    backend releases XREAD listeners immediately rather than waiting for
    the 5-10 s timeout.
    """
    session = await get_chat_session(session_id, user_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found or access denied",
        )
    await stream_registry.disconnect_all_listeners(session_id)
    return Response(status_code=204)


@router.patch(
    "/sessions/{session_id}/title",
    summary="Update session title",
    dependencies=[Security(auth.requires_user)],
    status_code=200,
    responses={404: {"description": "Session not found or access denied"}},
)
async def update_session_title_route(
    session_id: str,
    request: UpdateSessionTitleRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> dict:
    """
    Update the title of a chat session.

    Allows the user to rename their chat session.

    Args:
        session_id: The session ID to update.
        request: Request body containing the new title.
        user_id: The authenticated user's ID.

    Returns:
        dict: Status of the update.

    Raises:
        HTTPException: 404 if session not found or not owned by user.
    """
    success = await update_session_title(session_id, user_id, request.title)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found or access denied",
        )
    return {"status": "ok"}


@router.get(
    "/sessions/{session_id}",
)
async def get_session(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
    limit: int = Query(default=50, ge=1, le=200),
    before_sequence: int | None = Query(default=None, ge=0),
) -> SessionDetailResponse:
    """
    Retrieve the details of a specific chat session.

    Supports cursor-based pagination via ``limit`` and ``before_sequence``.
    When no pagination params are provided, returns the most recent messages.
    """
    page = await get_chat_messages_paginated(
        session_id, limit, before_sequence, user_id=user_id
    )
    if page is None:
        raise NotFoundError(f"Session {session_id} not found.")

    messages = [
        _strip_injected_context(message.model_dump()) for message in page.messages
    ]

    # Only check active stream on initial load (not on "load more" requests)
    active_stream_info = None
    if before_sequence is None:
        active_session, last_message_id = await stream_registry.get_active_session(
            session_id, user_id
        )
        if active_session:
            active_stream_info = ActiveStreamInfo(
                turn_id=active_session.turn_id,
                last_message_id=last_message_id,
            )

    # Skip session metadata on "load more" — frontend only needs messages
    if before_sequence is not None:
        return SessionDetailResponse(
            id=page.session.session_id,
            created_at=page.session.started_at.isoformat(),
            updated_at=page.session.updated_at.isoformat(),
            user_id=page.session.user_id or None,
            messages=messages,
            active_stream=None,
            has_more_messages=page.has_more,
            oldest_sequence=page.oldest_sequence,
            total_prompt_tokens=0,
            total_completion_tokens=0,
        )

    total_prompt = sum(u.prompt_tokens for u in page.session.usage)
    total_completion = sum(u.completion_tokens for u in page.session.usage)

    return SessionDetailResponse(
        id=page.session.session_id,
        created_at=page.session.started_at.isoformat(),
        updated_at=page.session.updated_at.isoformat(),
        user_id=page.session.user_id or None,
        messages=messages,
        active_stream=active_stream_info,
        has_more_messages=page.has_more,
        oldest_sequence=page.oldest_sequence,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        metadata=page.session.metadata,
    )


@router.get(
    "/usage",
)
async def get_copilot_usage(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> CoPilotUsageStatus:
    """Get CoPilot usage status for the authenticated user.

    Returns current token usage vs limits for daily and weekly windows.
    Global defaults sourced from LaunchDarkly (falling back to config).
    Includes the user's rate-limit tier.
    """
    daily_limit, weekly_limit, tier = await get_global_rate_limits(
        user_id, config.daily_token_limit, config.weekly_token_limit
    )
    return await get_usage_status(
        user_id=user_id,
        daily_token_limit=daily_limit,
        weekly_token_limit=weekly_limit,
        rate_limit_reset_cost=config.rate_limit_reset_cost,
        tier=tier,
    )


class RateLimitResetResponse(BaseModel):
    """Response from resetting the daily rate limit."""

    success: bool
    credits_charged: int = Field(description="Credits charged (in cents)")
    remaining_balance: int = Field(description="Credit balance after charge (in cents)")
    usage: CoPilotUsageStatus = Field(description="Updated usage status after reset")


@router.post(
    "/usage/reset",
    status_code=200,
    responses={
        400: {
            "description": "Bad Request (feature disabled or daily limit not reached)"
        },
        402: {"description": "Payment Required (insufficient credits)"},
        429: {
            "description": "Too Many Requests (max daily resets exceeded or reset in progress)"
        },
        503: {
            "description": "Service Unavailable (Redis reset failed; credits refunded or support needed)"
        },
    },
)
async def reset_copilot_usage(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> RateLimitResetResponse:
    """Reset the daily CoPilot rate limit by spending credits.

    Allows users who have hit their daily token limit to spend credits
    to reset their daily usage counter and continue working.
    Returns 400 if the feature is disabled or the user is not over the limit.
    Returns 402 if the user has insufficient credits.
    """
    cost = config.rate_limit_reset_cost
    if cost <= 0:
        raise HTTPException(
            status_code=400,
            detail="Rate limit reset is not available.",
        )

    if not settings.config.enable_credit:
        raise HTTPException(
            status_code=400,
            detail="Rate limit reset is not available (credit system is disabled).",
        )

    daily_limit, weekly_limit, tier = await get_global_rate_limits(
        user_id, config.daily_token_limit, config.weekly_token_limit
    )

    if daily_limit <= 0:
        raise HTTPException(
            status_code=400,
            detail="No daily limit is configured — nothing to reset.",
        )

    # Check max daily resets.  get_daily_reset_count returns None when Redis
    # is unavailable; reject the reset in that case to prevent unlimited
    # free resets when the counter store is down.
    reset_count = await get_daily_reset_count(user_id)
    if reset_count is None:
        raise HTTPException(
            status_code=503,
            detail="Unable to verify reset eligibility — please try again later.",
        )
    if config.max_daily_resets > 0 and reset_count >= config.max_daily_resets:
        raise HTTPException(
            status_code=429,
            detail=f"You've used all {config.max_daily_resets} resets for today.",
        )

    # Acquire a per-user lock to prevent TOCTOU races (concurrent resets).
    if not await acquire_reset_lock(user_id):
        raise HTTPException(
            status_code=429,
            detail="A reset is already in progress. Please try again.",
        )

    try:
        # Verify the user is actually at or over their daily limit.
        # (rate_limit_reset_cost intentionally omitted — this object is only
        # used for limit checks, not returned to the client.)
        usage_status = await get_usage_status(
            user_id=user_id,
            daily_token_limit=daily_limit,
            weekly_token_limit=weekly_limit,
            tier=tier,
        )
        if daily_limit > 0 and usage_status.daily.used < daily_limit:
            raise HTTPException(
                status_code=400,
                detail="You have not reached your daily limit yet.",
            )

        # If the weekly limit is also exhausted, resetting the daily counter
        # won't help — the user would still be blocked by the weekly limit.
        if weekly_limit > 0 and usage_status.weekly.used >= weekly_limit:
            raise HTTPException(
                status_code=400,
                detail="Your weekly limit is also reached. Resetting the daily limit won't help.",
            )

        # Charge credits.
        credit_model = await get_user_credit_model(user_id)
        try:
            remaining = await credit_model.spend_credits(
                user_id=user_id,
                cost=cost,
                metadata=UsageTransactionMetadata(
                    reason="CoPilot daily rate limit reset",
                ),
            )
        except InsufficientBalanceError as e:
            raise HTTPException(
                status_code=402,
                detail="Insufficient credits to reset your rate limit.",
            ) from e

        # Reset daily usage in Redis.  If this fails, refund the credits
        # so the user is not charged for a service they did not receive.
        if not await reset_daily_usage(user_id, daily_token_limit=daily_limit):
            # Compensate: refund the charged credits.
            refunded = False
            try:
                await credit_model.top_up_credits(user_id, cost)
                refunded = True
                logger.warning(
                    "Refunded %d credits to user %s after Redis reset failure",
                    cost,
                    user_id[:8],
                )
            except Exception:
                logger.error(
                    "CRITICAL: Failed to refund %d credits to user %s "
                    "after Redis reset failure — manual intervention required",
                    cost,
                    user_id[:8],
                    exc_info=True,
                )
            if refunded:
                raise HTTPException(
                    status_code=503,
                    detail="Rate limit reset failed — please try again later. "
                    "Your credits have not been charged.",
                )
            raise HTTPException(
                status_code=503,
                detail="Rate limit reset failed and the automatic refund "
                "also failed. Please contact support for assistance.",
            )

        # Track the reset count for daily cap enforcement.
        await increment_daily_reset_count(user_id)
    finally:
        await release_reset_lock(user_id)

    # Return updated usage status.
    updated_usage = await get_usage_status(
        user_id=user_id,
        daily_token_limit=daily_limit,
        weekly_token_limit=weekly_limit,
        rate_limit_reset_cost=config.rate_limit_reset_cost,
        tier=tier,
    )

    return RateLimitResetResponse(
        success=True,
        credits_charged=cost,
        remaining_balance=remaining,
        usage=updated_usage,
    )


@router.post(
    "/sessions/{session_id}/cancel",
    status_code=200,
)
async def cancel_session_task(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> CancelSessionResponse:
    """Cancel the active streaming task for a session.

    Publishes a cancel event to the executor via RabbitMQ FANOUT, then
    polls Redis until the task status flips from ``running`` or a timeout
    (5 s) is reached.  Returns only after the cancellation is confirmed.
    """
    await _validate_and_get_session(session_id, user_id)

    active_session, _ = await stream_registry.get_active_session(session_id, user_id)
    if not active_session:
        return CancelSessionResponse(cancelled=True, reason="no_active_session")

    await enqueue_cancel_task(session_id)
    logger.info(f"[CANCEL] Published cancel for session ...{session_id[-8:]}")

    # Poll until the executor confirms the task is no longer running.
    poll_interval = 0.5
    max_wait = 5.0
    waited = 0.0
    while waited < max_wait:
        await asyncio.sleep(poll_interval)
        waited += poll_interval
        session_state = await stream_registry.get_session(session_id)
        if session_state is None or session_state.status != "running":
            logger.info(
                f"[CANCEL] Session ...{session_id[-8:]} confirmed stopped "
                f"(status={session_state.status if session_state else 'gone'}) after {waited:.1f}s"
            )
            return CancelSessionResponse(cancelled=True)

    logger.warning(
        f"[CANCEL] Session ...{session_id[-8:]} not confirmed after {max_wait}s, force-completing"
    )
    await stream_registry.mark_session_completed(session_id, error_message="Cancelled")
    return CancelSessionResponse(cancelled=True)


@router.post(
    "/sessions/{session_id}/stream",
    responses={
        202: {
            "model": QueuePendingMessageResponse,
            "description": (
                "Session has a turn in flight — message queued into the pending "
                "buffer and will be picked up between tool-call rounds by the "
                "executor currently processing the turn."
            ),
        },
        404: {"description": "Session not found or access denied"},
        429: {"description": "Token rate-limit or call-frequency cap exceeded"},
    },
)
async def stream_chat_post(
    session_id: str,
    request: StreamChatRequest,
    user_id: str = Security(auth.get_user_id),
):
    """Start a new turn OR queue a follow-up — decided server-side.

    - **Session idle**: starts a turn.  Returns an SSE stream (``text/event-stream``)
      with Vercel AI SDK chunks (text fragments, tool-call UI, tool results).
      The generation runs in a background task that survives client disconnects;
      reconnect via ``GET /sessions/{session_id}/stream`` to resume.

    - **Session has a turn in flight**: pushes the message into the per-session
      pending buffer and returns ``202 application/json`` with
      ``QueuePendingMessageResponse``.  The executor running the current turn
      drains the buffer between tool-call rounds (baseline) or at the start of
      the next turn (SDK).  Clients should detect the 202 and surface the
      message as a queued-chip in the UI.

    Args:
        session_id: The chat session identifier.
        request: Request body with message, is_user_message, and optional context.
        user_id: Authenticated user ID.
    """
    import asyncio
    import time

    stream_start_time = time.perf_counter()
    # Wall-clock arrival time, propagated to the executor so the turn-start
    # drain can order pending messages relative to this request (pending
    # pushed BEFORE this instant were typed earlier; pending pushed AFTER
    # are race-path follow-ups typed while /stream was still processing).
    request_arrival_at = time.time()
    log_meta = {"component": "ChatStream", "session_id": session_id, "user_id": user_id}

    logger.info(
        f"[TIMING] stream_chat_post STARTED, session={session_id}, "
        f"user={user_id}, message_len={len(request.message)}",
        extra={"json_fields": log_meta},
    )
    await _validate_and_get_session(session_id, user_id)

    # Self-defensive queue-fallback: if a turn is already running, don't race
    # it on the cluster lock — drop the message into the pending buffer and
    # return 202 so the caller can render a chip.  Both UI chips and autopilot
    # block follow-ups route through this path; keeping the decision on the
    # server means every caller gets uniform behaviour.
    if (
        request.is_user_message
        and request.message
        and await is_turn_in_flight(session_id)
    ):
        response = await queue_pending_for_http(
            session_id=session_id,
            user_id=user_id,
            message=request.message,
            context=request.context,
            file_ids=request.file_ids,
        )
        return JSONResponse(status_code=202, content=response.model_dump())

    logger.info(
        f"[TIMING] session validated in {(time.perf_counter() - stream_start_time) * 1000:.1f}ms",
        extra={
            "json_fields": {
                **log_meta,
                "duration_ms": (time.perf_counter() - stream_start_time) * 1000,
            }
        },
    )

    # Pre-turn rate limit check (token-based).
    # check_rate_limit short-circuits internally when both limits are 0.
    # Global defaults sourced from LaunchDarkly, falling back to config.
    if user_id:
        try:
            daily_limit, weekly_limit, _ = await get_global_rate_limits(
                user_id, config.daily_token_limit, config.weekly_token_limit
            )
            await check_rate_limit(
                user_id=user_id,
                daily_token_limit=daily_limit,
                weekly_token_limit=weekly_limit,
            )
        except RateLimitExceeded as e:
            raise HTTPException(status_code=429, detail=str(e)) from e

    # Enrich message with file metadata if file_ids are provided.
    # Also sanitise file_ids so only validated, workspace-scoped IDs are
    # forwarded downstream (e.g. to the executor via enqueue_copilot_turn).
    sanitized_file_ids: list[str] | None = None
    if request.file_ids:
        files = await resolve_workspace_files(user_id, request.file_ids)
        sanitized_file_ids = [wf.id for wf in files] or None
        request.message += build_files_block(files)

    # Atomically append user message to session BEFORE creating task to avoid
    # race condition where GET_SESSION sees task as "running" but message isn't
    # saved yet.  append_and_save_message returns None when a duplicate is
    # detected — in that case skip enqueue to avoid processing the message twice.
    is_duplicate_message = False
    if request.message:
        message = ChatMessage(
            role="user" if request.is_user_message else "assistant",
            content=request.message,
        )
        logger.info(f"[STREAM] Saving user message to session {session_id}")
        is_duplicate_message = (
            await append_and_save_message(session_id, message)
        ) is None
        logger.info(f"[STREAM] User message saved for session {session_id}")
        if not is_duplicate_message and request.is_user_message:
            track_user_message(
                user_id=user_id,
                session_id=session_id,
                message_length=len(request.message),
            )

    # Create a task in the stream registry for reconnection support.
    # For duplicate messages, skip create_session entirely so the infra-retry
    # client subscribes to the *existing* turn's Redis stream and receives the
    # in-progress executor output rather than an empty stream.
    turn_id = ""
    if not is_duplicate_message:
        turn_id = str(uuid4())
        log_meta["turn_id"] = turn_id
        session_create_start = time.perf_counter()
        await stream_registry.create_session(
            session_id=session_id,
            user_id=user_id,
            tool_call_id="chat_stream",
            tool_name="chat",
            turn_id=turn_id,
        )
        logger.info(
            f"[TIMING] create_session completed in {(time.perf_counter() - session_create_start) * 1000:.1f}ms",
            extra={
                "json_fields": {
                    **log_meta,
                    "duration_ms": (time.perf_counter() - session_create_start) * 1000,
                }
            },
        )
        await enqueue_copilot_turn(
            session_id=session_id,
            user_id=user_id,
            message=request.message,
            turn_id=turn_id,
            is_user_message=request.is_user_message,
            context=request.context,
            file_ids=sanitized_file_ids,
            mode=request.mode,
            model=request.model,
            request_arrival_at=request_arrival_at,
        )
    else:
        logger.info(
            f"[STREAM] Duplicate message detected for session {session_id}, skipping enqueue"
        )

    setup_time = (time.perf_counter() - stream_start_time) * 1000
    logger.info(
        f"[TIMING] Task enqueued to RabbitMQ, setup={setup_time:.1f}ms",
        extra={"json_fields": {**log_meta, "setup_time_ms": setup_time}},
    )

    # Per-turn stream is always fresh (unique turn_id), subscribe from beginning
    subscribe_from_id = "0-0"

    # SSE endpoint that subscribes to the task's stream
    async def event_generator() -> AsyncGenerator[str, None]:
        import time as time_module

        event_gen_start = time_module.perf_counter()
        logger.info(
            f"[TIMING] event_generator STARTED, turn={turn_id}, session={session_id}, "
            f"user={user_id}",
            extra={"json_fields": log_meta},
        )
        subscriber_queue = None
        first_chunk_yielded = False
        chunks_yielded = 0
        try:
            # Subscribe from the position we captured before enqueuing
            # This avoids replaying old messages while catching all new ones
            subscriber_queue = await stream_registry.subscribe_to_session(
                session_id=session_id,
                user_id=user_id,
                last_message_id=subscribe_from_id,
            )

            if subscriber_queue is None:
                yield StreamFinish().to_sse()
                return

            # Read from the subscriber queue and yield to SSE
            logger.info(
                "[TIMING] Starting to read from subscriber_queue",
                extra={"json_fields": log_meta},
            )
            while True:
                try:
                    chunk = await asyncio.wait_for(subscriber_queue.get(), timeout=10.0)
                    chunks_yielded += 1

                    if not first_chunk_yielded:
                        first_chunk_yielded = True
                        elapsed = time_module.perf_counter() - event_gen_start
                        logger.info(
                            f"[TIMING] FIRST CHUNK from queue at {elapsed:.2f}s, "
                            f"type={type(chunk).__name__}",
                            extra={
                                "json_fields": {
                                    **log_meta,
                                    "chunk_type": type(chunk).__name__,
                                    "elapsed_ms": elapsed * 1000,
                                }
                            },
                        )

                    yield chunk.to_sse()

                    if isinstance(chunk, StreamFinish):
                        total_time = time_module.perf_counter() - event_gen_start
                        logger.info(
                            f"[TIMING] StreamFinish received in {total_time:.2f}s; "
                            f"n_chunks={chunks_yielded}",
                            extra={
                                "json_fields": {
                                    **log_meta,
                                    "chunks_yielded": chunks_yielded,
                                    "total_time_ms": total_time * 1000,
                                }
                            },
                        )
                        break

                except asyncio.TimeoutError:
                    yield StreamHeartbeat().to_sse()

        except GeneratorExit:
            logger.info(
                f"[TIMING] GeneratorExit (client disconnected), chunks={chunks_yielded}",
                extra={
                    "json_fields": {
                        **log_meta,
                        "chunks_yielded": chunks_yielded,
                        "reason": "client_disconnect",
                    }
                },
            )
        except Exception as e:
            elapsed = (time_module.perf_counter() - event_gen_start) * 1000
            logger.error(
                f"[TIMING] event_generator ERROR after {elapsed:.1f}ms: {e}",
                extra={
                    "json_fields": {**log_meta, "elapsed_ms": elapsed, "error": str(e)}
                },
            )
            # Surface error to frontend so it doesn't appear stuck
            yield StreamError(
                errorText="An error occurred. Please try again.",
                code="stream_error",
            ).to_sse()
            yield StreamFinish().to_sse()
        finally:
            # Unsubscribe when client disconnects or stream ends
            if subscriber_queue is not None:
                try:
                    await stream_registry.unsubscribe_from_session(
                        session_id, subscriber_queue
                    )
                except Exception as unsub_err:
                    logger.error(
                        f"Error unsubscribing from session {session_id}: {unsub_err}",
                        exc_info=True,
                    )
            # AI SDK protocol termination - always yield even if unsubscribe fails
            total_time = time_module.perf_counter() - event_gen_start
            logger.info(
                f"[TIMING] event_generator FINISHED in {total_time:.2f}s; "
                f"turn={turn_id}, session={session_id}, n_chunks={chunks_yielded}",
                extra={
                    "json_fields": {
                        **log_meta,
                        "total_time_ms": total_time * 1000,
                        "chunks_yielded": chunks_yielded,
                    }
                },
            )
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "x-vercel-ai-ui-message-stream": "v1",  # AI SDK protocol header
        },
    )


@router.get(
    "/sessions/{session_id}/messages/pending",
    response_model=PeekPendingMessagesResponse,
    responses={
        404: {"description": "Session not found or access denied"},
    },
)
async def get_pending_messages(
    session_id: str,
    user_id: str = Security(auth.get_user_id),
):
    """Peek at the pending-message buffer without consuming it.

    Returns the current contents of the session's pending message buffer
    so the frontend can restore the queued-message indicator after a page
    refresh and clear it correctly once a turn drains the buffer.
    """
    await _validate_and_get_session(session_id, user_id)
    pending = await peek_pending_messages(session_id)
    return PeekPendingMessagesResponse(
        messages=[m.content for m in pending],
        count=len(pending),
    )


@router.get(
    "/sessions/{session_id}/stream",
)
async def resume_session_stream(
    session_id: str,
    user_id: str = Security(auth.get_user_id),
):
    """
    Resume an active stream for a session.

    Called by the AI SDK's ``useChat(resume: true)`` on page load.
    Checks for an active (in-progress) task on the session and either replays
    the full SSE stream or returns 204 No Content if nothing is running.

    Args:
        session_id: The chat session identifier.
        user_id: Optional authenticated user ID.

    Returns:
        StreamingResponse (SSE) when an active stream exists,
        or 204 No Content when there is nothing to resume.
    """
    import asyncio

    active_session, last_message_id = await stream_registry.get_active_session(
        session_id, user_id
    )

    if not active_session:
        return Response(status_code=204)

    # Always replay from the beginning ("0-0") on resume.
    # We can't use last_message_id because it's the latest ID in the backend
    # stream, not the latest the frontend received — the gap causes lost
    # messages. The frontend deduplicates replayed content.
    subscriber_queue = await stream_registry.subscribe_to_session(
        session_id=session_id,
        user_id=user_id,
        last_message_id="0-0",
    )

    if subscriber_queue is None:
        return Response(status_code=204)

    async def event_generator() -> AsyncGenerator[str, None]:
        chunk_count = 0
        first_chunk_type: str | None = None
        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(subscriber_queue.get(), timeout=10.0)
                    if chunk_count < 3:
                        logger.info(
                            "Resume stream chunk",
                            extra={
                                "session_id": session_id,
                                "chunk_type": str(chunk.type),
                            },
                        )
                    if not first_chunk_type:
                        first_chunk_type = str(chunk.type)
                    chunk_count += 1
                    yield chunk.to_sse()

                    if isinstance(chunk, StreamFinish):
                        break
                except asyncio.TimeoutError:
                    yield StreamHeartbeat().to_sse()
        except GeneratorExit:
            pass
        except Exception as e:
            logger.error(f"Error in resume stream for session {session_id}: {e}")
        finally:
            try:
                await stream_registry.unsubscribe_from_session(
                    session_id, subscriber_queue
                )
            except Exception as unsub_err:
                logger.error(
                    f"Error unsubscribing from session {active_session.session_id}: {unsub_err}",
                    exc_info=True,
                )
            logger.info(
                "Resume stream completed",
                extra={
                    "session_id": session_id,
                    "n_chunks": chunk_count,
                    "first_chunk_type": first_chunk_type,
                },
            )
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "x-vercel-ai-ui-message-stream": "v1",
        },
    )


@router.patch(
    "/sessions/{session_id}/assign-user",
    dependencies=[Security(auth.requires_user)],
)
async def session_assign_user(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> dict:
    """
    Assign an authenticated user to a chat session.

    Used (typically post-login) to claim an existing anonymous session as the current authenticated user.

    Args:
        session_id: The identifier for the (previously anonymous) session.
        user_id: The authenticated user's ID to associate with the session.

    Returns:
        dict: Status of the assignment.

    """
    await chat_service.assign_user_to_session(session_id, user_id)
    return {"status": "ok"}


# ========== Suggested Prompts ==========


class SuggestedTheme(BaseModel):
    """A themed group of suggested prompts."""

    name: str
    prompts: list[str]


class SuggestedPromptsResponse(BaseModel):
    """Response model for user-specific suggested prompts grouped by theme."""

    themes: list[SuggestedTheme]


@router.get(
    "/suggested-prompts",
    dependencies=[Security(auth.requires_user)],
)
async def get_suggested_prompts(
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> SuggestedPromptsResponse:
    """
    Get LLM-generated suggested prompts grouped by theme.

    Returns personalized quick-action prompts based on the user's
    business understanding. Returns empty themes list if no custom
    prompts are available.
    """
    understanding = await get_business_understanding(user_id)
    if understanding is None or not understanding.suggested_prompts:
        return SuggestedPromptsResponse(themes=[])

    themes = [
        SuggestedTheme(name=name, prompts=prompts)
        for name, prompts in understanding.suggested_prompts.items()
    ]
    return SuggestedPromptsResponse(themes=themes)


# ========== Configuration ==========


@router.get("/config/ttl", status_code=200)
async def get_ttl_config() -> dict:
    """
    Get the stream TTL configuration.

    Returns the Time-To-Live settings for chat streams, which determines
    how long clients can reconnect to an active stream.

    Returns:
        dict: TTL configuration with seconds and milliseconds values.
    """
    return {
        "stream_ttl_seconds": config.stream_ttl,
        "stream_ttl_ms": config.stream_ttl * 1000,
    }


# ========== Health Check ==========


@router.get("/health", status_code=200)
async def health_check() -> dict:
    """
    Health check endpoint for the chat service.

    Performs a full cycle test of session creation and retrieval. Should always return healthy
    if the service and data layer are operational.

    Returns:
        dict: A status dictionary indicating health, service name, and API version.

    """
    from backend.data.user import get_or_create_user

    # Ensure health check user exists (required for FK constraint)
    health_check_user_id = "health-check-user"
    await get_or_create_user(
        {
            "sub": health_check_user_id,
            "email": "health-check@system.local",
            "user_metadata": {"name": "Health Check User"},
        }
    )

    # Create and retrieve session to verify full data layer
    session = await create_chat_session(health_check_user_id, dry_run=False)
    await get_chat_session(session.session_id, health_check_user_id)

    return {
        "status": "healthy",
        "service": "chat",
        "version": "0.1.0",
    }


# ========== Schema Export (for OpenAPI / Orval codegen) ==========

ToolResponseUnion = (
    AgentsFoundResponse
    | NoResultsResponse
    | AgentDetailsResponse
    | SetupRequirementsResponse
    | ExecutionStartedResponse
    | NeedLoginResponse
    | ErrorResponse
    | InputValidationErrorResponse
    | AgentOutputResponse
    | UnderstandingUpdatedResponse
    | AgentPreviewResponse
    | AgentSavedResponse
    | ClarificationNeededResponse
    | SuggestedGoalResponse
    | BlockListResponse
    | BlockDetailsResponse
    | BlockOutputResponse
    | DocSearchResultsResponse
    | DocPageResponse
    | MCPToolsDiscoveredResponse
    | MCPToolOutputResponse
    | MemoryStoreResponse
    | MemorySearchResponse
    | MemoryForgetCandidatesResponse
    | MemoryForgetConfirmResponse
)


@router.get(
    "/schema/tool-responses",
    response_model=ToolResponseUnion,
    include_in_schema=True,
    summary="[Dummy] Tool response type export for codegen",
    description="This endpoint is not meant to be called. It exists solely to "
    "expose tool response models in the OpenAPI schema for frontend codegen.",
)
async def _tool_response_schema() -> ToolResponseUnion:  # type: ignore[return]
    """Never called at runtime. Exists only so Orval generates TS types."""
    raise HTTPException(status_code=501, detail="Schema-only endpoint")
