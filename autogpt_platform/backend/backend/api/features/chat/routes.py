"""Chat API routes for chat session management and streaming via SSE."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import uuid4

from autogpt_libs import auth
from fastapi import APIRouter, HTTPException, Query, Response, Security
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.copilot import service as chat_service
from backend.copilot import stream_registry
from backend.copilot.active_turns import (
    ConcurrentTurnLimitError,
    concurrent_turn_limit_message,
)
from backend.copilot.builder_context import resolve_session_permissions
from backend.copilot.config import ChatConfig, CopilotLlmModel, CopilotMode
from backend.copilot.db import get_chat_messages_paginated
from backend.copilot.executor.utils import enqueue_cancel_task, schedule_chat_turn
from backend.copilot.model import (
    ChatSession,
    ChatSessionMetadata,
    create_chat_session,
    delete_chat_session,
    get_chat_session,
    get_or_create_builder_session,
    get_user_sessions,
    update_session_title,
)
from backend.copilot.pending_message_helpers import (
    QueuePendingMessageResponse,
    StreamRegistryUnavailable,
    is_turn_in_flight,
    queue_pending_for_http,
)
from backend.copilot.pending_messages import peek_pending_messages
from backend.copilot.rate_limit import (
    CoPilotUsagePublic,
    RateLimitExceeded,
    RateLimitUnavailable,
    acquire_reset_lock,
    check_rate_limit,
    get_daily_reset_count,
    get_global_rate_limits,
    get_usage_status,
    increment_daily_reset_count,
    release_reset_lock,
    reset_daily_usage,
)
from backend.copilot.response_model import (
    StreamError,
    StreamFinish,
    StreamFinishStep,
    StreamHeartbeat,
    StreamStart,
    StreamStartStep,
)
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
    TodoWriteResponse,
    UnderstandingUpdatedResponse,
)
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

    message: str = Field(max_length=64_000)
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
    message_id: str | None = Field(
        default=None,
        max_length=64,
        description=(
            "Optional per-click UUID generated by the frontend.  Becomes "
            "the persisted ``ChatMessage.id`` (PK).  Frontend / network / "
            "RMQ-redelivery retransmits of the same logical send reuse "
            "the id, so the Postgres unique-constraint on the PK is the "
            "atomic dedup primitive: a duplicate INSERT returns a "
            "subscribe-only response without creating a parallel turn.  "
            "Distinct user clicks (even with identical text) MUST send "
            "different ids — the frontend's per-click ``crypto.randomUUID()`` "
            "guarantees that."
        ),
    )


class QueuePendingMessageRequest(BaseModel):
    """Request model for queueing a follow-up while a turn is running."""

    message: str = Field(max_length=64_000)
    context: dict[str, str] | None = None
    file_ids: list[str] | None = Field(default=None, max_length=20)


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
    """Request model for creating (or get-or-creating) a chat session.

    Two modes, selected by the body:

    - Default: create a fresh session. ``dry_run`` is a **top-level**
      field — do not nest it inside ``metadata``.
    - Builder-bound: when ``builder_graph_id`` is set, the endpoint
      switches to **get-or-create** keyed on
      ``(user_id, builder_graph_id)``.  The builder panel calls this on
      mount so the chat persists across refreshes.  Graph ownership is
      validated inside :func:`get_or_create_builder_session`. Write-side
      scope is enforced per-tool (``edit_agent`` / ``run_agent`` reject
      any ``agent_id`` other than the bound graph) and a small blacklist
      hides tools that conflict with the panel's scope
      (``create_agent`` / ``customize_agent`` / ``get_agent_building_guide``
      — see :data:`BUILDER_BLOCKED_TOOLS`). Read-side lookups
      (``find_block``, ``find_agent``, ``search_docs``, …) stay open.

    Extra/unknown fields are rejected (422) to prevent silent mis-use.
    """

    model_config = ConfigDict(extra="forbid")

    dry_run: bool = False
    builder_graph_id: str | None = Field(default=None, max_length=128)


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
    # ISO-8601 timestamp (UTC) marking when the backend registered the turn
    # as running. Lets the frontend seed its elapsed-time counter so restored
    # turns show honest "time since turn started" instead of the misleading
    # "time since this mount resumed the SSE".
    started_at: str | None = None


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
                # Use the canonical helper so the hash-tag braces match every
                # other writer; building the key inline drops the braces and
                # silently misses every running session on cluster mode.
                pipe.hget(
                    stream_registry.get_session_meta_key(session.session_id),
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
    """Create (or get-or-create) a chat session.

    Two modes, selected by the request body:

    - Default: create a fresh session for the user. ``dry_run=True`` forces
      run_block and run_agent calls to use dry-run simulation.
    - Builder-bound: when ``builder_graph_id`` is set, get-or-create keyed
      on ``(user_id, builder_graph_id)``. Returns the existing session for
      that graph or creates one locked to it.  Graph ownership is validated
      inside :func:`get_or_create_builder_session`; raises 404 on
      unauthorized access.  Write-side scope is enforced per-tool
      (``edit_agent`` / ``run_agent`` reject any ``agent_id`` other than
      the bound graph) and a small blacklist hides tools that conflict
      with the panel's scope (see :data:`BUILDER_BLOCKED_TOOLS`).

    Args:
        user_id: The authenticated user ID parsed from the JWT (required).
        request: Optional request body with ``dry_run`` and/or
            ``builder_graph_id``.

    Returns:
        CreateSessionResponse: Details of the resulting session.
    """
    dry_run = request.dry_run if request else False
    builder_graph_id = request.builder_graph_id if request else None

    logger.info(
        f"Creating session with user_id: "
        f"...{user_id[-8:] if len(user_id) > 8 else '<redacted>'}"
        f"{', dry_run=True' if dry_run else ''}"
        f"{f', builder_graph_id={builder_graph_id}' if builder_graph_id else ''}"
    )

    if builder_graph_id:
        session = await get_or_create_builder_session(user_id, builder_graph_id)
    else:
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
                started_at=active_session.created_at.isoformat(),
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
) -> CoPilotUsagePublic:
    """Get CoPilot usage status for the authenticated user.

    Returns the percentage of the daily/weekly allowance used — not the
    raw spend or cap — so clients cannot derive per-turn cost or platform
    margins. Global defaults sourced from LaunchDarkly (falling back to
    config). Includes the user's rate-limit tier.
    """
    daily_limit, weekly_limit, tier = await get_global_rate_limits(
        user_id,
        config.daily_cost_limit_microdollars,
        config.weekly_cost_limit_microdollars,
    )
    status = await get_usage_status(
        user_id=user_id,
        daily_cost_limit=daily_limit,
        weekly_cost_limit=weekly_limit,
        rate_limit_reset_cost=config.rate_limit_reset_cost,
        tier=tier,
    )
    return CoPilotUsagePublic.from_status(status)


class RateLimitResetResponse(BaseModel):
    """Response from resetting the daily rate limit."""

    success: bool
    credits_charged: int = Field(description="Credits charged (in cents)")
    remaining_balance: int = Field(description="Credit balance after charge (in cents)")
    usage: CoPilotUsagePublic = Field(
        description="Updated usage status after reset (percentages only)"
    )


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

    Allows users who have hit their daily cost limit to spend credits
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
        user_id,
        config.daily_cost_limit_microdollars,
        config.weekly_cost_limit_microdollars,
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
            daily_cost_limit=daily_limit,
            weekly_cost_limit=weekly_limit,
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
        if not await reset_daily_usage(user_id, daily_cost_limit=daily_limit):
            # Compensate: refund the charged credits as a GRANT (no Stripe
            # charge — TOP_UP is reserved for real user-initiated checkouts).
            refunded = False
            try:
                await credit_model.grant_credits(
                    user_id,
                    cost,
                    "Refund for failed CoPilot rate-limit reset",
                )
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

    # Return updated usage status (public schema — percentages only).
    updated_usage = await get_usage_status(
        user_id=user_id,
        daily_cost_limit=daily_limit,
        weekly_cost_limit=weekly_limit,
        rate_limit_reset_cost=config.rate_limit_reset_cost,
        tier=tier,
    )

    return RateLimitResetResponse(
        success=True,
        credits_charged=cost,
        remaining_balance=remaining,
        usage=CoPilotUsagePublic.from_status(updated_usage),
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


def _ui_message_stream_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "x-vercel-ai-ui-message-stream": "v1",
    }


def _empty_ui_message_stream_response() -> StreamingResponse:
    # Stable placeholder messageId for the empty queued-mid-turn stream.
    # Real turns generate per-message UUIDs via the executor; this stream
    # has no message to attach to, but the AI SDK parser still requires a
    # non-empty ``messageId`` field on ``StreamStart``.
    message_id = uuid4().hex

    async def event_generator() -> AsyncGenerator[str, None]:
        # Vercel AI SDK's UI-message-stream parser expects symmetric
        # start/finish framing at both stream and step level — every
        # non-empty turn emits the pair.  Without an opener, today's parser
        # tolerates the closer (no active parts to flush) but a future SDK
        # tightening would silently break the queue-mid-turn UX.  Emit the
        # full empty pair so the contract stays correct.
        yield StreamStart(messageId=message_id).to_sse()
        yield StreamStartStep().to_sse()
        yield StreamFinishStep().to_sse()
        yield StreamFinish().to_sse()
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers=_ui_message_stream_headers(),
    )


@router.post(
    "/sessions/{session_id}/stream",
    responses={
        404: {"description": "Session not found or access denied"},
        429: {
            "description": "Cost rate-limit, call-frequency cap, or "
            "per-user concurrent-turn limit exceeded"
        },
        503: {
            "description": "Chat service degraded (Redis unavailable for rate "
            "limit or stream registry); client should honour the Retry-After "
            "header before retrying."
        },
    },
)
async def stream_chat_post(
    session_id: str,
    request: StreamChatRequest,
    user_id: str = Security(auth.get_user_id),
):
    """Start a new turn and return an AI SDK UI message stream.

    Returns an SSE stream (``text/event-stream``) with Vercel AI SDK chunks
    (text fragments, tool-call UI, tool results). The generation runs in a
    background task that survives client disconnects; reconnect via
    ``GET /sessions/{session_id}/stream`` to resume.

    Follow-up messages typed while a turn is already running should use
    ``POST /sessions/{session_id}/messages/pending``. If an older client still
    posts that follow-up here, we queue it defensively but still return a valid
    empty UI-message stream so AI SDK transports never receive a JSON body from
    the stream endpoint.

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
    session = await _validate_and_get_session(session_id, user_id)

    try:
        turn_in_flight = (
            request.is_user_message
            and request.message
            and await is_turn_in_flight(session_id)
        )
    except StreamRegistryUnavailable as exc:
        # Same fail-closed mapping as the RateLimitUnavailable branch below:
        # the pre-flight chain runs is_turn_in_flight BEFORE check_rate_limit,
        # so a Redis brown-out at this step would otherwise surface as a raw
        # 500 instead of the polished 503 + Retry-After.
        raise HTTPException(
            status_code=503,
            detail="Chat service degraded, retry shortly",
            headers={"Retry-After": "30"},
        ) from exc

    if turn_in_flight:
        try:
            await queue_pending_for_http(
                session_id=session_id,
                user_id=user_id,
                message=request.message,
                context=request.context,
                file_ids=request.file_ids,
            )
            return _empty_ui_message_stream_response()
        except HTTPException as exc:
            if exc.status_code != 409:
                raise

    # Permission resolution is only needed below for the actual turn — keep
    # it after the queue-fall-through so a queued mid-turn request returns
    # without paying the work.
    builder_permissions = resolve_session_permissions(session)

    logger.info(
        f"[TIMING] session validated in {(time.perf_counter() - stream_start_time) * 1000:.1f}ms",
        extra={
            "json_fields": {
                **log_meta,
                "duration_ms": (time.perf_counter() - stream_start_time) * 1000,
            }
        },
    )

    # Pre-turn rate limit check (cost-based, microdollars).
    # check_rate_limit short-circuits internally when both limits are 0.
    # Global defaults sourced from LaunchDarkly, falling back to config.
    if user_id:
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
            raise HTTPException(status_code=429, detail=str(e)) from e
        except RateLimitUnavailable as e:
            # Fail-closed on Redis brown-out: the user may already be at or
            # past their USD cap and we cannot prove otherwise. 503 + a short
            # Retry-After is the right UX (transient outage, retry shortly),
            # not 429 ("you hit your limit").
            raise HTTPException(
                status_code=503,
                detail="Rate limit service degraded, retry shortly",
                headers={"Retry-After": "30"},
            ) from e

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
    # detected — both the trailing-same-role check and the
    # ``ChatMessage.id`` PK collision (frontend-supplied per-click UUID
    # → Postgres unique constraint) feed into that signal.  In either
    # case we skip enqueue and let the SSE generator subscribe to any
    # existing in-flight turn for this session.
    #
    # Note: the in-flight branch is handled at the top of this handler
    # via ``queue_pending_for_http`` (see ``is_turn_in_flight`` check
    # near the start) — that path returns early.  Any request that
    # reaches this point is starting a fresh turn, so we always mint a
    # ``turn_id`` unless ``append_and_save_message`` reports a duplicate.
    try:
        turn_id = await schedule_chat_turn(
            session_id=session_id,
            user_id=user_id,
            message=request.message,
            message_id=request.message_id,
            is_user_message=request.is_user_message,
            context=request.context,
            file_ids=sanitized_file_ids,
            mode=request.mode,
            model=request.model,
            permissions=builder_permissions,
            request_arrival_at=request_arrival_at,
        )
    except ConcurrentTurnLimitError as exc:
        raise HTTPException(
            status_code=429, detail=concurrent_turn_limit_message()
        ) from exc

    if turn_id is None:
        logger.info(
            f"[STREAM] Duplicate message detected for session {session_id}, skipping enqueue"
        )
    else:
        log_meta["turn_id"] = turn_id

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
        headers=_ui_message_stream_headers(),
    )


@router.post(
    "/sessions/{session_id}/messages/pending",
    response_model=QueuePendingMessageResponse,
    responses={
        404: {"description": "Session not found or access denied"},
        409: {"description": "Session has no active turn to receive pending messages"},
        429: {"description": "Call-frequency cap exceeded"},
        503: {
            "description": "Chat service degraded (Redis unavailable); "
            "client should honour the Retry-After header before retrying."
        },
    },
)
async def queue_pending_message(
    session_id: str,
    request: QueuePendingMessageRequest,
    user_id: str = Security(auth.get_user_id),
):
    """Queue a follow-up message while the session has an active turn."""
    await _validate_and_get_session(session_id, user_id)
    try:
        turn_in_flight = await is_turn_in_flight(session_id)
    except StreamRegistryUnavailable as exc:
        raise HTTPException(
            status_code=503,
            detail="Chat service degraded, retry shortly",
            headers={"Retry-After": "30"},
        ) from exc
    if not turn_in_flight:
        raise HTTPException(
            status_code=409,
            detail="Session has no active turn. Start a new turn with POST /stream.",
        )
    return await queue_pending_for_http(
        session_id=session_id,
        user_id=user_id,
        message=request.message,
        context=request.context,
        file_ids=request.file_ids,
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
    last_chunk_id: str | None = Query(default=None, include_in_schema=False),
    user_id: str = Security(auth.get_user_id),
):
    """
    Resume an active stream for a session.

    Called by the AI SDK's ``useChat(resume: true)`` on page load.
    Checks for an active (in-progress) task on the session and either replays
    the full SSE stream or returns 204 No Content if nothing is running.

    Always replays the active turn from ``0-0``. The AI SDK UI-message parser
    keeps text/reasoning part state inside a single parser instance; resuming
    from a Redis cursor can skip the ``*-start`` events required by later
    ``*-delta`` chunks.
    """
    import asyncio

    active_session, _latest_backend_id = await stream_registry.get_active_session(
        session_id, user_id
    )

    if not active_session:
        return Response(status_code=204)

    if last_chunk_id:
        logger.info(
            "Ignoring deprecated last_chunk_id on stream resume",
            extra={"session_id": session_id, "last_chunk_id": last_chunk_id},
        )

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
        headers=_ui_message_stream_headers(),
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
    | TodoWriteResponse
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
