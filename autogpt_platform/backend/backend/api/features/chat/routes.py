"""Chat API routes for chat session management and streaming via SSE."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Annotated
from uuid import uuid4

from autogpt_libs import auth
from fastapi import APIRouter, Depends, HTTPException, Query, Response, Security
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.copilot import service as chat_service
from backend.copilot import stream_registry
from backend.copilot.config import ChatConfig
from backend.copilot.executor.utils import enqueue_cancel_task, enqueue_copilot_turn
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    append_and_save_message,
    create_chat_session,
    delete_chat_session,
    get_chat_session,
    get_user_sessions,
)
from backend.copilot.response_model import StreamError, StreamFinish, StreamHeartbeat
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
    NeedLoginResponse,
    NoResultsResponse,
    SetupRequirementsResponse,
    SuggestedGoalResponse,
    UnderstandingUpdatedResponse,
)
from backend.copilot.tracking import track_user_message
from backend.util.exceptions import NotFoundError

config = ChatConfig()


logger = logging.getLogger(__name__)


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

# ========== Request/Response Models ==========


class StreamChatRequest(BaseModel):
    """Request model for streaming chat with optional context."""

    message: str
    is_user_message: bool = True
    context: dict[str, str] | None = None  # {url: str, content: str}


class CreateSessionResponse(BaseModel):
    """Response model containing information on a newly created chat session."""

    id: str
    created_at: str
    user_id: str | None


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


class SessionSummaryResponse(BaseModel):
    """Response model for a session summary (without messages)."""

    id: str
    created_at: str
    updated_at: str
    title: str | None = None


class ListSessionsResponse(BaseModel):
    """Response model for listing chat sessions."""

    sessions: list[SessionSummaryResponse]
    total: int


class CancelSessionResponse(BaseModel):
    """Response model for the cancel session endpoint."""

    cancelled: bool
    reason: str | None = None


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

    return ListSessionsResponse(
        sessions=[
            SessionSummaryResponse(
                id=session.session_id,
                created_at=session.started_at.isoformat(),
                updated_at=session.updated_at.isoformat(),
                title=session.title,
            )
            for session in sessions
        ],
        total=total_count,
    )


@router.post(
    "/sessions",
)
async def create_session(
    user_id: Annotated[str, Depends(auth.get_user_id)],
) -> CreateSessionResponse:
    """
    Create a new chat session.

    Initiates a new chat session for the authenticated user.

    Args:
        user_id: The authenticated user ID parsed from the JWT (required).

    Returns:
        CreateSessionResponse: Details of the created session.

    """
    logger.info(
        f"Creating session with user_id: "
        f"...{user_id[-8:] if len(user_id) > 8 else '<redacted>'}"
    )

    session = await create_chat_session(user_id)

    return CreateSessionResponse(
        id=session.session_id,
        created_at=session.started_at.isoformat(),
        user_id=session.user_id,
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

    return Response(status_code=204)


@router.get(
    "/sessions/{session_id}",
)
async def get_session(
    session_id: str,
    user_id: Annotated[str | None, Depends(auth.get_user_id)],
) -> SessionDetailResponse:
    """
    Retrieve the details of a specific chat session.

    Looks up a chat session by ID for the given user (if authenticated) and returns all session data including messages.
    If there's an active stream for this session, returns active_stream info for reconnection.

    Args:
        session_id: The unique identifier for the desired chat session.
        user_id: The optional authenticated user ID, or None for anonymous access.

    Returns:
        SessionDetailResponse: Details for the requested session, including active_stream info if applicable.

    """
    session = await get_chat_session(session_id, user_id)
    if not session:
        raise NotFoundError(f"Session {session_id} not found.")

    messages = [message.model_dump() for message in session.messages]

    # Check if there's an active stream for this session
    active_stream_info = None
    active_session, last_message_id = await stream_registry.get_active_session(
        session_id, user_id
    )
    logger.info(
        f"[GET_SESSION] session={session_id}, active_session={active_session is not None}, "
        f"msg_count={len(messages)}, last_role={messages[-1].get('role') if messages else 'none'}"
    )
    if active_session:
        # Keep the assistant message (including tool_calls) so the frontend can
        # render the correct tool UI (e.g. CreateAgent with mini game).
        # convertChatSessionToUiMessages handles isComplete=false by setting
        # tool parts without output to state "input-available".
        active_stream_info = ActiveStreamInfo(
            turn_id=active_session.turn_id,
            last_message_id=last_message_id,
        )

    return SessionDetailResponse(
        id=session.session_id,
        created_at=session.started_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        user_id=session.user_id or None,
        messages=messages,
        active_stream=active_stream_info,
    )


@router.post(
    "/sessions/{session_id}/cancel",
    status_code=200,
)
async def cancel_session_task(
    session_id: str,
    user_id: Annotated[str | None, Depends(auth.get_user_id)],
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
)
async def stream_chat_post(
    session_id: str,
    request: StreamChatRequest,
    user_id: str | None = Depends(auth.get_user_id),
):
    """
    Stream chat responses for a session (POST with context support).

    Streams the AI/completion responses in real time over Server-Sent Events (SSE), including:
      - Text fragments as they are generated
      - Tool call UI elements (if invoked)
      - Tool execution results

    The AI generation runs in a background task that continues even if the client disconnects.
    All chunks are written to a per-turn Redis stream for reconnection support. If the client
    disconnects, they can reconnect using GET /sessions/{session_id}/stream to resume.

    Args:
        session_id: The chat session identifier to associate with the streamed messages.
        request: Request body containing message, is_user_message, and optional context.
        user_id: Optional authenticated user ID.
    Returns:
        StreamingResponse: SSE-formatted response chunks.

    """
    import asyncio
    import time

    stream_start_time = time.perf_counter()
    log_meta = {"component": "ChatStream", "session_id": session_id}
    if user_id:
        log_meta["user_id"] = user_id

    logger.info(
        f"[TIMING] stream_chat_post STARTED, session={session_id}, "
        f"user={user_id}, message_len={len(request.message)}",
        extra={"json_fields": log_meta},
    )
    await _validate_and_get_session(session_id, user_id)
    logger.info(
        f"[TIMING] session validated in {(time.perf_counter() - stream_start_time) * 1000:.1f}ms",
        extra={
            "json_fields": {
                **log_meta,
                "duration_ms": (time.perf_counter() - stream_start_time) * 1000,
            }
        },
    )

    # Atomically append user message to session BEFORE creating task to avoid
    # race condition where GET_SESSION sees task as "running" but message isn't
    # saved yet.  append_and_save_message re-fetches inside a lock to prevent
    # message loss from concurrent requests.
    if request.message:
        message = ChatMessage(
            role="user" if request.is_user_message else "assistant",
            content=request.message,
        )
        if request.is_user_message:
            track_user_message(
                user_id=user_id,
                session_id=session_id,
                message_length=len(request.message),
            )
        logger.info(f"[STREAM] Saving user message to session {session_id}")
        await append_and_save_message(session_id, message)
        logger.info(f"[STREAM] User message saved for session {session_id}")

    # Create a task in the stream registry for reconnection support
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

    # Per-turn stream is always fresh (unique turn_id), subscribe from beginning
    subscribe_from_id = "0-0"

    await enqueue_copilot_turn(
        session_id=session_id,
        user_id=user_id,
        message=request.message,
        turn_id=turn_id,
        is_user_message=request.is_user_message,
        context=request.context,
    )

    setup_time = (time.perf_counter() - stream_start_time) * 1000
    logger.info(
        f"[TIMING] Task enqueued to RabbitMQ, setup={setup_time:.1f}ms",
        extra={"json_fields": {**log_meta, "setup_time_ms": setup_time}},
    )

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
                yield "data: [DONE]\n\n"
                return

            # Read from the subscriber queue and yield to SSE
            logger.info(
                "[TIMING] Starting to read from subscriber_queue",
                extra={"json_fields": log_meta},
            )
            while True:
                try:
                    chunk = await asyncio.wait_for(subscriber_queue.get(), timeout=30.0)
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

                    # Check for finish signal
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
            pass  # Client disconnected - background task continues
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
    "/sessions/{session_id}/stream",
)
async def resume_session_stream(
    session_id: str,
    user_id: str | None = Depends(auth.get_user_id),
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
                    chunk = await asyncio.wait_for(subscriber_queue.get(), timeout=30.0)
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
    status_code=200,
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
    session = await create_chat_session(health_check_user_id)
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
