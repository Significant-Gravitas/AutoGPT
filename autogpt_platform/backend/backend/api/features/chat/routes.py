"""Chat API routes for chat session management and streaming via SSE."""

import logging
import uuid as uuid_module
from collections.abc import AsyncGenerator
from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Security
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.util.exceptions import NotFoundError

from . import service as chat_service
from . import stream_registry
from .completion_handler import process_operation_failure, process_operation_success
from .config import ChatConfig
from .model import ChatSession, create_chat_session, get_chat_session, get_user_sessions
from .response_model import StreamFinish, StreamHeartbeat, StreamStart

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

    task_id: str
    last_message_id: str  # Redis Stream message ID for resumption
    operation_id: str  # Operation ID for completion tracking
    tool_name: str  # Name of the tool being executed


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


class OperationCompleteRequest(BaseModel):
    """Request model for external completion webhook."""

    success: bool
    result: dict | str | None = None
    error: str | None = None


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
    If there's an active stream for this session, returns the task_id for reconnection.

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
    active_task, last_message_id = await stream_registry.get_active_task_for_session(
        session_id, user_id
    )
    if active_task:
        # Filter out the in-progress assistant message from the session response.
        # The client will receive the complete assistant response through the SSE
        # stream replay instead, preventing duplicate content.
        if messages and messages[-1].get("role") == "assistant":
            messages = messages[:-1]

        # Use "0-0" as last_message_id to replay the stream from the beginning.
        # Since we filtered out the cached assistant message, the client needs
        # the full stream to reconstruct the response.
        active_stream_info = ActiveStreamInfo(
            task_id=active_task.task_id,
            last_message_id="0-0",
            operation_id=active_task.operation_id,
            tool_name=active_task.tool_name,
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
    All chunks are written to Redis for reconnection support. If the client disconnects,
    they can reconnect using GET /tasks/{task_id}/stream to resume from where they left off.

    Args:
        session_id: The chat session identifier to associate with the streamed messages.
        request: Request body containing message, is_user_message, and optional context.
        user_id: Optional authenticated user ID.
    Returns:
        StreamingResponse: SSE-formatted response chunks. First chunk is a "start" event
        containing the task_id for reconnection.

    """
    import asyncio

    session = await _validate_and_get_session(session_id, user_id)

    # Create a task in the stream registry for reconnection support
    task_id = str(uuid_module.uuid4())
    operation_id = str(uuid_module.uuid4())
    await stream_registry.create_task(
        task_id=task_id,
        session_id=session_id,
        user_id=user_id,
        tool_call_id="chat_stream",  # Not a tool call, but needed for the model
        tool_name="chat",
        operation_id=operation_id,
    )

    # Background task that runs the AI generation independently of SSE connection
    async def run_ai_generation():
        try:
            # Emit a start event with task_id for reconnection
            start_chunk = StreamStart(messageId=task_id, taskId=task_id)
            await stream_registry.publish_chunk(task_id, start_chunk)

            async for chunk in chat_service.stream_chat_completion(
                session_id,
                request.message,
                is_user_message=request.is_user_message,
                user_id=user_id,
                session=session,  # Pass pre-fetched session to avoid double-fetch
                context=request.context,
            ):
                # Write to Redis (subscribers will receive via XREAD)
                await stream_registry.publish_chunk(task_id, chunk)

            # Mark task as completed
            await stream_registry.mark_task_completed(task_id, "completed")
        except Exception as e:
            logger.error(
                f"Error in background AI generation for session {session_id}: {e}"
            )
            await stream_registry.mark_task_completed(task_id, "failed")

    # Start the AI generation in a background task
    bg_task = asyncio.create_task(run_ai_generation())
    await stream_registry.set_task_asyncio_task(task_id, bg_task)

    # SSE endpoint that subscribes to the task's stream
    async def event_generator() -> AsyncGenerator[str, None]:
        subscriber_queue = None
        try:
            # Subscribe to the task stream (this replays existing messages + live updates)
            subscriber_queue = await stream_registry.subscribe_to_task(
                task_id=task_id,
                user_id=user_id,
                last_message_id="0-0",  # Get all messages from the beginning
            )

            if subscriber_queue is None:
                yield StreamFinish().to_sse()
                yield "data: [DONE]\n\n"
                return

            # Read from the subscriber queue and yield to SSE
            while True:
                try:
                    chunk = await asyncio.wait_for(subscriber_queue.get(), timeout=30.0)
                    yield chunk.to_sse()

                    # Check for finish signal
                    if isinstance(chunk, StreamFinish):
                        break
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield StreamHeartbeat().to_sse()

        except GeneratorExit:
            pass  # Client disconnected - background task continues
        except Exception as e:
            logger.error(f"Error in SSE stream for task {task_id}: {e}")
        finally:
            # Unsubscribe when client disconnects or stream ends to prevent resource leak
            if subscriber_queue is not None:
                try:
                    await stream_registry.unsubscribe_from_task(
                        task_id, subscriber_queue
                    )
                except Exception as unsub_err:
                    logger.error(
                        f"Error unsubscribing from task {task_id}: {unsub_err}",
                        exc_info=True,
                    )
            # AI SDK protocol termination - always yield even if unsubscribe fails
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
async def stream_chat_get(
    session_id: str,
    message: Annotated[str, Query(min_length=1, max_length=10000)],
    user_id: str | None = Depends(auth.get_user_id),
    is_user_message: bool = Query(default=True),
):
    """
    Stream chat responses for a session (GET - legacy endpoint).

    Streams the AI/completion responses in real time over Server-Sent Events (SSE), including:
      - Text fragments as they are generated
      - Tool call UI elements (if invoked)
      - Tool execution results

    Args:
        session_id: The chat session identifier to associate with the streamed messages.
        message: The user's new message to process.
        user_id: Optional authenticated user ID.
        is_user_message: Whether the message is a user message.
    Returns:
        StreamingResponse: SSE-formatted response chunks.

    """
    session = await _validate_and_get_session(session_id, user_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        chunk_count = 0
        first_chunk_type: str | None = None
        async for chunk in chat_service.stream_chat_completion(
            session_id,
            message,
            is_user_message=is_user_message,
            user_id=user_id,
            session=session,  # Pass pre-fetched session to avoid double-fetch
        ):
            if chunk_count < 3:
                logger.info(
                    "Chat stream chunk",
                    extra={
                        "session_id": session_id,
                        "chunk_type": str(chunk.type),
                    },
                )
            if not first_chunk_type:
                first_chunk_type = str(chunk.type)
            chunk_count += 1
            yield chunk.to_sse()
        logger.info(
            "Chat stream completed",
            extra={
                "session_id": session_id,
                "chunk_count": chunk_count,
                "first_chunk_type": first_chunk_type,
            },
        )
        # AI SDK protocol termination
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


# ========== Task Streaming (SSE Reconnection) ==========


@router.get(
    "/tasks/{task_id}/stream",
)
async def stream_task(
    task_id: str,
    user_id: str | None = Depends(auth.get_user_id),
    last_message_id: str = Query(
        default="0-0",
        description="Last Redis Stream message ID received (e.g., '1706540123456-0'). Use '0-0' for full replay.",
    ),
):
    """
    Reconnect to a long-running task's SSE stream.

    When a long-running operation (like agent generation) starts, the client
    receives a task_id. If the connection drops, the client can reconnect
    using this endpoint to resume receiving updates.

    Args:
        task_id: The task ID from the operation_started response.
        user_id: Authenticated user ID for ownership validation.
        last_message_id: Last Redis Stream message ID received ("0-0" for full replay).

    Returns:
        StreamingResponse: SSE-formatted response chunks starting after last_message_id.

    Raises:
        HTTPException: 404 if task not found, 410 if task expired, 403 if access denied.
    """
    # Check task existence and expiry before subscribing
    task, error_code = await stream_registry.get_task_with_expiry_info(task_id)

    if error_code == "TASK_EXPIRED":
        raise HTTPException(
            status_code=410,
            detail={
                "code": "TASK_EXPIRED",
                "message": "This operation has expired. Please try again.",
            },
        )

    if error_code == "TASK_NOT_FOUND":
        raise HTTPException(
            status_code=404,
            detail={
                "code": "TASK_NOT_FOUND",
                "message": f"Task {task_id} not found.",
            },
        )

    # Validate ownership if task has an owner
    if task and task.user_id and user_id != task.user_id:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "ACCESS_DENIED",
                "message": "You do not have access to this task.",
            },
        )

    # Get subscriber queue from stream registry
    subscriber_queue = await stream_registry.subscribe_to_task(
        task_id=task_id,
        user_id=user_id,
        last_message_id=last_message_id,
    )

    if subscriber_queue is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "TASK_NOT_FOUND",
                "message": f"Task {task_id} not found or access denied.",
            },
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        import asyncio

        heartbeat_interval = 15.0  # Send heartbeat every 15 seconds
        try:
            while True:
                try:
                    # Wait for next chunk with timeout for heartbeats
                    chunk = await asyncio.wait_for(
                        subscriber_queue.get(), timeout=heartbeat_interval
                    )
                    yield chunk.to_sse()

                    # Check for finish signal
                    if isinstance(chunk, StreamFinish):
                        break
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield StreamHeartbeat().to_sse()
        except Exception as e:
            logger.error(f"Error in task stream {task_id}: {e}", exc_info=True)
        finally:
            # Unsubscribe when client disconnects or stream ends
            try:
                await stream_registry.unsubscribe_from_task(task_id, subscriber_queue)
            except Exception as unsub_err:
                logger.error(
                    f"Error unsubscribing from task {task_id}: {unsub_err}",
                    exc_info=True,
                )
            # AI SDK protocol termination - always yield even if unsubscribe fails
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


@router.get(
    "/tasks/{task_id}",
)
async def get_task_status(
    task_id: str,
    user_id: str | None = Depends(auth.get_user_id),
) -> dict:
    """
    Get the status of a long-running task.

    Args:
        task_id: The task ID to check.
        user_id: Authenticated user ID for ownership validation.

    Returns:
        dict: Task status including task_id, status, tool_name, and operation_id.

    Raises:
        NotFoundError: If task_id is not found or user doesn't have access.
    """
    task = await stream_registry.get_task(task_id)

    if task is None:
        raise NotFoundError(f"Task {task_id} not found.")

    # Validate ownership - if task has an owner, requester must match
    if task.user_id and user_id != task.user_id:
        raise NotFoundError(f"Task {task_id} not found.")

    return {
        "task_id": task.task_id,
        "session_id": task.session_id,
        "status": task.status,
        "tool_name": task.tool_name,
        "operation_id": task.operation_id,
        "created_at": task.created_at.isoformat(),
    }


# ========== External Completion Webhook ==========


@router.post(
    "/operations/{operation_id}/complete",
    status_code=200,
)
async def complete_operation(
    operation_id: str,
    request: OperationCompleteRequest,
    x_api_key: str | None = Header(default=None),
) -> dict:
    """
    External completion webhook for long-running operations.

    Called by Agent Generator (or other services) when an operation completes.
    This triggers the stream registry to publish completion and continue LLM generation.

    Args:
        operation_id: The operation ID to complete.
        request: Completion payload with success status and result/error.
        x_api_key: Internal API key for authentication.

    Returns:
        dict: Status of the completion.

    Raises:
        HTTPException: If API key is invalid or operation not found.
    """
    # Validate internal API key - reject if not configured or invalid
    if not config.internal_api_key:
        logger.error(
            "Operation complete webhook rejected: CHAT_INTERNAL_API_KEY not configured"
        )
        raise HTTPException(
            status_code=503,
            detail="Webhook not available: internal API key not configured",
        )
    if x_api_key != config.internal_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Find task by operation_id
    task = await stream_registry.find_task_by_operation_id(operation_id)
    if task is None:
        raise HTTPException(
            status_code=404,
            detail=f"Operation {operation_id} not found",
        )

    logger.info(
        f"Received completion webhook for operation {operation_id} "
        f"(task_id={task.task_id}, success={request.success})"
    )

    if request.success:
        await process_operation_success(task, request.result)
    else:
        await process_operation_failure(task, request.error)

    return {"status": "ok", "task_id": task.task_id}


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
