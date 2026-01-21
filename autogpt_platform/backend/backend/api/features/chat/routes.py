"""Chat API routes for chat session management and streaming via SSE."""

import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, Depends, Query, Security
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.util.exceptions import NotFoundError

from . import service as chat_service
from .config import ChatConfig
from .model import ChatSession, create_chat_session, get_chat_session, get_user_sessions

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


class SessionDetailResponse(BaseModel):
    """Response model providing complete details for a chat session, including messages."""

    id: str
    created_at: str
    updated_at: str
    user_id: str | None
    messages: list[dict]


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

    Args:
        session_id: The unique identifier for the desired chat session.
        user_id: The optional authenticated user ID, or None for anonymous access.

    Returns:
        SessionDetailResponse: Details for the requested session, or None if not found.

    """
    session = await get_chat_session(session_id, user_id)
    if not session:
        raise NotFoundError(f"Session {session_id} not found.")

    messages = [message.model_dump() for message in session.messages]
    logger.info(
        f"Returning session {session_id}: "
        f"message_count={len(messages)}, "
        f"roles={[m.get('role') for m in messages]}"
    )

    return SessionDetailResponse(
        id=session.session_id,
        created_at=session.started_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        user_id=session.user_id or None,
        messages=messages,
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

    Args:
        session_id: The chat session identifier to associate with the streamed messages.
        request: Request body containing message, is_user_message, and optional context.
        user_id: Optional authenticated user ID.
    Returns:
        StreamingResponse: SSE-formatted response chunks.

    """
    session = await _validate_and_get_session(session_id, user_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        chunk_count = 0
        first_chunk_type: str | None = None
        async for chunk in chat_service.stream_chat_completion(
            session_id,
            request.message,
            is_user_message=request.is_user_message,
            user_id=user_id,
            session=session,  # Pass pre-fetched session to avoid double-fetch
            context=request.context,
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
