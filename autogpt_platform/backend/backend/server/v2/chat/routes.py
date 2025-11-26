"""Chat API routes for chat session management and streaming via SSE."""

import logging
from collections.abc import AsyncGenerator
from typing import Annotated

from autogpt_libs import auth
from fastapi import APIRouter, Depends, Query, Security
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import backend.server.v2.chat.service as chat_service
from backend.server.v2.chat.config import ChatConfig
from backend.util.exceptions import NotFoundError

config = ChatConfig()


logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["chat"],
)

# ========== Request/Response Models ==========


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


# ========== Routes ==========


@router.post(
    "/sessions",
)
async def create_session(
    user_id: Annotated[str | None, Depends(auth.get_user_id)],
) -> CreateSessionResponse:
    """
    Create a new chat session.

    Initiates a new chat session for either an authenticated or anonymous user.

    Args:
        user_id: The optional authenticated user ID parsed from the JWT. If missing, creates an anonymous session.

    Returns:
        CreateSessionResponse: Details of the created session.

    """
    logger.info(f"Creating session with user_id: {user_id}")

    session = await chat_service.create_chat_session(user_id)

    return CreateSessionResponse(
        id=session.session_id,
        created_at=session.started_at.isoformat(),
        user_id=session.user_id or None,
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
        SessionDetailResponse: Details for the requested session; raises NotFoundError if not found.

    """
    session = await chat_service.get_session(session_id, user_id)
    if not session:
        raise NotFoundError(f"Session {session_id} not found")
    return SessionDetailResponse(
        id=session.session_id,
        created_at=session.started_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        user_id=session.user_id or None,
        messages=[message.model_dump() for message in session.messages],
    )


@router.get(
    "/sessions/{session_id}/stream",
)
async def stream_chat(
    session_id: str,
    message: Annotated[str, Query(min_length=1, max_length=10000)],
    user_id: str | None = Depends(auth.get_user_id),
    is_user_message: bool = Query(default=True),
):
    """
    Stream chat responses for a session.

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
    # Validate session exists before starting the stream
    # This prevents errors after the response has already started
    session = await chat_service.get_session(session_id, user_id)

    if not session:
        raise NotFoundError(f"Session {session_id} not found. ")
    if session.user_id is None and user_id is not None:
        session = await chat_service.assign_user_to_session(session_id, user_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        async for chunk in chat_service.stream_chat_completion(
            session_id,
            message,
            is_user_message=is_user_message,
            user_id=user_id,
            session=session,  # Pass pre-fetched session to avoid double-fetch
        ):
            yield chunk.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
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

    Performs a full cycle test of session creation, assignment, and retrieval. Should always return healthy
    if the service and data layer are operational.

    Returns:
        dict: A status dictionary indicating health, service name, and API version.

    """
    session = await chat_service.create_chat_session(None)
    await chat_service.assign_user_to_session(session.session_id, "test_user")
    await chat_service.get_session(session.session_id, "test_user")

    return {
        "status": "healthy",
        "service": "chat",
        "version": "0.1.0",
    }
