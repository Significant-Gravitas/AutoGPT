"""Chat API routes for SSE streaming and session management."""

import logging
from typing import Annotated

import prisma.models
from autogpt_libs import auth
from fastapi import APIRouter, Depends, HTTPException, Query, Security
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from prisma.enums import ChatMessageRole
from pydantic import BaseModel, Field
from starlette.status import HTTP_404_NOT_FOUND

from backend.server.v2.chat import chat, db
from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)

# Optional bearer token authentication
optional_bearer = HTTPBearer(auto_error=False)

router = APIRouter(
    tags=["chat"],
    responses={
        404: {"description": "Resource not found"},
        401: {"description": "Unauthorized"},
    },
)


def get_optional_user_id(
    credentials: HTTPAuthorizationCredentials | None = Security(optional_bearer),
) -> str | None:
    """Get user ID from auth token if present, otherwise None for anonymous."""
    if not credentials:
        return None

    try:
        # Parse JWT token to get user ID
        from autogpt_libs.auth.jwt_utils import parse_jwt_token

        payload = parse_jwt_token(credentials.credentials)
        return payload.get("sub")
    except Exception as e:
        logger.debug(f"Auth token validation failed (anonymous access): {e}")
        return None


# ========== Request/Response Models ==========


class CreateSessionRequest(BaseModel):
    """Request model for creating a new chat session."""

    metadata: dict | None = Field(
        default_factory=dict,
        description="Optional metadata",
    )


class CreateSessionResponse(BaseModel):
    """Response model for created chat session."""

    id: str
    created_at: str
    user_id: str


class SendMessageRequest(BaseModel):
    """Request model for sending a chat message."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Message content",
    )
    model: str = Field(default="gpt-4o", description="AI model to use")
    max_context_messages: int = Field(
        default=50,
        ge=1,
        le=100,
        description="Max context messages",
    )


class SendMessageResponse(BaseModel):
    """Response model for non-streaming message."""

    message_id: str
    content: str
    role: str
    tokens_used: dict | None = None


class SessionListResponse(BaseModel):
    """Response model for session list."""

    sessions: list[dict]
    total: int
    limit: int
    offset: int


class SessionDetailResponse(BaseModel):
    """Response model for session details."""

    id: str
    created_at: str
    updated_at: str
    user_id: str
    messages: list[dict]
    metadata: dict


# ========== Routes ==========


@router.post(
    "/sessions",
)
async def create_session(
    request: CreateSessionRequest,
    user_id: Annotated[str | None, Depends(get_optional_user_id)],
) -> CreateSessionResponse:
    """Create a new chat session for the authenticated or anonymous user.

    Args:
        request: Session creation parameters
        user_id: Optional authenticated user ID

    Returns:
        Created session details

    """
    try:
        logger.info(f"Creating session with user_id: {user_id}")

        # Create the session (anonymous if no user_id)
        # Use a special anonymous user ID if not authenticated
        import uuid

        session_user_id = user_id if user_id else f"anon_{uuid.uuid4().hex[:12]}"
        logger.info(f"Using session_user_id: {session_user_id}")

        session = await db.create_chat_session(user_id=session_user_id)

        logger.info(f"Created chat session {session.id} for user {user_id}")

        return CreateSessionResponse(
            id=session.id,
            created_at=session.createdAt.isoformat(),
            user_id=session.userId,
        )
    except Exception as e:
        logger.error(f"Failed to create session: {e!s}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e!s}")


@router.get(
    "/sessions",
    dependencies=[Security(auth.requires_user)],
)
async def list_sessions(
    user_id: Annotated[str, Security(auth.get_user_id)],
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    include_last_message: Annotated[bool, Query()] = True,
) -> SessionListResponse:
    """List chat sessions for the authenticated user.

    Args:
        limit: Maximum number of sessions to return
        offset: Number of sessions to skip
        include_last_message: Whether to include the last message
        user_id: Authenticated user ID

    Returns:
        List of user's chat sessions

    """
    try:
        sessions = await db.list_chat_sessions(
            user_id=user_id,
            limit=limit,
            offset=offset,
            include_last_message=include_last_message,
        )

        # Format sessions for response
        session_list = []
        for session in sessions:
            session_dict = {
                "id": session.id,
                "created_at": session.createdAt.isoformat(),
                "updated_at": session.updatedAt.isoformat(),
            }

            # Add last message if included
            if include_last_message and session.messages:
                last_msg = session.messages[0]
                session_dict["last_message"] = {
                    "content": last_msg.content[:100],  # Preview
                    "role": last_msg.role,
                    "created_at": last_msg.createdAt.isoformat(),
                }

            session_list.append(session_dict)

        return SessionListResponse(
            sessions=session_list,
            total=len(session_list),  # TODO: Add total count query
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.exception(f"Failed to list sessions: {e!s}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")


@router.get(
    "/sessions/{session_id}",
)
async def get_session(
    session_id: str,
    user_id: Annotated[str | None, Depends(get_optional_user_id)],
    include_messages: Annotated[bool, Query()] = True,
) -> SessionDetailResponse:
    """Get details of a specific chat session.

    Args:
        session_id: ID of the session to retrieve
        include_messages: Whether to include all messages
        user_id: Authenticated user ID

    Returns:
        Session details with optional messages

    """
    try:
        # For anonymous sessions, we don't check ownership
        if user_id:
            # Authenticated user - verify ownership
            session = await db.get_chat_session(
                session_id=session_id,
                user_id=user_id,
                include_messages=include_messages,
            )
        else:
            # Anonymous user - just get the session by ID
            from backend.data.db import prisma

            if include_messages:
                session = await prisma.chatsession.find_unique(
                    where={"id": session_id},
                    include={"messages": True},
                )
                # Sort messages if they were included
                if session and session.messages:
                    session.messages.sort(key=lambda m: m.sequence)
            else:
                session = await prisma.chatsession.find_unique(
                    where={"id": session_id},
                )
            if not session:
                msg = f"Session {session_id} not found"
                raise NotFoundError(msg)

        # Format messages if included
        messages = []
        if include_messages and session.messages:
            for msg in session.messages:
                messages.append(
                    {
                        "id": msg.id,
                        "content": msg.content,
                        "role": msg.role,
                        "created_at": msg.createdAt.isoformat(),
                        "tool_calls": msg.toolCalls,
                        "tool_call_id": msg.toolCallId,
                        "tokens": (
                            {
                                "prompt": msg.promptTokens,
                                "completion": msg.completionTokens,
                                "total": msg.totalTokens,
                            }
                            if msg.totalTokens
                            else None
                        ),
                    },
                )

        return SessionDetailResponse(
            id=session.id,
            created_at=session.createdAt.isoformat(),
            updated_at=session.updatedAt.isoformat(),
            user_id=session.userId,
            messages=messages,
            metadata={},  # TODO: Add session metadata support
        )
    except NotFoundError:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    except Exception as e:
        logger.exception(f"Failed to get session: {e!s}")
        raise HTTPException(status_code=500, detail="Failed to get session")


@router.delete("/sessions/{session_id}", dependencies=[Security(auth.requires_user)])
async def delete_session(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> dict:
    """Delete a chat session and all its messages.

    Args:
        session_id: ID of the session to delete
        user_id: Authenticated user ID

    Returns:
        Deletion confirmation

    """
    try:
        # Verify ownership first
        await db.get_chat_session(
            session_id=session_id,
            user_id=user_id,
        )

        # Delete the session (cascade deletes messages)
        await prisma.models.ChatSession.prisma().delete(where={"id": session_id})

        logger.info(f"Deleted session {session_id} for user {user_id}")

        return {"status": "success", "message": f"Session {session_id} deleted"}
    except NotFoundError:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    except Exception as e:
        logger.exception(f"Failed to delete session: {e!s}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


@router.post(
    "/sessions/{session_id}/messages",
    dependencies=[Security(auth.requires_user)],
)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> SendMessageResponse:
    """Send a message to a chat session (non-streaming).

    This endpoint processes the message and returns the complete response.
    For streaming responses, use the /stream endpoint.

    Args:
        session_id: ID of the session
        request: Message parameters
        user_id: Authenticated user ID

    Returns:
        Complete assistant response

    """
    try:
        # Verify session ownership
        await db.get_chat_session(
            session_id=session_id,
            user_id=user_id,
        )

        # Store user message
        await db.create_chat_message(
            session_id=session_id,
            content=request.message,
            role=ChatMessageRole.USER,
        )

        # Collect the complete response using the refactored function
        full_response = ""
        async for chunk in chat.stream_chat_completion(
            session_id=session_id,
            user_message=request.message,
            user_id=user_id,
            model=request.model,
            max_messages=request.max_context_messages,
        ):
            # Parse SSE data
            if chunk.startswith("data: "):
                import json

                try:
                    data = json.loads(chunk[6:].strip())
                    if data.get("type") == "text_chunk":
                        full_response += data.get("content", "")
                except json.JSONDecodeError:
                    continue

        # Get the last assistant message for token counts
        messages = await db.get_chat_messages(session_id=session_id, limit=1)
        last_msg = messages[0] if messages else None

        tokens_used = None
        if last_msg and last_msg.totalTokens:
            tokens_used = {
                "prompt": last_msg.promptTokens,
                "completion": last_msg.completionTokens,
                "total": last_msg.totalTokens,
            }

        return SendMessageResponse(
            message_id=last_msg.id if last_msg else "",
            content=full_response.strip(),
            role="ASSISTANT",
            tokens_used=tokens_used,
        )
    except NotFoundError:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    except Exception as e:
        logger.exception(f"Failed to send message: {e!s}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {e!s}")


@router.get(
    "/sessions/{session_id}/stream",
)
async def stream_chat(
    session_id: str,
    message: Annotated[str, Query(min_length=1, max_length=10000)] = ...,
    model: Annotated[str, Query()] = "gpt-4o",
    max_context: Annotated[int, Query(ge=1, le=100)] = 50,
    user_id: str | None = Depends(get_optional_user_id),
):
    """Stream chat responses using Server-Sent Events (SSE).

    This endpoint streams the AI response in real-time, including:
    - Text chunks as they're generated
    - Tool call UI elements
    - Tool execution results

    Args:
        session_id: ID of the session
        message: User's message
        model: AI model to use
        max_context: Maximum context messages
        user_id: Optional authenticated user ID

    Returns:
        SSE stream of response chunks

    """
    try:

        # Get session - allow anonymous access by session ID
        # For anonymous users, we just verify the session exists
        if user_id:
            session = await db.get_chat_session(
                session_id=session_id,
                user_id=user_id,
            )
        else:
            # For anonymous, just verify session exists (no ownership check)
            from backend.data.db import prisma

            session = await prisma.chatsession.find_unique(
                where={"id": session_id},
            )
            if not session:
                msg = f"Session {session_id} not found"
                raise NotFoundError(msg)

        # Use the session's user_id for tool execution
        effective_user_id = user_id if user_id else session.userId

        logger.info(f"Starting SSE stream for session {session_id}")

        # Get the streaming generator using the refactored function
        stream_generator = chat.stream_chat_completion(
            session_id=session_id,
            user_message=message,
            user_id=effective_user_id,
            model=model,
            max_messages=max_context,
        )

        # Return as SSE stream
        return StreamingResponse(
            stream_generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
                "Access-Control-Allow-Origin": "*",  # TODO: Configure proper CORS
            },
        )
    except NotFoundError:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    except Exception as e:
        logger.exception(f"Failed to stream chat: {e!s}")
        raise HTTPException(status_code=500, detail=f"Failed to stream: {e!s}")


@router.patch(
    "/sessions/{session_id}/assign-user", dependencies=[Security(auth.requires_user)]
)
async def assign_user_to_session(
    session_id: str,
    user_id: Annotated[str, Security(auth.get_user_id)],
) -> dict:
    """Assign an authenticated user to an anonymous session.

    This is called after a user logs in to claim their anonymous session.

    Args:
        session_id: ID of the anonymous session
        user_id: Authenticated user ID

    Returns:
        Success status

    """
    try:
        # Get the session (should be anonymous)
        from backend.data.db import prisma

        session = await prisma.chatsession.find_unique(
            where={"id": session_id},
        )

        if not session:
            raise HTTPException(
                status_code=HTTP_404_NOT_FOUND,
                detail=f"Session {session_id} not found",
            )

        # Check if session is anonymous (starts with anon_)
        if not session.userId.startswith("anon_"):
            raise HTTPException(
                status_code=400,
                detail="Session already has an assigned user",
            )

        # Update the session with the real user ID
        await prisma.chatsession.update(
            where={"id": session_id},
            data={"userId": user_id},
        )

        logger.info(f"Assigned user {user_id} to session {session_id}")

        return {
            "status": "success",
            "message": f"Session {session_id} assigned to user",
            "user_id": user_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to assign user to session: {e!s}")
        raise HTTPException(status_code=500, detail="Failed to assign user")


# ========== Health Check ==========


@router.get("/health")
async def health_check() -> dict:
    """Check if the chat service is healthy.

    Returns:
        Health status

    """
    try:
        # Try to get the OpenAI client to verify connectivity
        from backend.server.v2.chat.config import get_config

        config = get_config()

        return {
            "status": "healthy",
            "service": "chat",
            "version": "2.0",
            "model": config.model,
            "has_api_key": config.api_key is not None,
        }
    except Exception as e:
        logger.exception(f"Health check failed: {e!s}")
        return {
            "status": "unhealthy",
            "error": str(e),
        }
