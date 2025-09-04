"""Database operations for chat functionality."""

import logging
from typing import Any

import prisma.errors
import prisma.models
import prisma.types
from openai.types.chat import ChatCompletionMessageParam
from prisma import Json
from prisma.enums import ChatMessageRole

from backend.util.exceptions import NotFoundError

logger = logging.getLogger(__name__)


# ========== ChatSession Functions ==========


async def create_chat_session(
    user_id: str,
) -> prisma.models.ChatSession:
    """Create a new chat session for a user.

    Args:
        user_id: The ID of the user creating the session

    Returns:
        The created ChatSession object

    """
    # For anonymous users, create a temporary user record
    if user_id.startswith("anon_"):
        # Check if anonymous user already exists
        existing_user = await prisma.models.User.prisma().find_unique(
            where={"id": user_id},
        )

        if not existing_user:
            # Create anonymous user with minimal data
            await prisma.models.User.prisma().create(
                data={
                    "id": user_id,
                    "email": f"{user_id}@anonymous.local",
                    "name": "Anonymous User",
                },
            )
            logger.info(f"Created anonymous user: {user_id}")

    return await prisma.models.ChatSession.prisma().create(
        data={
            "userId": user_id,
        },
    )


async def get_chat_session(
    session_id: str,
    user_id: str | None = None,
    include_messages: bool = False,
) -> prisma.models.ChatSession:
    """Get a chat session by ID.

    Args:
        session_id: The ID of the session
        user_id: Optional user ID to verify ownership
        include_messages: Whether to include messages in the response

    Returns:
        The ChatSession object

    Raises:
        NotFoundError: If the session doesn't exist or user doesn't have access

    """
    where_clause: dict[str, Any] = {"id": session_id}
    if user_id:
        where_clause["userId"] = user_id

    session = await prisma.models.ChatSession.prisma().find_first(
        where=prisma.types.ChatSessionWhereInput(**where_clause),  # type: ignore
        include={"messages": include_messages} if include_messages else None,
    )

    if not session:
        msg = f"Chat session {session_id} not found"
        raise NotFoundError(msg)

    return session


async def list_chat_sessions(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    include_last_message: bool = False,
) -> list[prisma.models.ChatSession]:
    """List chat sessions for a user.

    Args:
        user_id: The ID of the user
        limit: Maximum number of sessions to return
        offset: Number of sessions to skip
        include_last_message: Whether to include the last message for each session

    Returns:
        List of ChatSession objects

    """
    where_clause: dict[str, Any] = {"userId": user_id}

    include_clause = None
    if include_last_message:
        include_clause = {"messages": {"take": 1, "order": [{"sequence": "desc"}]}}

    return await prisma.models.ChatSession.prisma().find_many(
        where=prisma.types.ChatSessionWhereInput(**where_clause),  # type: ignore
        include=include_clause,  # type: ignore
        order=[{"updatedAt": "desc"}],
        skip=offset,
        take=limit,
    )


# ========== ChatMessage Functions ==========


async def create_chat_message(
    session_id: str,
    content: str,
    role: ChatMessageRole,
    sequence: int | None = None,
    tool_call_id: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    parent_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    error: str | None = None,
) -> prisma.models.ChatMessage:
    """Create a new chat message.

    Args:
        session_id: The ID of the chat session
        content: The message content
        role: The role of the message sender
        sequence: Optional sequence number (auto-incremented if not provided)
        tool_call_id: For tool responses
        tool_calls: List of tool calls made by assistant
        parent_id: Parent message ID for threading
        metadata: Additional metadata
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used
        error: Error message if any

    Returns:
        The created ChatMessage object

    """
    # Auto-increment sequence if not provided
    if sequence is None:
        last_message = await prisma.models.ChatMessage.prisma().find_first(
            where={"sessionId": session_id},
            order=[{"sequence": "desc"}],
        )
        sequence = (last_message.sequence + 1) if last_message else 0

    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    # Build the data dict dynamically to avoid setting None values
    data: dict[str, Any] = {
        "sessionId": session_id,
        "content": content,
        "role": role,
        "sequence": sequence,
    }

    # Only add optional fields if they have values
    if tool_call_id:
        data["toolCallId"] = tool_call_id
    if tool_calls:
        data["toolCalls"] = Json(tool_calls)  # type: ignore
    if parent_id:
        data["parentId"] = parent_id
    if metadata:
        data["metadata"] = Json(metadata)
    if prompt_tokens is not None:
        data["promptTokens"] = prompt_tokens
    if completion_tokens is not None:
        data["completionTokens"] = completion_tokens
    if total_tokens is not None:
        data["totalTokens"] = total_tokens
    if error:
        data["error"] = error

    message = await prisma.models.ChatMessage.prisma().create(data=prisma.types.ChatMessageCreateInput(**data))  # type: ignore

    # Update session's updatedAt timestamp
    await prisma.models.ChatSession.prisma().update(where={"id": session_id}, data={})

    return message


async def get_chat_messages(
    session_id: str,
    limit: int | None = None,
    offset: int = 0,
    parent_id: str | None = None,
    include_children: bool = False,
) -> list[prisma.models.ChatMessage]:
    """Get messages for a chat session.

    Args:
        session_id: The ID of the chat session
        limit: Maximum number of messages to return
        offset: Number of messages to skip
        parent_id: Filter by parent message (for threaded conversations)
        include_children: Whether to include child messages

    Returns:
        List of ChatMessage objects ordered by sequence

    """
    where_clause: dict[str, Any] = {"sessionId": session_id}

    if parent_id is not None:
        where_clause["parentId"] = parent_id

    include_clause = {"children": True} if include_children else None

    return await prisma.models.ChatMessage.prisma().find_many(
        where=prisma.types.ChatMessageWhereInput(**where_clause),  # type: ignore
        include=include_clause,  # type: ignore
        order=[{"sequence": "asc"}],
        skip=offset,
        take=limit,
    )


# ========== Helper Functions ==========


async def get_conversation_context(
    session_id: str,
    max_messages: int = 50,
    include_system: bool = True,
) -> list[ChatCompletionMessageParam]:
    """Get the conversation context formatted for OpenAI API.

    Args:
        session_id: The ID of the chat session
        max_messages: Maximum number of messages to include
        include_system: Whether to include system messages

    Returns:
        List of ChatCompletionMessageParam for OpenAI API

    """
    messages = await get_chat_messages(session_id, limit=max_messages)

    context: list[ChatCompletionMessageParam] = []
    for msg in messages:
        if not include_system and msg.role == ChatMessageRole.SYSTEM:
            continue

        # Handle role - it might be a string or an enum
        role_value = msg.role.value if hasattr(msg.role, "value") else msg.role
        role = role_value.lower()

        message: dict[str, Any]

        # Build the message based on role
        if role == "assistant" and msg.toolCalls:
            # Assistant message with tool calls
            message = {
                "role": "assistant",
                "content": msg.content if msg.content else None,
                "tool_calls": msg.toolCalls,
            }
        elif role == "tool":
            # Tool response message
            message = {
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.toolCallId or "",
            }
        elif role == "system":
            # System message
            message = {
                "role": "system",
                "content": msg.content,
            }
        elif role == "user":
            # User message
            message = {
                "role": "user",
                "content": msg.content,
            }
        else:
            # Default assistant message
            message = {
                "role": "assistant",
                "content": msg.content,
            }

        context.append(message)  # type: ignore

    return context
