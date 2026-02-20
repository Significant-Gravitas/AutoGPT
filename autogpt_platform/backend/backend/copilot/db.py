"""Database operations for chat sessions."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, cast

from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession
from prisma.types import (
    ChatMessageCreateInput,
    ChatSessionCreateInput,
    ChatSessionUpdateInput,
    ChatSessionWhereInput,
)

from backend.data import db
from backend.util.json import SafeJson

from .model import ChatMessage, ChatSession, ChatSessionInfo

logger = logging.getLogger(__name__)


async def get_chat_session(session_id: str) -> ChatSession | None:
    """Get a chat session by ID from the database."""
    session = await PrismaChatSession.prisma().find_unique(
        where={"id": session_id},
        include={"Messages": {"order_by": {"sequence": "asc"}}},
    )
    return ChatSession.from_db(session) if session else None


async def create_chat_session(
    session_id: str,
    user_id: str,
) -> ChatSessionInfo:
    """Create a new chat session in the database."""
    data = ChatSessionCreateInput(
        id=session_id,
        userId=user_id,
        credentials=SafeJson({}),
        successfulAgentRuns=SafeJson({}),
        successfulAgentSchedules=SafeJson({}),
    )
    prisma_session = await PrismaChatSession.prisma().create(data=data)
    return ChatSessionInfo.from_db(prisma_session)


async def update_chat_session(
    session_id: str,
    credentials: dict[str, Any] | None = None,
    successful_agent_runs: dict[str, Any] | None = None,
    successful_agent_schedules: dict[str, Any] | None = None,
    total_prompt_tokens: int | None = None,
    total_completion_tokens: int | None = None,
    title: str | None = None,
) -> ChatSession | None:
    """Update a chat session's metadata."""
    data: ChatSessionUpdateInput = {"updatedAt": datetime.now(UTC)}

    if credentials is not None:
        data["credentials"] = SafeJson(credentials)
    if successful_agent_runs is not None:
        data["successfulAgentRuns"] = SafeJson(successful_agent_runs)
    if successful_agent_schedules is not None:
        data["successfulAgentSchedules"] = SafeJson(successful_agent_schedules)
    if total_prompt_tokens is not None:
        data["totalPromptTokens"] = total_prompt_tokens
    if total_completion_tokens is not None:
        data["totalCompletionTokens"] = total_completion_tokens
    if title is not None:
        data["title"] = title

    session = await PrismaChatSession.prisma().update(
        where={"id": session_id},
        data=data,
        include={"Messages": {"order_by": {"sequence": "asc"}}},
    )
    return ChatSession.from_db(session) if session else None


async def add_chat_message(
    session_id: str,
    role: str,
    sequence: int,
    content: str | None = None,
    name: str | None = None,
    tool_call_id: str | None = None,
    refusal: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    function_call: dict[str, Any] | None = None,
) -> ChatMessage:
    """Add a message to a chat session."""
    # Build input dict dynamically rather than using ChatMessageCreateInput directly
    # because Prisma's TypedDict validation rejects optional fields set to None.
    # We only include fields that have values, then cast at the end.
    data: dict[str, Any] = {
        "Session": {"connect": {"id": session_id}},
        "role": role,
        "sequence": sequence,
    }

    # Add optional string fields
    if content is not None:
        data["content"] = content
    if name is not None:
        data["name"] = name
    if tool_call_id is not None:
        data["toolCallId"] = tool_call_id
    if refusal is not None:
        data["refusal"] = refusal

    # Add optional JSON fields only when they have values
    if tool_calls is not None:
        data["toolCalls"] = SafeJson(tool_calls)
    if function_call is not None:
        data["functionCall"] = SafeJson(function_call)

    # Run message create and session timestamp update in parallel for lower latency
    _, message = await asyncio.gather(
        PrismaChatSession.prisma().update(
            where={"id": session_id},
            data={"updatedAt": datetime.now(UTC)},
        ),
        PrismaChatMessage.prisma().create(data=cast(ChatMessageCreateInput, data)),
    )
    return ChatMessage.from_db(message)


async def add_chat_messages_batch(
    session_id: str,
    messages: list[dict[str, Any]],
    start_sequence: int,
) -> tuple[list[ChatMessage], int]:
    """Add multiple messages to a chat session in a batch.

    Uses collision detection with retry: tries to create messages starting
    at start_sequence. If a unique constraint violation occurs (e.g., the
    streaming loop and long-running callback race), queries MAX(sequence)
    and retries with the correct next sequence number. This avoids
    unnecessary upserts and DB queries in the common case (no collision).

    Returns:
        Tuple of (messages, final_message_count) where final_message_count
        is the total number of messages in the session after insertion.
        This allows callers to update their counters even when collision
        detection adjusts start_sequence.
    """
    if not messages:
        # No messages to add - return current count
        return [], start_sequence

    max_retries = 3
    for attempt in range(max_retries):
        try:
            created_messages = []
            async with db.transaction() as tx:
                for i, msg in enumerate(messages):
                    # Build input dict dynamically rather than using ChatMessageCreateInput
                    # directly because Prisma's TypedDict validation rejects optional fields
                    # set to None. We only include fields that have values, then cast.
                    data: dict[str, Any] = {
                        "Session": {"connect": {"id": session_id}},
                        "role": msg["role"],
                        "sequence": start_sequence + i,
                    }

                    # Add optional string fields
                    if msg.get("content") is not None:
                        data["content"] = msg["content"]
                    if msg.get("name") is not None:
                        data["name"] = msg["name"]
                    if msg.get("tool_call_id") is not None:
                        data["toolCallId"] = msg["tool_call_id"]
                    if msg.get("refusal") is not None:
                        data["refusal"] = msg["refusal"]

                    # Add optional JSON fields only when they have values
                    if msg.get("tool_calls") is not None:
                        data["toolCalls"] = SafeJson(msg["tool_calls"])
                    if msg.get("function_call") is not None:
                        data["functionCall"] = SafeJson(msg["function_call"])

                    created = await PrismaChatMessage.prisma(tx).create(
                        data=cast(ChatMessageCreateInput, data)
                    )
                    created_messages.append(created)

                # Update session's updatedAt timestamp within the same transaction.
                # Note: Token usage (total_prompt_tokens, total_completion_tokens) is updated
                # separately via update_chat_session() after streaming completes.
                await PrismaChatSession.prisma(tx).update(
                    where={"id": session_id},
                    data={"updatedAt": datetime.now(UTC)},
                )

            # Return messages and final message count (for shared counter sync)
            final_count = start_sequence + len(messages)
            return [ChatMessage.from_db(m) for m in created_messages], final_count

        except Exception as e:
            # Check if it's a unique constraint violation
            error_msg = str(e).lower()
            is_unique_constraint = (
                "unique constraint" in error_msg or "duplicate key" in error_msg
            )

            if is_unique_constraint and attempt < max_retries - 1:
                # Collision detected - query MAX(sequence)+1 and retry with correct offset
                logger.info(
                    f"Collision detected for session {session_id} at sequence "
                    f"{start_sequence}, querying DB for latest sequence"
                )
                start_sequence = await get_next_sequence(session_id)
                logger.info(
                    f"Retrying batch insert with start_sequence={start_sequence}"
                )
                continue
            else:
                # Not a collision or max retries exceeded - propagate error
                raise

    # Should never reach here due to raise in exception handler
    raise RuntimeError(f"Failed to insert messages after {max_retries} attempts")


async def get_user_chat_sessions(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[ChatSessionInfo]:
    """Get chat sessions for a user, ordered by most recent."""
    prisma_sessions = await PrismaChatSession.prisma().find_many(
        where={"userId": user_id},
        order={"updatedAt": "desc"},
        take=limit,
        skip=offset,
    )
    return [ChatSessionInfo.from_db(s) for s in prisma_sessions]


async def get_user_session_count(user_id: str) -> int:
    """Get the total number of chat sessions for a user."""
    return await PrismaChatSession.prisma().count(where={"userId": user_id})


async def delete_chat_session(session_id: str, user_id: str | None = None) -> bool:
    """Delete a chat session and all its messages.

    Args:
        session_id: The session ID to delete.
        user_id: If provided, validates that the session belongs to this user
            before deletion. This prevents unauthorized deletion of other
            users' sessions.

    Returns:
        True if deleted successfully, False otherwise.
    """
    try:
        # Build typed where clause with optional user_id validation
        where_clause: ChatSessionWhereInput = {"id": session_id}
        if user_id is not None:
            where_clause["userId"] = user_id

        result = await PrismaChatSession.prisma().delete_many(where=where_clause)
        if result == 0:
            logger.warning(
                f"No session deleted for {session_id} "
                f"(user_id validation: {user_id is not None})"
            )
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to delete chat session {session_id}: {e}")
        return False


async def get_next_sequence(session_id: str) -> int:
    """Get the next sequence number for a new message in this session.

    Uses MAX(sequence) + 1 for robustness. Returns 0 if no messages exist.
    More robust than COUNT(*) because it's immune to deleted messages.
    """
    result = await db.prisma.query_raw(
        """
        SELECT COALESCE(MAX(sequence) + 1, 0) as next_seq
        FROM "ChatMessage"
        WHERE "sessionId" = $1
        """,
        session_id,
    )
    if not result or len(result) == 0:
        return 0
    return int(result[0]["next_seq"])


async def update_tool_message_content(
    session_id: str,
    tool_call_id: str,
    new_content: str,
) -> bool:
    """Update the content of a tool message in chat history.

    Used by background tasks to update pending operation messages with final results.

    Args:
        session_id: The chat session ID.
        tool_call_id: The tool call ID to find the message.
        new_content: The new content to set.

    Returns:
        True if a message was updated, False otherwise.
    """
    try:
        result = await PrismaChatMessage.prisma().update_many(
            where={
                "sessionId": session_id,
                "toolCallId": tool_call_id,
            },
            data={
                "content": new_content,
            },
        )
        if result == 0:
            logger.warning(
                f"No message found to update for session {session_id}, "
                f"tool_call_id {tool_call_id}"
            )
            return False
        return True
    except Exception as e:
        logger.error(
            f"Failed to update tool message for session {session_id}, "
            f"tool_call_id {tool_call_id}: {e}"
        )
        return False
