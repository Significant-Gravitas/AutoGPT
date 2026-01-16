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

from backend.data.db import transaction
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)


async def get_chat_session(session_id: str) -> PrismaChatSession | None:
    """Get a chat session by ID from the database."""
    session = await PrismaChatSession.prisma().find_unique(
        where={"id": session_id},
        include={"Messages": True},
    )
    if session and session.Messages:
        # Sort messages by sequence in Python - Prisma Python client doesn't support
        # order_by in include clauses (unlike Prisma JS), so we sort after fetching
        session.Messages.sort(key=lambda m: m.sequence)
    return session


async def create_chat_session(
    session_id: str,
    user_id: str,
) -> PrismaChatSession:
    """Create a new chat session in the database."""
    data = ChatSessionCreateInput(
        id=session_id,
        userId=user_id,
        credentials=SafeJson({}),
        successfulAgentRuns=SafeJson({}),
        successfulAgentSchedules=SafeJson({}),
    )
    return await PrismaChatSession.prisma().create(
        data=data,
        include={"Messages": True},
    )


async def update_chat_session(
    session_id: str,
    credentials: dict[str, Any] | None = None,
    successful_agent_runs: dict[str, Any] | None = None,
    successful_agent_schedules: dict[str, Any] | None = None,
    total_prompt_tokens: int | None = None,
    total_completion_tokens: int | None = None,
    title: str | None = None,
) -> PrismaChatSession | None:
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
        include={"Messages": True},
    )
    if session and session.Messages:
        # Sort in Python - Prisma Python doesn't support order_by in include clauses
        session.Messages.sort(key=lambda m: m.sequence)
    return session


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
) -> PrismaChatMessage:
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
    return message


async def add_chat_messages_batch(
    session_id: str,
    messages: list[dict[str, Any]],
    start_sequence: int,
) -> list[PrismaChatMessage]:
    """Add multiple messages to a chat session in a batch.

    Uses a transaction for atomicity - if any message creation fails,
    the entire batch is rolled back.
    """
    if not messages:
        return []

    created_messages = []

    async with transaction() as tx:
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

    return created_messages


async def get_user_chat_sessions(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[PrismaChatSession]:
    """Get chat sessions for a user, ordered by most recent."""
    return await PrismaChatSession.prisma().find_many(
        where={"userId": user_id},
        order={"updatedAt": "desc"},
        take=limit,
        skip=offset,
    )


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


async def get_chat_session_message_count(session_id: str) -> int:
    """Get the number of messages in a chat session."""
    count = await PrismaChatMessage.prisma().count(where={"sessionId": session_id})
    return count
