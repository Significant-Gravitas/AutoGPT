"""Database operations for chat sessions."""

import logging
from datetime import UTC, datetime
from typing import Any

from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession
from prisma.types import (
    ChatMessageCreateInput,
    ChatSessionCreateInput,
    ChatSessionUpdateInput,
)

from backend.util import json

logger = logging.getLogger(__name__)


async def get_chat_session(session_id: str) -> PrismaChatSession | None:
    """Get a chat session by ID from the database."""
    session = await PrismaChatSession.prisma().find_unique(
        where={"id": session_id},
        include={"Messages": True},
    )
    if session and session.Messages:
        # Sort messages by sequence in Python since Prisma doesn't support order_by in include
        session.Messages.sort(key=lambda m: m.sequence)
    return session


async def create_chat_session(
    session_id: str,
    user_id: str | None,
) -> PrismaChatSession:
    """Create a new chat session in the database."""
    data: ChatSessionCreateInput = {
        "id": session_id,
        "userId": user_id,
        "credentials": json.dumps({}),
        "successfulAgentRuns": json.dumps({}),
        "successfulAgentSchedules": json.dumps({}),
    }
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
        data["credentials"] = json.dumps(credentials)
    if successful_agent_runs is not None:
        data["successfulAgentRuns"] = json.dumps(successful_agent_runs)
    if successful_agent_schedules is not None:
        data["successfulAgentSchedules"] = json.dumps(successful_agent_schedules)
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
    data: ChatMessageCreateInput = {
        "Session": {"connect": {"id": session_id}},
        "role": role,
        "sequence": sequence,
    }

    if content is not None:
        data["content"] = content
    if name is not None:
        data["name"] = name
    if tool_call_id is not None:
        data["toolCallId"] = tool_call_id
    if refusal is not None:
        data["refusal"] = refusal
    if tool_calls is not None:
        data["toolCalls"] = json.dumps(tool_calls)
    if function_call is not None:
        data["functionCall"] = json.dumps(function_call)

    # Update session's updatedAt timestamp
    await PrismaChatSession.prisma().update(
        where={"id": session_id},
        data={"updatedAt": datetime.now(UTC)},
    )

    return await PrismaChatMessage.prisma().create(data=data)


async def add_chat_messages_batch(
    session_id: str,
    messages: list[dict[str, Any]],
    start_sequence: int,
) -> list[PrismaChatMessage]:
    """Add multiple messages to a chat session in a batch."""
    if not messages:
        return []

    created_messages = []
    for i, msg in enumerate(messages):
        data: ChatMessageCreateInput = {
            "Session": {"connect": {"id": session_id}},
            "role": msg["role"],
            "sequence": start_sequence + i,
        }

        if msg.get("content") is not None:
            data["content"] = msg["content"]
        if msg.get("name") is not None:
            data["name"] = msg["name"]
        if msg.get("tool_call_id") is not None:
            data["toolCallId"] = msg["tool_call_id"]
        if msg.get("refusal") is not None:
            data["refusal"] = msg["refusal"]
        if msg.get("tool_calls") is not None:
            data["toolCalls"] = json.dumps(msg["tool_calls"])
        if msg.get("function_call") is not None:
            data["functionCall"] = json.dumps(msg["function_call"])

        created = await PrismaChatMessage.prisma().create(data=data)
        created_messages.append(created)

    # Update session's updatedAt timestamp
    await PrismaChatSession.prisma().update(
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


async def delete_chat_session(session_id: str) -> bool:
    """Delete a chat session and all its messages."""
    try:
        await PrismaChatSession.prisma().delete(where={"id": session_id})
        return True
    except Exception as e:
        logger.error(f"Failed to delete chat session {session_id}: {e}")
        return False


async def get_chat_session_message_count(session_id: str) -> int:
    """Get the number of messages in a chat session."""
    count = await PrismaChatMessage.prisma().count(where={"sessionId": session_id})
    return count
