"""Database operations for chat sessions."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from prisma.errors import UniqueViolationError
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession
from prisma.types import (
    ChatMessageCreateInput,
    ChatSessionCreateInput,
    ChatSessionUpdateInput,
    ChatSessionWhereInput,
)
from pydantic import BaseModel

from backend.data import db
from backend.util.json import SafeJson, sanitize_string

from .model import (
    ChatMessage,
    ChatSession,
    ChatSessionInfo,
    ChatSessionMetadata,
    cache_chat_session,
)
from .model import get_chat_session as get_chat_session_cached

logger = logging.getLogger(__name__)


class PaginatedMessages(BaseModel):
    """Result of a paginated message query."""

    messages: list[ChatMessage]
    has_more: bool
    oldest_sequence: int | None
    session: ChatSessionInfo


async def get_chat_session(session_id: str) -> ChatSession | None:
    """Get a chat session by ID from the database."""
    session = await PrismaChatSession.prisma().find_unique(
        where={"id": session_id},
        include={"Messages": {"order_by": {"sequence": "asc"}}},
    )
    return ChatSession.from_db(session) if session else None


async def get_chat_session_metadata(session_id: str) -> ChatSessionInfo | None:
    """Get chat session metadata (without messages) for ownership validation."""
    session = await PrismaChatSession.prisma().find_unique(
        where={"id": session_id},
    )
    return ChatSessionInfo.from_db(session) if session else None


async def get_chat_messages_paginated(
    session_id: str,
    limit: int = 50,
    before_sequence: int | None = None,
    user_id: str | None = None,
) -> PaginatedMessages | None:
    """Get paginated messages for a session, newest first.

    Verifies session existence (and ownership when ``user_id`` is provided)
    in parallel with the message query.  Returns ``None`` when the session
    is not found or does not belong to the user.

    Args:
        session_id: The chat session ID.
        limit: Max messages to return.
        before_sequence: Cursor — return messages with sequence < this value.
        user_id: If provided, filters via ``Session.userId`` so only the
            session owner's messages are returned (acts as an ownership guard).
    """
    # Build session-existence / ownership check
    session_where: ChatSessionWhereInput = {"id": session_id}
    if user_id is not None:
        session_where["userId"] = user_id

    # Build message include — fetch paginated messages in the same query
    msg_include: dict[str, Any] = {
        "order_by": {"sequence": "desc"},
        "take": limit + 1,
    }
    if before_sequence is not None:
        msg_include["where"] = {"sequence": {"lt": before_sequence}}

    # Single query: session existence/ownership + paginated messages
    session = await PrismaChatSession.prisma().find_first(
        where=session_where,
        include={"Messages": msg_include},
    )

    if session is None:
        return None

    session_info = ChatSessionInfo.from_db(session)
    results = list(session.Messages) if session.Messages else []

    has_more = len(results) > limit
    results = results[:limit]

    # Reverse to ascending order
    results.reverse()

    # Tool-call boundary fix: if the oldest message is a tool message,
    # expand backward to include the preceding assistant message that
    # owns the tool_calls, so convertChatSessionMessagesToUiMessages
    # can pair them correctly.
    _BOUNDARY_SCAN_LIMIT = 10
    if results and results[0].role == "tool":
        boundary_where: dict[str, Any] = {
            "sessionId": session_id,
            "sequence": {"lt": results[0].sequence},
        }
        if user_id is not None:
            boundary_where["Session"] = {"is": {"userId": user_id}}
        extra = await PrismaChatMessage.prisma().find_many(
            where=boundary_where,
            order={"sequence": "desc"},
            take=_BOUNDARY_SCAN_LIMIT,
        )
        # Find the first non-tool message (should be the assistant)
        boundary_msgs = []
        found_owner = False
        for msg in extra:
            boundary_msgs.append(msg)
            if msg.role != "tool":
                found_owner = True
                break
        boundary_msgs.reverse()
        if not found_owner:
            logger.warning(
                "Boundary expansion did not find owning assistant message "
                "for session=%s before sequence=%s (%d msgs scanned)",
                session_id,
                results[0].sequence,
                len(extra),
            )
        if boundary_msgs:
            results = boundary_msgs + results
            # Only mark has_more if the expanded boundary isn't the
            # very start of the conversation (sequence 0).
            if boundary_msgs[0].sequence > 0:
                has_more = True

    messages = [ChatMessage.from_db(m) for m in results]
    oldest_sequence = messages[0].sequence if messages else None

    return PaginatedMessages(
        messages=messages,
        has_more=has_more,
        oldest_sequence=oldest_sequence,
        session=session_info,
    )


async def create_chat_session(
    session_id: str,
    user_id: str,
    metadata: ChatSessionMetadata | None = None,
) -> ChatSessionInfo:
    """Create a new chat session in the database."""
    data = ChatSessionCreateInput(
        id=session_id,
        userId=user_id,
        credentials=SafeJson({}),
        successfulAgentRuns=SafeJson({}),
        successfulAgentSchedules=SafeJson({}),
        metadata=SafeJson((metadata or ChatSessionMetadata()).model_dump()),
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
    """Update a chat session's mutable fields.

    Note: ``metadata`` (which includes ``dry_run``) is intentionally omitted —
    it is set once at creation time and treated as immutable for the lifetime
    of the session.
    """
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


async def update_chat_session_title(
    session_id: str,
    user_id: str,
    title: str,
    *,
    only_if_empty: bool = False,
) -> bool:
    """Update the title of a chat session, scoped to the owning user.

    Always filters by (session_id, user_id) so callers cannot mutate another
    user's session even when they know the session_id.

    Args:
        only_if_empty: When True, uses an atomic ``UPDATE WHERE title IS NULL``
            guard so auto-generated titles never overwrite a user-set title.

    Returns True if a row was updated, False otherwise (session not found,
    wrong user, or — when only_if_empty — title was already set).
    """
    where: ChatSessionWhereInput = {"id": session_id, "userId": user_id}
    if only_if_empty:
        where["title"] = None
    result = await PrismaChatSession.prisma().update_many(
        where=where,
        data={"title": title, "updatedAt": datetime.now(UTC)},
    )
    return result > 0


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
    # Build ChatMessageCreateInput with only non-None values
    # (Prisma TypedDict rejects optional fields set to None)
    data: ChatMessageCreateInput = {
        "Session": {"connect": {"id": session_id}},
        "role": role,
        "sequence": sequence,
    }

    # Add optional string fields — sanitize to strip PostgreSQL-incompatible
    # control characters (null bytes etc.) that may appear in tool outputs.
    if content is not None:
        data["content"] = sanitize_string(content)
    if name is not None:
        data["name"] = name
    if tool_call_id is not None:
        data["toolCallId"] = tool_call_id
    if refusal is not None:
        data["refusal"] = sanitize_string(refusal)

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
        PrismaChatMessage.prisma().create(data=data),
    )
    return ChatMessage.from_db(message)


async def add_chat_messages_batch(
    session_id: str,
    messages: list[dict[str, Any]],
    start_sequence: int,
) -> int:
    """Add multiple messages to a chat session in a batch.

    Uses collision detection with retry: tries to create messages starting
    at start_sequence. If a unique constraint violation occurs (e.g., the
    streaming loop and long-running callback race), queries the latest
    sequence and retries with the correct offset. This avoids unnecessary
    upserts and DB queries in the common case (no collision).

    Returns:
        Next sequence number for the next message to be inserted. This equals
        start_sequence + len(messages) and allows callers to update their
        counters even when collision detection adjusts start_sequence.
    """
    if not messages:
        # No messages to add - return current count
        return start_sequence

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # Single timestamp for all messages and session update
            now = datetime.now(UTC)

            async with db.transaction() as tx:
                # Build all message data
                messages_data = []
                for i, msg in enumerate(messages):
                    # Build ChatMessageCreateInput with only non-None values
                    # (Prisma TypedDict rejects optional fields set to None)
                    # Note: create_many doesn't support nested creates, use sessionId directly
                    data: ChatMessageCreateInput = {
                        "sessionId": session_id,
                        "role": msg["role"],
                        "sequence": start_sequence + i,
                        "createdAt": now,
                    }

                    # Add optional string fields — sanitize to strip
                    # PostgreSQL-incompatible control characters.
                    if msg.get("content") is not None:
                        data["content"] = sanitize_string(msg["content"])
                    if msg.get("name") is not None:
                        data["name"] = msg["name"]
                    if msg.get("tool_call_id") is not None:
                        data["toolCallId"] = msg["tool_call_id"]
                    if msg.get("refusal") is not None:
                        data["refusal"] = sanitize_string(msg["refusal"])

                    # Add optional JSON fields only when they have values
                    if msg.get("tool_calls") is not None:
                        data["toolCalls"] = SafeJson(msg["tool_calls"])
                    if msg.get("function_call") is not None:
                        data["functionCall"] = SafeJson(msg["function_call"])

                    if msg.get("duration_ms") is not None:
                        data["durationMs"] = msg["duration_ms"]

                    messages_data.append(data)

                # Run create_many and session update in parallel within transaction
                # Both use the same timestamp for consistency
                await asyncio.gather(
                    PrismaChatMessage.prisma(tx).create_many(data=messages_data),
                    PrismaChatSession.prisma(tx).update(
                        where={"id": session_id},
                        data={"updatedAt": now},
                    ),
                )

            # Return next sequence number for counter sync
            return start_sequence + len(messages)

        except UniqueViolationError:
            if attempt < max_retries - 1:
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
                # Max retries exceeded - propagate error
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

    Optimized to select only the sequence column using raw SQL.
    The unique index on (sessionId, sequence) makes this query fast.
    """
    results = await db.query_raw_with_schema(
        'SELECT "sequence" FROM {schema_prefix}"ChatMessage" WHERE "sessionId" = $1 ORDER BY "sequence" DESC LIMIT 1',
        session_id,
    )
    return 0 if not results else results[0]["sequence"] + 1


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
                "content": sanitize_string(new_content),
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


async def update_message_content_by_sequence(
    session_id: str,
    sequence: int,
    new_content: str,
) -> bool:
    """Update the content of a specific message by its sequence number.

    Used to persist content modifications (e.g. user-context prefix injection)
    to a message that was already saved to the DB.

    Args:
        session_id: The chat session ID.
        sequence: The 0-based sequence number of the message to update.
        new_content: The new content to set.

    Returns:
        True if a message was updated, False otherwise.
    """
    try:
        result = await PrismaChatMessage.prisma().update_many(
            where={"sessionId": session_id, "sequence": sequence},
            data={"content": sanitize_string(new_content)},
        )
        if result == 0:
            logger.warning(
                f"No message found to update for session {session_id}, sequence {sequence}"
            )
            return False
        return True
    except Exception as e:
        logger.error(
            f"Failed to update message for session {session_id}, sequence {sequence}: {e}"
        )
        return False


async def set_turn_duration(session_id: str, duration_ms: int) -> None:
    """Set durationMs on the last assistant message in a session.

    Updates the Redis cache in-place instead of invalidating it.
    Invalidation would delete the key, creating a window where concurrent
    ``get_chat_session`` calls re-populate the cache from DB — potentially
    with stale data if the DB write from the previous turn hasn't propagated.
    This race caused duplicate user messages on the next turn.
    """
    last_msg = await PrismaChatMessage.prisma().find_first(
        where={"sessionId": session_id, "role": "assistant"},
        order={"sequence": "desc"},
    )
    if last_msg:
        await PrismaChatMessage.prisma().update(
            where={"id": last_msg.id},
            data={"durationMs": duration_ms},
        )
        # Update cache in-place rather than invalidating to avoid a
        # race window where the empty cache gets re-populated with
        # stale data by a concurrent get_chat_session call.
        session = await get_chat_session_cached(session_id)
        if session and session.messages:
            for msg in reversed(session.messages):
                if msg.role == "assistant":
                    msg.duration_ms = duration_ms
                    break
            await cache_chat_session(session)
