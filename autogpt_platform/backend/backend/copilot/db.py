"""Database operations for chat sessions."""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from prisma.errors import UniqueViolationError
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession
from prisma.models import ChatSessionCallbackToken as PrismaChatSessionCallbackToken
from prisma.types import (
    ChatMessageCreateInput,
    ChatSessionCreateInput,
    ChatSessionUpdateInput,
    ChatSessionWhereInput,
)
from pydantic import BaseModel

from backend.data import db
from backend.util.json import SafeJson, sanitize_string

from .model import ChatMessage, ChatSession, ChatSessionInfo
from .session_types import ChatSessionStartType

logger = logging.getLogger(__name__)
_UNSET = object()


class ChatSessionCallbackTokenInfo(BaseModel):
    id: str
    user_id: str
    source_session_id: str | None = None
    callback_session_message: str
    expires_at: datetime
    consumed_at: datetime | None = None
    consumed_session_id: str | None = None

    @classmethod
    def from_db(
        cls,
        token: PrismaChatSessionCallbackToken,
    ) -> "ChatSessionCallbackTokenInfo":
        return cls(
            id=token.id,
            user_id=token.userId,
            source_session_id=token.sourceSessionId,
            callback_session_message=token.callbackSessionMessage,
            expires_at=token.expiresAt,
            consumed_at=token.consumedAt,
            consumed_session_id=token.consumedSessionId,
        )


async def get_chat_session(session_id: str) -> ChatSession | None:
    """Get a chat session by ID from the database."""
    session = await PrismaChatSession.prisma().find_unique(
        where={"id": session_id},
        include={"Messages": {"order_by": {"sequence": "asc"}}},
    )
    return ChatSession.from_db(session) if session else None


async def has_recent_manual_message(user_id: str, since: datetime) -> bool:
    message = await PrismaChatMessage.prisma().find_first(
        where={
            "role": "user",
            "createdAt": {"gte": since},
            "Session": {
                "is": {
                    "userId": user_id,
                    "startType": ChatSessionStartType.MANUAL.value,
                }
            },
        }
    )
    return message is not None


async def has_session_since(user_id: str, since: datetime) -> bool:
    session = await PrismaChatSession.prisma().find_first(
        where={"userId": user_id, "createdAt": {"gte": since}}
    )
    return session is not None


async def session_exists_for_execution_tag(user_id: str, execution_tag: str) -> bool:
    session = await PrismaChatSession.prisma().find_first(
        where={"userId": user_id, "executionTag": execution_tag}
    )
    return session is not None


async def get_manual_chat_sessions_since(
    user_id: str,
    since_utc: datetime,
    limit: int,
) -> list[ChatSessionInfo]:
    sessions = await PrismaChatSession.prisma().find_many(
        where={
            "userId": user_id,
            "startType": ChatSessionStartType.MANUAL.value,
            "updatedAt": {"gte": since_utc},
        },
        order={"updatedAt": "desc"},
        take=limit,
    )
    return [ChatSessionInfo.from_db(session) for session in sessions]


async def get_chat_messages_since(
    session_id: str,
    since_utc: datetime,
) -> list[ChatMessage]:
    messages = await PrismaChatMessage.prisma().find_many(
        where={
            "sessionId": session_id,
            "createdAt": {"gte": since_utc},
        },
        order={"sequence": "asc"},
    )
    return [ChatMessage.from_db(message) for message in messages]


def _build_chat_message_create_input(
    *,
    session_id: str,
    sequence: int,
    now: datetime,
    msg: dict[str, Any],
) -> ChatMessageCreateInput:
    data: ChatMessageCreateInput = {
        "sessionId": session_id,
        "role": msg["role"],
        "sequence": sequence,
        "createdAt": now,
    }

    if msg.get("content") is not None:
        data["content"] = sanitize_string(msg["content"])
    if msg.get("name") is not None:
        data["name"] = msg["name"]
    if msg.get("tool_call_id") is not None:
        data["toolCallId"] = msg["tool_call_id"]
    if msg.get("refusal") is not None:
        data["refusal"] = sanitize_string(msg["refusal"])
    if msg.get("tool_calls") is not None:
        data["toolCalls"] = SafeJson(msg["tool_calls"])
    if msg.get("function_call") is not None:
        data["functionCall"] = SafeJson(msg["function_call"])

    return data


async def create_chat_session(
    session_id: str,
    user_id: str,
    start_type: ChatSessionStartType = ChatSessionStartType.MANUAL,
    execution_tag: str | None = None,
    session_config: dict[str, Any] | None = None,
) -> ChatSessionInfo:
    """Create a new chat session in the database."""
    data = ChatSessionCreateInput(
        id=session_id,
        userId=user_id,
        credentials=SafeJson({}),
        successfulAgentRuns=SafeJson({}),
        successfulAgentSchedules=SafeJson({}),
        startType=start_type.value,
        executionTag=execution_tag,
        sessionConfig=SafeJson(session_config or {}),
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
    start_type: ChatSessionStartType | None = None,
    execution_tag: str | None | object = _UNSET,
    session_config: dict[str, Any] | None = None,
    completion_report: dict[str, Any] | None | object = _UNSET,
    completion_report_repair_count: int | None = None,
    completion_report_repair_queued_at: datetime | None | object = _UNSET,
    completed_at: datetime | None | object = _UNSET,
    notification_email_sent_at: datetime | None | object = _UNSET,
    notification_email_skipped_at: datetime | None | object = _UNSET,
) -> ChatSession | None:
    """Update a chat session's metadata."""
    data: ChatSessionUpdateInput = {"updatedAt": datetime.now(UTC)}
    should_clear_completion_report = completion_report is None

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
    if start_type is not None:
        data["startType"] = start_type.value
    if execution_tag is not _UNSET:
        data["executionTag"] = execution_tag
    if session_config is not None:
        data["sessionConfig"] = SafeJson(session_config)
    if completion_report is not _UNSET and completion_report is not None:
        data["completionReport"] = SafeJson(completion_report)
    if completion_report_repair_count is not None:
        data["completionReportRepairCount"] = completion_report_repair_count
    if completion_report_repair_queued_at is not _UNSET:
        data["completionReportRepairQueuedAt"] = completion_report_repair_queued_at
    if completed_at is not _UNSET:
        data["completedAt"] = completed_at
    if notification_email_sent_at is not _UNSET:
        data["notificationEmailSentAt"] = notification_email_sent_at
    if notification_email_skipped_at is not _UNSET:
        data["notificationEmailSkippedAt"] = notification_email_skipped_at

    session = await PrismaChatSession.prisma().update(
        where={"id": session_id},
        data=data,
        include={"Messages": {"order_by": {"sequence": "asc"}}},
    )

    if should_clear_completion_report:
        await db.execute_raw_with_schema(
            'UPDATE {schema_prefix}"ChatSession" SET "completionReport" = NULL WHERE "id" = $1',
            session_id,
        )
        session = await PrismaChatSession.prisma().find_unique(
            where={"id": session_id},
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
                messages_data = [
                    _build_chat_message_create_input(
                        session_id=session_id,
                        sequence=start_sequence + i,
                        now=now,
                        msg=msg,
                    )
                    for i, msg in enumerate(messages)
                ]

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
    with_auto: bool = False,
) -> list[ChatSessionInfo]:
    """Get chat sessions for a user, ordered by most recent."""
    prisma_sessions = await PrismaChatSession.prisma().find_many(
        where={
            "userId": user_id,
            **({} if with_auto else {"startType": ChatSessionStartType.MANUAL.value}),
        },
        order={"updatedAt": "desc"},
        take=limit,
        skip=offset,
    )
    return [ChatSessionInfo.from_db(s) for s in prisma_sessions]


async def get_pending_notification_chat_sessions(
    limit: int = 200,
) -> list[ChatSessionInfo]:
    sessions = await PrismaChatSession.prisma().find_many(
        where={
            "startType": {"not": ChatSessionStartType.MANUAL.value},
            "notificationEmailSentAt": None,
            "notificationEmailSkippedAt": None,
        },
        order={"updatedAt": "asc"},
        take=limit,
    )
    return [ChatSessionInfo.from_db(session) for session in sessions]


async def get_recent_sent_email_chat_sessions(
    user_id: str,
    limit: int,
) -> list[ChatSessionInfo]:
    sessions = await PrismaChatSession.prisma().find_many(
        where={
            "userId": user_id,
            "startType": {"not": ChatSessionStartType.MANUAL.value},
            "notificationEmailSentAt": {"not": None},
        },
        order={"notificationEmailSentAt": "desc"},
        take=max(limit * 3, limit),
    )
    return [
        session_info
        for session_info in (ChatSessionInfo.from_db(session) for session in sessions)
        if session_info.notification_email_sent_at and session_info.completion_report
    ][:limit]


async def get_recent_completion_report_chat_sessions(
    user_id: str,
    limit: int,
) -> list[ChatSessionInfo]:
    sessions = await PrismaChatSession.prisma().find_many(
        where={
            "userId": user_id,
            "startType": {"not": ChatSessionStartType.MANUAL.value},
        },
        order={"updatedAt": "desc"},
        take=max(limit * 5, 10),
    )
    return [
        session_info
        for session_info in (ChatSessionInfo.from_db(session) for session in sessions)
        if session_info.completion_report is not None
    ][:limit]


async def get_user_session_count(
    user_id: str,
    with_auto: bool = False,
) -> int:
    """Get the total number of chat sessions for a user."""
    return await PrismaChatSession.prisma().count(
        where={
            "userId": user_id,
            **({} if with_auto else {"startType": ChatSessionStartType.MANUAL.value}),
        }
    )


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


async def create_chat_session_callback_token(
    user_id: str,
    source_session_id: str,
    callback_session_message: str,
    expires_at: datetime,
) -> ChatSessionCallbackTokenInfo:
    token = await PrismaChatSessionCallbackToken.prisma().create(
        data={
            "userId": user_id,
            "sourceSessionId": source_session_id,
            "callbackSessionMessage": callback_session_message,
            "expiresAt": expires_at,
        }
    )
    return ChatSessionCallbackTokenInfo.from_db(token)


async def get_chat_session_callback_token(
    token_id: str,
) -> ChatSessionCallbackTokenInfo | None:
    token = await PrismaChatSessionCallbackToken.prisma().find_unique(
        where={"id": token_id}
    )
    return ChatSessionCallbackTokenInfo.from_db(token) if token else None


async def mark_chat_session_callback_token_consumed(
    token_id: str,
    consumed_session_id: str,
) -> None:
    await PrismaChatSessionCallbackToken.prisma().update(
        where={"id": token_id},
        data={
            "consumedAt": datetime.now(UTC),
            "consumedSessionId": consumed_session_id,
        },
    )
