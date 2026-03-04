import asyncio
import logging
import uuid
from datetime import UTC, datetime
from typing import Any, Self, cast
from weakref import WeakValueDictionary

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import FunctionCall
from openai.types.chat.chat_completion_message_tool_call_param import (
    ChatCompletionMessageToolCallParam,
    Function,
)
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession
from pydantic import BaseModel

from backend.data.db_accessors import chat_db
from backend.data.redis_client import get_redis_async
from backend.util import json
from backend.util.exceptions import DatabaseError, RedisError

from .config import ChatConfig

logger = logging.getLogger(__name__)
config = ChatConfig()


# Redis cache key prefix for chat sessions
CHAT_SESSION_CACHE_PREFIX = "chat:session:"


def _get_session_cache_key(session_id: str) -> str:
    """Get the Redis cache key for a chat session."""
    return f"{CHAT_SESSION_CACHE_PREFIX}{session_id}"


# ===================== Chat data models ===================== #


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    refusal: str | None = None
    tool_calls: list[dict] | None = None
    function_call: dict | None = None

    @staticmethod
    def from_db(prisma_message: PrismaChatMessage) -> "ChatMessage":
        """Convert a Prisma ChatMessage to a Pydantic ChatMessage."""
        return ChatMessage(
            role=prisma_message.role,
            content=prisma_message.content,
            name=prisma_message.name,
            tool_call_id=prisma_message.toolCallId,
            refusal=prisma_message.refusal,
            tool_calls=_parse_json_field(prisma_message.toolCalls),
            function_call=_parse_json_field(prisma_message.functionCall),
        )


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatSessionInfo(BaseModel):
    session_id: str
    user_id: str
    title: str | None = None
    usage: list[Usage]
    credentials: dict[str, dict] = {}  # Map of provider -> credential metadata
    started_at: datetime
    updated_at: datetime
    successful_agent_runs: dict[str, int] = {}
    successful_agent_schedules: dict[str, int] = {}

    @classmethod
    def from_db(cls, prisma_session: PrismaChatSession) -> Self:
        """Convert Prisma ChatSession to Pydantic ChatSession."""
        # Parse JSON fields from Prisma
        credentials = _parse_json_field(prisma_session.credentials, default={})
        successful_agent_runs = _parse_json_field(
            prisma_session.successfulAgentRuns, default={}
        )
        successful_agent_schedules = _parse_json_field(
            prisma_session.successfulAgentSchedules, default={}
        )

        # Calculate usage from token counts
        usage = []
        if prisma_session.totalPromptTokens or prisma_session.totalCompletionTokens:
            usage.append(
                Usage(
                    prompt_tokens=prisma_session.totalPromptTokens or 0,
                    completion_tokens=prisma_session.totalCompletionTokens or 0,
                    total_tokens=(prisma_session.totalPromptTokens or 0)
                    + (prisma_session.totalCompletionTokens or 0),
                )
            )

        return cls(
            session_id=prisma_session.id,
            user_id=prisma_session.userId,
            title=prisma_session.title,
            usage=usage,
            credentials=credentials,
            started_at=prisma_session.createdAt,
            updated_at=prisma_session.updatedAt,
            successful_agent_runs=successful_agent_runs,
            successful_agent_schedules=successful_agent_schedules,
        )


class ChatSession(ChatSessionInfo):
    messages: list[ChatMessage]

    @classmethod
    def new(cls, user_id: str) -> Self:
        return cls(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            title=None,
            messages=[],
            usage=[],
            credentials={},
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    @classmethod
    def from_db(cls, prisma_session: PrismaChatSession) -> Self:
        """Convert Prisma ChatSession to Pydantic ChatSession."""
        if prisma_session.Messages is None:
            raise ValueError(
                f"Prisma session {prisma_session.id} is missing Messages relation"
            )

        return cls(
            **ChatSessionInfo.from_db(prisma_session).model_dump(),
            messages=[ChatMessage.from_db(m) for m in prisma_session.Messages],
        )

    def add_tool_call_to_current_turn(self, tool_call: dict) -> None:
        """Attach a tool_call to the current turn's assistant message.

        Searches backwards for the most recent assistant message (stopping at
        any user message boundary). If found, appends the tool_call to it.
        Otherwise creates a new assistant message with the tool_call.
        """
        for msg in reversed(self.messages):
            if msg.role == "user":
                break
            if msg.role == "assistant":
                if not msg.tool_calls:
                    msg.tool_calls = []
                msg.tool_calls.append(tool_call)
                return

        self.messages.append(
            ChatMessage(role="assistant", content="", tool_calls=[tool_call])
        )

    def to_openai_messages(self) -> list[ChatCompletionMessageParam]:
        messages = []
        for message in self.messages:
            if message.role == "developer":
                m = ChatCompletionDeveloperMessageParam(
                    role="developer",
                    content=message.content or "",
                )
                if message.name:
                    m["name"] = message.name
                messages.append(m)
            elif message.role == "system":
                m = ChatCompletionSystemMessageParam(
                    role="system",
                    content=message.content or "",
                )
                if message.name:
                    m["name"] = message.name
                messages.append(m)
            elif message.role == "user":
                m = ChatCompletionUserMessageParam(
                    role="user",
                    content=message.content or "",
                )
                if message.name:
                    m["name"] = message.name
                messages.append(m)
            elif message.role == "assistant":
                m = ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=message.content or "",
                )
                if message.function_call:
                    m["function_call"] = FunctionCall(
                        arguments=message.function_call["arguments"],
                        name=message.function_call["name"],
                    )
                if message.refusal:
                    m["refusal"] = message.refusal
                if message.tool_calls:
                    t: list[ChatCompletionMessageToolCallParam] = []
                    for tool_call in message.tool_calls:
                        # Tool calls are stored with nested structure: {id, type, function: {name, arguments}}
                        function_data = tool_call.get("function", {})

                        # Skip tool calls that are missing required fields
                        if "id" not in tool_call or "name" not in function_data:
                            logger.warning(
                                f"Skipping invalid tool call: missing required fields. "
                                f"Got: {tool_call.keys()}, function keys: {function_data.keys()}"
                            )
                            continue

                        # Arguments are stored as a JSON string
                        arguments_str = function_data.get("arguments", "{}")

                        t.append(
                            ChatCompletionMessageToolCallParam(
                                id=tool_call["id"],
                                type="function",
                                function=Function(
                                    arguments=arguments_str,
                                    name=function_data["name"],
                                ),
                            )
                        )
                    m["tool_calls"] = t
                if message.name:
                    m["name"] = message.name
                messages.append(m)
            elif message.role == "tool":
                messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        content=message.content or "",
                        tool_call_id=message.tool_call_id or "",
                    )
                )
            elif message.role == "function":
                messages.append(
                    ChatCompletionFunctionMessageParam(
                        role="function",
                        content=message.content,
                        name=message.name or "",
                    )
                )
        return self._merge_consecutive_assistant_messages(messages)

    @staticmethod
    def _merge_consecutive_assistant_messages(
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        """Merge consecutive assistant messages into single messages.

        Long-running tool flows can create split assistant messages: one with
        text content and another with tool_calls. Anthropic's API requires
        tool_result blocks to reference a tool_use in the immediately preceding
        assistant message, so these splits cause 400 errors via OpenRouter.
        """
        if len(messages) < 2:
            return messages

        result: list[ChatCompletionMessageParam] = [messages[0]]
        for msg in messages[1:]:
            prev = result[-1]
            if prev.get("role") != "assistant" or msg.get("role") != "assistant":
                result.append(msg)
                continue

            prev = cast(ChatCompletionAssistantMessageParam, prev)
            curr = cast(ChatCompletionAssistantMessageParam, msg)

            curr_content = curr.get("content") or ""
            if curr_content:
                prev_content = prev.get("content") or ""
                prev["content"] = (
                    f"{prev_content}\n{curr_content}" if prev_content else curr_content
                )

            curr_tool_calls = curr.get("tool_calls")
            if curr_tool_calls:
                prev_tool_calls = prev.get("tool_calls")
                prev["tool_calls"] = (
                    list(prev_tool_calls) + list(curr_tool_calls)
                    if prev_tool_calls
                    else list(curr_tool_calls)
                )
        return result


def _parse_json_field(value: str | dict | list | None, default: Any = None) -> Any:
    """Parse a JSON field that may be stored as string or already parsed."""
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


# ================ Chat cache + DB operations ================ #

# NOTE: Database calls are automatically routed through DatabaseManager if Prisma is not
#       connected directly.


async def cache_chat_session(session: ChatSession) -> None:
    """Cache a chat session in Redis (without persisting to the database)."""
    redis_key = _get_session_cache_key(session.session_id)
    async_redis = await get_redis_async()
    await async_redis.setex(redis_key, config.session_ttl, session.model_dump_json())


async def invalidate_session_cache(session_id: str) -> None:
    """Invalidate a chat session from Redis cache.

    Used by background tasks to ensure fresh data is loaded on next access.
    This is best-effort - Redis failures are logged but don't fail the operation.
    """
    try:
        redis_key = _get_session_cache_key(session_id)
        async_redis = await get_redis_async()
        await async_redis.delete(redis_key)
    except Exception as e:
        # Best-effort: log but don't fail - cache will expire naturally
        logger.warning(f"Failed to invalidate session cache for {session_id}: {e}")


async def get_chat_session(
    session_id: str,
    user_id: str | None = None,
) -> ChatSession | None:
    """Get a chat session by ID.

    Checks Redis cache first, falls back to database if not found.
    Caches database results back to Redis.

    Args:
        session_id: The session ID to fetch.
        user_id: If provided, validates that the session belongs to this user.
            If None, ownership is not validated (admin/system access).
    """
    # Try cache first
    try:
        session = await _get_session_from_cache(session_id)
        if session:
            # Verify user ownership if user_id was provided for validation
            if user_id is not None and session.user_id != user_id:
                logger.warning(
                    f"Session {session_id} user id mismatch: {session.user_id} != {user_id}"
                )
                return None
            return session
    except RedisError:
        logger.warning(f"Cache error for session {session_id}, trying database")
    except Exception as e:
        logger.warning(f"Unexpected cache error for session {session_id}: {e}")

    # Fall back to database
    logger.debug(f"Session {session_id} not in cache, checking database")
    session = await _get_session_from_db(session_id)

    if session is None:
        logger.warning(f"Session {session_id} not found in cache or database")
        return None

    # Verify user ownership if user_id was provided for validation
    if user_id is not None and session.user_id != user_id:
        logger.warning(
            f"Session {session_id} user id mismatch: {session.user_id} != {user_id}"
        )
        return None

    # Cache the session from DB
    try:
        await cache_chat_session(session)
        logger.info(f"Cached session {session_id} from database")
    except Exception as e:
        logger.warning(f"Failed to cache session {session_id}: {e}")

    return session


async def _get_session_from_cache(session_id: str) -> ChatSession | None:
    """Get a chat session from Redis cache."""
    redis_key = _get_session_cache_key(session_id)
    async_redis = await get_redis_async()
    raw_session: bytes | None = await async_redis.get(redis_key)

    if raw_session is None:
        return None

    try:
        session = ChatSession.model_validate_json(raw_session)
        logger.info(
            f"Loading session {session_id} from cache: "
            f"message_count={len(session.messages)}, "
            f"roles={[m.role for m in session.messages]}"
        )
        return session
    except Exception as e:
        logger.error(f"Failed to deserialize session {session_id}: {e}", exc_info=True)
        raise RedisError(f"Corrupted session data for {session_id}") from e


async def _get_session_from_db(session_id: str) -> ChatSession | None:
    """Get a chat session from the database."""
    session = await chat_db().get_chat_session(session_id)
    if not session:
        return None

    logger.info(
        f"Loaded session {session_id} from DB: "
        f"has_messages={bool(session.messages)}, "
        f"message_count={len(session.messages)}, "
        f"roles={[m.role for m in session.messages]}"
    )

    return session


async def upsert_chat_session(
    session: ChatSession,
) -> ChatSession:
    """Update a chat session in both cache and database.

    Uses session-level locking to prevent race conditions when concurrent
    operations (e.g., background title update and main stream handler)
    attempt to upsert the same session simultaneously.

    Raises:
        DatabaseError: If the database write fails. The cache is still updated
            as a best-effort optimization, but the error is propagated to ensure
            callers are aware of the persistence failure.
        RedisError: If the cache write fails (after successful DB write).
    """
    # Acquire session-specific lock to prevent concurrent upserts
    lock = await _get_session_lock(session.session_id)

    async with lock:
        # Always query DB for existing message count to ensure consistency
        existing_message_count = await chat_db().get_next_sequence(session.session_id)

        db_error: Exception | None = None

        # Save to database (primary storage)
        try:
            await _save_session_to_db(
                session,
                existing_message_count,
                skip_existence_check=existing_message_count > 0,
            )
        except Exception as e:
            logger.error(
                f"Failed to save session {session.session_id} to database: {e}"
            )
            db_error = e

        # Save to cache (best-effort, even if DB failed)
        try:
            await cache_chat_session(session)
        except Exception as e:
            # If DB succeeded but cache failed, raise cache error
            if db_error is None:
                raise RedisError(
                    f"Failed to persist chat session {session.session_id} to Redis: {e}"
                ) from e
            # If both failed, log cache error but raise DB error (more critical)
            logger.warning(
                f"Cache write also failed for session {session.session_id}: {e}"
            )

        # Propagate DB error after attempting cache (prevents data loss)
        if db_error is not None:
            raise DatabaseError(
                f"Failed to persist chat session {session.session_id} to database"
            ) from db_error

        return session


async def _save_session_to_db(
    session: ChatSession,
    existing_message_count: int,
    *,
    skip_existence_check: bool = False,
) -> None:
    """Save or update a chat session in the database.

    Args:
        skip_existence_check: When True, skip the ``get_chat_session`` query
            and assume the session row already exists.  Saves one DB round trip
            for incremental saves during streaming.
    """
    db = chat_db()

    if not skip_existence_check:
        # Check if session exists in DB
        existing = await db.get_chat_session(session.session_id)

        if not existing:
            # Create new session
            await db.create_chat_session(
                session_id=session.session_id,
                user_id=session.user_id,
            )
            existing_message_count = 0

    # Calculate total tokens from usage
    total_prompt = sum(u.prompt_tokens for u in session.usage)
    total_completion = sum(u.completion_tokens for u in session.usage)

    # Update session metadata
    await db.update_chat_session(
        session_id=session.session_id,
        credentials=session.credentials,
        successful_agent_runs=session.successful_agent_runs,
        successful_agent_schedules=session.successful_agent_schedules,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
    )

    # Add new messages (only those after existing count)
    new_messages = session.messages[existing_message_count:]
    if new_messages:
        messages_data = []
        for msg in new_messages:
            messages_data.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    "name": msg.name,
                    "tool_call_id": msg.tool_call_id,
                    "refusal": msg.refusal,
                    "tool_calls": msg.tool_calls,
                    "function_call": msg.function_call,
                }
            )
        logger.info(
            f"Saving {len(new_messages)} new messages to DB for session {session.session_id}: "
            f"roles={[m['role'] for m in messages_data]}, "
            f"start_sequence={existing_message_count}"
        )
        await db.add_chat_messages_batch(
            session_id=session.session_id,
            messages=messages_data,
            start_sequence=existing_message_count,
        )


async def append_and_save_message(session_id: str, message: ChatMessage) -> ChatSession:
    """Atomically append a message to a session and persist it.

    Acquires the session lock, re-fetches the latest session state,
    appends the message, and saves — preventing message loss when
    concurrent requests modify the same session.
    """
    lock = await _get_session_lock(session_id)

    async with lock:
        session = await get_chat_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        session.messages.append(message)
        existing_message_count = await chat_db().get_next_sequence(session_id)

        try:
            await _save_session_to_db(session, existing_message_count)
        except Exception as e:
            raise DatabaseError(
                f"Failed to persist message to session {session_id}"
            ) from e

        try:
            await cache_chat_session(session)
        except Exception as e:
            logger.warning(f"Cache write failed for session {session_id}: {e}")

        return session


async def create_chat_session(user_id: str) -> ChatSession:
    """Create a new chat session and persist it.

    Raises:
        DatabaseError: If the database write fails. We fail fast to ensure
            callers never receive a non-persisted session that only exists
            in cache (which would be lost when the cache expires).
    """
    session = ChatSession.new(user_id)

    # Create in database first - fail fast if this fails
    try:
        await chat_db().create_chat_session(
            session_id=session.session_id,
            user_id=user_id,
        )
    except Exception as e:
        logger.error(f"Failed to create session {session.session_id} in database: {e}")
        raise DatabaseError(
            f"Failed to create chat session {session.session_id} in database"
        ) from e

    # Cache the session (best-effort optimization, DB is source of truth)
    try:
        await cache_chat_session(session)
    except Exception as e:
        logger.warning(f"Failed to cache new session {session.session_id}: {e}")

    return session


async def get_user_sessions(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[ChatSessionInfo], int]:
    """Get chat sessions for a user from the database with total count.

    Returns:
        A tuple of (sessions, total_count) where total_count is the overall
        number of sessions for the user (not just the current page).
    """
    db = chat_db()
    sessions = await db.get_user_chat_sessions(user_id, limit, offset)
    total_count = await db.get_user_session_count(user_id)

    return sessions, total_count


async def delete_chat_session(session_id: str, user_id: str | None = None) -> bool:
    """Delete a chat session from both cache and database.

    Args:
        session_id: The session ID to delete.
        user_id: If provided, validates that the session belongs to this user
            before deletion. This prevents unauthorized deletion.

    Returns:
        True if deleted successfully, False otherwise.
    """
    # Delete from database first (with optional user_id validation)
    # This confirms ownership before invalidating cache
    deleted = await chat_db().delete_chat_session(session_id, user_id)

    if not deleted:
        return False

    # Only invalidate cache and clean up lock after DB confirms deletion
    try:
        redis_key = _get_session_cache_key(session_id)
        async_redis = await get_redis_async()
        await async_redis.delete(redis_key)
    except Exception as e:
        logger.warning(f"Failed to delete session {session_id} from cache: {e}")

    # Clean up session lock (belt-and-suspenders with WeakValueDictionary)
    async with _session_locks_mutex:
        _session_locks.pop(session_id, None)

    return True


async def update_session_title(session_id: str, title: str) -> bool:
    """Update only the title of a chat session.

    This is a lightweight operation that doesn't touch messages, avoiding
    race conditions with concurrent message updates. Use this for background
    title generation instead of upsert_chat_session.

    Args:
        session_id: The session ID to update.
        title: The new title to set.

    Returns:
        True if updated successfully, False otherwise.
    """
    try:
        result = await chat_db().update_chat_session(session_id=session_id, title=title)
        if result is None:
            logger.warning(f"Session {session_id} not found for title update")
            return False

        # Update title in cache if it exists (instead of invalidating).
        # This prevents race conditions where cache invalidation causes
        # the frontend to see stale DB data while streaming is still in progress.
        try:
            cached = await _get_session_from_cache(session_id)
            if cached:
                cached.title = title
                await cache_chat_session(cached)
        except Exception as e:
            # Not critical - title will be correct on next full cache refresh
            logger.warning(
                f"Failed to update title in cache for session {session_id}: {e}"
            )

        return True
    except Exception as e:
        logger.error(f"Failed to update title for session {session_id}: {e}")
        return False


# ==================== Chat session locks ==================== #

_session_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()
_session_locks_mutex = asyncio.Lock()


async def _get_session_lock(session_id: str) -> asyncio.Lock:
    """Get or create a lock for a specific session to prevent concurrent upserts.

    This was originally added to solve the specific problem of race conditions between
    the session title thread and the conversation thread, which always occurs on the
    same instance as we prevent rapid request sends on the frontend.

    Uses WeakValueDictionary for automatic cleanup: locks are garbage collected
    when no coroutine holds a reference to them, preventing memory leaks from
    unbounded growth of session locks. Explicit cleanup also occurs
    in `delete_chat_session()`.
    """
    async with _session_locks_mutex:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            _session_locks[session_id] = lock
        return lock
