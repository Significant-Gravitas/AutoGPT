import logging
import uuid
from datetime import UTC, datetime

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

from backend.data.redis_client import get_redis_async
from backend.util import json
from backend.util.exceptions import RedisError

from . import db as chat_db
from .config import ChatConfig

logger = logging.getLogger(__name__)
config = ChatConfig()


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    refusal: str | None = None
    tool_calls: list[dict] | None = None
    function_call: dict | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatSession(BaseModel):
    session_id: str
    user_id: str | None
    title: str | None = None
    messages: list[ChatMessage]
    usage: list[Usage]
    credentials: dict[str, dict] = {}  # Map of provider -> credential metadata
    started_at: datetime
    updated_at: datetime
    successful_agent_runs: dict[str, int] = {}
    successful_agent_schedules: dict[str, int] = {}

    @staticmethod
    def new(user_id: str | None) -> "ChatSession":
        return ChatSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            title=None,
            messages=[],
            usage=[],
            credentials={},
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    @staticmethod
    def from_prisma(
        prisma_session: PrismaChatSession,
        prisma_messages: list[PrismaChatMessage] | None = None,
    ) -> "ChatSession":
        """Convert Prisma models to Pydantic ChatSession."""
        messages = []
        if prisma_messages:
            for msg in prisma_messages:
                tool_calls = None
                if msg.toolCalls:
                    tool_calls = (
                        json.loads(msg.toolCalls)
                        if isinstance(msg.toolCalls, str)
                        else msg.toolCalls
                    )

                function_call = None
                if msg.functionCall:
                    function_call = (
                        json.loads(msg.functionCall)
                        if isinstance(msg.functionCall, str)
                        else msg.functionCall
                    )

                messages.append(
                    ChatMessage(
                        role=msg.role,
                        content=msg.content,
                        name=msg.name,
                        tool_call_id=msg.toolCallId,
                        refusal=msg.refusal,
                        tool_calls=tool_calls,
                        function_call=function_call,
                    )
                )

        # Parse JSON fields from Prisma
        credentials = (
            json.loads(prisma_session.credentials)
            if isinstance(prisma_session.credentials, str)
            else prisma_session.credentials or {}
        )
        successful_agent_runs = (
            json.loads(prisma_session.successfulAgentRuns)
            if isinstance(prisma_session.successfulAgentRuns, str)
            else prisma_session.successfulAgentRuns or {}
        )
        successful_agent_schedules = (
            json.loads(prisma_session.successfulAgentSchedules)
            if isinstance(prisma_session.successfulAgentSchedules, str)
            else prisma_session.successfulAgentSchedules or {}
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

        return ChatSession(
            session_id=prisma_session.id,
            user_id=prisma_session.userId,
            title=prisma_session.title,
            messages=messages,
            usage=usage,
            credentials=credentials,
            started_at=prisma_session.createdAt,
            updated_at=prisma_session.updatedAt,
            successful_agent_runs=successful_agent_runs,
            successful_agent_schedules=successful_agent_schedules,
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
        return messages


async def _get_session_from_cache(session_id: str) -> ChatSession | None:
    """Get a chat session from Redis cache."""
    redis_key = f"chat:session:{session_id}"
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


async def _cache_session(session: ChatSession) -> None:
    """Cache a chat session in Redis."""
    redis_key = f"chat:session:{session.session_id}"
    async_redis = await get_redis_async()
    await async_redis.setex(redis_key, config.session_ttl, session.model_dump_json())


async def _get_session_from_db(session_id: str) -> ChatSession | None:
    """Get a chat session from the database."""
    prisma_session = await chat_db.get_chat_session(session_id)
    if not prisma_session:
        return None

    messages = prisma_session.Messages
    logger.info(
        f"Loading session {session_id} from DB: "
        f"has_messages={messages is not None}, "
        f"message_count={len(messages) if messages else 0}, "
        f"roles={[m.role for m in messages] if messages else []}"
    )

    return ChatSession.from_prisma(prisma_session, messages)


async def _save_session_to_db(
    session: ChatSession, existing_message_count: int
) -> None:
    """Save or update a chat session in the database."""
    # Check if session exists in DB
    existing = await chat_db.get_chat_session(session.session_id)

    if not existing:
        # Create new session
        await chat_db.create_chat_session(
            session_id=session.session_id,
            user_id=session.user_id,
        )
        existing_message_count = 0

    # Calculate total tokens from usage
    total_prompt = sum(u.prompt_tokens for u in session.usage)
    total_completion = sum(u.completion_tokens for u in session.usage)

    # Update session metadata
    await chat_db.update_chat_session(
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
        await chat_db.add_chat_messages_batch(
            session_id=session.session_id,
            messages=messages_data,
            start_sequence=existing_message_count,
        )


async def get_chat_session(
    session_id: str,
    user_id: str | None,
) -> ChatSession | None:
    """Get a chat session by ID.

    Checks Redis cache first, falls back to database if not found.
    Caches database results back to Redis.
    """
    # Try cache first
    try:
        session = await _get_session_from_cache(session_id)
        if session:
            # Verify user ownership
            if session.user_id is not None and session.user_id != user_id:
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
    logger.info(f"Session {session_id} not in cache, checking database")
    session = await _get_session_from_db(session_id)

    if session is None:
        logger.warning(f"Session {session_id} not found in cache or database")
        return None

    # Verify user ownership
    if session.user_id is not None and session.user_id != user_id:
        logger.warning(
            f"Session {session_id} user id mismatch: {session.user_id} != {user_id}"
        )
        return None

    # Cache the session from DB
    try:
        await _cache_session(session)
        logger.info(f"Cached session {session_id} from database")
    except Exception as e:
        logger.warning(f"Failed to cache session {session_id}: {e}")

    return session


async def upsert_chat_session(
    session: ChatSession,
) -> ChatSession:
    """Update a chat session in both cache and database."""
    # Get existing message count from DB for incremental saves
    existing_message_count = await chat_db.get_chat_session_message_count(
        session.session_id
    )

    # Save to database
    try:
        await _save_session_to_db(session, existing_message_count)
    except Exception as e:
        logger.error(f"Failed to save session {session.session_id} to database: {e}")
        # Continue to cache even if DB fails

    # Save to cache
    try:
        await _cache_session(session)
    except Exception as e:
        raise RedisError(
            f"Failed to persist chat session {session.session_id} to Redis: {e}"
        ) from e

    return session


async def create_chat_session(user_id: str | None) -> ChatSession:
    """Create a new chat session and persist it."""
    session = ChatSession.new(user_id)

    # Create in database first
    try:
        await chat_db.create_chat_session(
            session_id=session.session_id,
            user_id=user_id,
        )
    except Exception as e:
        logger.error(f"Failed to create session in database: {e}")
        # Continue even if DB fails - cache will still work

    # Cache the session
    try:
        await _cache_session(session)
    except Exception as e:
        logger.warning(f"Failed to cache new session: {e}")

    return session


async def get_user_sessions(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[ChatSession]:
    """Get all chat sessions for a user from the database."""
    prisma_sessions = await chat_db.get_user_chat_sessions(user_id, limit, offset)

    sessions = []
    for prisma_session in prisma_sessions:
        # Convert without messages for listing (lighter weight)
        sessions.append(ChatSession.from_prisma(prisma_session, None))

    return sessions


async def delete_chat_session(session_id: str) -> bool:
    """Delete a chat session from both cache and database."""
    # Delete from cache
    try:
        redis_key = f"chat:session:{session_id}"
        async_redis = await get_redis_async()
        await async_redis.delete(redis_key)
    except Exception as e:
        logger.warning(f"Failed to delete session {session_id} from cache: {e}")

    # Delete from database
    return await chat_db.delete_chat_session(session_id)
