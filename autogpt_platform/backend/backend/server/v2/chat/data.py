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
from pydantic import BaseModel

from backend.server.v2.chat.config import ChatConfig
from backend.util.cache import async_redis

logger = logging.getLogger(__name__)
config = ChatConfig()


class ChatMessage(BaseModel):
    role: str
    content: str
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
    messages: list[ChatMessage]
    usage: list[Usage]
    started_at: datetime
    updated_at: datetime

    @staticmethod
    def new(user_id: str | None) -> "ChatSession":
        return ChatSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            messages=[],
            usage=[],
            started_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )

    def to_openai_messages(self) -> list[ChatCompletionMessageParam]:
        messages = []
        for message in self.messages:
            if message.role == "developer":
                m = ChatCompletionDeveloperMessageParam(
                    role="developer",
                    content=message.content,
                )
                if message.name:
                    m["name"] = message.name
                messages.append(m)
            elif message.role == "system":
                m = ChatCompletionSystemMessageParam(
                    role="system",
                    content=message.content,
                )
                if message.name:
                    m["name"] = message.name
                messages.append(m)
            elif message.role == "user":
                m = ChatCompletionUserMessageParam(
                    role="user",
                    content=message.content,
                )
                if message.name:
                    m["name"] = message.name
                messages.append(m)
            elif message.role == "assistant":
                m = ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=message.content,
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
                        t.append(
                            ChatCompletionMessageToolCallParam(
                                id=tool_call["id"],
                                type="function",
                                function=Function(
                                    arguments=tool_call["arguments"],
                                    name=tool_call["name"],
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
                        content=message.content,
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


async def get_chat_session(
    session_id: str,
    user_id: str | None,
) -> ChatSession | None:
    """Get a chat session by ID."""
    redis_key = f"chat:session:{session_id}"

    raw_session: bytes = await async_redis.get(redis_key)

    if not raw_session:
        logger.warning(f"Session {session_id} not found")
        return None

    session = ChatSession.model_validate_json(raw_session)

    if session.user_id != user_id:
        logger.warning(
            f"Session {session_id} user id mismatch: {session.user_id} != {user_id}"
        )
        return None

    return session


async def upsert_chat_session(
    session: ChatSession,
) -> ChatSession:
    """Update a chat session with the given messages."""

    redis_key = f"chat:session:{session.session_id}"

    resp = await async_redis.setex(
        redis_key, config.session_ttl, session.model_dump_json()
    )

    if not resp:
        raise Exception(f"Failed to update chat session: {resp}")

    return session
