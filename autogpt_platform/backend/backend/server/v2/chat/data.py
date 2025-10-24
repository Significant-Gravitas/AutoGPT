import json
from pydantic import BaseModel

from backend.util.cache import async_redis
from backend.server.v2.chat.config import ChatConfig
from backend.util.exceptions import NotFoundError

config = ChatConfig()


class ChatMessage(BaseModel):
    role: str
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    refusal: str | None = None
    tool_calls: list[dict] | None = None
    function_call: dict | None = None

class ChatSession(BaseModel):
    session_id: str
    user_id: str | None
    messages: list[ChatMessage]
    
async def get_chat_session(
    session_id: str,
    user_id: str,
) -> ChatSession:
    """Get a chat session by ID."""
    redis_key = f"chat:session:{session_id}"

        
    raw_session: bytes = await async_redis.get(
        redis_key
    )
    
    if not raw_session:
        raise NotFoundError(f"Chat session not found session_id: {session_id}")
    
    return ChatSession.model_validate_json(raw_session)


async def upsert_chat_session(
    session_id: str,
    user_id: str,
    messages: list[ChatMessage],
) -> ChatSession:
    """Update a chat session with the given messages."""
    session = ChatSession(
        session_id=session_id,
        user_id=user_id,
        messages=messages,
    )
    
    redis_key = f"chat:session:{session_id}"
    
    resp = await async_redis.setex(
        redis_key,
        config.session_ttl,
        session.model_dump_json()
    )
    
    if resp != True:
        raise Exception(f"Failed to update chat session: {resp}")
    
    return session


async def test_chatsession_serialization_deserialization():
    # Example messages using correct types
    messages = [
        ChatMessage(
            content="Hello, how are you?",
            role="user"
        ),
        ChatMessage(
            content="I'm fine, thank you!",
            role="assistant",
            tool_calls=[{
                "id": "t123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": "{\"city\": \"New York\"}"
                }
            }]
        ),
        ChatMessage(
            content="I'm using the tool to get the weather",
            role="tool",
            tool_call_id="t123"
        ),
    ]

    s = await upsert_chat_session(
        session_id="s1",
        user_id="u123",
        messages=messages,
    )
    
    s2 = await get_chat_session(
        session_id="s1",
        user_id="u123",
    )
    
    assert s2 == s

    
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_chatsession_serialization_deserialization())
    print("ChatSession serialization/deserialization test passed.")