from backend.server.v2.chat.data import ChatMessage, ChatSession, get_chat_session, upsert_chat_session
import pytest
from datetime import datetime, UTC
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



@pytest.mark.asyncio
async def test_chatsession_serialization_deserialization():
    s = ChatSession.new(user_id="abc123")
    s.messages = messages
    serialized = s.model_dump_json()
    s2 = ChatSession.model_validate_json(serialized)
    assert s2.model_dump() == s.model_dump()


@pytest.mark.asyncio
async def test_chatsession_redis_storage():
    
    s = ChatSession.new(user_id=None)
    s.messages = messages
    
    s = await upsert_chat_session(s)
    
    s2 = await get_chat_session(
        session_id=s.session_id,
        user_id=s.user_id,
    )
    
    assert s2 == s
    
@pytest.mark.asyncio
async def test_chatsession_redis_storage_user_id_mismatch():
    
    s = ChatSession.new(user_id="abc123")
    s.messages = messages
    s = await upsert_chat_session(s)
    
    s2 = await get_chat_session(s.session_id, None)
    
    assert s2 is None