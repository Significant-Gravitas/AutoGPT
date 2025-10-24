from backend.server.v2.chat.data import ChatMessage, ChatSession, get_chat_session, upsert_chat_session
import pytest

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
    s = ChatSession(
        session_id="s1",
        user_id="u123",
        messages=messages,
    )
    serialized = s.model_dump_json()
    s2 = ChatSession.model_validate_json(serialized)
    assert s2.model_dump() == s.model_dump()


@pytest.mark.asyncio
async def test_chatsession_redis_storage():
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
    