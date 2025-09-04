"""Tests for chat database operations."""

from datetime import datetime

import prisma.enums
import prisma.errors
import prisma.models
import prisma.types
import pytest
from prisma.enums import ChatMessageRole

from backend.server.v2.chat import db
from backend.util.exceptions import NotFoundError


@pytest.mark.asyncio
async def test_create_chat_session(mocker) -> None:
    """Test creating a new chat session."""
    # Mock data
    mock_session = prisma.models.ChatSession(
        id="session123",
        userId="test-user",
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
    )

    # Mock prisma call
    mock_chat_session = mocker.patch("prisma.models.ChatSession.prisma")
    mock_chat_session.return_value.create = mocker.AsyncMock(return_value=mock_session)

    # Call function
    result = await db.create_chat_session("test-user")

    # Verify results
    assert result.id == "session123"
    assert result.userId == "test-user"

    # Verify the create was called with correct data
    mock_chat_session.return_value.create.assert_called_once_with(
        data={"userId": "test-user"},
    )


@pytest.mark.asyncio
async def test_get_chat_session(mocker) -> None:
    """Test getting a chat session by ID."""
    # Mock data
    mock_session = prisma.models.ChatSession(
        id="session123",
        userId="test-user",
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
    )

    # Mock prisma call
    mock_chat_session = mocker.patch("prisma.models.ChatSession.prisma")
    mock_chat_session.return_value.find_first = mocker.AsyncMock(
        return_value=mock_session,
    )

    # Call function
    result = await db.get_chat_session("session123", user_id="test-user")

    # Verify results
    assert result.id == "session123"
    assert result.userId == "test-user"

    # Verify the find_first was called with correct parameters
    mock_chat_session.return_value.find_first.assert_called_once_with(
        where={"id": "session123", "userId": "test-user"},
        include=None,
    )


@pytest.mark.asyncio
async def test_get_chat_session_not_found(mocker) -> None:
    """Test getting a non-existent chat session."""
    # Mock prisma call to return None
    mock_chat_session = mocker.patch("prisma.models.ChatSession.prisma")
    mock_chat_session.return_value.find_first = mocker.AsyncMock(return_value=None)

    # Call function and expect error
    with pytest.raises(NotFoundError) as exc_info:
        await db.get_chat_session("nonexistent", user_id="test-user")

    assert "Chat session nonexistent not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_chat_session_with_messages(mocker) -> None:
    """Test getting a chat session with messages included."""
    # Mock data
    mock_messages = [
        prisma.models.ChatMessage(
            id="msg1",
            sessionId="session123",
            content="Hello",
            role=ChatMessageRole.USER,
            sequence=0,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        ),
    ]

    mock_session = prisma.models.ChatSession(
        id="session123",
        userId="test-user",
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
        messages=mock_messages,
    )

    # Mock prisma call
    mock_chat_session = mocker.patch("prisma.models.ChatSession.prisma")
    mock_chat_session.return_value.find_first = mocker.AsyncMock(
        return_value=mock_session,
    )

    # Call function
    result = await db.get_chat_session("session123", include_messages=True)

    # Verify results
    assert result.id == "session123"
    assert result.messages is not None
    assert len(result.messages) == 1
    assert result.messages[0].content == "Hello"

    # Verify the find_first was called with include parameter
    mock_chat_session.return_value.find_first.assert_called_once_with(
        where={"id": "session123"},
        include={"messages": True},
    )


@pytest.mark.asyncio
async def test_list_chat_sessions(mocker) -> None:
    """Test listing chat sessions for a user."""
    # Mock data
    mock_sessions = [
        prisma.models.ChatSession(
            id="session1",
            userId="test-user",
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        ),
        prisma.models.ChatSession(
            id="session2",
            userId="test-user",
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        ),
    ]

    # Mock prisma call
    mock_chat_session = mocker.patch("prisma.models.ChatSession.prisma")
    mock_chat_session.return_value.find_many = mocker.AsyncMock(
        return_value=mock_sessions,
    )

    # Call function
    result = await db.list_chat_sessions("test-user", limit=10, offset=0)

    # Verify results
    assert len(result) == 2
    assert result[0].id == "session1"
    assert result[1].id == "session2"

    # Verify the find_many was called with correct parameters
    mock_chat_session.return_value.find_many.assert_called_once_with(
        where={"userId": "test-user"},
        include=None,
        order_by={"updatedAt": "desc"},
        skip=0,
        take=10,
    )


@pytest.mark.asyncio
async def test_list_chat_sessions_with_last_message(mocker) -> None:
    """Test listing chat sessions with the last message included."""
    # Mock data
    mock_sessions = [
        prisma.models.ChatSession(
            id="session1",
            userId="test-user",
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            messages=[
                prisma.models.ChatMessage(
                    id="msg1",
                    sessionId="session1",
                    content="Last message",
                    role=ChatMessageRole.ASSISTANT,
                    sequence=5,
                    createdAt=datetime.now(),
                    updatedAt=datetime.now(),
                ),
            ],
        ),
    ]

    # Mock prisma call
    mock_chat_session = mocker.patch("prisma.models.ChatSession.prisma")
    mock_chat_session.return_value.find_many = mocker.AsyncMock(
        return_value=mock_sessions,
    )

    # Call function
    result = await db.list_chat_sessions("test-user", include_last_message=True)

    # Verify results
    assert len(result) == 1
    assert result[0].messages is not None
    assert len(result[0].messages) == 1
    assert result[0].messages[0].content == "Last message"

    # Verify the find_many was called with include parameter
    expected_include = {"messages": {"take": 1, "order_by": {"sequence": "desc"}}}
    mock_chat_session.return_value.find_many.assert_called_once_with(
        where={"userId": "test-user"},
        include=expected_include,
        order_by={"updatedAt": "desc"},
        skip=0,
        take=50,
    )


@pytest.mark.asyncio
async def test_create_chat_message(mocker) -> None:
    """Test creating a new chat message."""
    # Mock data
    mock_message = prisma.models.ChatMessage(
        id="msg123",
        sessionId="session123",
        content="Test message",
        role=ChatMessageRole.USER,
        sequence=0,
        toolCallId=None,
        toolCalls=None,
        parentId=None,
        metadata=prisma.Json({}),
        promptTokens=10,
        completionTokens=20,
        totalTokens=30,
        error=None,
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
    )

    # Mock prisma calls
    mock_chat_message = mocker.patch("prisma.models.ChatMessage.prisma")
    mock_chat_message.return_value.find_first = mocker.AsyncMock(
        return_value=None,
    )  # No existing messages
    mock_chat_message.return_value.create = mocker.AsyncMock(return_value=mock_message)

    mock_chat_session = mocker.patch("prisma.models.ChatSession.prisma")
    mock_chat_session.return_value.update = mocker.AsyncMock()

    # Call function
    result = await db.create_chat_message(
        session_id="session123",
        content="Test message",
        role=ChatMessageRole.USER,
        prompt_tokens=10,
        completion_tokens=20,
    )

    # Verify results
    assert result.id == "msg123"
    assert result.content == "Test message"
    assert result.role == ChatMessageRole.USER
    assert result.sequence == 0
    assert result.totalTokens == 30

    # Verify the create was called with correct data
    mock_chat_message.return_value.create.assert_called_once_with(
        data={
            "sessionId": "session123",
            "content": "Test message",
            "role": ChatMessageRole.USER,
            "sequence": 0,
            "toolCallId": None,
            "toolCalls": None,
            "parentId": None,
            "metadata": {},
            "promptTokens": 10,
            "completionTokens": 20,
            "totalTokens": 30,
            "error": None,
        },
    )

    # Verify session was updated
    mock_chat_session.return_value.update.assert_called_once_with(
        where={"id": "session123"},
        data={},
    )


@pytest.mark.asyncio
async def test_create_chat_message_with_auto_sequence(mocker) -> None:
    """Test creating a chat message with auto-incremented sequence."""
    # Mock existing message
    mock_last_message = prisma.models.ChatMessage(
        id="msg1",
        sessionId="session123",
        content="Previous message",
        role=ChatMessageRole.USER,
        sequence=5,
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
    )

    mock_new_message = prisma.models.ChatMessage(
        id="msg2",
        sessionId="session123",
        content="New message",
        role=ChatMessageRole.ASSISTANT,
        sequence=6,
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
        metadata=prisma.Json({}),
        totalTokens=None,
    )

    # Mock prisma calls
    mock_chat_message = mocker.patch("prisma.models.ChatMessage.prisma")
    mock_chat_message.return_value.find_first = mocker.AsyncMock(
        return_value=mock_last_message,
    )
    mock_chat_message.return_value.create = mocker.AsyncMock(
        return_value=mock_new_message,
    )

    mock_chat_session = mocker.patch("prisma.models.ChatSession.prisma")
    mock_chat_session.return_value.update = mocker.AsyncMock()

    # Call function without sequence
    result = await db.create_chat_message(
        session_id="session123",
        content="New message",
        role=ChatMessageRole.ASSISTANT,
    )

    # Verify results
    assert result.sequence == 6

    # Verify the create was called with auto-incremented sequence
    create_call_args = mock_chat_message.return_value.create.call_args[1]["data"]
    assert create_call_args["sequence"] == 6


@pytest.mark.asyncio
async def test_create_chat_message_with_tool_calls(mocker) -> None:
    """Test creating a chat message with tool calls."""
    # Mock data
    tool_calls = [
        {
            "id": "call123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
            },
        },
    ]

    mock_message = prisma.models.ChatMessage(
        id="msg123",
        sessionId="session123",
        content="",
        role=ChatMessageRole.ASSISTANT,
        sequence=1,
        toolCallId=None,
        toolCalls=prisma.Json(tool_calls),  # type: ignore
        parentId=None,
        metadata=prisma.Json({}),
        createdAt=datetime.now(),
        updatedAt=datetime.now(),
        totalTokens=None,
    )

    # Mock prisma calls
    mock_chat_message = mocker.patch("prisma.models.ChatMessage.prisma")
    mock_chat_message.return_value.find_first = mocker.AsyncMock(return_value=None)
    mock_chat_message.return_value.create = mocker.AsyncMock(return_value=mock_message)

    mock_chat_session = mocker.patch("prisma.models.ChatSession.prisma")
    mock_chat_session.return_value.update = mocker.AsyncMock()

    # Call function
    result = await db.create_chat_message(
        session_id="session123",
        content="",
        role=ChatMessageRole.ASSISTANT,
        tool_calls=tool_calls,
    )

    # Verify results
    assert result.toolCalls == tool_calls

    # Verify the create was called with tool calls
    create_call_args = mock_chat_message.return_value.create.call_args[1]["data"]
    assert create_call_args["toolCalls"] == tool_calls


@pytest.mark.asyncio
async def test_get_chat_messages(mocker) -> None:
    """Test getting messages for a chat session."""
    # Mock data
    mock_messages = [
        prisma.models.ChatMessage(
            id="msg1",
            sessionId="session123",
            content="First message",
            role=ChatMessageRole.USER,
            sequence=0,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        ),
        prisma.models.ChatMessage(
            id="msg2",
            sessionId="session123",
            content="Second message",
            role=ChatMessageRole.ASSISTANT,
            sequence=1,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        ),
    ]

    # Mock prisma call
    mock_chat_message = mocker.patch("prisma.models.ChatMessage.prisma")
    mock_chat_message.return_value.find_many = mocker.AsyncMock(
        return_value=mock_messages,
    )

    # Call function
    result = await db.get_chat_messages("session123", limit=10)

    # Verify results
    assert len(result) == 2
    assert result[0].content == "First message"
    assert result[1].content == "Second message"
    assert result[0].sequence < result[1].sequence

    # Verify the find_many was called with correct parameters
    mock_chat_message.return_value.find_many.assert_called_once_with(
        where={"sessionId": "session123"},
        include=None,
        order_by={"sequence": "asc"},
        skip=0,
        take=10,
    )


@pytest.mark.asyncio
async def test_get_chat_messages_with_parent_filter(mocker) -> None:
    """Test getting messages filtered by parent ID."""
    # Mock data
    mock_messages = [
        prisma.models.ChatMessage(
            id="msg2",
            sessionId="session123",
            content="Child message",
            role=ChatMessageRole.ASSISTANT,
            sequence=2,
            parentId="msg1",
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        ),
    ]

    # Mock prisma call
    mock_chat_message = mocker.patch("prisma.models.ChatMessage.prisma")
    mock_chat_message.return_value.find_many = mocker.AsyncMock(
        return_value=mock_messages,
    )

    # Call function
    result = await db.get_chat_messages("session123", parent_id="msg1")

    # Verify results
    assert len(result) == 1
    assert result[0].parentId == "msg1"

    # Verify the find_many was called with parent filter
    mock_chat_message.return_value.find_many.assert_called_once_with(
        where={"sessionId": "session123", "parentId": "msg1"},
        include=None,
        order_by={"sequence": "asc"},
        skip=0,
        take=None,
    )


@pytest.mark.asyncio
async def test_get_conversation_context(mocker) -> None:
    """Test getting conversation context formatted for OpenAI API."""
    # Mock data
    mock_messages = [
        prisma.models.ChatMessage(
            id="msg1",
            sessionId="session123",
            content="You are a helpful assistant.",
            role=ChatMessageRole.SYSTEM,
            sequence=0,
            toolCallId=None,
            toolCalls=None,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        ),
        prisma.models.ChatMessage(
            id="msg2",
            sessionId="session123",
            content="Hello!",
            role=ChatMessageRole.USER,
            sequence=1,
            toolCallId=None,
            toolCalls=None,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        ),
        prisma.models.ChatMessage(
            id="msg3",
            sessionId="session123",
            content="",
            role=ChatMessageRole.ASSISTANT,
            sequence=2,
            toolCallId=None,
            toolCalls=prisma.Json(
                [
                    {
                        "id": "call123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "SF"}',
                        },
                    },
                ]
            ),  # type: ignore
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        ),
        prisma.models.ChatMessage(
            id="msg4",
            sessionId="session123",
            content="Sunny, 72°F",
            role=ChatMessageRole.TOOL,
            sequence=3,
            toolCallId="call123",
            toolCalls=None,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        ),
    ]

    # Mock get_chat_messages
    mocker.patch(
        "backend.server.v2.chat.db.get_chat_messages",
        return_value=mock_messages,
    )

    # Call function
    result = await db.get_conversation_context("session123")

    # Verify results
    assert len(result) == 4

    # Check system message
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are a helpful assistant."

    # Check user message
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "Hello!"

    # Check assistant message with tool calls
    assert result[2]["role"] == "assistant"
    assert result[2].get("content") == ""
    assert "tool_calls" in result[2]
    tool_calls = result[2]["tool_calls"]
    assert isinstance(tool_calls, list) and len(tool_calls) > 0
    assert tool_calls[0]["id"] == "call123"

    # Check tool response
    assert result[3]["role"] == "tool"
    assert result[3]["content"] == "Sunny, 72°F"
    assert result[3]["tool_call_id"] == "call123"


@pytest.mark.asyncio
async def test_get_conversation_context_without_system(mocker) -> None:
    """Test getting conversation context without system messages."""
    # Mock data
    mock_messages = [
        prisma.models.ChatMessage(
            id="msg1",
            sessionId="session123",
            content="System prompt",
            role=ChatMessageRole.SYSTEM,
            sequence=0,
            toolCallId=None,
            toolCalls=None,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        ),
        prisma.models.ChatMessage(
            id="msg2",
            sessionId="session123",
            content="User message",
            role=ChatMessageRole.USER,
            sequence=1,
            toolCallId=None,
            toolCalls=None,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
        ),
    ]

    # Mock get_chat_messages
    mocker.patch(
        "backend.server.v2.chat.db.get_chat_messages",
        return_value=mock_messages,
    )

    # Call function
    result = await db.get_conversation_context("session123", include_system=False)

    # Verify results - should only have user message
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "User message"
