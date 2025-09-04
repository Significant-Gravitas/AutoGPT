"""Unit tests for the chat streaming functions."""

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ChatMessageRole

from backend.server.v2.chat.chat import (
    get_openai_client,
    stream_chat_completion,
    stream_chat_response,
)
from backend.server.v2.chat.tool_exports import execute_tool

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletionMessageParam


class MockDelta:
    """Mock OpenAI delta object."""

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    """Mock OpenAI choice object."""

    def __init__(
        self,
        delta: MockDelta | None = None,
        finish_reason: str | None = None,
    ) -> None:
        self.delta = delta or MockDelta()
        self.finish_reason = finish_reason


class MockChunk:
    """Mock OpenAI stream chunk."""

    def __init__(self, choices: list[MockChoice] | None = None) -> None:
        self.choices = choices or []


class MockToolCall:
    """Mock tool call object."""

    def __init__(
        self,
        index: int,
        id: str | None = None,
        function: Any | None = None,
    ) -> None:
        self.index = index
        self.id = id
        self.function = function


class MockFunction:
    """Mock function object for tool calls."""

    def __init__(self, name: str | None = None, arguments: str | None = None) -> None:
        self.name = name
        self.arguments = arguments


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("backend.server.v2.chat.chat.AsyncOpenAI") as mock_client:
        instance = MagicMock()
        mock_client.return_value = instance
        yield instance


@pytest.fixture
def mock_db():
    """Create mock database module."""
    with patch("backend.server.v2.chat.chat.db") as mock_db:
        mock_db.create_chat_message = AsyncMock(return_value={"id": "msg-123"})
        mock_db.get_conversation_context = AsyncMock(return_value=[])
        yield mock_db


@pytest.fixture
def mock_config():
    """Create mock config."""
    with patch("backend.server.v2.chat.chat.get_config") as mock_get_config:
        mock_config = MagicMock()
        mock_config.model = "gpt-4o"
        mock_config.api_key = "test-key"
        mock_config.base_url = None
        mock_config.cache_client = True
        mock_config.get_system_prompt.return_value = "You are a helpful assistant."
        mock_get_config.return_value = mock_config
        yield mock_config


@pytest.fixture
def mock_tools():
    """Create mock tools module."""
    # Mock the tools list directly since it's imported as a list
    tools_list = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "Test tool",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    with patch("backend.server.v2.chat.chat.tools", tools_list):
        yield


class TestGetOpenAIClient:
    """Test cases for get_openai_client function."""

    def test_get_client_with_cache(self, mock_config) -> None:
        """Test getting cached client."""
        # Reset global cache
        import backend.server.v2.chat.chat as chat_module

        chat_module._client_cache = None

        with patch("backend.server.v2.chat.chat.AsyncOpenAI") as mock_client:
            instance = MagicMock()
            mock_client.return_value = instance

            # First call creates client
            client1 = get_openai_client()
            assert client1 == instance
            mock_client.assert_called_once()

            # Second call returns cached client
            client2 = get_openai_client()
            assert client2 == instance
            assert mock_client.call_count == 1  # Still only called once

    def test_get_client_force_new(self, mock_config) -> None:
        """Test forcing new client creation."""
        # Reset global cache
        import backend.server.v2.chat.chat as chat_module

        chat_module._client_cache = None

        with patch("backend.server.v2.chat.chat.AsyncOpenAI") as mock_client:
            instance = mock_client.return_value

            # First call
            client1 = get_openai_client()
            assert client1 == instance
            assert mock_client.call_count == 1

            # Force new client (creates a new one despite cache)
            client2 = get_openai_client(force_new=True)
            assert client2 == instance  # Same mock instance returned
            assert mock_client.call_count == 2  # But constructor called twice

    def test_get_client_with_config(self, mock_config) -> None:
        """Test client creation with config settings."""
        mock_config.api_key = "custom-key"
        mock_config.base_url = "https://custom.api/v1"

        with patch("backend.server.v2.chat.chat.AsyncOpenAI") as mock_client:
            with patch("backend.server.v2.chat.chat._client_cache", None):
                get_openai_client()
                mock_client.assert_called_once_with(
                    api_key="custom-key",
                    base_url="https://custom.api/v1",
                )


class TestStreamChatResponse:
    """Test cases for stream_chat_response function."""

    @pytest.mark.asyncio
    async def test_stream_text_response(
        self, mock_openai_client, mock_config, mock_tools
    ) -> None:
        """Test streaming a simple text response without tool calls."""
        # Setup messages
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        # Create mock stream chunks
        chunks = [
            MockChunk([MockChoice(MockDelta(content="Hello "))]),
            MockChunk([MockChoice(MockDelta(content="world!"))]),
            MockChunk([MockChoice(finish_reason="stop")]),
        ]

        # Create an async generator for the mock stream
        async def async_chunks():
            for chunk in chunks:
                yield chunk

        # Mock the OpenAI completion
        with patch("backend.server.v2.chat.chat.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=async_chunks())

            # Collect streamed responses
            responses = []
            async for chunk in stream_chat_response(
                messages=messages,
                user_id="test-user",
                model="gpt-4o",
            ):
                if chunk.startswith("data: "):
                    data = json.loads(chunk[6:].strip())
                    responses.append(data)

            # Verify responses
            text_chunks = [r for r in responses if r.get("type") == "text_chunk"]
            assert len(text_chunks) == 2  # "Hello " and "world!"

            # Check for stream end
            end_markers = [r for r in responses if r.get("type") == "stream_end"]
            assert len(end_markers) == 1

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(
        self, mock_openai_client, mock_config, mock_tools
    ) -> None:
        """Test streaming with tool calls."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Execute tool"},
        ]

        # Create chunks with tool calls
        tool_call = MockToolCall(
            index=0,
            id="call-123",
            function=MockFunction(name="test_tool", arguments='{"input": "test"}'),
        )

        chunks_with_tools = [
            MockChunk([MockChoice(MockDelta(content="Let me help "))]),
            MockChunk([MockChoice(MockDelta(tool_calls=[tool_call]))]),
            MockChunk([MockChoice(finish_reason="tool_calls")]),
        ]

        # Second response after tool execution
        chunks_after_tools = [
            MockChunk([MockChoice(MockDelta(content="Tool result processed"))]),
            MockChunk([MockChoice(finish_reason="stop")]),
        ]

        async def mock_stream_with_tools():
            for chunk in chunks_with_tools:
                yield chunk

        async def mock_stream_after_tools():
            for chunk in chunks_after_tools:
                yield chunk

        # Mock the OpenAI completions and tool execution
        with patch("backend.server.v2.chat.chat.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                side_effect=[mock_stream_with_tools(), mock_stream_after_tools()],
            )

            with patch("backend.server.v2.chat.chat.execute_tool") as mock_execute:
                mock_execute.return_value = {"result": "Tool executed successfully"}

                # Collect streamed responses
                responses = []
                async for chunk in stream_chat_response(
                    messages=messages,
                    user_id="test-user",
                ):
                    if chunk.startswith("data: "):
                        data = json.loads(chunk[6:].strip())
                        responses.append(data)

                # Verify tool was executed
                mock_execute.assert_called_once()

                # Check for tool call notification
                tool_calls = [r for r in responses if r.get("type") == "tool_call"]
                assert len(tool_calls) == 1

                # Check for tool response
                tool_responses = [
                    r for r in responses if r.get("type") == "tool_response"
                ]
                assert len(tool_responses) == 1

    @pytest.mark.asyncio
    async def test_callbacks(self, mock_openai_client, mock_config, mock_tools) -> None:
        """Test that callbacks are invoked correctly."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Test"},
        ]

        chunks = [
            MockChunk([MockChoice(MockDelta(content="Response"))]),
            MockChunk([MockChoice(finish_reason="stop")]),
        ]

        async def async_chunks():
            for chunk in chunks:
                yield chunk

        # Mock callbacks
        assistant_callback = AsyncMock()
        tool_callback = AsyncMock()

        with patch("backend.server.v2.chat.chat.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(return_value=async_chunks())

            # Stream with callbacks
            responses = []
            async for chunk in stream_chat_response(
                messages=messages,
                user_id="test-user",
                on_assistant_message=assistant_callback,
                on_tool_response=tool_callback,
            ):
                responses.append(chunk)

            # Verify assistant callback was called
            assistant_callback.assert_called_once_with("Response", None)

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_openai_client, mock_config) -> None:
        """Test error handling in stream."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": "Test"},
        ]

        with patch("backend.server.v2.chat.chat.get_openai_client") as mock_get_client:
            mock_client = MagicMock()
            mock_get_client.return_value = mock_client
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("API Error")
            )

            # Collect streamed responses
            responses = []
            async for chunk in stream_chat_response(
                messages=messages,
                user_id="test-user",
            ):
                if chunk.startswith("data: "):
                    data = json.loads(chunk[6:].strip())
                    responses.append(data)

            # Should have error response
            error_responses = [r for r in responses if r.get("type") == "error"]
            assert len(error_responses) == 1
            assert "API Error" in error_responses[0].get("message", "")


class TestExecuteTool:
    """Test cases for execute_tool function."""

    @pytest.mark.asyncio
    async def test_execute_known_tool(self) -> None:
        """Test executing a known tool."""
        with patch("backend.server.v2.chat.tool_exports.find_agent_tool") as mock_tool:
            mock_instance = MagicMock()
            mock_tool.return_value = mock_instance
            mock_instance.execute = AsyncMock(
                return_value=MagicMock(
                    model_dump_json=lambda indent=2: json.dumps(
                        {"type": "agent_carousel", "agents": [{"id": "agent1"}]}
                    )
                )
            )

            result = await execute_tool(
                "find_agent",
                {"search_query": "test"},
                user_id="user",
                session_id="session",
            )

            result_dict = json.loads(result)
            assert isinstance(result_dict, dict)
            assert "agents" in result_dict
            mock_instance.execute.assert_called_once_with(
                "user", "session", search_query="test"
            )

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self) -> None:
        """Test executing an unknown tool."""
        result = await execute_tool(
            "unknown_tool",
            {},
            user_id="user",
            session_id="session",
        )

        result_dict = json.loads(result)
        assert result_dict["type"] == "error"
        assert "Unknown tool" in result_dict["message"]


class TestStreamChatCompletion:
    """Test cases for stream_chat_completion wrapper function."""

    @pytest.mark.asyncio
    async def test_database_operations(self, mock_db, mock_config, mock_tools) -> None:
        """Test that stream_chat_completion handles database operations."""

        # Mock the pure streaming function
        async def mock_stream():
            yield 'data: {"type": "text_chunk", "content": "Test"}\n\n'
            yield 'data: {"type": "stream_end"}\n\n'

        with patch(
            "backend.server.v2.chat.chat.stream_chat_response"
        ) as mock_stream_response:
            mock_stream_response.return_value = mock_stream()

            # Collect responses
            responses = []
            async for chunk in stream_chat_completion(
                session_id="session",
                user_message="Hello",
                user_id="user",
            ):
                responses.append(chunk)

            # Verify database operations
            assert mock_db.create_chat_message.called
            assert mock_db.get_conversation_context.called

            # Check user message was saved
            user_msg_calls = [
                call
                for call in mock_db.create_chat_message.call_args_list
                if call.kwargs.get("role") == ChatMessageRole.USER
            ]
            assert len(user_msg_calls) == 1
            assert user_msg_calls[0].kwargs["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_system_prompt_injection(self, mock_db, mock_config) -> None:
        """Test that system prompt is added when needed."""
        mock_db.get_conversation_context.return_value = []  # No existing messages

        async def mock_stream():
            yield 'data: {"type": "stream_end"}\n\n'

        with patch(
            "backend.server.v2.chat.chat.stream_chat_response"
        ) as mock_stream_response:
            mock_stream_response.return_value = mock_stream()

            async for _ in stream_chat_completion(
                session_id="session",
                user_message="Hello",
                user_id="user",
            ):
                pass

            # Check that stream_chat_response was called with system message
            call_args = mock_stream_response.call_args
            messages = call_args.kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_callbacks_creation(self, mock_db, mock_config) -> None:
        """Test that callbacks are properly created and passed."""

        async def mock_stream():
            yield 'data: {"type": "stream_end"}\n\n'

        with patch(
            "backend.server.v2.chat.chat.stream_chat_response"
        ) as mock_stream_response:
            mock_stream_response.return_value = mock_stream()

            async for _ in stream_chat_completion(
                session_id="session",
                user_message="Test",
                user_id="user",
            ):
                pass

            # Verify callbacks were passed
            call_args = mock_stream_response.call_args
            assert call_args.kwargs["on_assistant_message"] is not None
            assert call_args.kwargs["on_tool_response"] is not None
            assert call_args.kwargs["session_id"] == "session"
