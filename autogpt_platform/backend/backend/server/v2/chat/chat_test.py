"""Unit tests for the chat streaming service."""

import json
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ChatMessageRole

from backend.server.v2.chat.chat import (
    ChatStreamingService,
    get_chat_service,
    stream_chat_completion,
)


class MockDelta:
    """Mock OpenAI delta object."""

    def __init__(
        self, content: Optional[str] = None, tool_calls: Optional[List] = None
    ):
        self.content = content
        self.tool_calls = tool_calls


class MockChoice:
    """Mock OpenAI choice object."""

    def __init__(
        self, delta: Optional[MockDelta] = None, finish_reason: Optional[str] = None
    ):
        self.delta = delta or MockDelta()
        self.finish_reason = finish_reason


class MockChunk:
    """Mock OpenAI stream chunk."""

    def __init__(self, choices: Optional[List[MockChoice]] = None):
        self.choices = choices or []


class MockToolCall:
    """Mock tool call object."""

    def __init__(
        self, index: int, id: Optional[str] = None, function: Optional[Any] = None
    ):
        self.index = index
        self.id = id
        self.function = function


class MockFunction:
    """Mock function object for tool calls."""

    def __init__(self, name: Optional[str] = None, arguments: Optional[str] = None):
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
        }
    ]
    with patch("backend.server.v2.chat.chat.tools", tools_list):
        with patch("backend.server.v2.chat.tools.execute_test_tool", AsyncMock(
            return_value="Tool executed successfully"
        )):
            yield


@pytest.fixture
def chat_service(mock_openai_client):
    """Create a chat service instance with mocked dependencies."""
    service = ChatStreamingService(api_key="test-key")
    return service


class TestChatStreamingService:
    """Test cases for ChatStreamingService class."""

    @pytest.mark.asyncio
    async def test_init(self):
        """Test service initialization."""
        with patch("backend.server.v2.chat.chat.AsyncOpenAI") as mock_client:
            ChatStreamingService(api_key="test-key")
            mock_client.assert_called_once_with(api_key="test-key")

    @pytest.mark.asyncio
    async def test_init_with_env_var(self):
        """Test service initialization with environment variable."""
        with patch("backend.server.v2.chat.chat.os.getenv") as mock_getenv:
            with patch("backend.server.v2.chat.chat.AsyncOpenAI") as mock_client:
                mock_getenv.return_value = "env-api-key"
                ChatStreamingService()
                mock_client.assert_called_once_with(api_key="env-api-key")

    @pytest.mark.asyncio
    async def test_stream_text_response(self, chat_service, mock_db, mock_tools):
        """Test streaming a simple text response without tool calls."""
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

        # Mock the OpenAI completion to return an async iterator
        mock_completion = async_chunks()
        chat_service.client.chat.completions.create = AsyncMock(
            return_value=mock_completion
        )

        # Collect streamed responses
        responses = []
        async for chunk in chat_service.stream_chat_response(
            session_id="test-session", user_message="Hello", user_id="test-user"
        ):
            responses.append(chunk)

        # Verify responses
        assert len(responses) > 0

        # Check that messages were saved to database
        assert (
            mock_db.create_chat_message.call_count >= 2
        )  # User message + Assistant message

        # Verify user message was saved
        user_msg_call = mock_db.create_chat_message.call_args_list[0]
        assert user_msg_call.kwargs["content"] == "Hello"
        assert user_msg_call.kwargs["role"] == ChatMessageRole.USER

    @pytest.mark.asyncio
    async def test_stream_with_tool_calls(self, chat_service, mock_db, mock_tools):
        """Test streaming with tool calls."""
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

        # Mock the OpenAI completions
        mock_completion1 = mock_stream_with_tools()
        mock_completion2 = mock_stream_after_tools()

        chat_service.client.chat.completions.create = AsyncMock(
            side_effect=[mock_completion1, mock_completion2]
        )

        # Mock tool execution
        with patch.object(
            chat_service, "_execute_tool", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = "Tool executed successfully"

            # Collect streamed responses
            responses = []
            async for chunk in chat_service.stream_chat_response(
                session_id="test-session",
                user_message="Execute tool",
                user_id="test-user",
            ):
                responses.append(chunk)

            # Verify tool was executed
            mock_execute.assert_called_once()

            # Verify multiple database saves (user, assistant with tools, tool result, final assistant)
            assert mock_db.create_chat_message.call_count >= 3

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, chat_service):
        """Test successful tool execution."""
        with patch(
            "backend.server.v2.chat.chat.tools.execute_test_tool",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_exec.return_value = "Success"

            result = await chat_service._execute_tool(
                "test_tool", {"param": "value"}, user_id="user", session_id="session"
            )

            assert result == "Success"
            mock_exec.assert_called_once_with(
                {"param": "value"}, user_id="user", session_id="session"
            )

    @pytest.mark.asyncio
    async def test_execute_tool_not_found(self, chat_service):
        """Test tool execution when tool doesn't exist."""
        result = await chat_service._execute_tool(
            "nonexistent_tool", {}, user_id="user", session_id="session"
        )

        assert "not implemented" in result.lower()

    def test_create_tool_call_ui(self, chat_service):
        """Test tool call UI generation."""
        html = chat_service._create_tool_call_ui("test_tool", {"param": "value"})

        assert "test_tool" in html
        assert "param" in html
        assert "value" in html
        assert "tool-call-container" in html

    def test_create_executing_ui(self, chat_service):
        """Test executing UI generation."""
        html = chat_service._create_executing_ui("test_tool")

        assert "test_tool" in html
        assert "Executing" in html
        assert "tool-executing" in html

    def test_create_result_ui(self, chat_service):
        """Test result UI generation."""
        html = chat_service._create_result_ui("Test result content")

        assert "Test result content" in html
        assert "Tool Result" in html
        assert "tool-result" in html

    def test_create_processing_ui(self, chat_service):
        """Test processing UI generation."""
        html = chat_service._create_processing_ui()

        assert "Processing" in html
        assert "tool results" in html

    @pytest.mark.asyncio
    async def test_error_handling(self, chat_service, mock_db):
        """Test error handling in stream."""
        # Mock an error in OpenAI call
        chat_service.client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        # Collect streamed responses
        responses = []
        async for chunk in chat_service.stream_chat_response(
            session_id="test-session", user_message="Test", user_id="test-user"
        ):
            responses.append(chunk)

        # Should have error response
        assert len(responses) > 0
        assert "error" in responses[0].lower()
        assert "API Error" in responses[0]


class TestModuleFunctions:
    """Test module-level functions."""

    def test_get_chat_service_singleton(self):
        """Test that get_chat_service returns singleton."""
        with patch("backend.server.v2.chat.chat.AsyncOpenAI"):
            service1 = get_chat_service()
            service2 = get_chat_service()

            assert service1 is service2

    @pytest.mark.asyncio
    async def test_stream_chat_completion(self, mock_db, mock_tools):
        """Test the main stream_chat_completion function."""
        with patch("backend.server.v2.chat.chat.get_chat_service") as mock_get_service:
            mock_service = MagicMock()
            mock_get_service.return_value = mock_service

            async def mock_stream():
                yield "data: test\n\n"

            mock_service.stream_chat_response = MagicMock(return_value=mock_stream())

            # Collect responses
            responses = []
            async for chunk in stream_chat_completion(
                session_id="session", user_message="message", user_id="user"
            ):
                responses.append(chunk)

            assert len(responses) == 1
            assert responses[0] == "data: test\n\n"

            # Verify service method was called correctly
            mock_service.stream_chat_response.assert_called_once_with(
                session_id="session",
                user_message="message",
                user_id="user",
                model="gpt-4o",
                max_messages=50,
            )


class TestIntegration:
    """Integration tests for the chat service."""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, mock_db, mock_tools):
        """Test a complete conversation flow with tool calls."""
        with patch("backend.server.v2.chat.chat.AsyncOpenAI") as mock_client_class:
            # Setup mock client
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Create service
            service = ChatStreamingService(api_key="test-key")

            # Mock conversation context
            mock_db.get_conversation_context.return_value = [
                {"role": "system", "content": "You are a helpful assistant."}
            ]

            # Create tool call chunks
            tool_call = MockToolCall(
                index=0,
                id="call-456",
                function=MockFunction(
                    name="find_agent", arguments='{"search_query": "data"}'
                ),
            )

            initial_chunks = [
                MockChunk([MockChoice(MockDelta(content="I'll search for "))]),
                MockChunk([MockChoice(MockDelta(content="agents for you."))]),
                MockChunk([MockChoice(MockDelta(tool_calls=[tool_call]))]),
                MockChunk([MockChoice(finish_reason="tool_calls")]),
            ]

            final_chunks = [
                MockChunk([MockChoice(MockDelta(content="Found 2 agents"))]),
                MockChunk([MockChoice(finish_reason="stop")]),
            ]

            async def mock_initial_stream():
                for chunk in initial_chunks:
                    yield chunk

            async def mock_final_stream():
                for chunk in final_chunks:
                    yield chunk

            mock_completion1 = mock_initial_stream()
            mock_completion2 = mock_final_stream()

            mock_client.chat.completions.create = AsyncMock(
                side_effect=[mock_completion1, mock_completion2]
            )

            # Mock tool execution
            with patch("backend.server.v2.chat.chat.tools") as mock_tools_module:
                mock_tools_module.execute_find_agent = AsyncMock(
                    return_value="Found agents: Agent1, Agent2"
                )
                mock_tools_module.tools = []

                # Collect all responses
                responses = []
                async for chunk in service.stream_chat_response(
                    session_id="test-session",
                    user_message="Find data agents",
                    user_id="test-user",
                ):
                    if chunk.startswith("data: "):
                        try:
                            data = json.loads(chunk[6:].strip())
                            responses.append(data)
                        except json.JSONDecodeError:
                            pass

                # Verify we got text, HTML (tool UI), and final text
                text_responses = [r for r in responses if r.get("type") == "text"]
                html_responses = [r for r in responses if r.get("type") == "html"]

                assert len(text_responses) > 0  # Should have text responses
                assert len(html_responses) > 0  # Should have tool UI responses

                # Verify database interactions
                assert mock_db.create_chat_message.called
                assert mock_db.get_conversation_context.called
