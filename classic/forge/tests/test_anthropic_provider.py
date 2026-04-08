"""Tests for Anthropic provider: message prep, tool parsing, retry, error handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from forge.llm.providers.anthropic import (
    ANTHROPIC_CHAT_MODELS,
    AnthropicCredentials,
    AnthropicModelName,
    AnthropicProvider,
    AnthropicSettings,
)
from forge.llm.providers.schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
    ChatMessage,
    CompletionModelFunction,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ToolResultMessage,
)
from forge.models.json_schema import JSONSchema


# ---------------------------------------------------------------------------
# Helper to create a provider without real credentials
# ---------------------------------------------------------------------------
@pytest.fixture
def provider():
    settings = AnthropicSettings(
        name="test_anthropic",
        description="Test",
        configuration=ModelProviderConfiguration(),
        credentials=AnthropicCredentials(api_key="test-key"),  # type: ignore
        budget=ModelProviderBudget(),
    )
    with patch("anthropic.AsyncAnthropic"):
        p = AnthropicProvider(settings=settings)
    return p


@pytest.fixture
def search_function():
    return CompletionModelFunction(
        name="web_search",
        description="Search the web",
        parameters={
            "query": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Search query",
                required=True,
            ),
        },
    )


@pytest.fixture
def write_function():
    return CompletionModelFunction(
        name="write_file",
        description="Write content to a file",
        parameters={
            "path": JSONSchema(
                type=JSONSchema.Type.STRING, description="Path", required=True
            ),
            "content": JSONSchema(
                type=JSONSchema.Type.STRING, description="Content", required=True
            ),
        },
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
class TestAnthropicModels:
    def test_all_models_have_function_call_api(self):
        for name, info in ANTHROPIC_CHAT_MODELS.items():
            assert info.has_function_call_api is True, f"{name} missing function call"

    def test_opus_46_exists(self):
        assert AnthropicModelName.CLAUDE4_6_OPUS_v1 in ANTHROPIC_CHAT_MODELS

    def test_rolling_opus_points_to_latest(self):
        assert AnthropicModelName.CLAUDE_OPUS in ANTHROPIC_CHAT_MODELS

    def test_rolling_sonnet_exists(self):
        assert AnthropicModelName.CLAUDE_SONNET in ANTHROPIC_CHAT_MODELS

    def test_rolling_haiku_exists(self):
        assert AnthropicModelName.CLAUDE_HAIKU in ANTHROPIC_CHAT_MODELS

    def test_claude4_models_support_extended_thinking(self):
        for name in [
            AnthropicModelName.CLAUDE4_SONNET_v1,
            AnthropicModelName.CLAUDE4_OPUS_v1,
            AnthropicModelName.CLAUDE4_5_OPUS_v1,
            AnthropicModelName.CLAUDE4_6_OPUS_v1,
        ]:
            assert ANTHROPIC_CHAT_MODELS[name].supports_extended_thinking is True

    def test_pre_claude4_models_do_not_support_extended_thinking(self):
        """Only Claude 4+ supports extended thinking. Flagging earlier models
        would cause API errors when thinking_budget_tokens is configured."""
        for name in [
            AnthropicModelName.CLAUDE3_OPUS_v1,
            AnthropicModelName.CLAUDE3_SONNET_v1,
            AnthropicModelName.CLAUDE3_HAIKU_v1,
            AnthropicModelName.CLAUDE3_5_SONNET_v1,
            AnthropicModelName.CLAUDE3_5_SONNET_v2,
        ]:
            assert (
                ANTHROPIC_CHAT_MODELS[name].supports_extended_thinking is False
            ), f"{name} should NOT support extended thinking"


# ---------------------------------------------------------------------------
# AnthropicCredentials
# ---------------------------------------------------------------------------
class TestAnthropicCredentials:
    def test_get_api_access_kwargs(self):
        creds = AnthropicCredentials(api_key="sk-test")  # type: ignore
        kwargs = creds.get_api_access_kwargs()
        assert kwargs["api_key"] == "sk-test"
        assert "base_url" not in kwargs  # None values excluded

    def test_get_api_access_kwargs_with_base_url(self):
        creds = AnthropicCredentials(
            api_key=SecretStr("sk-test"),
            api_base=SecretStr("https://custom.api"),
        )
        kwargs = creds.get_api_access_kwargs()
        assert kwargs["api_key"] == "sk-test"
        assert kwargs["base_url"] == "https://custom.api"


# ---------------------------------------------------------------------------
# _get_chat_completion_args
# ---------------------------------------------------------------------------
class TestAnthropicGetChatCompletionArgs:
    def test_system_message_extracted(self, provider):
        messages = [
            ChatMessage.system("You are helpful"),
            ChatMessage.user("Hello"),
        ]
        anthropic_msgs, kwargs = provider._get_chat_completion_args(
            prompt_messages=messages
        )
        # System message should be in kwargs, not in messages
        assert kwargs["system"] == "You are helpful"
        assert len(anthropic_msgs) == 1
        assert anthropic_msgs[0]["role"] == "user"

    def test_multiple_system_messages_merged(self, provider):
        messages = [
            ChatMessage.system("Rule 1"),
            ChatMessage.system("Rule 2"),
            ChatMessage.user("Hi"),
        ]
        _, kwargs = provider._get_chat_completion_args(prompt_messages=messages)
        assert "Rule 1" in kwargs["system"]
        assert "Rule 2" in kwargs["system"]

    def test_consecutive_user_messages_merged(self, provider):
        messages = [
            ChatMessage.user("Part 1"),
            ChatMessage.user("Part 2"),
        ]
        anthropic_msgs, _ = provider._get_chat_completion_args(prompt_messages=messages)
        # Should be merged into one user message
        assert len(anthropic_msgs) == 1
        assert "Part 1" in anthropic_msgs[0]["content"]
        assert "Part 2" in anthropic_msgs[0]["content"]

    def test_user_messages_not_merged_when_separated(self, provider):
        messages = [
            ChatMessage.user("First"),
            AssistantChatMessage(content="Response"),
            ChatMessage.user("Second"),
        ]
        anthropic_msgs, _ = provider._get_chat_completion_args(prompt_messages=messages)
        assert len(anthropic_msgs) == 3

    def test_assistant_message_with_tool_calls(self, provider):
        tc = AssistantToolCall(
            id="tool_1",
            type="function",
            function=AssistantFunctionCall(name="search", arguments={"query": "test"}),
        )
        messages = [
            ChatMessage.user("Search"),
            AssistantChatMessage(content="Searching...", tool_calls=[tc]),
        ]
        anthropic_msgs, _ = provider._get_chat_completion_args(prompt_messages=messages)
        assistant_msg = anthropic_msgs[1]
        assert assistant_msg["role"] == "assistant"
        # Should have content blocks
        assert isinstance(assistant_msg["content"], list)
        # First block is text, second is tool_use
        text_blocks = [b for b in assistant_msg["content"] if b["type"] == "text"]
        tool_blocks = [b for b in assistant_msg["content"] if b["type"] == "tool_use"]
        assert len(text_blocks) == 1
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "search"
        assert tool_blocks[0]["input"] == {"query": "test"}

    def test_assistant_message_tool_calls_without_content(self, provider):
        tc = AssistantToolCall(
            id="tool_1",
            type="function",
            function=AssistantFunctionCall(name="noop", arguments={}),
        )
        messages = [
            ChatMessage.user("Do it"),
            AssistantChatMessage(content="", tool_calls=[tc]),
        ]
        anthropic_msgs, _ = provider._get_chat_completion_args(prompt_messages=messages)
        assistant_msg = anthropic_msgs[1]
        content = assistant_msg["content"]
        # No text block when content is empty
        text_blocks = [b for b in content if b["type"] == "text"]
        assert len(text_blocks) == 0

    def test_assistant_message_without_content_or_tool_calls_skipped(self, provider):
        messages = [
            ChatMessage.user("Hi"),
            AssistantChatMessage(content=""),
        ]
        anthropic_msgs, _ = provider._get_chat_completion_args(prompt_messages=messages)
        # Empty assistant message should be skipped
        assert len(anthropic_msgs) == 1

    def test_tool_result_message(self, provider):
        messages = [
            ToolResultMessage(tool_call_id="tool_1", content="Search results here"),
        ]
        anthropic_msgs, _ = provider._get_chat_completion_args(prompt_messages=messages)
        msg = anthropic_msgs[0]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "tool_1"

    def test_tool_result_error_message(self, provider):
        messages = [
            ToolResultMessage(
                tool_call_id="tool_1", content="Error occurred", is_error=True
            ),
        ]
        anthropic_msgs, _ = provider._get_chat_completion_args(prompt_messages=messages)
        assert anthropic_msgs[0]["content"][0]["is_error"] is True

    def test_functions_converted_to_tools(self, provider, search_function):
        messages = [ChatMessage.user("search")]
        _, kwargs = provider._get_chat_completion_args(
            prompt_messages=messages, functions=[search_function]
        )
        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 1
        tool = kwargs["tools"][0]
        assert tool["name"] == "web_search"
        assert tool["description"] == "Search the web"
        assert "input_schema" in tool
        assert "query" in tool["input_schema"]["properties"]
        assert tool["input_schema"]["required"] == ["query"]

    def test_default_max_tokens(self, provider):
        messages = [ChatMessage.user("hi")]
        _, kwargs = provider._get_chat_completion_args(prompt_messages=messages)
        assert kwargs["max_tokens"] == 4096

    def test_custom_max_tokens(self, provider):
        messages = [ChatMessage.user("hi")]
        _, kwargs = provider._get_chat_completion_args(
            prompt_messages=messages, max_output_tokens=8192
        )
        assert kwargs["max_tokens"] == 8192

    def test_extended_thinking_enabled(self, provider):
        messages = [ChatMessage.user("think")]
        _, kwargs = provider._get_chat_completion_args(
            prompt_messages=messages, thinking_budget_tokens=2048
        )
        assert kwargs["thinking"]["type"] == "enabled"
        assert kwargs["thinking"]["budget_tokens"] == 2048

    def test_extended_thinking_minimum_1024(self, provider):
        messages = [ChatMessage.user("think")]
        _, kwargs = provider._get_chat_completion_args(
            prompt_messages=messages, thinking_budget_tokens=500
        )
        assert kwargs["thinking"]["budget_tokens"] == 1024

    def test_extended_thinking_with_tools_adds_beta_header(
        self, provider, search_function
    ):
        messages = [ChatMessage.user("think and search")]
        _, kwargs = provider._get_chat_completion_args(
            prompt_messages=messages,
            functions=[search_function],
            thinking_budget_tokens=2048,
        )
        assert "anthropic-beta" in kwargs.get("extra_headers", {})

    def test_extended_thinking_zero_budget_not_enabled(self, provider):
        messages = [ChatMessage.user("hi")]
        _, kwargs = provider._get_chat_completion_args(
            prompt_messages=messages, thinking_budget_tokens=0
        )
        assert "thinking" not in kwargs

    def test_extra_request_headers(self, provider):
        provider._configuration.extra_request_headers = {"X-Custom": "value"}
        messages = [ChatMessage.user("hi")]
        _, kwargs = provider._get_chat_completion_args(prompt_messages=messages)
        assert kwargs["extra_headers"]["X-Custom"] == "value"


# ---------------------------------------------------------------------------
# _parse_assistant_tool_calls
# ---------------------------------------------------------------------------
class TestAnthropicParseToolCalls:
    def _make_anthropic_response(self, content_blocks):
        msg = MagicMock()
        msg.content = content_blocks
        return msg

    def _make_text_block(self, text):
        block = MagicMock()
        block.type = "text"
        block.text = text
        return block

    def _make_tool_use_block(self, id, name, input_data):
        block = MagicMock()
        block.type = "tool_use"
        block.id = id
        block.name = name
        block.input = input_data
        return block

    def test_parse_single_tool_call(self, provider):
        blocks = [
            self._make_text_block("Searching..."),
            self._make_tool_use_block("tool_1", "search", {"query": "test"}),
        ]
        response = self._make_anthropic_response(blocks)
        tool_calls = provider._parse_assistant_tool_calls(response)
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "tool_1"
        assert tool_calls[0].function.name == "search"
        assert tool_calls[0].function.arguments == {"query": "test"}

    def test_parse_multiple_tool_calls(self, provider):
        blocks = [
            self._make_tool_use_block("t1", "search", {"q": "a"}),
            self._make_tool_use_block("t2", "write", {"p": "b"}),
        ]
        response = self._make_anthropic_response(blocks)
        tool_calls = provider._parse_assistant_tool_calls(response)
        assert len(tool_calls) == 2

    def test_text_only_returns_empty(self, provider):
        blocks = [self._make_text_block("Just text")]
        response = self._make_anthropic_response(blocks)
        tool_calls = provider._parse_assistant_tool_calls(response)
        assert tool_calls == []

    def test_empty_content_returns_empty(self, provider):
        response = self._make_anthropic_response([])
        tool_calls = provider._parse_assistant_tool_calls(response)
        assert tool_calls == []

    def test_tool_call_type_is_function(self, provider):
        blocks = [self._make_tool_use_block("t1", "fn", {})]
        response = self._make_anthropic_response(blocks)
        tool_calls = provider._parse_assistant_tool_calls(response)
        assert tool_calls[0].type == "function"


# ---------------------------------------------------------------------------
# _get_tool_error_message
# ---------------------------------------------------------------------------
class TestGetToolErrorMessage:
    def _make_tool_call(self, name, arguments):
        return AssistantToolCall(
            id="tool_1",
            type="function",
            function=AssistantFunctionCall(name=name, arguments=arguments),
        )

    def test_no_errors_returns_default_message(self, provider):
        tc = self._make_tool_call("search", {"q": "test"})
        msg = provider._get_tool_error_message(tc, [], None)
        assert "parsing" in msg.lower()

    def test_matching_error_included(self, provider, search_function):
        tc = self._make_tool_call("web_search", {"wrong": "arg"})
        err = MagicMock()
        err.name = "web_search"
        err.__str__ = MagicMock(return_value="Invalid args for web_search")
        msg = provider._get_tool_error_message(tc, [err], [search_function])
        assert "web_search" in msg

    def test_no_matching_error_returns_default(self, provider):
        tc = self._make_tool_call("search", {"q": "test"})
        err = MagicMock()
        err.name = "other_function"
        msg = provider._get_tool_error_message(tc, [err], None)
        assert "validation failed" in msg.lower()

    def test_empty_arguments_shows_no_arguments(self, provider):
        tc = self._make_tool_call("search", {})
        err = MagicMock()
        err.name = "search"
        err.__str__ = MagicMock(return_value="Error")
        msg = provider._get_tool_error_message(tc, [err], None)
        assert "(no arguments)" in msg

    def test_expected_parameters_shown(self, provider, search_function):
        tc = self._make_tool_call("web_search", {"wrong": "arg"})
        err = MagicMock()
        err.name = "web_search"
        err.__str__ = MagicMock(return_value="Error")
        msg = provider._get_tool_error_message(tc, [err], [search_function])
        assert "Expected parameters" in msg
        assert "query" in msg


# ---------------------------------------------------------------------------
# create_chat_completion
# ---------------------------------------------------------------------------
class TestAnthropicCreateChatCompletion:
    def _make_response(self, text="Response", tool_use_blocks=None):
        blocks = []
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        blocks.append(text_block)

        if tool_use_blocks:
            blocks.extend(tool_use_blocks)

        response = MagicMock()
        response.content = blocks
        response.usage = MagicMock()
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50
        response.model_dump = MagicMock(
            return_value={
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            }
        )
        return response

    @pytest.mark.asyncio
    async def test_successful_completion(self, provider):
        response = self._make_response(text="Hello!")
        provider._client.messages.create = AsyncMock(return_value=response)

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("Hi")],
            model_name=AnthropicModelName.CLAUDE4_SONNET_v1,
        )
        assert result.response.content == "Hello!"

    @pytest.mark.asyncio
    async def test_completion_with_parser(self, provider):
        response = self._make_response(text="parsed")
        provider._client.messages.create = AsyncMock(return_value=response)

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("Hi")],
            model_name=AnthropicModelName.CLAUDE4_SONNET_v1,
            completion_parser=lambda msg: msg.content.upper(),
        )
        assert result.parsed_result == "PARSED"

    @pytest.mark.asyncio
    async def test_completion_tracks_cost(self, provider):
        response = self._make_response()
        provider._client.messages.create = AsyncMock(return_value=response)
        initial_cost = provider._budget.total_cost

        await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("Hi")],
            model_name=AnthropicModelName.CLAUDE4_SONNET_v1,
        )
        assert provider._budget.total_cost > initial_cost

    @pytest.mark.asyncio
    async def test_prefill_response_merged(self, provider):
        response = self._make_response(text=" continued text")
        provider._client.messages.create = AsyncMock(return_value=response)

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("Hi")],
            model_name=AnthropicModelName.CLAUDE4_SONNET_v1,
            prefill_response="Start:",
        )
        # Prefill should be merged into the first text block
        assert result.response.content.startswith("Start:")

    @pytest.mark.asyncio
    async def test_parse_failure_retries(self, provider):
        response1 = self._make_response(text="bad")
        response2 = self._make_response(text="good")
        provider._client.messages.create = AsyncMock(side_effect=[response1, response2])

        call_count = 0

        def flaky_parser(msg):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Parse failed")
            return msg.content

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("test")],
            model_name=AnthropicModelName.CLAUDE4_SONNET_v1,
            completion_parser=flaky_parser,
        )
        assert result.parsed_result == "good"

    @pytest.mark.asyncio
    async def test_parse_failure_exhausts_retries(self, provider):
        response = self._make_response(text="bad")
        provider._client.messages.create = AsyncMock(return_value=response)
        provider._configuration.fix_failed_parse_tries = 2

        with pytest.raises(ValueError, match="Always fails"):
            await provider.create_chat_completion(
                model_prompt=[ChatMessage.user("test")],
                model_name=AnthropicModelName.CLAUDE4_SONNET_v1,
                completion_parser=lambda _: (_ for _ in ()).throw(
                    ValueError("Always fails")
                ),
            )


# ---------------------------------------------------------------------------
# Token limits
# ---------------------------------------------------------------------------
class TestAnthropicTokenLimits:
    def test_get_token_limit_matches_model_definition(self, provider):
        limit = provider.get_token_limit(AnthropicModelName.CLAUDE4_SONNET_v1)
        assert (
            limit
            == ANTHROPIC_CHAT_MODELS[AnthropicModelName.CLAUDE4_SONNET_v1].max_tokens
        )

    def test_all_models_have_expected_context(self, provider):
        for name, info in ANTHROPIC_CHAT_MODELS.items():
            assert info.max_tokens in (
                200000,
                1000000,
            ), f"{name} has unexpected context size {info.max_tokens}"


class TestAnthropicTokenCounting:
    def test_count_tokens_returns_positive(self, provider):
        count = provider.count_tokens(
            "hello world this is a test", AnthropicModelName.CLAUDE4_SONNET_v1
        )
        assert count > 0

    def test_count_tokens_longer_text_has_more_tokens(self, provider):
        short = provider.count_tokens("hi", AnthropicModelName.CLAUDE4_SONNET_v1)
        long = provider.count_tokens(
            "this is a much longer sentence with many words in it",
            AnthropicModelName.CLAUDE4_SONNET_v1,
        )
        assert long > short

    def test_count_message_tokens_returns_positive(self, provider):
        msg = ChatMessage.user("hello world")
        count = provider.count_message_tokens(msg, AnthropicModelName.CLAUDE4_SONNET_v1)
        assert count > 0

    def test_count_message_tokens_accepts_list(self, provider):
        msgs = [ChatMessage.user("hello"), ChatMessage.system("be helpful")]
        count = provider.count_message_tokens(
            msgs, AnthropicModelName.CLAUDE4_SONNET_v1
        )
        single = provider.count_message_tokens(
            msgs[0], AnthropicModelName.CLAUDE4_SONNET_v1
        )
        assert count > single
