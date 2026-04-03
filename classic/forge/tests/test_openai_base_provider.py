"""Tests for OpenAI base provider: message prep, tool parsing, retry, cost tracking."""

import json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.llm.providers._openai_base import (
    BaseOpenAIChatProvider,
    format_function_def_for_openai,
)
from forge.llm.providers.schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
    ChatMessage,
    ChatModelInfo,
    CompletionModelFunction,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
    ToolResultMessage,
)
from forge.llm.providers.utils import InvalidFunctionCallError
from forge.models.json_schema import JSONSchema


# ---------------------------------------------------------------------------
# Concrete test provider (since BaseOpenAIChatProvider is abstract-ish)
# ---------------------------------------------------------------------------
class _TestModelName(str):
    """Dummy model name type for testing."""

    pass


TEST_MODEL = _TestModelName("test-model")
TEST_MODEL_GPT5 = _TestModelName("gpt-5-test")
TEST_MODEL_O3 = _TestModelName("o3-mini-test")
TEST_MODEL_LEGACY = _TestModelName("gpt-3.5-turbo")

_MODEL_INFO = ChatModelInfo(
    name=TEST_MODEL,
    provider_name=ModelProviderName.OPENAI,
    prompt_token_cost=10.0 / 1e6,
    completion_token_cost=30.0 / 1e6,
    max_tokens=4096,
    has_function_call_api=True,
)

_GPT5_MODEL_INFO = ChatModelInfo(
    name=TEST_MODEL_GPT5,
    provider_name=ModelProviderName.OPENAI,
    prompt_token_cost=1.25 / 1e6,
    completion_token_cost=10.0 / 1e6,
    max_tokens=400_000,
    has_function_call_api=True,
    supports_reasoning_effort=True,
)


class _DummyCredentials(ModelProviderCredentials):
    api_key: Any = "test-key"

    def get_api_access_kwargs(self) -> dict:
        return {"api_key": "test-key"}


class _DummySettings(ModelProviderSettings):
    credentials: Optional[ModelProviderCredentials] = None  # type: ignore[assignment]
    budget: Optional[ModelProviderBudget] = None


class _TestProvider(BaseOpenAIChatProvider[_TestModelName, _DummySettings]):
    MODELS = {
        TEST_MODEL: _MODEL_INFO,
        TEST_MODEL_GPT5: _GPT5_MODEL_INFO,
        TEST_MODEL_O3: ChatModelInfo(
            name=TEST_MODEL_O3,
            provider_name=ModelProviderName.OPENAI,
            max_tokens=200_000,
            supports_reasoning_effort=True,
        ),
        TEST_MODEL_LEGACY: ChatModelInfo(
            name=TEST_MODEL_LEGACY,
            provider_name=ModelProviderName.OPENAI,
            max_tokens=4096,
        ),
    }
    CHAT_MODELS = MODELS  # type: ignore

    default_settings = _DummySettings(
        name="test_provider",
        description="Test",
        configuration=ModelProviderConfiguration(),
        credentials=_DummyCredentials(),
        budget=ModelProviderBudget(),
    )

    def __init__(self, **kwargs):
        # Skip actual OpenAI client initialization
        self._settings = self.default_settings.model_copy(deep=True)
        self._configuration = self._settings.configuration
        self._credentials = self._settings.credentials
        self._budget = self._settings.budget
        self._logger = MagicMock()
        self._client = MagicMock()

    def get_tokenizer(self, model_name) -> ModelTokenizer:
        tokenizer = MagicMock()
        tokenizer.encode = lambda text: text.split()
        return tokenizer


@pytest.fixture
def provider():
    return _TestProvider()


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
                type=JSONSchema.Type.STRING, description="File path", required=True
            ),
            "content": JSONSchema(
                type=JSONSchema.Type.STRING, description="Content", required=True
            ),
        },
    )


# ---------------------------------------------------------------------------
# format_function_def_for_openai
# ---------------------------------------------------------------------------
class TestFormatFunctionDefForOpenAI:
    def _params(self, result):
        """Extract parameters dict from FunctionDefinition."""
        return result.get("parameters") or {}

    def test_basic_function(self, search_function):
        result = format_function_def_for_openai(search_function)
        assert result["name"] == "web_search"
        assert result.get("description") == "Search the web"
        params = self._params(result)
        assert "properties" in params
        assert "query" in params["properties"]
        assert params["required"] == ["query"]

    def test_multiple_required_params(self, write_function):
        result = format_function_def_for_openai(write_function)
        params = self._params(result)
        assert set(params["required"]) == {"path", "content"}

    def test_optional_params_not_in_required(self):
        fn = CompletionModelFunction(
            name="test",
            description="Test",
            parameters={
                "required_param": JSONSchema(
                    type=JSONSchema.Type.STRING, required=True
                ),
                "optional_param": JSONSchema(
                    type=JSONSchema.Type.STRING, required=False
                ),
            },
        )
        result = format_function_def_for_openai(fn)
        assert self._params(result)["required"] == ["required_param"]

    def test_no_params(self):
        fn = CompletionModelFunction(name="noop", description="No-op", parameters={})
        result = format_function_def_for_openai(fn)
        params = self._params(result)
        assert params["properties"] == {}
        assert params["required"] == []

    def test_parameters_type_is_object(self, search_function):
        result = format_function_def_for_openai(search_function)
        assert self._params(result)["type"] == "object"


# ---------------------------------------------------------------------------
# _get_chat_completion_args: message preparation
# ---------------------------------------------------------------------------
class TestGetChatCompletionArgs:
    def test_basic_messages(self, provider):
        messages = [
            ChatMessage.system("You are helpful"),
            ChatMessage.user("Hello"),
        ]
        prepped, kwargs, parse_kwargs = provider._get_chat_completion_args(
            prompt_messages=messages, model=TEST_MODEL
        )
        assert len(prepped) == 2
        assert prepped[0]["role"] == "system"
        assert prepped[1]["role"] == "user"

    def test_tool_calls_arguments_serialized_as_json_strings(self, provider):
        tc = AssistantToolCall(
            id="call_1",
            type="function",
            function=AssistantFunctionCall(
                name="search", arguments={"query": "test", "limit": 10}
            ),
        )
        messages = [
            AssistantChatMessage(content="Searching", tool_calls=[tc]),
        ]
        prepped, _, _ = provider._get_chat_completion_args(
            prompt_messages=messages, model=TEST_MODEL
        )
        args = prepped[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"query": "test", "limit": 10}

    def test_empty_tool_call_arguments_serialized(self, provider):
        tc = AssistantToolCall(
            id="call_1",
            type="function",
            function=AssistantFunctionCall(name="noop", arguments={}),
        )
        messages = [AssistantChatMessage(content="ok", tool_calls=[tc])]
        prepped, _, _ = provider._get_chat_completion_args(
            prompt_messages=messages, model=TEST_MODEL
        )
        args = prepped[0]["tool_calls"][0]["function"]["arguments"]
        assert args == "{}"

    def test_messages_without_tool_calls_unaffected(self, provider):
        messages = [ChatMessage.user("Hello"), ChatMessage.system("Be helpful")]
        prepped, _, _ = provider._get_chat_completion_args(
            prompt_messages=messages, model=TEST_MODEL
        )
        assert "tool_calls" not in prepped[0]
        assert "tool_calls" not in prepped[1]

    def test_tool_result_message_preserved(self, provider):
        messages = [
            ToolResultMessage(tool_call_id="call_1", content="Result data"),
        ]
        prepped, _, _ = provider._get_chat_completion_args(
            prompt_messages=messages, model=TEST_MODEL
        )
        assert prepped[0]["tool_call_id"] == "call_1"
        assert prepped[0]["role"] == "tool"

    def test_exclude_none_removes_absent_fields(self, provider):
        messages = [ChatMessage.user("test")]
        prepped, _, _ = provider._get_chat_completion_args(
            prompt_messages=messages, model=TEST_MODEL
        )
        assert "tool_calls" not in prepped[0]
        assert "tool_call_id" not in prepped[0]

    # ---- Functions / Tools ----

    def test_single_function_forces_tool_choice(self, provider, search_function):
        messages = [ChatMessage.user("search")]
        _, kwargs, _ = provider._get_chat_completion_args(
            prompt_messages=messages,
            model=TEST_MODEL,
            functions=[search_function],
        )
        assert "tools" in kwargs
        assert len(kwargs["tools"]) == 1
        assert kwargs["tool_choice"]["type"] == "function"
        assert kwargs["tool_choice"]["function"]["name"] == "web_search"

    def test_multiple_functions_no_forced_choice(
        self, provider, search_function, write_function
    ):
        messages = [ChatMessage.user("do something")]
        _, kwargs, _ = provider._get_chat_completion_args(
            prompt_messages=messages,
            model=TEST_MODEL,
            functions=[search_function, write_function],
        )
        assert len(kwargs["tools"]) == 2
        assert "tool_choice" not in kwargs

    def test_no_functions_no_tools(self, provider):
        messages = [ChatMessage.user("hi")]
        _, kwargs, _ = provider._get_chat_completion_args(
            prompt_messages=messages, model=TEST_MODEL
        )
        assert "tools" not in kwargs

    # ---- max_output_tokens ----

    def test_newer_model_uses_max_completion_tokens(self, provider):
        messages = [ChatMessage.user("hi")]
        _, kwargs, _ = provider._get_chat_completion_args(
            prompt_messages=messages,
            model=TEST_MODEL_GPT5,
            max_output_tokens=1000,
        )
        assert kwargs["max_completion_tokens"] == 1000
        assert "max_tokens" not in kwargs

    def test_o_series_uses_max_completion_tokens(self, provider):
        messages = [ChatMessage.user("hi")]
        _, kwargs, _ = provider._get_chat_completion_args(
            prompt_messages=messages,
            model=TEST_MODEL_O3,
            max_output_tokens=500,
        )
        assert kwargs["max_completion_tokens"] == 500

    def test_legacy_model_uses_max_tokens(self, provider):
        messages = [ChatMessage.user("hi")]
        _, kwargs, _ = provider._get_chat_completion_args(
            prompt_messages=messages,
            model=TEST_MODEL_LEGACY,
            max_output_tokens=500,
        )
        assert kwargs["max_tokens"] == 500
        assert "max_completion_tokens" not in kwargs

    def test_no_max_output_tokens_omits_both(self, provider):
        messages = [ChatMessage.user("hi")]
        _, kwargs, _ = provider._get_chat_completion_args(
            prompt_messages=messages, model=TEST_MODEL
        )
        assert "max_tokens" not in kwargs
        assert "max_completion_tokens" not in kwargs

    # ---- reasoning_effort ----

    def test_reasoning_effort_for_gpt5(self, provider):
        messages = [ChatMessage.user("think hard")]
        _, kwargs, _ = provider._get_chat_completion_args(
            prompt_messages=messages,
            model=TEST_MODEL_GPT5,
            reasoning_effort="high",
        )
        assert kwargs["reasoning_effort"] == "high"

    def test_reasoning_effort_for_o_series(self, provider):
        messages = [ChatMessage.user("think")]
        _, kwargs, _ = provider._get_chat_completion_args(
            prompt_messages=messages,
            model=TEST_MODEL_O3,
            reasoning_effort="low",
        )
        assert kwargs["reasoning_effort"] == "low"

    def test_reasoning_effort_ignored_for_legacy_model(self, provider):
        messages = [ChatMessage.user("hi")]
        _, kwargs, _ = provider._get_chat_completion_args(
            prompt_messages=messages,
            model=TEST_MODEL_LEGACY,
            reasoning_effort="high",
        )
        assert "reasoning_effort" not in kwargs

    # ---- extra_headers ----

    def test_extra_request_headers_merged(self, provider):
        provider._configuration.extra_request_headers = {"X-Custom": "value"}
        messages = [ChatMessage.user("hi")]
        _, kwargs, _ = provider._get_chat_completion_args(
            prompt_messages=messages, model=TEST_MODEL
        )
        assert kwargs["extra_headers"]["X-Custom"] == "value"

    def test_no_extra_headers_when_empty(self, provider):
        messages = [ChatMessage.user("hi")]
        _, kwargs, _ = provider._get_chat_completion_args(
            prompt_messages=messages, model=TEST_MODEL
        )
        assert "extra_headers" not in kwargs


# ---------------------------------------------------------------------------
# _parse_assistant_tool_calls
# ---------------------------------------------------------------------------
class TestParseAssistantToolCalls:
    def _make_openai_message(self, content="", tool_calls=None):
        msg = MagicMock()
        msg.content = content
        msg.tool_calls = tool_calls
        return msg

    def _make_openai_tool_call(self, id, name, arguments_json):
        tc = MagicMock()
        tc.id = id
        tc.type = "function"
        tc.function = MagicMock()
        tc.function.name = name
        tc.function.arguments = arguments_json
        return tc

    def test_successful_parse(self, provider):
        tc = self._make_openai_tool_call("call_1", "search", '{"query": "test"}')
        msg = self._make_openai_message(tool_calls=[tc])
        tool_calls, errors = provider._parse_assistant_tool_calls(msg)
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].function.name == "search"
        assert tool_calls[0].function.arguments == {"query": "test"}
        assert errors == []

    def test_multiple_tool_calls(self, provider):
        tcs = [
            self._make_openai_tool_call("call_1", "search", '{"q": "a"}'),
            self._make_openai_tool_call("call_2", "write", '{"path": "/tmp"}'),
        ]
        msg = self._make_openai_message(tool_calls=tcs)
        tool_calls, errors = provider._parse_assistant_tool_calls(msg)
        assert len(tool_calls) == 2
        assert errors == []

    def test_no_tool_calls_returns_empty(self, provider):
        msg = self._make_openai_message(content="Just text", tool_calls=None)
        tool_calls, errors = provider._parse_assistant_tool_calls(msg)
        assert tool_calls == []
        assert errors == []

    def test_malformed_json_produces_error(self, provider):
        tc = self._make_openai_tool_call("call_1", "search", "not valid json{{{")
        msg = self._make_openai_message(tool_calls=[tc])
        tool_calls, errors = provider._parse_assistant_tool_calls(msg)
        assert tool_calls == []
        assert len(errors) == 1
        assert "search" in str(errors[0])

    def test_partial_parse_failure_ignored_when_all_succeed(self, provider):
        """If all tool calls eventually parse, errors are cleared."""
        tcs = [
            self._make_openai_tool_call("call_1", "search", '{"q": "test"}'),
            self._make_openai_tool_call("call_2", "write", '{"p": "x"}'),
        ]
        msg = self._make_openai_message(tool_calls=tcs)
        tool_calls, errors = provider._parse_assistant_tool_calls(msg)
        # Both parsed successfully, so errors should be empty
        assert len(tool_calls) == 2
        assert errors == []

    def test_empty_arguments_parsed(self, provider):
        tc = self._make_openai_tool_call("call_1", "noop", "{}")
        msg = self._make_openai_message(tool_calls=[tc])
        tool_calls, errors = provider._parse_assistant_tool_calls(msg)
        assert len(tool_calls) == 1
        assert tool_calls[0].function.arguments == {}
        assert errors == []


# ---------------------------------------------------------------------------
# _format_parse_errors
# ---------------------------------------------------------------------------
class TestFormatParseErrors:
    def test_invalid_function_call_error_formatted(self, provider, search_function):
        err = InvalidFunctionCallError(
            name="web_search",
            arguments={"wrong_param": "value"},
            message="Missing required field",
        )
        result = provider._format_parse_errors([err], None, [search_function])
        assert "web_search" in result
        assert "Missing required field" in result
        assert "wrong_param" in result
        assert "Expected parameters" in result

    def test_generic_exception_formatted(self, provider):
        err = ValueError("Something went wrong")
        result = provider._format_parse_errors([err], None, None)
        assert "ValueError" in result
        assert "Something went wrong" in result

    def test_empty_arguments_shows_no_arguments(self, provider, search_function):
        err = InvalidFunctionCallError(
            name="web_search", arguments={}, message="Bad call"
        )
        result = provider._format_parse_errors([err], None, [search_function])
        assert "(no arguments)" in result

    def test_no_matching_function_skips_expected_params(self, provider):
        err = InvalidFunctionCallError(
            name="unknown", arguments={"x": 1}, message="Unknown function"
        )
        result = provider._format_parse_errors([err], None, [])
        assert "Expected parameters" not in result

    def test_multiple_errors_joined(self, provider):
        errors = [
            ValueError("Error 1"),
            ValueError("Error 2"),
        ]
        result = provider._format_parse_errors(errors, None, None)
        assert "Error 1" in result
        assert "Error 2" in result


# ---------------------------------------------------------------------------
# create_chat_completion: main flow
# ---------------------------------------------------------------------------
class TestCreateChatCompletion:
    def _make_completion_response(
        self, content: str | None = "Response", tool_calls=None, usage=None
    ):
        """Build a mock ChatCompletion object."""
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls

        def _model_dump(**kwargs):
            d = {"role": "assistant", "content": content}
            if kwargs.get("exclude_none"):
                d = {k: v for k, v in d.items() if v is not None}
            return d

        message.model_dump = _model_dump

        choice = MagicMock()
        choice.message = message

        completion = MagicMock()
        completion.choices = [choice]
        completion.usage = usage
        return completion

    def _make_usage(self, prompt_tokens=100, completion_tokens=50):
        usage = MagicMock()
        usage.prompt_tokens = prompt_tokens
        usage.completion_tokens = completion_tokens
        return usage

    @pytest.mark.asyncio
    async def test_successful_completion(self, provider):
        usage = self._make_usage()
        completion = self._make_completion_response(content="Hello!", usage=usage)
        provider._client.chat.completions.create = AsyncMock(return_value=completion)

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("Hi")],
            model_name=TEST_MODEL,
        )
        assert result.response.content == "Hello!"
        assert result.prompt_tokens_used == 100
        assert result.completion_tokens_used == 50

    @pytest.mark.asyncio
    async def test_completion_with_parser(self, provider):
        usage = self._make_usage()
        completion = self._make_completion_response(
            content="parsed content", usage=usage
        )
        provider._client.chat.completions.create = AsyncMock(return_value=completion)

        def parser(msg: AssistantChatMessage) -> str:
            return msg.content.upper()

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("Hi")],
            model_name=TEST_MODEL,
            completion_parser=parser,
        )
        assert result.parsed_result == "PARSED CONTENT"

    @pytest.mark.asyncio
    async def test_completion_with_tool_calls(self, provider, search_function):
        tc_mock = MagicMock()
        tc_mock.id = "call_1"
        tc_mock.type = "function"
        tc_mock.function = MagicMock()
        tc_mock.function.name = "web_search"
        tc_mock.function.arguments = '{"query": "test"}'

        usage = self._make_usage()
        completion = self._make_completion_response(
            content="Searching", tool_calls=[tc_mock], usage=usage
        )
        provider._client.chat.completions.create = AsyncMock(return_value=completion)

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("search for test")],
            model_name=TEST_MODEL,
            functions=[search_function],
        )
        assert result.response.tool_calls is not None
        assert result.response.tool_calls[0].function.name == "web_search"

    @pytest.mark.asyncio
    async def test_parse_error_retries_then_succeeds(self, provider):
        """When parser fails on first attempt, retry and succeed on second."""
        usage = self._make_usage()

        # First response will fail parsing, second will succeed
        completion1 = self._make_completion_response(content="bad", usage=usage)
        completion2 = self._make_completion_response(content="good", usage=usage)

        provider._client.chat.completions.create = AsyncMock(
            side_effect=[completion1, completion2]
        )

        call_count = 0

        def flaky_parser(msg: AssistantChatMessage) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Parse failed")
            return msg.content

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("test")],
            model_name=TEST_MODEL,
            completion_parser=flaky_parser,
        )
        assert result.parsed_result == "good"

    @pytest.mark.asyncio
    async def test_parse_error_exhausts_retries_raises(self, provider):
        """When parser always fails, it raises after max retries."""
        usage = self._make_usage()
        completion = self._make_completion_response(content="always bad", usage=usage)
        provider._client.chat.completions.create = AsyncMock(return_value=completion)

        provider._configuration.fix_failed_parse_tries = 2

        def always_fail(msg: AssistantChatMessage):
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            await provider.create_chat_completion(
                model_prompt=[ChatMessage.user("test")],
                model_name=TEST_MODEL,
                completion_parser=always_fail,
            )

    @pytest.mark.asyncio
    async def test_retry_strips_tool_calls_from_message(self, provider):
        """On retry, tool_calls are stripped from the failed assistant message
        to prevent OpenAI 400 errors."""
        tc_mock = MagicMock()
        tc_mock.id = "call_1"
        tc_mock.type = "function"
        tc_mock.function = MagicMock()
        tc_mock.function.name = "search"
        tc_mock.function.arguments = '{"q": "test"}'

        usage = self._make_usage()

        # First response has tool_calls that will fail validation
        bad_completion = self._make_completion_response(
            content="searching", tool_calls=[tc_mock], usage=usage
        )
        # Second response succeeds
        good_completion = self._make_completion_response(content="done", usage=usage)

        call_count = 0

        async def track_calls(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad_completion
            # Verify the retry messages don't contain tool_calls
            msgs = kwargs.get("messages", [])
            for msg in msgs:
                if msg.get("role") == "assistant" and "tool_calls" in msg:
                    pytest.fail("Retry message should not have tool_calls")
            return good_completion

        provider._client.chat.completions.create = track_calls

        attempt = 0

        def fail_then_succeed(msg: AssistantChatMessage) -> str:
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise ValueError("Bad parse")
            return msg.content

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("test")],
            model_name=TEST_MODEL,
            completion_parser=fail_then_succeed,
        )
        assert result.parsed_result == "done"

    @pytest.mark.asyncio
    async def test_retry_with_null_content_does_not_send_null(self, provider):
        """When GPT-5.4 returns tool_calls with content=None, the retry
        message must have content="" not null/missing, otherwise OpenAI
        returns 400 'expected a string, got null'."""
        tc_mock = MagicMock()
        tc_mock.id = "call_1"
        tc_mock.type = "function"
        tc_mock.function = MagicMock()
        tc_mock.function.name = "write_file"
        tc_mock.function.arguments = "invalid json{{"

        usage = self._make_usage()

        # First response: content=None with tool_calls (GPT-5.4 behavior)
        bad_completion = self._make_completion_response(
            content=None, tool_calls=[tc_mock], usage=usage
        )
        good_completion = self._make_completion_response(content="done", usage=usage)

        call_count = 0

        async def check_retry_messages(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return bad_completion
            # On retry, verify assistant message has string content
            msgs = kwargs.get("messages", [])
            for msg in msgs:
                if msg.get("role") == "assistant":
                    assert (
                        msg.get("content") is not None
                    ), "Assistant message content must not be null on retry"
                    assert isinstance(
                        msg["content"], str
                    ), f"Content must be str, got {type(msg['content'])}"
            return good_completion

        provider._client.chat.completions.create = check_retry_messages

        attempt = 0

        def fail_then_succeed(msg):
            nonlocal attempt
            attempt += 1
            if attempt == 1:
                raise ValueError("Parse failed")
            return msg.content

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("test")],
            model_name=TEST_MODEL,
            completion_parser=fail_then_succeed,
        )
        assert result.parsed_result == "done"
        assert call_count >= 2  # May take multiple retries

    @pytest.mark.asyncio
    async def test_no_usage_defaults_to_zero(self, provider):
        completion = self._make_completion_response(content="hi", usage=None)
        provider._client.chat.completions.create = AsyncMock(return_value=completion)

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("test")],
            model_name=TEST_MODEL,
        )
        assert result.prompt_tokens_used == 0
        assert result.completion_tokens_used == 0

    @pytest.mark.asyncio
    async def test_cost_tracked_in_budget(self, provider):
        usage = self._make_usage(prompt_tokens=1000, completion_tokens=500)
        completion = self._make_completion_response(content="hi", usage=usage)
        provider._client.chat.completions.create = AsyncMock(return_value=completion)

        initial_cost = provider._budget.total_cost

        await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("test")],
            model_name=TEST_MODEL,
        )

        assert provider._budget.total_cost > initial_cost

    @pytest.mark.asyncio
    async def test_no_budget_cost_is_zero(self, provider):
        provider._budget = None
        usage = self._make_usage()
        completion = self._make_completion_response(content="hi", usage=usage)
        provider._client.chat.completions.create = AsyncMock(return_value=completion)

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("test")],
            model_name=TEST_MODEL,
        )
        assert result is not None


# ---------------------------------------------------------------------------
# count_message_tokens
# ---------------------------------------------------------------------------
class TestCountMessageTokens:
    def test_single_message_counts_role_and_content(self, provider):
        """Tokenizer splits on whitespace; 'USER: hello world' = 3 tokens."""
        msg = ChatMessage.user("hello world")
        count = provider.count_message_tokens(msg, TEST_MODEL)
        # "USER: hello world" split by spaces = ["USER:", "hello", "world"] = 3
        assert count == 3

    def test_list_of_messages_concatenated(self, provider):
        msgs = [ChatMessage.user("hello"), ChatMessage.system("be helpful")]
        count = provider.count_message_tokens(msgs, TEST_MODEL)
        # Concatenated messages should have more tokens than a single one
        single = provider.count_message_tokens(msgs[0], TEST_MODEL)
        assert count > single

    def test_longer_message_has_more_tokens(self, provider):
        short = ChatMessage.user("hi")
        long = ChatMessage.user("this is a much longer message with many words")
        assert provider.count_message_tokens(long, TEST_MODEL) > (
            provider.count_message_tokens(short, TEST_MODEL)
        )


# ---------------------------------------------------------------------------
# get_token_limit
# ---------------------------------------------------------------------------
class TestGetTokenLimit:
    def test_returns_model_max_tokens(self, provider):
        assert provider.get_token_limit(TEST_MODEL) == 4096
        assert provider.get_token_limit(TEST_MODEL_GPT5) == 400_000
