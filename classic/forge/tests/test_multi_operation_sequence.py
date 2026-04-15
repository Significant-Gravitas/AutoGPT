"""Multi-operation sequence tests for every configured LLM model.

These tests exercise the real production code paths by running 3 sequential
chat completion operations per model, which catches bugs like:
- Message history corruption between calls (the GPT-5.2 tool_calls bug)
- Cost accumulation errors across calls
- State leakage between operations
- Incorrect max_tokens/max_completion_tokens branching per model
"""

import json
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from forge.llm.providers._openai_base import BaseOpenAIChatProvider
from forge.llm.providers.anthropic import (
    ANTHROPIC_CHAT_MODELS,
    AnthropicCredentials,
    AnthropicModelName,
    AnthropicProvider,
    AnthropicSettings,
)
from forge.llm.providers.multi import CHAT_MODELS
from forge.llm.providers.openai import OPEN_AI_CHAT_MODELS, OpenAIModelName
from forge.llm.providers.schema import (
    ChatMessage,
    CompletionModelFunction,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
    ToolResultMessage,
)
from forge.models.json_schema import JSONSchema

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
SEARCH_FUNCTION = CompletionModelFunction(
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

WRITE_FUNCTION = CompletionModelFunction(
    name="write_file",
    description="Write a file",
    parameters={
        "path": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="File path",
            required=True,
        ),
        "content": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="Content",
            required=True,
        ),
    },
)


# ---------------------------------------------------------------------------
# OpenAI-like provider for testing all OpenAI model configs
# ---------------------------------------------------------------------------
class _DummyCredentials(ModelProviderCredentials):
    api_key: Any = "test-key"

    def get_api_access_kwargs(self) -> dict:
        return {"api_key": "test-key"}


class _DummySettings(ModelProviderSettings):
    credentials: Optional[ModelProviderCredentials] = None  # type: ignore[assignment]
    budget: Optional[ModelProviderBudget] = None


class _TestOpenAIProvider(BaseOpenAIChatProvider):
    MODELS = OPEN_AI_CHAT_MODELS
    CHAT_MODELS = OPEN_AI_CHAT_MODELS  # type: ignore

    default_settings = _DummySettings(
        name="test",
        description="Test",
        configuration=ModelProviderConfiguration(),
        credentials=_DummyCredentials(),
        budget=ModelProviderBudget(),
    )

    def __init__(self):
        self._settings = self.default_settings.model_copy(deep=True)
        self._configuration = self._settings.configuration
        self._credentials = self._settings.credentials
        self._budget = self._settings.budget
        self._logger = MagicMock()
        self._client = MagicMock()

    def get_tokenizer(self, model_name) -> ModelTokenizer:
        tok = MagicMock()
        tok.encode = lambda text: text.split()
        return tok


def _make_openai_completion(
    content="Response", tool_calls=None, prompt_tok=100, completion_tok=50
):
    """Build a mock ChatCompletion matching the OpenAI SDK structure."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    message.model_dump = MagicMock(
        return_value={"role": "assistant", "content": content}
    )

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = prompt_tok
    usage.completion_tokens = completion_tok

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    return completion


def _make_openai_tool_call(call_id, name, arguments):
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


# ---------------------------------------------------------------------------
# Collect all OpenAI-like models for parameterized testing
# ---------------------------------------------------------------------------
# Pick representative models from each family to keep test time reasonable
OPENAI_TEST_MODELS = [
    OpenAIModelName.GPT3,
    OpenAIModelName.GPT4,
    OpenAIModelName.GPT4_TURBO,
    OpenAIModelName.GPT4_O,
    OpenAIModelName.GPT4_1,
    OpenAIModelName.O1,
    OpenAIModelName.O3_MINI,
    OpenAIModelName.O4_MINI,
    OpenAIModelName.GPT5,
    OpenAIModelName.GPT5_2,
    OpenAIModelName.GPT5_3,
    OpenAIModelName.GPT5_4,
    OpenAIModelName.GPT5_MINI,
    OpenAIModelName.GPT5_NANO,
    OpenAIModelName.GPT5_PRO,
    OpenAIModelName.GPT5_3_PRO,
    OpenAIModelName.GPT5_4_PRO,
]


@pytest.fixture
def openai_provider():
    return _TestOpenAIProvider()


class TestOpenAIMultiOperationSequence:
    """Run 3 sequential operations for each OpenAI model,
    verifying message history, cost tracking, and tool calls don't corrupt state."""

    @pytest.mark.parametrize("model_name", OPENAI_TEST_MODELS, ids=str)
    @pytest.mark.asyncio
    async def test_three_text_completions_in_sequence(
        self, openai_provider, model_name
    ):
        """3 plain text completions — costs accumulate, no state leaks."""
        responses = [
            _make_openai_completion(
                f"Response {i}", prompt_tok=100 + i * 10, completion_tok=50
            )
            for i in range(3)
        ]
        openai_provider._client.chat.completions.create = AsyncMock(
            side_effect=responses
        )

        results = []
        for i in range(3):
            result = await openai_provider.create_chat_completion(
                model_prompt=[ChatMessage.user(f"Message {i}")],
                model_name=model_name,
            )
            results.append(result)

        # Each call should return distinct content
        assert results[0].response.content == "Response 0"
        assert results[1].response.content == "Response 1"
        assert results[2].response.content == "Response 2"

        # Cost should accumulate across all 3 calls
        assert openai_provider._budget.total_cost > 0
        assert openai_provider._budget.usage.prompt_tokens == sum(
            100 + i * 10 for i in range(3)
        )

    @pytest.mark.parametrize("model_name", OPENAI_TEST_MODELS, ids=str)
    @pytest.mark.asyncio
    async def test_three_tool_call_completions_in_sequence(
        self, openai_provider, model_name
    ):
        """3 sequential tool calls — the exact bug pattern that broke GPT-5.2.

        Each response has tool_calls. The key thing: when we feed history from
        one call into the next, the tool_calls arguments must be JSON strings
        not dicts, and tool response messages must follow tool_call messages.
        """
        responses = []
        for i in range(3):
            tc = _make_openai_tool_call(
                f"call_{i}", "web_search", {"query": f"search {i}"}
            )
            responses.append(_make_openai_completion(f"Searching {i}", tool_calls=[tc]))
        openai_provider._client.chat.completions.create = AsyncMock(
            side_effect=responses
        )

        for i in range(3):
            result = await openai_provider.create_chat_completion(
                model_prompt=[ChatMessage.user(f"Search for {i}")],
                model_name=model_name,
                functions=[SEARCH_FUNCTION],
            )
            assert result.response.tool_calls is not None
            assert result.response.tool_calls[0].function.name == "web_search"
            assert result.response.tool_calls[0].function.arguments == {
                "query": f"search {i}"
            }

    @pytest.mark.parametrize("model_name", OPENAI_TEST_MODELS, ids=str)
    @pytest.mark.asyncio
    async def test_mixed_operations_text_then_tool_then_text(
        self, openai_provider, model_name
    ):
        """Text -> tool call -> text: different response types in sequence."""
        tc = _make_openai_tool_call(
            "call_1", "write_file", {"path": "/tmp/f", "content": "data"}
        )
        responses = [
            _make_openai_completion("Plain response"),
            _make_openai_completion("Writing file", tool_calls=[tc]),
            _make_openai_completion("Done"),
        ]
        openai_provider._client.chat.completions.create = AsyncMock(
            side_effect=responses
        )

        # Op 1: plain text
        r1 = await openai_provider.create_chat_completion(
            model_prompt=[ChatMessage.user("hello")],
            model_name=model_name,
        )
        assert r1.response.tool_calls is None

        # Op 2: tool call
        r2 = await openai_provider.create_chat_completion(
            model_prompt=[ChatMessage.user("write a file")],
            model_name=model_name,
            functions=[WRITE_FUNCTION],
        )
        assert r2.response.tool_calls is not None

        # Op 3: plain text again
        r3 = await openai_provider.create_chat_completion(
            model_prompt=[ChatMessage.user("done")],
            model_name=model_name,
        )
        assert r3.response.content == "Done"

    @pytest.mark.parametrize("model_name", OPENAI_TEST_MODELS, ids=str)
    @pytest.mark.asyncio
    async def test_conversation_history_with_tool_calls_serializes_correctly(
        self, openai_provider, model_name
    ):
        """Simulate building up a conversation with tool calls in history.

        This is the EXACT scenario that caused the GPT-5.2 400 error:
        tool_calls arguments must be JSON strings when sent back to the API.
        """
        # First call
        tc = _make_openai_tool_call("call_1", "web_search", {"query": "AI news"})
        openai_provider._client.chat.completions.create = AsyncMock(
            return_value=_make_openai_completion("Searching", tool_calls=[tc])
        )

        r1 = await openai_provider.create_chat_completion(
            model_prompt=[ChatMessage.user("search for AI news")],
            model_name=model_name,
            functions=[SEARCH_FUNCTION],
        )

        # Build conversation history including the tool call and its result
        history = [
            ChatMessage.user("search for AI news"),
            r1.response,  # AssistantChatMessage with tool_calls
            ToolResultMessage(tool_call_id="call_1", content="Found 10 results"),
            ChatMessage.user("summarize the results"),
        ]

        # Second call with history — this is where the bug occurred
        openai_provider._client.chat.completions.create = AsyncMock(
            return_value=_make_openai_completion("Here is a summary of AI news...")
        )

        # This call must prep messages correctly: tool_calls args as JSON strings
        prepped_msgs, kwargs, _ = openai_provider._get_chat_completion_args(
            prompt_messages=history,
            model=model_name,
        )

        # Verify the assistant message's tool_calls have string arguments
        assistant_msg = next(m for m in prepped_msgs if m["role"] == "assistant")
        for tc_dict in assistant_msg.get("tool_calls", []):
            args = tc_dict["function"]["arguments"]
            assert isinstance(args, str), (
                f"Model {model_name}: tool_calls arguments must be JSON string, "
                f"got {type(args).__name__}: {args}"
            )
            # Must be valid JSON
            parsed = json.loads(args)
            assert isinstance(parsed, dict)

        # The tool result message must be present
        tool_msg = next(m for m in prepped_msgs if m.get("role") == "tool")
        assert tool_msg["tool_call_id"] == "call_1"

    @pytest.mark.parametrize("model_name", OPENAI_TEST_MODELS, ids=str)
    def test_max_tokens_parameter_correct_for_model(self, openai_provider, model_name):
        """Each model must use the right max_tokens parameter name."""
        messages = [ChatMessage.user("hi")]
        _, kwargs, _ = openai_provider._get_chat_completion_args(
            prompt_messages=messages,
            model=model_name,
            max_output_tokens=1000,
        )

        # model_name is a str enum — use .value for string operations,
        # matching how the production code uses model.startswith()
        model_val = model_name.value
        uses_new_param = (
            model_val.startswith("o1")
            or model_val.startswith("o3")
            or model_val.startswith("o4")
            or model_val.startswith("gpt-5")
            or model_val.startswith("gpt-4.1")
            or model_val.startswith("gpt-4o")
        )

        if uses_new_param:
            assert (
                kwargs.get("max_completion_tokens") == 1000
            ), f"{model_name} should use max_completion_tokens"
            assert "max_tokens" not in kwargs
        else:
            assert (
                kwargs.get("max_tokens") == 1000
            ), f"{model_name} should use max_tokens"
            assert "max_completion_tokens" not in kwargs


# ---------------------------------------------------------------------------
# Anthropic multi-operation sequence
# ---------------------------------------------------------------------------
ANTHROPIC_TEST_MODELS = [
    AnthropicModelName.CLAUDE3_HAIKU_v1,
    AnthropicModelName.CLAUDE3_5_SONNET_v2,
    AnthropicModelName.CLAUDE4_SONNET_v1,
    AnthropicModelName.CLAUDE4_OPUS_v1,
    AnthropicModelName.CLAUDE4_5_OPUS_v1,
    AnthropicModelName.CLAUDE4_6_OPUS_v1,
]


def _make_anthropic_response(
    text="Response", tool_use_blocks=None, input_tok=100, output_tok=50
):
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
    response.usage.input_tokens = input_tok
    response.usage.output_tokens = output_tok
    response.model_dump = MagicMock(
        return_value={
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        }
    )
    return response


def _make_anthropic_tool_use(tool_id, name, input_data):
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data
    return block


@pytest.fixture
def anthropic_provider():
    from unittest.mock import patch

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


class TestAnthropicMultiOperationSequence:
    @pytest.mark.parametrize("model_name", ANTHROPIC_TEST_MODELS, ids=str)
    @pytest.mark.asyncio
    async def test_three_text_completions_in_sequence(
        self, anthropic_provider, model_name
    ):
        responses = [
            _make_anthropic_response(f"Response {i}", input_tok=100 + i * 10)
            for i in range(3)
        ]
        anthropic_provider._client.messages.create = AsyncMock(side_effect=responses)

        results = []
        for i in range(3):
            result = await anthropic_provider.create_chat_completion(
                model_prompt=[ChatMessage.user(f"Message {i}")],
                model_name=model_name,
            )
            results.append(result)

        assert results[0].response.content == "Response 0"
        assert results[1].response.content == "Response 1"
        assert results[2].response.content == "Response 2"
        assert anthropic_provider._budget.total_cost > 0

    @pytest.mark.parametrize("model_name", ANTHROPIC_TEST_MODELS, ids=str)
    @pytest.mark.asyncio
    async def test_three_tool_call_completions_in_sequence(
        self, anthropic_provider, model_name
    ):
        responses = []
        for i in range(3):
            tool_block = _make_anthropic_tool_use(
                f"tool_{i}", "web_search", {"query": f"q{i}"}
            )
            responses.append(
                _make_anthropic_response(f"Searching {i}", tool_use_blocks=[tool_block])
            )
        anthropic_provider._client.messages.create = AsyncMock(side_effect=responses)

        for i in range(3):
            result = await anthropic_provider.create_chat_completion(
                model_prompt=[ChatMessage.user(f"Search {i}")],
                model_name=model_name,
                functions=[SEARCH_FUNCTION],
            )
            assert result.response.tool_calls is not None
            assert result.response.tool_calls[0].function.name == "web_search"

    @pytest.mark.parametrize("model_name", ANTHROPIC_TEST_MODELS, ids=str)
    @pytest.mark.asyncio
    async def test_conversation_with_tool_history(self, anthropic_provider, model_name):
        """Build conversation history with tool calls, then make another call.
        Verifies Anthropic message format stays correct across operations."""
        # First call returns a tool use
        tool_block = _make_anthropic_tool_use("tool_1", "web_search", {"query": "test"})
        anthropic_provider._client.messages.create = AsyncMock(
            return_value=_make_anthropic_response(
                "Searching", tool_use_blocks=[tool_block]
            )
        )

        r1 = await anthropic_provider.create_chat_completion(
            model_prompt=[ChatMessage.user("search test")],
            model_name=model_name,
            functions=[SEARCH_FUNCTION],
        )

        # Build history for second call
        history = [
            ChatMessage.user("search test"),
            r1.response,
            ToolResultMessage(tool_call_id="tool_1", content="Found results"),
            ChatMessage.user("summarize"),
        ]

        # Verify message prep handles this history correctly
        anthropic_msgs, kwargs = anthropic_provider._get_chat_completion_args(
            prompt_messages=history, functions=[SEARCH_FUNCTION]
        )

        # Should have: user, assistant (with tool_use), user (with tool_result), user
        roles = [m["role"] for m in anthropic_msgs]
        assert "assistant" in roles
        assert "user" in roles

        # Assistant message should have tool_use blocks
        assistant_msg = next(m for m in anthropic_msgs if m["role"] == "assistant")
        assert isinstance(assistant_msg["content"], list)
        tool_use_blocks = [
            b for b in assistant_msg["content"] if b["type"] == "tool_use"
        ]
        assert len(tool_use_blocks) == 1
        assert tool_use_blocks[0]["name"] == "web_search"

    @pytest.mark.parametrize("model_name", ANTHROPIC_TEST_MODELS, ids=str)
    @pytest.mark.asyncio
    async def test_cost_accumulates_correctly_across_operations(
        self, anthropic_provider, model_name
    ):
        """Cost tracking must work correctly across 3 sequential calls."""
        model_info = ANTHROPIC_CHAT_MODELS[model_name]
        responses = [
            _make_anthropic_response(f"R{i}", input_tok=1000, output_tok=500)
            for i in range(3)
        ]
        anthropic_provider._client.messages.create = AsyncMock(side_effect=responses)

        for i in range(3):
            await anthropic_provider.create_chat_completion(
                model_prompt=[ChatMessage.user(f"M{i}")],
                model_name=model_name,
            )

        expected_cost = 3 * (
            1000 * model_info.prompt_token_cost + 500 * model_info.completion_token_cost
        )
        assert anthropic_provider._budget.total_cost == pytest.approx(expected_cost)
        assert anthropic_provider._budget.usage.prompt_tokens == 3000
        assert anthropic_provider._budget.usage.completion_tokens == 1500


# ---------------------------------------------------------------------------
# Cross-model: verify all registered models have consistent configs
# ---------------------------------------------------------------------------
class TestAllRegisteredModelsConsistency:
    def test_every_model_has_positive_cost_or_zero(self):
        """No model should have negative costs."""
        for name, info in CHAT_MODELS.items():
            assert info.prompt_token_cost >= 0, f"{name} has negative prompt cost"
            assert (
                info.completion_token_cost >= 0
            ), f"{name} has negative completion cost"

    def test_completion_cost_greater_or_equal_to_prompt_cost(self):
        """For all models, output tokens should cost >= input tokens."""
        for name, info in CHAT_MODELS.items():
            assert info.completion_token_cost >= info.prompt_token_cost, (
                f"{name}: completion cost ({info.completion_token_cost}) < "
                f"prompt cost ({info.prompt_token_cost})"
            )

    def test_all_non_llamafile_models_have_function_call_api(self):
        """All models except Llamafile's Mistral should support function calling."""
        for name, info in CHAT_MODELS.items():
            if info.provider_name == ModelProviderName.LLAMAFILE:
                continue  # Llamafile Mistral intentionally lacks function calling
            assert (
                info.has_function_call_api is True
            ), f"{name} does not support function calling"
