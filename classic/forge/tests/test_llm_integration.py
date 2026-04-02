"""Live integration tests against real LLM APIs.

These tests make actual API calls and cost real money. They are skipped
unless the relevant API key is set in the environment. Run explicitly with:

    OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... \
        poetry run pytest forge/tests/test_llm_integration.py -v

Each test is cheap (~100-500 tokens) but validates the full round-trip:
message prep → API call → response parsing → tool call handling.
"""

import json
import os
from typing import Any

import pytest

from forge.llm.providers.schema import (
    AssistantChatMessage,
    ChatMessage,
    CompletionModelFunction,
)
from forge.models.json_schema import JSONSchema

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------
HAS_OPENAI_KEY = bool(os.environ.get("OPENAI_API_KEY"))
HAS_ANTHROPIC_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))

skip_no_openai = pytest.mark.skipif(not HAS_OPENAI_KEY, reason="OPENAI_API_KEY not set")
skip_no_anthropic = pytest.mark.skipif(
    not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set"
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
SIMPLE_FUNCTION = CompletionModelFunction(
    name="get_weather",
    description="Get the current weather for a city",
    parameters={
        "city": JSONSchema(
            type=JSONSchema.Type.STRING,
            description="City name",
            required=True,
        ),
    },
)


def _parse_to_dict(msg: AssistantChatMessage) -> dict[str, Any]:
    """Simple parser that extracts text and tool calls."""
    result: dict[str, Any] = {"content": msg.content}
    if msg.tool_calls:
        result["tool_calls"] = [
            {"name": tc.function.name, "arguments": tc.function.arguments}
            for tc in msg.tool_calls
        ]
    return result


# ---------------------------------------------------------------------------
# OpenAI integration tests
# ---------------------------------------------------------------------------
@skip_no_openai
class TestOpenAIIntegration:
    """Live tests against OpenAI API."""

    @pytest.fixture
    def provider(self):
        from forge.llm.providers.openai import OpenAIProvider

        return OpenAIProvider()

    @pytest.mark.asyncio
    async def test_simple_completion(self, provider):
        """Basic text completion round-trip."""
        from forge.llm.providers.openai import OpenAIModelName

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("Reply with exactly: PONG")],
            model_name=OpenAIModelName.GPT4_O_MINI,
            completion_parser=_parse_to_dict,
            max_output_tokens=50,
        )
        assert "PONG" in result.parsed_result["content"].upper()
        assert result.prompt_tokens_used > 0
        assert result.completion_tokens_used > 0

    @pytest.mark.asyncio
    async def test_tool_call_completion(self, provider):
        """Tool call round-trip — the main bug area."""
        from forge.llm.providers.openai import OpenAIModelName

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("What's the weather in Paris?")],
            model_name=OpenAIModelName.GPT4_O_MINI,
            completion_parser=_parse_to_dict,
            functions=[SIMPLE_FUNCTION],
            max_output_tokens=100,
        )
        parsed = result.parsed_result
        assert "tool_calls" in parsed
        assert len(parsed["tool_calls"]) >= 1
        tc = parsed["tool_calls"][0]
        assert tc["name"] == "get_weather"
        assert isinstance(tc["arguments"], dict)
        assert "city" in tc["arguments"]

    @pytest.mark.asyncio
    async def test_gpt5_text_completion(self, provider):
        """GPT-5 class model — validates no-text-content handling."""
        from forge.llm.providers.openai import OpenAIModelName

        # Use the cheapest GPT-5 variant
        model = OpenAIModelName.GPT5_NANO
        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("Reply with exactly: HELLO")],
            model_name=model,
            completion_parser=_parse_to_dict,
            max_output_tokens=50,
        )
        assert result.parsed_result is not None

    @pytest.mark.asyncio
    @pytest.mark.flaky(reruns=2)
    async def test_gpt5_tool_call(self, provider):
        """GPT-5 with tool calls — the exact scenario that broke GPT-5.2."""
        from forge.llm.providers.openai import OpenAIModelName

        model = OpenAIModelName.GPT5_MINI
        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("What's the weather in Tokyo?")],
            model_name=model,
            completion_parser=_parse_to_dict,
            functions=[SIMPLE_FUNCTION],
            max_output_tokens=100,
        )
        parsed = result.parsed_result
        assert "tool_calls" in parsed
        tc = parsed["tool_calls"][0]
        assert tc["name"] == "get_weather"
        assert isinstance(tc["arguments"], dict)

    @pytest.mark.asyncio
    async def test_conversation_with_tool_history(self, provider):
        """Multi-turn with tool calls in history — the GPT-5.2 400 bug."""
        from forge.llm.providers.openai import OpenAIModelName
        from forge.llm.providers.schema import ToolResultMessage

        model = OpenAIModelName.GPT4_O_MINI

        # First call: get a tool call
        r1 = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("What's the weather in London?")],
            model_name=model,
            completion_parser=_parse_to_dict,
            functions=[SIMPLE_FUNCTION],
            max_output_tokens=100,
        )
        assert r1.response.tool_calls

        # Build history with tool call + result
        history = [
            ChatMessage.user("What's the weather in London?"),
            r1.response,
            ToolResultMessage(
                tool_call_id=r1.response.tool_calls[0].id,
                content=json.dumps({"temperature": 15, "condition": "cloudy"}),
            ),
            ChatMessage.user("Thanks! Now summarize that in one sentence."),
        ]

        # Second call with history — this is where the 400 error happened
        r2 = await provider.create_chat_completion(
            model_prompt=history,
            model_name=model,
            completion_parser=_parse_to_dict,
            max_output_tokens=100,
        )
        assert r2.parsed_result["content"]  # Should have text response


# ---------------------------------------------------------------------------
# Anthropic integration tests
# ---------------------------------------------------------------------------
@skip_no_anthropic
class TestAnthropicIntegration:
    """Live tests against Anthropic API."""

    @pytest.fixture
    def provider(self):
        from forge.llm.providers.anthropic import AnthropicProvider

        return AnthropicProvider()

    @pytest.mark.asyncio
    async def test_simple_completion(self, provider):
        """Basic text completion round-trip."""
        from forge.llm.providers.anthropic import AnthropicModelName

        result = await provider.create_chat_completion(
            model_prompt=[
                ChatMessage.system("You are helpful."),
                ChatMessage.user("Reply with exactly: PONG"),
            ],
            model_name=AnthropicModelName.CLAUDE4_5_HAIKU_v1,
            completion_parser=_parse_to_dict,
            max_output_tokens=50,
        )
        assert "PONG" in result.parsed_result["content"].upper()
        assert result.prompt_tokens_used > 0

    @pytest.mark.asyncio
    async def test_tool_call_completion(self, provider):
        """Tool call round-trip."""
        from forge.llm.providers.anthropic import AnthropicModelName

        result = await provider.create_chat_completion(
            model_prompt=[
                ChatMessage.system("Use the get_weather tool to answer."),
                ChatMessage.user("What's the weather in Berlin?"),
            ],
            model_name=AnthropicModelName.CLAUDE4_5_HAIKU_v1,
            completion_parser=_parse_to_dict,
            functions=[SIMPLE_FUNCTION],
            max_output_tokens=200,
        )
        parsed = result.parsed_result
        assert "tool_calls" in parsed
        tc = parsed["tool_calls"][0]
        assert tc["name"] == "get_weather"
        assert isinstance(tc["arguments"], dict)

    @pytest.mark.asyncio
    async def test_conversation_with_tool_history(self, provider):
        """Multi-turn with tool calls in history."""
        from forge.llm.providers.anthropic import AnthropicModelName
        from forge.llm.providers.schema import ToolResultMessage

        model = AnthropicModelName.CLAUDE4_5_HAIKU_v1

        r1 = await provider.create_chat_completion(
            model_prompt=[
                ChatMessage.system("Use tools when asked about weather."),
                ChatMessage.user("What's the weather in Sydney?"),
            ],
            model_name=model,
            completion_parser=_parse_to_dict,
            functions=[SIMPLE_FUNCTION],
            max_output_tokens=200,
        )
        assert r1.response.tool_calls

        history = [
            ChatMessage.system("Use tools when asked about weather."),
            ChatMessage.user("What's the weather in Sydney?"),
            r1.response,
            ToolResultMessage(
                tool_call_id=r1.response.tool_calls[0].id,
                content=json.dumps({"temperature": 22, "condition": "sunny"}),
            ),
            ChatMessage.user("Summarize that in one sentence."),
        ]

        r2 = await provider.create_chat_completion(
            model_prompt=history,
            model_name=model,
            completion_parser=_parse_to_dict,
            max_output_tokens=100,
        )
        assert r2.parsed_result["content"]

    @pytest.mark.asyncio
    async def test_token_counting_returns_positive(self, provider):
        """Verify token counting actually works (was returning 0 before fix)."""
        from forge.llm.providers.anthropic import AnthropicModelName

        count = provider.count_tokens(
            "This is a test sentence for token counting.",
            AnthropicModelName.CLAUDE4_5_HAIKU_v1,
        )
        assert count > 5  # Should be ~9 tokens


# ---------------------------------------------------------------------------
# MultiProvider integration tests
# ---------------------------------------------------------------------------
class TestMultiProviderIntegration:
    """Tests that go through the MultiProvider routing layer."""

    @pytest.fixture
    def provider(self):
        from forge.llm.providers.multi import MultiProvider

        return MultiProvider()

    @skip_no_openai
    @pytest.mark.asyncio
    async def test_routes_openai_model(self, provider):
        from forge.llm.providers.openai import OpenAIModelName

        result = await provider.create_chat_completion(
            model_prompt=[ChatMessage.user("Reply with: OK")],
            model_name=OpenAIModelName.GPT4_O_MINI,
            completion_parser=_parse_to_dict,
            max_output_tokens=10,
        )
        assert result.parsed_result["content"]

    @skip_no_anthropic
    @pytest.mark.asyncio
    async def test_routes_anthropic_model(self, provider):
        from forge.llm.providers.anthropic import AnthropicModelName

        result = await provider.create_chat_completion(
            model_prompt=[
                ChatMessage.system("Be brief."),
                ChatMessage.user("Reply with: OK"),
            ],
            model_name=AnthropicModelName.CLAUDE4_5_HAIKU_v1,
            completion_parser=_parse_to_dict,
            max_output_tokens=10,
        )
        assert result.parsed_result["content"]
