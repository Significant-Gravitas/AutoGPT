"""``structured_complete`` on the native Anthropic API: JSON mode is ignored
there, so the output comes from one tool built from the response model, read
back from the tool call, in the model's native spelling. Forced where the
model accepts that; left to the model (``auto``) with a prompt line asking
for the call where it doesn't (Opus 5.5).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anthropic
import pytest

from backend.copilot.dream.structured_output import output_tool_name
from backend.util.llm.providers import ProviderResponse
from backend.util.llm.tool_use import auto_tool_choice, force_tool_choice

from ._test_data import (
    ANTHROPIC,
    CALL_PROVIDER,
    MESSAGES,
    ONE_FACT,
    ROUTING,
    SampleOutput,
    sync_ctx,
    tool_response,
)
from .complete import structured_complete
from .context import InferenceError


class TestAnthropicToolPath:
    """The native Anthropic API ignores JSON mode, so structured output
    comes from one forced tool built from the response model, read back
    from the tool call, with the model in its native spelling."""

    @pytest.mark.asyncio
    async def test_forces_one_tool_and_sends_the_native_model(self):
        fake = tool_response(
            ONE_FACT, prompt_tokens=12, completion_tokens=4, cache_read_tokens=3
        )
        call_provider = AsyncMock(return_value=fake)
        with patch(ROUTING, return_value=ANTHROPIC), patch(
            CALL_PROVIDER, call_provider
        ):
            result = await structured_complete(
                sync_ctx("anthropic", "claude-sonnet-5"), MESSAGES, SampleOutput
            )

        assert result.value.facts[0].content == "x"
        kwargs = call_provider.call_args.kwargs
        assert kwargs["provider"] == "anthropic"
        assert kwargs["model"] == "claude-sonnet-5"
        assert kwargs["force_json_output"] is False
        tool_name = output_tool_name(SampleOutput)
        assert [tool["name"] for tool in kwargs["tools"]] == [tool_name]
        assert kwargs["tools"][0]["input_schema"]["required"] == ["facts"]
        assert kwargs["tool_choice"] == force_tool_choice(tool_name)
        assert kwargs["tools"][0]["description"] == (
            "Return the SampleOutput result: call this once, with every "
            "field the schema requires."
        )
        # A forced tool needs no prompt line asking for it.
        assert kwargs["messages"] == MESSAGES
        # Usage names the model that was called, the spelling the price card
        # resolves, with its tokens.
        assert result.usage.model == "claude-sonnet-5"
        assert (result.usage.input_tokens, result.usage.cache_read_tokens) == (12, 3)

    @pytest.mark.asyncio
    async def test_opus_5_5_gets_auto_and_the_prompt_asks_for_the_call(self):
        """Opus 5.5 answers a forced ``tool_choice`` with a 400, so the tool
        goes out under ``auto``, and both its description and the last user
        turn ask for one call with the complete result."""
        call_provider = AsyncMock(return_value=tool_response(ONE_FACT))
        with patch(ROUTING, return_value=ANTHROPIC), patch(
            CALL_PROVIDER, call_provider
        ):
            result = await structured_complete(
                sync_ctx("anthropic", "claude-opus-5-5"),
                [
                    {"role": "system", "content": "you consolidate facts"},
                    {"role": "user", "content": "give me a fact"},
                ],
                SampleOutput,
            )

        assert result.value.facts[0].content == "x"
        kwargs = call_provider.call_args.kwargs
        tool_name = output_tool_name(SampleOutput)
        assert kwargs["model"] == "claude-opus-5-5"
        assert kwargs["tool_choice"] == auto_tool_choice()
        # The sync description asks for one complete call in either mode.
        assert "call this once" in kwargs["tools"][0]["description"]
        system, user = kwargs["messages"]
        assert system == {"role": "system", "content": "you consolidate facts"}
        assert user["role"] == "user"
        assert user["content"].startswith("give me a fact\n\n")
        assert tool_name in user["content"]

    @pytest.mark.asyncio
    async def test_parses_the_message_text_when_no_tool_was_called(self):
        fake = ProviderResponse(
            content='{"facts": [{"content": "text", "confidence": 1.0}]}',
            prompt_tokens=1,
            completion_tokens=1,
        )
        with patch(ROUTING, return_value=ANTHROPIC), patch(
            CALL_PROVIDER, AsyncMock(return_value=fake)
        ):
            result = await structured_complete(
                sync_ctx("anthropic", "claude-opus-5.5"), MESSAGES, SampleOutput
            )
        assert result.value.facts[0].content == "text"
        assert result.usage.model == "claude-opus-5-5"

    @pytest.mark.asyncio
    async def test_tool_arguments_off_schema_raise_with_usage(self):
        """The tool call was billed even when its arguments don't fit."""
        fake = tool_response('{"facts": "not a list"}', prompt_tokens=7)
        with patch(ROUTING, return_value=ANTHROPIC), patch(
            CALL_PROVIDER, AsyncMock(return_value=fake)
        ):
            with pytest.raises(InferenceError, match="did not match") as exc_info:
                await structured_complete(
                    sync_ctx("anthropic", "claude-sonnet-5"), MESSAGES, SampleOutput
                )
        assert exc_info.value.usage is not None
        assert exc_info.value.usage.input_tokens == 7

    @pytest.mark.asyncio
    async def test_non_anthropic_model_fails_before_any_call(self):
        call_provider = AsyncMock()
        with patch(ROUTING, return_value=ANTHROPIC), patch(
            CALL_PROVIDER, call_provider
        ):
            with pytest.raises(InferenceError, match="requires an Anthropic model"):
                await structured_complete(
                    sync_ctx("anthropic", "openai/gpt-4.1-mini"), MESSAGES, SampleOutput
                )
        call_provider.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_through_the_real_anthropic_messages_call(self):
        """End to end through ``call_provider``: the tool reaches the Messages
        API carrying the schema, and its tool_use block comes back as the
        parsed value."""
        tool_name = output_tool_name(SampleOutput)
        message = anthropic.types.Message(
            id="msg-1",
            type="message",
            role="assistant",
            model="claude-sonnet-5",
            content=[
                anthropic.types.ToolUseBlock(
                    type="tool_use",
                    id="toolu_1",
                    name=tool_name,
                    input={"facts": [{"content": "e2e", "confidence": 0.5}]},
                )
            ],
            stop_reason="tool_use",
            usage=anthropic.types.Usage(input_tokens=20, output_tokens=6),
        )
        create = AsyncMock(return_value=message)
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        with patch(ROUTING, return_value=ANTHROPIC), patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=client,
        ):
            result = await structured_complete(
                sync_ctx("anthropic", "claude-sonnet-5"),
                [
                    {"role": "system", "content": "you consolidate facts"},
                    {"role": "user", "content": "facts"},
                ],
                SampleOutput,
            )

        assert result.value.facts[0].content == "e2e"
        assert result.usage.output_tokens == 6
        sent = create.call_args.kwargs
        assert sent["model"] == "claude-sonnet-5"
        assert sent["tool_choice"] == force_tool_choice(tool_name)
        (tool,) = sent["tools"]
        assert set(tool["input_schema"]["properties"]) == {"facts"}
        assert tool["input_schema"]["required"] == ["facts"]

    @pytest.mark.asyncio
    async def test_a_text_answer_under_auto_still_parses(self):
        """End to end on Opus 5.5: ``auto`` goes out, the model answers in
        text rather than calling the tool, and the JSON in that text
        (fenced, behind a line of prose) is still the parsed value."""
        message = anthropic.types.Message(
            id="msg-2",
            type="message",
            role="assistant",
            model="claude-opus-5-5",
            content=[
                anthropic.types.TextBlock(
                    type="text",
                    text=(
                        "Here are the facts:\n```json\n"
                        '{"facts": [{"content": "prose", "confidence": 0.4}]}'
                        "\n```"
                    ),
                )
            ],
            stop_reason="end_turn",
            usage=anthropic.types.Usage(input_tokens=9, output_tokens=5),
        )
        create = AsyncMock(return_value=message)
        client = SimpleNamespace(messages=SimpleNamespace(create=create))
        with patch(ROUTING, return_value=ANTHROPIC), patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=client,
        ):
            result = await structured_complete(
                sync_ctx("anthropic", "claude-opus-5-5"),
                [
                    {"role": "system", "content": "you consolidate facts"},
                    {"role": "user", "content": "facts"},
                ],
                SampleOutput,
            )

        assert result.value.facts[0].content == "prose"
        assert result.usage.model == "claude-opus-5-5"
        assert result.usage.output_tokens == 5
        sent = create.call_args.kwargs
        assert sent["model"] == "claude-opus-5-5"
        assert sent["tool_choice"] == auto_tool_choice()
        assert output_tool_name(SampleOutput) in sent["messages"][-1]["content"]
