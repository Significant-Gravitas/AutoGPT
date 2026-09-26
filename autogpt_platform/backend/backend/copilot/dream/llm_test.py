"""Regression tests for dream-pass LLM JSON handling.

The structured_completion wrapper requests JSON mode, but some OpenRouter
upstreams (Claude family, certain Gemini variants) still wrap responses in
```json ... ``` markdown fences. Without stripping them the dream pass
aborts on the consolidation step with "Expecting value: line 1 column 1". This file pins
the fence-stripper that prevents the regression.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from backend.copilot.transport_routing import ProviderRoutingKwargs
from backend.util.llm.conversions import ToolCall, ToolContentBlock
from backend.util.llm.providers import ProviderResponse
from backend.util.llm.tool_use import auto_tool_choice, force_tool_choice

from .llm import (
    DreamLLMError,
    _extract_first_json_object,
    _normalize_ollama_host,
    _strip_json_code_fence,
    structured_completion,
)
from .structured_output import (
    OUTPUT_TOOL_CALL_ONCE,
    output_tool_name,
    structured_request,
    with_output_tool_instruction,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        # No fence — content passes through.
        ('{"a": 1}', '{"a": 1}'),
        # Fence with ```json tag (most common Claude / Gemini wrap).
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        # Fence without language tag.
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        # Trailing whitespace after closing fence.
        ('```json\n{"a": 1}\n```   ', '{"a": 1}'),
        # Multiline JSON inside fence — newlines inside the body preserved
        # by the parser; only the fence delimiters are removed.
        ('```json\n{\n  "a": 1\n}\n```', '{\n  "a": 1\n}'),
    ],
)
def test_strip_json_code_fence(raw: str, expected: str):
    assert _strip_json_code_fence(raw) == expected


def test_strip_json_code_fence_leaves_unfenced_content_alone():
    """A single-line response without fences must be returned verbatim."""
    raw = '{"facts": [{"content": "test"}]}'
    assert _strip_json_code_fence(raw) == raw


def test_strip_json_code_fence_handles_only_opening_fence():
    """If the model opened a fence but never closed it, drop the opener anyway
    so json.loads at least gets a chance to parse the body."""
    raw = '```json\n{"a": 1}'
    assert _strip_json_code_fence(raw) == '{"a": 1}'


def test_strip_json_code_fence_no_newline_after_opener_returns_raw():
    """Pathological case — opening fence with no newline before content. We
    leave it alone so json.loads surfaces the original parse error rather
    than masking it with a guess."""
    raw = "```json{}"
    assert _strip_json_code_fence(raw) == raw


@pytest.mark.parametrize(
    "raw,expected",
    [
        # JSON-only — start of string.
        ('{"a": 1}', '{"a": 1}'),
        # Prose prefix then JSON object.
        ('I\'ll analyze the proposals...\n\n{"writes": []}', '{"writes": []}'),
        # Prose prefix then JSON array.
        ("Here we go:\n[1, 2, 3]\ntrailing", "[1, 2, 3]"),
        # Nested braces inside the object — depth counted correctly.
        (
            'Sure thing:\n{"a": {"b": {"c": 1}}, "d": 2}\nThanks!',
            '{"a": {"b": {"c": 1}}, "d": 2}',
        ),
        # Braces inside strings must NOT throw off the depth count.
        (
            'before {"text": "this has } and { inside", "ok": true} after',
            '{"text": "this has } and { inside", "ok": true}',
        ),
        # Escaped quotes inside strings.
        (
            'pre {"a": "\\"quoted\\"", "b": 1} post',
            '{"a": "\\"quoted\\"", "b": 1}',
        ),
    ],
)
def test_extract_first_json_object(raw: str, expected: str):
    assert _extract_first_json_object(raw) == expected


def test_extract_first_json_object_returns_none_when_no_object():
    assert _extract_first_json_object("just some prose without any braces") is None


def test_extract_first_json_object_returns_none_when_unbalanced():
    """An opening brace with no matching close — fallback returns None
    instead of guessing, so the caller surfaces a real parse error
    rather than silently truncating."""
    assert _extract_first_json_object('{"a": "no closer') is None


# ---------------------------------------------------------------------------
# _normalize_ollama_host
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_url,expected",
    [
        # OpenAI-compat ``/v1`` suffix is stripped — ollama's native
        # client wants the raw host.
        ("http://localhost:11434/v1", "http://localhost:11434"),
        ("https://ollama.lan:11434/v1/", "https://ollama.lan:11434"),
        # No path → preserved.
        ("http://localhost:11434", "http://localhost:11434"),
        # Bare host:port (no scheme) — pass through unchanged.
        ("localhost:11434", "localhost:11434"),
        # Empty/None → fall back to default.
        ("", "localhost:11434"),
        (None, "localhost:11434"),
    ],
)
def test_normalize_ollama_host(base_url: str | None, expected: str):
    assert _normalize_ollama_host(base_url) == expected


# ---------------------------------------------------------------------------
# structured_completion delegation contract
# ---------------------------------------------------------------------------


class _SampleFact(BaseModel):
    content: str
    confidence: float


class _SampleOutput(BaseModel):
    facts: list[_SampleFact]


def _openrouter_routing(api_key: str = "sk-or-test") -> ProviderRoutingKwargs:
    return ProviderRoutingKwargs(
        provider="open_router",
        api_key=api_key,
        base_url=None,
        supports_flex=True,
        cost_log_provider="open_router",
    )


def _ollama_routing(
    base_url: str = "http://localhost:11434/v1",
) -> ProviderRoutingKwargs:
    return ProviderRoutingKwargs(
        provider="ollama",
        api_key="ollama-placeholder",
        base_url=base_url,
        supports_flex=False,
        cost_log_provider="ollama",
    )


def _anthropic_routing(api_key: str = "") -> ProviderRoutingKwargs:
    return ProviderRoutingKwargs(
        provider="anthropic",
        api_key=api_key,
        base_url=None,
        supports_flex=False,
        cost_log_provider="anthropic",
    )


class TestStructuredCompletionDelegation:
    """Confirms ``structured_completion`` delegates to ``call_provider``
    with kwargs derived from ``routing_kwargs_for_chat_transport()`` and
    converts ``ProviderResponse`` → the dream's typed Pydantic +
    ``CompletionUsage`` shape, regardless of which transport is active."""

    @pytest.mark.asyncio
    async def test_delegates_to_call_provider_with_openrouter_args(self):
        fake_response = ProviderResponse(
            content='{"facts": [{"content": "x", "confidence": 0.9}]}',
            prompt_tokens=12,
            completion_tokens=4,
            cost_usd=0.0042,
        )
        call_provider_mock = AsyncMock(return_value=fake_response)

        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_openrouter_routing(),
        ), patch("backend.copilot.dream.llm.call_provider", call_provider_mock):
            result = await structured_completion(
                model="anthropic/claude-sonnet-4-6",
                messages=[
                    {"role": "system", "content": "you are helpful"},
                    {"role": "user", "content": "give me a fact"},
                ],
                response_model=_SampleOutput,
                temperature=0.3,
                max_output_tokens=512,
            )

        # Result shape is the dream's typed wrapper
        assert isinstance(result.value, _SampleOutput)
        assert len(result.value.facts) == 1
        assert result.value.facts[0].confidence == 0.9
        # Cost + tokens flow through
        assert result.usage.cost_usd == 0.0042
        assert result.usage.input_tokens == 12
        assert result.usage.output_tokens == 4
        # Delegation routed to OpenRouter with the right knobs
        call_provider_mock.assert_awaited_once()
        kwargs = call_provider_mock.call_args.kwargs
        assert kwargs["provider"] == "open_router"
        assert kwargs["api_key"] == "sk-or-test"
        assert kwargs["model"] == "anthropic/claude-sonnet-4-6"
        assert kwargs["max_tokens"] == 512
        assert kwargs["temperature"] == 0.3
        # JSON-mode must be requested — without it OpenRouter→Claude
        # falls back to free-form text and our prose parser does extra
        # work.
        assert kwargs["force_json_output"] is True
        # The forced tool is the native Anthropic path's device only.
        assert kwargs["tools"] is None
        assert kwargs["tool_choice"] is None

    @pytest.mark.asyncio
    async def test_caller_supplied_phase_timeout_reaches_call_provider(self):
        """Recombine/sanitize carry 16384-token output budgets because
        real responses exceed 8192 tokens — at real decode speeds those
        responses take far longer than the shared 120s default, so the
        orchestrator passes a per-phase wall-clock budget. The wrapper
        must thread it through to ``call_provider`` verbatim, otherwise
        the token-cap raise is dead letter: any long response dies on
        TimeoutError instead of finishing."""
        fake = ProviderResponse(
            content='{"facts": [{"content": "x", "confidence": 0.9}]}',
            prompt_tokens=1,
            completion_tokens=1,
        )
        call_provider_mock = AsyncMock(return_value=fake)

        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_openrouter_routing(),
        ), patch("backend.copilot.dream.llm.call_provider", call_provider_mock):
            await structured_completion(
                model="anthropic/claude-opus-4-7",
                messages=[{"role": "user", "content": "hi"}],
                response_model=_SampleOutput,
                timeout_seconds=600,
            )

        assert call_provider_mock.call_args.kwargs["timeout_seconds"] == 600

    @pytest.mark.asyncio
    async def test_omitted_timeout_defers_to_call_provider(self):
        """Callers that don't pass a phase budget (none in-tree, but the
        signature allows it) keep the conservative shared default rather
        than an unbounded request."""
        fake = ProviderResponse(
            content='{"facts": [{"content": "x", "confidence": 0.9}]}',
            prompt_tokens=1,
            completion_tokens=1,
        )
        call_provider_mock = AsyncMock(return_value=fake)

        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_openrouter_routing(),
        ), patch("backend.copilot.dream.llm.call_provider", call_provider_mock):
            await structured_completion(
                model="anthropic/claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                response_model=_SampleOutput,
            )

        # Forwards the sentinel rather than a def-time snapshot of the setting,
        # so call_provider resolves the live value — same contract as every
        # other entry point.
        assert call_provider_mock.call_args.kwargs["timeout_seconds"] is None

    @pytest.mark.asyncio
    async def test_routes_to_ollama_under_local_transport(self):
        """Local transport: dispatch becomes ``provider="ollama"`` with
        ``ollama_host`` derived from ``CHAT_BASE_URL`` (sans the
        OpenAI-compat ``/v1`` suffix). Closes the dream-pass-on-local
        hole identified in the local-AI memory integration."""
        fake_response = ProviderResponse(
            content='{"facts": [{"content": "y", "confidence": 0.5}]}',
            prompt_tokens=8,
            completion_tokens=3,
        )
        call_provider_mock = AsyncMock(return_value=fake_response)

        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_ollama_routing("http://localhost:11434/v1"),
        ), patch("backend.copilot.dream.llm.call_provider", call_provider_mock):
            result = await structured_completion(
                model="hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M",
                messages=[{"role": "user", "content": "hi"}],
                response_model=_SampleOutput,
            )

        assert isinstance(result.value, _SampleOutput)
        kwargs = call_provider_mock.call_args.kwargs
        assert kwargs["provider"] == "ollama"
        # Empty api_key is fine for ollama; the placeholder ChatConfig
        # would have set is passed through verbatim.
        assert kwargs["api_key"] == "ollama-placeholder"
        # ``/v1`` is the OpenAI-compat suffix; ollama's native client
        # wants the raw host, so the wrapper strips it.
        assert kwargs["ollama_host"] == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_raises_when_no_openrouter_key(self):
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_openrouter_routing(api_key=""),
        ):
            with pytest.raises(DreamLLMError, match="OPEN_ROUTER_API_KEY"):
                await structured_completion(
                    model="anthropic/claude-sonnet-4-6",
                    messages=[{"role": "user", "content": "hi"}],
                    response_model=_SampleOutput,
                )

    @pytest.mark.asyncio
    async def test_raises_with_subscription_hint_when_no_anthropic_key(self):
        """Subscription mode lands on ``provider="anthropic"`` with an
        empty api_key (the Claude Code OAuth token can't authenticate
        against the Messages API per the Feb-2026 Anthropic ToS). The
        wrapper must surface a friendly error pointing operators at
        the ``ANTHROPIC_API_KEY`` env var instead of leaking a 401."""
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_anthropic_routing(api_key=""),
        ):
            with pytest.raises(DreamLLMError, match="ANTHROPIC_API_KEY") as exc_info:
                await structured_completion(
                    model="anthropic/claude-sonnet-4-6",
                    messages=[{"role": "user", "content": "hi"}],
                    response_model=_SampleOutput,
                )
        # The hint is user-facing self-serve documentation — the anchor must
        # match the real heading slug in docs/platform/copilot-local-llm.md
        # ("### Subscription mode caveat").
        assert "docs/platform/copilot-local-llm.md#subscription-mode-caveat" in str(
            exc_info.value
        )

    @pytest.mark.asyncio
    async def test_empty_content_raises_dream_llm_error(self):
        fake = ProviderResponse(content="", prompt_tokens=1, completion_tokens=0)
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_openrouter_routing(),
        ), patch(
            "backend.copilot.dream.llm.call_provider",
            new=AsyncMock(return_value=fake),
        ):
            with pytest.raises(DreamLLMError, match="empty"):
                await structured_completion(
                    model="x",
                    messages=[{"role": "user", "content": "hi"}],
                    response_model=_SampleOutput,
                )

    @pytest.mark.asyncio
    async def test_invalid_pydantic_shape_raises_dream_llm_error(self):
        """Pydantic validation lives in this wrapper, not in
        ``call_provider`` — pin the boundary so a future schema change
        doesn't accidentally drop validation."""
        fake = ProviderResponse(
            content='{"facts": [{"content": 1.0, "confidence": "wrong"}]}',
            prompt_tokens=1,
            completion_tokens=1,
        )
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_openrouter_routing(),
        ), patch(
            "backend.copilot.dream.llm.call_provider",
            new=AsyncMock(return_value=fake),
        ):
            with pytest.raises(DreamLLMError, match="did not match"):
                await structured_completion(
                    model="x",
                    messages=[{"role": "user", "content": "hi"}],
                    response_model=_SampleOutput,
                )

    @pytest.mark.asyncio
    async def test_provider_failure_is_wrapped_in_dream_llm_error(self):
        """``call_provider`` raising must surface as ``DreamLLMError`` so
        the orchestrator's per-phase failure handler triggers — not as a
        raw RuntimeError that crashes the pass."""
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_openrouter_routing(),
        ), patch(
            "backend.copilot.dream.llm.call_provider",
            new=AsyncMock(side_effect=RuntimeError("upstream 502")),
        ):
            with pytest.raises(DreamLLMError, match="upstream 502"):
                await structured_completion(
                    model="x",
                    messages=[{"role": "user", "content": "hi"}],
                    response_model=_SampleOutput,
                )

    @pytest.mark.asyncio
    async def test_recovers_from_prose_prefixed_json(self):
        """Even when the strengthened system prompts hold, occasional
        model outputs still slip prose before the JSON. The wrapper's
        balanced-brace fallback should recover so the phase doesn't
        fail unnecessarily."""
        fake = ProviderResponse(
            content=(
                "Looking at the inputs, I need to ...\n\n"
                '{"facts": [{"content": "recovered", "confidence": 1.0}]}'
            ),
            prompt_tokens=10,
            completion_tokens=20,
        )
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_openrouter_routing(),
        ), patch(
            "backend.copilot.dream.llm.call_provider",
            new=AsyncMock(return_value=fake),
        ):
            result = await structured_completion(
                model="x",
                messages=[{"role": "user", "content": "hi"}],
                response_model=_SampleOutput,
            )
        assert result.value.facts[0].content == "recovered"


# ---------------------------------------------------------------------------
# Native Anthropic sync path: forced tool instead of JSON mode
# ---------------------------------------------------------------------------


def _bad_request(message: str) -> anthropic.BadRequestError:
    response = httpx.Response(
        400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    return anthropic.BadRequestError(message, response=response, body=None)


# Anthropic's 400 for a forced ``tool_choice`` on a model that takes none.
_FORCED_TOOL_REJECTION = (
    'tool_choice: type "tool" and "any" are not supported for this model.'
)


def _tool_response(arguments: str, **usage: int) -> ProviderResponse:
    """What ``call_provider`` returns for a forced tool call: ``content``
    is the tool NAME, the JSON is in the tool call's arguments."""
    name = output_tool_name(_SampleOutput)
    return ProviderResponse(
        content=name,
        prompt_tokens=usage.get("prompt_tokens", 1),
        completion_tokens=usage.get("completion_tokens", 1),
        cache_read_tokens=usage.get("cache_read_tokens", 0),
        tool_calls=[
            ToolContentBlock(
                id="toolu_1",
                type="tool_use",
                function=ToolCall(name=name, arguments=arguments),
            )
        ],
    )


class TestAnthropicToolPath:
    """The native Anthropic API ignores JSON mode, so structured output
    comes from one forced tool built from the response model, read back
    from the tool call, with the model in its native spelling."""

    @pytest.mark.asyncio
    async def test_forces_one_tool_and_sends_the_native_model(self):
        fake = _tool_response(
            '{"facts": [{"content": "x", "confidence": 0.9}]}',
            prompt_tokens=12,
            completion_tokens=4,
            cache_read_tokens=3,
        )
        call_provider_mock = AsyncMock(return_value=fake)

        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_anthropic_routing(api_key="sk-ant-test"),
        ), patch("backend.copilot.dream.llm.call_provider", call_provider_mock):
            result = await structured_completion(
                model="anthropic/claude-sonnet-5",
                messages=[{"role": "user", "content": "give me a fact"}],
                response_model=_SampleOutput,
            )

        assert result.value.facts[0].content == "x"
        kwargs = call_provider_mock.call_args.kwargs
        assert kwargs["provider"] == "anthropic"
        assert kwargs["model"] == "claude-sonnet-5"
        assert kwargs["force_json_output"] is False
        tool_name = output_tool_name(_SampleOutput)
        assert [tool["name"] for tool in kwargs["tools"]] == [tool_name]
        assert kwargs["tools"][0]["input_schema"]["required"] == ["facts"]
        assert kwargs["tool_choice"] == force_tool_choice(tool_name)
        # A forced tool needs no prompt line asking for it.
        assert kwargs["messages"] == [{"role": "user", "content": "give me a fact"}]
        # Usage names the model that was called, the spelling the price
        # card resolves, with its tokens.
        assert result.usage.model == "claude-sonnet-5"
        assert (result.usage.input_tokens, result.usage.cache_read_tokens) == (12, 3)

    @pytest.mark.asyncio
    async def test_opus_5_5_gets_auto_and_the_prompt_asks_for_the_call(self):
        """Opus 5.5 answers a forced ``tool_choice`` with a 400, so the tool
        goes out under ``auto``, and both its description and the last user
        turn ask for one call with the complete result."""
        fake = _tool_response('{"facts": [{"content": "x", "confidence": 0.9}]}')
        call_provider_mock = AsyncMock(return_value=fake)

        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_anthropic_routing(api_key="sk-ant-test"),
        ), patch("backend.copilot.dream.llm.call_provider", call_provider_mock):
            result = await structured_completion(
                model="anthropic/claude-opus-5.5",
                messages=[
                    {"role": "system", "content": "you consolidate facts"},
                    {"role": "user", "content": "give me a fact"},
                ],
                response_model=_SampleOutput,
            )

        assert result.value.facts[0].content == "x"
        kwargs = call_provider_mock.call_args.kwargs
        tool_name = output_tool_name(_SampleOutput)
        assert kwargs["model"] == "claude-opus-5-5"
        assert kwargs["tool_choice"] == auto_tool_choice()
        assert kwargs["tools"][0]["description"].endswith(OUTPUT_TOOL_CALL_ONCE)
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
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_anthropic_routing(api_key="sk-ant-test"),
        ), patch(
            "backend.copilot.dream.llm.call_provider",
            new=AsyncMock(return_value=fake),
        ):
            result = await structured_completion(
                model="claude-opus-5.5",
                messages=[{"role": "user", "content": "hi"}],
                response_model=_SampleOutput,
            )
        assert result.value.facts[0].content == "text"
        assert result.usage.model == "claude-opus-5-5"

    @pytest.mark.asyncio
    async def test_tool_arguments_off_schema_raise_with_usage(self):
        """The tool call was billed even when its arguments don't fit."""
        fake = _tool_response('{"facts": "not a list"}', prompt_tokens=7)
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_anthropic_routing(api_key="sk-ant-test"),
        ), patch(
            "backend.copilot.dream.llm.call_provider",
            new=AsyncMock(return_value=fake),
        ):
            with pytest.raises(DreamLLMError, match="did not match") as exc_info:
                await structured_completion(
                    model="anthropic/claude-sonnet-5",
                    messages=[{"role": "user", "content": "hi"}],
                    response_model=_SampleOutput,
                )
        assert exc_info.value.usage is not None
        assert exc_info.value.usage.input_tokens == 7

    @pytest.mark.asyncio
    async def test_non_anthropic_model_fails_before_any_call(self):
        call_provider_mock = AsyncMock()
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_anthropic_routing(api_key="sk-ant-test"),
        ), patch("backend.copilot.dream.llm.call_provider", call_provider_mock):
            with pytest.raises(DreamLLMError, match="requires an Anthropic model"):
                await structured_completion(
                    model="openai/gpt-4.1-mini",
                    messages=[{"role": "user", "content": "hi"}],
                    response_model=_SampleOutput,
                )
        call_provider_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_through_the_real_anthropic_messages_call(self):
        """End to end through ``call_provider``: the tool reaches the
        Messages API carrying the schema, and its tool_use block comes back
        as the parsed value."""
        tool_name = output_tool_name(_SampleOutput)
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
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_anthropic_routing(api_key="sk-ant-test"),
        ), patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=client,
        ):
            result = await structured_completion(
                model="anthropic/claude-sonnet-5",
                messages=[
                    {"role": "system", "content": "you consolidate facts"},
                    {"role": "user", "content": "facts"},
                ],
                response_model=_SampleOutput,
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
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_anthropic_routing(api_key="sk-ant-test"),
        ), patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=client,
        ):
            result = await structured_completion(
                model="anthropic/claude-opus-5.5",
                messages=[
                    {"role": "system", "content": "you consolidate facts"},
                    {"role": "user", "content": "facts"},
                ],
                response_model=_SampleOutput,
            )

        assert result.value.facts[0].content == "prose"
        assert result.usage.model == "claude-opus-5-5"
        assert result.usage.output_tokens == 5
        sent = create.call_args.kwargs
        assert sent["model"] == "claude-opus-5-5"
        assert sent["tool_choice"] == auto_tool_choice()
        assert output_tool_name(_SampleOutput) in sent["messages"][-1]["content"]

    @pytest.mark.asyncio
    async def test_forced_tool_rejection_retries_once_with_auto(self, caplog):
        """A model missing from the forced-tool list answers the forced
        choice with a 400. The call goes once more under ``auto``, the
        prompt asking for the tool, and the swap is logged as a warning."""
        fake = _tool_response('{"facts": [{"content": "healed", "confidence": 0.7}]}')
        call_provider_mock = AsyncMock(
            side_effect=[_bad_request(_FORCED_TOOL_REJECTION), fake]
        )

        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_anthropic_routing(api_key="sk-ant-test"),
        ), patch(
            "backend.copilot.dream.llm.call_provider", call_provider_mock
        ), caplog.at_level(
            logging.WARNING, logger="backend.copilot.dream.llm"
        ):
            result = await structured_completion(
                model="anthropic/claude-sonnet-5",
                messages=[{"role": "user", "content": "hi"}],
                response_model=_SampleOutput,
            )

        assert result.value.facts[0].content == "healed"
        tool_name = output_tool_name(_SampleOutput)
        first, second = call_provider_mock.call_args_list
        assert first.kwargs["tool_choice"] == force_tool_choice(tool_name)
        assert first.kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert second.kwargs["tool_choice"] == auto_tool_choice()
        assert second.kwargs["model"] == "claude-sonnet-5"
        assert tool_name in second.kwargs["messages"][-1]["content"]
        assert "retrying once with tool_choice=auto" in caplog.text

    @pytest.mark.asyncio
    async def test_forced_tool_rejection_is_retried_only_once(self):
        call_provider_mock = AsyncMock(
            side_effect=[
                _bad_request(_FORCED_TOOL_REJECTION),
                _bad_request(_FORCED_TOOL_REJECTION),
            ]
        )
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_anthropic_routing(api_key="sk-ant-test"),
        ), patch("backend.copilot.dream.llm.call_provider", call_provider_mock):
            with pytest.raises(DreamLLMError, match="tool_choice") as exc_info:
                await structured_completion(
                    model="anthropic/claude-sonnet-5",
                    messages=[{"role": "user", "content": "hi"}],
                    response_model=_SampleOutput,
                )
        assert call_provider_mock.await_count == 2
        # Neither call came back with a response, so nothing was billed.
        assert exc_info.value.usage is None

    @pytest.mark.asyncio
    async def test_other_bad_requests_are_not_retried(self):
        call_provider_mock = AsyncMock(
            side_effect=_bad_request("messages: at least one message is required")
        )
        with patch(
            "backend.copilot.dream.llm.routing_kwargs_for_chat_transport",
            return_value=_anthropic_routing(api_key="sk-ant-test"),
        ), patch("backend.copilot.dream.llm.call_provider", call_provider_mock):
            with pytest.raises(DreamLLMError, match="at least one message"):
                await structured_completion(
                    model="anthropic/claude-sonnet-5",
                    messages=[{"role": "user", "content": "hi"}],
                    response_model=_SampleOutput,
                )
        assert call_provider_mock.await_count == 1


class TestStructuredRequest:
    """Which structured-output device each provider and model gets."""

    def test_opus_5_5_offers_the_tool_under_auto(self):
        request = structured_request(
            "anthropic", "anthropic/claude-opus-5.5", _SampleOutput
        )
        assert request.model == "claude-opus-5-5"
        assert request.tool_choice == auto_tool_choice()
        assert not request.forces_output_tool

    def test_sonnet_5_forces_the_tool_and_leaves_the_prompt_alone(self):
        request = structured_request(
            "anthropic", "anthropic/claude-sonnet-5", _SampleOutput
        )
        assert request.tool_choice == force_tool_choice(output_tool_name(_SampleOutput))
        messages = [{"role": "user", "content": "hi"}]
        assert request.prompt(messages) is messages

    def test_json_mode_providers_get_no_tool(self):
        request = structured_request(
            "open_router", "anthropic/claude-opus-5.5", _SampleOutput
        )
        assert request.force_json_output
        assert request.tools is None and request.tool_choice is None
        messages = [{"role": "user", "content": "hi"}]
        assert request.prompt(messages) is messages

    def test_instruction_gets_its_own_turn_after_a_non_user_message(self):
        messages = [{"role": "system", "content": "sys"}]
        prompted = with_output_tool_instruction(messages, "emit_x")
        assert [m["role"] for m in prompted] == ["system", "user"]
        assert "emit_x" in prompted[-1]["content"]
        assert messages == [{"role": "system", "content": "sys"}]
