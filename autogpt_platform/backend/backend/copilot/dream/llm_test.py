"""Regression tests for dream-pass LLM JSON handling.

The structured_completion wrapper requests JSON mode, but some OpenRouter
upstreams (Claude family, certain Gemini variants) still wrap responses in
```json ... ``` markdown fences. Without stripping them the dream pass
aborts on the consolidation step with "Expecting value: line 1 column 1". This file pins
the fence-stripper that prevents the regression.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from backend.copilot.transport_routing import ProviderRoutingKwargs
from backend.util.llm.providers import ProviderResponse

from .llm import (
    DreamLLMError,
    _extract_first_json_object,
    _normalize_ollama_host,
    _strip_json_code_fence,
    structured_completion,
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
