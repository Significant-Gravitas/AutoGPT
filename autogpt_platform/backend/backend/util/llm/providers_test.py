"""Tests for ``util.llm.providers.call_provider``.

These tests mock each provider's SDK at its construction point so the
test suite never makes a real network call. The goal is to lock in
that ``call_provider`` dispatches to the right SDK with the right
arguments, and that the normalized ``ProviderResponse`` carries the
fields each provider exposes.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.llm.providers import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ProviderResponse,
    call_provider,
)


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


# ---------------------------------------------------------------------------
# Execution mode gating
# ---------------------------------------------------------------------------


class TestExecutionModeStubs:
    """Batch + flex modes raise NotImplementedError until Steps 4 / 8 land.

    Pinning this behavior keeps the contract honest: callers that opt in
    early get a loud failure, not silent fallback to sync.
    """

    @pytest.mark.asyncio
    async def test_batch_mode_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Step 4"):
            await call_provider(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                execution_mode="batch",
            )

    @pytest.mark.asyncio
    async def test_flex_mode_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="Step 8"):
            await call_provider(
                provider="openai",
                model="gpt-4o",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                execution_mode="flex",
            )


# ---------------------------------------------------------------------------
# Unsupported provider
# ---------------------------------------------------------------------------


class TestUnsupportedProvider:
    @pytest.mark.asyncio
    async def test_raises_value_error_with_provider_name(self):
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            await call_provider(
                provider="made_up_provider",  # type: ignore[arg-type]
                model="x",
                api_key="x",
                messages=[_msg("user", "hi")],
                max_tokens=10,
            )


# ---------------------------------------------------------------------------
# OpenAI Responses API
# ---------------------------------------------------------------------------


class TestOpenAIResponses:
    @pytest.mark.asyncio
    async def test_dispatches_to_openai_responses_create(self):
        fake_response = SimpleNamespace()  # extractors are also mocked

        async_create = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(responses=SimpleNamespace(create=async_create))

        with (
            patch(
                "backend.util.llm.providers.openai.AsyncOpenAI",
                return_value=fake_client,
            ),
            patch(
                "backend.util.llm.providers.extract_responses_tool_calls",
                return_value=None,
            ),
            patch(
                "backend.util.llm.providers.extract_responses_content",
                return_value="hello world",
            ),
            patch(
                "backend.util.llm.providers.extract_responses_reasoning",
                return_value=None,
            ),
            patch(
                "backend.util.llm.providers.extract_responses_usage",
                return_value=(11, 22),
            ),
        ):
            result = await call_provider(
                provider="openai",
                model="gpt-4o",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=100,
            )

        assert isinstance(result, ProviderResponse)
        assert result.content == "hello world"
        assert result.prompt_tokens == 11
        assert result.completion_tokens == 22
        async_create.assert_awaited_once()
        kwargs = async_create.call_args.kwargs
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["max_output_tokens"] == 100
        assert kwargs["store"] is False


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------


class TestAnthropicMessages:
    @pytest.mark.asyncio
    async def test_dispatches_to_anthropic_messages_create_and_normalizes_usage(
        self,
    ):
        fake_usage = SimpleNamespace(
            input_tokens=5,
            output_tokens=7,
            cache_read_input_tokens=3,
            cache_creation_input_tokens=2,
        )
        fake_text_block = SimpleNamespace(type="text", text="ok")
        fake_response = SimpleNamespace(
            content=[fake_text_block],
            usage=fake_usage,
            stop_reason="end_turn",
        )

        async_create = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=async_create))

        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            result = await call_provider(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="sk-test",
                messages=[
                    _msg("system", "you are helpful"),
                    _msg("user", "hi"),
                ],
                max_tokens=200,
                temperature=0.3,
            )

        assert isinstance(result, ProviderResponse)
        assert result.content == "ok"
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 7
        assert result.cache_read_tokens == 3
        assert result.cache_creation_tokens == 2

        kwargs = async_create.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-6"
        assert kwargs["max_tokens"] == 200
        assert kwargs["temperature"] == 0.3
        # System prompt wrapped in a cache-controlled text block.
        assert isinstance(kwargs["system"], list)
        assert kwargs["system"][0]["text"] == "you are helpful"
        assert kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_omits_system_field_for_whitespace_only_prompt(self):
        """Anthropic rejects empty text blocks (HTTP 400). For
        whitespace-only system content we must omit the field entirely."""
        fake_usage = SimpleNamespace(input_tokens=1, output_tokens=1)
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="x")],
            usage=fake_usage,
            stop_reason="end_turn",
        )
        async_create = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=async_create))
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            await call_provider(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="sk-test",
                messages=[_msg("system", "   "), _msg("user", "hi")],
                max_tokens=10,
            )
        assert "system" not in async_create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_passes_tool_choice_when_provided(self):
        """Tool-use forced output relies on tool_choice making it into
        create_kwargs. If this is dropped, the model is free to emit prose."""
        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="x")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )
        async_create = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=async_create))
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            await call_provider(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                tool_choice={"type": "tool", "name": "submit_dream_ops"},
            )
        assert async_create.call_args.kwargs["tool_choice"] == {
            "type": "tool",
            "name": "submit_dream_ops",
        }

    @pytest.mark.asyncio
    async def test_raises_when_content_empty(self):
        fake_response = SimpleNamespace(
            content=[],
            usage=SimpleNamespace(input_tokens=1, output_tokens=0),
            stop_reason="end_turn",
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=SimpleNamespace(
                messages=SimpleNamespace(create=AsyncMock(return_value=fake_response))
            ),
        ):
            with pytest.raises(ValueError, match="No content"):
                await call_provider(
                    provider="anthropic",
                    model="claude-sonnet-4-6",
                    api_key="sk-test",
                    messages=[_msg("user", "hi")],
                    max_tokens=10,
                )


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------


class TestGroq:
    @pytest.mark.asyncio
    async def test_rejects_tools_param(self):
        """Groq SDK doesn't support tool calling; pin the error so we
        don't silently strip tools and surprise callers."""
        with pytest.raises(ValueError, match="Groq does not support tools"):
            await call_provider(
                provider="groq",
                model="mixtral-8x7b",
                api_key="x",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                tools=[{"name": "x", "parameters": {}}],
            )

    @pytest.mark.asyncio
    async def test_dispatches_with_response_format_when_force_json(self):
        fake_usage = SimpleNamespace(prompt_tokens=4, completion_tokens=6)
        fake_message = SimpleNamespace(content="{}")
        fake_choice = SimpleNamespace(message=fake_message)
        fake_response = SimpleNamespace(choices=[fake_choice], usage=fake_usage)
        async_create = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=async_create))
        )
        with patch(
            "groq.AsyncGroq",
            return_value=fake_client,
        ):
            result = await call_provider(
                provider="groq",
                model="mixtral-8x7b",
                api_key="x",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                force_json_output=True,
            )

        assert result.content == "{}"
        assert result.prompt_tokens == 4
        assert result.completion_tokens == 6
        kwargs = async_create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}


# ---------------------------------------------------------------------------
# OpenAI-compat (OpenRouter, Llama API, AI/ML, v0)
# ---------------------------------------------------------------------------


def _fake_openai_chat_response(
    content: str, prompt: int, completion: int, *, cost: float | None = None
) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        model_extra={"cost": cost} if cost is not None else None,
    )
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


class TestOpenRouter:
    @pytest.mark.asyncio
    async def test_dispatches_with_openrouter_extras(self):
        fake_response = _fake_openai_chat_response(
            "hello", prompt=10, completion=5, cost=0.0123
        )
        async_create = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=async_create))
        )
        with (
            patch(
                "backend.util.llm.providers.openai.AsyncOpenAI",
                return_value=fake_client,
            ),
            patch(
                "backend.util.llm.providers.extract_openai_reasoning",
                return_value=None,
            ),
            patch(
                "backend.util.llm.providers.extract_openai_tool_calls",
                return_value=None,
            ),
            patch(
                "backend.util.llm.providers.extract_openrouter_cost",
                return_value=0.0123,
            ),
        ):
            result = await call_provider(
                provider="open_router",
                model="anthropic/claude-sonnet-4-6",
                api_key="sk-or-test",
                messages=[_msg("user", "hi")],
                max_tokens=100,
            )

        assert result.content == "hello"
        assert result.cost_usd == 0.0123
        kwargs = async_create.call_args.kwargs
        # OpenRouter-specific extras must be present so the response
        # carries cost data.
        assert kwargs["extra_body"] == {"usage": {"include": True}}
        assert "HTTP-Referer" in kwargs["extra_headers"]


class TestLlamaAPI:
    @pytest.mark.asyncio
    async def test_omits_openrouter_extras(self):
        fake_response = _fake_openai_chat_response("x", prompt=1, completion=1)
        async_create = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=async_create))
        )
        with (
            patch(
                "backend.util.llm.providers.openai.AsyncOpenAI",
                return_value=fake_client,
            ),
            patch(
                "backend.util.llm.providers.extract_openai_reasoning",
                return_value=None,
            ),
            patch(
                "backend.util.llm.providers.extract_openai_tool_calls",
                return_value=None,
            ),
        ):
            await call_provider(
                provider="llama_api",
                model="llama-3-70b",
                api_key="x",
                messages=[_msg("user", "hi")],
                max_tokens=10,
            )
        kwargs = async_create.call_args.kwargs
        # Llama API path doesn't ask for openrouter cost extras
        assert "extra_body" not in kwargs


class TestAIMLAPI:
    @pytest.mark.asyncio
    async def test_dispatches_with_default_headers(self):
        fake_response = _fake_openai_chat_response("x", prompt=1, completion=1)
        async_create = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=async_create))
        )
        constructor = MagicMock(return_value=fake_client)
        with (
            patch(
                "backend.util.llm.providers.openai.AsyncOpenAI",
                constructor,
            ),
            patch(
                "backend.util.llm.providers.extract_openai_reasoning",
                return_value=None,
            ),
            patch(
                "backend.util.llm.providers.extract_openai_tool_calls",
                return_value=None,
            ),
        ):
            await call_provider(
                provider="aiml_api",
                model="anything",
                api_key="x",
                messages=[_msg("user", "hi")],
                max_tokens=10,
            )
        # AI/ML expects branded headers on the client construction.
        ctor_kwargs = constructor.call_args.kwargs
        assert ctor_kwargs["default_headers"]["X-Project"] == "AutoGPT"


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class TestOllama:
    @pytest.mark.asyncio
    async def test_rejects_tools_param(self):
        with pytest.raises(ValueError, match="Ollama does not support tools"):
            await call_provider(
                provider="ollama",
                model="llama3",
                api_key="",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                tools=[{"name": "x"}],
            )


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


class TestTimeoutWrapping:
    @pytest.mark.asyncio
    async def test_timeout_raises_TimeoutError_with_provider_name(self):
        async def hang(*args, **kwargs):
            import asyncio as a

            await a.sleep(10)

        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=hang),
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            with pytest.raises(TimeoutError, match="anthropic/"):
                await call_provider(
                    provider="anthropic",
                    model="claude-sonnet-4-6",
                    api_key="x",
                    messages=[_msg("user", "hi")],
                    max_tokens=10,
                    timeout_seconds=0.05,
                )


# ---------------------------------------------------------------------------
# UTF-8 sanitization
# ---------------------------------------------------------------------------


class TestUtf8Sanitization:
    @pytest.mark.asyncio
    async def test_replaces_unpaired_surrogates_before_dispatch(self):
        """Caller hands us a message with an unpaired surrogate; we must
        sanitize before the SDK tries to UTF-8 encode the request body."""
        bad = "\ud800lol"  # unpaired high surrogate
        captured: dict = {}

        fake_response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )

        async def fake_create(**kwargs):
            captured["messages"] = kwargs["messages"]
            return fake_response

        fake_client = SimpleNamespace(
            messages=SimpleNamespace(create=fake_create),
        )
        msgs = [_msg("user", bad)]
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            await call_provider(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="x",
                messages=msgs,
                max_tokens=10,
            )
        sent = captured["messages"][0]["content"]
        # Sanitization replaces the surrogate; the suffix survives.
        assert "lol" in sent
        # Roundtrip through UTF-8 must succeed now.
        sent.encode("utf-8")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_timeout_is_120_seconds(self):
        # Pin the SLA — anything bigger and a stalled provider can park
        # an executor thread for too long.
        assert DEFAULT_REQUEST_TIMEOUT_SECONDS == 120
