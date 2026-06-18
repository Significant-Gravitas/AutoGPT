"""Tests for ``util.llm.providers.call_provider``.

These tests mock each provider's SDK at its construction point so the
test suite never makes a real network call. The goal is to lock in
that ``call_provider`` dispatches to the right SDK with the right
arguments, and that the normalized ``ProviderResponse`` carries the
fields each provider exposes.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic
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
    """Flex mode + batch-on-non-Anthropic raise NotImplementedError until
    later steps land. Pinning this keeps the contract honest: callers
    that opt in early get a loud failure, not silent fallback to sync.

    Anthropic batch lands in Step 4 (this commit) — see
    ``TestBatchSubmission`` below for its tests.
    """

    @pytest.mark.asyncio
    async def test_batch_mode_on_openai_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="OpenAI batch"):
            await call_provider(
                provider="openai",
                model="gpt-4o",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                execution_mode="batch",
                custom_id="probe",
            )

    @pytest.mark.asyncio
    async def test_batch_mode_on_groq_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="provider='anthropic'"):
            await call_provider(
                provider="groq",
                model="mixtral",
                api_key="x",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                execution_mode="batch",
                custom_id="probe",
            )

    @pytest.mark.asyncio
    async def test_batch_mode_requires_custom_id(self):
        with pytest.raises(ValueError, match="custom_id"):
            await call_provider(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                execution_mode="batch",
            )

    @pytest.mark.asyncio
    async def test_flex_mode_passes_service_tier_to_openai_responses(self):
        """OpenAI native exposes ``service_tier="flex"`` via the Responses
        API. The helper must inject it via ``extra_body`` so the SDK
        forwards it on payloads even on SDK versions that don't surface
        ``service_tier`` as a typed kwarg."""
        from backend.util.llm.providers import _call_openai_responses

        captured: dict[str, object] = {}

        class _StubResponses:
            async def create(self, **kwargs):
                captured.update(kwargs)
                resp = MagicMock()
                resp.output = []
                resp.usage = MagicMock(input_tokens=1, output_tokens=1)
                return resp

        class _StubClient:
            def __init__(self, *args, **kwargs):
                self.responses = _StubResponses()

        with patch(
            "backend.util.llm.providers.openai.AsyncOpenAI", new=_StubClient
        ), patch(
            "backend.util.llm.providers.extract_responses_content",
            return_value="",
        ), patch(
            "backend.util.llm.providers.extract_responses_tool_calls",
            return_value=[],
        ), patch(
            "backend.util.llm.providers.extract_responses_usage",
            return_value=(1, 1),
        ), patch(
            "backend.util.llm.providers.extract_responses_reasoning",
            return_value=None,
        ):
            await _call_openai_responses(
                model="gpt-4o",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                temperature=None,
                tools=None,
                force_json_output=False,
                parallel_tool_calls=False,
                timeout_seconds=30.0,
                service_tier="flex",
            )

        assert captured.get("extra_body") == {"service_tier": "flex"}

    @pytest.mark.asyncio
    async def test_no_service_tier_passes_none_extra_body_not_omit_sentinel(self):
        """Sentry AUTOGPT-SERVER-99S: ``extra_body=openai.omit`` crashes the
        real SDK — ``make_request_options`` only checks ``is not None``, so
        the Omit sentinel lands in ``options.extra_json`` and
        ``_merge_mappings`` raises ``TypeError: 'Omit' object is not a
        mapping`` on EVERY plain OpenAI Responses call (no service_tier).
        ``omit`` is only valid for typed params; ``extra_body``'s absent
        value is ``None``."""
        from backend.util.llm.providers import _call_openai_responses

        captured: dict[str, object] = {}

        class _StubResponses:
            async def create(self, **kwargs):
                captured.update(kwargs)
                resp = MagicMock()
                resp.output = []
                resp.usage = MagicMock(input_tokens=1, output_tokens=1)
                return resp

        class _StubClient:
            def __init__(self, *args, **kwargs):
                self.responses = _StubResponses()

        with patch(
            "backend.util.llm.providers.openai.AsyncOpenAI", new=_StubClient
        ), patch(
            "backend.util.llm.providers.extract_responses_content",
            return_value="",
        ), patch(
            "backend.util.llm.providers.extract_responses_tool_calls",
            return_value=[],
        ), patch(
            "backend.util.llm.providers.extract_responses_usage",
            return_value=(1, 1),
        ), patch(
            "backend.util.llm.providers.extract_responses_reasoning",
            return_value=None,
        ):
            await _call_openai_responses(
                model="gpt-4o",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                temperature=None,
                tools=None,
                force_json_output=False,
                parallel_tool_calls=False,
                timeout_seconds=30.0,
                service_tier=None,
            )

        assert captured.get("extra_body") is None

    @pytest.mark.asyncio
    async def test_flex_mode_passes_service_tier_via_openrouter_extra_body(
        self,
    ):
        """OpenRouter forwards ``service_tier`` to OpenAI/Google upstreams
        via ``extra_body``. The merge with the OpenRouter-extras
        ``usage.include`` block must keep both keys — neither overrides
        the other."""
        from backend.util.llm.providers import _call_openai_compat

        captured: dict[str, object] = {}

        class _StubCompletions:
            async def create(self, **kwargs):
                captured.update(kwargs)
                msg = MagicMock()
                msg.content = ""
                choice = MagicMock()
                choice.message = msg
                resp = MagicMock()
                resp.choices = [choice]
                resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
                return resp

        class _StubClient:
            def __init__(self, *args, **kwargs):
                self.chat = SimpleNamespace(completions=_StubCompletions())

        with patch(
            "backend.util.llm.providers.openai.AsyncOpenAI", new=_StubClient
        ), patch(
            "backend.util.llm.providers._extract_openai_compat_cache_tokens",
            return_value=(0, 0),
        ), patch(
            "backend.util.llm.providers.extract_openai_tool_calls",
            return_value=None,
        ), patch(
            "backend.util.llm.providers.extract_openai_reasoning",
            return_value=None,
        ), patch(
            "backend.util.llm.providers.extract_openrouter_cost",
            return_value=None,
        ):
            await _call_openai_compat(
                base_url="https://openrouter.ai/api/v1",
                model="openai/gpt-4o",
                api_key="ork-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                temperature=None,
                tools=None,
                force_json_output=False,
                parallel_tool_calls=False,
                timeout_seconds=30.0,
                include_openrouter_extras=True,
                service_tier="flex",
            )

        extra_body = captured.get("extra_body")
        assert isinstance(extra_body, dict)
        assert extra_body.get("service_tier") == "flex"
        # OpenRouter cost-include must survive the merge — losing it
        # silently disables cost-based rate limiting on flex turns.
        assert extra_body.get("usage") == {"include": True}

    @pytest.mark.asyncio
    async def test_flex_mode_falls_through_to_sync_for_unsupported_provider(
        self, caplog
    ):
        """Anthropic + Groq + the open-weight gateways have no flex tier.
        Callers must get a sync execution instead of a hard error, with
        a log line so the dashboard can surface the silent fallback."""
        sync_mock = AsyncMock(
            return_value=MagicMock(
                content="ok",
                prompt_tokens=1,
                completion_tokens=1,
            )
        )
        with patch(
            "backend.util.llm.providers._dispatch_sync", new=sync_mock
        ), caplog.at_level("WARNING"):
            await call_provider(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="sk-ant-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                execution_mode="flex",
            )
        sync_mock.assert_awaited_once()
        assert sync_mock.call_args.kwargs.get("service_tier") is None
        assert any(
            "execution_mode='flex' requested for provider=anthropic" in rec.message
            for rec in caplog.records
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


@pytest.fixture(autouse=True)
def _reset_chat_base_url_cache():
    """The ChatConfig.base_url read is memoized module-wide; clear it so
    each test's patched ``backend.copilot.config.ChatConfig`` is the one
    actually constructed."""
    from backend.util.llm.providers import _read_chat_config_base_url

    _read_chat_config_base_url.cache_clear()
    yield
    _read_chat_config_base_url.cache_clear()


def _fake_ollama_chat_client(
    captured: dict,
    *,
    content: str = "ok",
    prompt_eval_count: int | None = 7,
    eval_count: int | None = 9,
) -> SimpleNamespace:
    """Fake ``ollama.AsyncClient`` exposing a ``chat`` coroutine that
    records its kwargs and returns a ChatResponse-shaped object."""

    async def chat(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            message=SimpleNamespace(content=content),
            prompt_eval_count=prompt_eval_count,
            eval_count=eval_count,
        )

    return SimpleNamespace(chat=chat)


@contextmanager
def _patched_ollama_dispatch(
    captured: dict,
    *,
    content: str = "ok",
    chat_base_url: str | None = None,
):
    """Patch set for an Ollama dispatch test: mocked SSRF validation, a
    stubbed copilot ``ChatConfig`` (so no real settings load happens),
    and a fake ``ollama.AsyncClient``. Yields ``(validate_mock,
    client_ctor_mock)``."""
    validate_mock = AsyncMock(return_value=None)
    client_ctor = MagicMock(
        return_value=_fake_ollama_chat_client(captured, content=content)
    )
    with (
        patch("backend.util.llm.providers.validate_url_host", new=validate_mock),
        patch(
            "backend.copilot.config.ChatConfig",
            return_value=SimpleNamespace(base_url=chat_base_url),
        ),
        patch("backend.util.llm.providers.ollama.AsyncClient", new=client_ctor),
    ):
        yield validate_mock, client_ctor


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

    @pytest.mark.asyncio
    async def test_dispatches_via_chat_api_with_structured_messages(self):
        """The dream prompts carry a real system + user conversation;
        flattening them into a repr'd ``generate`` prompt fed the model
        bracketed Python list syntax instead of a chat. The message list
        must reach the chat API intact, roles included."""
        captured: dict = {}
        with _patched_ollama_dispatch(captured):
            await call_provider(
                provider="ollama",
                model="llama3",
                api_key="",
                messages=[
                    _msg("system", "you are a memory consolidator"),
                    _msg("user", "consolidate these transcripts"),
                ],
                max_tokens=4096,
            )
        assert captured["messages"] == [
            {"role": "system", "content": "you are a memory consolidator"},
            {"role": "user", "content": "consolidate these transcripts"},
        ]
        assert captured["stream"] is False

    @pytest.mark.asyncio
    async def test_force_json_output_requests_json_format(self):
        """``structured_completion`` passes ``force_json_output=True`` for
        every dream phase and the dream docs promise forced JSON works on
        every transport — Ollama honors it via ``format="json"``."""
        captured: dict = {}
        with _patched_ollama_dispatch(captured, content="{}"):
            await call_provider(
                provider="ollama",
                model="llama3",
                api_key="",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                force_json_output=True,
            )
        assert captured["format"] == "json"

    @pytest.mark.asyncio
    async def test_plain_completion_does_not_force_json_format(self):
        captured: dict = {}
        with _patched_ollama_dispatch(captured):
            await call_provider(
                provider="ollama",
                model="llama3",
                api_key="",
                messages=[_msg("user", "hi")],
                max_tokens=10,
            )
        assert captured["format"] is None

    @pytest.mark.asyncio
    async def test_max_tokens_caps_output_via_num_predict_not_num_ctx(self):
        """``num_ctx`` is the INPUT context window — setting it from
        ``max_tokens`` silently truncated the consolidate/recombine
        prompts (whole transcripts + fact lists) at the output budget.
        The output cap belongs on ``num_predict``; ``num_ctx`` stays at
        the model default."""
        captured: dict = {}
        with _patched_ollama_dispatch(captured):
            await call_provider(
                provider="ollama",
                model="llama3",
                api_key="",
                messages=[_msg("user", "hi")],
                max_tokens=4096,
            )
        assert captured["options"]["num_predict"] == 4096
        assert "num_ctx" not in captured["options"]

    @pytest.mark.asyncio
    async def test_normalizes_chat_response_into_provider_response(self):
        captured: dict = {}
        with _patched_ollama_dispatch(captured, content='{"facts": []}') as (_, ctor):
            result = await call_provider(
                provider="ollama",
                model="llama3",
                api_key="",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                ollama_host="http://gpu-box:11434",
                timeout_seconds=42.0,
            )
        assert isinstance(result, ProviderResponse)
        assert result.content == '{"facts": []}'
        assert result.prompt_tokens == 7
        assert result.completion_tokens == 9
        ctor.assert_called_once_with(host="http://gpu-box:11434", timeout=42.0)


class TestOllamaTrustedHosts:
    """SSRF trust list for the Ollama dispatch. The host can be
    user-supplied (block input field) so it must be validated — but the
    allowlist has to include BOTH operator-set endpoints: the block
    layer's ``Config.ollama_host`` AND the copilot ``CHAT_BASE_URL``
    the dream pass routes through on the local transport. Trusting only
    the former made every local dream phase fail with a misleading
    'blocked IP' error whenever the two were configured differently."""

    @pytest.mark.asyncio
    async def test_trust_list_includes_chat_base_url_and_block_config_host(self):
        from backend.util.llm.providers import settings

        captured: dict = {}
        with _patched_ollama_dispatch(
            captured,
            chat_base_url="http://host.docker.internal:11434/v1",
        ) as (validate_mock, _):
            await call_provider(
                provider="ollama",
                model="llama3",
                api_key="",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                ollama_host="http://host.docker.internal:11434",
            )
        validate_mock.assert_awaited_once()
        assert validate_mock.call_args.args[0] == "http://host.docker.internal:11434"
        assert validate_mock.call_args.kwargs["trusted_hostnames"] == [
            settings.config.ollama_host,
            "http://host.docker.internal:11434/v1",
        ]

    @pytest.mark.asyncio
    async def test_chat_config_failure_falls_back_to_block_config_trust_list(
        self, caplog
    ):
        """A misconfigured copilot ``ChatConfig`` (its validators raise at
        construction) must not take down block-layer Ollama calls — the
        trust list degrades to ``Config.ollama_host`` only."""
        from backend.util.llm.providers import settings

        captured: dict = {}
        validate_mock = AsyncMock(return_value=None)
        with (
            patch(
                "backend.util.llm.providers.validate_url_host",
                new=validate_mock,
            ),
            patch(
                "backend.copilot.config.ChatConfig",
                side_effect=ValueError("misconfigured transport"),
            ),
            patch(
                "backend.util.llm.providers.ollama.AsyncClient",
                return_value=_fake_ollama_chat_client(captured),
            ),
            caplog.at_level("WARNING"),
        ):
            result = await call_provider(
                provider="ollama",
                model="llama3",
                api_key="",
                messages=[_msg("user", "hi")],
                max_tokens=10,
            )
        assert isinstance(result, ProviderResponse)
        assert validate_mock.call_args.kwargs["trusted_hostnames"] == [
            settings.config.ollama_host
        ]
        assert any(
            "Could not read ChatConfig.base_url" in rec.message
            for rec in caplog.records
        )

    @pytest.mark.asyncio
    async def test_lan_host_matching_chat_base_url_passes_real_ssrf_check(self):
        """Regression for the local dream-pass failure: operator points
        ``CHAT_BASE_URL`` at a LAN GPU box (private IP) while the block
        layer's ``ollama_host`` stays at its localhost default. The real
        ``validate_url_host`` must accept the host via the trust match
        instead of rejecting the private IP."""
        captured: dict = {}
        client_ctor = MagicMock(return_value=_fake_ollama_chat_client(captured))
        with (
            patch(
                "backend.copilot.config.ChatConfig",
                return_value=SimpleNamespace(base_url="http://192.168.1.50:11434/v1"),
            ),
            patch("backend.util.llm.providers.ollama.AsyncClient", new=client_ctor),
        ):
            result = await call_provider(
                provider="ollama",
                model="llama3",
                api_key="",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                ollama_host="http://192.168.1.50:11434",
            )
        assert isinstance(result, ProviderResponse)
        client_ctor.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_config_is_constructed_once_across_dispatches(self):
        """``ChatConfig`` is a pydantic-settings class with
        ``env_file=".env"`` — constructing it per dispatch meant dotenv
        disk I/O and repeated validator warnings on every Ollama call.
        The base_url read must be memoized after the first success while
        every dispatch still gets the cached value in its trust list."""
        captured: dict = {}
        validate_mock = AsyncMock(return_value=None)
        chat_config_ctor = MagicMock(
            return_value=SimpleNamespace(
                base_url="http://host.docker.internal:11434/v1"
            )
        )
        with (
            patch("backend.util.llm.providers.validate_url_host", new=validate_mock),
            patch("backend.copilot.config.ChatConfig", new=chat_config_ctor),
            patch(
                "backend.util.llm.providers.ollama.AsyncClient",
                return_value=_fake_ollama_chat_client(captured),
            ),
        ):
            for _ in range(3):
                await call_provider(
                    provider="ollama",
                    model="llama3",
                    api_key="",
                    messages=[_msg("user", "hi")],
                    max_tokens=10,
                )
        chat_config_ctor.assert_called_once()
        assert validate_mock.await_count == 3
        for call in validate_mock.call_args_list:
            assert (
                "http://host.docker.internal:11434/v1"
                in call.kwargs["trusted_hostnames"]
            )

    @pytest.mark.asyncio
    async def test_chat_config_failure_is_not_cached_as_permanent_none(self):
        """A transient ``ChatConfig`` construction failure degrades that
        one dispatch to the block-layer trust list, but must not be
        memoized — once the config is fixed, the next dispatch picks up
        ``CHAT_BASE_URL`` again."""
        from backend.util.llm.providers import settings

        captured: dict = {}
        validate_mock = AsyncMock(return_value=None)
        chat_config_ctor = MagicMock(
            side_effect=[
                ValueError("transient misconfiguration"),
                SimpleNamespace(base_url="http://192.168.1.50:11434/v1"),
            ]
        )
        with (
            patch("backend.util.llm.providers.validate_url_host", new=validate_mock),
            patch("backend.copilot.config.ChatConfig", new=chat_config_ctor),
            patch(
                "backend.util.llm.providers.ollama.AsyncClient",
                return_value=_fake_ollama_chat_client(captured),
            ),
        ):
            for _ in range(2):
                await call_provider(
                    provider="ollama",
                    model="llama3",
                    api_key="",
                    messages=[_msg("user", "hi")],
                    max_tokens=10,
                )
        first_call, second_call = validate_mock.call_args_list
        assert first_call.kwargs["trusted_hostnames"] == [settings.config.ollama_host]
        assert second_call.kwargs["trusted_hostnames"] == [
            settings.config.ollama_host,
            "http://192.168.1.50:11434/v1",
        ]

    @pytest.mark.asyncio
    async def test_unconfigured_private_host_is_still_rejected(self):
        """The widened trust list must not weaken the SSRF guard: a
        private-IP host that appears in NEITHER config value is rejected
        before any client is constructed."""
        client_ctor = MagicMock()
        with (
            patch(
                "backend.copilot.config.ChatConfig",
                return_value=SimpleNamespace(base_url=None),
            ),
            patch("backend.util.llm.providers.ollama.AsyncClient", new=client_ctor),
        ):
            with pytest.raises(ValueError, match="blocked or private IP"):
                await call_provider(
                    provider="ollama",
                    model="llama3",
                    api_key="",
                    messages=[_msg("user", "hi")],
                    max_tokens=10,
                    ollama_host="http://169.254.169.254:11434",
                )
        client_ctor.assert_not_called()


# ---------------------------------------------------------------------------
# Temperature threading
# ---------------------------------------------------------------------------


class TestTemperatureThreading:
    """The dream pass tunes temperature per phase (0.2 consolidate /
    0.9 recombine / 0.0 sanitize) on every transport. The dispatcher
    used to forward it only to Anthropic — every other provider
    silently ran at its default (~1.0), degrading memory-write quality
    with no error. Pin that each path forwards the caller's value, and
    that ``None`` keeps the field off the wire (some reasoning models
    reject ``temperature`` — the Anthropic-style guard applies
    everywhere)."""

    def _openai_compat_patches(self, async_create):
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=async_create))
        )
        return (
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
                return_value=None,
            ),
        )

    @pytest.mark.asyncio
    async def test_openrouter_receives_recombine_temperature(self):
        fake_response = _fake_openai_chat_response("x", prompt=1, completion=1)
        async_create = AsyncMock(return_value=fake_response)
        p1, p2, p3, p4 = self._openai_compat_patches(async_create)
        with p1, p2, p3, p4:
            await call_provider(
                provider="open_router",
                model="anthropic/claude-sonnet-4-6",
                api_key="sk-or-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                temperature=0.9,
            )
        assert async_create.call_args.kwargs["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_openrouter_omits_temperature_when_caller_unset(self):
        fake_response = _fake_openai_chat_response("x", prompt=1, completion=1)
        async_create = AsyncMock(return_value=fake_response)
        p1, p2, p3, p4 = self._openai_compat_patches(async_create)
        with p1, p2, p3, p4:
            await call_provider(
                provider="open_router",
                model="openai/o3",
                api_key="sk-or-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
            )
        assert "temperature" not in async_create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_openai_responses_receives_consolidate_temperature(self):
        async_create = AsyncMock(return_value=SimpleNamespace())
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
                return_value="ok",
            ),
            patch(
                "backend.util.llm.providers.extract_responses_reasoning",
                return_value=None,
            ),
            patch(
                "backend.util.llm.providers.extract_responses_usage",
                return_value=(1, 1),
            ),
        ):
            await call_provider(
                provider="openai",
                model="gpt-4o",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                temperature=0.2,
            )
        assert async_create.call_args.kwargs["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_openai_responses_omits_temperature_when_caller_unset(self):
        """Reasoning models reject ``temperature`` — when the caller
        doesn't set one, the SDK must receive the omit sentinel so the
        field never reaches the wire."""
        import openai

        async_create = AsyncMock(return_value=SimpleNamespace())
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
                return_value="ok",
            ),
            patch(
                "backend.util.llm.providers.extract_responses_reasoning",
                return_value=None,
            ),
            patch(
                "backend.util.llm.providers.extract_responses_usage",
                return_value=(1, 1),
            ),
        ):
            await call_provider(
                provider="openai",
                model="o3",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
            )
        assert async_create.call_args.kwargs["temperature"] is openai.omit

    @pytest.mark.asyncio
    async def test_groq_receives_zero_sanitize_temperature(self):
        """0.0 is falsy but explicitly set (the sanitize phase runs
        deterministic) — the None-guard must forward it, not drop it."""
        fake_usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="x"))],
            usage=fake_usage,
        )
        async_create = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=async_create))
        )
        with patch("groq.AsyncGroq", return_value=fake_client):
            await call_provider(
                provider="groq",
                model="mixtral-8x7b",
                api_key="x",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                temperature=0.0,
            )
        assert async_create.call_args.kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_groq_omits_temperature_when_caller_unset(self):
        fake_usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
        fake_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="x"))],
            usage=fake_usage,
        )
        async_create = AsyncMock(return_value=fake_response)
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=async_create))
        )
        with patch("groq.AsyncGroq", return_value=fake_client):
            await call_provider(
                provider="groq",
                model="mixtral-8x7b",
                api_key="x",
                messages=[_msg("user", "hi")],
                max_tokens=10,
            )
        assert "temperature" not in async_create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_ollama_receives_temperature_in_options(self):
        captured: dict = {}
        with _patched_ollama_dispatch(captured):
            await call_provider(
                provider="ollama",
                model="llama3",
                api_key="",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                temperature=0.2,
            )
        assert captured["options"]["temperature"] == 0.2

    @pytest.mark.asyncio
    async def test_ollama_omits_temperature_when_caller_unset(self):
        captured: dict = {}
        with _patched_ollama_dispatch(captured):
            await call_provider(
                provider="ollama",
                model="llama3",
                api_key="",
                messages=[_msg("user", "hi")],
                max_tokens=10,
            )
        assert "temperature" not in captured["options"]


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


# ---------------------------------------------------------------------------
# Anthropic batch submission + poll + download
# ---------------------------------------------------------------------------


from backend.util.llm.providers import (  # noqa: E402
    BatchResultRow,
    BatchSubmissionRef,
    download_batch_results,
    poll_batch,
)


class TestBatchSubmission:
    @pytest.mark.asyncio
    async def test_submits_to_anthropic_batches_create(self):
        fake_batch = SimpleNamespace(id="msgbatch_abc123")
        async_create = AsyncMock(return_value=fake_batch)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(batches=SimpleNamespace(create=async_create)),
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            ref = await call_provider(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="sk-ant-test",
                messages=[
                    _msg("system", "be helpful"),
                    _msg("user", "give me a fact"),
                ],
                max_tokens=200,
                temperature=0.3,
                execution_mode="batch",
                custom_id="passid-1:consolidate",
            )

        assert isinstance(ref, BatchSubmissionRef)
        assert ref.provider == "anthropic"
        assert ref.provider_batch_id == "msgbatch_abc123"
        assert ref.custom_id == "passid-1:consolidate"

        async_create.assert_awaited_once()
        kwargs = async_create.call_args.kwargs
        requests = kwargs["requests"]
        assert len(requests) == 1
        req = requests[0]
        assert req["custom_id"] == "passid-1:consolidate"
        params = req["params"]
        assert params["model"] == "claude-sonnet-4-6"
        assert params["max_tokens"] == 200
        assert params["temperature"] == 0.3
        # System prompt landed as cache-controlled text block.
        assert isinstance(params["system"], list)
        assert params["system"][0]["text"] == "be helpful"
        assert params["system"][0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_submit_includes_tool_choice_for_structured_output(self):
        """Step 5 leans on this: pin that ``tool_choice`` and ``tools``
        make it into the batch ``params`` so the model is constrained to
        emit one tool_use block. Without it, the model is free to write
        prose and the dream's JSON parser does extra work."""
        fake_batch = SimpleNamespace(id="msgbatch_xyz")
        async_create = AsyncMock(return_value=fake_batch)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(batches=SimpleNamespace(create=async_create)),
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            await call_provider(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="sk-ant-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                tools=[{"name": "x", "parameters": {}}],
                tool_choice={"type": "tool", "name": "x"},
                execution_mode="batch",
                custom_id="x",
            )
        params = async_create.call_args.kwargs["requests"][0]["params"]
        assert params["tool_choice"] == {"type": "tool", "name": "x"}
        assert params["tools"][0]["name"] == "x"


class TestPollBatch:
    @pytest.mark.asyncio
    async def test_maps_ended_status(self):
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(
                batches=SimpleNamespace(
                    retrieve=AsyncMock(
                        return_value=SimpleNamespace(processing_status="ended")
                    )
                )
            ),
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            assert (
                await poll_batch(
                    provider="anthropic",
                    provider_batch_id="msgbatch_1",
                    api_key="sk-ant-test",
                )
                == "ended"
            )

    @pytest.mark.asyncio
    async def test_maps_in_progress_to_processing(self):
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(
                batches=SimpleNamespace(
                    retrieve=AsyncMock(
                        return_value=SimpleNamespace(processing_status="in_progress")
                    )
                )
            ),
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            assert (
                await poll_batch(
                    provider="anthropic",
                    provider_batch_id="x",
                    api_key="sk-ant-test",
                )
                == "processing"
            )

    @pytest.mark.asyncio
    async def test_unknown_status_reports_pending(self):
        """Future-proof: if Anthropic adds a new ``processing_status``
        value we don't recognize, we keep polling rather than failing
        the whole pass."""
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(
                batches=SimpleNamespace(
                    retrieve=AsyncMock(
                        return_value=SimpleNamespace(
                            processing_status="some_future_state"
                        )
                    )
                )
            ),
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            assert (
                await poll_batch(
                    provider="anthropic",
                    provider_batch_id="x",
                    api_key="sk-ant-test",
                )
                == "pending"
            )

    @pytest.mark.asyncio
    async def test_openai_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="anthropic"):
            await poll_batch(
                provider="openai",
                provider_batch_id="x",
                api_key="x",
            )


class TestDownloadBatchResults:
    @pytest.mark.asyncio
    async def test_extracts_text_block_content(self):
        """A succeeded result with a single text block flattens to the
        text content. That's the most common dream-pass result shape
        when tool_use isn't in play."""
        results_iter = _fake_async_iter(
            [
                SimpleNamespace(
                    custom_id="passid-1:consolidate",
                    result=SimpleNamespace(
                        type="succeeded",
                        message=SimpleNamespace(
                            content=[
                                SimpleNamespace(type="text", text='{"facts": []}')
                            ],
                            usage=SimpleNamespace(
                                input_tokens=10,
                                output_tokens=20,
                                cache_read_input_tokens=3,
                                cache_creation_input_tokens=2,
                            ),
                        ),
                    ),
                )
            ]
        )
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(
                batches=SimpleNamespace(results=AsyncMock(return_value=results_iter))
            ),
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            rows = await download_batch_results(
                provider="anthropic",
                provider_batch_id="msgbatch_done",
                api_key="sk-ant-test",
            )
        assert len(rows) == 1
        row = rows[0]
        assert isinstance(row, BatchResultRow)
        assert row.custom_id == "passid-1:consolidate"
        assert row.content == '{"facts": []}'
        assert row.input_tokens == 10
        assert row.output_tokens == 20
        assert row.cache_read_tokens == 3
        assert row.cache_creation_tokens == 2
        assert row.error is None

    @pytest.mark.asyncio
    async def test_extracts_tool_use_input_as_json(self):
        """Step 5 leans on this: when the dream pass uses
        ``tool_choice={"type":"tool","name":...}`` to force structured
        output, the result is one ``tool_use`` block whose ``input``
        IS the structured payload. Flattening it to a JSON string is
        what the dream parser then validates against the Pydantic
        schema."""
        results_iter = _fake_async_iter(
            [
                SimpleNamespace(
                    custom_id="passid-1:sanitize",
                    result=SimpleNamespace(
                        type="succeeded",
                        message=SimpleNamespace(
                            content=[
                                SimpleNamespace(
                                    type="tool_use",
                                    input={
                                        "writes": [],
                                        "summary_for_user": "ok",
                                    },
                                )
                            ],
                            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                        ),
                    ),
                )
            ]
        )
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(
                batches=SimpleNamespace(results=AsyncMock(return_value=results_iter))
            ),
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            rows = await download_batch_results(
                provider="anthropic",
                provider_batch_id="x",
                api_key="x",
            )
        assert rows[0].content == '{"writes": [], "summary_for_user": "ok"}'

    @pytest.mark.asyncio
    async def test_errored_row_carries_error_string_with_zero_tokens(self):
        """Per-request errors do not fail the whole batch — they surface
        as a row with ``error`` set and token counts at 0. The
        BatchExecutor's dispatch handler reads this to mark the phase
        failed in the JobStatus row."""
        results_iter = _fake_async_iter(
            [
                SimpleNamespace(
                    custom_id="passid-1:recombine",
                    result=SimpleNamespace(
                        type="errored",
                        error=SimpleNamespace(message="content moderation"),
                    ),
                )
            ]
        )
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(
                batches=SimpleNamespace(results=AsyncMock(return_value=results_iter))
            ),
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            rows = await download_batch_results(
                provider="anthropic",
                provider_batch_id="x",
                api_key="x",
            )
        assert rows[0].error is not None
        assert "content moderation" in rows[0].error
        assert rows[0].input_tokens == 0
        assert rows[0].output_tokens == 0


def _fake_async_iter(items):
    """Build an awaitable that yields an async iterator over ``items``.

    The Anthropic SDK's ``batches.results()`` returns an awaitable
    yielding an async-iterable streamer; this helper mocks that shape
    without dragging the real SDK into the tests.
    """

    class _Iter:
        def __init__(self, items):
            self._items = list(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    return _Iter(items)


# ---------------------------------------------------------------------------
# Streaming + non-streaming OpenAI-compat helpers (chat dispatch)
# ---------------------------------------------------------------------------


class TestCallProviderStream:
    """Regression coverage for ``call_provider_stream``. The chat
    baseline path delegates here; behavior changes break chat in
    user-facing ways (tokens stop streaming, tool calls drop, etc.)."""

    @pytest.mark.asyncio
    async def test_uses_provided_client_directly(self):
        """When a ``client`` is passed, the helper does NOT construct a
        new one — preserves the chat layer's module-cached
        Langfuse-wrapped client + its pooled HTTP connections."""
        from backend.util.llm.providers import call_provider_stream

        sentinel_stream = MagicMock(name="async_stream")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=sentinel_stream)

        result = await call_provider_stream(
            client=client,
            model="gpt-4o",
            messages=[_msg("user", "hi")],
            extra_body={"foo": "bar"},
            stream_options={"include_usage": True},
        )

        assert result is sentinel_stream
        client.chat.completions.create.assert_awaited_once()
        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["model"] == "gpt-4o"
        assert kwargs["extra_body"] == {"foo": "bar"}
        assert kwargs["stream_options"] == {"include_usage": True}

    @pytest.mark.asyncio
    async def test_builds_client_via_factory_when_not_provided(self):
        """When no ``client`` is passed, the helper instantiates one
        via ``client_factory(base_url, api_key)``. The chat layer uses
        this branch to inject ``LangfuseAsyncOpenAI`` as the factory."""
        from backend.util.llm.providers import call_provider_stream

        captured: dict[str, object] = {}

        class _StubClient:
            def __init__(self, *, base_url: str, api_key: str):
                captured["base_url"] = base_url
                captured["api_key"] = api_key
                self.chat = SimpleNamespace(completions=SimpleNamespace())
                self.chat.completions.create = AsyncMock(return_value="stream")

        result = await call_provider_stream(
            model="claude-sonnet-4-6",
            messages=[_msg("user", "hi")],
            base_url="https://api.test/v1",
            api_key="sk-test",
            client_factory=_StubClient,
        )

        assert result == "stream"
        assert captured == {
            "base_url": "https://api.test/v1",
            "api_key": "sk-test",
        }

    @pytest.mark.asyncio
    async def test_omit_max_tokens_drops_field_from_create_call(self):
        """OpenRouter's thinking routes inject their own ``max_tokens``
        default and 400 if the client sends one too. The chat baseline
        passes ``openai.omit`` to skip the field; the helper must
        actually drop it from create_kwargs (not pass it as the omit
        sentinel which would still register as 'present' on the dict)."""
        import openai

        from backend.util.llm.providers import call_provider_stream

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value="s")
        await call_provider_stream(
            client=client,
            model="x",
            messages=[_msg("user", "hi")],
            max_tokens=openai.omit,
        )
        kwargs = client.chat.completions.create.await_args.kwargs
        assert "max_tokens" not in kwargs

    @pytest.mark.asyncio
    async def test_real_max_tokens_lands_in_create_kwargs(self):
        """Direct-Anthropic mode passes an explicit max_tokens because
        Anthropic refuses to default it when thinking is enabled. That
        value must reach the SDK call."""
        from backend.util.llm.providers import call_provider_stream

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value="s")
        await call_provider_stream(
            client=client,
            model="claude-opus-4-1",
            messages=[_msg("user", "hi")],
            max_tokens=8192,
        )
        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["max_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_raises_when_neither_client_nor_credentials(self):
        from backend.util.llm.providers import call_provider_stream

        with pytest.raises(ValueError, match="pass either"):
            await call_provider_stream(
                model="x",
                messages=[_msg("user", "hi")],
            )


class TestCallProviderOpenAICompatSync:
    @pytest.mark.asyncio
    async def test_uses_provided_client_directly(self):
        from backend.util.llm.providers import call_provider_openai_compat_sync

        sentinel = MagicMock(name="response")
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=sentinel)

        result = await call_provider_openai_compat_sync(
            client=client,
            model="claude-haiku-4-5",
            messages=[_msg("user", "hi")],
            max_tokens=20,
            extra_body={"usage": {"include": True}},
        )

        assert result is sentinel
        kwargs = client.chat.completions.create.await_args.kwargs
        assert kwargs["model"] == "claude-haiku-4-5"
        assert kwargs["max_tokens"] == 20
        assert kwargs["extra_body"] == {"usage": {"include": True}}
        assert "stream" not in kwargs

    @pytest.mark.asyncio
    async def test_raises_when_neither_client_nor_credentials(self):
        from backend.util.llm.providers import call_provider_openai_compat_sync

        with pytest.raises(ValueError, match="pass either"):
            await call_provider_openai_compat_sync(
                model="x",
                messages=[_msg("user", "hi")],
                max_tokens=20,
            )


class TestCancelBatch:
    @pytest.mark.asyncio
    async def test_anthropic_cancel_calls_sdk_and_returns_true(self):
        from backend.util.llm.providers import cancel_batch

        async_cancel = AsyncMock(return_value=SimpleNamespace(id="msgbatch_x"))
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(batches=SimpleNamespace(cancel=async_cancel))
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            ok = await cancel_batch(
                provider="anthropic",
                provider_batch_id="msgbatch_x",
                api_key="sk-test",
            )
        assert ok is True
        async_cancel.assert_awaited_once_with("msgbatch_x")

    @pytest.mark.asyncio
    async def test_non_anthropic_provider_returns_false(self):
        from backend.util.llm.providers import cancel_batch

        ok = await cancel_batch(
            provider="openai", provider_batch_id="b1", api_key="sk-test"
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_cancel_swallows_sdk_error_returns_false(self):
        from backend.util.llm.providers import cancel_batch

        async_cancel = AsyncMock(side_effect=RuntimeError("boom"))
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(batches=SimpleNamespace(cancel=async_cancel))
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            ok = await cancel_batch(
                provider="anthropic", provider_batch_id="b1", api_key="sk-test"
            )
        assert ok is False


class TestAnthropicTemperatureDeprecation:
    """Anthropic rejects ``temperature`` on its newest models with
    '`temperature` is deprecated for this model.' (verified live:
    claude-opus-4-7 and claude-opus-4-8 reject; sonnet-4-6 accepts).
    The dream pass's recombine phase died on this nightly — the param
    must be stripped for known-rejecting models, and the sync path
    must self-heal for unknown future ones."""

    def _fake_response(self):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="ok")],
            usage=SimpleNamespace(input_tokens=1, output_tokens=1),
            stop_reason="end_turn",
        )

    @pytest.mark.asyncio
    async def test_sync_omits_temperature_for_deprecating_model(self):
        async_create = AsyncMock(return_value=self._fake_response())
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=async_create))
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            await call_provider(
                provider="anthropic",
                model="claude-opus-4-7",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                temperature=0.9,
            )
        assert "temperature" not in async_create.call_args.kwargs

    @pytest.mark.asyncio
    async def test_sync_retries_without_temperature_on_deprecation_error(self):
        """Self-healing for future models the deny-list doesn't know:
        one retry without the param, same call otherwise."""
        import httpx

        err = anthropic.BadRequestError(
            message="`temperature` is deprecated for this model.",
            response=httpx.Response(
                400, request=httpx.Request("POST", "https://api.anthropic.com")
            ),
            body={"error": {"message": "`temperature` is deprecated for this model."}},
        )
        async_create = AsyncMock(side_effect=[err, self._fake_response()])
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=async_create))
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            result = await call_provider(
                provider="anthropic",
                model="claude-sonnet-9-9",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                temperature=0.9,
            )
        assert isinstance(result, ProviderResponse)
        assert async_create.await_count == 2
        first, second = async_create.await_args_list
        assert first.kwargs["temperature"] == 0.9
        assert "temperature" not in second.kwargs

    @pytest.mark.asyncio
    async def test_sync_retry_detection_is_case_insensitive(self):
        """The deprecation message wording/casing isn't contractual — a
        future model returning 'Temperature ... Deprecated' must still
        trigger the self-healing retry, not fall through and fail."""
        import httpx

        err = anthropic.BadRequestError(
            message="Temperature is Deprecated for this model.",
            response=httpx.Response(
                400, request=httpx.Request("POST", "https://api.anthropic.com")
            ),
            body={"error": {"message": "Temperature is Deprecated for this model."}},
        )
        async_create = AsyncMock(side_effect=[err, self._fake_response()])
        fake_client = SimpleNamespace(messages=SimpleNamespace(create=async_create))
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            result = await call_provider(
                provider="anthropic",
                model="claude-sonnet-9-9",
                api_key="sk-test",
                messages=[_msg("user", "hi")],
                max_tokens=10,
                temperature=0.9,
            )
        assert isinstance(result, ProviderResponse)
        assert async_create.await_count == 2
        first, second = async_create.await_args_list
        assert first.kwargs["temperature"] == 0.9
        assert "temperature" not in second.kwargs

    @pytest.mark.asyncio
    async def test_batch_omits_temperature_for_deprecating_model(self):
        """Batch errors come back hours later in the result rows — the
        param must never be submitted for known-rejecting models."""
        fake_batch = SimpleNamespace(id="msgbatch_x")
        async_create = AsyncMock(return_value=fake_batch)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(batches=SimpleNamespace(create=async_create)),
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            await call_provider(
                provider="anthropic",
                model="claude-opus-4-7",
                api_key="sk-ant-test",
                messages=[_msg("user", "recombine this")],
                max_tokens=100,
                temperature=0.9,
                execution_mode="batch",
                custom_id="p1_recombine",
            )
        params = async_create.call_args.kwargs["requests"][0]["params"]
        assert "temperature" not in params

    @pytest.mark.asyncio
    async def test_batch_keeps_temperature_for_supporting_model(self):
        fake_batch = SimpleNamespace(id="msgbatch_y")
        async_create = AsyncMock(return_value=fake_batch)
        fake_client = SimpleNamespace(
            messages=SimpleNamespace(batches=SimpleNamespace(create=async_create)),
        )
        with patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=fake_client,
        ):
            await call_provider(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="sk-ant-test",
                messages=[_msg("user", "consolidate this")],
                max_tokens=100,
                temperature=0.2,
                execution_mode="batch",
                custom_id="p1_consolidate",
            )
        params = async_create.call_args.kwargs["requests"][0]["params"]
        assert params["temperature"] == 0.2
