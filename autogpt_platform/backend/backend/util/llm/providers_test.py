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
                tools=None,
                force_json_output=False,
                parallel_tool_calls=False,
                timeout_seconds=30.0,
                service_tier="flex",
            )

        assert captured.get("extra_body") == {"service_tier": "flex"}

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
