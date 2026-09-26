"""``structured_complete``: one sync call on the context's route, parsed.

The delegation contract: the route's provider and model with the platform's
key, JSON mode, the job's timeout, and which failures carry billed usage.
Moved here with the call from ``dream/llm.py``; the parse helpers' own tests
stay in ``dream/llm_test.py``. The native Anthropic path is in
``complete_anthropic_test.py`` and its forced-tool retry in
``complete_retry_test.py``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from backend.util.llm.providers import ProviderResponse

from ._test_data import (
    CALL_PROVIDER,
    MESSAGES,
    ONE_FACT,
    ROUTING,
    SampleOutput,
    platform_routing,
    sync_ctx,
)
from .complete import _normalize_ollama_host, structured_complete
from .context import InferenceError
from .routing import anthropic_batch_route


class TestDelegation:
    """``structured_complete`` hands ``call_provider`` the route's provider
    and model with the platform's key, and turns the ``ProviderResponse``
    into the typed value and an ``InferenceUsage``."""

    @pytest.mark.asyncio
    async def test_delegates_to_call_provider_with_the_route(self):
        fake = ProviderResponse(
            content=ONE_FACT, prompt_tokens=12, completion_tokens=4, cost_usd=0.0042
        )
        call_provider = AsyncMock(return_value=fake)
        with patch(ROUTING, return_value=platform_routing()), patch(
            CALL_PROVIDER, call_provider
        ):
            result = await structured_complete(
                sync_ctx(),
                MESSAGES,
                SampleOutput,
                temperature=0.3,
                max_output_tokens=512,
            )

        assert result.value.facts[0].confidence == 0.9
        assert result.usage.model_dump() == {
            "model": "anthropic/claude-sonnet-4-6",
            "input_tokens": 12,
            "output_tokens": 4,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "cost_usd": 0.0042,
            "cost_source": "provider",
            "payer": "platform_allowance",
        }
        kwargs = call_provider.call_args.kwargs
        assert kwargs["provider"] == "open_router"
        assert kwargs["api_key"] == "sk-or-test"
        assert kwargs["model"] == "anthropic/claude-sonnet-4-6"
        assert kwargs["max_tokens"] == 512
        assert kwargs["temperature"] == 0.3
        # JSON mode: without it OpenRouter→Claude falls back to free text.
        assert kwargs["force_json_output"] is True
        # The forced tool is the native Anthropic path's device only.
        assert kwargs["tools"] is None
        assert kwargs["tool_choice"] is None

    @pytest.mark.asyncio
    async def test_the_jobs_timeout_reaches_call_provider(self):
        """Recombine/sanitize carry 16384-token output budgets, which take far
        longer than the shared 120s default to decode; the job's own budget
        must reach ``call_provider`` or the token cap is dead letter."""
        call_provider = AsyncMock(
            return_value=ProviderResponse(
                content=ONE_FACT, prompt_tokens=1, completion_tokens=1
            )
        )
        with patch(ROUTING, return_value=platform_routing()), patch(
            CALL_PROVIDER, call_provider
        ):
            await structured_complete(
                sync_ctx(timeout_seconds=600), MESSAGES, SampleOutput
            )
            await structured_complete(sync_ctx(), MESSAGES, SampleOutput)

        first, second = call_provider.call_args_list
        assert first.kwargs["timeout_seconds"] == 600
        # No budget forwards the sentinel, so call_provider reads the live
        # default rather than a def-time snapshot of it.
        assert second.kwargs["timeout_seconds"] is None

    @pytest.mark.asyncio
    async def test_an_unpriced_call_reports_an_unknown_cost(self):
        call_provider = AsyncMock(
            return_value=ProviderResponse(
                content=ONE_FACT, prompt_tokens=1, completion_tokens=1
            )
        )
        with patch(ROUTING, return_value=platform_routing()), patch(
            CALL_PROVIDER, call_provider
        ):
            result = await structured_complete(sync_ctx(), MESSAGES, SampleOutput)
        assert (result.usage.cost_usd, result.usage.cost_source) == (None, "none")

    @pytest.mark.asyncio
    async def test_routes_to_ollama_with_the_raw_host(self):
        """Local: ``provider="ollama"`` with ``ollama_host`` from
        ``CHAT_BASE_URL`` minus the OpenAI-compat ``/v1``."""
        call_provider = AsyncMock(
            return_value=ProviderResponse(
                content='{"facts": [{"content": "y", "confidence": 0.5}]}',
                prompt_tokens=8,
                completion_tokens=3,
            )
        )
        platform = platform_routing(
            "ollama", "ollama-placeholder", "http://localhost:11434/v1"
        )
        with patch(ROUTING, return_value=platform), patch(CALL_PROVIDER, call_provider):
            result = await structured_complete(
                sync_ctx(
                    "ollama", "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M", payer="local"
                ),
                MESSAGES,
                SampleOutput,
            )

        assert result.usage.payer == "local"
        kwargs = call_provider.call_args.kwargs
        assert kwargs["provider"] == "ollama"
        assert kwargs["api_key"] == "ollama-placeholder"
        assert kwargs["ollama_host"] == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_no_key_is_an_inference_error_before_any_call(self):
        call_provider = AsyncMock()
        with patch(ROUTING, return_value=platform_routing(api_key="")), patch(
            CALL_PROVIDER, call_provider
        ):
            with pytest.raises(InferenceError, match="OPEN_ROUTER_API_KEY"):
                await structured_complete(sync_ctx(), MESSAGES, SampleOutput)
        call_provider.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_batch_route_is_refused(self):
        ctx = sync_ctx().model_copy(update={"route": anthropic_batch_route("claude-x")})
        with pytest.raises(InferenceError, match="provider_batch"):
            await structured_complete(ctx, MESSAGES, SampleOutput)

    @pytest.mark.asyncio
    async def test_provider_failure_is_an_inference_error_without_usage(self):
        """So a caller's per-call failure handling runs, and bills nothing."""
        with patch(ROUTING, return_value=platform_routing()), patch(
            CALL_PROVIDER, AsyncMock(side_effect=RuntimeError("upstream 502"))
        ):
            with pytest.raises(InferenceError, match="upstream 502") as exc_info:
                await structured_complete(sync_ctx(), MESSAGES, SampleOutput)
        assert exc_info.value.usage is None

    @pytest.mark.parametrize(
        "content,match",
        [
            ("", "empty"),
            ('{"facts": [{"content": 1.0, "confidence": "wrong"}]}', "did not match"),
            ("no json here", "non-JSON"),
        ],
    )
    @pytest.mark.asyncio
    async def test_a_billed_answer_that_does_not_parse_carries_its_usage(
        self, content, match
    ):
        fake = ProviderResponse(content=content, prompt_tokens=9, completion_tokens=2)
        with patch(ROUTING, return_value=platform_routing()), patch(
            CALL_PROVIDER, AsyncMock(return_value=fake)
        ):
            with pytest.raises(InferenceError, match=match) as exc_info:
                await structured_complete(sync_ctx(), MESSAGES, SampleOutput)
        assert exc_info.value.usage is not None
        assert exc_info.value.usage.input_tokens == 9

    @pytest.mark.asyncio
    async def test_recovers_from_prose_prefixed_json(self):
        fake = ProviderResponse(
            content=(
                "Looking at the inputs, I need to ...\n\n"
                '{"facts": [{"content": "recovered", "confidence": 1.0}]}'
            ),
            prompt_tokens=10,
            completion_tokens=20,
        )
        with patch(ROUTING, return_value=platform_routing()), patch(
            CALL_PROVIDER, AsyncMock(return_value=fake)
        ):
            result = await structured_complete(sync_ctx(), MESSAGES, SampleOutput)
        assert result.value.facts[0].content == "recovered"


@pytest.mark.parametrize(
    "base_url,expected",
    [
        # The OpenAI-compat ``/v1`` suffix goes: ollama's client wants the host.
        ("http://localhost:11434/v1", "http://localhost:11434"),
        ("https://ollama.lan:11434/v1/", "https://ollama.lan:11434"),
        ("http://localhost:11434", "http://localhost:11434"),
        # A bare host:port passes through unchanged.
        ("localhost:11434", "localhost:11434"),
        # Empty/None falls back to the default.
        ("", "localhost:11434"),
        (None, "localhost:11434"),
    ],
)
def test_normalize_ollama_host(base_url: str | None, expected: str):
    assert _normalize_ollama_host(base_url) == expected
