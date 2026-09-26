"""``structured_complete``: one sync call on the context's route, parsed.

Moved here with the call from ``dream/llm.py``; the parse helpers' own tests
stay in ``dream/llm_test.py``.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from backend.copilot.dream.structured_output import output_tool_name
from backend.copilot.transport_routing import ProviderRoutingKwargs
from backend.util.llm.conversions import ToolCall, ToolContentBlock
from backend.util.llm.providers import ProviderResponse
from backend.util.llm.tool_use import auto_tool_choice, force_tool_choice

from .complete import _normalize_ollama_host, structured_complete
from .context import (
    InferenceContext,
    InferenceError,
    InferenceJob,
    InferenceScope,
    RouteDecision,
)
from .routing import anthropic_batch_route

_ROUTING = "backend.copilot.inference.routing.routing_kwargs_for_chat_transport"
_CALL_PROVIDER = "backend.copilot.inference.complete.call_provider"


class _SampleFact(BaseModel):
    content: str
    confidence: float


class _SampleOutput(BaseModel):
    facts: list[_SampleFact]


_MESSAGES = [{"role": "user", "content": "give me a fact"}]
_ONE_FACT = '{"facts": [{"content": "x", "confidence": 0.9}]}'


def _platform(provider="open_router", api_key="sk-or-test", base_url=None):
    return ProviderRoutingKwargs(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        supports_flex=provider == "open_router",
        cost_log_provider={"open_router": "open_router", "ollama": "ollama"}.get(
            provider, "anthropic"
        ),
    )


def _ctx(
    provider="open_router",
    model="anthropic/claude-sonnet-4-6",
    *,
    timeout_seconds: float | None = None,
    payer="platform_allowance",
) -> InferenceContext:
    return InferenceContext(
        scope=InferenceScope(user_id="u1"),
        job=InferenceJob(
            kind="dream",
            phase="consolidate",
            correlation_id="pass-1",
            latency_class="deferred",
            tier="standard",
            timeout_seconds=timeout_seconds,
        ),
        route=RouteDecision(
            engine="provider_sync",
            auth_provider="platform",
            provider=provider,
            model=model,
            payer=payer,
            execution_path="sync_baseline",
            cost_log_provider="test",
            reason="test",
        ),
    )


class TestDelegation:
    """``structured_complete`` hands ``call_provider`` the route's provider
    and model with the platform's key, and turns the ``ProviderResponse``
    into the typed value and an ``InferenceUsage``."""

    @pytest.mark.asyncio
    async def test_delegates_to_call_provider_with_the_route(self):
        fake = ProviderResponse(
            content=_ONE_FACT, prompt_tokens=12, completion_tokens=4, cost_usd=0.0042
        )
        call_provider = AsyncMock(return_value=fake)
        with patch(_ROUTING, return_value=_platform()), patch(
            _CALL_PROVIDER, call_provider
        ):
            result = await structured_complete(
                _ctx(),
                _MESSAGES,
                _SampleOutput,
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
                content=_ONE_FACT, prompt_tokens=1, completion_tokens=1
            )
        )
        with patch(_ROUTING, return_value=_platform()), patch(
            _CALL_PROVIDER, call_provider
        ):
            await structured_complete(
                _ctx(timeout_seconds=600), _MESSAGES, _SampleOutput
            )
            await structured_complete(_ctx(), _MESSAGES, _SampleOutput)

        first, second = call_provider.call_args_list
        assert first.kwargs["timeout_seconds"] == 600
        # No budget forwards the sentinel, so call_provider reads the live
        # default rather than a def-time snapshot of it.
        assert second.kwargs["timeout_seconds"] is None

    @pytest.mark.asyncio
    async def test_an_unpriced_call_reports_an_unknown_cost(self):
        call_provider = AsyncMock(
            return_value=ProviderResponse(
                content=_ONE_FACT, prompt_tokens=1, completion_tokens=1
            )
        )
        with patch(_ROUTING, return_value=_platform()), patch(
            _CALL_PROVIDER, call_provider
        ):
            result = await structured_complete(_ctx(), _MESSAGES, _SampleOutput)
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
        platform = _platform(
            "ollama", "ollama-placeholder", "http://localhost:11434/v1"
        )
        with patch(_ROUTING, return_value=platform), patch(
            _CALL_PROVIDER, call_provider
        ):
            result = await structured_complete(
                _ctx("ollama", "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M", payer="local"),
                _MESSAGES,
                _SampleOutput,
            )

        assert result.usage.payer == "local"
        kwargs = call_provider.call_args.kwargs
        assert kwargs["provider"] == "ollama"
        assert kwargs["api_key"] == "ollama-placeholder"
        assert kwargs["ollama_host"] == "http://localhost:11434"

    @pytest.mark.asyncio
    async def test_no_key_is_an_inference_error_before_any_call(self):
        call_provider = AsyncMock()
        with patch(_ROUTING, return_value=_platform(api_key="")), patch(
            _CALL_PROVIDER, call_provider
        ):
            with pytest.raises(InferenceError, match="OPEN_ROUTER_API_KEY"):
                await structured_complete(_ctx(), _MESSAGES, _SampleOutput)
        call_provider.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_batch_route_is_refused(self):
        ctx = _ctx().model_copy(update={"route": anthropic_batch_route("claude-x")})
        with pytest.raises(InferenceError, match="provider_batch"):
            await structured_complete(ctx, _MESSAGES, _SampleOutput)

    @pytest.mark.asyncio
    async def test_provider_failure_is_an_inference_error_without_usage(self):
        """So a caller's per-call failure handling runs, and bills nothing."""
        with patch(_ROUTING, return_value=_platform()), patch(
            _CALL_PROVIDER, AsyncMock(side_effect=RuntimeError("upstream 502"))
        ):
            with pytest.raises(InferenceError, match="upstream 502") as exc_info:
                await structured_complete(_ctx(), _MESSAGES, _SampleOutput)
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
        with patch(_ROUTING, return_value=_platform()), patch(
            _CALL_PROVIDER, AsyncMock(return_value=fake)
        ):
            with pytest.raises(InferenceError, match=match) as exc_info:
                await structured_complete(_ctx(), _MESSAGES, _SampleOutput)
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
        with patch(_ROUTING, return_value=_platform()), patch(
            _CALL_PROVIDER, AsyncMock(return_value=fake)
        ):
            result = await structured_complete(_ctx(), _MESSAGES, _SampleOutput)
        assert result.value.facts[0].content == "recovered"


def _bad_request(message: str) -> anthropic.BadRequestError:
    response = httpx.Response(
        400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    return anthropic.BadRequestError(message, response=response, body=None)


def _server_error(message: str) -> anthropic.InternalServerError:
    response = httpx.Response(
        500, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    return anthropic.InternalServerError(message, response=response, body=None)


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


_ANTHROPIC = _platform("anthropic", "sk-ant-test")


class TestAnthropicToolPath:
    """The native Anthropic API ignores JSON mode, so structured output
    comes from one forced tool built from the response model, read back
    from the tool call, with the model in its native spelling."""

    @pytest.mark.asyncio
    async def test_forces_one_tool_and_sends_the_native_model(self):
        fake = _tool_response(
            _ONE_FACT, prompt_tokens=12, completion_tokens=4, cache_read_tokens=3
        )
        call_provider = AsyncMock(return_value=fake)
        with patch(_ROUTING, return_value=_ANTHROPIC), patch(
            _CALL_PROVIDER, call_provider
        ):
            result = await structured_complete(
                _ctx("anthropic", "claude-sonnet-5"), _MESSAGES, _SampleOutput
            )

        assert result.value.facts[0].content == "x"
        kwargs = call_provider.call_args.kwargs
        assert kwargs["provider"] == "anthropic"
        assert kwargs["model"] == "claude-sonnet-5"
        assert kwargs["force_json_output"] is False
        tool_name = output_tool_name(_SampleOutput)
        assert [tool["name"] for tool in kwargs["tools"]] == [tool_name]
        assert kwargs["tools"][0]["input_schema"]["required"] == ["facts"]
        assert kwargs["tool_choice"] == force_tool_choice(tool_name)
        assert kwargs["tools"][0]["description"] == (
            "Return the _SampleOutput result: call this once, with every "
            "field the schema requires."
        )
        # A forced tool needs no prompt line asking for it.
        assert kwargs["messages"] == _MESSAGES
        # Usage names the model that was called, the spelling the price card
        # resolves, with its tokens.
        assert result.usage.model == "claude-sonnet-5"
        assert (result.usage.input_tokens, result.usage.cache_read_tokens) == (12, 3)

    @pytest.mark.asyncio
    async def test_opus_5_5_gets_auto_and_the_prompt_asks_for_the_call(self):
        """Opus 5.5 answers a forced ``tool_choice`` with a 400, so the tool
        goes out under ``auto``, and both its description and the last user
        turn ask for one call with the complete result."""
        call_provider = AsyncMock(return_value=_tool_response(_ONE_FACT))
        with patch(_ROUTING, return_value=_ANTHROPIC), patch(
            _CALL_PROVIDER, call_provider
        ):
            result = await structured_complete(
                _ctx("anthropic", "claude-opus-5-5"),
                [
                    {"role": "system", "content": "you consolidate facts"},
                    {"role": "user", "content": "give me a fact"},
                ],
                _SampleOutput,
            )

        assert result.value.facts[0].content == "x"
        kwargs = call_provider.call_args.kwargs
        tool_name = output_tool_name(_SampleOutput)
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
        with patch(_ROUTING, return_value=_ANTHROPIC), patch(
            _CALL_PROVIDER, AsyncMock(return_value=fake)
        ):
            result = await structured_complete(
                _ctx("anthropic", "claude-opus-5.5"), _MESSAGES, _SampleOutput
            )
        assert result.value.facts[0].content == "text"
        assert result.usage.model == "claude-opus-5-5"

    @pytest.mark.asyncio
    async def test_tool_arguments_off_schema_raise_with_usage(self):
        """The tool call was billed even when its arguments don't fit."""
        fake = _tool_response('{"facts": "not a list"}', prompt_tokens=7)
        with patch(_ROUTING, return_value=_ANTHROPIC), patch(
            _CALL_PROVIDER, AsyncMock(return_value=fake)
        ):
            with pytest.raises(InferenceError, match="did not match") as exc_info:
                await structured_complete(
                    _ctx("anthropic", "claude-sonnet-5"), _MESSAGES, _SampleOutput
                )
        assert exc_info.value.usage is not None
        assert exc_info.value.usage.input_tokens == 7

    @pytest.mark.asyncio
    async def test_non_anthropic_model_fails_before_any_call(self):
        call_provider = AsyncMock()
        with patch(_ROUTING, return_value=_ANTHROPIC), patch(
            _CALL_PROVIDER, call_provider
        ):
            with pytest.raises(InferenceError, match="requires an Anthropic model"):
                await structured_complete(
                    _ctx("anthropic", "openai/gpt-4.1-mini"), _MESSAGES, _SampleOutput
                )
        call_provider.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_through_the_real_anthropic_messages_call(self):
        """End to end through ``call_provider``: the tool reaches the Messages
        API carrying the schema, and its tool_use block comes back as the
        parsed value."""
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
        with patch(_ROUTING, return_value=_ANTHROPIC), patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=client,
        ):
            result = await structured_complete(
                _ctx("anthropic", "claude-sonnet-5"),
                [
                    {"role": "system", "content": "you consolidate facts"},
                    {"role": "user", "content": "facts"},
                ],
                _SampleOutput,
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
        with patch(_ROUTING, return_value=_ANTHROPIC), patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=client,
        ):
            result = await structured_complete(
                _ctx("anthropic", "claude-opus-5-5"),
                [
                    {"role": "system", "content": "you consolidate facts"},
                    {"role": "user", "content": "facts"},
                ],
                _SampleOutput,
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
        healed = _tool_response('{"facts": [{"content": "healed", "confidence": 0.7}]}')
        call_provider = AsyncMock(
            side_effect=[_bad_request(_FORCED_TOOL_REJECTION), healed]
        )
        with patch(_ROUTING, return_value=_ANTHROPIC), patch(
            _CALL_PROVIDER, call_provider
        ), caplog.at_level(
            logging.WARNING, logger="backend.copilot.inference.complete"
        ):
            result = await structured_complete(
                _ctx("anthropic", "claude-sonnet-5"),
                [{"role": "user", "content": "hi"}],
                _SampleOutput,
            )

        assert result.value.facts[0].content == "healed"
        tool_name = output_tool_name(_SampleOutput)
        first, second = call_provider.call_args_list
        assert first.kwargs["tool_choice"] == force_tool_choice(tool_name)
        assert first.kwargs["messages"] == [{"role": "user", "content": "hi"}]
        assert second.kwargs["tool_choice"] == auto_tool_choice()
        assert second.kwargs["model"] == "claude-sonnet-5"
        assert tool_name in second.kwargs["messages"][-1]["content"]
        assert "retrying once with tool_choice=auto" in caplog.text

    @pytest.mark.asyncio
    async def test_forced_tool_rejection_is_retried_only_once(self):
        call_provider = AsyncMock(
            side_effect=[
                _bad_request(_FORCED_TOOL_REJECTION),
                _bad_request(_FORCED_TOOL_REJECTION),
            ]
        )
        with patch(_ROUTING, return_value=_ANTHROPIC), patch(
            _CALL_PROVIDER, call_provider
        ):
            with pytest.raises(InferenceError, match="tool_choice") as exc_info:
                await structured_complete(
                    _ctx("anthropic", "claude-sonnet-5"),
                    [{"role": "user", "content": "hi"}],
                    _SampleOutput,
                )
        assert call_provider.await_count == 2
        # Neither call came back with a response, so nothing was billed.
        assert exc_info.value.usage is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [_server_error(_FORCED_TOOL_REJECTION), RuntimeError(_FORCED_TOOL_REJECTION)],
        ids=["500", "runtime-error"],
    )
    async def test_only_a_400_rejection_is_retried(self, error: Exception):
        """The documented text is not enough: a 5xx or an exception of our
        own quoting it keeps the forced tool and fails once."""
        call_provider = AsyncMock(side_effect=error)
        with patch(_ROUTING, return_value=_ANTHROPIC), patch(
            _CALL_PROVIDER, call_provider
        ):
            with pytest.raises(InferenceError, match="tool_choice"):
                await structured_complete(
                    _ctx("anthropic", "claude-sonnet-5"),
                    [{"role": "user", "content": "hi"}],
                    _SampleOutput,
                )
        assert call_provider.await_count == 1

    @pytest.mark.asyncio
    async def test_other_bad_requests_are_not_retried(self):
        call_provider = AsyncMock(
            side_effect=_bad_request("messages: at least one message is required")
        )
        with patch(_ROUTING, return_value=_ANTHROPIC), patch(
            _CALL_PROVIDER, call_provider
        ):
            with pytest.raises(InferenceError, match="at least one message"):
                await structured_complete(
                    _ctx("anthropic", "claude-sonnet-5"),
                    [{"role": "user", "content": "hi"}],
                    _SampleOutput,
                )
        assert call_provider.await_count == 1


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
