"""Tests for AIConditionBlock – regression coverage for max_tokens and error propagation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.ai_condition import (
    MIN_LLM_OUTPUT_TOKENS,
    AIConditionBlock,
    _parse_boolean_response,
)
from backend.blocks.llm import (
    DEFAULT_LLM_MODEL,
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    AICredentials,
    LLMModel,
    LLMResponse,
)
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError
from backend.util.llm.providers import SMALL_CALL_REASONING_FLOOR

_TEST_AI_CREDENTIALS = cast(AICredentials, TEST_CREDENTIALS_INPUT)


# ---------------------------------------------------------------------------
# Helper to collect all yields from the async generator
# ---------------------------------------------------------------------------


async def _collect_outputs(block: AIConditionBlock, input_data, credentials):
    outputs: dict[str, object] = {}
    async for name, value in block.run(input_data, credentials=credentials):
        outputs[name] = value
    return outputs


def _make_input(**overrides) -> AIConditionBlock.Input:
    defaults: dict = {
        "input_value": "hello@example.com",
        "condition": "the input is an email address",
        "yes_value": "yes!",
        "no_value": "no!",
        "model": DEFAULT_LLM_MODEL,
        "credentials": TEST_CREDENTIALS_INPUT,
    }
    defaults.update(overrides)
    return AIConditionBlock.Input(**defaults)


def _mock_llm_response(
    response_text: str,
    *,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    provider_cost: float | None = None,
) -> LLMResponse:
    return LLMResponse(
        raw_response="",
        prompt=[],
        response=response_text,
        tool_calls=None,
        prompt_tokens=10,
        completion_tokens=5,
        reasoning=None,
        cache_read_tokens=cache_read_tokens,
        cache_creation_tokens=cache_creation_tokens,
        provider_cost=provider_cost,
    )


# ---------------------------------------------------------------------------
# _parse_boolean_response unit tests
# ---------------------------------------------------------------------------


class TestParseBooleanResponse:
    def test_true_exact(self):
        assert _parse_boolean_response("true") == (True, None)

    def test_false_exact(self):
        assert _parse_boolean_response("false") == (False, None)

    def test_true_with_whitespace(self):
        assert _parse_boolean_response("  True  ") == (True, None)

    def test_yes_fuzzy(self):
        assert _parse_boolean_response("Yes") == (True, None)

    def test_no_fuzzy(self):
        assert _parse_boolean_response("no") == (False, None)

    def test_one_fuzzy(self):
        assert _parse_boolean_response("1") == (True, None)

    def test_zero_fuzzy(self):
        assert _parse_boolean_response("0") == (False, None)

    def test_unclear_response(self):
        result, error = _parse_boolean_response("I'm not sure")
        assert result is False
        assert error is not None
        assert "Unclear" in error

    def test_conflicting_tokens(self):
        result, error = _parse_boolean_response("true and false")
        assert result is False
        assert error is not None


# ---------------------------------------------------------------------------
# Regression: max_tokens is set to MIN_LLM_OUTPUT_TOKENS
# ---------------------------------------------------------------------------


class TestMaxTokensRegression:
    @pytest.mark.asyncio
    async def test_llm_call_receives_min_output_tokens(self):
        """max_tokens must be MIN_LLM_OUTPUT_TOKENS (16) – the previous value
        of 1 was too low and caused OpenAI to reject the request."""
        block = AIConditionBlock()
        captured_kwargs: dict = {}

        async def spy_llm_call(**kwargs):
            captured_kwargs.update(kwargs)
            return _mock_llm_response("true")

        block.llm_call = spy_llm_call  # type: ignore[assignment]

        input_data = _make_input()
        await _collect_outputs(block, input_data, credentials=TEST_CREDENTIALS)

        assert captured_kwargs["max_tokens"] == MIN_LLM_OUTPUT_TOKENS
        assert captured_kwargs["max_tokens"] == 16


# ---------------------------------------------------------------------------
# The models #17071 moves workflows onto reason or think before they answer.
# Their thinking counts against the 16-token cap, so the request asks each one
# for its least reasoning, and raises the cap where it still reasons.
# ---------------------------------------------------------------------------


async def _run_to_the_provider(model: LLMModel) -> dict[str, Any]:
    """Run the block through the real ``llm_call`` and provider code, with the
    provider's SDK client replaced, and return the request it was sent."""
    openai_create = AsyncMock(return_value=SimpleNamespace())
    anthropic_create = AsyncMock(
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text="true")],
            usage=SimpleNamespace(
                input_tokens=40,
                output_tokens=1,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            stop_reason="end_turn",
        )
    )
    with (
        patch(
            "backend.util.llm.providers.openai.AsyncOpenAI",
            return_value=SimpleNamespace(
                responses=SimpleNamespace(create=openai_create)
            ),
        ),
        patch(
            "backend.util.llm.providers.anthropic.AsyncAnthropic",
            return_value=SimpleNamespace(
                messages=SimpleNamespace(create=anthropic_create)
            ),
        ),
        patch(
            "backend.util.llm.providers.extract_responses_tool_calls",
            return_value=None,
        ),
        patch(
            "backend.util.llm.providers.extract_responses_content",
            return_value="true",
        ),
        patch(
            "backend.util.llm.providers.extract_responses_reasoning",
            return_value=None,
        ),
        patch(
            "backend.util.llm.providers.extract_responses_usage",
            return_value=(40, 1),
        ),
    ):
        outputs = await _collect_outputs(
            AIConditionBlock(), _make_input(model=model), credentials=TEST_CREDENTIALS
        )
    assert outputs["result"] is True and "error" not in outputs
    (create,) = [c for c in (openai_create, anthropic_create) if c.await_count]
    return create.call_args.kwargs


class TestReasoningReplacementsAnswer:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("model", "effort", "cap"),
        [
            (LLMModel.GPT6_LUNA, "none", MIN_LLM_OUTPUT_TOKENS),
            (LLMModel.GPT6_1_TERRA, "none", MIN_LLM_OUTPUT_TOKENS),
            (LLMModel.GPT5_6_SOL, "none", MIN_LLM_OUTPUT_TOKENS),
            # No "none" or "minimal": it reasons at "low", so the cap leaves room.
            (LLMModel.GPT6_1_SOL, "low", SMALL_CALL_REASONING_FLOOR),
        ],
    )
    async def test_openai_replacements_get_their_least_effort(
        self, model: LLMModel, effort: str, cap: int
    ):
        request = await _run_to_the_provider(model)
        assert request["model"] == model.value
        assert request["reasoning"] == {"effort": effort}
        assert request["max_output_tokens"] == cap

    @pytest.mark.asyncio
    async def test_sonnet_5_5_answers_without_thinking_first(self):
        request = await _run_to_the_provider(LLMModel.CLAUDE_5_5_SONNET)
        assert request["model"] == LLMModel.CLAUDE_5_5_SONNET.value
        assert request["thinking"] == {"type": "between_tools"}
        assert request["max_tokens"] == MIN_LLM_OUTPUT_TOKENS


# ---------------------------------------------------------------------------
# Regression: exceptions from llm_call must propagate
# ---------------------------------------------------------------------------


class TestExceptionPropagation:
    @pytest.mark.asyncio
    async def test_llm_call_exception_propagates(self):
        """If llm_call raises, the exception must NOT be swallowed.
        Previously the block caught all exceptions and silently returned
        result=False."""
        block = AIConditionBlock()

        async def boom(**kwargs):
            raise RuntimeError("LLM provider error")

        block.llm_call = boom  # type: ignore[assignment]

        input_data = _make_input()
        with pytest.raises(RuntimeError, match="LLM provider error"):
            await _collect_outputs(block, input_data, credentials=TEST_CREDENTIALS)


# ---------------------------------------------------------------------------
# Regression: cache tokens and provider_cost must be propagated to stats
# ---------------------------------------------------------------------------


class TestCacheTokenPropagation:
    @pytest.mark.asyncio
    async def test_cache_tokens_propagated_to_stats(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """cache_read_tokens and cache_creation_tokens must be forwarded to
        NodeExecutionStats so that usage dashboards count cached tokens."""
        block = AIConditionBlock()

        async def spy_llm(**kwargs):
            return _mock_llm_response(
                "true",
                cache_read_tokens=7,
                cache_creation_tokens=3,
                provider_cost=0.0012,
            )

        monkeypatch.setattr(block, "llm_call", spy_llm)

        input_data = _make_input()
        await _collect_outputs(block, input_data, credentials=TEST_CREDENTIALS)

        assert block.execution_stats.cache_read_token_count == 7
        assert block.execution_stats.cache_creation_token_count == 3
        assert block.execution_stats.provider_cost == 0.0012


# ---------------------------------------------------------------------------
# Regression: an unclear answer still records the call's usage
# ---------------------------------------------------------------------------


class TestUsageOnErrorBranch:
    @pytest.mark.asyncio
    async def test_unclear_answer_keeps_the_calls_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """The provider billed the call, so its usage is recorded although the
        answer could not be read: the error output ends the run before any
        later line of the block runs."""
        block = AIConditionBlock()

        async def unclear_llm(**kwargs):
            return _mock_llm_response("maybe", provider_cost=0.002)

        monkeypatch.setattr(block, "llm_call", unclear_llm)

        with pytest.raises(BlockExecutionError, match="Unclear AI response"):
            async for _ in block.execute(
                _make_input().model_dump(mode="json"),
                execution_context=ExecutionContext(),
                credentials=TEST_CREDENTIALS,
            ):
                pass

        assert block.execution_stats.input_token_count > 0
        assert block.execution_stats.provider_cost == 0.002


class TestUsageOfAStoppedCall:
    @pytest.mark.asyncio
    async def test_a_stop_mid_call_records_the_requests_estimated_input(self):
        block = AIConditionBlock()
        request_sent = asyncio.Event()

        async def provider(**_: Any) -> None:
            request_sent.set()
            await asyncio.Event().wait()

        with patch("backend.util.llm.providers.call_provider", provider):
            run = asyncio.ensure_future(
                _collect_outputs(
                    block,
                    _make_input(input_value="hello@example.com " * 100),
                    credentials=TEST_CREDENTIALS,
                )
            )
            await asyncio.wait_for(request_sent.wait(), timeout=30)
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run

        assert block.execution_stats.abandoned_input_token_count > 100
        assert block.execution_stats.input_token_count == 0
