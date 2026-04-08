"""Tests for AIConditionBlock – regression coverage for max_tokens and error propagation."""

from __future__ import annotations

from typing import cast

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
    LLMResponse,
)

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


def _mock_llm_response(response_text: str) -> LLMResponse:
    return LLMResponse(
        raw_response="",
        prompt=[],
        response=response_text,
        tool_calls=None,
        prompt_tokens=10,
        completion_tokens=5,
        reasoning=None,
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
