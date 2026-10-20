"""Regression tests for the dream's LLM JSON handling.

``inference.complete.structured_complete`` requests JSON mode, but some
OpenRouter upstreams (Claude family, certain Gemini variants) still wrap
responses in ```json ... ``` markdown fences. Without stripping them the dream
pass aborts on the consolidation step with "Expecting value: line 1 column 1".
This file pins the fence-stripper and prose recovery that prevent the
regression, and which structured-output device each provider and model gets
(``structured_output.py``); the call itself is tested in
``inference/complete_test.py``.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from backend.copilot.inference.context import InferenceError, InferenceUsage
from backend.util.llm.tool_use import auto_tool_choice, force_tool_choice

from .llm import (
    _extract_first_json_object,
    _strip_json_code_fence,
    parse_structured_output,
)
from .structured_output import (
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
# parse_structured_output
# ---------------------------------------------------------------------------


class _Fact(BaseModel):
    content: str


_USAGE = InferenceUsage(model="m", input_tokens=9, payer="platform_allowance")


def test_parse_structured_output_validates_fenced_json():
    fenced = '```json\n{"content": "x"}\n```'
    assert parse_structured_output(fenced, _Fact, _USAGE) == _Fact(content="x")


@pytest.mark.parametrize(
    "content,match",
    [("", "empty"), ("prose only", "non-JSON"), ('{"other": 1}', "did not match")],
)
def test_parse_failures_carry_the_billed_usage(content: str, match: str):
    with pytest.raises(InferenceError, match=match) as exc_info:
        parse_structured_output(content, _Fact, _USAGE)
    assert exc_info.value.usage == _USAGE


# ---------------------------------------------------------------------------
# structured_request: which structured-output device each call gets
# ---------------------------------------------------------------------------


class _SampleFact(BaseModel):
    content: str
    confidence: float


class _SampleOutput(BaseModel):
    facts: list[_SampleFact]


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
