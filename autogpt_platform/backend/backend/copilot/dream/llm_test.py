"""Regression tests for dream-pass LLM JSON handling.

The structured_completion wrapper requests JSON mode, but some OpenRouter
upstreams (Claude family, certain Gemini variants) still wrap responses in
```json ... ``` markdown fences. Without stripping them the dream pass
aborts on the consolidation step with "Expecting value: line 1 column 1". This file pins
the fence-stripper that prevents the regression.
"""

from __future__ import annotations

import pytest

from .llm import _extract_first_json_object, _strip_json_code_fence


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
