"""Tests for backend.util.truncate.

These tests pin down the contract of :func:`truncate` — its docstring promises
that the returned value's string representation does not exceed ``size_limit``
characters. Historically ``_truncate_string_middle`` reserved the entire
``limit`` budget for kept content and then *appended* the
``"… (omitted N chars)…"`` sentinel on top, so truncated strings always
overshot the limit by ~22 chars. This file locks in the corrected behaviour.
"""

import pytest

from backend.util.truncate import truncate

SENTINEL_CHARS_OVERHEAD = 24  # generous upper bound for "… (omitted N chars)…"


class TestTruncateString:
    @pytest.mark.parametrize("size_limit", [50, 100, 200, 500])
    def test_truncated_string_respects_size_limit(self, size_limit):
        # GIVEN a string much larger than the requested limit
        big = "a" * 5000

        # WHEN we truncate it
        out = truncate(big, size_limit)

        # THEN the string representation must fit within the limit (the
        # whole point of the function is to bound output size).
        assert isinstance(out, str)
        assert len(str(out)) <= size_limit, (
            f"truncate(string, {size_limit}) returned {len(str(out))} chars; "
            f"output: {out!r}"
        )

    def test_short_string_returned_unchanged(self):
        # Strings already within the limit should pass through untouched.
        s = "hello world"
        assert truncate(s, 100) == s
        assert truncate(s, len(s)) == s

    def test_truncated_string_preserves_head_and_tail(self):
        # We still want to show the start and end of the original — that is
        # the whole purpose of "middle" truncation.
        s = "START_" + ("x" * 1000) + "_END"
        out = truncate(s, 80)
        assert isinstance(out, str)
        assert out.startswith("S")
        assert out.endswith("D")
        assert "omitted" in out
        assert len(out) <= 80

    def test_truncate_string_with_small_limit_is_safe(self):
        # Edge case: previously limit=1 produced ``head + sentinel + entire
        # original tail`` because ``value[-0:]`` returns the whole string.
        s = "abcdefghijklmnopqrstuvwxyz"
        out = truncate(s, 1)
        # We don't require it to be <= 1 (the sentinel itself is longer than
        # 1 char), but we *do* require the original string is no longer
        # smuggled in unredacted.
        assert s not in str(out)


class TestTruncateContainers:
    def test_dict_with_long_string_value_respects_limit(self):
        big = "z" * 5000
        out = truncate({"key": big}, 200)
        assert len(str(out)) <= 200 + SENTINEL_CHARS_OVERHEAD

    def test_list_with_long_string_respects_limit(self):
        big_items = ["q" * 1000 for _ in range(20)]
        out = truncate(big_items, 300)
        # Container truncation goes through the binary-search path which is
        # already bounded; the leaf string truncator was the bug.
        assert len(str(out)) <= 300 + SENTINEL_CHARS_OVERHEAD

    def test_non_string_non_container_passthrough(self):
        # Numbers, booleans, None should be returned as-is regardless of limit.
        assert truncate(42, 5) == 42
        assert truncate(True, 5) is True
        assert truncate(None, 5) is None
