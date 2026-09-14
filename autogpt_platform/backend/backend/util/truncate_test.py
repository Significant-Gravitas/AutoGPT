"""Unit tests for backend.util.truncate string size_limit guarantees."""

import re

import pytest

from backend.util.truncate import _truncate_string_middle, truncate

LONG = "0123456789" * 10  # 100 chars


def test_truncate_respects_limit_30():
    result = truncate(LONG, 30)
    assert len(result) <= 30


def test_truncate_respects_limit_1():
    result = truncate(LONG, 1)
    assert len(result) <= 1


def test_truncate_respects_limit_0():
    result = truncate(LONG, 0)
    assert len(result) <= 0
    assert result == ""


def test_truncate_within_limit_unchanged():
    value = "short"
    assert truncate(value, 10) == value
    assert truncate(value, len(value)) == value


def test_truncate_exact_limit_unchanged():
    value = "0123456789"
    assert truncate(value, 10) == value


def test_truncate_string_middle_never_exceeds_nonneg_limit():
    for limit in (0, 1, 5, 10, 21, 22, 30, 50, 100, 200):
        result = _truncate_string_middle(LONG, limit)
        assert len(result) <= max(0, limit), (limit, len(result), result)


def test_truncate_string_middle_negative_limit_empty():
    assert _truncate_string_middle(LONG, -1) == ""


def test_truncate_keeps_head_and_tail_when_space_allows():
    result = truncate(LONG, 30)
    assert result.startswith("0123")
    assert result.endswith("56789")
    assert "omitted" in result


@pytest.mark.parametrize("length", [0, 1, 9, 10, 99, 100, 101, 999, 1000, 1001])
def test_string_boundaries_preserve_content_and_report_exact_omissions(length):
    value = "".join(chr(0x4E00 + index) for index in range(length))
    for limit in [-5, 0, 1, 2, 19, 20, 21, 22, 23, 30, 50, 100, 999, 1000, 1001]:
        result = truncate(value, limit)
        assert len(result) <= max(0, limit)
        if limit <= 0:
            assert result == ""
        elif length <= limit:
            assert result == value
        elif match := re.search(r"… \(omitted (\d+) chars\)…", result):
            head, tail = result[: match.start()], result[match.end() :]
            assert value.startswith(head)
            assert value.endswith(tail)
            assert int(match.group(1)) == length - len(head) - len(tail)
            retained = len(head) + len(tail)
            assert all(
                candidate + len(f"… (omitted {length - candidate} chars)…") > limit
                for candidate in range(retained + 1, min(length, limit + 1))
            )
        else:
            assert result == value[:limit]
            assert len(f"… (omitted {length} chars)…") > limit


@pytest.mark.parametrize(
    "length, limit, omitted",
    [(200, 121, 101), (200, 122, 99), (1100, 122, 1001), (1100, 123, 999)],
)
def test_omission_count_digit_boundaries_keep_the_largest_fitting_content(
    length, limit, omitted
):
    value = "x" * length
    result = truncate(value, limit)
    marker = f"… (omitted {omitted} chars)…"
    assert marker in result
    assert len(result) == limit
    assert len(result.replace(marker, "")) == length - omitted


@pytest.mark.parametrize(
    "length, limit, expected",
    [
        (100, 21, "x" * 21),
        (100, 22, "x… (omitted 99 chars)…"),
        (100, 23, "x… (omitted 98 chars)…x"),
        (200, 22, "… (omitted 200 chars)…"),
        (200, 23, "x… (omitted 199 chars)…"),
        (200, 24, "x… (omitted 198 chars)…x"),
    ],
)
def test_marker_only_and_one_character_budgets_do_not_expand_the_string(
    length, limit, expected
):
    assert truncate("x" * length, limit) == expected


@pytest.mark.parametrize("limit", [1, 21, 22, 30, 80])
def test_unicode_limits_count_characters_and_preserve_prefix_and_suffix(limit):
    value = "頭🧪e\u0301🙂尾" * 50
    result = truncate(value, limit)
    assert len(result) <= limit
    result.encode("utf-8").decode("utf-8")
    if match := re.search(r"… \(omitted (\d+) chars\)…", result):
        head, tail = result[: match.start()], result[match.end() :]
        assert head == value[: len(head)]
        assert tail == value[len(value) - len(tail) :]
        assert int(match.group(1)) == len(value) - len(head) - len(tail)
    else:
        assert result == value[:limit]
