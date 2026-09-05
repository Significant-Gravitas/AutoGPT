import re

from backend.util.truncate import _truncate_string_middle, truncate


def test_reproduction_issue_14328():
    """Verify that truncate strictly respects size_limit for plain strings (fixes #14328)."""
    value = "0123456789" * 10  # 100 characters

    res_30 = truncate(value, 30)
    assert len(res_30) <= 30
    assert "… (omitted " in res_30

    res_1 = truncate(value, 1)
    assert len(res_1) <= 1
    assert res_1 == "…"


def test_string_within_limit_unchanged():
    """Values already within size_limit must not be altered."""
    assert truncate("hello world", 20) == "hello world"
    assert truncate("hello world", 11) == "hello world"
    assert _truncate_string_middle("abc", 5) == "abc"


def test_non_positive_limit():
    """Non-positive limits must return empty string and not fail."""
    assert truncate("hello world", 0) == ""
    assert truncate("hello world", -5) == ""
    assert _truncate_string_middle("hello world", 0) == ""
    assert _truncate_string_middle("hello world", -10) == ""


def test_small_limits():
    """Small limits fall back to concise ellipsis preserving the bound."""
    text = "abcdefghijklmnopqrstuvwxyz"
    assert truncate(text, 1) == "…"
    assert truncate(text, 2) == "a…"
    assert truncate(text, 3) == "a…z"
    assert truncate(text, 4) == "ab…z"
    assert truncate(text, 5) == "ab…yz"
    for lim in range(1, 10):
        res = truncate(text, lim)
        assert len(res) <= lim


def test_sentinel_accuracy_and_retained_content():
    """When the detailed sentinel fits, omitted count and retained boundaries must be exact."""
    value = "0123456789" * 10  # 100 characters
    limit = 35

    res = truncate(value, limit)
    assert len(res) <= limit

    match = re.search(r"^(.*?)… \(omitted (\d+) chars\)…(.*?)$", res)
    assert match is not None
    head, omitted_str, tail = match.groups()
    omitted = int(omitted_str)

    assert value.startswith(head)
    assert value.endswith(tail)
    assert len(head) + omitted + len(tail) == len(value)
    assert len(head) >= 1
    assert len(tail) >= 1


def test_recursive_truncation_nested_structures():
    """Recursive truncation on lists and dicts should respect size limit."""
    data = {
        "title": "A" * 100,
        "items": ["B" * 50 for _ in range(10)],
    }
    truncated = truncate(data, 100)
    assert len(str(truncated)) <= 100


def test_structured_values_at_small_and_non_positive_limits():
    """Structured values (lists and dicts) must never exceed size_limit, even for small/non-positive limits."""
    for val in [["x"], ["hello", "world"], {"a": "b"}, {"key": "very long value"}]:
        assert truncate(val, 0) == ""
        assert truncate(val, -3) == ""
        for lim in range(1, 5):
            res = truncate(val, lim)
            assert len(str(res)) <= lim, f"len({res!r}) > {lim} for value {val!r}"
