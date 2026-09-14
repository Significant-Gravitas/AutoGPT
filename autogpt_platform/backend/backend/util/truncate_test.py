"""Unit tests for backend.util.truncate string size_limit guarantees."""

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
