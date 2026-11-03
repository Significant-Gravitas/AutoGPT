import pytest

from backend.util.truncate import truncate


def test_truncate_string_respects_size_limit():
    value = "0123456789" * 10

    result = truncate(value, 30)

    assert len(result) <= 30
    assert result.startswith(value[0])
    assert result.endswith(value[-1])
    assert "omitted 91 chars" in result


@pytest.mark.parametrize("size_limit", [0, 1, 5, 20])
def test_truncate_string_respects_short_size_limits(size_limit: int):
    result = truncate("0123456789" * 10, size_limit)

    assert len(result) <= size_limit


def test_truncate_string_returns_value_within_limit_unchanged():
    value = "short value"

    assert truncate(value, len(value)) == value


def test_truncate_string_preserves_ends_when_full_marker_has_no_context():
    value = "0123456789" * 10

    result = truncate(value, 22)

    assert len(result) == 22
    assert result.startswith(value[0])
    assert result.endswith(value[-1])


@pytest.mark.parametrize("value", ["value", {"key": "value"}, 42])
def test_truncate_rejects_negative_size_limit(value: object):
    with pytest.raises(ValueError, match="size_limit must be non-negative"):
        truncate(value, -1)
