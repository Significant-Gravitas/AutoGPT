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


def test_truncate_dict_respects_size_limit():
    """A dict is bounded by the number of entries, not just by their values."""
    value = {f"field_{i:02d}": "v" * 40 for i in range(12)}

    assert len(str(truncate(value, 100))) <= 100


def test_truncate_dict_marks_the_entries_it_drops():
    result = truncate({f"field_{i:02d}": "v" * 40 for i in range(12)}, 100)

    assert any("omitted" in key for key in result)


def test_truncate_dict_keeps_every_key_when_they_fit():
    result = truncate({"a": "x" * 500, "b": "y" * 500}, 100)

    assert sorted(result) == ["a", "b"]
    assert len(str(result)) <= 100


def test_truncate_nested_dict_respects_size_limit():
    value = {"outer": {f"field_{i:02d}": "v" * 30 for i in range(10)}}

    assert len(str(truncate(value, 100))) <= 100


@pytest.mark.parametrize("size_limit", [2, 5, 20, 100])
def test_truncate_containers_respect_short_size_limits(size_limit: int):
    for value in (
        {"alpha": "a" * 50, "beta": "b" * 50},
        ["x" * 50, "y" * 50, "z" * 50],
    ):
        assert len(str(truncate(value, size_limit))) <= size_limit


def test_truncate_shortest_container_representation_is_empty():
    """``{}`` is two characters, so sizes below that cannot be met."""
    assert truncate({"a": "b" * 100}, 0) == {}


def test_truncate_exhausts_the_string_budget_before_dropping_entries():
    """A key survives when shrinking its value is enough to fit."""
    result = truncate({"a": "b" * 100}, 10)

    assert list(result) == ["a"]
    assert len(str(result)) <= 10
