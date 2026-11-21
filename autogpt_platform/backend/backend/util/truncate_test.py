"""Tests for the size-bounded truncation helper.

These cover the maximal-limits fast path in :func:`truncate` and the plain
string path.
"""

import random

import pytest

import backend.util.truncate as truncate_module
from backend.util.truncate import _truncate_value, truncate

OMISSION_MARKER = "… (omitted"


def measured(value: object) -> int:
    """The size definition `truncate` itself enforces."""
    return len(str(value))


class TestAlreadyFittingPayloads:
    """A payload that already fits must come back untouched."""

    def test_dict_at_the_limit_is_returned_verbatim(self):
        value = {"response": "y" * 84}
        assert measured(value) == 100

        result = truncate(value, 100)

        assert result == value
        assert OMISSION_MARKER not in str(result)

    def test_single_string_list_is_returned_verbatim(self):
        value = ["x" * 53]
        assert measured(value) == 57

        result = truncate(value, 64)

        assert result == value

    def test_result_never_exceeds_the_limit_it_was_given(self):
        for length, size_limit in ((13, 17), (14, 18), (16, 21)):
            value = [("x" * length)]
            assert measured(value) <= size_limit

            result = truncate(value, size_limit)

            assert measured(result) <= size_limit
            assert result == value

    def test_a_fitting_payload_is_probed_once(self, monkeypatch):
        value = {"response": "y" * 84}
        probes = []
        real = truncate_module._truncate_value

        def spy(val, *limits):
            if val is value:
                probes.append(limits)
            return real(val, *limits)

        monkeypatch.setattr(truncate_module, "_truncate_value", spy)

        assert truncate(value, 100) == value
        assert probes == [(100, 2**12)]

    def test_empty_and_scalar_payloads_are_untouched(self):
        for value in ({}, [], {"a": [], "b": {}, "c": ""}, 0, False, None):
            assert truncate(value, 100) == value


class TestOversizedPayloads:
    """When the maximal-limits candidate does not fit, the search still runs."""

    def test_oversized_string_field_is_truncated_middle_out(self):
        value = {"k": "x" * 5_000}

        result = truncate(value, 200)

        assert result != value
        assert OMISSION_MARKER in str(result)
        assert measured(result) <= 200

    def test_search_path_is_used_when_maximal_candidate_overshoots(self):
        # Multi-byte payload whose maximal-limits candidate measures 56 > 40,
        # so the fast path must decline and leave the search result in place.
        size_limit = 40
        value = {"a": {"b": "…" * (size_limit + 1)}}
        maximal = _truncate_value(value, size_limit, 2**12)
        assert measured(maximal) > size_limit

        result = truncate(value, size_limit)

        assert result != maximal
        assert OMISSION_MARKER in str(result)

    def test_long_lists_stay_capped_at_the_list_limit(self):
        value = [("w" * 4) for _ in range(6_000)]

        result = truncate(value, 8 * 1024 * 1024)

        assert isinstance(result, list)
        # 4096 retained items plus the middle-out sentinel
        assert len(result) == 2**12 + 1
        assert OMISSION_MARKER in str(result[len(result) // 2])

    def test_records_are_all_kept_when_cutting_their_strings_is_enough(self):
        rows = [{"id": i, "note": "x" * 200} for i in range(100)]

        result = truncate({"rows": rows}, 6_000)

        assert [row["id"] for row in result["rows"]] == list(range(100))
        assert measured(result) <= 6_000

    def test_nothing_fits_falls_back_to_the_tightest_truncation(self):
        value = {"k": "x" * 1_000}

        result = truncate(value, 1)

        assert result == {"k": "…"}


class TestCostOfOversizedPayloads:
    """An oversized payload costs a few builds, whatever it is made of."""

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param({"x": {f"{i:x}": 0 for i in range(3_000)}}, id="many-keys"),
            pytest.param({"x": [[1e15] * 300 for _ in range(40)]}, id="list-of-lists"),
            pytest.param([f"line {i}\n" * 30 for i in range(500)], id="long-strings"),
        ],
    )
    def test_the_whole_value_is_built_at_most_five_times(self, monkeypatch, value):
        builds = []
        real = truncate_module._truncate_value

        def spy(val, *limits):
            if val is value:
                builds.append(limits)
            return real(val, *limits)

        monkeypatch.setattr(truncate_module, "_truncate_value", spy)

        truncate(value, measured(value) // 3)

        assert len(builds) <= 5


@pytest.mark.parametrize("alphabet", ["abc def\n", "a'\"\n\\…é"])
def test_the_result_keeps_the_most_list_items_that_fit(alphabet: str):
    rng = random.Random(alphabet)

    def payload(depth: int) -> object:
        roll = rng.random()
        if depth > 2 or roll < 0.4:
            return rng.choice(
                [
                    "".join(rng.choices(alphabet, k=rng.randint(0, 50))),
                    rng.randint(-(10**6), 10**12),
                    None,
                ]
            )
        if roll < 0.75:
            return [payload(depth + 1) for _ in range(rng.randint(0, 12))]
        return {f"k{i}": payload(depth + 1) for i in range(rng.randint(0, 5))}

    for _ in range(300):
        value = [payload(1) for _ in range(rng.randint(1, 8))]
        size_limit = rng.randint(0, measured(value))
        str_min = min(8, size_limit)
        cuts = [_truncate_value(value, str_min, n) for n in range(1, 14)]
        fitting = [cut for cut in cuts if measured(cut) <= size_limit]

        result = truncate(value, size_limit)

        if not fitting:
            assert measured(result) == min(measured(cut) for cut in cuts)
            continue
        assert measured(result) <= size_limit
        assert kept_items(result) == kept_items(fitting[-1])


def kept_items(value: object) -> int:
    """List items kept at every depth, not counting the omission markers."""
    if isinstance(value, dict):
        return sum(kept_items(v) for v in value.values())
    if isinstance(value, list):
        return sum(1 + kept_items(v) for v in value if not _is_list_marker(v))
    return 0


def _is_list_marker(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(OMISSION_MARKER)
        and value.endswith(" items)…")
    )


class TestPlainStrings:
    """Plain strings bypass the search entirely, as before."""

    def test_short_string_is_unchanged(self):
        assert truncate("hello", 100) == "hello"

    def test_long_string_is_truncated_to_the_limit(self):
        result = truncate("x" * 500, 100)

        assert len(result) <= 100
        assert OMISSION_MARKER in result


class TestRepeatedCalls:
    """`truncate` is a pure function; repeated calls must not drift."""

    def test_repeated_calls_are_deterministic(self):
        value = {"s": "a" * 70, "l": [("b" * 30) for _ in range(5)], "n": 12_345}

        first = truncate(value, 100)
        second = truncate(value, 100)
        third = truncate(value, 100)

        assert first == second == third

    def test_input_is_not_mutated(self):
        value = {"s": "a" * 200, "l": [("b" * 90) for _ in range(9)]}
        before = str(value)

        truncate(value, 100)

        assert str(value) == before

    def test_truncating_a_fitting_result_again_is_a_no_op(self):
        value = {"response": "y" * 84}

        once = truncate(value, 100)
        twice = truncate(once, 100)

        assert twice == once


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
