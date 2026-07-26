"""Tests for the size-bounded truncation helper.

These cover the maximal-limits fast path in :func:`truncate`, including the
cases where it diverges from the previous grid search, so the divergence is
pinned rather than incidental.
"""

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
        # `measure` is not monotonic in str_limit: the omission marker can be
        # longer than the text it replaces, so a mid-range str_limit inflates
        # a payload that already fitted. At these limits the grid search
        # inflated ["x" * 13] from 17 chars to 32, overshooting size_limit.
        for length, size_limit in ((13, 17), (14, 18), (16, 21)):
            value = [("x" * length)]
            assert measured(value) <= size_limit

            result = truncate(value, size_limit)

            assert measured(result) <= size_limit
            assert result == value

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
        # Multi-byte payload whose maximal-limits candidate measures 76 > 40,
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

    def test_nothing_fits_falls_back_to_the_tightest_truncation(self):
        value = {"k": "x" * 1_000}

        result = truncate(value, 1)

        assert result != value
        assert OMISSION_MARKER in str(result)


class TestPlainStrings:
    """Plain strings bypass the search entirely, as before."""

    def test_short_string_is_unchanged(self):
        assert truncate("hello", 100) == "hello"

    def test_long_string_is_truncated_to_the_limit(self):
        result = truncate("x" * 500, 100)

        assert len(result) > 0
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
