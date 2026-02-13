"""
Comprehensive test suite for list concatenation and manipulation blocks.

Tests cover:
- ConcatenateListsBlock: basic concatenation, deduplication, None removal
- FlattenListBlock: nested list flattening with depth control
- InterleaveListsBlock: round-robin interleaving of multiple lists
- ZipListsBlock: zipping lists with truncation and padding
- ListDifferenceBlock: computing list differences (regular and symmetric)
- ListIntersectionBlock: finding common elements between lists
- Helper utility functions: validation, flattening, deduplication, etc.
"""

import pytest

from backend.blocks.data_manipulation import (
    _MAX_FLATTEN_DEPTH,
    ConcatenateListsBlock,
    FlattenListBlock,
    InterleaveListsBlock,
    ListDifferenceBlock,
    ListIntersectionBlock,
    ZipListsBlock,
    _compute_nesting_depth,
    _concatenate_lists_simple,
    _deduplicate_list,
    _filter_none_values,
    _flatten_nested_list,
    _interleave_lists,
    _make_hashable,
    _validate_all_lists,
    _validate_list_input,
)
from backend.util.test import execute_block_test

# =============================================================================
# Helper Function Tests
# =============================================================================


class TestValidateListInput:
    """Tests for the _validate_list_input helper."""

    def test_valid_list_returns_none(self):
        assert _validate_list_input([1, 2, 3], 0) is None

    def test_empty_list_returns_none(self):
        assert _validate_list_input([], 0) is None

    def test_none_returns_none(self):
        assert _validate_list_input(None, 0) is None

    def test_string_returns_error(self):
        result = _validate_list_input("hello", 0)
        assert result is not None
        assert "str" in result
        assert "index 0" in result

    def test_integer_returns_error(self):
        result = _validate_list_input(42, 1)
        assert result is not None
        assert "int" in result
        assert "index 1" in result

    def test_dict_returns_error(self):
        result = _validate_list_input({"a": 1}, 2)
        assert result is not None
        assert "dict" in result
        assert "index 2" in result

    def test_tuple_returns_error(self):
        result = _validate_list_input((1, 2), 3)
        assert result is not None
        assert "tuple" in result

    def test_boolean_returns_error(self):
        result = _validate_list_input(True, 0)
        assert result is not None
        assert "bool" in result

    def test_float_returns_error(self):
        result = _validate_list_input(3.14, 0)
        assert result is not None
        assert "float" in result


class TestValidateAllLists:
    """Tests for the _validate_all_lists helper."""

    def test_all_valid_lists(self):
        assert _validate_all_lists([[1], [2], [3]]) is None

    def test_empty_outer_list(self):
        assert _validate_all_lists([]) is None

    def test_mixed_valid_and_none(self):
        # None is skipped, so this should pass
        assert _validate_all_lists([[1], None, [3]]) is None

    def test_invalid_item_returns_error(self):
        result = _validate_all_lists([[1], "bad", [3]])
        assert result is not None
        assert "index 1" in result

    def test_first_invalid_is_returned(self):
        result = _validate_all_lists(["first_bad", "second_bad"])
        assert result is not None
        assert "index 0" in result

    def test_all_none_passes(self):
        assert _validate_all_lists([None, None, None]) is None


class TestConcatenateListsSimple:
    """Tests for the _concatenate_lists_simple helper."""

    def test_basic_concatenation(self):
        assert _concatenate_lists_simple([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_empty_lists(self):
        assert _concatenate_lists_simple([[], []]) == []

    def test_single_list(self):
        assert _concatenate_lists_simple([[1, 2, 3]]) == [1, 2, 3]

    def test_no_lists(self):
        assert _concatenate_lists_simple([]) == []

    def test_skip_none_values(self):
        assert _concatenate_lists_simple([[1, 2], None, [3, 4]]) == [1, 2, 3, 4]  # type: ignore[arg-type]

    def test_mixed_types(self):
        result = _concatenate_lists_simple([[1, "a"], [True, 3.14]])
        assert result == [1, "a", True, 3.14]

    def test_nested_lists_preserved(self):
        result = _concatenate_lists_simple([[[1, 2]], [[3, 4]]])
        assert result == [[1, 2], [3, 4]]

    def test_large_number_of_lists(self):
        lists = [[i] for i in range(100)]
        result = _concatenate_lists_simple(lists)
        assert result == list(range(100))


class TestFlattenNestedList:
    """Tests for the _flatten_nested_list helper."""

    def test_already_flat(self):
        assert _flatten_nested_list([1, 2, 3]) == [1, 2, 3]

    def test_one_level_nesting(self):
        assert _flatten_nested_list([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_deep_nesting(self):
        assert _flatten_nested_list([1, [2, [3, [4, [5]]]]]) == [1, 2, 3, 4, 5]

    def test_empty_list(self):
        assert _flatten_nested_list([]) == []

    def test_mixed_nesting(self):
        assert _flatten_nested_list([1, [2, 3], 4, [5, [6]]]) == [1, 2, 3, 4, 5, 6]

    def test_max_depth_zero(self):
        # max_depth=0 means no flattening at all
        result = _flatten_nested_list([[1, 2], [3, 4]], max_depth=0)
        assert result == [[1, 2], [3, 4]]

    def test_max_depth_one(self):
        result = _flatten_nested_list([[1, [2, 3]], [4, [5]]], max_depth=1)
        assert result == [1, [2, 3], 4, [5]]

    def test_max_depth_two(self):
        result = _flatten_nested_list([[[1, 2], [3]], [[4, [5]]]], max_depth=2)
        assert result == [1, 2, 3, 4, [5]]

    def test_unlimited_depth(self):
        deeply_nested = [[[[[[[1]]]]]]]
        assert _flatten_nested_list(deeply_nested, max_depth=-1) == [1]

    def test_preserves_non_list_iterables(self):
        result = _flatten_nested_list(["hello", [1, 2]])
        assert result == ["hello", 1, 2]

    def test_preserves_dicts(self):
        result = _flatten_nested_list([{"a": 1}, [{"b": 2}]])
        assert result == [{"a": 1}, {"b": 2}]

    def test_excessive_depth_raises_recursion_error(self):
        """Deeply nested lists beyond 1000 levels should raise RecursionError."""
        # Build a list nested 1100 levels deep
        nested = [42]
        for _ in range(1100):
            nested = [nested]
        with pytest.raises(RecursionError, match="maximum.*depth"):
            _flatten_nested_list(nested, max_depth=-1)


class TestDeduplicateList:
    """Tests for the _deduplicate_list helper."""

    def test_no_duplicates(self):
        assert _deduplicate_list([1, 2, 3]) == [1, 2, 3]

    def test_with_duplicates(self):
        assert _deduplicate_list([1, 2, 2, 3, 3, 3]) == [1, 2, 3]

    def test_all_duplicates(self):
        assert _deduplicate_list([1, 1, 1]) == [1]

    def test_empty_list(self):
        assert _deduplicate_list([]) == []

    def test_preserves_order(self):
        result = _deduplicate_list([3, 1, 2, 1, 3])
        assert result == [3, 1, 2]

    def test_string_duplicates(self):
        assert _deduplicate_list(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_mixed_types(self):
        result = _deduplicate_list([1, "1", 1, "1"])
        assert result == [1, "1"]

    def test_dict_duplicates(self):
        result = _deduplicate_list([{"a": 1}, {"a": 1}, {"b": 2}])
        assert result == [{"a": 1}, {"b": 2}]

    def test_list_duplicates(self):
        result = _deduplicate_list([[1, 2], [1, 2], [3, 4]])
        assert result == [[1, 2], [3, 4]]

    def test_none_duplicates(self):
        result = _deduplicate_list([None, 1, None, 2])
        assert result == [None, 1, 2]

    def test_single_element(self):
        assert _deduplicate_list([42]) == [42]


class TestMakeHashable:
    """Tests for the _make_hashable helper."""

    def test_integer(self):
        assert _make_hashable(42) == 42

    def test_string(self):
        assert _make_hashable("hello") == "hello"

    def test_none(self):
        assert _make_hashable(None) is None

    def test_dict_returns_tuple(self):
        result = _make_hashable({"a": 1})
        assert isinstance(result, tuple)
        # Should be hashable
        hash(result)

    def test_list_returns_tuple(self):
        result = _make_hashable([1, 2, 3])
        assert result == (1, 2, 3)

    def test_same_dict_same_hash(self):
        assert _make_hashable({"a": 1, "b": 2}) == _make_hashable({"a": 1, "b": 2})

    def test_different_dict_different_hash(self):
        assert _make_hashable({"a": 1}) != _make_hashable({"a": 2})

    def test_dict_key_order_independent(self):
        """Dicts with same keys in different insertion order produce same result."""
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        assert _make_hashable(d1) == _make_hashable(d2)

    def test_tuple_hashable(self):
        result = _make_hashable((1, 2, 3))
        assert result == (1, 2, 3)
        hash(result)

    def test_boolean(self):
        result = _make_hashable(True)
        assert result is True

    def test_float(self):
        result = _make_hashable(3.14)
        assert result == 3.14


class TestFilterNoneValues:
    """Tests for the _filter_none_values helper."""

    def test_removes_none(self):
        assert _filter_none_values([1, None, 2, None, 3]) == [1, 2, 3]

    def test_no_none(self):
        assert _filter_none_values([1, 2, 3]) == [1, 2, 3]

    def test_all_none(self):
        assert _filter_none_values([None, None, None]) == []

    def test_empty_list(self):
        assert _filter_none_values([]) == []

    def test_preserves_falsy_values(self):
        assert _filter_none_values([0, False, "", None, []]) == [0, False, "", []]


class TestComputeNestingDepth:
    """Tests for the _compute_nesting_depth helper."""

    def test_flat_list(self):
        assert _compute_nesting_depth([1, 2, 3]) == 1

    def test_one_level(self):
        assert _compute_nesting_depth([[1, 2], [3, 4]]) == 2

    def test_deep_nesting(self):
        assert _compute_nesting_depth([[[[]]]]) == 4

    def test_mixed_depth(self):
        depth = _compute_nesting_depth([1, [2, [3]]])
        assert depth == 3

    def test_empty_list(self):
        assert _compute_nesting_depth([]) == 1

    def test_non_list(self):
        assert _compute_nesting_depth(42) == 0

    def test_string_not_recursed(self):
        # Strings should not be treated as nested lists
        assert _compute_nesting_depth(["hello"]) == 1


class TestInterleaveListsHelper:
    """Tests for the _interleave_lists helper."""

    def test_equal_length_lists(self):
        result = _interleave_lists([[1, 2, 3], ["a", "b", "c"]])
        assert result == [1, "a", 2, "b", 3, "c"]

    def test_unequal_length_lists(self):
        result = _interleave_lists([[1, 2, 3], ["a"]])
        assert result == [1, "a", 2, 3]

    def test_empty_input(self):
        assert _interleave_lists([]) == []

    def test_single_list(self):
        assert _interleave_lists([[1, 2, 3]]) == [1, 2, 3]

    def test_three_lists(self):
        result = _interleave_lists([[1], [2], [3]])
        assert result == [1, 2, 3]

    def test_with_none_list(self):
        result = _interleave_lists([[1, 2], None, [3, 4]])  # type: ignore[arg-type]
        assert result == [1, 3, 2, 4]

    def test_all_empty_lists(self):
        assert _interleave_lists([[], [], []]) == []

    def test_all_none_lists(self):
        """All-None inputs should return empty list, not crash."""
        assert _interleave_lists([None, None, None]) == []  # type: ignore[arg-type]


class TestComputeNestingDepthEdgeCases:
    """Tests for _compute_nesting_depth with deeply nested input."""

    def test_deeply_nested_does_not_crash(self):
        """Deeply nested lists beyond 1000 levels should not raise RecursionError."""
        nested = [42]
        for _ in range(1100):
            nested = [nested]
        # Should return a depth value without crashing
        depth = _compute_nesting_depth(nested)
        assert depth >= _MAX_FLATTEN_DEPTH


class TestMakeHashableMixedKeys:
    """Tests for _make_hashable with mixed-type dict keys."""

    def test_mixed_type_dict_keys(self):
        """Dicts with mixed-type keys (int and str) should not crash sorted()."""
        d = {1: "one", "two": 2}
        result = _make_hashable(d)
        assert isinstance(result, tuple)
        hash(result)  # Should be hashable without error

    def test_mixed_type_keys_deterministic(self):
        """Same dict with mixed keys produces same result."""
        d1 = {1: "a", "b": 2}
        d2 = {1: "a", "b": 2}
        assert _make_hashable(d1) == _make_hashable(d2)


class TestZipListsNoneHandling:
    """Tests for ZipListsBlock with None values in input."""

    def setup_method(self):
        self.block = ZipListsBlock()

    def test_zip_truncate_with_none(self):
        """_zip_truncate should handle None values in input lists."""
        result = self.block._zip_truncate([[1, 2], None, [3, 4]])  # type: ignore[arg-type]
        assert result == [[1, 3], [2, 4]]

    def test_zip_pad_with_none(self):
        """_zip_pad should handle None values in input lists."""
        result = self.block._zip_pad([[1, 2, 3], None, ["a"]], fill_value="X")  # type: ignore[arg-type]
        assert result == [[1, "a"], [2, "X"], [3, "X"]]

    def test_zip_truncate_all_none(self):
        """All-None inputs should return empty list."""
        result = self.block._zip_truncate([None, None])  # type: ignore[arg-type]
        assert result == []

    def test_zip_pad_all_none(self):
        """All-None inputs should return empty list."""
        result = self.block._zip_pad([None, None], fill_value=0)  # type: ignore[arg-type]
        assert result == []


# =============================================================================
# Block Built-in Tests (using test_input/test_output)
# =============================================================================


class TestConcatenateListsBlockBuiltin:
    """Run the built-in test_input/test_output tests for ConcatenateListsBlock."""

    @pytest.mark.asyncio
    async def test_builtin_tests(self):
        block = ConcatenateListsBlock()
        await execute_block_test(block)


class TestFlattenListBlockBuiltin:
    """Run the built-in test_input/test_output tests for FlattenListBlock."""

    @pytest.mark.asyncio
    async def test_builtin_tests(self):
        block = FlattenListBlock()
        await execute_block_test(block)


class TestInterleaveListsBlockBuiltin:
    """Run the built-in test_input/test_output tests for InterleaveListsBlock."""

    @pytest.mark.asyncio
    async def test_builtin_tests(self):
        block = InterleaveListsBlock()
        await execute_block_test(block)


class TestZipListsBlockBuiltin:
    """Run the built-in test_input/test_output tests for ZipListsBlock."""

    @pytest.mark.asyncio
    async def test_builtin_tests(self):
        block = ZipListsBlock()
        await execute_block_test(block)


class TestListDifferenceBlockBuiltin:
    """Run the built-in test_input/test_output tests for ListDifferenceBlock."""

    @pytest.mark.asyncio
    async def test_builtin_tests(self):
        block = ListDifferenceBlock()
        await execute_block_test(block)


class TestListIntersectionBlockBuiltin:
    """Run the built-in test_input/test_output tests for ListIntersectionBlock."""

    @pytest.mark.asyncio
    async def test_builtin_tests(self):
        block = ListIntersectionBlock()
        await execute_block_test(block)


# =============================================================================
# ConcatenateListsBlock Manual Tests
# =============================================================================


class TestConcatenateListsBlockManual:
    """Manual test cases for ConcatenateListsBlock edge cases."""

    def setup_method(self):
        self.block = ConcatenateListsBlock()

    @pytest.mark.asyncio
    async def test_two_lists(self):
        """Test basic two-list concatenation."""
        results = {}
        async for name, value in self.block.run(
            ConcatenateListsBlock.Input(lists=[[1, 2], [3, 4]])
        ):
            results[name] = value
        assert results["concatenated_list"] == [1, 2, 3, 4]
        assert results["length"] == 4

    @pytest.mark.asyncio
    async def test_three_lists(self):
        """Test three-list concatenation."""
        results = {}
        async for name, value in self.block.run(
            ConcatenateListsBlock.Input(lists=[[1], [2], [3]])
        ):
            results[name] = value
        assert results["concatenated_list"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_five_lists(self):
        """Test concatenation of five lists."""
        results = {}
        async for name, value in self.block.run(
            ConcatenateListsBlock.Input(lists=[[1], [2], [3], [4], [5]])
        ):
            results[name] = value
        assert results["concatenated_list"] == [1, 2, 3, 4, 5]
        assert results["length"] == 5

    @pytest.mark.asyncio
    async def test_empty_lists_only(self):
        """Test concatenation of only empty lists."""
        results = {}
        async for name, value in self.block.run(
            ConcatenateListsBlock.Input(lists=[[], [], []])
        ):
            results[name] = value
        assert results["concatenated_list"] == []
        assert results["length"] == 0

    @pytest.mark.asyncio
    async def test_mixed_types_in_lists(self):
        """Test concatenation with mixed types."""
        results = {}
        async for name, value in self.block.run(
            ConcatenateListsBlock.Input(
                lists=[[1, "a"], [True, 3.14], [None, {"key": "val"}]]
            )
        ):
            results[name] = value
        assert results["concatenated_list"] == [
            1,
            "a",
            True,
            3.14,
            None,
            {"key": "val"},
        ]

    @pytest.mark.asyncio
    async def test_deduplication_enabled(self):
        """Test deduplication removes duplicates."""
        results = {}
        async for name, value in self.block.run(
            ConcatenateListsBlock.Input(
                lists=[[1, 2, 3], [2, 3, 4], [3, 4, 5]],
                deduplicate=True,
            )
        ):
            results[name] = value
        assert results["concatenated_list"] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_deduplication_preserves_order(self):
        """Test that deduplication preserves first-occurrence order."""
        results = {}
        async for name, value in self.block.run(
            ConcatenateListsBlock.Input(
                lists=[[3, 1, 2], [2, 4, 1]],
                deduplicate=True,
            )
        ):
            results[name] = value
        assert results["concatenated_list"] == [3, 1, 2, 4]

    @pytest.mark.asyncio
    async def test_remove_none_enabled(self):
        """Test None removal from concatenated results."""
        results = {}
        async for name, value in self.block.run(
            ConcatenateListsBlock.Input(
                lists=[[1, None], [None, 2], [3, None]],
                remove_none=True,
            )
        ):
            results[name] = value
        assert results["concatenated_list"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_dedup_and_remove_none_combined(self):
        """Test both deduplication and None removal together."""
        results = {}
        async for name, value in self.block.run(
            ConcatenateListsBlock.Input(
                lists=[[1, None, 2], [2, None, 3]],
                deduplicate=True,
                remove_none=True,
            )
        ):
            results[name] = value
        assert results["concatenated_list"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_nested_lists_preserved(self):
        """Test that nested lists are not flattened during concatenation."""
        results = {}
        async for name, value in self.block.run(
            ConcatenateListsBlock.Input(lists=[[[1, 2]], [[3, 4]]])
        ):
            results[name] = value
        assert results["concatenated_list"] == [[1, 2], [3, 4]]

    @pytest.mark.asyncio
    async def test_large_lists(self):
        """Test concatenation of large lists."""
        list_a = list(range(1000))
        list_b = list(range(1000, 2000))
        results = {}
        async for name, value in self.block.run(
            ConcatenateListsBlock.Input(lists=[list_a, list_b])
        ):
            results[name] = value
        assert results["concatenated_list"] == list(range(2000))
        assert results["length"] == 2000

    @pytest.mark.asyncio
    async def test_single_list_input(self):
        """Test concatenation with a single list."""
        results = {}
        async for name, value in self.block.run(
            ConcatenateListsBlock.Input(lists=[[1, 2, 3]])
        ):
            results[name] = value
        assert results["concatenated_list"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_block_id_is_valid_uuid(self):
        """Test that the block has a valid UUID4 ID."""
        import uuid

        parsed = uuid.UUID(self.block.id)
        assert parsed.version == 4

    @pytest.mark.asyncio
    async def test_block_category(self):
        """Test that the block has the correct category."""
        from backend.blocks._base import BlockCategory

        assert BlockCategory.BASIC in self.block.categories


# =============================================================================
# FlattenListBlock Manual Tests
# =============================================================================


class TestFlattenListBlockManual:
    """Manual test cases for FlattenListBlock."""

    def setup_method(self):
        self.block = FlattenListBlock()

    @pytest.mark.asyncio
    async def test_simple_flatten(self):
        """Test flattening a simple nested list."""
        results = {}
        async for name, value in self.block.run(
            FlattenListBlock.Input(nested_list=[[1, 2], [3, 4]])
        ):
            results[name] = value
        assert results["flattened_list"] == [1, 2, 3, 4]
        assert results["length"] == 4

    @pytest.mark.asyncio
    async def test_deeply_nested(self):
        """Test flattening a deeply nested structure."""
        results = {}
        async for name, value in self.block.run(
            FlattenListBlock.Input(nested_list=[1, [2, [3, [4, [5]]]]])
        ):
            results[name] = value
        assert results["flattened_list"] == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_partial_flatten(self):
        """Test flattening with max_depth=1."""
        results = {}
        async for name, value in self.block.run(
            FlattenListBlock.Input(
                nested_list=[[1, [2, 3]], [4, [5]]],
                max_depth=1,
            )
        ):
            results[name] = value
        assert results["flattened_list"] == [1, [2, 3], 4, [5]]

    @pytest.mark.asyncio
    async def test_already_flat_list(self):
        """Test flattening an already flat list."""
        results = {}
        async for name, value in self.block.run(
            FlattenListBlock.Input(nested_list=[1, 2, 3, 4])
        ):
            results[name] = value
        assert results["flattened_list"] == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_empty_nested_lists(self):
        """Test flattening with empty nested lists."""
        results = {}
        async for name, value in self.block.run(
            FlattenListBlock.Input(nested_list=[[], [1], [], [2], []])
        ):
            results[name] = value
        assert results["flattened_list"] == [1, 2]

    @pytest.mark.asyncio
    async def test_mixed_types_preserved(self):
        """Test that non-list types are preserved during flattening."""
        results = {}
        async for name, value in self.block.run(
            FlattenListBlock.Input(nested_list=["hello", [1, {"a": 1}], [True]])
        ):
            results[name] = value
        assert results["flattened_list"] == ["hello", 1, {"a": 1}, True]

    @pytest.mark.asyncio
    async def test_original_depth_reported(self):
        """Test that original nesting depth is correctly reported."""
        results = {}
        async for name, value in self.block.run(
            FlattenListBlock.Input(nested_list=[1, [2, [3]]])
        ):
            results[name] = value
        assert results["original_depth"] == 3

    @pytest.mark.asyncio
    async def test_block_id_is_valid_uuid(self):
        """Test that the block has a valid UUID4 ID."""
        import uuid

        parsed = uuid.UUID(self.block.id)
        assert parsed.version == 4


# =============================================================================
# InterleaveListsBlock Manual Tests
# =============================================================================


class TestInterleaveListsBlockManual:
    """Manual test cases for InterleaveListsBlock."""

    def setup_method(self):
        self.block = InterleaveListsBlock()

    @pytest.mark.asyncio
    async def test_equal_length_interleave(self):
        """Test interleaving two equal-length lists."""
        results = {}
        async for name, value in self.block.run(
            InterleaveListsBlock.Input(lists=[[1, 2, 3], ["a", "b", "c"]])
        ):
            results[name] = value
        assert results["interleaved_list"] == [1, "a", 2, "b", 3, "c"]

    @pytest.mark.asyncio
    async def test_unequal_length_interleave(self):
        """Test interleaving lists of different lengths."""
        results = {}
        async for name, value in self.block.run(
            InterleaveListsBlock.Input(lists=[[1, 2, 3, 4], ["a", "b"]])
        ):
            results[name] = value
        assert results["interleaved_list"] == [1, "a", 2, "b", 3, 4]

    @pytest.mark.asyncio
    async def test_three_lists_interleave(self):
        """Test interleaving three lists."""
        results = {}
        async for name, value in self.block.run(
            InterleaveListsBlock.Input(lists=[[1, 2], ["a", "b"], ["x", "y"]])
        ):
            results[name] = value
        assert results["interleaved_list"] == [1, "a", "x", 2, "b", "y"]

    @pytest.mark.asyncio
    async def test_single_element_lists(self):
        """Test interleaving single-element lists."""
        results = {}
        async for name, value in self.block.run(
            InterleaveListsBlock.Input(lists=[[1], [2], [3], [4]])
        ):
            results[name] = value
        assert results["interleaved_list"] == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_block_id_is_valid_uuid(self):
        """Test that the block has a valid UUID4 ID."""
        import uuid

        parsed = uuid.UUID(self.block.id)
        assert parsed.version == 4


# =============================================================================
# ZipListsBlock Manual Tests
# =============================================================================


class TestZipListsBlockManual:
    """Manual test cases for ZipListsBlock."""

    def setup_method(self):
        self.block = ZipListsBlock()

    @pytest.mark.asyncio
    async def test_basic_zip(self):
        """Test basic zipping of two lists."""
        results = {}
        async for name, value in self.block.run(
            ZipListsBlock.Input(lists=[[1, 2, 3], ["a", "b", "c"]])
        ):
            results[name] = value
        assert results["zipped_list"] == [[1, "a"], [2, "b"], [3, "c"]]

    @pytest.mark.asyncio
    async def test_truncate_to_shortest(self):
        """Test that default behavior truncates to shortest list."""
        results = {}
        async for name, value in self.block.run(
            ZipListsBlock.Input(lists=[[1, 2, 3], ["a", "b"]])
        ):
            results[name] = value
        assert results["zipped_list"] == [[1, "a"], [2, "b"]]
        assert results["length"] == 2

    @pytest.mark.asyncio
    async def test_pad_to_longest(self):
        """Test padding shorter lists with fill value."""
        results = {}
        async for name, value in self.block.run(
            ZipListsBlock.Input(
                lists=[[1, 2, 3], ["a"]],
                pad_to_longest=True,
                fill_value="X",
            )
        ):
            results[name] = value
        assert results["zipped_list"] == [[1, "a"], [2, "X"], [3, "X"]]

    @pytest.mark.asyncio
    async def test_pad_with_none(self):
        """Test padding with None (default fill value)."""
        results = {}
        async for name, value in self.block.run(
            ZipListsBlock.Input(
                lists=[[1, 2], ["a"]],
                pad_to_longest=True,
            )
        ):
            results[name] = value
        assert results["zipped_list"] == [[1, "a"], [2, None]]

    @pytest.mark.asyncio
    async def test_three_lists_zip(self):
        """Test zipping three lists."""
        results = {}
        async for name, value in self.block.run(
            ZipListsBlock.Input(lists=[[1, 2], ["a", "b"], [True, False]])
        ):
            results[name] = value
        assert results["zipped_list"] == [[1, "a", True], [2, "b", False]]

    @pytest.mark.asyncio
    async def test_empty_lists_zip(self):
        """Test zipping empty input."""
        results = {}
        async for name, value in self.block.run(ZipListsBlock.Input(lists=[])):
            results[name] = value
        assert results["zipped_list"] == []
        assert results["length"] == 0

    @pytest.mark.asyncio
    async def test_block_id_is_valid_uuid(self):
        """Test that the block has a valid UUID4 ID."""
        import uuid

        parsed = uuid.UUID(self.block.id)
        assert parsed.version == 4


# =============================================================================
# ListDifferenceBlock Manual Tests
# =============================================================================


class TestListDifferenceBlockManual:
    """Manual test cases for ListDifferenceBlock."""

    def setup_method(self):
        self.block = ListDifferenceBlock()

    @pytest.mark.asyncio
    async def test_basic_difference(self):
        """Test basic set difference."""
        results = {}
        async for name, value in self.block.run(
            ListDifferenceBlock.Input(
                list_a=[1, 2, 3, 4, 5],
                list_b=[3, 4, 5, 6, 7],
            )
        ):
            results[name] = value
        assert results["difference"] == [1, 2]

    @pytest.mark.asyncio
    async def test_symmetric_difference(self):
        """Test symmetric difference."""
        results = {}
        async for name, value in self.block.run(
            ListDifferenceBlock.Input(
                list_a=[1, 2, 3],
                list_b=[2, 3, 4],
                symmetric=True,
            )
        ):
            results[name] = value
        assert results["difference"] == [1, 4]

    @pytest.mark.asyncio
    async def test_no_difference(self):
        """Test when lists are identical."""
        results = {}
        async for name, value in self.block.run(
            ListDifferenceBlock.Input(
                list_a=[1, 2, 3],
                list_b=[1, 2, 3],
            )
        ):
            results[name] = value
        assert results["difference"] == []
        assert results["length"] == 0

    @pytest.mark.asyncio
    async def test_complete_difference(self):
        """Test when lists share no elements."""
        results = {}
        async for name, value in self.block.run(
            ListDifferenceBlock.Input(
                list_a=[1, 2, 3],
                list_b=[4, 5, 6],
            )
        ):
            results[name] = value
        assert results["difference"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_empty_list_a(self):
        """Test with empty list_a."""
        results = {}
        async for name, value in self.block.run(
            ListDifferenceBlock.Input(list_a=[], list_b=[1, 2, 3])
        ):
            results[name] = value
        assert results["difference"] == []

    @pytest.mark.asyncio
    async def test_empty_list_b(self):
        """Test with empty list_b."""
        results = {}
        async for name, value in self.block.run(
            ListDifferenceBlock.Input(list_a=[1, 2, 3], list_b=[])
        ):
            results[name] = value
        assert results["difference"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_string_difference(self):
        """Test difference with string elements."""
        results = {}
        async for name, value in self.block.run(
            ListDifferenceBlock.Input(
                list_a=["apple", "banana", "cherry"],
                list_b=["banana", "date"],
            )
        ):
            results[name] = value
        assert results["difference"] == ["apple", "cherry"]

    @pytest.mark.asyncio
    async def test_dict_difference(self):
        """Test difference with dictionary elements."""
        results = {}
        async for name, value in self.block.run(
            ListDifferenceBlock.Input(
                list_a=[{"a": 1}, {"b": 2}, {"c": 3}],
                list_b=[{"b": 2}],
            )
        ):
            results[name] = value
        assert results["difference"] == [{"a": 1}, {"c": 3}]

    @pytest.mark.asyncio
    async def test_block_id_is_valid_uuid(self):
        """Test that the block has a valid UUID4 ID."""
        import uuid

        parsed = uuid.UUID(self.block.id)
        assert parsed.version == 4


# =============================================================================
# ListIntersectionBlock Manual Tests
# =============================================================================


class TestListIntersectionBlockManual:
    """Manual test cases for ListIntersectionBlock."""

    def setup_method(self):
        self.block = ListIntersectionBlock()

    @pytest.mark.asyncio
    async def test_basic_intersection(self):
        """Test basic intersection."""
        results = {}
        async for name, value in self.block.run(
            ListIntersectionBlock.Input(
                list_a=[1, 2, 3, 4, 5],
                list_b=[3, 4, 5, 6, 7],
            )
        ):
            results[name] = value
        assert results["intersection"] == [3, 4, 5]
        assert results["length"] == 3

    @pytest.mark.asyncio
    async def test_no_intersection(self):
        """Test when lists share no elements."""
        results = {}
        async for name, value in self.block.run(
            ListIntersectionBlock.Input(
                list_a=[1, 2, 3],
                list_b=[4, 5, 6],
            )
        ):
            results[name] = value
        assert results["intersection"] == []
        assert results["length"] == 0

    @pytest.mark.asyncio
    async def test_identical_lists(self):
        """Test intersection of identical lists."""
        results = {}
        async for name, value in self.block.run(
            ListIntersectionBlock.Input(
                list_a=[1, 2, 3],
                list_b=[1, 2, 3],
            )
        ):
            results[name] = value
        assert results["intersection"] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_preserves_order_from_list_a(self):
        """Test that intersection preserves order from list_a."""
        results = {}
        async for name, value in self.block.run(
            ListIntersectionBlock.Input(
                list_a=[5, 3, 1],
                list_b=[1, 3, 5],
            )
        ):
            results[name] = value
        assert results["intersection"] == [5, 3, 1]

    @pytest.mark.asyncio
    async def test_empty_list_a(self):
        """Test with empty list_a."""
        results = {}
        async for name, value in self.block.run(
            ListIntersectionBlock.Input(list_a=[], list_b=[1, 2, 3])
        ):
            results[name] = value
        assert results["intersection"] == []

    @pytest.mark.asyncio
    async def test_empty_list_b(self):
        """Test with empty list_b."""
        results = {}
        async for name, value in self.block.run(
            ListIntersectionBlock.Input(list_a=[1, 2, 3], list_b=[])
        ):
            results[name] = value
        assert results["intersection"] == []

    @pytest.mark.asyncio
    async def test_string_intersection(self):
        """Test intersection with string elements."""
        results = {}
        async for name, value in self.block.run(
            ListIntersectionBlock.Input(
                list_a=["apple", "banana", "cherry"],
                list_b=["banana", "cherry", "date"],
            )
        ):
            results[name] = value
        assert results["intersection"] == ["banana", "cherry"]

    @pytest.mark.asyncio
    async def test_deduplication_in_intersection(self):
        """Test that duplicates in input don't cause duplicate results."""
        results = {}
        async for name, value in self.block.run(
            ListIntersectionBlock.Input(
                list_a=[1, 1, 2, 2, 3],
                list_b=[1, 2],
            )
        ):
            results[name] = value
        assert results["intersection"] == [1, 2]

    @pytest.mark.asyncio
    async def test_block_id_is_valid_uuid(self):
        """Test that the block has a valid UUID4 ID."""
        import uuid

        parsed = uuid.UUID(self.block.id)
        assert parsed.version == 4


# =============================================================================
# Block Method Tests
# =============================================================================


class TestConcatenateListsBlockMethods:
    """Tests for internal methods of ConcatenateListsBlock."""

    def setup_method(self):
        self.block = ConcatenateListsBlock()

    def test_validate_inputs_valid(self):
        assert self.block._validate_inputs([[1], [2]]) is None

    def test_validate_inputs_invalid(self):
        result = self.block._validate_inputs([[1], "bad"])
        assert result is not None

    def test_perform_concatenation(self):
        result = self.block._perform_concatenation([[1, 2], [3, 4]])
        assert result == [1, 2, 3, 4]

    def test_apply_deduplication(self):
        result = self.block._apply_deduplication([1, 2, 2, 3])
        assert result == [1, 2, 3]

    def test_apply_none_removal(self):
        result = self.block._apply_none_removal([1, None, 2])
        assert result == [1, 2]

    def test_post_process_all_options(self):
        result = self.block._post_process(
            [1, None, 2, None, 2], deduplicate=True, remove_none=True
        )
        assert result == [1, 2]

    def test_post_process_no_options(self):
        result = self.block._post_process(
            [1, None, 2, None, 2], deduplicate=False, remove_none=False
        )
        assert result == [1, None, 2, None, 2]


class TestFlattenListBlockMethods:
    """Tests for internal methods of FlattenListBlock."""

    def setup_method(self):
        self.block = FlattenListBlock()

    def test_compute_depth_flat(self):
        assert self.block._compute_depth([1, 2, 3]) == 1

    def test_compute_depth_nested(self):
        assert self.block._compute_depth([[1, [2]]]) == 3

    def test_flatten_unlimited(self):
        result = self.block._flatten([1, [2, [3]]], max_depth=-1)
        assert result == [1, 2, 3]

    def test_flatten_limited(self):
        result = self.block._flatten([1, [2, [3]]], max_depth=1)
        assert result == [1, 2, [3]]

    def test_validate_max_depth_valid(self):
        assert self.block._validate_max_depth(-1) is None
        assert self.block._validate_max_depth(0) is None
        assert self.block._validate_max_depth(5) is None

    def test_validate_max_depth_invalid(self):
        result = self.block._validate_max_depth(-2)
        assert result is not None


class TestZipListsBlockMethods:
    """Tests for internal methods of ZipListsBlock."""

    def setup_method(self):
        self.block = ZipListsBlock()

    def test_zip_truncate(self):
        result = self.block._zip_truncate([[1, 2, 3], ["a", "b"]])
        assert result == [[1, "a"], [2, "b"]]

    def test_zip_pad(self):
        result = self.block._zip_pad([[1, 2, 3], ["a"]], fill_value="X")
        assert result == [[1, "a"], [2, "X"], [3, "X"]]

    def test_zip_pad_empty(self):
        result = self.block._zip_pad([], fill_value=None)
        assert result == []

    def test_validate_inputs(self):
        assert self.block._validate_inputs([[1], [2]]) is None
        result = self.block._validate_inputs([[1], "bad"])
        assert result is not None


class TestListDifferenceBlockMethods:
    """Tests for internal methods of ListDifferenceBlock."""

    def setup_method(self):
        self.block = ListDifferenceBlock()

    def test_compute_difference(self):
        result = self.block._compute_difference([1, 2, 3], [2, 3, 4])
        assert result == [1]

    def test_compute_symmetric_difference(self):
        result = self.block._compute_symmetric_difference([1, 2, 3], [2, 3, 4])
        assert result == [1, 4]

    def test_compute_difference_empty(self):
        result = self.block._compute_difference([], [1, 2])
        assert result == []

    def test_compute_symmetric_difference_identical(self):
        result = self.block._compute_symmetric_difference([1, 2], [1, 2])
        assert result == []


class TestListIntersectionBlockMethods:
    """Tests for internal methods of ListIntersectionBlock."""

    def setup_method(self):
        self.block = ListIntersectionBlock()

    def test_compute_intersection(self):
        result = self.block._compute_intersection([1, 2, 3], [2, 3, 4])
        assert result == [2, 3]

    def test_compute_intersection_empty(self):
        result = self.block._compute_intersection([], [1, 2])
        assert result == []

    def test_compute_intersection_no_overlap(self):
        result = self.block._compute_intersection([1, 2], [3, 4])
        assert result == []
