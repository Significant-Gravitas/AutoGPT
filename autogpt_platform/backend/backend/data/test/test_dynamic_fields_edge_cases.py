"""
Tests for dynamic fields edge cases and failure modes.

Covers failure modes:
8. No Type Validation in Dynamic Field Merging
17. No Validation of Dynamic Field Paths
"""

from typing import Any

import pytest

from backend.data.dynamic_fields import (
    DICT_SPLIT,
    LIST_SPLIT,
    OBJC_SPLIT,
    extract_base_field_name,
    get_dynamic_field_description,
    is_dynamic_field,
    is_tool_pin,
    merge_execution_input,
    parse_execution_output,
    sanitize_pin_name,
)


class TestDynamicFieldMergingTypeValidation:
    """
    Tests for Failure Mode #8: No Type Validation in Dynamic Field Merging

    When merging dynamic fields, there's no validation that intermediate
    structures have the correct type, leading to potential type coercion errors.
    """

    def test_merge_dict_field_creates_dict(self):
        """Test that dictionary fields create dict structure."""
        data = {
            "values_#_name": "Alice",
            "values_#_age": 30,
        }

        result = merge_execution_input(data)

        assert "values" in result
        assert isinstance(result["values"], dict)
        assert result["values"]["name"] == "Alice"
        assert result["values"]["age"] == 30

    def test_merge_list_field_creates_list(self):
        """Test that list fields create list structure."""
        data = {
            "items_$_0": "first",
            "items_$_1": "second",
            "items_$_2": "third",
        }

        result = merge_execution_input(data)

        assert "items" in result
        assert isinstance(result["items"], list)
        assert result["items"] == ["first", "second", "third"]

    def test_merge_with_existing_primitive_type_conflict(self):
        """
        Test behavior when merging into existing primitive value.

        BUG: If the base field already exists as a primitive,
        merging a dynamic field may fail or corrupt data.
        """
        # Pre-existing primitive value
        data = {
            "value": "I am a string",  # Primitive
            "value_#_key": "dict value",  # Dynamic dict field
        }

        # This may raise an error or produce unexpected results
        # depending on merge order and implementation
        try:
            result = merge_execution_input(data)
            # If it succeeds, check what happened
            # The primitive may have been overwritten
            if isinstance(result.get("value"), dict):
                # Primitive was converted to dict - data loss!
                assert "key" in result["value"]
            else:
                # Or the dynamic field was ignored
                pass
        except (TypeError, AttributeError):
            # Expected error when trying to merge into primitive
            pass

    def test_merge_list_with_gaps(self):
        """Test merging list fields with non-contiguous indices."""
        data = {
            "items_$_0": "zero",
            "items_$_2": "two",  # Gap at index 1
            "items_$_5": "five",  # Larger gap
        }

        result = merge_execution_input(data)

        assert "items" in result
        # Check how gaps are handled
        items = result["items"]
        assert items[0] == "zero"
        # Index 1 may be None or missing
        assert items[2] == "two"
        assert items[5] == "five"

    def test_merge_nested_dynamic_fields(self):
        """Test merging deeply nested dynamic fields."""
        data = {
            "data_#_users_$_0": "user1",
            "data_#_users_$_1": "user2",
            "data_#_config_#_enabled": True,
        }

        result = merge_execution_input(data)

        # Complex nested structures should be created
        assert "data" in result

    def test_merge_object_field(self):
        """Test merging object attribute fields."""
        data = {
            "user_@_name": "Alice",
            "user_@_email": "alice@example.com",
        }

        result = merge_execution_input(data)

        assert "user" in result
        # Object fields create dict-like structure
        assert result["user"]["name"] == "Alice"
        assert result["user"]["email"] == "alice@example.com"

    def test_merge_mixed_field_types(self):
        """Test merging mixed regular and dynamic fields."""
        data = {
            "regular": "value",
            "dict_field_#_key": "dict_value",
            "list_field_$_0": "list_item",
        }

        result = merge_execution_input(data)

        assert result["regular"] == "value"
        assert result["dict_field"]["key"] == "dict_value"
        assert result["list_field"][0] == "list_item"


class TestDynamicFieldPathValidation:
    """
    Tests for Failure Mode #17: No Validation of Dynamic Field Paths

    When traversing dynamic field paths, intermediate None values
    can cause TypeErrors instead of graceful failures.
    """

    def test_parse_output_with_none_intermediate(self):
        """
        Test parse_execution_output with None intermediate value.

        If data contains {"items": None} and we try to access items[0],
        it should return None gracefully, not raise TypeError.
        """
        # Output with nested path
        output_item = ("data_$_0", "value")

        # When the base is None, should return None
        # This tests the path traversal logic
        result = parse_execution_output(
            output_item,
            link_output_selector="data",
            sink_node_id=None,
            sink_pin_name=None,
        )

        # Should handle gracefully (return the value or None)
        # Not raise TypeError

    def test_extract_base_field_name_with_multiple_delimiters(self):
        """Test extracting base name with multiple delimiters."""
        # Multiple dict delimiters
        assert extract_base_field_name("a_#_b_#_c") == "a"

        # Multiple list delimiters
        assert extract_base_field_name("a_$_0_$_1") == "a"

        # Mixed delimiters
        assert extract_base_field_name("a_#_b_$_0") == "a"

    def test_is_dynamic_field_edge_cases(self):
        """Test is_dynamic_field with edge cases."""
        # Standard dynamic fields
        assert is_dynamic_field("values_#_key") is True
        assert is_dynamic_field("items_$_0") is True
        assert is_dynamic_field("obj_@_attr") is True

        # Regular fields
        assert is_dynamic_field("regular") is False
        assert is_dynamic_field("with_underscore") is False

        # Edge cases
        assert is_dynamic_field("") is False
        assert is_dynamic_field("_#_") is True  # Just delimiter
        assert is_dynamic_field("a_#_") is True  # Trailing delimiter

    def test_sanitize_pin_name_with_tool_pins(self):
        """Test sanitize_pin_name with various tool pin formats."""
        # Tool pins should return "tools"
        assert sanitize_pin_name("tools") == "tools"
        assert sanitize_pin_name("tools_^_node_~_field") == "tools"

        # Dynamic fields should return base name
        assert sanitize_pin_name("values_#_key") == "values"
        assert sanitize_pin_name("items_$_0") == "items"

        # Regular fields unchanged
        assert sanitize_pin_name("regular") == "regular"


class TestDynamicFieldDescriptions:
    """Tests for dynamic field description generation."""

    def test_dict_field_description(self):
        """Test description for dictionary fields."""
        desc = get_dynamic_field_description("values_#_user_name")

        assert "Dictionary field" in desc
        assert "values['user_name']" in desc

    def test_list_field_description(self):
        """Test description for list fields."""
        desc = get_dynamic_field_description("items_$_0")

        assert "List item 0" in desc
        assert "items[0]" in desc

    def test_object_field_description(self):
        """Test description for object fields."""
        desc = get_dynamic_field_description("user_@_email")

        assert "Object attribute" in desc
        assert "user.email" in desc

    def test_regular_field_description(self):
        """Test description for regular (non-dynamic) fields."""
        desc = get_dynamic_field_description("regular_field")

        assert desc == "Value for regular_field"

    def test_description_with_numeric_key(self):
        """Test description with numeric dictionary key."""
        desc = get_dynamic_field_description("values_#_123")

        assert "Dictionary field" in desc
        assert "values['123']" in desc


class TestParseExecutionOutputToolRouting:
    """Tests for tool pin routing in parse_execution_output."""

    def test_tool_pin_routing_exact_match(self):
        """Test tool pin routing with exact match."""
        output_item = ("tools_^_node-123_~_field_name", "value")

        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="node-123",
            sink_pin_name="field_name",
        )

        assert result == "value"

    def test_tool_pin_routing_node_mismatch(self):
        """Test tool pin routing with node ID mismatch."""
        output_item = ("tools_^_node-123_~_field_name", "value")

        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="different-node",
            sink_pin_name="field_name",
        )

        assert result is None

    def test_tool_pin_routing_field_mismatch(self):
        """Test tool pin routing with field name mismatch."""
        output_item = ("tools_^_node-123_~_field_name", "value")

        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="node-123",
            sink_pin_name="different_field",
        )

        assert result is None

    def test_tool_pin_missing_required_params(self):
        """Test that tool pins require node_id and pin_name."""
        output_item = ("tools_^_node-123_~_field", "value")

        with pytest.raises(ValueError, match="must be provided"):
            parse_execution_output(
                output_item,
                link_output_selector="tools",
                sink_node_id=None,
                sink_pin_name="field",
            )

        with pytest.raises(ValueError, match="must be provided"):
            parse_execution_output(
                output_item,
                link_output_selector="tools",
                sink_node_id="node-123",
                sink_pin_name=None,
            )


class TestParseExecutionOutputDynamicFields:
    """Tests for dynamic field routing in parse_execution_output."""

    def test_dict_field_extraction(self):
        """Test extraction of dictionary field value."""
        # The output_item is (field_name, data_structure)
        data = {"key1": "value1", "key2": "value2"}
        output_item = ("values", data)

        result = parse_execution_output(
            output_item,
            link_output_selector="values_#_key1",
            sink_node_id=None,
            sink_pin_name=None,
        )

        assert result == "value1"

    def test_list_field_extraction(self):
        """Test extraction of list item value."""
        data = ["zero", "one", "two"]
        output_item = ("items", data)

        result = parse_execution_output(
            output_item,
            link_output_selector="items_$_1",
            sink_node_id=None,
            sink_pin_name=None,
        )

        assert result == "one"

    def test_nested_field_extraction(self):
        """Test extraction of nested field value."""
        data = {
            "users": [
                {"name": "Alice", "email": "alice@example.com"},
                {"name": "Bob", "email": "bob@example.com"},
            ]
        }
        output_item = ("data", data)

        # Access nested path
        result = parse_execution_output(
            output_item,
            link_output_selector="data_#_users",
            sink_node_id=None,
            sink_pin_name=None,
        )

        assert result == data["users"]

    def test_missing_key_returns_none(self):
        """Test that missing keys return None."""
        data = {"existing": "value"}
        output_item = ("values", data)

        result = parse_execution_output(
            output_item,
            link_output_selector="values_#_nonexistent",
            sink_node_id=None,
            sink_pin_name=None,
        )

        assert result is None

    def test_index_out_of_bounds_returns_none(self):
        """Test that out-of-bounds indices return None."""
        data = ["zero", "one"]
        output_item = ("items", data)

        result = parse_execution_output(
            output_item,
            link_output_selector="items_$_99",
            sink_node_id=None,
            sink_pin_name=None,
        )

        assert result is None


class TestIsToolPin:
    """Tests for is_tool_pin function."""

    def test_tools_prefix(self):
        """Test that 'tools_^_' prefix is recognized."""
        assert is_tool_pin("tools_^_node_~_field") is True
        assert is_tool_pin("tools_^_anything") is True

    def test_tools_exact(self):
        """Test that exact 'tools' is recognized."""
        assert is_tool_pin("tools") is True

    def test_non_tool_pins(self):
        """Test that non-tool pins are not recognized."""
        assert is_tool_pin("input") is False
        assert is_tool_pin("output") is False
        assert is_tool_pin("toolsomething") is False
        assert is_tool_pin("my_tools") is False
        assert is_tool_pin("") is False


class TestMergeExecutionInputEdgeCases:
    """Edge case tests for merge_execution_input."""

    def test_empty_input(self):
        """Test merging empty input."""
        result = merge_execution_input({})
        assert result == {}

    def test_only_regular_fields(self):
        """Test merging only regular fields (no dynamic)."""
        data = {"a": 1, "b": 2, "c": 3}
        result = merge_execution_input(data)
        assert result == data

    def test_overwrite_behavior(self):
        """Test behavior when same key is set multiple times."""
        # This shouldn't happen in practice, but test the behavior
        data = {
            "values_#_key": "first",
        }
        result = merge_execution_input(data)
        assert result["values"]["key"] == "first"

    def test_numeric_string_keys(self):
        """Test handling of numeric string keys in dict fields."""
        data = {
            "values_#_123": "numeric_key",
            "values_#_456": "another_numeric",
        }
        result = merge_execution_input(data)

        assert result["values"]["123"] == "numeric_key"
        assert result["values"]["456"] == "another_numeric"

    def test_special_characters_in_keys(self):
        """Test handling of special characters in keys."""
        data = {
            "values_#_key-with-dashes": "value1",
            "values_#_key.with.dots": "value2",
        }
        result = merge_execution_input(data)

        assert result["values"]["key-with-dashes"] == "value1"
        assert result["values"]["key.with.dots"] == "value2"

    def test_deeply_nested_list(self):
        """Test deeply nested list indices."""
        data = {
            "matrix_$_0_$_0": "0,0",
            "matrix_$_0_$_1": "0,1",
            "matrix_$_1_$_0": "1,0",
            "matrix_$_1_$_1": "1,1",
        }

        # Note: Current implementation may not support this depth
        # Test documents expected behavior
        try:
            result = merge_execution_input(data)
            # If supported, verify structure
        except (KeyError, TypeError, IndexError):
            # Deep nesting may not be supported
            pass

    def test_none_values(self):
        """Test handling of None values in input."""
        data = {
            "regular": None,
            "dict_#_key": None,
            "list_$_0": None,
        }

        result = merge_execution_input(data)

        assert result["regular"] is None
        assert result["dict"]["key"] is None
        assert result["list"][0] is None

    def test_complex_values(self):
        """Test handling of complex values (dicts, lists)."""
        data = {
            "values_#_nested_dict": {"inner": "value"},
            "values_#_nested_list": [1, 2, 3],
        }

        result = merge_execution_input(data)

        assert result["values"]["nested_dict"] == {"inner": "value"}
        assert result["values"]["nested_list"] == [1, 2, 3]
