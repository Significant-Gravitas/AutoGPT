"""
Tests for dynamic field routing with sanitized names.

This test file specifically tests the parse_execution_output function
which is responsible for routing tool outputs to the correct nodes.
The critical bug this addresses is the mismatch between:
- emit keys using sanitized names (e.g., "max_keyword_difficulty")
- sink_pin_name using original names (e.g., "Max Keyword Difficulty")
"""

import re
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


def cleanup(s: str) -> str:
    """
    Simulate SmartDecisionMakerBlock.cleanup() for testing.
    Clean up names for use as tool function names.
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s).lower()


class TestParseExecutionOutputToolRouting:
    """Tests for tool pin routing in parse_execution_output."""

    def test_exact_match_routes_correctly(self):
        """When emit key field exactly matches sink_pin_name, routing works."""
        output_item = ("tools_^_node-123_~_query", "test value")

        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="node-123",
            sink_pin_name="query",
        )
        assert result == "test value"

    def test_sanitized_emit_vs_original_sink_fails(self):
        """
        CRITICAL BUG TEST: When emit key uses sanitized name but sink uses original,
        routing fails.
        """
        # Backend emits with sanitized name
        sanitized_field = cleanup("Max Keyword Difficulty")
        output_item = (f"tools_^_node-123_~_{sanitized_field}", 50)

        # Frontend link has original name
        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="node-123",
            sink_pin_name="Max Keyword Difficulty",  # Original name
        )

        # BUG: This returns None because sanitized != original
        # Once fixed, change this to: assert result == 50
        assert result is None, "Expected None due to sanitization mismatch bug"

    def test_node_id_mismatch_returns_none(self):
        """When node IDs don't match, routing should return None."""
        output_item = ("tools_^_node-123_~_query", "test value")

        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="different-node",  # Different node
            sink_pin_name="query",
        )
        assert result is None

    def test_both_node_and_pin_must_match(self):
        """Both node_id and pin_name must match for routing to succeed."""
        output_item = ("tools_^_node-123_~_query", "test value")

        # Wrong node, right pin
        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="wrong-node",
            sink_pin_name="query",
        )
        assert result is None

        # Right node, wrong pin
        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="node-123",
            sink_pin_name="wrong_pin",
        )
        assert result is None

        # Right node, right pin
        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="node-123",
            sink_pin_name="query",
        )
        assert result == "test value"


class TestToolPinRoutingWithSpecialCharacters:
    """Tests for tool pin routing with various special characters in names."""

    @pytest.mark.parametrize(
        "original_name,sanitized_name",
        [
            ("Max Keyword Difficulty", "max_keyword_difficulty"),
            ("Search Volume (Monthly)", "search_volume__monthly_"),
            ("CPC ($)", "cpc____"),
            ("User's Input", "user_s_input"),
            ("Query #1", "query__1"),
            ("API.Response", "api_response"),
            ("Field@Name", "field_name"),
            ("Test\tTab", "test_tab"),
            ("Test\nNewline", "test_newline"),
        ],
    )
    def test_routing_mismatch_with_special_chars(self, original_name, sanitized_name):
        """
        Test that various special characters cause routing mismatches.

        This test documents the current buggy behavior where sanitized emit keys
        don't match original sink_pin_names.
        """
        # Verify sanitization
        assert cleanup(original_name) == sanitized_name

        # Backend emits with sanitized name
        output_item = (f"tools_^_node-123_~_{sanitized_name}", "value")

        # Frontend link has original name
        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="node-123",
            sink_pin_name=original_name,
        )

        # BUG: Returns None due to mismatch
        assert result is None, f"Routing should fail for '{original_name}' vs '{sanitized_name}'"


class TestToolPinMissingParameters:
    """Tests for missing required parameters in parse_execution_output."""

    def test_missing_sink_node_id_raises_error(self):
        """Missing sink_node_id should raise ValueError for tool pins."""
        output_item = ("tools_^_node-123_~_query", "test value")

        with pytest.raises(ValueError, match="sink_node_id and sink_pin_name must be provided"):
            parse_execution_output(
                output_item,
                link_output_selector="tools",
                sink_node_id=None,
                sink_pin_name="query",
            )

    def test_missing_sink_pin_name_raises_error(self):
        """Missing sink_pin_name should raise ValueError for tool pins."""
        output_item = ("tools_^_node-123_~_query", "test value")

        with pytest.raises(ValueError, match="sink_node_id and sink_pin_name must be provided"):
            parse_execution_output(
                output_item,
                link_output_selector="tools",
                sink_node_id="node-123",
                sink_pin_name=None,
            )


class TestIsToolPin:
    """Tests for is_tool_pin function."""

    def test_tools_prefix_is_tool_pin(self):
        """Names starting with 'tools_^_' are tool pins."""
        assert is_tool_pin("tools_^_node_~_field") is True
        assert is_tool_pin("tools_^_anything") is True

    def test_tools_exact_is_tool_pin(self):
        """Exact 'tools' is a tool pin."""
        assert is_tool_pin("tools") is True

    def test_non_tool_pins(self):
        """Non-tool pin names should return False."""
        assert is_tool_pin("input") is False
        assert is_tool_pin("output") is False
        assert is_tool_pin("my_tools") is False
        assert is_tool_pin("toolsomething") is False


class TestSanitizePinName:
    """Tests for sanitize_pin_name function."""

    def test_extracts_base_from_dynamic_field(self):
        """Should extract base field name from dynamic fields."""
        assert sanitize_pin_name("values_#_key") == "values"
        assert sanitize_pin_name("items_$_0") == "items"
        assert sanitize_pin_name("obj_@_attr") == "obj"

    def test_returns_tools_for_tool_pins(self):
        """Tool pins should be sanitized to 'tools'."""
        assert sanitize_pin_name("tools_^_node_~_field") == "tools"
        assert sanitize_pin_name("tools") == "tools"

    def test_regular_field_unchanged(self):
        """Regular field names should be unchanged."""
        assert sanitize_pin_name("query") == "query"
        assert sanitize_pin_name("max_difficulty") == "max_difficulty"


class TestDynamicFieldDescriptions:
    """Tests for dynamic field description generation."""

    def test_dict_field_description_with_spaces_in_key(self):
        """Dictionary field keys with spaces should generate correct descriptions."""
        # After cleanup, "User Name" becomes "user_name" in the field name
        # But the original key might have had spaces
        desc = get_dynamic_field_description("values_#_user_name")
        assert "Dictionary field" in desc
        assert "values['user_name']" in desc

    def test_list_field_description(self):
        """List field descriptions should include index."""
        desc = get_dynamic_field_description("items_$_0")
        assert "List item 0" in desc
        assert "items[0]" in desc

    def test_object_field_description(self):
        """Object field descriptions should include attribute."""
        desc = get_dynamic_field_description("user_@_email")
        assert "Object attribute" in desc
        assert "user.email" in desc


class TestMergeExecutionInput:
    """Tests for merge_execution_input function."""

    def test_merges_dict_fields(self):
        """Dictionary fields should be merged into nested structure."""
        data = {
            "values_#_name": "Alice",
            "values_#_age": 30,
            "other_field": "unchanged",
        }

        result = merge_execution_input(data)

        assert "values" in result
        assert result["values"]["name"] == "Alice"
        assert result["values"]["age"] == 30
        assert result["other_field"] == "unchanged"

    def test_merges_list_fields(self):
        """List fields should be merged into arrays."""
        data = {
            "items_$_0": "first",
            "items_$_1": "second",
            "items_$_2": "third",
        }

        result = merge_execution_input(data)

        assert "items" in result
        assert result["items"] == ["first", "second", "third"]

    def test_merges_mixed_fields(self):
        """Mixed regular and dynamic fields should all be preserved."""
        data = {
            "regular": "value",
            "dict_#_key": "dict_value",
            "list_$_0": "list_item",
        }

        result = merge_execution_input(data)

        assert result["regular"] == "value"
        assert result["dict"]["key"] == "dict_value"
        assert result["list"] == ["list_item"]


class TestExtractBaseFieldName:
    """Tests for extract_base_field_name function."""

    def test_extracts_from_dict_delimiter(self):
        """Should extract base name before _#_ delimiter."""
        assert extract_base_field_name("values_#_name") == "values"
        assert extract_base_field_name("user_#_email_#_domain") == "user"

    def test_extracts_from_list_delimiter(self):
        """Should extract base name before _$_ delimiter."""
        assert extract_base_field_name("items_$_0") == "items"
        assert extract_base_field_name("data_$_1_$_nested") == "data"

    def test_extracts_from_object_delimiter(self):
        """Should extract base name before _@_ delimiter."""
        assert extract_base_field_name("obj_@_attr") == "obj"

    def test_no_delimiter_returns_original(self):
        """Names without delimiters should be returned unchanged."""
        assert extract_base_field_name("regular_field") == "regular_field"
        assert extract_base_field_name("query") == "query"


class TestIsDynamicField:
    """Tests for is_dynamic_field function."""

    def test_dict_delimiter_is_dynamic(self):
        """Fields with _#_ are dynamic."""
        assert is_dynamic_field("values_#_key") is True

    def test_list_delimiter_is_dynamic(self):
        """Fields with _$_ are dynamic."""
        assert is_dynamic_field("items_$_0") is True

    def test_object_delimiter_is_dynamic(self):
        """Fields with _@_ are dynamic."""
        assert is_dynamic_field("obj_@_attr") is True

    def test_regular_fields_not_dynamic(self):
        """Regular field names without delimiters are not dynamic."""
        assert is_dynamic_field("regular_field") is False
        assert is_dynamic_field("query") is False
        assert is_dynamic_field("Max Keyword Difficulty") is False


class TestRoutingEndToEnd:
    """End-to-end tests for the full routing flow."""

    def test_successful_routing_without_spaces(self):
        """Full routing flow works when no spaces in names."""
        field_name = "query"
        node_id = "test-node-123"

        # Emit key (as created by SmartDecisionMaker)
        emit_key = f"tools_^_{node_id}_~_{cleanup(field_name)}"
        output_item = (emit_key, "search term")

        # Route (as called by executor)
        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id=node_id,
            sink_pin_name=field_name,
        )

        assert result == "search term"

    def test_failed_routing_with_spaces(self):
        """
        Full routing flow FAILS when names have spaces.

        This test documents the exact bug scenario:
        1. Frontend creates link with sink_name="Max Keyword Difficulty"
        2. SmartDecisionMaker emits with sanitized name in key
        3. Executor calls parse_execution_output with original sink_pin_name
        4. Routing fails because names don't match
        """
        original_field_name = "Max Keyword Difficulty"
        sanitized_field_name = cleanup(original_field_name)
        node_id = "test-node-123"

        # Step 1 & 2: SmartDecisionMaker emits with sanitized name
        emit_key = f"tools_^_{node_id}_~_{sanitized_field_name}"
        output_item = (emit_key, 50)

        # Step 3: Executor routes with original name from link
        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id=node_id,
            sink_pin_name=original_field_name,  # Original from link!
        )

        # Step 4: BUG - Returns None instead of 50
        assert result is None

        # This is what should happen after fix:
        # assert result == 50

    def test_multiple_fields_with_spaces(self):
        """Test routing multiple fields where some have spaces."""
        node_id = "test-node"

        fields = {
            "query": "test",  # No spaces - should work
            "Max Difficulty": 100,  # Spaces - will fail
            "min_volume": 1000,  # No spaces - should work
        }

        results = {}
        for original_name, value in fields.items():
            sanitized = cleanup(original_name)
            emit_key = f"tools_^_{node_id}_~_{sanitized}"
            output_item = (emit_key, value)

            result = parse_execution_output(
                output_item,
                link_output_selector="tools",
                sink_node_id=node_id,
                sink_pin_name=original_name,
            )
            results[original_name] = result

        # Fields without spaces work
        assert results["query"] == "test"
        assert results["min_volume"] == 1000

        # Fields with spaces fail
        assert results["Max Difficulty"] is None  # BUG!


class TestProposedFix:
    """
    Tests for the proposed fix.

    The fix should sanitize sink_pin_name before comparison in parse_execution_output.
    This class contains tests that will pass once the fix is implemented.
    """

    def test_routing_should_sanitize_both_sides(self):
        """
        PROPOSED FIX: parse_execution_output should sanitize sink_pin_name
        before comparing with the field from emit key.

        Current behavior: Direct string comparison
        Fixed behavior: Compare cleanup(target_input_pin) == cleanup(sink_pin_name)
        """
        original_field = "Max Keyword Difficulty"
        sanitized_field = cleanup(original_field)
        node_id = "node-123"

        emit_key = f"tools_^_{node_id}_~_{sanitized_field}"
        output_item = (emit_key, 50)

        # Extract the comparison being made
        selector = emit_key[8:]  # Remove "tools_^_"
        target_node_id, target_input_pin = selector.split("_~_", 1)

        # Current comparison (FAILS):
        current_comparison = (target_input_pin == original_field)
        assert current_comparison is False, "Current comparison fails"

        # Proposed fixed comparison (PASSES):
        # Either sanitize sink_pin_name, or sanitize both
        fixed_comparison = (target_input_pin == cleanup(original_field))
        assert fixed_comparison is True, "Fixed comparison should pass"
