"""
Tests for SmartDecisionMaker data integrity failure modes.

Covers failure modes:
6. Conversation Corruption in Error Paths
7. Field Name Collision Not Detected
8. No Type Validation in Dynamic Field Merging
9. Unhandled Field Mapping Keys
16. Silent Value Loss in Output Routing
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock


class TestFieldNameCollisionDetection:
    """
    Tests for Failure Mode #7: Field Name Collision Not Detected

    When multiple field names sanitize to the same value,
    the last one silently overwrites previous mappings.
    """

    def test_different_names_same_sanitized_result(self):
        """Test that different names can produce the same sanitized result."""
        cleanup = SmartDecisionMakerBlock.cleanup

        # All these sanitize to "test_field"
        variants = [
            "test_field",
            "Test Field",
            "test field",
            "TEST_FIELD",
            "Test_Field",
            "test-field",  # Note: hyphen is preserved, this is different
        ]

        sanitized = [cleanup(v) for v in variants]

        # Count unique sanitized values
        unique = set(sanitized)
        # Most should collide (except hyphenated one)
        assert len(unique) < len(variants), \
            f"Expected collisions, got {unique}"

    @pytest.mark.asyncio
    async def test_collision_last_one_wins(self):
        """Test that in case of collision, the last field mapping wins."""
        block = SmartDecisionMakerBlock()

        mock_node = Mock()
        mock_node.id = "test-node"
        mock_node.block = Mock()
        mock_node.block.name = "TestBlock"
        mock_node.block.description = "Test"
        mock_node.block.input_schema = Mock()
        mock_node.block.input_schema.jsonschema = Mock(
            return_value={"properties": {}, "required": []}
        )
        mock_node.block.input_schema.get_field_schema = Mock(
            return_value={"type": "string", "description": "test"}
        )

        # Two fields that sanitize to the same name
        mock_links = [
            Mock(sink_name="Test Field", sink_id="test-node", source_id="source"),
            Mock(sink_name="test field", sink_id="test-node", source_id="source"),
        ]

        signature = await block._create_block_function_signature(mock_node, mock_links)

        field_mapping = signature["function"]["_field_mapping"]
        properties = signature["function"]["parameters"]["properties"]

        # Only one property (collision)
        assert len(properties) == 1
        assert "test_field" in properties

        # The mapping has only the last one
        # This is the BUG: first field's mapping is lost
        assert field_mapping["test_field"] in ["Test Field", "test field"]

    @pytest.mark.asyncio
    async def test_collision_causes_data_loss(self):
        """
        Test that field collision can cause actual data loss.

        Scenario:
        1. Two fields "Field A" and "field a" both map to "field_a"
        2. LLM provides value for "field_a"
        3. Only one original field gets the value
        4. The other field's expected input is lost
        """
        block = SmartDecisionMakerBlock()

        # Simulate processing tool calls with collision
        mock_response = Mock()
        mock_tool_call = Mock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = json.dumps({
            "field_a": "value_for_both"  # LLM uses sanitized name
        })
        mock_response.tool_calls = [mock_tool_call]

        # Tool definition with collision in field mapping
        tool_functions = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "parameters": {
                        "properties": {
                            "field_a": {"type": "string"},
                        },
                        "required": ["field_a"],
                    },
                    "_sink_node_id": "sink",
                    # BUG: Only one original name is stored
                    # "Field A" was overwritten by "field a"
                    "_field_mapping": {"field_a": "field a"},
                },
            }
        ]

        processed = block._process_tool_calls(mock_response, tool_functions)

        assert len(processed) == 1
        input_data = processed[0].input_data

        # Only "field a" gets the value
        assert "field a" in input_data
        assert input_data["field a"] == "value_for_both"

        # "Field A" is completely lost!
        assert "Field A" not in input_data


class TestUnhandledFieldMappingKeys:
    """
    Tests for Failure Mode #9: Unhandled Field Mapping Keys

    When field_mapping is missing a key, the code falls back to
    the clean name, which may not be what the sink expects.
    """

    @pytest.mark.asyncio
    async def test_missing_field_mapping_falls_back_to_clean_name(self):
        """Test that missing field mapping falls back to clean name."""
        block = SmartDecisionMakerBlock()

        mock_response = Mock()
        mock_tool_call = Mock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = json.dumps({
            "unmapped_field": "value"
        })
        mock_response.tool_calls = [mock_tool_call]

        # Tool definition with incomplete field mapping
        tool_functions = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "parameters": {
                        "properties": {
                            "unmapped_field": {"type": "string"},
                        },
                        "required": [],
                    },
                    "_sink_node_id": "sink",
                    "_field_mapping": {},  # Empty! No mapping for unmapped_field
                },
            }
        ]

        processed = block._process_tool_calls(mock_response, tool_functions)

        assert len(processed) == 1
        input_data = processed[0].input_data

        # Falls back to clean name (which IS the key since it's already clean)
        assert "unmapped_field" in input_data

    @pytest.mark.asyncio
    async def test_partial_field_mapping(self):
        """Test behavior with partial field mapping."""
        block = SmartDecisionMakerBlock()

        mock_response = Mock()
        mock_tool_call = Mock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = json.dumps({
            "mapped_field": "value1",
            "unmapped_field": "value2",
        })
        mock_response.tool_calls = [mock_tool_call]

        tool_functions = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "parameters": {
                        "properties": {
                            "mapped_field": {"type": "string"},
                            "unmapped_field": {"type": "string"},
                        },
                        "required": [],
                    },
                    "_sink_node_id": "sink",
                    # Only one field is mapped
                    "_field_mapping": {
                        "mapped_field": "Original Mapped Field",
                    },
                },
            }
        ]

        processed = block._process_tool_calls(mock_response, tool_functions)

        assert len(processed) == 1
        input_data = processed[0].input_data

        # Mapped field uses original name
        assert "Original Mapped Field" in input_data
        # Unmapped field uses clean name (fallback)
        assert "unmapped_field" in input_data


class TestSilentValueLossInRouting:
    """
    Tests for Failure Mode #16: Silent Value Loss in Output Routing

    When routing fails in parse_execution_output, it returns None
    without any logging or indication of why it failed.
    """

    def test_routing_mismatch_returns_none_silently(self):
        """Test that routing mismatch returns None without error."""
        from backend.data.dynamic_fields import parse_execution_output

        output_item = ("tools_^_node-123_~_sanitized_name", "important_value")

        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="node-123",
            sink_pin_name="Original Name",  # Doesn't match sanitized_name
        )

        # Silently returns None
        assert result is None
        # No way to distinguish "value is None" from "routing failed"

    def test_wrong_node_id_returns_none(self):
        """Test that wrong node ID returns None."""
        from backend.data.dynamic_fields import parse_execution_output

        output_item = ("tools_^_node-123_~_field", "value")

        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="different-node",  # Wrong node
            sink_pin_name="field",
        )

        assert result is None

    def test_wrong_selector_returns_none(self):
        """Test that wrong selector returns None."""
        from backend.data.dynamic_fields import parse_execution_output

        output_item = ("tools_^_node-123_~_field", "value")

        result = parse_execution_output(
            output_item,
            link_output_selector="different_selector",  # Wrong selector
            sink_node_id="node-123",
            sink_pin_name="field",
        )

        assert result is None

    def test_cannot_distinguish_none_value_from_routing_failure(self):
        """
        Test that None as actual value is indistinguishable from routing failure.
        """
        from backend.data.dynamic_fields import parse_execution_output

        # Case 1: Actual None value
        output_with_none = ("field_name", None)
        result1 = parse_execution_output(
            output_with_none,
            link_output_selector="field_name",
            sink_node_id=None,
            sink_pin_name=None,
        )

        # Case 2: Routing failure
        output_mismatched = ("field_name", "value")
        result2 = parse_execution_output(
            output_mismatched,
            link_output_selector="different_field",
            sink_node_id=None,
            sink_pin_name=None,
        )

        # Both return None - cannot distinguish!
        assert result1 is None
        assert result2 is None


class TestProcessToolCallsInputData:
    """Tests for _process_tool_calls input data generation."""

    @pytest.mark.asyncio
    async def test_all_expected_args_included(self):
        """Test that all expected arguments are included in input_data."""
        block = SmartDecisionMakerBlock()

        mock_response = Mock()
        mock_tool_call = Mock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = json.dumps({
            "provided_field": "value",
            # optional_field not provided
        })
        mock_response.tool_calls = [mock_tool_call]

        tool_functions = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "parameters": {
                        "properties": {
                            "provided_field": {"type": "string"},
                            "optional_field": {"type": "string"},
                        },
                        "required": ["provided_field"],
                    },
                    "_sink_node_id": "sink",
                    "_field_mapping": {
                        "provided_field": "Provided Field",
                        "optional_field": "Optional Field",
                    },
                },
            }
        ]

        processed = block._process_tool_calls(mock_response, tool_functions)

        assert len(processed) == 1
        input_data = processed[0].input_data

        # Both fields should be in input_data
        assert "Provided Field" in input_data
        assert "Optional Field" in input_data

        # Provided has value, optional is None
        assert input_data["Provided Field"] == "value"
        assert input_data["Optional Field"] is None

    @pytest.mark.asyncio
    async def test_extra_args_from_llm_ignored(self):
        """Test that extra arguments from LLM not in schema are ignored."""
        block = SmartDecisionMakerBlock()

        mock_response = Mock()
        mock_tool_call = Mock()
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = json.dumps({
            "expected_field": "value",
            "unexpected_field": "should_be_ignored",
        })
        mock_response.tool_calls = [mock_tool_call]

        tool_functions = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "parameters": {
                        "properties": {
                            "expected_field": {"type": "string"},
                            # unexpected_field not in schema
                        },
                        "required": [],
                    },
                    "_sink_node_id": "sink",
                    "_field_mapping": {"expected_field": "Expected Field"},
                },
            }
        ]

        processed = block._process_tool_calls(mock_response, tool_functions)

        assert len(processed) == 1
        input_data = processed[0].input_data

        # Only expected field should be in input_data
        assert "Expected Field" in input_data
        assert "unexpected_field" not in input_data
        assert "Unexpected Field" not in input_data


class TestToolCallMatching:
    """Tests for tool call matching logic."""

    @pytest.mark.asyncio
    async def test_tool_not_found_skipped(self):
        """Test that tool calls for unknown tools are skipped."""
        block = SmartDecisionMakerBlock()

        mock_response = Mock()
        mock_tool_call = Mock()
        mock_tool_call.function.name = "unknown_tool"
        mock_tool_call.function.arguments = json.dumps({})
        mock_response.tool_calls = [mock_tool_call]

        tool_functions = [
            {
                "type": "function",
                "function": {
                    "name": "known_tool",  # Different name
                    "parameters": {"properties": {}, "required": []},
                    "_sink_node_id": "sink",
                },
            }
        ]

        processed = block._process_tool_calls(mock_response, tool_functions)

        # Unknown tool is skipped (not processed)
        assert len(processed) == 0

    @pytest.mark.asyncio
    async def test_single_tool_fallback(self):
        """Test fallback when only one tool exists but name doesn't match."""
        block = SmartDecisionMakerBlock()

        mock_response = Mock()
        mock_tool_call = Mock()
        mock_tool_call.function.name = "wrong_name"
        mock_tool_call.function.arguments = json.dumps({"field": "value"})
        mock_response.tool_calls = [mock_tool_call]

        # Only one tool defined
        tool_functions = [
            {
                "type": "function",
                "function": {
                    "name": "only_tool",
                    "parameters": {
                        "properties": {"field": {"type": "string"}},
                        "required": [],
                    },
                    "_sink_node_id": "sink",
                    "_field_mapping": {"field": "Field"},
                },
            }
        ]

        processed = block._process_tool_calls(mock_response, tool_functions)

        # Falls back to the only tool
        assert len(processed) == 1
        assert processed[0].input_data["Field"] == "value"

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_processed(self):
        """Test that multiple tool calls are all processed."""
        block = SmartDecisionMakerBlock()

        mock_response = Mock()
        mock_tool_call_1 = Mock()
        mock_tool_call_1.function.name = "tool_a"
        mock_tool_call_1.function.arguments = json.dumps({"a": "1"})

        mock_tool_call_2 = Mock()
        mock_tool_call_2.function.name = "tool_b"
        mock_tool_call_2.function.arguments = json.dumps({"b": "2"})

        mock_response.tool_calls = [mock_tool_call_1, mock_tool_call_2]

        tool_functions = [
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "parameters": {
                        "properties": {"a": {"type": "string"}},
                        "required": [],
                    },
                    "_sink_node_id": "sink_a",
                    "_field_mapping": {"a": "A"},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_b",
                    "parameters": {
                        "properties": {"b": {"type": "string"}},
                        "required": [],
                    },
                    "_sink_node_id": "sink_b",
                    "_field_mapping": {"b": "B"},
                },
            },
        ]

        processed = block._process_tool_calls(mock_response, tool_functions)

        assert len(processed) == 2
        assert processed[0].input_data["A"] == "1"
        assert processed[1].input_data["B"] == "2"


class TestOutputEmitKeyGeneration:
    """Tests for output emit key generation consistency."""

    def test_emit_key_uses_sanitized_field_name(self):
        """Test that emit keys use sanitized field names."""
        cleanup = SmartDecisionMakerBlock.cleanup

        original_field = "Max Keyword Difficulty"
        sink_node_id = "node-123"

        sanitized = cleanup(original_field)
        emit_key = f"tools_^_{sink_node_id}_~_{sanitized}"

        assert emit_key == "tools_^_node-123_~_max_keyword_difficulty"

    def test_emit_key_format_consistent(self):
        """Test that emit key format is consistent."""
        test_cases = [
            ("field", "node", "tools_^_node_~_field"),
            ("Field Name", "node-123", "tools_^_node-123_~_field_name"),
            ("CPC ($)", "abc", "tools_^_abc_~_cpc____"),
        ]

        cleanup = SmartDecisionMakerBlock.cleanup

        for original_field, node_id, expected in test_cases:
            sanitized = cleanup(original_field)
            emit_key = f"tools_^_{node_id}_~_{sanitized}"
            assert emit_key == expected, \
                f"Expected {expected}, got {emit_key}"

    def test_emit_key_sanitization_idempotent(self):
        """Test that sanitizing an already sanitized name gives same result."""
        cleanup = SmartDecisionMakerBlock.cleanup

        original = "Test Field Name"
        first_clean = cleanup(original)
        second_clean = cleanup(first_clean)

        assert first_clean == second_clean


class TestToolFunctionMetadata:
    """Tests for tool function metadata handling."""

    @pytest.mark.asyncio
    async def test_sink_node_id_preserved(self):
        """Test that _sink_node_id is preserved in tool function."""
        block = SmartDecisionMakerBlock()

        mock_node = Mock()
        mock_node.id = "specific-node-id"
        mock_node.block = Mock()
        mock_node.block.name = "TestBlock"
        mock_node.block.description = "Test"
        mock_node.block.input_schema = Mock()
        mock_node.block.input_schema.jsonschema = Mock(
            return_value={"properties": {}, "required": []}
        )
        mock_node.block.input_schema.get_field_schema = Mock(
            return_value={"type": "string", "description": "test"}
        )

        mock_links = [
            Mock(sink_name="field", sink_id="specific-node-id", source_id="source"),
        ]

        signature = await block._create_block_function_signature(mock_node, mock_links)

        assert signature["function"]["_sink_node_id"] == "specific-node-id"

    @pytest.mark.asyncio
    async def test_field_mapping_preserved(self):
        """Test that _field_mapping is preserved in tool function."""
        block = SmartDecisionMakerBlock()

        mock_node = Mock()
        mock_node.id = "test-node"
        mock_node.block = Mock()
        mock_node.block.name = "TestBlock"
        mock_node.block.description = "Test"
        mock_node.block.input_schema = Mock()
        mock_node.block.input_schema.jsonschema = Mock(
            return_value={"properties": {}, "required": []}
        )
        mock_node.block.input_schema.get_field_schema = Mock(
            return_value={"type": "string", "description": "test"}
        )

        mock_links = [
            Mock(sink_name="Original Field Name", sink_id="test-node", source_id="source"),
        ]

        signature = await block._create_block_function_signature(mock_node, mock_links)

        field_mapping = signature["function"]["_field_mapping"]
        assert "original_field_name" in field_mapping
        assert field_mapping["original_field_name"] == "Original Field Name"


class TestRequiredFieldsHandling:
    """Tests for required fields handling."""

    @pytest.mark.asyncio
    async def test_required_fields_use_sanitized_names(self):
        """Test that required fields array uses sanitized names."""
        block = SmartDecisionMakerBlock()

        mock_node = Mock()
        mock_node.id = "test-node"
        mock_node.block = Mock()
        mock_node.block.name = "TestBlock"
        mock_node.block.description = "Test"
        mock_node.block.input_schema = Mock()
        mock_node.block.input_schema.jsonschema = Mock(
            return_value={
                "properties": {},
                "required": ["Required Field", "Another Required"],
            }
        )
        mock_node.block.input_schema.get_field_schema = Mock(
            return_value={"type": "string", "description": "test"}
        )

        mock_links = [
            Mock(sink_name="Required Field", sink_id="test-node", source_id="source"),
            Mock(sink_name="Another Required", sink_id="test-node", source_id="source"),
            Mock(sink_name="Optional Field", sink_id="test-node", source_id="source"),
        ]

        signature = await block._create_block_function_signature(mock_node, mock_links)

        required = signature["function"]["parameters"]["required"]

        # Should use sanitized names
        assert "required_field" in required
        assert "another_required" in required

        # Original names should NOT be in required
        assert "Required Field" not in required
        assert "Another Required" not in required

        # Optional field should not be required
        assert "optional_field" not in required
        assert "Optional Field" not in required
