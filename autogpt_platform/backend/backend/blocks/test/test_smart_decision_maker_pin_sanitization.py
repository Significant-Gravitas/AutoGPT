"""
Comprehensive tests for SmartDecisionMakerBlock pin name sanitization.

This test file addresses the critical bug where field names with spaces/special characters
(e.g., "Max Keyword Difficulty") are not consistently sanitized between frontend and backend,
causing tool calls to "go into the void".

The core issue:
- Frontend connects link with original name: tools_^_{node_id}_~_Max Keyword Difficulty
- Backend emits with sanitized name: tools_^_{node_id}_~_max_keyword_difficulty
- parse_execution_output compares sink_pin_name directly without sanitization
- Result: mismatch causes tool calls to fail silently
"""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock
from backend.data.dynamic_fields import (
    parse_execution_output,
    sanitize_pin_name,
)


class TestCleanupFunction:
    """Tests for the SmartDecisionMakerBlock.cleanup() static method."""

    def test_cleanup_spaces_to_underscores(self):
        """Spaces should be replaced with underscores."""
        assert SmartDecisionMakerBlock.cleanup("Max Keyword Difficulty") == "max_keyword_difficulty"

    def test_cleanup_mixed_case_to_lowercase(self):
        """Mixed case should be converted to lowercase."""
        assert SmartDecisionMakerBlock.cleanup("MaxKeywordDifficulty") == "maxkeyworddifficulty"
        assert SmartDecisionMakerBlock.cleanup("UPPER_CASE") == "upper_case"

    def test_cleanup_special_characters(self):
        """Special characters should be replaced with underscores."""
        assert SmartDecisionMakerBlock.cleanup("field@name!") == "field_name_"
        assert SmartDecisionMakerBlock.cleanup("value#1") == "value_1"
        assert SmartDecisionMakerBlock.cleanup("test$value") == "test_value"
        assert SmartDecisionMakerBlock.cleanup("a%b^c") == "a_b_c"

    def test_cleanup_preserves_valid_characters(self):
        """Valid characters (alphanumeric, underscore, hyphen) should be preserved."""
        assert SmartDecisionMakerBlock.cleanup("valid_name-123") == "valid_name-123"
        assert SmartDecisionMakerBlock.cleanup("abc123") == "abc123"

    def test_cleanup_empty_string(self):
        """Empty string should return empty string."""
        assert SmartDecisionMakerBlock.cleanup("") == ""

    def test_cleanup_only_special_chars(self):
        """String of only special characters should return underscores."""
        assert SmartDecisionMakerBlock.cleanup("@#$%") == "____"

    def test_cleanup_unicode_characters(self):
        """Unicode characters should be replaced with underscores."""
        assert SmartDecisionMakerBlock.cleanup("café") == "caf_"
        assert SmartDecisionMakerBlock.cleanup("日本語") == "___"

    def test_cleanup_multiple_consecutive_spaces(self):
        """Multiple consecutive spaces should become multiple underscores."""
        assert SmartDecisionMakerBlock.cleanup("a   b") == "a___b"

    def test_cleanup_leading_trailing_spaces(self):
        """Leading/trailing spaces should become underscores."""
        assert SmartDecisionMakerBlock.cleanup(" name ") == "_name_"

    def test_cleanup_realistic_field_names(self):
        """Test realistic field names from actual use cases."""
        # From the reported bug
        assert SmartDecisionMakerBlock.cleanup("Max Keyword Difficulty") == "max_keyword_difficulty"
        # Other realistic names
        assert SmartDecisionMakerBlock.cleanup("Search Query") == "search_query"
        assert SmartDecisionMakerBlock.cleanup("API Response (JSON)") == "api_response__json_"
        assert SmartDecisionMakerBlock.cleanup("User's Input") == "user_s_input"


class TestFieldMappingCreation:
    """Tests for field mapping creation in function signatures."""

    @pytest.mark.asyncio
    async def test_field_mapping_with_spaces_in_names(self):
        """Test that field mapping correctly maps clean names back to original names with spaces."""
        block = SmartDecisionMakerBlock()

        mock_node = Mock()
        mock_node.id = "test-node-id"
        mock_node.block = Mock()
        mock_node.block.name = "TestBlock"
        mock_node.block.description = "Test description"
        mock_node.block.input_schema = Mock()
        mock_node.block.input_schema.jsonschema = Mock(
            return_value={"properties": {}, "required": ["Max Keyword Difficulty"]}
        )

        def get_field_schema(field_name):
            if field_name == "Max Keyword Difficulty":
                return {"type": "integer", "description": "Maximum keyword difficulty (0-100)"}
            raise KeyError(f"Field {field_name} not found")

        mock_node.block.input_schema.get_field_schema = get_field_schema

        mock_links = [
            Mock(
                source_name="tools_^_test_~_max_keyword_difficulty",
                sink_name="Max Keyword Difficulty",  # Original name with spaces
                sink_id="test-node-id",
                source_id="smart_node_id",
            ),
        ]

        signature = await block._create_block_function_signature(mock_node, mock_links)

        # Verify the cleaned name is used in properties
        properties = signature["function"]["parameters"]["properties"]
        assert "max_keyword_difficulty" in properties

        # Verify the field mapping maps back to original
        field_mapping = signature["function"]["_field_mapping"]
        assert field_mapping["max_keyword_difficulty"] == "Max Keyword Difficulty"

    @pytest.mark.asyncio
    async def test_field_mapping_with_multiple_special_char_names(self):
        """Test field mapping with multiple fields containing special characters."""
        block = SmartDecisionMakerBlock()

        mock_node = Mock()
        mock_node.id = "test-node-id"
        mock_node.block = Mock()
        mock_node.block.name = "SEO Tool"
        mock_node.block.description = "SEO analysis tool"
        mock_node.block.input_schema = Mock()
        mock_node.block.input_schema.jsonschema = Mock(
            return_value={"properties": {}, "required": []}
        )

        def get_field_schema(field_name):
            schemas = {
                "Max Keyword Difficulty": {"type": "integer", "description": "Max difficulty"},
                "Search Volume (Monthly)": {"type": "integer", "description": "Monthly volume"},
                "CPC ($)": {"type": "number", "description": "Cost per click"},
                "Target URL": {"type": "string", "description": "URL to analyze"},
            }
            if field_name in schemas:
                return schemas[field_name]
            raise KeyError(f"Field {field_name} not found")

        mock_node.block.input_schema.get_field_schema = get_field_schema

        mock_links = [
            Mock(sink_name="Max Keyword Difficulty", sink_id="test-node-id", source_id="smart_node_id"),
            Mock(sink_name="Search Volume (Monthly)", sink_id="test-node-id", source_id="smart_node_id"),
            Mock(sink_name="CPC ($)", sink_id="test-node-id", source_id="smart_node_id"),
            Mock(sink_name="Target URL", sink_id="test-node-id", source_id="smart_node_id"),
        ]

        signature = await block._create_block_function_signature(mock_node, mock_links)

        properties = signature["function"]["parameters"]["properties"]
        field_mapping = signature["function"]["_field_mapping"]

        # Verify all cleaned names are in properties
        assert "max_keyword_difficulty" in properties
        assert "search_volume__monthly_" in properties
        assert "cpc____" in properties
        assert "target_url" in properties

        # Verify field mappings
        assert field_mapping["max_keyword_difficulty"] == "Max Keyword Difficulty"
        assert field_mapping["search_volume__monthly_"] == "Search Volume (Monthly)"
        assert field_mapping["cpc____"] == "CPC ($)"
        assert field_mapping["target_url"] == "Target URL"


class TestFieldNameCollision:
    """Tests for detecting field name collisions after sanitization."""

    @pytest.mark.asyncio
    async def test_collision_detection_same_sanitized_name(self):
        """Test behavior when two different names sanitize to the same value."""
        block = SmartDecisionMakerBlock()

        # These two different names will sanitize to the same value
        name1 = "max keyword difficulty"  # -> max_keyword_difficulty
        name2 = "Max Keyword Difficulty"  # -> max_keyword_difficulty
        name3 = "MAX_KEYWORD_DIFFICULTY"  # -> max_keyword_difficulty

        assert SmartDecisionMakerBlock.cleanup(name1) == SmartDecisionMakerBlock.cleanup(name2)
        assert SmartDecisionMakerBlock.cleanup(name2) == SmartDecisionMakerBlock.cleanup(name3)

    @pytest.mark.asyncio
    async def test_collision_in_function_signature(self):
        """Test that collisions in sanitized names could cause issues."""
        block = SmartDecisionMakerBlock()

        mock_node = Mock()
        mock_node.id = "test-node-id"
        mock_node.block = Mock()
        mock_node.block.name = "TestBlock"
        mock_node.block.description = "Test description"
        mock_node.block.input_schema = Mock()
        mock_node.block.input_schema.jsonschema = Mock(
            return_value={"properties": {}, "required": []}
        )

        def get_field_schema(field_name):
            return {"type": "string", "description": f"Field: {field_name}"}

        mock_node.block.input_schema.get_field_schema = get_field_schema

        # Two different fields that sanitize to the same name
        mock_links = [
            Mock(sink_name="Test Field", sink_id="test-node-id", source_id="smart_node_id"),
            Mock(sink_name="test field", sink_id="test-node-id", source_id="smart_node_id"),
        ]

        signature = await block._create_block_function_signature(mock_node, mock_links)

        properties = signature["function"]["parameters"]["properties"]
        field_mapping = signature["function"]["_field_mapping"]

        # Both sanitize to "test_field" - only one will be in properties
        assert "test_field" in properties
        # The field_mapping will have the last one written
        assert field_mapping["test_field"] in ["Test Field", "test field"]


class TestOutputRouting:
    """Tests for output routing with sanitized names."""

    def test_emit_key_format_with_spaces(self):
        """Test that emit keys use sanitized field names."""
        block = SmartDecisionMakerBlock()

        original_field_name = "Max Keyword Difficulty"
        sink_node_id = "node-123"

        sanitized_name = block.cleanup(original_field_name)
        emit_key = f"tools_^_{sink_node_id}_~_{sanitized_name}"

        assert emit_key == "tools_^_node-123_~_max_keyword_difficulty"

    def test_parse_execution_output_exact_match(self):
        """Test parse_execution_output with exact matching names."""
        output_item = ("tools_^_node-123_~_max_keyword_difficulty", 50)

        # When sink_pin_name matches the sanitized name, it should work
        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="node-123",
            sink_pin_name="max_keyword_difficulty",
        )
        assert result == 50

    def test_parse_execution_output_mismatch_original_vs_sanitized(self):
        """
        CRITICAL TEST: This reproduces the exact bug reported.

        When frontend creates a link with original name "Max Keyword Difficulty"
        but backend emits with sanitized name "max_keyword_difficulty",
        the tool call should still be routed correctly.

        CURRENT BEHAVIOR (BUG): Returns None because names don't match
        EXPECTED BEHAVIOR: Should return the value (50) after sanitizing both names
        """
        output_item = ("tools_^_node-123_~_max_keyword_difficulty", 50)

        # This is what happens: sink_pin_name comes from frontend link (unsanitized)
        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="node-123",
            sink_pin_name="Max Keyword Difficulty",  # Original name with spaces
        )

        # BUG: This currently returns None because:
        # - target_input_pin = "max_keyword_difficulty" (from emit key, sanitized)
        # - sink_pin_name = "Max Keyword Difficulty" (from link, original)
        # - They don't match, so routing fails
        #
        # TODO: When the bug is fixed, change this assertion to:
        # assert result == 50
        assert result is None  # Current buggy behavior

    def test_parse_execution_output_with_sanitized_sink_pin(self):
        """Test that if sink_pin_name is pre-sanitized, routing works."""
        output_item = ("tools_^_node-123_~_max_keyword_difficulty", 50)

        # If sink_pin_name is already sanitized, routing works
        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="node-123",
            sink_pin_name="max_keyword_difficulty",  # Pre-sanitized
        )
        assert result == 50


class TestProcessToolCallsMapping:
    """Tests for _process_tool_calls method field mapping."""

    @pytest.mark.asyncio
    async def test_process_tool_calls_maps_clean_to_original(self):
        """Test that _process_tool_calls correctly maps clean names back to original."""
        block = SmartDecisionMakerBlock()

        mock_response = Mock()
        mock_tool_call = Mock()
        mock_tool_call.function.name = "seo_tool"
        mock_tool_call.function.arguments = json.dumps({
            "max_keyword_difficulty": 50,  # LLM uses clean name
            "search_query": "test query",
        })
        mock_response.tool_calls = [mock_tool_call]

        tool_functions = [
            {
                "type": "function",
                "function": {
                    "name": "seo_tool",
                    "parameters": {
                        "properties": {
                            "max_keyword_difficulty": {"type": "integer"},
                            "search_query": {"type": "string"},
                        },
                        "required": ["max_keyword_difficulty", "search_query"],
                    },
                    "_sink_node_id": "test-sink-node",
                    "_field_mapping": {
                        "max_keyword_difficulty": "Max Keyword Difficulty",  # Original name
                        "search_query": "Search Query",
                    },
                },
            }
        ]

        processed = block._process_tool_calls(mock_response, tool_functions)

        assert len(processed) == 1
        tool_info = processed[0]

        # Verify input_data uses ORIGINAL field names
        assert "Max Keyword Difficulty" in tool_info.input_data
        assert "Search Query" in tool_info.input_data
        assert tool_info.input_data["Max Keyword Difficulty"] == 50
        assert tool_info.input_data["Search Query"] == "test query"


class TestToolOutputEmitting:
    """Tests for the tool output emitting in traditional mode."""

    @pytest.mark.asyncio
    async def test_emit_keys_use_sanitized_names(self):
        """Test that emit keys always use sanitized field names."""
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "seo_tool"
        mock_tool_call.function.arguments = json.dumps({
            "max_keyword_difficulty": 50,
        })

        mock_response = MagicMock()
        mock_response.response = None
        mock_response.tool_calls = [mock_tool_call]
        mock_response.prompt_tokens = 50
        mock_response.completion_tokens = 25
        mock_response.reasoning = None
        mock_response.raw_response = {"role": "assistant", "content": None}

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "seo_tool",
                    "_sink_node_id": "test-sink-node-id",
                    "_field_mapping": {
                        "max_keyword_difficulty": "Max Keyword Difficulty",
                    },
                    "parameters": {
                        "properties": {
                            "max_keyword_difficulty": {"type": "integer"},
                        },
                        "required": ["max_keyword_difficulty"],
                    },
                },
            }
        ]

        with patch(
            "backend.blocks.llm.llm_call",
            new_callable=AsyncMock,
            return_value=mock_response,
        ), patch.object(
            block, "_create_tool_node_signatures", return_value=mock_tool_signatures
        ):
            input_data = SmartDecisionMakerBlock.Input(
                prompt="Test prompt",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=0,
            )

            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = MagicMock()

            outputs = {}
            async for output_name, output_data in block.run(
                input_data,
                credentials=llm_module.TEST_CREDENTIALS,
                graph_id="test-graph-id",
                node_id="test-node-id",
                graph_exec_id="test-exec-id",
                node_exec_id="test-node-exec-id",
                user_id="test-user-id",
                graph_version=1,
                execution_context=mock_execution_context,
                execution_processor=mock_execution_processor,
            ):
                outputs[output_name] = output_data

            # The emit key should use the sanitized field name
            # Even though the original was "Max Keyword Difficulty", emit uses sanitized
            assert "tools_^_test-sink-node-id_~_max_keyword_difficulty" in outputs
            assert outputs["tools_^_test-sink-node-id_~_max_keyword_difficulty"] == 50


class TestSanitizationConsistency:
    """Tests for ensuring sanitization is consistent throughout the pipeline."""

    @pytest.mark.asyncio
    async def test_full_round_trip_with_spaces(self):
        """
        Test the full round-trip of a field name with spaces through the system.

        This simulates:
        1. Frontend creates link with sink_name="Max Keyword Difficulty"
        2. Backend creates function signature with cleaned property name
        3. LLM responds with cleaned name
        4. Backend processes response and maps back to original
        5. Backend emits with sanitized name
        6. Routing should match (currently broken)
        """
        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        original_field_name = "Max Keyword Difficulty"
        cleaned_field_name = SmartDecisionMakerBlock.cleanup(original_field_name)

        # Step 1: Simulate frontend link creation
        mock_link = Mock()
        mock_link.sink_name = original_field_name  # Frontend uses original
        mock_link.sink_id = "test-sink-node-id"
        mock_link.source_id = "smart-node-id"

        # Step 2: Create function signature
        mock_node = Mock()
        mock_node.id = "test-sink-node-id"
        mock_node.block = Mock()
        mock_node.block.name = "SEO Tool"
        mock_node.block.description = "SEO analysis"
        mock_node.block.input_schema = Mock()
        mock_node.block.input_schema.jsonschema = Mock(
            return_value={"properties": {}, "required": [original_field_name]}
        )
        mock_node.block.input_schema.get_field_schema = Mock(
            return_value={"type": "integer", "description": "Max difficulty"}
        )

        signature = await block._create_block_function_signature(mock_node, [mock_link])

        # Verify cleaned name is in properties
        assert cleaned_field_name in signature["function"]["parameters"]["properties"]
        # Verify field mapping exists
        assert signature["function"]["_field_mapping"][cleaned_field_name] == original_field_name

        # Step 3: Simulate LLM response using cleaned name
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "seo_tool"
        mock_tool_call.function.arguments = json.dumps({
            cleaned_field_name: 50  # LLM uses cleaned name
        })

        mock_response = MagicMock()
        mock_response.response = None
        mock_response.tool_calls = [mock_tool_call]
        mock_response.prompt_tokens = 50
        mock_response.completion_tokens = 25
        mock_response.reasoning = None
        mock_response.raw_response = {"role": "assistant", "content": None}

        # Prepare tool_functions as they would be in run()
        tool_functions = [
            {
                "type": "function",
                "function": {
                    "name": "seo_tool",
                    "_sink_node_id": "test-sink-node-id",
                    "_field_mapping": signature["function"]["_field_mapping"],
                    "parameters": signature["function"]["parameters"],
                },
            }
        ]

        # Step 4: Process tool calls
        processed = block._process_tool_calls(mock_response, tool_functions)
        assert len(processed) == 1
        # Input data should have ORIGINAL name
        assert original_field_name in processed[0].input_data
        assert processed[0].input_data[original_field_name] == 50

        # Step 5: Emit key generation (from run method logic)
        field_mapping = processed[0].field_mapping
        for clean_arg_name in signature["function"]["parameters"]["properties"]:
            original = field_mapping.get(clean_arg_name, clean_arg_name)
            sanitized_arg_name = block.cleanup(original)
            emit_key = f"tools_^_test-sink-node-id_~_{sanitized_arg_name}"

            # Emit key uses sanitized name
            assert emit_key == f"tools_^_test-sink-node-id_~_{cleaned_field_name}"

        # Step 6: Routing check (this is where the bug manifests)
        emit_key = f"tools_^_test-sink-node-id_~_{cleaned_field_name}"
        output_item = (emit_key, 50)

        # Current routing uses original sink_name from link
        result = parse_execution_output(
            output_item,
            link_output_selector="tools",
            sink_node_id="test-sink-node-id",
            sink_pin_name=original_field_name,  # Frontend's original name
        )

        # BUG: This returns None because sanitized != original
        # When fixed, this should return 50
        assert result is None  # Current broken behavior

    def test_sanitization_is_idempotent(self):
        """Test that sanitizing an already sanitized name gives the same result."""
        original = "Max Keyword Difficulty"
        first_clean = SmartDecisionMakerBlock.cleanup(original)
        second_clean = SmartDecisionMakerBlock.cleanup(first_clean)

        assert first_clean == second_clean


class TestEdgeCases:
    """Tests for edge cases in the sanitization pipeline."""

    @pytest.mark.asyncio
    async def test_empty_field_name(self):
        """Test handling of empty field name."""
        assert SmartDecisionMakerBlock.cleanup("") == ""

    @pytest.mark.asyncio
    async def test_very_long_field_name(self):
        """Test handling of very long field names."""
        long_name = "A" * 1000 + " " + "B" * 1000
        cleaned = SmartDecisionMakerBlock.cleanup(long_name)
        assert "_" in cleaned  # Space was replaced
        assert len(cleaned) == len(long_name)

    @pytest.mark.asyncio
    async def test_field_name_with_newlines(self):
        """Test handling of field names with newlines."""
        name_with_newline = "First Line\nSecond Line"
        cleaned = SmartDecisionMakerBlock.cleanup(name_with_newline)
        assert "\n" not in cleaned
        assert "_" in cleaned

    @pytest.mark.asyncio
    async def test_field_name_with_tabs(self):
        """Test handling of field names with tabs."""
        name_with_tab = "First\tSecond"
        cleaned = SmartDecisionMakerBlock.cleanup(name_with_tab)
        assert "\t" not in cleaned
        assert "_" in cleaned

    @pytest.mark.asyncio
    async def test_numeric_field_name(self):
        """Test handling of purely numeric field names."""
        assert SmartDecisionMakerBlock.cleanup("123") == "123"
        assert SmartDecisionMakerBlock.cleanup("123 456") == "123_456"

    @pytest.mark.asyncio
    async def test_hyphenated_field_names(self):
        """Test that hyphens are preserved (valid in function names)."""
        assert SmartDecisionMakerBlock.cleanup("field-name") == "field-name"
        assert SmartDecisionMakerBlock.cleanup("Field-Name") == "field-name"


class TestDynamicFieldsWithSpaces:
    """Tests for dynamic fields with spaces in their names."""

    @pytest.mark.asyncio
    async def test_dynamic_dict_field_with_spaces(self):
        """Test dynamic dictionary fields where the key contains spaces."""
        block = SmartDecisionMakerBlock()

        mock_node = Mock()
        mock_node.id = "test-node-id"
        mock_node.block = Mock()
        mock_node.block.name = "CreateDictionary"
        mock_node.block.description = "Creates a dictionary"
        mock_node.block.input_schema = Mock()
        mock_node.block.input_schema.jsonschema = Mock(
            return_value={"properties": {}, "required": ["values"]}
        )
        mock_node.block.input_schema.get_field_schema = Mock(
            side_effect=KeyError("not found")
        )

        # Dynamic field with a key containing spaces
        mock_links = [
            Mock(
                sink_name="values_#_User Name",  # Dict key with space
                sink_id="test-node-id",
                source_id="smart_node_id",
            ),
        ]

        signature = await block._create_block_function_signature(mock_node, mock_links)

        properties = signature["function"]["parameters"]["properties"]
        field_mapping = signature["function"]["_field_mapping"]

        # The cleaned name should be in properties
        expected_clean = SmartDecisionMakerBlock.cleanup("values_#_User Name")
        assert expected_clean in properties

        # Field mapping should map back to original
        assert field_mapping[expected_clean] == "values_#_User Name"


class TestAgentModeWithSpaces:
    """Tests for agent mode with field names containing spaces."""

    @pytest.mark.asyncio
    async def test_agent_mode_tool_execution_with_spaces(self):
        """Test that agent mode correctly handles field names with spaces."""
        import threading
        from collections import defaultdict

        import backend.blocks.llm as llm_module
        from backend.data.execution import ExecutionContext

        block = SmartDecisionMakerBlock()

        original_field = "Max Keyword Difficulty"
        clean_field = SmartDecisionMakerBlock.cleanup(original_field)

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "seo_tool"
        mock_tool_call.function.arguments = json.dumps({
            clean_field: 50  # LLM uses clean name
        })

        mock_response_1 = MagicMock()
        mock_response_1.response = None
        mock_response_1.tool_calls = [mock_tool_call]
        mock_response_1.prompt_tokens = 50
        mock_response_1.completion_tokens = 25
        mock_response_1.reasoning = None
        mock_response_1.raw_response = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function"}],
        }

        mock_response_2 = MagicMock()
        mock_response_2.response = "Task completed"
        mock_response_2.tool_calls = []
        mock_response_2.prompt_tokens = 30
        mock_response_2.completion_tokens = 15
        mock_response_2.reasoning = None
        mock_response_2.raw_response = {"role": "assistant", "content": "Task completed"}

        llm_call_mock = AsyncMock()
        llm_call_mock.side_effect = [mock_response_1, mock_response_2]

        mock_tool_signatures = [
            {
                "type": "function",
                "function": {
                    "name": "seo_tool",
                    "_sink_node_id": "test-sink-node-id",
                    "_field_mapping": {
                        clean_field: original_field,
                    },
                    "parameters": {
                        "properties": {
                            clean_field: {"type": "integer"},
                        },
                        "required": [clean_field],
                    },
                },
            }
        ]

        mock_db_client = AsyncMock()
        mock_node = MagicMock()
        mock_node.block_id = "test-block-id"
        mock_db_client.get_node.return_value = mock_node

        mock_node_exec_result = MagicMock()
        mock_node_exec_result.node_exec_id = "test-tool-exec-id"

        # The input data should use ORIGINAL field name
        mock_input_data = {original_field: 50}
        mock_db_client.upsert_execution_input.return_value = (
            mock_node_exec_result,
            mock_input_data,
        )
        mock_db_client.get_execution_outputs_by_node_exec_id.return_value = {
            "result": {"status": "success"}
        }

        with patch("backend.blocks.llm.llm_call", llm_call_mock), patch.object(
            block, "_create_tool_node_signatures", return_value=mock_tool_signatures
        ), patch(
            "backend.blocks.smart_decision_maker.get_database_manager_async_client",
            return_value=mock_db_client,
        ):
            mock_execution_context = ExecutionContext(safe_mode=False)
            mock_execution_processor = AsyncMock()
            mock_execution_processor.running_node_execution = defaultdict(MagicMock)
            mock_execution_processor.execution_stats = MagicMock()
            mock_execution_processor.execution_stats_lock = threading.Lock()

            mock_node_stats = MagicMock()
            mock_node_stats.error = None
            mock_execution_processor.on_node_execution = AsyncMock(
                return_value=mock_node_stats
            )

            input_data = SmartDecisionMakerBlock.Input(
                prompt="Analyze keywords",
                model=llm_module.DEFAULT_LLM_MODEL,
                credentials=llm_module.TEST_CREDENTIALS_INPUT,
                agent_mode_max_iterations=3,
            )

            outputs = {}
            async for output_name, output_data in block.run(
                input_data,
                credentials=llm_module.TEST_CREDENTIALS,
                graph_id="test-graph-id",
                node_id="test-node-id",
                graph_exec_id="test-exec-id",
                node_exec_id="test-node-exec-id",
                user_id="test-user-id",
                graph_version=1,
                execution_context=mock_execution_context,
                execution_processor=mock_execution_processor,
            ):
                outputs[output_name] = output_data

            # Verify upsert was called with original field name
            upsert_calls = mock_db_client.upsert_execution_input.call_args_list
            assert len(upsert_calls) > 0
            # Check that the original field name was used
            for call in upsert_calls:
                input_name = call.kwargs.get("input_name") or call.args[2]
                # The input name should be the original (mapped back)
                assert input_name == original_field


class TestRequiredFieldsWithSpaces:
    """Tests for required field handling with spaces in names."""

    @pytest.mark.asyncio
    async def test_required_fields_use_clean_names(self):
        """Test that required fields array uses clean names for API compatibility."""
        block = SmartDecisionMakerBlock()

        mock_node = Mock()
        mock_node.id = "test-node-id"
        mock_node.block = Mock()
        mock_node.block.name = "TestBlock"
        mock_node.block.description = "Test"
        mock_node.block.input_schema = Mock()
        mock_node.block.input_schema.jsonschema = Mock(
            return_value={
                "properties": {},
                "required": ["Max Keyword Difficulty", "Search Query"],
            }
        )

        def get_field_schema(field_name):
            return {"type": "string", "description": f"Field: {field_name}"}

        mock_node.block.input_schema.get_field_schema = get_field_schema

        mock_links = [
            Mock(sink_name="Max Keyword Difficulty", sink_id="test-node-id", source_id="smart_node_id"),
            Mock(sink_name="Search Query", sink_id="test-node-id", source_id="smart_node_id"),
        ]

        signature = await block._create_block_function_signature(mock_node, mock_links)

        required = signature["function"]["parameters"]["required"]

        # Required array should use CLEAN names for API compatibility
        assert "max_keyword_difficulty" in required
        assert "search_query" in required
        # Original names should NOT be in required
        assert "Max Keyword Difficulty" not in required
        assert "Search Query" not in required
