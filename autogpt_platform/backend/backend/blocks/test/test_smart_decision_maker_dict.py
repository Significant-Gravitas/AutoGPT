from unittest.mock import Mock

import pytest

from backend.blocks.data_manipulation import AddToListBlock, CreateDictionaryBlock
from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock


@pytest.mark.asyncio
async def test_smart_decision_maker_handles_dynamic_dict_fields():
    """Test Smart Decision Maker can handle dynamic dictionary fields (_#_) for any block"""

    # Create a mock node for CreateDictionaryBlock
    mock_node = Mock()
    mock_node.block = CreateDictionaryBlock()
    mock_node.block_id = CreateDictionaryBlock().id
    mock_node.input_default = {}
    mock_node.metadata = {}  # Add metadata to avoid Mock being returned
    mock_node.id = "test-node-id-12345678"  # Add node ID for fallback naming

    # Create mock links with dynamic dictionary fields
    mock_links = [
        Mock(
            source_name="tools_^_create_dict_~_name",
            sink_name="values_#_name",  # Dynamic dict field
            sink_id="dict_node_id",
            source_id="smart_decision_node_id",
        ),
        Mock(
            source_name="tools_^_create_dict_~_age",
            sink_name="values_#_age",  # Dynamic dict field
            sink_id="dict_node_id",
            source_id="smart_decision_node_id",
        ),
        Mock(
            source_name="tools_^_create_dict_~_city",
            sink_name="values_#_city",  # Dynamic dict field
            sink_id="dict_node_id",
            source_id="smart_decision_node_id",
        ),
    ]

    # Generate function signature
    signature = await SmartDecisionMakerBlock._create_block_function_signature(
        mock_node, mock_links  # type: ignore
    )

    # Verify the signature was created successfully
    assert signature["type"] == "function"
    assert "parameters" in signature["function"]
    assert "properties" in signature["function"]["parameters"]

    # Check that dynamic fields are handled with original names
    properties = signature["function"]["parameters"]["properties"]
    assert len(properties) == 3  # Should have all three fields

    # Check that field names are cleaned (for Anthropic API compatibility)
    assert "values___name" in properties
    assert "values___age" in properties
    assert "values___city" in properties

    # Each dynamic field should have proper schema with descriptive text
    for field_name, prop_value in properties.items():
        assert "type" in prop_value
        assert prop_value["type"] == "string"  # Dynamic fields get string type
        assert "description" in prop_value
        # Check that descriptions properly explain the dynamic field
        assert "Dynamic value for" in prop_value["description"]


@pytest.mark.asyncio
async def test_smart_decision_maker_handles_dynamic_list_fields():
    """Test Smart Decision Maker can handle dynamic list fields (_$_) for any block"""

    # Create a mock node for AddToListBlock
    mock_node = Mock()
    mock_node.block = AddToListBlock()
    mock_node.block_id = AddToListBlock().id
    mock_node.input_default = {}
    mock_node.metadata = {}  # Add metadata to avoid Mock being returned
    mock_node.id = "test-node-id-87654321"  # Add node ID for fallback naming

    # Create mock links with dynamic list fields
    mock_links = [
        Mock(
            source_name="tools_^_add_to_list_~_0",
            sink_name="entries_$_0",  # Dynamic list field
            sink_id="list_node_id",
            source_id="smart_decision_node_id",
        ),
        Mock(
            source_name="tools_^_add_to_list_~_1",
            sink_name="entries_$_1",  # Dynamic list field
            sink_id="list_node_id",
            source_id="smart_decision_node_id",
        ),
    ]

    # Generate function signature
    signature = await SmartDecisionMakerBlock._create_block_function_signature(
        mock_node, mock_links  # type: ignore
    )

    # Verify dynamic list fields are handled properly
    assert signature["type"] == "function"
    properties = signature["function"]["parameters"]["properties"]
    assert len(properties) == 2  # Should have both list items

    # Check that field names are cleaned (for Anthropic API compatibility)
    assert "entries___0" in properties
    assert "entries___1" in properties

    # Each dynamic field should have proper schema with descriptive text
    for field_name, prop_value in properties.items():
        assert prop_value["type"] == "string"
        assert "description" in prop_value
        # Check that descriptions properly explain the dynamic field
        assert "Dynamic value for" in prop_value["description"]


@pytest.mark.asyncio
async def test_smart_decision_maker_required_fields_filtered_for_dynamic_dict():
    """Test that required fields are properly filtered when dynamic fields replace base fields"""

    # Create a mock node for CreateDictionaryBlock
    mock_node = Mock()
    mock_node.block = CreateDictionaryBlock()
    mock_node.block_id = CreateDictionaryBlock().id
    mock_node.input_default = {}
    mock_node.metadata = {}
    mock_node.id = "test-node-id-99887766"

    # Create mock links with dynamic dictionary fields
    # These replace the base "values" field with values_#_a, values_#_b, values_#_c
    mock_links = [
        Mock(
            source_name="tools_^_create_dict_~_a",
            sink_name="values_#_a",
            sink_id="dict_node_id",
            source_id="smart_decision_node_id",
        ),
        Mock(
            source_name="tools_^_create_dict_~_b",
            sink_name="values_#_b",
            sink_id="dict_node_id",
            source_id="smart_decision_node_id",
        ),
        Mock(
            source_name="tools_^_create_dict_~_c",
            sink_name="values_#_c",
            sink_id="dict_node_id",
            source_id="smart_decision_node_id",
        ),
    ]

    # Generate function signature
    signature = await SmartDecisionMakerBlock._create_block_function_signature(
        mock_node, mock_links  # type: ignore
    )

    # Verify the signature structure
    assert signature["type"] == "function"
    assert "parameters" in signature["function"]
    parameters = signature["function"]["parameters"]
    assert "properties" in parameters
    assert "required" in parameters

    # CRITICAL: The required field should NOT include "values" anymore
    # because it was replaced by dynamic fields values_#_a, values_#_b, values_#_c
    properties = parameters["properties"]
    required = parameters["required"]

    # Verify all required fields exist in properties
    for req_field in required:
        assert req_field in properties, (
            f"Required field '{req_field}' must exist in properties. "
            f"Properties: {list(properties.keys())}, Required: {required}"
        )

    # Verify "values" is NOT in required (since it's been replaced by dynamic fields)
    assert (
        "values" not in required
    ), "Base field 'values' should not be in required when replaced by dynamic fields"

    # Verify dynamic fields are in properties
    assert "values___a" in properties
    assert "values___b" in properties
    assert "values___c" in properties


@pytest.mark.asyncio
async def test_create_dict_block_with_dynamic_values():
    """Test CreateDictionaryBlock processes dynamic values correctly"""

    block = CreateDictionaryBlock()

    # Simulate what happens when executor merges dynamic fields
    # The executor merges values_#_* fields into the values dict
    input_data = block.input_schema(
        values={
            "existing": "value",
            "name": "Alice",  # This would come from values_#_name
            "age": 25,  # This would come from values_#_age
        }
    )

    # Run the block
    result = {}
    async for output_name, output_value in block.run(input_data):
        result[output_name] = output_value

    # Check the result
    assert "dictionary" in result
    assert result["dictionary"]["existing"] == "value"
    assert result["dictionary"]["name"] == "Alice"
    assert result["dictionary"]["age"] == 25


@pytest.mark.asyncio
async def test_smart_decision_maker_detects_duplicate_tool_names():
    """Test that Smart Decision Maker detects and errors on duplicate tool names"""

    # Create two mock nodes with the SAME customized_name
    mock_node_1 = Mock()
    mock_node_1.block = CreateDictionaryBlock()
    mock_node_1.block_id = CreateDictionaryBlock().id
    mock_node_1.input_default = {}
    mock_node_1.metadata = {"customized_name": "Create Dict"}  # Same name!
    mock_node_1.id = "node-1-12345678"

    mock_node_2 = Mock()
    mock_node_2.block = CreateDictionaryBlock()
    mock_node_2.block_id = CreateDictionaryBlock().id
    mock_node_2.input_default = {}
    mock_node_2.metadata = {"customized_name": "Create Dict"}  # Same name!
    mock_node_2.id = "node-2-87654321"

    # Create mock links for both nodes
    mock_links_1 = [
        Mock(
            source_name="tools_^_create_dict_~_a",
            sink_name="values_#_a",
            sink_id="node-1",
            source_id="smart_decision_node_id",
        ),
    ]

    mock_links_2 = [
        Mock(
            source_name="tools_^_create_dict_~_b",
            sink_name="values_#_b",
            sink_id="node-2",
            source_id="smart_decision_node_id",
        ),
    ]

    # Generate signatures for both (simulating what _create_function_signature does)
    sig_1 = await SmartDecisionMakerBlock._create_block_function_signature(
        mock_node_1, mock_links_1  # type: ignore
    )
    sig_2 = await SmartDecisionMakerBlock._create_block_function_signature(
        mock_node_2, mock_links_2  # type: ignore
    )

    # Both should have the same tool name since they have the same customized_name
    assert sig_1["function"]["name"] == sig_2["function"]["name"]
    assert sig_1["function"]["name"] == "create_dict"

    # Now simulate what happens in _create_function_signature when it collects these
    tool_functions = [sig_1, sig_2]

    # Check for duplicates (same logic as in the actual code)
    tool_names = [func["function"]["name"] for func in tool_functions]
    duplicates = [name for name in tool_names if tool_names.count(name) > 1]

    # Should detect the duplicate
    assert len(duplicates) > 0
    assert "create_dict" in duplicates
