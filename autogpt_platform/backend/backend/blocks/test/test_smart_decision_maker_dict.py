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
        if field_name == "values___name":
            assert "Dictionary field 'name'" in prop_value["description"]
            assert "values['name']" in prop_value["description"]


@pytest.mark.asyncio
async def test_smart_decision_maker_handles_dynamic_list_fields():
    """Test Smart Decision Maker can handle dynamic list fields (_$_) for any block"""

    # Create a mock node for AddToListBlock
    mock_node = Mock()
    mock_node.block = AddToListBlock()
    mock_node.block_id = AddToListBlock().id
    mock_node.input_default = {}

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
        # Check that descriptions properly explain the list field
        if field_name == "entries___0":
            assert "List item 0" in prop_value["description"]
            assert "entries[0]" in prop_value["description"]


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
