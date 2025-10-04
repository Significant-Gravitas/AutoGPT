"""Comprehensive tests for SmartDecisionMakerBlock dynamic field handling."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from backend.blocks.data_manipulation import AddToListBlock, CreateDictionaryBlock
from backend.blocks.smart_decision_maker import SmartDecisionMakerBlock
from backend.blocks.text import MatchTextPatternBlock
from backend.data.dynamic_fields import get_dynamic_field_description


@pytest.mark.asyncio
async def test_dynamic_field_description_generation():
    """Test that dynamic field descriptions are generated correctly."""
    # Test dictionary field description
    desc = get_dynamic_field_description("values_#_name")
    assert "Dictionary field 'name' for base field 'values'" in desc
    assert "values['name']" in desc

    # Test list field description
    desc = get_dynamic_field_description("items_$_0")
    assert "List item 0 for base field 'items'" in desc
    assert "items[0]" in desc

    # Test object field description
    desc = get_dynamic_field_description("user_@_email")
    assert "Object attribute 'email' for base field 'user'" in desc
    assert "user.email" in desc

    # Test regular field fallback
    desc = get_dynamic_field_description("regular_field")
    assert desc == "Value for regular_field"


@pytest.mark.asyncio
async def test_create_block_function_signature_with_dict_fields():
    """Test that function signatures are created correctly for dictionary dynamic fields."""
    block = SmartDecisionMakerBlock()

    # Create a mock node for CreateDictionaryBlock
    mock_node = Mock()
    mock_node.block = CreateDictionaryBlock()
    mock_node.block_id = CreateDictionaryBlock().id
    mock_node.input_default = {}

    # Create mock links with dynamic dictionary fields (source sanitized, sink original)
    mock_links = [
        Mock(
            source_name="tools_^_create_dict_~_values___name",  # Sanitized source
            sink_name="values_#_name",  # Original sink
            sink_id="dict_node_id",
            source_id="smart_decision_node_id",
        ),
        Mock(
            source_name="tools_^_create_dict_~_values___age",  # Sanitized source
            sink_name="values_#_age",  # Original sink
            sink_id="dict_node_id",
            source_id="smart_decision_node_id",
        ),
        Mock(
            source_name="tools_^_create_dict_~_values___email",  # Sanitized source
            sink_name="values_#_email",  # Original sink
            sink_id="dict_node_id",
            source_id="smart_decision_node_id",
        ),
    ]

    # Generate function signature
    signature = await block._create_block_function_signature(mock_node, mock_links)  # type: ignore

    # Verify the signature structure
    assert signature["type"] == "function"
    assert "function" in signature
    assert "parameters" in signature["function"]
    assert "properties" in signature["function"]["parameters"]

    # Check that dynamic fields are handled with original names
    properties = signature["function"]["parameters"]["properties"]
    assert len(properties) == 3

    # Check cleaned field names (for Anthropic API compatibility)
    assert "values___name" in properties
    assert "values___age" in properties
    assert "values___email" in properties

    # Check descriptions mention they are dictionary fields
    assert "Dictionary field" in properties["values___name"]["description"]
    assert "values['name']" in properties["values___name"]["description"]

    assert "Dictionary field" in properties["values___age"]["description"]
    assert "values['age']" in properties["values___age"]["description"]

    assert "Dictionary field" in properties["values___email"]["description"]
    assert "values['email']" in properties["values___email"]["description"]


@pytest.mark.asyncio
async def test_create_block_function_signature_with_list_fields():
    """Test that function signatures are created correctly for list dynamic fields."""
    block = SmartDecisionMakerBlock()

    # Create a mock node for AddToListBlock
    mock_node = Mock()
    mock_node.block = AddToListBlock()
    mock_node.block_id = AddToListBlock().id
    mock_node.input_default = {}

    # Create mock links with dynamic list fields
    mock_links = [
        Mock(
            source_name="tools_^_add_list_~_0",
            sink_name="entries_$_0",  # Dynamic list field
            sink_id="list_node_id",
            source_id="smart_decision_node_id",
        ),
        Mock(
            source_name="tools_^_add_list_~_1",
            sink_name="entries_$_1",  # Dynamic list field
            sink_id="list_node_id",
            source_id="smart_decision_node_id",
        ),
        Mock(
            source_name="tools_^_add_list_~_2",
            sink_name="entries_$_2",  # Dynamic list field
            sink_id="list_node_id",
            source_id="smart_decision_node_id",
        ),
    ]

    # Generate function signature
    signature = await block._create_block_function_signature(mock_node, mock_links)  # type: ignore

    # Verify the signature structure
    assert signature["type"] == "function"
    properties = signature["function"]["parameters"]["properties"]

    # Check cleaned field names (for Anthropic API compatibility)
    assert "entries___0" in properties
    assert "entries___1" in properties
    assert "entries___2" in properties

    # Check descriptions mention they are list items
    assert "List item 0" in properties["entries___0"]["description"]
    assert "entries[0]" in properties["entries___0"]["description"]

    assert "List item 1" in properties["entries___1"]["description"]
    assert "entries[1]" in properties["entries___1"]["description"]


@pytest.mark.asyncio
async def test_create_block_function_signature_with_object_fields():
    """Test that function signatures are created correctly for object dynamic fields."""
    block = SmartDecisionMakerBlock()

    # Create a mock node for MatchTextPatternBlock (simulating object fields)
    mock_node = Mock()
    mock_node.block = MatchTextPatternBlock()
    mock_node.block_id = MatchTextPatternBlock().id
    mock_node.input_default = {}

    # Create mock links with dynamic object fields
    mock_links = [
        Mock(
            source_name="tools_^_extract_~_user_name",
            sink_name="data_@_user_name",  # Dynamic object field
            sink_id="extract_node_id",
            source_id="smart_decision_node_id",
        ),
        Mock(
            source_name="tools_^_extract_~_user_email",
            sink_name="data_@_user_email",  # Dynamic object field
            sink_id="extract_node_id",
            source_id="smart_decision_node_id",
        ),
    ]

    # Generate function signature
    signature = await block._create_block_function_signature(mock_node, mock_links)  # type: ignore

    # Verify the signature structure
    properties = signature["function"]["parameters"]["properties"]

    # Check cleaned field names (for Anthropic API compatibility)
    assert "data___user_name" in properties
    assert "data___user_email" in properties

    # Check descriptions mention they are object attributes
    assert "Object attribute" in properties["data___user_name"]["description"]
    assert "data.user_name" in properties["data___user_name"]["description"]


@pytest.mark.asyncio
async def test_create_function_signature():
    """Test that the mapping between sanitized and original field names is built correctly."""
    block = SmartDecisionMakerBlock()

    # Mock the database client and connected nodes
    with patch(
        "backend.blocks.smart_decision_maker.get_database_manager_async_client"
    ) as mock_db:
        mock_client = AsyncMock()
        mock_db.return_value = mock_client

        # Create mock nodes and links
        mock_dict_node = Mock()
        mock_dict_node.block = CreateDictionaryBlock()
        mock_dict_node.block_id = CreateDictionaryBlock().id
        mock_dict_node.input_default = {}

        mock_list_node = Mock()
        mock_list_node.block = AddToListBlock()
        mock_list_node.block_id = AddToListBlock().id
        mock_list_node.input_default = {}

        # Mock links with dynamic fields
        dict_link1 = Mock(
            source_name="tools_^_create_dictionary_~_name",
            sink_name="values_#_name",
            sink_id="dict_node_id",
            source_id="test_node_id",
        )
        dict_link2 = Mock(
            source_name="tools_^_create_dictionary_~_age",
            sink_name="values_#_age",
            sink_id="dict_node_id",
            source_id="test_node_id",
        )
        list_link = Mock(
            source_name="tools_^_add_to_list_~_0",
            sink_name="entries_$_0",
            sink_id="list_node_id",
            source_id="test_node_id",
        )

        mock_client.get_connected_output_nodes.return_value = [
            (dict_link1, mock_dict_node),
            (dict_link2, mock_dict_node),
            (list_link, mock_list_node),
        ]

        # Call the method that builds signatures
        tool_functions = await block._create_function_signature("test_node_id")

        # Verify we got 2 tool functions (one for dict, one for list)
        assert len(tool_functions) == 2

        # Verify the tool functions contain the dynamic field names
        dict_tool = next(
            (
                tool
                for tool in tool_functions
                if tool["function"]["name"] == "createdictionaryblock"
            ),
            None,
        )
        assert dict_tool is not None
        dict_properties = dict_tool["function"]["parameters"]["properties"]
        assert "values___name" in dict_properties
        assert "values___age" in dict_properties

        list_tool = next(
            (
                tool
                for tool in tool_functions
                if tool["function"]["name"] == "addtolistblock"
            ),
            None,
        )
        assert list_tool is not None
        list_properties = list_tool["function"]["parameters"]["properties"]
        assert "entries___0" in list_properties


@pytest.mark.asyncio
async def test_output_yielding_with_dynamic_fields():
    """Test that outputs are yielded correctly with dynamic field names mapped back."""
    block = SmartDecisionMakerBlock()

    # No more sanitized mapping needed since we removed sanitization

    # Mock LLM response with tool calls
    mock_response = Mock()
    mock_response.tool_calls = [
        Mock(
            function=Mock(
                arguments=json.dumps(
                    {
                        "values___name": "Alice",
                        "values___age": 30,
                        "values___email": "alice@example.com",
                    }
                ),
            )
        )
    ]
    # Ensure function name is a real string, not a Mock name
    mock_response.tool_calls[0].function.name = "createdictionaryblock"
    mock_response.reasoning = "Creating a dictionary with user information"
    mock_response.raw_response = {"role": "assistant", "content": "test"}
    mock_response.prompt_tokens = 100
    mock_response.completion_tokens = 50

    # Mock the LLM call
    with patch(
        "backend.blocks.smart_decision_maker.llm.llm_call", new_callable=AsyncMock
    ) as mock_llm:
        mock_llm.return_value = mock_response

        # Mock the function signature creation
        with patch.object(
            block, "_create_function_signature", new_callable=AsyncMock
        ) as mock_sig:
            mock_sig.return_value = [
                {
                    "type": "function",
                    "function": {
                        "name": "createdictionaryblock",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "values___name": {"type": "string"},
                                "values___age": {"type": "number"},
                                "values___email": {"type": "string"},
                            },
                        },
                    },
                }
            ]

            # Create input data
            from backend.blocks import llm

            input_data = block.input_schema(
                prompt="Create a user dictionary",
                credentials=llm.TEST_CREDENTIALS_INPUT,
                model=llm.LlmModel.GPT4O,
            )

            # Run the block
            outputs = {}
            async for output_name, output_value in block.run(
                input_data,
                credentials=llm.TEST_CREDENTIALS,
                graph_id="test_graph",
                node_id="test_node",
                graph_exec_id="test_exec",
                node_exec_id="test_node_exec",
                user_id="test_user",
            ):
                outputs[output_name] = output_value

            # Verify the outputs use sanitized field names (matching frontend normalizeToolName)
            assert "tools_^_createdictionaryblock_~_values___name" in outputs
            assert outputs["tools_^_createdictionaryblock_~_values___name"] == "Alice"

            assert "tools_^_createdictionaryblock_~_values___age" in outputs
            assert outputs["tools_^_createdictionaryblock_~_values___age"] == 30

            assert "tools_^_createdictionaryblock_~_values___email" in outputs
            assert (
                outputs["tools_^_createdictionaryblock_~_values___email"]
                == "alice@example.com"
            )


@pytest.mark.asyncio
async def test_mixed_regular_and_dynamic_fields():
    """Test handling of blocks with both regular and dynamic fields."""
    block = SmartDecisionMakerBlock()

    # Create a mock node
    mock_node = Mock()
    mock_node.block = Mock()
    mock_node.block.name = "TestBlock"
    mock_node.block.description = "A test block"
    mock_node.block.input_schema = Mock()

    # Mock the get_field_schema to return a proper schema for regular fields
    def get_field_schema(field_name):
        if field_name == "regular_field":
            return {"type": "string", "description": "A regular field"}
        elif field_name == "values":
            return {"type": "object", "description": "A dictionary field"}
        else:
            raise KeyError(f"Field {field_name} not found")

    mock_node.block.input_schema.get_field_schema = get_field_schema
    mock_node.block.input_schema.jsonschema = Mock(
        return_value={"properties": {}, "required": []}
    )

    # Create links with both regular and dynamic fields
    mock_links = [
        Mock(
            source_name="tools_^_test_~_regular",
            sink_name="regular_field",  # Regular field
            sink_id="test_node_id",
            source_id="smart_decision_node_id",
        ),
        Mock(
            source_name="tools_^_test_~_dict_key",
            sink_name="values_#_key1",  # Dynamic dict field
            sink_id="test_node_id",
            source_id="smart_decision_node_id",
        ),
        Mock(
            source_name="tools_^_test_~_dict_key2",
            sink_name="values_#_key2",  # Dynamic dict field
            sink_id="test_node_id",
            source_id="smart_decision_node_id",
        ),
    ]

    # Generate function signature
    signature = await block._create_block_function_signature(mock_node, mock_links)  # type: ignore

    # Check properties
    properties = signature["function"]["parameters"]["properties"]
    assert len(properties) == 3

    # Regular field should have its original schema
    assert "regular_field" in properties
    assert properties["regular_field"]["description"] == "A regular field"

    # Dynamic fields should have generated descriptions
    assert "values___key1" in properties
    assert "Dictionary field" in properties["values___key1"]["description"]

    assert "values___key2" in properties
    assert "Dictionary field" in properties["values___key2"]["description"]


@pytest.mark.asyncio
async def test_validation_errors_dont_pollute_conversation():
    """Test that validation errors are only used during retries and don't pollute the conversation."""
    block = SmartDecisionMakerBlock()

    # Track conversation history changes
    conversation_snapshots = []

    # Mock response with invalid tool call (missing required parameter)
    invalid_response = Mock()
    invalid_response.tool_calls = [
        Mock(
            function=Mock(
                arguments=json.dumps({"wrong_param": "value"}),  # Wrong parameter name
            )
        )
    ]
    # Ensure function name is a real string, not a Mock name
    invalid_response.tool_calls[0].function.name = "test_tool"
    invalid_response.reasoning = None
    invalid_response.raw_response = {"role": "assistant", "content": "invalid"}
    invalid_response.prompt_tokens = 100
    invalid_response.completion_tokens = 50

    # Mock valid response after retry
    valid_response = Mock()
    valid_response.tool_calls = [
        Mock(function=Mock(arguments=json.dumps({"correct_param": "value"})))
    ]
    # Ensure function name is a real string, not a Mock name
    valid_response.tool_calls[0].function.name = "test_tool"
    valid_response.reasoning = None
    valid_response.raw_response = {"role": "assistant", "content": "valid"}
    valid_response.prompt_tokens = 100
    valid_response.completion_tokens = 50

    call_count = 0

    async def mock_llm_call(**kwargs):
        nonlocal call_count
        # Capture conversation state
        conversation_snapshots.append(kwargs.get("prompt", []).copy())
        call_count += 1
        if call_count == 1:
            return invalid_response
        else:
            return valid_response

    # Mock the LLM call
    with patch(
        "backend.blocks.smart_decision_maker.llm.llm_call", new_callable=AsyncMock
    ) as mock_llm:
        mock_llm.side_effect = mock_llm_call

        # Mock the function signature creation
        with patch.object(
            block, "_create_function_signature", new_callable=AsyncMock
        ) as mock_sig:
            mock_sig.return_value = [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "correct_param": {
                                    "type": "string",
                                    "description": "The correct parameter",
                                }
                            },
                            "required": ["correct_param"],
                        },
                    },
                }
            ]

            # Create input data
            from backend.blocks import llm

            input_data = block.input_schema(
                prompt="Test prompt",
                credentials=llm.TEST_CREDENTIALS_INPUT,
                model=llm.LlmModel.GPT4O,
                retry=3,  # Allow retries
            )

            # Run the block
            outputs = {}
            async for output_name, output_value in block.run(
                input_data,
                credentials=llm.TEST_CREDENTIALS,
                graph_id="test_graph",
                node_id="test_node",
                graph_exec_id="test_exec",
                node_exec_id="test_node_exec",
                user_id="test_user",
            ):
                outputs[output_name] = output_value

            # Verify we had 2 LLM calls (initial + retry)
            assert call_count == 2

            # Check the final conversation output
            final_conversation = outputs.get("conversations", [])

            # The final conversation should NOT contain the validation error message
            error_messages = [
                msg
                for msg in final_conversation
                if msg.get("role") == "user"
                and "parameter errors" in msg.get("content", "")
            ]
            assert (
                len(error_messages) == 0
            ), "Validation error leaked into final conversation"

            # The final conversation should only have the successful response
            assert final_conversation[-1]["content"] == "valid"
