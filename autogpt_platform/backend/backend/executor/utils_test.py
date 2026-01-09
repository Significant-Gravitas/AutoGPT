from typing import cast

import pytest
from pytest_mock import MockerFixture

from backend.data.dynamic_fields import merge_execution_input, parse_execution_output
from backend.util.mock import MockObject


def test_parse_execution_output():
    # Test case for basic output
    output = ("result", "value")
    assert parse_execution_output(output, "result") == "value"

    # Test case for list output
    output = ("result", [10, 20, 30])
    assert parse_execution_output(output, "result_$_1") == 20

    # Test case for dict output
    output = ("result", {"key1": "value1", "key2": "value2"})
    assert parse_execution_output(output, "result_#_key1") == "value1"

    # Test case for object output
    class Sample:
        def __init__(self):
            self.attr1 = "value1"
            self.attr2 = "value2"

    output = ("result", Sample())
    assert parse_execution_output(output, "result_@_attr1") == "value1"

    # Test case for nested list output
    output = ("result", [[1, 2], [3, 4]])
    assert parse_execution_output(output, "result_$_0_$_1") == 2
    assert parse_execution_output(output, "result_$_1_$_0") == 3

    # Test case for list containing dict
    output = ("result", [{"key1": "value1"}, {"key2": "value2"}])
    assert parse_execution_output(output, "result_$_0_#_key1") == "value1"
    assert parse_execution_output(output, "result_$_1_#_key2") == "value2"

    # Test case for dict containing list
    output = ("result", {"key1": [1, 2], "key2": [3, 4]})
    assert parse_execution_output(output, "result_#_key1_$_1") == 2
    assert parse_execution_output(output, "result_#_key2_$_0") == 3

    # Test case for complex nested structure
    class NestedSample:
        def __init__(self):
            self.attr1 = [1, 2]
            self.attr2 = {"key": "value"}

    output = ("result", [NestedSample(), {"key": [1, 2]}])
    assert parse_execution_output(output, "result_$_0_@_attr1_$_1") == 2
    assert parse_execution_output(output, "result_$_0_@_attr2_#_key") == "value"
    assert parse_execution_output(output, "result_$_1_#_key_$_0") == 1

    # Test case for non-existent paths
    output = ("result", [1, 2, 3])
    assert parse_execution_output(output, "result_$_5") is None
    assert parse_execution_output(output, "result_#_key") is None
    assert parse_execution_output(output, "result_@_attr") is None
    assert parse_execution_output(output, "wrong_name") is None

    # Test cases for delimiter processing order
    # Test case 1: List -> Dict -> List
    output = ("result", [[{"key": [1, 2]}], [3, 4]])
    assert parse_execution_output(output, "result_$_0_$_0_#_key_$_1") == 2

    # Test case 2: Dict -> List -> Object
    class NestedObj:
        def __init__(self):
            self.value = "nested"

    output = ("result", {"key": [NestedObj(), 2]})
    assert parse_execution_output(output, "result_#_key_$_0_@_value") == "nested"

    # Test case 3: Object -> List -> Dict
    class ParentObj:
        def __init__(self):
            self.items = [{"nested": "value"}]

    output = ("result", ParentObj())
    assert parse_execution_output(output, "result_@_items_$_0_#_nested") == "value"

    # Test case 4: Complex nested structure with all types
    class ComplexObj:
        def __init__(self):
            self.data = [{"items": [{"value": "deep"}]}]

    output = ("result", {"key": [ComplexObj()]})
    assert (
        parse_execution_output(
            output, "result_#_key_$_0_@_data_$_0_#_items_$_0_#_value"
        )
        == "deep"
    )

    # Test case 5: Invalid paths that should return None
    output = ("result", [{"key": [1, 2]}])
    assert parse_execution_output(output, "result_$_0_#_wrong_key") is None
    assert parse_execution_output(output, "result_$_0_#_key_$_5") is None
    assert parse_execution_output(output, "result_$_0_@_attr") is None

    # Test case 6: Mixed delimiter types in wrong order
    output = ("result", {"key": [1, 2]})
    assert (
        parse_execution_output(output, "result_#_key_$_1_@_attr") is None
    )  # Should fail at @_attr
    assert (
        parse_execution_output(output, "result_@_attr_$_0_#_key") is None
    )  # Should fail at @_attr

    # Test case 7: Tool pin routing with matching node ID and pin name
    output = ("tools_^_node123_~_query", "search term")
    assert parse_execution_output(output, "tools", "node123", "query") == "search term"

    # Test case 8: Tool pin routing with node ID mismatch
    output = ("tools_^_node123_~_query", "search term")
    assert parse_execution_output(output, "tools", "node456", "query") is None

    # Test case 9: Tool pin routing with pin name mismatch
    output = ("tools_^_node123_~_query", "search term")
    assert parse_execution_output(output, "tools", "node123", "different_pin") is None

    # Test case 10: Tool pin routing with complex field names
    output = ("tools_^_node789_~_nested_field", {"key": "value"})
    result = parse_execution_output(output, "tools", "node789", "nested_field")
    assert result == {"key": "value"}

    # Test case 11: Tool pin routing missing required parameters should raise error
    output = ("tools_^_node123_~_query", "search term")
    try:
        parse_execution_output(output, "tools", "node123")  # Missing sink_pin_name
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "must be provided for tool pin routing" in str(e)

    # Test case 12: Non-tool pin with similar pattern should use normal logic
    output = ("tools_^_node123_~_query", "search term")
    assert parse_execution_output(output, "different_name", "node123", "query") is None


def test_merge_execution_input():
    # Test case for basic list extraction
    data = {
        "list_$_0": "a",
        "list_$_1": "b",
    }
    result = merge_execution_input(data)
    assert "list" in result
    assert result["list"] == ["a", "b"]

    # Test case for basic dict extraction
    data = {
        "dict_#_key1": "value1",
        "dict_#_key2": "value2",
    }
    result = merge_execution_input(data)
    assert "dict" in result
    assert result["dict"] == {"key1": "value1", "key2": "value2"}

    # Test case for object extraction
    class Sample:
        def __init__(self):
            self.attr1 = None
            self.attr2 = None

    data = {
        "object_@_attr1": "value1",
        "object_@_attr2": "value2",
    }
    result = merge_execution_input(data)
    assert "object" in result
    assert isinstance(result["object"], MockObject)
    assert result["object"].attr1 == "value1"
    assert result["object"].attr2 == "value2"

    # Test case for nested list extraction
    data = {
        "nested_list_$_0_$_0": "a",
        "nested_list_$_0_$_1": "b",
        "nested_list_$_1_$_0": "c",
    }
    result = merge_execution_input(data)
    assert "nested_list" in result
    assert result["nested_list"] == [["a", "b"], ["c"]]

    # Test case for list containing dict
    data = {
        "list_with_dict_$_0_#_key1": "value1",
        "list_with_dict_$_0_#_key2": "value2",
        "list_with_dict_$_1_#_key3": "value3",
    }
    result = merge_execution_input(data)
    assert "list_with_dict" in result
    assert result["list_with_dict"] == [
        {"key1": "value1", "key2": "value2"},
        {"key3": "value3"},
    ]

    # Test case for dict containing list
    data = {
        "dict_with_list_#_key1_$_0": "value1",
        "dict_with_list_#_key1_$_1": "value2",
        "dict_with_list_#_key2_$_0": "value3",
    }
    result = merge_execution_input(data)
    assert "dict_with_list" in result
    assert result["dict_with_list"] == {
        "key1": ["value1", "value2"],
        "key2": ["value3"],
    }

    # Test case for complex nested structure
    data = {
        "complex_$_0_#_key1_$_0": "value1",
        "complex_$_0_#_key1_$_1": "value2",
        "complex_$_0_#_key2_@_attr1": "value3",
        "complex_$_1_#_key3_$_0": "value4",
    }
    result = merge_execution_input(data)
    assert "complex" in result
    assert result["complex"][0]["key1"] == ["value1", "value2"]
    assert isinstance(result["complex"][0]["key2"], MockObject)
    assert result["complex"][0]["key2"].attr1 == "value3"
    assert result["complex"][1]["key3"] == ["value4"]

    # Test case for invalid list index
    data = {"list_$_invalid": "value"}
    with pytest.raises(ValueError, match="index must be an integer"):
        merge_execution_input(data)

    # Test cases for delimiter ordering
    # Test case 1: List -> Dict -> List
    data = {
        "nested_$_0_#_key_$_0": "value1",
        "nested_$_0_#_key_$_1": "value2",
    }
    result = merge_execution_input(data)
    assert "nested" in result
    assert result["nested"][0]["key"] == ["value1", "value2"]

    # Test case 2: Dict -> List -> Object
    data = {
        "nested_#_key_$_0_@_attr": "value1",
        "nested_#_key_$_1_@_attr": "value2",
    }
    result = merge_execution_input(data)
    assert "nested" in result
    assert isinstance(result["nested"]["key"][0], MockObject)
    assert result["nested"]["key"][0].attr == "value1"
    assert result["nested"]["key"][1].attr == "value2"

    # Test case 3: Object -> List -> Dict
    data = {
        "nested_@_items_$_0_#_key": "value1",
        "nested_@_items_$_1_#_key": "value2",
    }
    result = merge_execution_input(data)
    assert "nested" in result
    nested = result["nested"]
    assert isinstance(nested, MockObject)
    items = nested.items
    assert isinstance(items, list)
    assert items[0]["key"] == "value1"
    assert items[1]["key"] == "value2"

    # Test case 4: Complex nested structure with all types
    data = {
        "deep_#_key_$_0_@_data_$_0_#_items_$_0_#_value": "deep_value",
        "deep_#_key_$_0_@_data_$_1_#_items_$_0_#_value": "another_value",
    }
    result = merge_execution_input(data)
    assert "deep" in result
    deep_key = result["deep"]["key"][0]
    assert deep_key is not None
    data0 = getattr(deep_key, "data", None)
    assert isinstance(data0, list)
    # Check items0
    items0 = None
    if len(data0) > 0 and isinstance(data0[0], dict) and "items" in data0[0]:
        items0 = data0[0]["items"]
    assert isinstance(items0, list)
    items0 = cast(list, items0)
    assert len(items0) > 0
    assert isinstance(items0[0], dict)
    assert items0[0]["value"] == "deep_value"  # type: ignore
    # Check items1
    items1 = None
    if len(data0) > 1 and isinstance(data0[1], dict) and "items" in data0[1]:
        items1 = data0[1]["items"]
    assert isinstance(items1, list)
    items1 = cast(list, items1)
    assert len(items1) > 0
    assert isinstance(items1[0], dict)
    assert items1[0]["value"] == "another_value"  # type: ignore

    # Test case 5: Mixed delimiter types in different orders
    # the last one should replace the type
    data = {
        "mixed_$_0_#_key_@_attr": "value1",  # List -> Dict -> Object
        "mixed_#_key_$_0_@_attr": "value2",  # Dict -> List -> Object
        "mixed_@_attr_$_0_#_key": "value3",  # Object -> List -> Dict
    }
    result = merge_execution_input(data)
    assert "mixed" in result
    assert result["mixed"].attr[0]["key"] == "value3"


@pytest.mark.asyncio
async def test_add_graph_execution_is_repeatable(mocker: MockerFixture):
    """
    Verify that calling the function with its own output creates the same execution again.
    """
    from backend.data.execution import GraphExecutionWithNodes
    from backend.data.model import CredentialsMetaInput
    from backend.executor.utils import add_graph_execution
    from backend.integrations.providers import ProviderName

    # Mock data
    graph_id = "test-graph-id"
    user_id = "test-user-id"
    inputs = {"test_input": "test_value"}
    preset_id = "test-preset-id"
    graph_version = 1
    graph_credentials_inputs = {
        "cred_key": CredentialsMetaInput(
            id="cred-id", provider=ProviderName("test_provider"), type="oauth2"
        )
    }
    nodes_input_masks = {"node1": {"input1": "masked_value"}}

    # Mock the graph object returned by validate_and_construct_node_execution_input
    mock_graph = mocker.MagicMock()
    mock_graph.version = graph_version

    # Mock the starting nodes input and compiled nodes input masks
    starting_nodes_input = [
        ("node1", {"input1": "value1"}),
        ("node2", {"input1": "value2"}),
    ]
    compiled_nodes_input_masks = {"node1": {"input1": "compiled_mask"}}

    # Mock the graph execution object
    mock_graph_exec = mocker.MagicMock(spec=GraphExecutionWithNodes)
    mock_graph_exec.id = "execution-id-123"
    mock_graph_exec.node_executions = []  # Add this to avoid AttributeError
    mock_graph_exec.to_graph_execution_entry.return_value = mocker.MagicMock()

    # Mock the queue and event bus
    mock_queue = mocker.AsyncMock()
    mock_event_bus = mocker.MagicMock()
    mock_event_bus.publish = mocker.AsyncMock()

    # Setup mocks
    mock_validate = mocker.patch(
        "backend.executor.utils.validate_and_construct_node_execution_input"
    )
    mock_edb = mocker.patch("backend.executor.utils.execution_db")
    mock_prisma = mocker.patch("backend.executor.utils.prisma")
    mock_udb = mocker.patch("backend.executor.utils.user_db")
    mock_gdb = mocker.patch("backend.executor.utils.graph_db")
    mock_get_queue = mocker.patch("backend.executor.utils.get_async_execution_queue")
    mock_get_event_bus = mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus"
    )

    # Setup mock returns
    # The function returns (graph, starting_nodes_input, compiled_nodes_input_masks, nodes_to_skip)
    nodes_to_skip: set[str] = set()
    mock_validate.return_value = (
        mock_graph,
        starting_nodes_input,
        compiled_nodes_input_masks,
        nodes_to_skip,
    )
    mock_prisma.is_connected.return_value = True
    mock_edb.create_graph_execution = mocker.AsyncMock(return_value=mock_graph_exec)
    mock_edb.update_graph_execution_stats = mocker.AsyncMock(
        return_value=mock_graph_exec
    )
    mock_edb.update_node_execution_status_batch = mocker.AsyncMock()
    # Mock user and settings data
    mock_user = mocker.MagicMock()
    mock_user.timezone = "UTC"
    mock_settings = mocker.MagicMock()
    mock_settings.human_in_the_loop_safe_mode = True

    mock_udb.get_user_by_id = mocker.AsyncMock(return_value=mock_user)
    mock_gdb.get_graph_settings = mocker.AsyncMock(return_value=mock_settings)
    mock_get_queue.return_value = mock_queue
    mock_get_event_bus.return_value = mock_event_bus

    # Call the function - first execution
    result1 = await add_graph_execution(
        graph_id=graph_id,
        user_id=user_id,
        inputs=inputs,
        preset_id=preset_id,
        graph_version=graph_version,
        graph_credentials_inputs=graph_credentials_inputs,
        nodes_input_masks=nodes_input_masks,
    )

    # Store the parameters used in the first call to create_graph_execution
    first_call_kwargs = mock_edb.create_graph_execution.call_args[1]

    # Verify the create_graph_execution was called with correct parameters
    mock_edb.create_graph_execution.assert_called_once_with(
        user_id=user_id,
        graph_id=graph_id,
        graph_version=mock_graph.version,
        inputs=inputs,
        credential_inputs=graph_credentials_inputs,
        nodes_input_masks=nodes_input_masks,
        starting_nodes_input=starting_nodes_input,
        preset_id=preset_id,
        parent_graph_exec_id=None,
    )

    # Set up the graph execution mock to have properties we can extract
    mock_graph_exec.graph_id = graph_id
    mock_graph_exec.user_id = user_id
    mock_graph_exec.graph_version = graph_version
    mock_graph_exec.inputs = inputs
    mock_graph_exec.credential_inputs = graph_credentials_inputs
    mock_graph_exec.nodes_input_masks = nodes_input_masks
    mock_graph_exec.preset_id = preset_id

    # Create a second mock execution for the sanity check
    mock_graph_exec_2 = mocker.MagicMock(spec=GraphExecutionWithNodes)
    mock_graph_exec_2.id = "execution-id-456"
    mock_graph_exec_2.to_graph_execution_entry.return_value = mocker.MagicMock()

    # Reset mocks and set up for second call
    mock_edb.create_graph_execution.reset_mock()
    mock_edb.create_graph_execution.return_value = mock_graph_exec_2
    mock_validate.reset_mock()

    # Sanity check: call add_graph_execution with properties from first result
    # This should create the same execution parameters
    result2 = await add_graph_execution(
        graph_id=mock_graph_exec.graph_id,
        user_id=mock_graph_exec.user_id,
        inputs=mock_graph_exec.inputs,
        preset_id=mock_graph_exec.preset_id,
        graph_version=mock_graph_exec.graph_version,
        graph_credentials_inputs=mock_graph_exec.credential_inputs,
        nodes_input_masks=mock_graph_exec.nodes_input_masks,
    )

    # Verify that create_graph_execution was called with identical parameters
    second_call_kwargs = mock_edb.create_graph_execution.call_args[1]

    # The sanity check: both calls should use identical parameters
    assert first_call_kwargs == second_call_kwargs

    # Both executions should succeed (though they create different objects)
    assert result1 == mock_graph_exec
    assert result2 == mock_graph_exec_2


# ============================================================================
# Tests for Optional Credentials Feature
# ============================================================================


@pytest.mark.asyncio
async def test_validate_node_input_credentials_returns_nodes_to_skip(
    mocker: MockerFixture,
):
    """
    Test that _validate_node_input_credentials returns nodes_to_skip set
    for nodes with credentials_optional=True and missing credentials.
    """
    from backend.executor.utils import _validate_node_input_credentials

    # Create a mock node with credentials_optional=True
    mock_node = mocker.MagicMock()
    mock_node.id = "node-with-optional-creds"
    mock_node.credentials_optional = True
    mock_node.input_default = {}  # No credentials configured

    # Create a mock block with credentials field
    mock_block = mocker.MagicMock()
    mock_credentials_field_type = mocker.MagicMock()
    mock_block.input_schema.get_credentials_fields.return_value = {
        "credentials": mock_credentials_field_type
    }
    mock_node.block = mock_block

    # Create mock graph
    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]

    # Call the function
    errors, nodes_to_skip = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="test-user-id",
        nodes_input_masks=None,
    )

    # Node should be in nodes_to_skip, not in errors
    assert mock_node.id in nodes_to_skip
    assert mock_node.id not in errors


@pytest.mark.asyncio
async def test_validate_node_input_credentials_required_missing_creds_error(
    mocker: MockerFixture,
):
    """
    Test that _validate_node_input_credentials returns errors
    for nodes with credentials_optional=False and missing credentials.
    """
    from backend.executor.utils import _validate_node_input_credentials

    # Create a mock node with credentials_optional=False (required)
    mock_node = mocker.MagicMock()
    mock_node.id = "node-with-required-creds"
    mock_node.credentials_optional = False
    mock_node.input_default = {}  # No credentials configured

    # Create a mock block with credentials field
    mock_block = mocker.MagicMock()
    mock_credentials_field_type = mocker.MagicMock()
    mock_block.input_schema.get_credentials_fields.return_value = {
        "credentials": mock_credentials_field_type
    }
    mock_node.block = mock_block

    # Create mock graph
    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]

    # Call the function
    errors, nodes_to_skip = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="test-user-id",
        nodes_input_masks=None,
    )

    # Node should be in errors, not in nodes_to_skip
    assert mock_node.id in errors
    assert "credentials" in errors[mock_node.id]
    assert "required" in errors[mock_node.id]["credentials"].lower()
    assert mock_node.id not in nodes_to_skip


@pytest.mark.asyncio
async def test_validate_graph_with_credentials_returns_nodes_to_skip(
    mocker: MockerFixture,
):
    """
    Test that validate_graph_with_credentials returns nodes_to_skip set
    from _validate_node_input_credentials.
    """
    from backend.executor.utils import validate_graph_with_credentials

    # Mock _validate_node_input_credentials to return specific values
    mock_validate = mocker.patch(
        "backend.executor.utils._validate_node_input_credentials"
    )
    expected_errors = {"node1": {"field": "error"}}
    expected_nodes_to_skip = {"node2", "node3"}
    mock_validate.return_value = (expected_errors, expected_nodes_to_skip)

    # Mock GraphModel with validate_graph_get_errors method
    mock_graph = mocker.MagicMock()
    mock_graph.validate_graph_get_errors.return_value = {}

    # Call the function
    errors, nodes_to_skip = await validate_graph_with_credentials(
        graph=mock_graph,
        user_id="test-user-id",
        nodes_input_masks=None,
    )

    # Verify nodes_to_skip is passed through
    assert nodes_to_skip == expected_nodes_to_skip
    assert "node1" in errors


@pytest.mark.asyncio
async def test_add_graph_execution_with_nodes_to_skip(mocker: MockerFixture):
    """
    Test that add_graph_execution properly passes nodes_to_skip
    to the graph execution entry.
    """
    from backend.data.execution import GraphExecutionWithNodes
    from backend.executor.utils import add_graph_execution

    # Mock data
    graph_id = "test-graph-id"
    user_id = "test-user-id"
    inputs = {"test_input": "test_value"}
    graph_version = 1

    # Mock the graph object
    mock_graph = mocker.MagicMock()
    mock_graph.version = graph_version

    # Starting nodes and masks
    starting_nodes_input = [("node1", {"input1": "value1"})]
    compiled_nodes_input_masks = {}
    nodes_to_skip = {"skipped-node-1", "skipped-node-2"}

    # Mock the graph execution object
    mock_graph_exec = mocker.MagicMock(spec=GraphExecutionWithNodes)
    mock_graph_exec.id = "execution-id-123"
    mock_graph_exec.node_executions = []

    # Track what's passed to to_graph_execution_entry
    captured_kwargs = {}

    def capture_to_entry(**kwargs):
        captured_kwargs.update(kwargs)
        return mocker.MagicMock()

    mock_graph_exec.to_graph_execution_entry.side_effect = capture_to_entry

    # Setup mocks
    mock_validate = mocker.patch(
        "backend.executor.utils.validate_and_construct_node_execution_input"
    )
    mock_edb = mocker.patch("backend.executor.utils.execution_db")
    mock_prisma = mocker.patch("backend.executor.utils.prisma")
    mock_udb = mocker.patch("backend.executor.utils.user_db")
    mock_gdb = mocker.patch("backend.executor.utils.graph_db")
    mock_get_queue = mocker.patch("backend.executor.utils.get_async_execution_queue")
    mock_get_event_bus = mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus"
    )

    # Setup returns - include nodes_to_skip in the tuple
    mock_validate.return_value = (
        mock_graph,
        starting_nodes_input,
        compiled_nodes_input_masks,
        nodes_to_skip,  # This should be passed through
    )
    mock_prisma.is_connected.return_value = True
    mock_edb.create_graph_execution = mocker.AsyncMock(return_value=mock_graph_exec)
    mock_edb.update_graph_execution_stats = mocker.AsyncMock(
        return_value=mock_graph_exec
    )
    mock_edb.update_node_execution_status_batch = mocker.AsyncMock()

    mock_user = mocker.MagicMock()
    mock_user.timezone = "UTC"
    mock_settings = mocker.MagicMock()
    mock_settings.human_in_the_loop_safe_mode = True

    mock_udb.get_user_by_id = mocker.AsyncMock(return_value=mock_user)
    mock_gdb.get_graph_settings = mocker.AsyncMock(return_value=mock_settings)
    mock_get_queue.return_value = mocker.AsyncMock()
    mock_get_event_bus.return_value = mocker.MagicMock(publish=mocker.AsyncMock())

    # Call the function
    await add_graph_execution(
        graph_id=graph_id,
        user_id=user_id,
        inputs=inputs,
        graph_version=graph_version,
    )

    # Verify nodes_to_skip was passed to to_graph_execution_entry
    assert "nodes_to_skip" in captured_kwargs
    assert captured_kwargs["nodes_to_skip"] == nodes_to_skip
