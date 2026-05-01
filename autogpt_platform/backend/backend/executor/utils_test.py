from datetime import datetime, timezone
from typing import cast

import pytest
from pytest_mock import MockerFixture

from backend.data.dynamic_fields import merge_execution_input, parse_execution_output
from backend.data.execution import ExecutionStatus, GraphExecutionWithNodes
from backend.data.model import User
from backend.executor.utils import (
    CRED_ERR_INVALID_PREFIX,
    CRED_ERR_INVALID_TYPE_MISMATCH,
    CRED_ERR_NOT_AVAILABLE_PREFIX,
    CRED_ERR_REQUIRED,
    CRED_ERR_UNKNOWN_PREFIX,
    add_graph_execution,
    is_credential_validation_error_message,
)
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
    mock_graph_exec.status = ExecutionStatus.QUEUED  # Required for race condition check
    mock_graph_exec.graph_version = graph_version
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
    mock_wdb = mocker.patch("backend.executor.utils.workspace_db")
    mock_workspace = mocker.MagicMock()
    mock_workspace.id = "test-workspace-id"
    mock_wdb.get_or_create_workspace = mocker.AsyncMock(return_value=mock_workspace)

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
    mock_settings.sensitive_action_safe_mode = False

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
        is_dry_run=False,
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
    mock_graph_exec_2.node_executions = []
    mock_graph_exec_2.status = ExecutionStatus.QUEUED
    mock_graph_exec_2.graph_version = graph_version
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
# Regression test: RPC layer returns typed User model, not raw dict
# ============================================================================


@pytest.mark.asyncio
async def test_add_graph_execution_via_rpc_returns_typed_user(
    mocker: MockerFixture,
):
    """
    Regression test: `add_graph_execution` accesses `user.timezone` on the User
    returned by `get_user_by_id`. This test verifies the downstream code path
    completes without AttributeError when `get_user_by_id` returns a proper typed
    User model. Note: the mock returns a User directly — _get_return deserialization
    is not exercised here; see TestGetReturn in util/service_test.py for that.
    """
    graph_id = "test-graph-id"
    user_id = "test-user-id"

    mock_graph = mocker.MagicMock()
    mock_graph.version = 1

    mock_graph_exec = mocker.MagicMock(spec=GraphExecutionWithNodes)
    mock_graph_exec.id = "exec-id-rpc"
    mock_graph_exec.node_executions = []
    mock_graph_exec.status = ExecutionStatus.QUEUED
    mock_graph_exec.graph_version = 1
    mock_graph_exec.to_graph_execution_entry.return_value = mocker.MagicMock()

    mock_queue = mocker.AsyncMock()
    mock_event_bus = mocker.MagicMock()
    mock_event_bus.publish = mocker.AsyncMock()

    mock_validate = mocker.patch(
        "backend.executor.utils.validate_and_construct_node_execution_input"
    )
    mock_validate.return_value = (mock_graph, [], {}, set())

    mock_prisma = mocker.patch("backend.executor.utils.prisma")
    mock_prisma.is_connected.return_value = (
        False  # prisma not connected: uses RPC path instead
    )

    # The RPC layer (_get_return) deserializes JSON dicts into typed Pydantic models.
    # The mock simulates what add_graph_execution receives after that deserialization:
    # a proper User model, not a raw dict.
    mock_user = User(
        id=user_id,
        email="test@example.com",
        name=None,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        stripe_customer_id=None,
        top_up_config=None,
        timezone="UTC",
    )

    mock_db_client = mocker.MagicMock()
    mock_db_client.get_user_by_id = mocker.AsyncMock(return_value=mock_user)
    mock_db_client.get_graph_settings = mocker.AsyncMock(
        return_value=mocker.MagicMock(
            human_in_the_loop_safe_mode=False, sensitive_action_safe_mode=False
        )
    )
    mock_db_client.create_graph_execution = mocker.AsyncMock(
        return_value=mock_graph_exec
    )
    mock_db_client.update_graph_execution_stats = mocker.AsyncMock(
        return_value=mock_graph_exec
    )
    mock_db_client.update_node_execution_status_batch = mocker.AsyncMock()
    mock_workspace = mocker.MagicMock()
    mock_workspace.id = "ws-id"
    mock_db_client.get_or_create_workspace = mocker.AsyncMock(
        return_value=mock_workspace
    )
    mock_db_client.increment_onboarding_runs = mocker.AsyncMock()

    mocker.patch(
        "backend.executor.utils.get_database_manager_async_client",
        return_value=mock_db_client,
    )
    mocker.patch(
        "backend.executor.utils.get_async_execution_queue", return_value=mock_queue
    )
    mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus",
        return_value=mock_event_bus,
    )

    # Must not raise AttributeError: 'dict' object has no attribute 'timezone'
    result = await add_graph_execution(
        graph_id=graph_id,
        user_id=user_id,
    )
    assert result == mock_graph_exec


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
    mock_block.input_schema.get_required_fields.return_value = {"credentials"}
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

    # Optional-creds + missing => skip the node (don't run it with None creds)
    # and don't record an error. This contract is relied on by the executor
    # which would otherwise try to run a block whose credentials never
    # arrived.
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
    mock_block.input_schema.get_required_fields.return_value = {"credentials"}
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
    mock_graph_exec.status = ExecutionStatus.QUEUED  # Required for race condition check
    mock_graph_exec.graph_version = graph_version

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
    mock_wdb = mocker.patch("backend.executor.utils.workspace_db")
    mock_workspace = mocker.MagicMock()
    mock_workspace.id = "test-workspace-id"
    mock_wdb.get_or_create_workspace = mocker.AsyncMock(return_value=mock_workspace)

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
    mock_settings.sensitive_action_safe_mode = False

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

    # Verify workspace_id is set in the execution context
    assert "execution_context" in captured_kwargs
    assert captured_kwargs["execution_context"].workspace_id == "test-workspace-id"


@pytest.mark.asyncio
async def test_stop_graph_execution_in_review_status_cancels_pending_reviews(
    mocker: MockerFixture,
):
    """Test that stopping an execution in REVIEW status cancels pending reviews."""
    from backend.data.execution import ExecutionStatus, GraphExecutionMeta
    from backend.executor.utils import stop_graph_execution

    user_id = "test-user"
    graph_exec_id = "test-exec-123"

    # Mock graph execution in REVIEW status
    mock_graph_exec = mocker.MagicMock(spec=GraphExecutionMeta)
    mock_graph_exec.id = graph_exec_id
    mock_graph_exec.status = ExecutionStatus.REVIEW

    # Mock dependencies
    mock_get_queue = mocker.patch("backend.executor.utils.get_async_execution_queue")
    mock_queue_client = mocker.AsyncMock()
    mock_get_queue.return_value = mock_queue_client

    mock_prisma = mocker.patch("backend.executor.utils.prisma")
    mock_prisma.is_connected.return_value = True

    mock_human_review_db = mocker.patch("backend.executor.utils.human_review_db")
    mock_human_review_db.cancel_pending_reviews_for_execution = mocker.AsyncMock(
        return_value=2  # 2 reviews cancelled
    )

    mock_execution_db = mocker.patch("backend.executor.utils.execution_db")
    mock_execution_db.get_graph_execution_meta = mocker.AsyncMock(
        return_value=mock_graph_exec
    )
    mock_execution_db.update_graph_execution_stats = mocker.AsyncMock()

    mock_get_event_bus = mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus"
    )
    mock_event_bus = mocker.MagicMock()
    mock_event_bus.publish = mocker.AsyncMock()
    mock_get_event_bus.return_value = mock_event_bus

    mock_get_child_executions = mocker.patch(
        "backend.executor.utils._get_child_executions"
    )
    mock_get_child_executions.return_value = []  # No children

    # Call stop_graph_execution with timeout to allow status check
    await stop_graph_execution(
        user_id=user_id,
        graph_exec_id=graph_exec_id,
        wait_timeout=1.0,  # Wait to allow status check
        cascade=True,
    )

    # Verify pending reviews were cancelled
    mock_human_review_db.cancel_pending_reviews_for_execution.assert_called_once_with(
        graph_exec_id, user_id
    )

    # Verify execution status was updated to TERMINATED
    mock_execution_db.update_graph_execution_stats.assert_called_once()
    call_kwargs = mock_execution_db.update_graph_execution_stats.call_args[1]
    assert call_kwargs["graph_exec_id"] == graph_exec_id
    assert call_kwargs["status"] == ExecutionStatus.TERMINATED


@pytest.mark.asyncio
async def test_stop_graph_execution_with_database_manager_when_prisma_disconnected(
    mocker: MockerFixture,
):
    """Test that stop uses database manager when Prisma is not connected."""
    from backend.data.execution import ExecutionStatus, GraphExecutionMeta
    from backend.executor.utils import stop_graph_execution

    user_id = "test-user"
    graph_exec_id = "test-exec-456"

    # Mock graph execution in REVIEW status
    mock_graph_exec = mocker.MagicMock(spec=GraphExecutionMeta)
    mock_graph_exec.id = graph_exec_id
    mock_graph_exec.status = ExecutionStatus.REVIEW

    # Mock dependencies
    mock_get_queue = mocker.patch("backend.executor.utils.get_async_execution_queue")
    mock_queue_client = mocker.AsyncMock()
    mock_get_queue.return_value = mock_queue_client

    # Prisma is NOT connected
    mock_prisma = mocker.patch("backend.executor.utils.prisma")
    mock_prisma.is_connected.return_value = False

    # Mock database manager client
    mock_get_db_manager = mocker.patch(
        "backend.executor.utils.get_database_manager_async_client"
    )
    mock_db_manager = mocker.AsyncMock()
    mock_db_manager.get_graph_execution_meta = mocker.AsyncMock(
        return_value=mock_graph_exec
    )
    mock_db_manager.cancel_pending_reviews_for_execution = mocker.AsyncMock(
        return_value=3  # 3 reviews cancelled
    )
    mock_db_manager.update_graph_execution_stats = mocker.AsyncMock()
    mock_get_db_manager.return_value = mock_db_manager

    mock_get_event_bus = mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus"
    )
    mock_event_bus = mocker.MagicMock()
    mock_event_bus.publish = mocker.AsyncMock()
    mock_get_event_bus.return_value = mock_event_bus

    mock_get_child_executions = mocker.patch(
        "backend.executor.utils._get_child_executions"
    )
    mock_get_child_executions.return_value = []  # No children

    # Call stop_graph_execution with timeout
    await stop_graph_execution(
        user_id=user_id,
        graph_exec_id=graph_exec_id,
        wait_timeout=1.0,
        cascade=True,
    )

    # Verify database manager was used for cancel_pending_reviews
    mock_db_manager.cancel_pending_reviews_for_execution.assert_called_once_with(
        graph_exec_id, user_id
    )

    # Verify execution status was updated via database manager
    mock_db_manager.update_graph_execution_stats.assert_called_once()


@pytest.mark.asyncio
async def test_stop_graph_execution_cascades_to_child_with_reviews(
    mocker: MockerFixture,
):
    """Test that stopping parent execution cascades to children and cancels their reviews."""
    from backend.data.execution import ExecutionStatus, GraphExecutionMeta
    from backend.executor.utils import stop_graph_execution

    user_id = "test-user"
    parent_exec_id = "parent-exec"
    child_exec_id = "child-exec"

    # Mock parent execution in RUNNING status
    mock_parent_exec = mocker.MagicMock(spec=GraphExecutionMeta)
    mock_parent_exec.id = parent_exec_id
    mock_parent_exec.status = ExecutionStatus.RUNNING

    # Mock child execution in REVIEW status
    mock_child_exec = mocker.MagicMock(spec=GraphExecutionMeta)
    mock_child_exec.id = child_exec_id
    mock_child_exec.status = ExecutionStatus.REVIEW

    # Mock dependencies
    mock_get_queue = mocker.patch("backend.executor.utils.get_async_execution_queue")
    mock_queue_client = mocker.AsyncMock()
    mock_get_queue.return_value = mock_queue_client

    mock_prisma = mocker.patch("backend.executor.utils.prisma")
    mock_prisma.is_connected.return_value = True

    mock_human_review_db = mocker.patch("backend.executor.utils.human_review_db")
    mock_human_review_db.cancel_pending_reviews_for_execution = mocker.AsyncMock(
        return_value=1  # 1 child review cancelled
    )

    # Mock execution_db to return different status based on which execution is queried
    mock_execution_db = mocker.patch("backend.executor.utils.execution_db")

    # Track call count to simulate status transition
    call_count = {"count": 0}

    async def get_exec_meta_side_effect(execution_id, user_id):
        call_count["count"] += 1
        if execution_id == parent_exec_id:
            # After a few calls (child processing happens), transition parent to TERMINATED
            # This simulates the executor service processing the stop request
            if call_count["count"] > 3:
                mock_parent_exec.status = ExecutionStatus.TERMINATED
            return mock_parent_exec
        elif execution_id == child_exec_id:
            return mock_child_exec
        return None

    mock_execution_db.get_graph_execution_meta = mocker.AsyncMock(
        side_effect=get_exec_meta_side_effect
    )
    mock_execution_db.update_graph_execution_stats = mocker.AsyncMock()

    mock_get_event_bus = mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus"
    )
    mock_event_bus = mocker.MagicMock()
    mock_event_bus.publish = mocker.AsyncMock()
    mock_get_event_bus.return_value = mock_event_bus

    # Mock _get_child_executions to return the child
    mock_get_child_executions = mocker.patch(
        "backend.executor.utils._get_child_executions"
    )

    def get_children_side_effect(parent_id):
        if parent_id == parent_exec_id:
            return [mock_child_exec]
        return []

    mock_get_child_executions.side_effect = get_children_side_effect

    # Call stop_graph_execution on parent with cascade=True
    await stop_graph_execution(
        user_id=user_id,
        graph_exec_id=parent_exec_id,
        wait_timeout=1.0,
        cascade=True,
    )

    # Verify child reviews were cancelled
    mock_human_review_db.cancel_pending_reviews_for_execution.assert_called_once_with(
        child_exec_id, user_id
    )

    # Verify both parent and child status updates
    assert mock_execution_db.update_graph_execution_stats.call_count >= 1


# ---------------------------------------------------------------------------
# Credential validation error marker parity.
#
# ``is_credential_validation_error_message`` is shared by the executor
# dry-run path and the copilot credential-race fallback.  Adding a new
# credential error string in ``_validate_node_input_credentials`` without
# updating the matcher would silently regress the copilot UX to a plain
# text error.  These tests pin the contract:
#
# 1. Every ``CRED_ERR_*`` constant emitted by the raise sites is
#    recognised by the public matcher (including reasonable formatted
#    variants with runtime suffixes from ``f"{PREFIX} {e}"``).
# 2. The matcher is case-insensitive and unaffected by trailing detail.
# 3. Non-credential messages fall through.
# ---------------------------------------------------------------------------


def test_credential_error_markers_cover_all_raise_sites():
    """Each credential error string emitted by
    ``_validate_node_input_credentials`` must be recognised by
    ``is_credential_validation_error_message``. This guards against
    drift when a new credential error is introduced without updating
    the matcher."""
    # Exact-match raise sites
    assert is_credential_validation_error_message(CRED_ERR_REQUIRED)
    assert is_credential_validation_error_message(CRED_ERR_INVALID_TYPE_MISMATCH)

    # Prefix raise sites with typical runtime suffixes (matching the
    # f-strings inside ``_validate_node_input_credentials``)
    assert is_credential_validation_error_message(
        f"{CRED_ERR_INVALID_PREFIX} 1 validation error for ApiKeyCredentials"
    )
    assert is_credential_validation_error_message(
        f"{CRED_ERR_NOT_AVAILABLE_PREFIX} connection refused"
    )
    assert is_credential_validation_error_message(
        f"{CRED_ERR_UNKNOWN_PREFIX}abc-123-def"
    )


def test_credential_error_marker_matching_is_case_insensitive():
    """The matcher lowercases inputs before comparing — ensure that
    stays true for each marker so log-normalised copies still match."""
    assert is_credential_validation_error_message(CRED_ERR_REQUIRED.upper())
    assert is_credential_validation_error_message(CRED_ERR_REQUIRED.lower())
    assert is_credential_validation_error_message(
        f"{CRED_ERR_INVALID_PREFIX.upper()} BAD FIELD"
    )
    assert is_credential_validation_error_message(
        f"{CRED_ERR_UNKNOWN_PREFIX.upper()}XYZ"
    )


def test_non_credential_errors_are_not_matched():
    """Unrelated graph validation errors must not hit the credential
    branch — otherwise the copilot would hide structural errors behind
    the credential setup card."""
    assert not is_credential_validation_error_message("")
    assert not is_credential_validation_error_message(
        "missing input {'required_field'}"
    )
    assert not is_credential_validation_error_message("Input field 'url' is required")
    # A message that happens to contain "credentials" somewhere but
    # doesn't start with any known prefix must not match.
    assert not is_credential_validation_error_message(
        "Block configuration says credentials are fine"
    )


# ============================================================================
# Tests for auto_credentials validation in _validate_node_input_credentials
# (Fix 3: SECRT-1772 + Fix 4: Path 4)
# ============================================================================


@pytest.mark.asyncio
async def test_validate_node_input_credentials_auto_creds_valid(
    mocker: MockerFixture,
):
    """
    [SECRT-1772] When a node has auto_credentials with a valid _credentials_id
    that exists in the store, validation should pass without errors.
    """
    from backend.executor.utils import _validate_node_input_credentials

    mock_node = mocker.MagicMock()
    mock_node.id = "node-with-auto-creds"
    mock_node.credentials_optional = False
    mock_node.input_default = {
        "spreadsheet": {
            "_credentials_id": "valid-cred-id",
            "id": "file-123",
            "name": "test.xlsx",
        }
    }

    mock_block = mocker.MagicMock()
    # No regular credentials fields
    mock_block.input_schema.get_credentials_fields.return_value = {}
    # Has auto_credentials fields
    mock_block.input_schema.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        }
    }
    mock_node.block = mock_block

    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]

    # Mock the credentials store to return valid credentials
    mock_store = mocker.MagicMock()
    mock_creds = mocker.MagicMock()
    mock_creds.id = "valid-cred-id"
    mock_store.get_creds_by_id = mocker.AsyncMock(return_value=mock_creds)
    mocker.patch(
        "backend.executor.utils.get_integration_credentials_store",
        return_value=mock_store,
    )

    errors, nodes_to_skip = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="test-user",
        nodes_input_masks=None,
    )

    assert mock_node.id not in errors
    assert mock_node.id not in nodes_to_skip


@pytest.mark.asyncio
async def test_validate_node_input_credentials_auto_creds_missing(
    mocker: MockerFixture,
):
    """
    [SECRT-1772] When a node has auto_credentials with a _credentials_id
    that doesn't exist for the current user, validation should report an error.
    """
    from backend.executor.utils import _validate_node_input_credentials

    mock_node = mocker.MagicMock()
    mock_node.id = "node-with-bad-auto-creds"
    mock_node.credentials_optional = False
    mock_node.input_default = {
        "spreadsheet": {
            "_credentials_id": "other-users-cred-id",
            "id": "file-123",
            "name": "test.xlsx",
        }
    }

    mock_block = mocker.MagicMock()
    mock_block.input_schema.get_credentials_fields.return_value = {}
    mock_block.input_schema.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        }
    }
    # The auto-credentials validator respects optional fields — mark the
    # spreadsheet field as required so the missing-cred error is recorded.
    mock_block.input_schema.get_required_fields.return_value = ["spreadsheet"]
    mock_node.block = mock_block

    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]

    # Mock the credentials store to return None (cred not found for this user)
    mock_store = mocker.MagicMock()
    mock_store.get_creds_by_id = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "backend.executor.utils.get_integration_credentials_store",
        return_value=mock_store,
    )

    errors, nodes_to_skip = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="different-user",
        nodes_input_masks=None,
    )

    assert mock_node.id in errors
    assert "spreadsheet" in errors[mock_node.id]
    # Error message uses the CRED_ERR_UNKNOWN_PREFIX marker so the copilot
    # credential-race fallback recognises it as a credentials gate failure.
    assert (
        errors[mock_node.id]["spreadsheet"].lower().startswith("unknown credentials #")
    )


@pytest.mark.asyncio
async def test_validate_node_input_credentials_both_regular_and_auto(
    mocker: MockerFixture,
):
    """
    [SECRT-1772] A node that has BOTH regular credentials AND auto_credentials
    should have both validated.
    """
    from backend.executor.utils import _validate_node_input_credentials

    mock_node = mocker.MagicMock()
    mock_node.id = "node-with-both-creds"
    mock_node.credentials_optional = False
    mock_node.input_default = {
        "credentials": {
            "id": "regular-cred-id",
            "provider": "github",
            "type": "api_key",
        },
        "spreadsheet": {
            "_credentials_id": "auto-cred-id",
            "id": "file-123",
            "name": "test.xlsx",
        },
    }

    mock_credentials_field_type = mocker.MagicMock()
    mock_credentials_meta = mocker.MagicMock()
    mock_credentials_meta.id = "regular-cred-id"
    mock_credentials_meta.provider = "github"
    mock_credentials_meta.type = "api_key"
    mock_credentials_field_type.model_validate.return_value = mock_credentials_meta

    mock_block = mocker.MagicMock()
    # Regular credentials field
    mock_block.input_schema.get_credentials_fields.return_value = {
        "credentials": mock_credentials_field_type,
    }
    # Auto-credentials field
    mock_block.input_schema.get_auto_credentials_fields.return_value = {
        "auto_credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        }
    }
    mock_node.block = mock_block

    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]

    # Mock the credentials store to return valid credentials for both
    mock_store = mocker.MagicMock()
    mock_regular_creds = mocker.MagicMock()
    mock_regular_creds.id = "regular-cred-id"
    mock_regular_creds.provider = "github"
    mock_regular_creds.type = "api_key"

    mock_auto_creds = mocker.MagicMock()
    mock_auto_creds.id = "auto-cred-id"

    def get_creds_side_effect(user_id, cred_id):
        if cred_id == "regular-cred-id":
            return mock_regular_creds
        elif cred_id == "auto-cred-id":
            return mock_auto_creds
        return None

    mock_store.get_creds_by_id = mocker.AsyncMock(side_effect=get_creds_side_effect)
    mocker.patch(
        "backend.executor.utils.get_integration_credentials_store",
        return_value=mock_store,
    )

    errors, nodes_to_skip = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="test-user",
        nodes_input_masks=None,
    )

    # Both should validate successfully - no errors
    assert mock_node.id not in errors
    assert mock_node.id not in nodes_to_skip


@pytest.mark.asyncio
async def test_validate_node_input_credentials_auto_creds_optional_missing(
    mocker: MockerFixture,
):
    """When a node marks credentials optional and the auto-credential is
    missing, validation should not record an error — the node is simply
    marked for skip or runs without credentials."""
    from backend.executor.utils import _validate_node_input_credentials

    mock_node = mocker.MagicMock()
    mock_node.id = "node-optional-auto-creds"
    mock_node.credentials_optional = True
    mock_node.input_default = {
        "spreadsheet": {
            "_credentials_id": "other-users-cred-id",
            "id": "file-123",
            "name": "test.xlsx",
        }
    }

    mock_block = mocker.MagicMock()
    mock_block.input_schema.get_credentials_fields.return_value = {}
    mock_block.input_schema.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        }
    }
    mock_block.input_schema.get_required_fields.return_value = ["spreadsheet"]
    mock_node.block = mock_block

    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]

    mock_store = mocker.MagicMock()
    mock_store.get_creds_by_id = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "backend.executor.utils.get_integration_credentials_store",
        return_value=mock_store,
    )

    errors, nodes_to_skip = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="different-user",
        nodes_input_masks=None,
    )

    # Optional auto-credential that's missing must NOT error — instead the
    # node lands in nodes_to_skip so the executor doesn't try to run it.
    assert mock_node.id not in errors
    assert mock_node.id in nodes_to_skip


@pytest.mark.asyncio
async def test_validate_node_input_credentials_auto_creds_uses_marker_prefix(
    mocker: MockerFixture,
):
    """Auto-credential errors must use ``CRED_ERR_*`` prefixes so the copilot
    credential-race fallback recognises them — otherwise dry runs fail
    before the user gets a chance to re-auth."""
    from backend.executor.utils import (
        _validate_node_input_credentials,
        is_credential_validation_error_message,
    )

    mock_node = mocker.MagicMock()
    mock_node.id = "node-missing-auto-creds-key"
    mock_node.credentials_optional = False
    # Missing _credentials_id entirely — e.g. after a fork.
    mock_node.input_default = {
        "spreadsheet": {
            "id": "file-123",
            "name": "test.xlsx",
        }
    }

    mock_block = mocker.MagicMock()
    mock_block.input_schema.get_credentials_fields.return_value = {}
    mock_block.input_schema.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        }
    }
    mock_block.input_schema.get_required_fields.return_value = ["spreadsheet"]
    mock_node.block = mock_block

    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]

    errors, _ = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="some-user",
        nodes_input_masks=None,
    )

    assert mock_node.id in errors
    message = errors[mock_node.id]["spreadsheet"]
    assert is_credential_validation_error_message(message)


@pytest.mark.asyncio
async def test_validate_node_input_credentials_auto_creds_empty_string_id(
    mocker: MockerFixture,
):
    """A ``_credentials_id`` set to an empty string is a corrupted state —
    the validator must treat it like a missing credential, not silently
    pass. Without this guard, ``if cred_id and isinstance(cred_id, str)``
    evaluated to False and the node ran with no credentials injected."""
    from backend.executor.utils import _validate_node_input_credentials

    mock_node = mocker.MagicMock()
    mock_node.id = "node-with-empty-string-cred"
    mock_node.credentials_optional = False
    mock_node.input_default = {
        "spreadsheet": {
            "_credentials_id": "",  # corrupted
            "id": "file-123",
            "name": "test.xlsx",
        }
    }

    mock_block = mocker.MagicMock()
    mock_block.input_schema.get_credentials_fields.return_value = {}
    mock_block.input_schema.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        }
    }
    mock_block.input_schema.get_required_fields.return_value = ["spreadsheet"]
    mock_node.block = mock_block

    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]

    errors, _ = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="some-user",
        nodes_input_masks=None,
    )

    assert mock_node.id in errors
    assert "spreadsheet" in errors[mock_node.id]


@pytest.mark.asyncio
async def test_validate_node_input_credentials_auto_creds_skipped_when_none(
    mocker: MockerFixture,
):
    """
    When a node has auto_credentials but the field value has _credentials_id=None
    (e.g., from upstream connection), validation should skip it without error.
    """
    from backend.executor.utils import _validate_node_input_credentials

    mock_node = mocker.MagicMock()
    mock_node.id = "node-with-chained-auto-creds"
    mock_node.credentials_optional = False
    mock_node.input_default = {
        "spreadsheet": {
            "_credentials_id": None,
            "id": "file-123",
            "name": "test.xlsx",
        }
    }

    mock_block = mocker.MagicMock()
    mock_block.input_schema.get_credentials_fields.return_value = {}
    mock_block.input_schema.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        }
    }
    mock_node.block = mock_block

    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]

    errors, nodes_to_skip = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="test-user",
        nodes_input_masks=None,
    )

    # No error - chained data with None cred_id is valid
    assert mock_node.id not in errors


@pytest.mark.asyncio
async def test_validate_node_input_credentials_auto_creds_optional_none_value_skips(
    mocker: MockerFixture,
):
    """Sentry HIGH regression: if input_default[field_name] is explicitly
    ``None`` (e.g. cleared by ``_reassign_ids`` on fork) and the field is
    optional, the validator previously silently skipped the whole
    auto-credentials block — ``has_missing_credentials`` never flipped
    true and the node never landed in ``nodes_to_skip``. Then
    ``_acquire_auto_credentials`` would hit ``field_data is None`` at
    runtime and raise ``ValueError`` instead of cleanly skipping.

    The validator must treat an explicitly-None value as a missing
    credential and, for optional fields, add the node to
    ``nodes_to_skip`` instead."""
    from backend.executor.utils import _validate_node_input_credentials

    mock_node = mocker.MagicMock()
    mock_node.id = "node-cleared-on-fork"
    mock_node.credentials_optional = True
    # input_default has the field but its value was cleared to None
    mock_node.input_default = {"spreadsheet": None}

    mock_block = mocker.MagicMock()
    mock_block.input_schema.get_credentials_fields.return_value = {}
    mock_block.input_schema.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        }
    }
    mock_block.input_schema.get_required_fields.return_value = ["spreadsheet"]
    mock_node.block = mock_block

    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]

    errors, nodes_to_skip = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="some-user",
        nodes_input_masks=None,
    )

    # Optional + missing → node MUST land in nodes_to_skip so the executor
    # never enters a run that would crash in `_acquire_auto_credentials`.
    assert mock_node.id not in errors
    assert mock_node.id in nodes_to_skip


@pytest.mark.asyncio
async def test_validate_node_input_credentials_field_level_optional_none_value_skips(
    mocker: MockerFixture,
):
    """Cursor Medium (thread PRRT_kwDOJKSTjM58r_37): a node with
    ``credentials_optional=False`` (the default) but whose auto-credential
    field is NOT in ``required_fields`` (typical — the ``spreadsheet``
    field on Google Sheets blocks defaults to None, so pydantic marks it
    non-required at the schema level). The per-field check correctly
    flags ``field_is_optional=True`` via ``field_name not in
    required_fields``, but the POST-LOOP guard used ``is_creds_optional``
    (the node-level flag) only — so the node silently passed validation
    and crashed at runtime inside ``_acquire_auto_credentials`` with
    ``ValueError('No file selected')``.

    Pin the contract: when ANY per-field branch decides the field is
    optional and missing, the node must land in ``nodes_to_skip``
    regardless of the node-level ``credentials_optional`` flag."""
    from backend.executor.utils import _validate_node_input_credentials

    mock_node = mocker.MagicMock()
    mock_node.id = "node-field-optional-cleared"
    # Node-level flag is False (the common case) — the field is only
    # field-level optional because it's absent from required_fields.
    mock_node.credentials_optional = False
    mock_node.input_default = {"spreadsheet": None}

    mock_block = mocker.MagicMock()
    mock_block.input_schema.get_credentials_fields.return_value = {}
    mock_block.input_schema.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        }
    }
    # Field-level optional: `spreadsheet` is NOT in required_fields because
    # its pydantic default is None.
    mock_block.input_schema.get_required_fields.return_value = []
    mock_node.block = mock_block

    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]

    errors, nodes_to_skip = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="some-user",
        nodes_input_masks=None,
    )

    assert mock_node.id not in errors
    assert mock_node.id in nodes_to_skip


@pytest.mark.asyncio
async def test_validate_node_input_credentials_auto_creds_required_none_value_errors(
    mocker: MockerFixture,
):
    """Sentry HIGH regression, required-field variant. If
    ``input_default[field_name]`` is explicitly ``None`` and the field is
    required, the validator must surface a
    ``CRED_ERR_NOT_AVAILABLE_PREFIX`` error so the dry-run gate fires
    before we enter `run()` — rather than silently letting the node pass
    validation and crashing inside `_acquire_auto_credentials`."""
    from backend.executor.utils import (
        _validate_node_input_credentials,
        is_credential_validation_error_message,
    )

    mock_node = mocker.MagicMock()
    mock_node.id = "node-cleared-on-fork-required"
    mock_node.credentials_optional = False
    mock_node.input_default = {"spreadsheet": None}

    mock_block = mocker.MagicMock()
    mock_block.input_schema.get_credentials_fields.return_value = {}
    mock_block.input_schema.get_auto_credentials_fields.return_value = {
        "credentials": {
            "field_name": "spreadsheet",
            "config": {"provider": "google", "type": "oauth2"},
        }
    }
    mock_block.input_schema.get_required_fields.return_value = ["spreadsheet"]
    mock_node.block = mock_block

    mock_graph = mocker.MagicMock()
    mock_graph.nodes = [mock_node]

    errors, _ = await _validate_node_input_credentials(
        graph=mock_graph,
        user_id="some-user",
        nodes_input_masks=None,
    )

    assert mock_node.id in errors
    assert "spreadsheet" in errors[mock_node.id]
    assert is_credential_validation_error_message(errors[mock_node.id]["spreadsheet"])


# ============================================================================
# Tests for CredentialsFieldInfo auto_credential tag (Fix 4: Path 4)
# ============================================================================


def test_credentials_field_info_auto_credential_tag():
    """
    [Path 4] CredentialsFieldInfo should support is_auto_credential and
    input_field_name fields for distinguishing auto from regular credentials.
    """
    from backend.data.model import CredentialsFieldInfo

    # Regular credential should have is_auto_credential=False by default
    regular = CredentialsFieldInfo.model_validate(
        {
            "credentials_provider": ["github"],
            "credentials_types": ["api_key"],
        },
        by_alias=True,
    )
    assert regular.is_auto_credential is False
    assert regular.input_field_name is None

    # Auto credential should have is_auto_credential=True
    auto = CredentialsFieldInfo.model_validate(
        {
            "credentials_provider": ["google"],
            "credentials_types": ["oauth2"],
            "is_auto_credential": True,
            "input_field_name": "spreadsheet",
        },
        by_alias=True,
    )
    assert auto.is_auto_credential is True
    assert auto.input_field_name == "spreadsheet"


def test_make_node_credentials_input_map_excludes_auto_creds(
    mocker: MockerFixture,
):
    """
    [Path 4] make_node_credentials_input_map should only include regular credentials,
    not auto_credentials (which are resolved at execution time).
    """
    from backend.data.model import CredentialsFieldInfo, CredentialsMetaInput
    from backend.executor.utils import make_node_credentials_input_map
    from backend.integrations.providers import ProviderName

    # Create a mock graph with aggregate_credentials_inputs that returns
    # both regular and auto credentials
    mock_graph = mocker.MagicMock()

    regular_field_info = CredentialsFieldInfo.model_validate(
        {
            "credentials_provider": ["github"],
            "credentials_types": ["api_key"],
            "is_auto_credential": False,
        },
        by_alias=True,
    )

    # Mock regular_credentials_inputs property (auto_credentials are excluded)
    mock_graph.regular_credentials_inputs = {
        "github_creds": (regular_field_info, {("node-1", "credentials")}, True),
    }

    graph_credentials_input = {
        "github_creds": CredentialsMetaInput(
            id="cred-123",
            provider=ProviderName("github"),
            type="api_key",
        ),
    }

    result = make_node_credentials_input_map(mock_graph, graph_credentials_input)

    # Regular credentials should be mapped
    assert "node-1" in result
    assert "credentials" in result["node-1"]

    # Auto credentials should NOT appear in the result
    # (they would have been mapped to the kwarg_name "credentials" not "spreadsheet")
    for node_id, fields in result.items():
        for field_name, value in fields.items():
            # Verify no auto-credential phantom entries
            if isinstance(value, dict):
                assert "_credentials_id" not in value
