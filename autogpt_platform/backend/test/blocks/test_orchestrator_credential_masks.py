"""
Tests for OrchestratorBlock credential input mask forwarding.

Verifies that _execute_single_tool_with_manager merges
execution_processor.nodes_input_masks into node inputs AND forwards
them to on_node_execution, so Library/AutoPilot credential overrides
reach tool nodes.
"""

import threading
import types
import uuid
from collections import defaultdict
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.orchestrator import ExecutionParams, OrchestratorBlock, ToolInfo
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    NodeExecutionResult,
)
from backend.data.model import GraphExecutionStats, NodeExecutionStats
from backend.executor.utils import NodeExecutionProgress


def _uid() -> str:
    return str(uuid.uuid4())


def _make_tool_info(sink_node_id: str) -> ToolInfo:
    """Create a minimal ToolInfo for testing."""
    tool_call = types.SimpleNamespace(
        id=f"call_{_uid()[:8]}",
        function=types.SimpleNamespace(
            name="test_tool",
            arguments="{}",
        ),
    )
    tool_def = {
        "function": {
            "_sink_node_id": sink_node_id,
            "_field_mapping": {},
            "parameters": {"properties": {"value": {"type": "string"}}},
        }
    }
    return ToolInfo(
        tool_call=tool_call,
        tool_name="test_tool",
        tool_def=tool_def,
        input_data={"value": "hello"},
        field_mapping={},
    )


def _make_execution_params() -> ExecutionParams:
    return ExecutionParams(
        user_id=_uid(),
        graph_id=_uid(),
        node_id=_uid(),
        graph_version=1,
        graph_exec_id=_uid(),
        node_exec_id=_uid(),
        execution_context=ExecutionContext(dry_run=False),
    )


def _make_mock_node(sink_node_id: str, block_id: str) -> MagicMock:
    node = MagicMock()
    node.id = sink_node_id
    node.block_id = block_id
    node.input_default = {"name": "result"}
    return node


def _make_node_exec_result(
    sink_node_id: str, graph_exec_id: str
) -> NodeExecutionResult:
    return NodeExecutionResult(
        user_id=_uid(),
        graph_id=_uid(),
        graph_version=1,
        graph_exec_id=graph_exec_id,
        node_exec_id=_uid(),
        node_id=sink_node_id,
        block_id=_uid(),
        status=ExecutionStatus.QUEUED,
        input_data={"value": "hello"},
        output_data={},
        add_time=datetime.now(),
        queue_time=None,
        start_time=None,
        end_time=None,
    )


def _setup_mocks(sink_node_id, params, masks):
    """Set up common mock objects for tool execution tests."""
    block_id = _uid()
    mock_node = _make_mock_node(sink_node_id, block_id)
    node_exec_result = _make_node_exec_result(sink_node_id, params.graph_exec_id)

    mock_db = AsyncMock()
    mock_db.get_node.return_value = mock_node
    mock_db.upsert_execution_input.return_value = (
        node_exec_result,
        {"name": "result", "value": "hello"},
    )
    # Tool dispatch reads outputs back after on_node_execution and json.dumps()
    # them; return a JSON-serializable dict so the post-execution path runs
    # cleanly instead of recursing on an unmocked MagicMock.
    mock_db.get_execution_outputs_by_node_exec_id.return_value = {"result": "ok"}

    mock_processor = MagicMock()
    mock_processor.nodes_input_masks = masks
    mock_processor.running_node_execution = defaultdict(NodeExecutionProgress)
    mock_processor.execution_stats = GraphExecutionStats()
    mock_processor.execution_stats_lock = threading.Lock()
    mock_processor.on_node_execution = AsyncMock(return_value=NodeExecutionStats())
    # charge_node_usage is awaited after a successful tool run; it must be an
    # AsyncMock or the await raises TypeError mid-flight.
    mock_processor.charge_node_usage = AsyncMock(return_value=None)

    return mock_db, mock_processor


def _get_on_node_execution_args(mock_processor):
    """Extract args from the on_node_execution call."""
    call_kwargs = mock_processor.on_node_execution.call_args
    node_exec = call_kwargs.kwargs.get(
        "node_exec", call_kwargs.args[0] if len(call_kwargs.args) > 0 else None
    )
    masks = call_kwargs.kwargs.get(
        "nodes_input_masks", call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
    )
    return node_exec, masks


@pytest.mark.asyncio
async def test_execute_single_tool_merges_masks_into_inputs():
    """
    _execute_single_tool_with_manager must merge the sink node\'s mask
    from execution_processor.nodes_input_masks into node_exec_entry.inputs
    before calling on_node_execution -- mirroring the normal queue-based path.
    """
    sink_node_id = _uid()
    params = _make_execution_params()

    expected_masks = {
        sink_node_id: {"credentials": {"api_key": "sk-test-secret"}},
    }

    mock_db, mock_processor = _setup_mocks(sink_node_id, params, expected_masks)
    block = OrchestratorBlock()

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        await block._execute_single_tool_with_manager(
            tool_info=_make_tool_info(sink_node_id),
            execution_params=params,
            execution_processor=mock_processor,
        )

    mock_processor.on_node_execution.assert_called_once()
    node_exec, actual_masks = _get_on_node_execution_args(mock_processor)

    # Masks forwarded to on_node_execution for downstream propagation
    assert actual_masks is expected_masks

    # Masks merged into node_exec_entry.inputs (the actual bug fix)
    assert (
        "credentials" in node_exec.inputs
    ), f"Credential mask not merged into inputs: {node_exec.inputs!r}"
    assert node_exec.inputs["credentials"] == {"api_key": "sk-test-secret"}


@pytest.mark.asyncio
async def test_execute_single_tool_no_masks_no_crash():
    """
    When nodes_input_masks is None, on_node_execution receives None
    and inputs are unmodified.
    """
    sink_node_id = _uid()
    params = _make_execution_params()

    mock_db, mock_processor = _setup_mocks(sink_node_id, params, masks=None)
    block = OrchestratorBlock()

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        await block._execute_single_tool_with_manager(
            tool_info=_make_tool_info(sink_node_id),
            execution_params=params,
            execution_processor=mock_processor,
        )

    mock_processor.on_node_execution.assert_called_once()
    node_exec, actual_masks = _get_on_node_execution_args(mock_processor)

    assert actual_masks is None
    assert "credentials" not in node_exec.inputs


@pytest.mark.asyncio
async def test_execute_single_tool_masks_for_different_node_not_merged():
    """
    Masks for other nodes must not leak into the sink node\'s inputs.
    The full masks dict is still forwarded for downstream propagation.
    """
    sink_node_id = _uid()
    other_node_id = _uid()
    params = _make_execution_params()

    masks = {other_node_id: {"credentials": {"api_key": "sk-other"}}}

    mock_db, mock_processor = _setup_mocks(sink_node_id, params, masks)
    block = OrchestratorBlock()

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        await block._execute_single_tool_with_manager(
            tool_info=_make_tool_info(sink_node_id),
            execution_params=params,
            execution_processor=mock_processor,
        )

    mock_processor.on_node_execution.assert_called_once()
    node_exec, actual_masks = _get_on_node_execution_args(mock_processor)

    # Full masks forwarded for downstream nodes
    assert actual_masks is masks
    # Sink node inputs unmodified
    assert "credentials" not in node_exec.inputs


@pytest.mark.asyncio
async def test_execute_single_tool_non_mapping_masks_coerced_to_none():
    """
    Guard: when nodes_input_masks is set to a non-Mapping value
    (e.g. an unexpected type from a stale/uninitialised processor), the
    orchestrator coerces it to None instead of crashing, and on_node_execution
    receives None.
    """
    sink_node_id = _uid()
    params = _make_execution_params()

    # Non-Mapping sentinel (a list) — exercises the `not isinstance(..., Mapping)`
    # guard branch added to harden against ExecutionProcessor instances that
    # don't expose a dict-like nodes_input_masks attribute.
    mock_db, mock_processor = _setup_mocks(
        sink_node_id, params, masks=["not", "a", "mapping"]
    )
    block = OrchestratorBlock()

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        await block._execute_single_tool_with_manager(
            tool_info=_make_tool_info(sink_node_id),
            execution_params=params,
            execution_processor=mock_processor,
        )

    mock_processor.on_node_execution.assert_called_once()
    node_exec, actual_masks = _get_on_node_execution_args(mock_processor)

    # Non-Mapping was coerced to None before being forwarded
    assert actual_masks is None
    # No merge happened; inputs are unchanged
    assert "credentials" not in node_exec.inputs


@pytest.mark.asyncio
async def test_execute_single_tool_empty_masks_no_merge():
    """
    When nodes_input_masks is an empty Mapping (truthy guard is False),
    no merge occurs and the empty dict is still forwarded.
    Covers the `if nodes_input_masks and ...` branch where the dict
    is present but the sink node has no entry.
    """
    sink_node_id = _uid()
    params = _make_execution_params()

    empty_masks: dict[str, dict] = {}

    mock_db, mock_processor = _setup_mocks(sink_node_id, params, empty_masks)
    block = OrchestratorBlock()

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        await block._execute_single_tool_with_manager(
            tool_info=_make_tool_info(sink_node_id),
            execution_params=params,
            execution_processor=mock_processor,
        )

    mock_processor.on_node_execution.assert_called_once()
    node_exec, actual_masks = _get_on_node_execution_args(mock_processor)

    assert actual_masks is empty_masks
    assert "credentials" not in node_exec.inputs
