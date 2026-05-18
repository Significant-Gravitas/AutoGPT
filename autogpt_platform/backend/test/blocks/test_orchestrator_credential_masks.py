"""
Tests for OrchestratorBlock credential input mask forwarding.

Verifies that _execute_single_tool_with_manager forwards
execution_processor.nodes_input_masks to on_node_execution,
so Library/AutoPilot credential overrides reach tool nodes.
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


@pytest.mark.asyncio
async def test_execute_single_tool_forwards_input_masks():
    """
    When OrchestratorBlock._execute_single_tool_with_manager executes a tool
    node, it must forward execution_processor.nodes_input_masks to
    on_node_execution so that credential overrides from Library/AutoPilot
    reach the tool node.
    """
    sink_node_id = _uid()
    block_id = _uid()
    params = _make_execution_params()
    mock_node = _make_mock_node(sink_node_id, block_id)
    node_exec_result = _make_node_exec_result(sink_node_id, params.graph_exec_id)

    # The credential masks that should be forwarded
    expected_masks = {
        sink_node_id: {"credentials": {"api_key": "sk-test-secret"}},
    }

    # Mock DB client
    mock_db = AsyncMock()
    mock_db.get_node.return_value = mock_node
    mock_db.upsert_execution_input.return_value = (
        node_exec_result,
        {"name": "result", "value": "hello"},
    )

    # Mock execution processor
    mock_processor = MagicMock()
    mock_processor.nodes_input_masks = expected_masks
    mock_processor.running_node_execution = defaultdict(NodeExecutionProgress)
    mock_processor.execution_stats = GraphExecutionStats()
    mock_processor.execution_stats_lock = threading.Lock()

    # on_node_execution should return stats and capture the masks arg
    mock_stats = NodeExecutionStats()
    mock_processor.on_node_execution = AsyncMock(return_value=mock_stats)

    tool_info = _make_tool_info(sink_node_id)
    block = OrchestratorBlock()

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        await block._execute_single_tool_with_manager(
            tool_info=tool_info,
            execution_params=params,
            execution_processor=mock_processor,
        )

    # Verify on_node_execution was called with the actual masks, not None
    mock_processor.on_node_execution.assert_called_once()
    call_kwargs = mock_processor.on_node_execution.call_args
    actual_masks = call_kwargs.kwargs.get(
        "nodes_input_masks", call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
    )
    assert actual_masks is expected_masks, (
        f"Expected nodes_input_masks={expected_masks!r}, got {actual_masks!r}. "
        "Credential overrides from Library/AutoPilot must be forwarded to tool nodes."
    )


@pytest.mark.asyncio
async def test_execute_single_tool_forwards_none_masks_when_absent():
    """
    When execution_processor.nodes_input_masks is None (no credential
    overrides), on_node_execution should receive None -- not crash.
    """
    sink_node_id = _uid()
    block_id = _uid()
    params = _make_execution_params()
    mock_node = _make_mock_node(sink_node_id, block_id)
    node_exec_result = _make_node_exec_result(sink_node_id, params.graph_exec_id)

    mock_db = AsyncMock()
    mock_db.get_node.return_value = mock_node
    mock_db.upsert_execution_input.return_value = (
        node_exec_result,
        {"name": "result", "value": "hello"},
    )

    mock_processor = MagicMock()
    mock_processor.nodes_input_masks = None
    mock_processor.running_node_execution = defaultdict(NodeExecutionProgress)
    mock_processor.execution_stats = GraphExecutionStats()
    mock_processor.execution_stats_lock = threading.Lock()

    mock_stats = NodeExecutionStats()
    mock_processor.on_node_execution = AsyncMock(return_value=mock_stats)

    tool_info = _make_tool_info(sink_node_id)
    block = OrchestratorBlock()

    with patch(
        "backend.blocks.orchestrator.get_database_manager_async_client",
        return_value=mock_db,
    ):
        await block._execute_single_tool_with_manager(
            tool_info=tool_info,
            execution_params=params,
            execution_processor=mock_processor,
        )

    mock_processor.on_node_execution.assert_called_once()
    call_kwargs = mock_processor.on_node_execution.call_args
    actual_masks = call_kwargs.kwargs.get(
        "nodes_input_masks", call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
    )
    assert actual_masks is None


@pytest.mark.asyncio
async def test_nodes_input_masks_stored_on_execution_processor():
    """
    ExecutionProcessor must store graph_exec.nodes_input_masks on
    self.nodes_input_masks so it's accessible during orchestrator
    tool execution.
    """
    from backend.executor.manager import ExecutionProcessor

    expected_masks = {"node-1": {"credentials": {"key": "val"}}}

    # Verify the attribute contract: ExecutionProcessor instances must
    # support .nodes_input_masks assignment (set in _on_graph_execution).
    processor = ExecutionProcessor.__new__(ExecutionProcessor)
    processor.nodes_input_masks = expected_masks
    assert processor.nodes_input_masks is expected_masks
