"""Tests for GraphExecution.from_db — verify node executions without 'name'
in input_data (e.g. OrchestratorBlock) are skipped gracefully."""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from prisma.enums import AgentExecutionStatus

from backend.blocks._base import BlockType
from backend.data.execution import GraphExecution


def _make_node_execution(
    *,
    exec_id: str,
    block_id: str,
    input_data: dict | None = None,
    output_data: list | None = None,
    status: AgentExecutionStatus = AgentExecutionStatus.COMPLETED,
) -> MagicMock:
    """Create a minimal mock AgentNodeExecution for from_db."""
    ne = MagicMock()
    ne.id = exec_id
    ne.agentNodeId = f"node-{exec_id}"
    ne.agentGraphExecutionId = "graph-exec-1"
    ne.executionStatus = status
    ne.addedTime = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ne.queuedTime = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ne.startedTime = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ne.endedTime = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ne.stats = None
    ne.executionData = input_data or {}
    ne.Input = []
    ne.Output = output_data or []

    node = MagicMock()
    node.agentBlockId = block_id
    ne.Node = node

    ne.GraphExecution = None
    return ne


def _make_graph_execution(node_executions: list) -> MagicMock:
    """Create a minimal mock AgentGraphExecution."""
    ge = MagicMock()
    ge.id = "graph-exec-1"
    ge.userId = "user-1"
    ge.agentGraphId = "graph-1"
    ge.agentGraphVersion = 1
    ge.inputs = None  # Force fallback to node-based extraction
    ge.credentialInputs = None
    ge.nodesInputMasks = None
    ge.agentPresetId = None
    ge.executionStatus = AgentExecutionStatus.COMPLETED
    ge.startedAt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ge.endedAt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ge.stats = None
    ge.isShared = False
    ge.shareToken = None
    ge.NodeExecutions = node_executions
    return ge


INPUT_BLOCK_ID = "input-block-id"
OUTPUT_BLOCK_ID = "output-block-id"
ORCHESTRATOR_BLOCK_ID = "orchestrator-block-id"


def _mock_get_block(block_id: str):
    """Return a mock block with the right block_type for each ID."""
    block = MagicMock()
    if block_id == INPUT_BLOCK_ID:
        block.block_type = BlockType.INPUT
    elif block_id == OUTPUT_BLOCK_ID:
        block.block_type = BlockType.OUTPUT
    else:
        block.block_type = BlockType.STANDARD
    return block


@patch("backend.data.execution.get_block", side_effect=_mock_get_block)
def test_from_db_skips_node_executions_without_name(mock_get_block: MagicMock):
    """Node executions without 'name' in input_data (e.g. OrchestratorBlock)
    must not cause a KeyError and should be silently skipped."""
    named_input = _make_node_execution(
        exec_id="ne-input-1",
        block_id=INPUT_BLOCK_ID,
        input_data={"name": "query", "value": "hello"},
    )
    unnamed_input = _make_node_execution(
        exec_id="ne-orchestrator-1",
        block_id=INPUT_BLOCK_ID,
        input_data={"value": "no name field here"},
    )
    named_output = _make_node_execution(
        exec_id="ne-output-1",
        block_id=OUTPUT_BLOCK_ID,
        input_data={"name": "result", "value": "world"},
    )
    unnamed_output = _make_node_execution(
        exec_id="ne-orchestrator-2",
        block_id=OUTPUT_BLOCK_ID,
        input_data={"value": "no name here either"},
    )
    standard_node = _make_node_execution(
        exec_id="ne-standard-1",
        block_id=ORCHESTRATOR_BLOCK_ID,
        input_data={"some_key": "some_value"},
    )

    graph_exec_db = _make_graph_execution(
        [named_input, unnamed_input, named_output, unnamed_output, standard_node]
    )

    result = GraphExecution.from_db(graph_exec_db)

    # Named input extracted correctly
    assert result.inputs == {"query": "hello"}
    # Named output extracted; unnamed output skipped
    assert dict(result.outputs) == {"result": ["world"]}
