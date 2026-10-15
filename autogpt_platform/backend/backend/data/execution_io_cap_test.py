"""Coverage for the cap-hit warning path on NodeExecutionResult.from_db."""

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock

from prisma.enums import AgentExecutionStatus

from backend.data.execution import NodeExecutionResult
from backend.data.includes import MAX_NODE_INPUT_OUTPUT_FETCH


def _make_io_row(name: str, value: str) -> MagicMock:
    row = MagicMock()
    row.name = name
    row.data = value
    return row


def _make_node_exec(*, inputs: list, outputs: list, exec_id: str = "ne-1") -> MagicMock:
    ne = MagicMock()
    ne.id = exec_id
    ne.stats = {}
    ne.executionData = None
    ne.Input = inputs
    ne.Output = outputs
    ne.agentGraphExecutionId = "ge-1"
    ne.agentNodeId = "n-1"
    ne.executionStatus = AgentExecutionStatus.COMPLETED
    ne.addedTime = datetime.now(timezone.utc)
    ne.queuedTime = None
    ne.startedTime = None
    ne.endedTime = None

    ne.Node = MagicMock()
    ne.Node.agentBlockId = "blk-1"

    ne.GraphExecution = MagicMock()
    ne.GraphExecution.userId = "u-1"
    ne.GraphExecution.agentGraphId = "g-1"
    ne.GraphExecution.agentGraphVersion = 1
    return ne


def test_node_exec_input_cap_logs_warning(caplog):
    inputs = [
        _make_io_row(f"in_{i}", f"v{i}") for i in range(MAX_NODE_INPUT_OUTPUT_FETCH)
    ]
    ne = _make_node_exec(inputs=inputs, outputs=[])

    with caplog.at_level(logging.WARNING, logger="backend.data.execution"):
        result = NodeExecutionResult.from_db(ne)

    assert any(
        "Input rows hit MAX_NODE_INPUT_OUTPUT_FETCH" in m for m in caplog.messages
    )
    assert isinstance(result, NodeExecutionResult)


def test_node_exec_output_cap_logs_warning(caplog):
    outputs = [_make_io_row("out", f"v{i}") for i in range(MAX_NODE_INPUT_OUTPUT_FETCH)]
    ne = _make_node_exec(inputs=[], outputs=outputs)

    with caplog.at_level(logging.WARNING, logger="backend.data.execution"):
        result = NodeExecutionResult.from_db(ne)

    assert any(
        "Output rows hit MAX_NODE_INPUT_OUTPUT_FETCH" in m for m in caplog.messages
    )
    assert isinstance(result, NodeExecutionResult)


def test_node_exec_under_cap_does_not_warn(caplog):
    inputs = [_make_io_row(f"in_{i}", f"v{i}") for i in range(3)]
    outputs = [_make_io_row("out", f"v{i}") for i in range(3)]
    ne = _make_node_exec(inputs=inputs, outputs=outputs)

    with caplog.at_level(logging.WARNING, logger="backend.data.execution"):
        result = NodeExecutionResult.from_db(ne)

    assert not any("MAX_NODE_INPUT_OUTPUT_FETCH" in m for m in caplog.messages)
    assert isinstance(result, NodeExecutionResult)
