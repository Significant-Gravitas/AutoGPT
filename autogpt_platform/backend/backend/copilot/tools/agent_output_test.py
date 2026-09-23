"""Tests for AgentOutputTool response building — run-health surfacing.

A COMPLETED graph run can contain FAILED nodes (node errors don't fail the
run) or produce no outputs; the response message must surface that instead
of reading as success.
"""

from unittest.mock import MagicMock

from backend.data.execution import (
    ExecutionStatus,
    GraphExecution,
    GraphExecutionWithNodes,
)

from .agent_output import AgentOutputTool


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.id = "lib-1"
    agent.name = "Test Agent"
    agent.graph_id = "graph-1"
    return agent


def _execution(*, with_nodes: bool, outputs: dict, node_executions: list):
    spec = GraphExecutionWithNodes if with_nodes else GraphExecution
    execution = MagicMock(spec=spec)
    execution.id = "exec-1"
    execution.status = ExecutionStatus.COMPLETED
    execution.started_at = None
    execution.ended_at = None
    execution.outputs = outputs
    execution.inputs = None
    if with_nodes:
        execution.node_executions = node_executions
    return execution


def _failed_node() -> MagicMock:
    ne = MagicMock()
    ne.status = ExecutionStatus.FAILED
    ne.node_id = "n1"
    ne.block_id = "block-x"
    ne.output_data = {"error": ["boom exploded"]}
    ne.input_data = {}
    ne.start_time = None
    ne.end_time = None
    return ne


def test_completed_with_failed_node_warns():
    response = AgentOutputTool()._build_response(
        agent=_agent(),
        execution=_execution(
            with_nodes=True,
            outputs={"result": ["ok"]},
            node_executions=[_failed_node()],
        ),
        available_executions=[],
        session_id="sess-1",
    )
    assert "FAILED despite the COMPLETED" in response.message
    assert response.execution.nodes_failed is not None
    assert "show_execution_details" not in response.message


def test_completed_empty_outputs_without_nodes_nudges_details():
    """Without the per-node trace loaded, the warning nudges a re-call with
    show_execution_details=true."""
    response = AgentOutputTool()._build_response(
        agent=_agent(),
        execution=_execution(with_nodes=False, outputs={}, node_executions=[]),
        available_executions=[],
        session_id="sess-1",
    )
    assert "produced no outputs" in response.message
    assert "show_execution_details=true" in response.message


def test_healthy_completed_has_no_warning():
    response = AgentOutputTool()._build_response(
        agent=_agent(),
        execution=_execution(
            with_nodes=True, outputs={"result": ["ok"]}, node_executions=[]
        ),
        available_executions=[],
        session_id="sess-1",
    )
    assert "WARNING" not in response.message
    assert response.execution.nodes_failed is None
