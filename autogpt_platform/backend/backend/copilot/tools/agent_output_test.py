"""Tests for AgentOutputTool response building — run-health surfacing.

A COMPLETED graph run can contain FAILED nodes (node errors don't fail the
run) or produce no outputs; the response message must surface that instead
of reading as success.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from backend.data.execution import (
    ExecutionStatus,
    GraphExecution,
    GraphExecutionMeta,
    GraphExecutionWithNodes,
)

from .agent_output import AgentOutputTool

_MODULE = "backend.copilot.tools.agent_output"


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.id = "lib-1"
    agent.name = "Test Agent"
    agent.graph_id = "graph-1"
    return agent


def _execution(
    *,
    with_nodes: bool,
    outputs: dict,
    node_executions: list,
    node_error_count: int = 0,
):
    spec = GraphExecutionWithNodes if with_nodes else GraphExecution
    execution = MagicMock(spec=spec)
    execution.id = "exec-1"
    execution.status = ExecutionStatus.COMPLETED
    execution.started_at = None
    execution.ended_at = None
    execution.outputs = outputs
    execution.inputs = None
    execution.stats = GraphExecutionMeta.Stats(node_error_count=node_error_count)
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


def test_completed_with_output_and_failed_node_warns_without_the_trace():
    """The default call loads no per-node trace. A run that produced output
    while a node failed must still warn, from the run's own failure count."""
    response = AgentOutputTool()._build_response(
        agent=_agent(),
        execution=_execution(
            with_nodes=False,
            outputs={"result": ["partial"]},
            node_executions=[],
            node_error_count=1,
        ),
        available_executions=[],
        session_id="sess-1",
    )
    assert "1 node(s) FAILED despite the COMPLETED" in response.message
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


def _library(**methods) -> MagicMock:
    lib = MagicMock()
    for name, mock in methods.items():
        setattr(lib, name, mock)
    return lib


async def _resolve(lib: MagicMock, **kwargs) -> tuple:
    args = {"agent_name": None, "library_agent_id": None, "store_slug": None}
    args.update(kwargs)
    with patch(f"{_MODULE}.library_db", return_value=lib):
        return await AgentOutputTool()._resolve_agent(user_id="user-1", **args)


async def test_resolve_unknown_library_id_names_the_workflow():
    lib = _library(get_library_agent=AsyncMock(side_effect=ValueError("missing")))
    agent, error = await _resolve(lib, library_agent_id="lib-9")
    assert agent is None
    assert error == "Library workflow 'lib-9' not found"


async def test_resolve_unknown_store_slug_names_the_workflow():
    with patch(
        f"{_MODULE}.fetch_graph_from_store_slug",
        AsyncMock(return_value=(None, None)),
    ):
        agent, error = await _resolve(_library(), store_slug="alice/digest")
    assert agent is None
    assert error == "Workflow 'alice/digest' not found in marketplace"


async def test_resolve_store_workflow_missing_from_library_says_add_it():
    graph = MagicMock()
    graph.id = "graph-7"
    lib = _library(get_library_agent_by_graph_id=AsyncMock(return_value=None))
    with patch(
        f"{_MODULE}.fetch_graph_from_store_slug",
        AsyncMock(return_value=(graph, None)),
    ):
        agent, error = await _resolve(lib, store_slug="alice/digest")
    assert agent is None
    assert error == (
        "Workflow 'alice/digest' is not in your library. "
        "Add it first to see outputs."
    )


async def test_resolve_name_with_no_match_names_workflows():
    lib = _library(list_library_agents=AsyncMock(return_value=MagicMock(agents=[])))
    agent, error = await _resolve(lib, agent_name="Digest")
    assert agent is None
    assert error == "No workflows matching 'Digest' found in your library"


async def test_resolve_name_search_failure_names_the_workflow():
    lib = _library(list_library_agents=AsyncMock(side_effect=RuntimeError("db down")))
    agent, error = await _resolve(lib, agent_name="Digest")
    assert agent is None
    assert error == "Error searching for workflow: db down"


async def test_resolve_with_no_identifier_asks_for_a_workflow_name():
    agent, error = await _resolve(_library())
    assert agent is None
    assert error == "Please specify a workflow name, library_agent_id, or store_slug"
