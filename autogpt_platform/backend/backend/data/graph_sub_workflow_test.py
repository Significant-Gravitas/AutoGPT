"""A workflow that would run itself as a sub-workflow, directly or through
others, is refused when it is saved and when it is run. Sub-workflows that
only share a workflow are not a cycle."""

import pytest

from backend.blocks.agent import AgentExecutorBlock
from backend.data.graph import (
    SUB_WORKFLOW_RUNS_ITSELF,
    BaseGraph,
    Graph,
    GraphModel,
    Node,
    make_graph_model,
)
from backend.util.exceptions import GraphValidationError


def test_saving_a_workflow_that_runs_itself_is_refused():
    # Saved as a new workflow: its sub-workflow node names the id it was sent
    # with, which saving rewrites to the new workflow's own id.
    graph = _workflow("as-sent", _runs("as-sent", "runs-itself"))
    graph.reassign_ids(user_id="user-1", reassign_graph_id=True)

    with pytest.raises(GraphValidationError) as refused:
        graph.validate_graph(for_run=False)

    assert refused.value.message == SUB_WORKFLOW_RUNS_ITSELF
    assert refused.value.node_errors == {
        graph.nodes[0].id: {"graph_id": SUB_WORKFLOW_RUNS_ITSELF}
    }


def test_saving_a_workflow_that_runs_itself_through_another_is_refused():
    graph = _workflow(
        "outer",
        _runs("inner", "outer-runs-inner"),
        sub_graphs=[_sub_graph("inner", _runs("outer", "inner-runs-outer"))],
    )

    with pytest.raises(GraphValidationError) as refused:
        graph.validate_graph(for_run=False)

    assert set(refused.value.node_errors) == {"outer-runs-inner", "inner-runs-outer"}


def test_a_run_of_a_workflow_that_runs_itself_reports_the_nodes_that_do():
    # A run loads every sub-graph the workflow reaches, so this is how a
    # workflow saved before the check is refused.
    graph = _workflow(
        "outer",
        _runs("middle", "outer-runs-middle"),
        _runs("leaf", "outer-runs-leaf"),
        sub_graphs=[
            _sub_graph("middle", _runs("outer", "middle-runs-outer")),
            _sub_graph("leaf"),
        ],
    )

    errors = graph.validate_graph_get_errors(for_run=True)

    assert errors == {
        "outer-runs-middle": {"graph_id": SUB_WORKFLOW_RUNS_ITSELF},
        "middle-runs-outer": {"graph_id": SUB_WORKFLOW_RUNS_ITSELF},
    }


def test_sub_workflows_that_share_a_workflow_are_not_a_cycle():
    graph = _workflow(
        "outer",
        _runs("left", "outer-runs-left"),
        _runs("right", "outer-runs-right"),
        _runs("shared", "outer-runs-shared"),
        _runs("shared", "outer-runs-shared-again"),
        sub_graphs=[
            _sub_graph("left", _runs("shared", "left-runs-shared")),
            _sub_graph("right", _runs("shared", "right-runs-shared")),
            _sub_graph("shared"),
        ],
    )

    graph.validate_graph(for_run=False)
    assert graph.validate_graph_get_errors(for_run=True) == {}


def test_versions_that_run_each_other_only_across_versions_are_not_a_cycle():
    # v1 of "a" runs v1 of "b", and v2 of "b" runs v2 of "a"; a run loads one
    # version of each, yet no chain runs a workflow twice
    graph = _workflow(
        "top",
        _runs("a", "top-runs-a", version=1),
        _runs("b", "top-runs-b", version=2),
        sub_graphs=[
            _sub_graph("a", _runs("b", "a-runs-b", version=1), version=1),
            _sub_graph("b", _runs("a", "b-runs-a", version=2), version=2),
        ],
    )

    graph.validate_graph(for_run=False)
    assert graph.validate_graph_get_errors(for_run=True) == {}


def _runs(graph_id: str, node_id: str, version: int = 1) -> Node:
    """A sub-workflow node that runs that version of the workflow `graph_id`."""
    return Node(
        id=node_id,
        block_id=AgentExecutorBlock().id,
        input_default={
            "user_id": "user-1",
            "graph_id": graph_id,
            "graph_version": version,
            "inputs": {},
            "input_schema": {},
            "output_schema": {},
        },
    )


def _sub_graph(graph_id: str, *nodes: Node, version: int = 1) -> BaseGraph:
    return BaseGraph(
        id=graph_id, version=version, name=graph_id, description="", nodes=list(nodes)
    )


def _workflow(
    graph_id: str, *nodes: Node, sub_graphs: list[BaseGraph] | None = None
) -> GraphModel:
    graph = Graph(
        id=graph_id,
        name=graph_id,
        description="",
        nodes=list(nodes),
        sub_graphs=sub_graphs or [],
    )
    return make_graph_model(graph, user_id="user-1")
