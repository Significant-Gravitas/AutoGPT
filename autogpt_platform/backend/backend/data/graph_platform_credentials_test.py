"""The agent's credentials input schema says which fields refuse the platform's
shared keys, as each block's own schema does, so the Library's run form,
schedule dialog and trigger setup don't start such a field on the platform's
key only for graph validation (``_validate_node_input_credentials``) to refuse
the run."""

from typing import Any

import pytest

from backend.blocks._base import Block
from backend.blocks.exa.search import ExaSearchBlock
from backend.blocks.exa.websets import ExaCreateWebsetBlock, ExaPreviewWebsetBlock
from backend.data.graph import BaseGraph, Graph, GraphModel, Node, make_graph_model

FLAG = "refuses_platform_credentials"


@pytest.mark.parametrize(
    "block, inputs, refuses",
    [
        pytest.param(ExaCreateWebsetBlock(), {}, True, id="create-webset"),
        pytest.param(ExaSearchBlock(), {}, False, id="search"),
        pytest.param(
            ExaPreviewWebsetBlock(),
            {"include_sample_items": True},
            True,
            id="preview-with-sample-items",
        ),
        pytest.param(
            ExaPreviewWebsetBlock(),
            {"include_sample_items": False},
            False,
            id="preview-without-sample-items",
        ),
        pytest.param(ExaPreviewWebsetBlock(), {}, False, id="preview-default"),
    ],
)
def test_a_field_is_marked_when_its_step_refuses_platform_keys(
    block: Block, inputs: dict[str, Any], refuses: bool
):
    field = _exa_field(_graph(_node("node-1", block, **inputs)))

    assert field.get(FLAG, False) is refuses
    # Left off, not false, for a field that takes the platform's key.
    assert (FLAG in field) is refuses


def test_a_field_shared_with_a_refusing_step_is_marked():
    # Both steps take an Exa API key, so they share one field, and the
    # platform's key in it would fail the Websets step.
    graph = _graph(
        _node("search", ExaSearchBlock()),
        _node("webset", ExaCreateWebsetBlock()),
    )

    assert _exa_field(graph)[FLAG] is True


@pytest.mark.parametrize(
    "refusing_id, other_id",
    [("a-webset", "z-search"), ("z-webset", "a-search")],
)
def test_a_refusing_step_marks_a_shared_field_wherever_it_sorts(
    refusing_id: str, other_id: str
):
    graph = _graph(
        _node(other_id, ExaSearchBlock()),
        _node(refusing_id, ExaCreateWebsetBlock()),
    )

    assert _exa_field(graph)[FLAG] is True


def test_a_refusing_step_in_a_sub_workflow_marks_the_agents_field():
    graph = _graph(
        _node("search", ExaSearchBlock()),
        sub_graph_nodes=[_node("webset", ExaCreateWebsetBlock())],
    )

    assert _exa_field(graph)[FLAG] is True


def _exa_field(graph: GraphModel) -> dict[str, Any]:
    properties = graph.credentials_input_schema["properties"]
    assert list(properties) == ["exa_api_key_credentials"]
    return properties["exa_api_key_credentials"]


def _node(node_id: str, block: Block, **inputs: Any) -> Node:
    return Node(id=node_id, block_id=block.id, input_default=inputs)


def _graph(*nodes: Node, sub_graph_nodes: list[Node] | None = None) -> GraphModel:
    sub_graphs = (
        [BaseGraph(id="sub-graph", name="Sub", description="", nodes=sub_graph_nodes)]
        if sub_graph_nodes
        else []
    )
    graph = Graph(
        id="graph",
        name="Agent",
        description="",
        nodes=list(nodes),
        sub_graphs=sub_graphs,
    )
    return make_graph_model(graph, user_id="user-a")
