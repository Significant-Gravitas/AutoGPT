"""A graph's metadata text is bounded when it is saved.

`GraphMeta` and `GraphModel` read stored rows through `GraphBaseMeta`, so the
bound cannot live on the model: it would make a row already longer than the
bound unreadable. It lives in the one write every save path reaches, which is
why a route, a fork, a copy and the copilot are all covered by one check.
"""

import pytest

from backend.data import graph as graph_db
from backend.data.graph import GRAPH_TEXT_MAX_LENGTHS, Graph

# Module-level, so the name is not mangled by the class-scope rule.
_create_graph = vars(graph_db)["__create_graph"]


def _graph(**overrides) -> Graph:
    fields: dict = {
        "name": "a graph",
        "description": "what it does",
        "nodes": [],
        "links": [],
    }
    fields.update(overrides)
    return Graph.model_validate(fields)


@pytest.mark.asyncio
@pytest.mark.parametrize("field_name", sorted(GRAPH_TEXT_MAX_LENGTHS))
async def test_a_save_refuses_metadata_over_the_bound(field_name: str):
    max_length = GRAPH_TEXT_MAX_LENGTHS[field_name]
    graph = _graph(**{field_name: "x" * (max_length + 1)})

    # `tx=None` is enough: the check runs before the first query, which is the
    # point — no oversized row reaches the database on any save path.
    with pytest.raises(ValueError, match=f"Graph {field_name} is too long"):
        await _create_graph(None, graph, "user-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("field_name", sorted(GRAPH_TEXT_MAX_LENGTHS))
async def test_a_save_accepts_metadata_at_the_bound(field_name: str):
    max_length = GRAPH_TEXT_MAX_LENGTHS[field_name]
    graph = _graph(**{field_name: "x" * max_length})

    # It goes on to fail on the absent transaction, in one of several ways
    # depending on whether a neighbouring test left the client connected. What
    # this test reads is only that the failure is not the refusal.
    with pytest.raises(Exception) as caught:
        await _create_graph(None, graph, "user-1")

    assert "is too long" not in str(caught.value)


@pytest.mark.asyncio
async def test_a_sub_graph_is_bounded_too():
    sub = _graph(name="x" * (GRAPH_TEXT_MAX_LENGTHS["name"] + 1))
    graph = _graph(sub_graphs=[sub])

    with pytest.raises(ValueError, match="Graph name is too long"):
        await _create_graph(None, graph, "user-1")
