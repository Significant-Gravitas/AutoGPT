"""A graph's metadata text is bounded when it is saved.

`GraphMeta` and `GraphModel` read stored rows through `GraphBaseMeta`, so the
bound lives in the one write every save path reaches — a route, a fork, a copy
and the copilot — rather than on the model.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from backend.data import graph as graph_db
from backend.data.graph import GRAPH_TEXT_MAX_LENGTHS, Graph, GraphModel

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


class _ReachedTheFirstQuery(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("field_name", sorted(GRAPH_TEXT_MAX_LENGTHS))
async def test_a_save_accepts_metadata_at_the_bound(field_name: str, mocker):
    max_length = GRAPH_TEXT_MAX_LENGTHS[field_name]
    graph = _graph(**{field_name: "x" * max_length})
    mocker.patch.object(graph_db, "graph_owner_id", side_effect=_ReachedTheFirstQuery)

    with pytest.raises(_ReachedTheFirstQuery):
        await _create_graph(None, graph, "user-1")


@pytest.mark.asyncio
async def test_a_workflow_named_at_the_bound_can_be_forked(mocker):
    source = _stored(mocker, name="x" * GRAPH_TEXT_MAX_LENGTHS["name"])

    with pytest.raises(_ReachedTheFirstQuery):
        await graph_db.fork_graph("graph-1", 1, "user-1")
    assert source.name.endswith(" (copy)")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "copy_with", [graph_db.fork_graph, graph_db.copy_graph], ids=["fork", "copy"]
)
async def test_a_stored_name_over_the_bound_is_cut_to_fit(copy_with, mocker):
    limit = GRAPH_TEXT_MAX_LENGTHS["name"]
    over = "x" * 500
    sub = {"name": over, "description": "", "nodes": [], "links": []}
    source = _stored(mocker, name=over, sub_graphs=[sub])

    with pytest.raises(_ReachedTheFirstQuery):
        await copy_with("graph-1", 1, "user-1")
    assert source.sub_graphs[0].name == "x" * (limit - 1) + "…"
    assert len(source.name) <= limit


def _stored(mocker, **overrides) -> GraphModel:
    """A stored workflow the fork or copy reads, which stops at its first query."""
    fields: dict = {
        "description": "",
        "nodes": [],
        "links": [],
        "user_id": "user-1",
        "created_at": datetime.now(UTC),
    }
    source = GraphModel.model_validate(fields | overrides)
    mocker.patch.object(graph_db, "get_graph", return_value=source)
    mocker.patch.object(graph_db, "transaction", return_value=AsyncMock())
    mocker.patch.object(graph_db, "graph_owner_id", side_effect=_ReachedTheFirstQuery)
    return source


def test_a_short_name_is_forked_whole():
    assert graph_db.fork_name("a workflow") == "a workflow (copy)"


@pytest.mark.asyncio
async def test_a_sub_graph_is_bounded_too():
    sub = _graph(name="x" * (GRAPH_TEXT_MAX_LENGTHS["name"] + 1))
    graph = _graph(sub_graphs=[sub])

    with pytest.raises(ValueError, match="Graph name is too long"):
        await _create_graph(None, graph, "user-1")
