"""A read loads a workflow's sub-workflows only up to a bound, measured in the
database before anything is fetched; a save that would pass it is rolled back."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from prisma.models import AgentGraph

import backend.data.graph as graph_db
from backend.api.features.library import _get_many
from backend.api.features.library import db as library_db
from backend.api.model import CreateGraph
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.basic import StoreValueBlock
from backend.data import sub_graph_budget
from backend.data.graph import (
    Graph,
    Link,
    Node,
    create_graph,
    delete_graph,
    foreign_sub_graph_refs,
    get_graph,
    get_graph_all_versions,
    get_graph_as_admin,
    get_sub_graphs_leaving_out_too_large,
)
from backend.data.sub_graph_budget import get_sub_graph_sizes
from backend.data.user import DEFAULT_USER_ID
from backend.util.exceptions import SubGraphsTooLargeError
from backend.util.test import SpinTestServer


@pytest.mark.asyncio(loop_scope="session")
async def test_a_stored_graph_is_measured_in_the_database(server: SpinTestServer):
    graph = _values(3, linked=True)
    graph.nodes[0].input_default = {"input": "x" * 10_000}
    async with _saved(graph) as (graph,):
        sizes = await get_sub_graph_sizes(
            [
                (graph.id, graph.version, DEFAULT_USER_ID),
                (graph.id, graph.version, "someone-else"),
            ]
        )

    [(ref, size)] = sizes.items()
    assert ref == (graph.id, graph.version, DEFAULT_USER_ID)
    assert (size.nodes, size.links) == (3, 2)
    assert size.bytes >= 10_000


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "limit, value",
    [("MAX_SUB_GRAPHS", 1), ("MAX_SUB_GRAPH_ROWS", 5), ("MAX_SUB_GRAPH_BYTES", 1)],
)
async def test_a_read_stops_before_loading_sub_graphs_past_the_bound(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch, limit: str, value: int
):
    async with _saved(_values(3), _values(3)) as (left, right):
        async with _saved(_runs(left, right)) as (parent,):
            loaded = await get_graph(
                parent.id,
                parent.version,
                user_id=DEFAULT_USER_ID,
                include_subgraphs=True,
            )
            assert loaded and {g.id for g in loaded.sub_graphs} == {left.id, right.id}

            monkeypatch.setattr(sub_graph_budget, limit, value)
            with pytest.raises(SubGraphsTooLargeError):
                await get_graph(
                    parent.id,
                    parent.version,
                    user_id=DEFAULT_USER_ID,
                    include_subgraphs=True,
                )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_links_pin_names_count_toward_the_byte_bound(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    pin = "tools_^_" + "x" * 50_000
    sub_graph = _values(2, linked=True)
    sub_graph.links[0].sink_name = pin
    async with _saved(sub_graph) as (sub_graph,):
        async with _saved(_runs(sub_graph)) as (parent,):
            [size] = (
                await get_sub_graph_sizes(
                    [(sub_graph.id, sub_graph.version, DEFAULT_USER_ID)]
                )
            ).values()
            assert size.bytes >= 2 * len(pin)

            monkeypatch.setattr(sub_graph_budget, "MAX_SUB_GRAPH_BYTES", 2 * len(pin))
            with pytest.raises(SubGraphsTooLargeError):
                await get_graph(
                    parent.id,
                    parent.version,
                    user_id=DEFAULT_USER_ID,
                    include_subgraphs=True,
                )


@pytest.mark.asyncio(loop_scope="session")
async def test_a_save_past_the_bound_stores_nothing(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    async with _saved(_values(3), _values(3)) as (left, right):
        monkeypatch.setattr(sub_graph_budget, "MAX_SUB_GRAPH_ROWS", 5)
        parent = _runs(left, right)

        with pytest.raises(SubGraphsTooLargeError):
            await create_graph(parent, DEFAULT_USER_ID)

        assert await AgentGraph.prisma().count(where={"id": parent.id}) == 0


@pytest.mark.asyncio(loop_scope="session")
async def test_a_level_naming_more_sub_graphs_than_the_bound_is_never_sized(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    async with _saved(_values(1), _values(1)) as (left, right):
        async with _saved(_runs(left, right)) as (parent,):
            monkeypatch.setattr(sub_graph_budget, "MAX_SUB_GRAPHS", 1)
            sizing = AsyncMock(side_effect=AssertionError("sized"))
            monkeypatch.setattr(graph_db, "get_sub_graph_sizes", sizing)

            with pytest.raises(SubGraphsTooLargeError):
                await get_graph(
                    parent.id,
                    parent.version,
                    user_id=DEFAULT_USER_ID,
                    include_subgraphs=True,
                )
    sizing.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_sizing_that_outlasts_its_timeout_refuses_the_read(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    slow = f"SELECT s.* FROM ({sub_graph_budget._SIZES_QUERY}) s, pg_sleep(2)"
    monkeypatch.setattr(sub_graph_budget, "_SIZES_QUERY", slow)
    monkeypatch.setattr(sub_graph_budget, "SIZE_QUERY_TIMEOUT_MS", 100)
    async with _saved(_values(1)) as (graph,):
        with pytest.raises(SubGraphsTooLargeError):
            await get_sub_graph_sizes([(graph.id, graph.version, DEFAULT_USER_ID)])


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "limit, value",
    [("MAX_SUB_GRAPHS", 1), ("MAX_SUB_GRAPH_ROWS", 5), ("MAX_SUB_GRAPH_BYTES", 1)],
)
async def test_the_marketplace_sub_graph_check_stops_past_the_bound(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch, limit: str, value: int
):
    async with _saved(_values(3), _values(3)) as (left, right):
        async with _saved(_runs(left, right)) as (parent,):
            row = await AgentGraph.prisma().find_first_or_raise(
                where={"id": parent.id}, include={"Nodes": True}
            )
            assert await foreign_sub_graph_refs(row) == []

            monkeypatch.setattr(sub_graph_budget, limit, value)
            with pytest.raises(SubGraphsTooLargeError):
                await foreign_sub_graph_refs(row)


@pytest.mark.asyncio(loop_scope="session")
async def test_an_update_past_the_bound_keeps_the_previous_version(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    async with _saved(_values(2), _values(6)) as (small, large):
        async with _saved(_runs(small)) as (workflow,):
            monkeypatch.setattr(sub_graph_budget, "MAX_SUB_GRAPH_ROWS", 5)
            update = _runs(large)
            update.id = workflow.id

            with pytest.raises(SubGraphsTooLargeError):
                await create_graph(update, DEFAULT_USER_ID)

            versions = await AgentGraph.prisma().find_many(where={"id": workflow.id})
            assert [(v.version, v.isActive) for v in versions] == [(1, True)]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_version_list_bounds_each_version_on_its_own(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    async with _saved(_values(4), _values(4)) as (first_sub, second_sub):
        async with _saved(_runs(first_sub)) as (workflow,):
            second = _runs(second_sub)
            second.id = workflow.id
            await create_graph(second, DEFAULT_USER_ID)
            monkeypatch.setattr(sub_graph_budget, "MAX_SUB_GRAPH_ROWS", 5)

            versions = await get_graph_all_versions(
                workflow.id, DEFAULT_USER_ID, include_subgraphs=True
            )

    assert [[g.id for g in v.sub_graphs] for v in versions] == [
        [second_sub.id],
        [first_sub.id],
    ]


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("newest_is_large", [True, False])
async def test_an_unversioned_read_bounds_the_version_it_returns(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch, newest_is_large: bool
):
    async with _saved(_values(2), _values(6)) as (small, large):
        active_sub, newest_sub = (small, large) if newest_is_large else (large, small)
        async with _saved(_runs(active_sub)) as (workflow,):
            newest = _runs(newest_sub)
            newest.id, newest.is_active = workflow.id, False
            await create_graph(newest, DEFAULT_USER_ID)
            monkeypatch.setattr(sub_graph_budget, "MAX_SUB_GRAPH_ROWS", 5)

            for read in (
                lambda: get_graph(
                    workflow.id, None, user_id=DEFAULT_USER_ID, include_subgraphs=True
                ),
                lambda: get_graph_as_admin(workflow.id, for_export=True),
            ):
                if newest_is_large:
                    loaded = await read()
                    assert loaded and loaded.version == 1
                    assert [g.id for g in loaded.sub_graphs] == [small.id]
                else:
                    with pytest.raises(SubGraphsTooLargeError):
                        await read()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_batch_leaves_out_only_the_graph_past_the_bound(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    async with _saved(_values(3), _values(6)) as (shared, large):
        async with _saved(_runs(shared), _runs(shared), _runs(large)) as roots:
            monkeypatch.setattr(sub_graph_budget, "MAX_SUB_GRAPH_ROWS", 5)
            each = await get_sub_graphs_leaving_out_too_large(
                [await _with_nodes(root) for root in roots]
            )

    assert [None if subs is None else [g.id for g in subs] for subs in each] == [
        [shared.id],
        [shared.id],
        None,
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_graph_left_out_of_a_batch_takes_nothing_from_the_others(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    # Each root takes one row at the first level, then the left-out one would
    # need 6 more and the other 4: only a bound per root lets the other load.
    async with _saved(_values(5), _values(4)) as (deep, kept):
        async with _saved(_runs(deep), _runs(kept)) as (middle, other_middle):
            async with _saved(_runs(middle), _runs(other_middle)) as roots:
                monkeypatch.setattr(sub_graph_budget, "MAX_SUB_GRAPH_ROWS", 5)
                each = await get_sub_graphs_leaving_out_too_large(
                    [await _with_nodes(root) for root in roots]
                )

    assert each[0] is None
    assert each[1] is not None
    assert {g.id for g in each[1]} == {other_middle.id, kept.id}


@pytest.mark.asyncio(loop_scope="session")
async def test_a_library_batch_leaves_out_only_the_card_past_the_bound(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    async with _saved(_values(2), _values(6)) as (small, large):
        ok, too_large = [
            await _in_library(server, _runs(sub)) for sub in (small, large)
        ]
        monkeypatch.setattr(sub_graph_budget, "MAX_SUB_GRAPH_ROWS", 5)

        batch = await _get_many.get_library_agents_by_ids(
            [ok, too_large], DEFAULT_USER_ID
        )

    assert [agent.id for agent in batch] == [ok]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_library_batch_whose_walk_fails_still_loads_card_by_card(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    walk = graph_db.get_sub_graphs_leaving_out_too_large

    async def fails_for_a_whole_batch(graphs, client=None):
        if len(graphs) > 1:
            raise RuntimeError("batch walk failed")
        return await walk(graphs, client)

    async with _saved(_values(2)) as (small,):
        cards = [await _in_library(server, _runs(small)) for _ in range(2)]
        monkeypatch.setattr(
            graph_db, "get_sub_graphs_leaving_out_too_large", fails_for_a_whole_batch
        )

        batch = await _get_many.get_library_agents_by_ids(cards, DEFAULT_USER_ID)

    assert [agent.id for agent in batch] == cards


async def _with_nodes(graph: Graph) -> AgentGraph:
    return await AgentGraph.prisma().find_first_or_raise(
        where={"id": graph.id}, include={"Nodes": True}
    )


async def _in_library(server: SpinTestServer, graph: Graph) -> str:
    """Save *graph* the way the API does, which also adds it to the library,
    and return its library agent's id."""
    saved = await server.agent_server.test_create_graph(
        CreateGraph(graph=graph), DEFAULT_USER_ID
    )
    agent = await library_db.get_library_agent_by_graph_id(DEFAULT_USER_ID, saved.id)
    assert agent is not None
    return agent.id


def _values(count: int, *, linked: bool = False) -> Graph:
    nodes = [Node(id=str(uuid4()), block_id=StoreValueBlock().id) for _ in range(count)]
    links = (
        [
            Link(source_id=a.id, sink_id=b.id, source_name="output", sink_name="input")
            for a, b in zip(nodes, nodes[1:])
        ]
        if linked
        else []
    )
    return Graph(
        id=str(uuid4()), name="Values", description="", nodes=nodes, links=links
    )


def _runs(*sub_graphs: Graph) -> Graph:
    return Graph(
        id=str(uuid4()),
        name="Runs others",
        description="",
        nodes=[
            Node(
                id=str(uuid4()),
                block_id=AgentExecutorBlock().id,
                input_default={
                    "user_id": DEFAULT_USER_ID,
                    "graph_id": sub_graph.id,
                    "graph_version": sub_graph.version,
                    "inputs": {},
                    "input_schema": {},
                    "output_schema": {},
                },
            )
            for sub_graph in sub_graphs
        ],
    )


@asynccontextmanager
async def _saved(*graphs: Graph) -> AsyncIterator[tuple[Graph, ...]]:
    for graph in graphs:
        await create_graph(graph, DEFAULT_USER_ID)
    try:
        yield graphs
    finally:
        for graph in graphs:
            await delete_graph(graph.id, user_id=DEFAULT_USER_ID)
