"""A read loads a workflow's sub-workflows only up to a bound, measured in the
database before anything is fetched; a save that would pass it is rolled back."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import pytest
from prisma.models import AgentGraph

from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.basic import StoreValueBlock
from backend.data import sub_graph_budget
from backend.data.graph import (
    Graph,
    Link,
    Node,
    create_graph,
    delete_graph,
    get_graph,
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
async def test_a_batch_counts_a_shared_sub_graph_once_and_leaves_out_only_the_graph_past_the_bound(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    async with _saved(_values(3), _values(6)) as (shared, large):
        async with _saved(_runs(shared), _runs(shared), _runs(large)) as roots:
            monkeypatch.setattr(sub_graph_budget, "MAX_SUB_GRAPH_ROWS", 5)
            rows = [
                await AgentGraph.prisma().find_first_or_raise(
                    where={"id": root.id}, include={"Nodes": True}
                )
                for root in roots
            ]

            each = await get_sub_graphs_leaving_out_too_large(rows)

    assert [None if subs is None else [g.id for g in subs] for subs in each] == [
        [shared.id],
        [shared.id],
        None,
    ]


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
