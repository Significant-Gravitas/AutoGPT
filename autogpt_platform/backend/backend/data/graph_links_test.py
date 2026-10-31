import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import prisma
import pytest
from prisma.enums import ResourceVisibility
from prisma.models import AgentGraph, AgentNode, AgentNodeLink

from backend.api.model import CreateGraph
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.basic import StoreValueBlock
from backend.data.graph import (
    Graph,
    GraphModel,
    Link,
    Node,
    _sub_graph_refs,
    get_connected_output_nodes,
    get_graph,
)
from backend.data.user import DEFAULT_USER_ID
from backend.util.test import SpinTestServer


def _graph_row(nodes: list[AgentNode]) -> AgentGraph:
    return AgentGraph(
        id="g-1",
        version=1,
        name="Links",
        description="",
        userId="owner",
        isActive=True,
        createdAt=datetime.now(timezone.utc),
        visibility=ResourceVisibility.PRIVATE,
        Nodes=nodes,
    )


def _node_row(node_id: str, **kwargs) -> AgentNode:
    return AgentNode(
        id=node_id,
        agentBlockId=StoreValueBlock().id,
        agentGraphId="g-1",
        agentGraphVersion=1,
        constantInput=json.dumps(kwargs.pop("constant_input", {})),
        metadata="{}",
        **kwargs,
    )


def _link_rows(count: int) -> list[AgentNodeLink]:
    return [
        AgentNodeLink(
            id=f"link-{i}",
            agentNodeSourceId="a",
            sourceName="output",
            agentNodeSinkId="b",
            sinkName="input",
            isStatic=False,
        )
        for i in range(count)
    ]


def test_from_db_keeps_one_link_per_pair_of_ends():
    links = _link_rows(5)
    row = _graph_row([_node_row("a", Output=links), _node_row("b", Input=links)])

    graph = GraphModel.from_db(row)

    assert [link.id for link in graph.links] == ["link-0"]


def test_from_db_does_not_compare_links_pairwise():
    """Loading `n` links with the same ends takes a number of link comparisons
    that grows with `n`, not with `n` squared."""
    count = 300
    links = _link_rows(count)
    row = _graph_row([_node_row("a", Output=links), _node_row("b", Input=links)])
    comparisons = 0
    compare = Link.__eq__

    def counting_eq(self, other):
        nonlocal comparisons
        comparisons += 1
        return compare(self, other)

    with patch.object(Link, "__eq__", counting_eq):
        GraphModel.from_db(row)

    assert comparisons <= 2 * count


def test_sub_graph_refs_read_the_block_id_column():
    """Nodes are loaded without their block row; the column has its id."""
    node = _node_row(
        "agent",
        constant_input={"graph_id": "sub-1", "graph_version": 3},
    )
    node.agentBlockId = AgentExecutorBlock().id
    assert node.AgentBlock is None

    assert _sub_graph_refs(_graph_row([node]), AgentExecutorBlock().id) == [
        ("sub-1", 3)
    ]


@pytest.mark.asyncio
async def test_get_graph_loads_nodes_without_their_block_row(mocker):
    graph_client = AsyncMock()
    graph_client.find_first.return_value = _graph_row([_node_row("a")])
    mocker.patch.object(prisma.models.AgentGraph, "prisma", return_value=graph_client)

    graph = await get_graph("g-1", 1, user_id="owner")

    assert graph is not None
    include = graph_client.find_first.call_args.kwargs["include"]
    assert "AgentBlock" not in include["Nodes"]["include"]


def _value_chain(links: list[Link]) -> Graph:
    return Graph(
        name="Repeated links",
        description="",
        nodes=[
            Node(id="a", block_id=StoreValueBlock().id, input_default={"input": 1}),
            Node(id="b", block_id=StoreValueBlock().id),
        ],
        links=links,
    )


def _a_to_b() -> Link:
    return Link(source_id="a", sink_id="b", source_name="output", sink_name="input")


@pytest.mark.asyncio(loop_scope="session")
async def test_saving_a_graph_keeps_one_link_per_pair_of_ends(server: SpinTestServer):
    other = Link(source_id="a", sink_id="b", source_name="output", sink_name="data")
    graph = _value_chain([_a_to_b(), _a_to_b(), _a_to_b(), other])

    created = await server.agent_server.test_create_graph(
        CreateGraph(graph=graph), DEFAULT_USER_ID
    )
    assert sorted(link.sink_name for link in created.links) == ["data", "input"]
    source_id = created.links[0].source_id
    assert (
        await AgentNodeLink.prisma().count(where={"agentNodeSourceId": source_id}) == 2
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_connected_output_nodes_return_each_link_and_sink_once(
    server: SpinTestServer,
):
    created = await server.agent_server.test_create_graph(
        CreateGraph(graph=_value_chain([_a_to_b()])), DEFAULT_USER_ID
    )
    [link] = created.links
    # Rows saved before links were de-duplicated on save.
    await AgentNodeLink.prisma().create_many(
        data=[
            {
                "agentNodeSourceId": link.source_id,
                "sourceName": link.source_name,
                "agentNodeSinkId": link.sink_id,
                "sinkName": link.sink_name,
            }
            for _ in range(20)
        ]
    )

    connected = await get_connected_output_nodes(link.source_id)
    loaded = await get_graph(created.id, created.version, DEFAULT_USER_ID)

    [(connected_link, sink)] = connected
    assert connected_link.ends == link.ends
    assert sink.id == link.sink_id
    assert sink.input_links == [] and sink.output_links == []
    assert loaded is not None and len(loaded.links) == 1
