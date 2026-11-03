"""A node's output registers the nodes it feeds side by side, while the links
into one node keep their order and the queue still gets them in link order."""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.data.execution import ExecutionContext
from backend.data.graph import Link, Node
from backend.executor import manager
from backend.util.testing import Rendezvous


@pytest.fixture
def enqueue_stand_ins(mocker: MockerFixture) -> None:
    """The node lock is Redis; a per-key asyncio lock keeps its exclusion."""
    locks: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def synchronized(key: str, timeout: int = 0):
        async with locks.setdefault(key, asyncio.Lock()):
            yield

    mocker.patch.object(manager, "synchronized", synchronized)
    mocker.patch.object(manager, "send_async_execution_update", AsyncMock())
    mocker.patch.object(
        manager, "validate_exec", lambda node, data, dry_run=False: (data, "complete")
    )


def _link(sink_id: str, sink_name: str) -> Link:
    return Link(
        source_id="source", sink_id=sink_id, source_name="result", sink_name=sink_name
    )


async def _enqueue(events: list[str], meeting: Rendezvous, links: list[Link]):
    """Register ``links``; every upsert but a ``late`` pin's meets the others."""

    async def get_node(node_id: str) -> Node:
        events.append(f"get {node_id}")
        return Node(id=node_id, block_id=f"block-{node_id}")

    async def upsert(node_id, graph_exec_id, input_name, input_data):
        events.append(f"start {node_id}:{input_name}")
        if not input_name.startswith("late"):
            await meeting.arrive()
        events.append(f"end {node_id}:{input_name}")
        return MagicMock(node_exec_id=f"{node_id}:{input_name}"), {
            input_name: input_data
        }

    db = MagicMock()
    db.get_node = AsyncMock(side_effect=get_node)
    db.upsert_execution_input = AsyncMock(side_effect=upsert)
    db.update_node_execution_status = AsyncMock(return_value=MagicMock())
    return await manager._enqueue_next_nodes(
        db_client=db,
        node=Node(id="source", block_id="block-source", output_links=links),
        output=("result", "hello"),
        user_id="user-1",
        graph_exec_id="run-1",
        graph_id="graph-1",
        graph_version=1,
        log_metadata=MagicMock(),
        nodes_input_masks=None,
        execution_context=ExecutionContext(),
    )


@pytest.mark.asyncio
async def test_links_into_different_nodes_register_side_by_side(
    enqueue_stand_ins: None,
) -> None:
    entries = await _enqueue(
        [], Rendezvous(3), [_link("a", "x"), _link("b", "y"), _link("c", "z")]
    )

    assert [entry.node_exec_id for entry in entries] == ["a:x", "b:y", "c:z"]
    assert [entry.inputs for entry in entries] == [
        {"x": "hello"},
        {"y": "hello"},
        {"z": "hello"},
    ]


@pytest.mark.asyncio
async def test_links_into_one_node_register_one_after_another_in_order(
    enqueue_stand_ins: None,
) -> None:
    """``a`` is fed twice: its second link starts only once its first is done,
    while ``b`` registers beside the first."""
    events: list[str] = []

    entries = await _enqueue(
        events,
        Rendezvous(2),
        [_link("a", "first"), _link("b", "other"), _link("a", "late")],
    )

    assert [entry.node_exec_id for entry in entries] == [
        "a:first",
        "b:other",
        "a:late",
    ]
    second_read_of_a = [i for i, event in enumerate(events) if event == "get a"][1]
    assert second_read_of_a > events.index("end a:first")
