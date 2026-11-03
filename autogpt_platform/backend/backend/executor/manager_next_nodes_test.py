"""A node's output registers the nodes it feeds side by side, a bounded number
at a time, while the links into one node keep their order and the queue still
gets them in link order."""

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


async def _enqueue(
    events: list[str],
    meeting: Rendezvous,
    links: list[Link],
    failing: frozenset[str] = frozenset(),
):
    """Register ``links``; every upsert but a ``late`` pin's meets the others,
    and a ``failing`` pin's upsert raises once it has.

    The stand-ins yield after each write, so a registration that overlapped
    one it should have waited for would interleave its events with it."""

    async def get_node(node_id: str) -> Node:
        events.append(f"get {node_id}")
        return Node(id=node_id, block_id=f"block-{node_id}")

    async def upsert(node_id, graph_exec_id, input_name, input_data):
        events.append(f"start {node_id}:{input_name}")
        if not input_name.startswith("late"):
            await meeting.arrive()
        if input_name in failing:
            raise RuntimeError(f"upsert failed for {input_name}")
        events.append(f"end {node_id}:{input_name}")
        await asyncio.sleep(0)
        return MagicMock(node_exec_id=f"{node_id}:{input_name}"), {
            input_name: input_data
        }

    async def mark_queued(exec_id, status, execution_data=None, stats=None):
        events.append(f"queued {exec_id}")
        await asyncio.sleep(0)
        return MagicMock()

    db = MagicMock()
    db.get_node = AsyncMock(side_effect=get_node)
    db.upsert_execution_input = AsyncMock(side_effect=upsert)
    db.update_node_execution_status = AsyncMock(side_effect=mark_queued)
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


def _most_upserts_at_once(events: list[str]) -> int:
    in_flight = most = 0
    for event in events:
        if event.startswith("start "):
            in_flight += 1
            most = max(most, in_flight)
        elif event.startswith("end "):
            in_flight -= 1
    return most


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
async def test_a_node_that_fails_to_register_leaves_the_others_queued(
    enqueue_stand_ins: None,
) -> None:
    """``b``'s upsert raises while ``a`` and ``c`` are in flight beside it."""
    entries = await _enqueue(
        [],
        Rendezvous(3),
        [_link("a", "x"), _link("b", "y"), _link("c", "z")],
        failing=frozenset({"y"}),
    )

    assert [entry.node_exec_id for entry in entries] == ["a:x", "c:z"]


@pytest.mark.asyncio
async def test_links_into_one_node_register_one_after_another_in_order(
    enqueue_stand_ins: None,
) -> None:
    """``a`` is fed twice: its second link starts only once its first is
    queued, while ``b`` registers beside the first."""
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
    assert second_read_of_a > events.index("queued a:first")


@pytest.mark.asyncio
async def test_an_output_registers_at_most_the_cap_of_its_nodes_at_once(
    enqueue_stand_ins: None, mocker: MockerFixture
) -> None:
    """With room for two, four nodes register two by two, and in link order."""
    mocker.patch.object(manager, "MAX_SINKS_REGISTERED_AT_ONCE", 2)
    events: list[str] = []

    entries = await _enqueue(
        events,
        Rendezvous(2),
        [_link("a", "w"), _link("b", "x"), _link("c", "y"), _link("d", "z")],
    )

    assert [entry.node_exec_id for entry in entries] == ["a:w", "b:x", "c:y", "d:z"]
    assert _most_upserts_at_once(events) == 2
