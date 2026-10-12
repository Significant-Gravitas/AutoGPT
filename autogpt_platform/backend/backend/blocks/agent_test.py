"""AgentExecutorBlock must finish with its sub-agent even when the sub-agent's
event stream is lost: it has no execution timeout of its own."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks.agent import AgentExecutorBlock
from backend.data.event_bus import EventStreamLostError
from backend.data.execution import ExecutionStatus, NodeExecutionResult

AGENT_OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4"


def _output_node(name: str, value: object) -> NodeExecutionResult:
    now = datetime.now(timezone.utc)
    return NodeExecutionResult(
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        graph_exec_id="exec-1",
        node_exec_id=f"node-exec-{name}",
        node_id=f"node-{name}",
        block_id=AGENT_OUTPUT_BLOCK_ID,
        status=ExecutionStatus.COMPLETED,
        input_data={"name": name},
        output_data={"output": [value]},
        add_time=now,
        queue_time=None,
        start_time=now,
        end_time=now,
    )


class _LostStreamBus:
    """Every listen subscribes, then loses its stream straight away."""

    def __init__(self) -> None:
        self.listens = 0

    async def listen(self, on_subscribed=None, **_):
        self.listens += 1
        if on_subscribed is not None:
            on_subscribed()
        raise EventStreamLostError("connection dropped")
        yield  # pragma: no cover — unreachable


@pytest.mark.asyncio
async def test_finishes_from_the_database_when_the_event_stream_is_lost():
    bus = _LostStreamBus()
    db = MagicMock()
    db.get_graph_execution = AsyncMock(
        return_value=SimpleNamespace(
            status=ExecutionStatus.COMPLETED,
            stats=SimpleNamespace(cost=7, node_exec_count=3),
        )
    )
    db.get_node_executions = AsyncMock(return_value=[_output_node("answer", 42)])
    block = AgentExecutorBlock()

    with (
        patch(
            "backend.executor.utils.get_async_execution_event_bus",
            return_value=bus,
        ),
        patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=db,
        ),
        patch("backend.blocks.agent._RESUBSCRIBE_DELAY_S", 0),
    ):
        outputs = await asyncio.wait_for(
            _collect(block),
            timeout=5,
        )

    assert outputs == [("answer", 42)]
    db.get_graph_execution.assert_awaited_with(user_id="user-1", execution_id="exec-1")
    assert block.execution_stats.extra_steps == 3


@pytest.mark.asyncio
async def test_keeps_waiting_while_the_database_says_it_is_still_running():
    bus = _LostStreamBus()
    db = MagicMock()
    db.get_graph_execution = AsyncMock(
        side_effect=[
            SimpleNamespace(status=ExecutionStatus.RUNNING, stats=None),
            SimpleNamespace(
                status=ExecutionStatus.COMPLETED,
                stats=SimpleNamespace(cost=0, node_exec_count=1),
            ),
        ]
    )
    db.get_node_executions = AsyncMock(
        side_effect=[[], [_output_node("answer", "late")]]
    )

    with (
        patch(
            "backend.executor.utils.get_async_execution_event_bus",
            return_value=bus,
        ),
        patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=db,
        ),
        patch("backend.blocks.agent._RESUBSCRIBE_DELAY_S", 0),
    ):
        outputs = await asyncio.wait_for(_collect(AgentExecutorBlock()), timeout=5)

    assert outputs == [("answer", "late")]
    assert bus.listens >= 2


async def _collect(block: AgentExecutorBlock) -> list[tuple[str, object]]:
    return [
        output
        async for output in block._run(
            graph_id="graph-1",
            graph_version=1,
            graph_exec_id="exec-1",
            user_id="user-1",
            logger=MagicMock(),
        )
    ]


class _BrokenBus:
    async def listen(self, on_subscribed=None, **_):
        raise RuntimeError("listener bug")
        yield  # pragma: no cover — unreachable


@pytest.mark.asyncio
async def test_an_unexpected_listener_error_fails_the_block_instead_of_hiding():
    """Only a lost stream is retried; anything else must surface."""
    with (
        patch(
            "backend.executor.utils.get_async_execution_event_bus",
            return_value=_BrokenBus(),
        ),
        patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=MagicMock(),
        ),
    ):
        with pytest.raises(RuntimeError, match="listener bug"):
            await asyncio.wait_for(_collect(AgentExecutorBlock()), timeout=5)
