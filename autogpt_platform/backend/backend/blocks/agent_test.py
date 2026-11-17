"""The sub-agent block waits on its child run with no timeout, and pub/sub
keeps no history: whatever the stream misses, the block must still finish with
every output, in the order the child produced them, and must not hang."""

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

import backend.blocks.agent as agent
import backend.data.event_bus as event_bus
import backend.data.redis_client as redis_client
from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.io import AgentOutputBlock
from backend.data.execution import (
    AsyncRedisExecutionEventBus,
    ExecutionEvent,
    ExecutionStatus,
    GraphExecutionEvent,
    GraphExecutionMeta,
    NodeExecutionEvent,
    exec_channel,
)
from backend.util.testing import is_tcp_port_reachable, redis_shard_proxies

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
_USER, _GRAPH, _CHILD = "user-1", "graph-1", "child-exec-1"
_WITHIN_S = 8.0


def _output(n: int, value: str) -> NodeExecutionEvent:
    """The child's output block ``n`` emitting ``value``, finished at ``n`` s."""
    return NodeExecutionEvent(
        user_id=_USER,
        graph_id=_GRAPH,
        graph_version=1,
        graph_exec_id=_CHILD,
        node_exec_id=f"node-exec-{n}",
        node_id=f"node-{n}",
        block_id=AgentOutputBlock().id,
        status=ExecutionStatus.COMPLETED,
        input_data={"name": "result"},
        output_data={"output": [value]},
        add_time=_T0 + timedelta(seconds=n),
        queue_time=_T0 + timedelta(seconds=n),
        start_time=_T0 + timedelta(seconds=n),
        end_time=_T0 + timedelta(seconds=n),
    )


def _meta(status: ExecutionStatus) -> GraphExecutionMeta:
    return GraphExecutionMeta(
        id=_CHILD,
        user_id=_USER,
        graph_id=_GRAPH,
        graph_version=1,
        inputs=None,
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=status,
        started_at=_T0,
        ended_at=_T0,
        stats=GraphExecutionMeta.Stats(cost=7, node_exec_count=3),
    )


def _finished_event() -> GraphExecutionEvent:
    meta = _meta(ExecutionStatus.COMPLETED).model_dump()
    return GraphExecutionEvent(**{**meta, "inputs": {}, "outputs": {}})


class _Stream:
    """Stands in for the execution event bus; the test feeds what its
    ``listen_with_resync`` yields (``None`` marks a confirmed subscription)."""

    def __init__(self) -> None:
        self.items: asyncio.Queue[ExecutionEvent | None] = asyncio.Queue()

    async def listen_with_resync(
        self, channel_key: str
    ) -> AsyncIterator[ExecutionEvent | None]:
        while True:
            yield await self.items.get()


def _db(
    *metas: GraphExecutionMeta | Exception, nodes: list, later_nodes: list | None = None
) -> MagicMock:
    """The database: each catch-up reads the next of ``metas`` (the last one
    repeats), then the child's finished node executions, newest first as
    ``get_node_executions`` orders them: ``nodes`` on the first read,
    ``later_nodes`` (if given) after that."""
    db = MagicMock()
    metas_left = list(metas)
    node_reads = [nodes] if later_nodes is None else [nodes, later_nodes]

    async def get_meta(**_kwargs: Any) -> GraphExecutionMeta:
        answer = metas_left.pop(0) if len(metas_left) > 1 else metas_left[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def get_nodes(**_kwargs: Any) -> list:
        visible = node_reads.pop(0) if len(node_reads) > 1 else node_reads[0]
        return sorted(visible, key=lambda node: node.end_time, reverse=True)

    db.get_graph_execution_meta = AsyncMock(side_effect=get_meta)
    db.get_node_executions = AsyncMock(side_effect=get_nodes)
    return db


async def _run(
    monkeypatch: pytest.MonkeyPatch, bus: object, db: MagicMock
) -> tuple[list[tuple[str, Any]], AgentExecutorBlock]:
    monkeypatch.setattr(
        "backend.executor.utils.get_async_execution_event_bus", lambda: bus
    )
    monkeypatch.setattr(agent, "execution_db", lambda: db)
    block = AgentExecutorBlock()
    outputs: list[tuple[str, Any]] = []
    async with asyncio.timeout(_WITHIN_S):
        async for output in block._run(
            graph_id=_GRAPH,
            graph_version=1,
            graph_exec_id=_CHILD,
            user_id=_USER,
            logger=MagicMock(),
        ):
            outputs.append(output)
    return outputs, block


async def test_a_child_that_ended_before_the_subscription_is_caught_up_in_order(
    monkeypatch: pytest.MonkeyPatch,
):
    stream = _Stream()
    stream.items.put_nowait(None)
    db = _db(_meta(ExecutionStatus.COMPLETED), nodes=[_output(1, "a"), _output(2, "b")])

    outputs, block = await _run(monkeypatch, stream, db)

    assert outputs == [("result", "a"), ("result", "b")]
    # Only output blocks matter: a big child's other nodes aren't loaded.
    read = db.get_node_executions.await_args
    assert read is not None and read.kwargs["block_ids"] == [AgentOutputBlock().id]
    assert block.execution_stats.reconciled_cost_delta == 7
    assert block.execution_stats.extra_steps == 3


async def test_a_lost_stream_is_caught_up_in_order_without_repeats(
    monkeypatch: pytest.MonkeyPatch,
):
    stream = _Stream()
    a, b, c = _output(1, "a"), _output(2, "b"), _output(3, "c")
    for item in (None, a, None, c, _finished_event()):
        stream.items.put_nowait(item)
    # b was published while the stream was down: only the resync's catch-up
    # can find it.
    db = _db(
        _meta(ExecutionStatus.RUNNING),
        _meta(ExecutionStatus.RUNNING),
        nodes=[],
        later_nodes=[a, b],
    )

    outputs, _ = await _run(monkeypatch, stream, db)

    assert outputs == [("result", "a"), ("result", "b"), ("result", "c")]


async def test_a_newer_output_waits_for_a_failed_catch_up_to_be_retried(
    monkeypatch: pytest.MonkeyPatch,
):
    """b was missed and the resync's catch-up failed; c then arrives live and
    must not overtake b."""
    stream = _Stream()
    a, b, c = _output(1, "a"), _output(2, "b"), _output(3, "c")
    for item in (None, a, None, c, _finished_event()):
        stream.items.put_nowait(item)
    db = _db(
        _meta(ExecutionStatus.RUNNING),
        ConnectionError("database manager unreachable"),
        _meta(ExecutionStatus.RUNNING),
        nodes=[],
        later_nodes=[a, b, c],
    )

    outputs, _ = await _run(monkeypatch, stream, db)

    assert outputs == [("result", "a"), ("result", "b"), ("result", "c")]


async def test_a_failed_catch_up_does_not_end_the_wait(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(agent, "_QUIET_RECHECK_S", 0.05)
    stream = _Stream()
    stream.items.put_nowait(None)
    db = _db(
        ConnectionError("database manager unreachable"),
        _meta(ExecutionStatus.COMPLETED),
        nodes=[_output(1, "a")],
    )

    outputs, _ = await _run(monkeypatch, stream, db)

    assert outputs == [("result", "a")]


async def test_an_end_that_was_never_published_is_found_by_the_quiet_recheck(
    monkeypatch: pytest.MonkeyPatch,
):
    """The stream is healthy, but the child's executor failed to publish its
    end: only asking the database ends the wait."""
    monkeypatch.setattr(agent, "_QUIET_RECHECK_S", 0.05)
    stream = _Stream()
    stream.items.put_nowait(None)
    db = _db(
        _meta(ExecutionStatus.RUNNING),
        _meta(ExecutionStatus.COMPLETED),
        nodes=[_output(1, "a")],
    )

    outputs, _ = await _run(monkeypatch, stream, db)

    assert outputs == [("result", "a")]


@pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="local redis cluster not reachable; skip sub-agent over real Redis",
)
async def test_a_child_that_ends_while_redis_is_dropped_still_ends_the_wait(
    monkeypatch: pytest.MonkeyPatch,
):
    """End to end over the real Redis: the child's last output and its end are
    published while the block's connection is down, so both are lost."""
    monkeypatch.setattr(event_bus, "RESUBSCRIBE_DELAY_S", 0.05)
    child = f"child-{uuid4()}"
    a, b = _output(1, "a"), _output(2, "b")
    for node in (a, b):
        node.graph_exec_id = child
    # Nothing in the database yet, so ``a`` can only arrive over the stream.
    db = _db(_meta(ExecutionStatus.RUNNING), nodes=[])
    publisher = AsyncRedisExecutionEventBus()
    channel = exec_channel(_USER, _GRAPH, child)
    outputs: list[tuple[str, Any]] = []

    async with redis_shard_proxies() as route:
        proxy = await route(f"{publisher.event_bus_name}/{channel}")
        monkeypatch.setattr(
            "backend.executor.utils.get_async_execution_event_bus",
            lambda: AsyncRedisExecutionEventBus(),
        )
        monkeypatch.setattr(agent, "execution_db", lambda: db)

        async def run_block() -> None:
            async for output in AgentExecutorBlock()._run(
                graph_id=_GRAPH,
                graph_version=1,
                graph_exec_id=child,
                user_id=_USER,
                logger=MagicMock(),
            ):
                outputs.append(output)

        block = asyncio.create_task(run_block())
        try:
            async with asyncio.timeout(_WITHIN_S):
                while not outputs:
                    await publisher.publish_event(a, channel)
                    await asyncio.sleep(0.1)
            proxy.drop()
            db.get_node_executions.side_effect = None
            db.get_node_executions.return_value = [b, a]
            db.get_graph_execution_meta.side_effect = None
            db.get_graph_execution_meta.return_value = _meta(ExecutionStatus.COMPLETED)
            await publisher.publish_event(b, channel)
            await publisher.publish_event(_finished_event(), channel)
            async with asyncio.timeout(_WITHIN_S):
                await block
        finally:
            block.cancel()
            await asyncio.gather(block, return_exceptions=True)
            await redis_client.disconnect_async()

    assert outputs == [("result", "a"), ("result", "b")]
