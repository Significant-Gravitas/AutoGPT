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

import backend.blocks._sub_agent_wait as sub_agent_wait
import backend.data.event_bus as event_bus
import backend.data.redis_client as redis_client
from backend.blocks import get_output_block_ids
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


@pytest.fixture(autouse=True, scope="module")
def _blocks_loaded() -> None:
    """Load the block registry, which takes seconds cold, before any test
    starts its _WITHIN_S bound."""
    get_output_block_ids()


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
    *metas: GraphExecutionMeta | Exception | None,
    nodes: list,
    later_nodes: list | None = None,
    deleted_runs: list | None = None,
) -> MagicMock:
    """The database: each catch-up reads the next of ``metas`` (the last one
    repeats), then the child's finished node executions, newest first as
    ``get_node_executions`` orders them: ``nodes`` on the first read,
    ``later_nodes`` (if given) after that."""
    db = MagicMock()
    metas_left = list(metas)
    node_reads = [nodes] if later_nodes is None else [nodes, later_nodes]

    async def get_meta(**_kwargs: Any) -> GraphExecutionMeta | None:
        answer = metas_left.pop(0) if len(metas_left) > 1 else metas_left[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def get_nodes(**_kwargs: Any) -> list:
        visible = node_reads.pop(0) if len(node_reads) > 1 else node_reads[0]
        return sorted(visible, key=lambda node: node.end_time, reverse=True)

    db.get_graph_execution_meta = AsyncMock(side_effect=get_meta)
    db.get_node_executions = AsyncMock(side_effect=get_nodes)
    # Runs deleted from history (soft-deleted); none unless given.
    db.get_graph_executions = AsyncMock(return_value=deleted_runs or [])
    return db


async def _run(
    monkeypatch: pytest.MonkeyPatch, bus: object, db: MagicMock
) -> tuple[list[tuple[str, Any]], AgentExecutorBlock]:
    monkeypatch.setattr(
        "backend.executor.utils.get_async_execution_event_bus", lambda: bus
    )
    monkeypatch.setattr(sub_agent_wait, "execution_db", lambda: db)
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
    must not overtake b, nor be lost when the database hasn't recorded it."""
    stream = _Stream()
    a, b, c = _output(1, "a"), _output(2, "b"), _output(3, "c")
    for item in (None, a, None, c, _finished_event()):
        stream.items.put_nowait(item)
    db = _db(
        _meta(ExecutionStatus.RUNNING),
        ConnectionError("database manager unreachable"),
        _meta(ExecutionStatus.RUNNING),
        nodes=[],
        later_nodes=[a, b],
    )

    outputs, _ = await _run(monkeypatch, stream, db)

    assert outputs == [("result", "a"), ("result", "b"), ("result", "c")]


async def test_a_live_end_during_a_pending_catch_up_ends_the_wait(
    monkeypatch: pytest.MonkeyPatch,
):
    """The end arrives while a failed catch-up is still owed, and the database
    hasn't recorded it yet: the end retries the catch-up at once, which yields
    what was missed, then the live end finishes the wait."""
    # Beyond the test's bound: only an immediate retry finishes in time.
    monkeypatch.setattr(sub_agent_wait, "CATCH_UP_RETRY_S", 60.0)
    stream = _Stream()
    a, b = _output(1, "a"), _output(2, "b")
    for item in (None, a, None, _finished_event()):
        stream.items.put_nowait(item)
    db = _db(
        _meta(ExecutionStatus.RUNNING),
        ConnectionError("database manager unreachable"),
        _meta(ExecutionStatus.RUNNING),
        nodes=[],
        later_nodes=[a, b],
    )

    outputs, block = await _run(monkeypatch, stream, db)

    assert outputs == [("result", "a"), ("result", "b")]
    assert block.execution_stats.reconciled_cost_delta == 7


async def test_an_end_whose_own_catch_up_fails_is_kept_for_the_retry(
    monkeypatch: pytest.MonkeyPatch,
):
    """The catch-up owed when the end arrives fails too; the child publishes
    nothing more, so the end must survive until the next attempt, which comes
    soon rather than after the 60 s quiet recheck."""
    monkeypatch.setattr(sub_agent_wait, "CATCH_UP_RETRY_S", 0.05)
    stream = _Stream()
    a, b = _output(1, "a"), _output(2, "b")
    for item in (None, a, None, _finished_event()):
        stream.items.put_nowait(item)
    db = _db(
        _meta(ExecutionStatus.RUNNING),
        ConnectionError("database manager unreachable"),
        ConnectionError("still unreachable"),
        _meta(ExecutionStatus.RUNNING),
        nodes=[],
        later_nodes=[a, b],
    )

    outputs, block = await _run(monkeypatch, stream, db)

    assert outputs == [("result", "a"), ("result", "b")]
    assert block.execution_stats.reconciled_cost_delta == 7


async def test_a_live_end_after_a_failed_quiet_recheck_is_caught_up_first(
    monkeypatch: pytest.MonkeyPatch,
):
    """The quiet-spell recheck fails, so whether the stream missed anything is
    unknown: the end that then arrives live must not finish the wait before a
    catch-up has read what the database holds."""
    monkeypatch.setattr(sub_agent_wait, "QUIET_RECHECK_S", 0.05)
    stream = _Stream()
    stream.items.put_nowait(None)
    b = _output(2, "b")  # never published: only the database has it
    db = _db(
        _meta(ExecutionStatus.RUNNING),
        ConnectionError("database manager unreachable"),
        _meta(ExecutionStatus.RUNNING),
        nodes=[],
        later_nodes=[b],
    )
    reads = db.get_graph_execution_meta.side_effect

    async def end_arrives_during_the_failed_recheck(**kwargs: Any):
        # The end is queued by the read that fails, so it can only arrive
        # after the recheck failed, however slow the runner.
        if db.get_graph_execution_meta.await_count == 2:
            stream.items.put_nowait(_finished_event())
        return await reads(**kwargs)

    db.get_graph_execution_meta.side_effect = end_arrives_during_the_failed_recheck

    outputs, block = await _run(monkeypatch, stream, db)

    assert outputs == [("result", "b")]
    assert block.execution_stats.reconciled_cost_delta == 7


async def test_a_run_gone_after_its_end_was_seen_ends_the_wait(
    monkeypatch: pytest.MonkeyPatch,
):
    """b arrives live while a catch-up is owed, and the run is gone by the
    time the end retries it: the wait ends on the end, with b."""
    stream = _Stream()
    a, b = _output(1, "a"), _output(2, "b")
    for item in (None, a, None, b, _finished_event()):
        stream.items.put_nowait(item)
    db = _db(
        _meta(ExecutionStatus.RUNNING),
        ConnectionError("database manager unreachable"),
        None,
        nodes=[],
    )

    outputs, block = await _run(monkeypatch, stream, db)

    assert outputs == [("result", "a"), ("result", "b")]
    assert block.execution_stats.reconciled_cost_delta == 7


async def test_a_run_gone_from_the_database_fails_the_wait(
    monkeypatch: pytest.MonkeyPatch,
):
    """Nothing will ever finish a run whose row is gone: waiting would hang
    the parent."""
    stream = _Stream()
    stream.items.put_nowait(None)
    db = _db(None, nodes=[])

    with pytest.raises(sub_agent_wait.SubAgentRunGoneError) as gone:
        await _run(monkeypatch, stream, db)
    # Shown to the user as the node's error.
    assert "sub-workflow" in str(gone.value) and "Graph" not in str(gone.value)


async def test_a_running_child_deleted_from_history_is_still_waited_on(
    monkeypatch: pytest.MonkeyPatch,
):
    """Deleting a run from history doesn't stop it: the child keeps running
    and publishing, so the parent waits for it and gets its outputs."""
    stream = _Stream()
    a = _output(1, "a")
    for item in (None, a, _finished_event()):
        stream.items.put_nowait(item)
    db = _db(None, nodes=[], deleted_runs=[_meta(ExecutionStatus.RUNNING)])

    outputs, block = await _run(monkeypatch, stream, db)

    assert outputs == [("result", "a")]
    assert block.execution_stats.reconciled_cost_delta == 7
    # Read among deleted runs, and only the parent's user's.
    db.get_graph_executions.assert_awaited_with(
        graph_exec_id=_CHILD, user_id=_USER, deleted=True
    )


async def test_a_database_down_for_good_still_ends_on_a_seen_end(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sub_agent_wait, "CATCH_UP_RETRY_S", 0.01)
    stream = _Stream()
    a = _output(1, "a")
    for item in (None, a, None, _finished_event()):
        stream.items.put_nowait(item)
    db = _db(
        _meta(ExecutionStatus.RUNNING),
        ConnectionError("database manager unreachable"),
        nodes=[],
    )

    outputs, block = await _run(monkeypatch, stream, db)

    assert outputs == [("result", "a")]
    assert block.execution_stats.reconciled_cost_delta == 7


async def test_failed_catch_ups_back_off(monkeypatch: pytest.MonkeyPatch):
    """At a fixed interval a long database outage would be read every few
    seconds by every waiting parent."""
    monkeypatch.setattr(sub_agent_wait, "CATCH_UP_RETRY_S", 0.05)
    stream = _Stream()
    stream.items.put_nowait(None)
    db = _db(ConnectionError("database manager unreachable"), nodes=[])

    waiting = asyncio.create_task(_run(monkeypatch, stream, db))
    await asyncio.sleep(1.0)
    waiting.cancel()
    await asyncio.gather(waiting, return_exceptions=True)

    # 0.05, 0.1, 0.2, 0.4 s apart: about five reads, not twenty.
    assert db.get_graph_execution_meta.await_count <= 8


async def test_a_busy_child_adds_no_reads_while_a_retry_is_scheduled(
    monkeypatch: pytest.MonkeyPatch,
):
    """While a failed catch-up's retry is scheduled, the child's finished
    nodes wait for it rather than each reading a database that keeps failing.
    It stays down, so the wait ends on the end seen live, with the outputs
    seen live."""
    # 0.1, 0.2 and 0.3 s of backoff before giving up: well inside _WITHIN_S.
    monkeypatch.setattr(sub_agent_wait, "CATCH_UP_RETRY_S", 0.1)
    monkeypatch.setattr(sub_agent_wait, "QUIET_RECHECK_S", 0.3)
    stream = _Stream()
    a, b = _output(1, "a"), _output(22, "b")
    busy = [
        _output(n, "-").model_copy(update={"block_id": AgentExecutorBlock().id})
        for n in range(2, 22)
    ]
    for item in (None, a, *busy, b, _finished_event()):
        stream.items.put_nowait(item)
    db = _db(ConnectionError("database manager unreachable"), nodes=[])

    outputs, _ = await _run(monkeypatch, stream, db)

    # The gap, the end, then the retries until the wait gives up: about four
    # reads, not one more for each of the 22 finished nodes.
    assert db.get_graph_execution_meta.await_count < 10
    assert outputs == [("result", "a"), ("result", "b")]


async def test_a_due_retry_runs_before_the_events_queued_meanwhile(
    monkeypatch: pytest.MonkeyPatch,
):
    """A retry can fall due while the child's updates are already queued: a
    queued event doesn't wait for a timeout, so unless the deadline is checked
    first the retry trails every one of them, and a busy child keeps it back."""

    async def no_more_events() -> AsyncIterator[ExecutionEvent | None]:
        await asyncio.Event().wait()
        yield None

    wait = sub_agent_wait.SubAgentWait(
        no_more_events(),
        user_id=_USER,
        graph_exec_id=_CHILD,
        log_id="test",
        logger=MagicMock(),
    )
    db = _db(_meta(ExecutionStatus.COMPLETED), nodes=[])
    reads = db.get_graph_execution_meta.side_effect
    queued_at_the_retry: list[int] = []

    async def read(**kwargs: Any) -> GraphExecutionMeta | None:
        queued_at_the_retry.append(wait._queue.qsize())
        return await reads(**kwargs)

    db.get_graph_execution_meta.side_effect = read
    monkeypatch.setattr(sub_agent_wait, "execution_db", lambda: db)
    # A failed catch-up's retry, already due, with three updates queued.
    wait._stale = True
    wait._retry_at = asyncio.get_running_loop().time() - 1.0
    running = _output(1, "a").model_copy(update={"status": ExecutionStatus.RUNNING})
    for _ in range(3):
        wait._queue.put_nowait(running)

    async with asyncio.timeout(_WITHIN_S):
        outputs = [output async for output in wait.outputs()]

    assert outputs == []
    assert queued_at_the_retry[0] == 3


async def test_outputs_held_while_a_catch_up_is_owed_are_capped(
    monkeypatch: pytest.MonkeyPatch,
):
    """A chatty child during a long outage mustn't hold every output in
    memory. Past the cap the catch-up's read returns them, still in order."""
    monkeypatch.setattr(sub_agent_wait, "MAX_HELD_OUTPUTS", 2)
    monkeypatch.setattr(sub_agent_wait, "CATCH_UP_RETRY_S", 60.0)
    stream = _Stream()
    produced = [_output(n, str(n)) for n in range(1, 6)]
    for item in (None, *produced, _finished_event()):
        stream.items.put_nowait(item)
    wait = sub_agent_wait.SubAgentWait(
        stream.listen_with_resync("unused"),
        user_id=_USER,
        graph_exec_id=_CHILD,
        log_id="test",
        logger=MagicMock(),
    )
    # The gap's catch-up fails; the end's, with all five recorded, succeeds.
    db = _db(
        ConnectionError("database manager unreachable"),
        _meta(ExecutionStatus.COMPLETED),
        nodes=produced,
    )
    reads = db.get_graph_execution_meta.side_effect
    held_at_each_read: list[int] = []

    async def read(**kwargs: Any) -> GraphExecutionMeta | None:
        held_at_each_read.append(len(wait._held))
        return await reads(**kwargs)

    db.get_graph_execution_meta.side_effect = read
    monkeypatch.setattr(sub_agent_wait, "execution_db", lambda: db)

    async with asyncio.timeout(_WITHIN_S):
        outputs = [output async for output in wait.outputs()]

    assert held_at_each_read == [0, 2]
    assert outputs == [("result", str(n)) for n in range(1, 6)]


async def test_an_owed_catch_up_runs_on_time_while_the_child_keeps_streaming(
    monkeypatch: pytest.MonkeyPatch,
):
    """A child that keeps sending non-terminal updates (nodes queued and
    started) faster than the retry interval must not push the owed retry out,
    or the outputs missed in a gap wait for the next node to finish."""
    monkeypatch.setattr(sub_agent_wait, "CATCH_UP_RETRY_S", 0.2)
    stream = _Stream()
    a, b = _output(1, "a"), _output(2, "b")
    still_running = a.model_copy(update={"status": ExecutionStatus.RUNNING})
    for item in (None, a, None):
        stream.items.put_nowait(item)
    # Read 1 finds nothing missed, read 2 (after the gap) fails, the retry
    # (read 3) finds b, which was published while the stream was down.
    db = _db(
        _meta(ExecutionStatus.RUNNING),
        ConnectionError("database manager unreachable"),
        _meta(ExecutionStatus.RUNNING),
        nodes=[],
        later_nodes=[a, b],
    )
    retried_while_streaming = False

    async def child_keeps_streaming() -> None:
        nonlocal retried_while_streaming
        try:
            async with asyncio.timeout(3):
                while db.get_graph_execution_meta.await_count < 3:
                    stream.items.put_nowait(still_running)
                    await asyncio.sleep(0.02)
            retried_while_streaming = True
        except TimeoutError:
            pass
        stream.items.put_nowait(_finished_event())

    child = asyncio.create_task(child_keeps_streaming())
    outputs, _ = await _run(monkeypatch, stream, db)
    await child

    assert retried_while_streaming
    assert outputs == [("result", "a"), ("result", "b")]


async def test_a_failed_catch_up_does_not_end_the_wait(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sub_agent_wait, "CATCH_UP_RETRY_S", 0.05)
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
    monkeypatch.setattr(sub_agent_wait, "QUIET_RECHECK_S", 0.05)
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
        monkeypatch.setattr(sub_agent_wait, "execution_db", lambda: db)

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
            reads_before_the_drop = db.get_node_executions.await_count
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
    # b was published while the connection was down: only a catch-up from
    # the database could have found it.
    assert db.get_node_executions.await_count > reads_before_the_drop
