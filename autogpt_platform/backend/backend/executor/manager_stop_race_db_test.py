"""A stop whose write lands after an executor has started the run, against the
real database (#17178).

``stop_graph_execution`` reads a QUEUED run, writes TERMINATED itself and sends its
cancel a second time. An executor that starts the run between that read and write
has registered it by the time of the write, but never saw the cancel the stop sent
before its read: the second cancel is the only one that can stop the workflow, and
the update the stop publishes can reach a page before the executor's RUNNING.

The stop, the executor and the database are real, with the stop held between its
read and its write by an Event. The broker is a stand-in that drops a cancel until
the executor has started the run, as the executor's cancel handler drops one for a
run it has not registered (#17182).
"""

import asyncio
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.api.model import CreateGraph
from backend.blocks.time_blocks import CountdownTimerBlock
from backend.data import execution as execution_db
from backend.data import graph
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionEntry,
    GraphExecutionMeta,
    create_graph_execution,
    get_graph_execution,
    update_graph_execution_stats,
)
from backend.executor import manager
from backend.executor.manager import ExecutionProcessor
from backend.executor.utils import CancelExecutionEvent, stop_graph_execution
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.usecases.sample import create_test_user
from backend.util.test import SpinTestServer

HELD_FOR = 60


@pytest.fixture
def processor() -> Iterator[ExecutionProcessor]:
    """An executor worker with its node-execution and node-evaluation loops
    and its credentials manager, as `on_graph_executor_start` sets it up."""
    processor = ExecutionProcessor()
    processor.creds_manager = IntegrationCredentialsManager()
    loops = [asyncio.new_event_loop() for _ in range(2)]
    threads = [threading.Thread(target=loop.run_forever, daemon=True) for loop in loops]
    for thread in threads:
        thread.start()
    processor.node_execution_loop, processor.node_evaluation_loop = loops
    yield processor
    for loop, thread in zip(loops, threads):
        # Closes the async Redis clients the loop's code opened, as asyncio.run
        # does; loop.close() alone leaves them open (#17087).
        asyncio.run_coroutine_threadsafe(loop.shutdown_asyncgens(), loop).result(10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()


@pytest.fixture(autouse=True)
def _no_credit_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager.settings.config, "enable_credit", False)


@pytest.fixture
def the_node(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[threading.Event, threading.Event]]:
    """The countdown node, whose run says it has started and then waits to be
    released. Returns the event it sets and the one it waits for."""
    started = threading.Event()
    release = threading.Event()

    async def run(
        self: CountdownTimerBlock, input_data: Any, **_: Any
    ) -> AsyncIterator[tuple[str, Any]]:
        started.set()
        await asyncio.to_thread(release.wait, HELD_FOR)
        yield "output_message", input_data.input_message

    monkeypatch.setattr(CountdownTimerBlock, "run", run)
    yield started, release
    release.set()


async def _a_queued_run(server: SpinTestServer) -> GraphExecutionEntry:
    """A run of one countdown node, QUEUED as `add_graph_execution` leaves it."""
    user = await create_test_user()
    workflow = await server.agent_server.test_create_graph(
        CreateGraph(
            graph=graph.Graph(
                name="Started between a stop's read and its write",
                description="One node, which a stop may end after it has started",
                nodes=[
                    graph.Node(
                        block_id=CountdownTimerBlock().id,
                        input_default={"seconds": 1, "input_message": "done"},
                    )
                ],
                links=[],
            )
        ),
        user.id,
    )
    run = await create_graph_execution(
        graph_id=workflow.id,
        graph_version=workflow.version,
        starting_nodes_input=[(node.id, node.input_default) for node in workflow.nodes],
        inputs={},
        user_id=user.id,
    )
    await update_graph_execution_stats(run.id, status=ExecutionStatus.QUEUED)
    return GraphExecutionEntry(
        user_id=user.id,
        graph_exec_id=run.id,
        graph_id=workflow.id,
        graph_version=workflow.version,
    )


def _the_executor_runs(
    processor: ExecutionProcessor,
    entry: GraphExecutionEntry,
    cancel: threading.Event,
) -> asyncio.Task[None]:
    return asyncio.create_task(
        asyncio.to_thread(processor.on_graph_execution, entry, cancel, MagicMock())
    )


def _a_stop_held_after_its_read(
    mocker: MockerFixture,
    entry: GraphExecutionEntry,
    started: threading.Event,
    cancel: threading.Event,
    on_update: Callable[[GraphExecutionMeta], None] | None = None,
) -> tuple[asyncio.Task[None], asyncio.Event, asyncio.Event]:
    """The real stop of ``entry``, held after it has read the run until its
    ``write`` event is set. Returns the stop, its ``read`` event and its ``write``
    event.

    Its cancels for ``entry`` reach ``cancel`` only once ``started`` is set, and
    the update it publishes goes to ``on_update``.
    """

    def deliver_cancel(*, message: str, **_: Any) -> None:
        cancelled = CancelExecutionEvent.model_validate_json(message).graph_exec_id
        if started.is_set() and cancelled == entry.graph_exec_id:
            cancel.set()

    queue = mocker.AsyncMock()
    queue.publish_message.side_effect = deliver_cancel
    mocker.patch("backend.executor.utils.get_async_execution_queue", return_value=queue)
    event_bus = mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus"
    ).return_value
    event_bus.publish = mocker.AsyncMock(side_effect=on_update)

    read, write = asyncio.Event(), asyncio.Event()
    real_read: Callable[..., Awaitable[Any]] = execution_db.get_graph_execution_meta

    async def held_read(**kwargs: Any) -> Any:
        run = await real_read(**kwargs)
        if not read.is_set():
            read.set()
            await write.wait()
        return run

    mocker.patch.object(execution_db, "get_graph_execution_meta", side_effect=held_read)
    stop = asyncio.create_task(
        stop_graph_execution(
            user_id=entry.user_id, graph_exec_id=entry.graph_exec_id, cascade=False
        )
    )
    return stop, read, write


async def test_a_start_between_the_stops_read_and_its_write_is_stopped_by_its_second_cancel(
    server: SpinTestServer,
    processor: ExecutionProcessor,
    the_node: tuple[threading.Event, threading.Event],
    mocker: MockerFixture,
):
    """The first cancel is dropped, as the executor had not started the run. The
    stop's write then ends a run that is already running, and only the cancel it
    sends after that write reaches the executor."""
    node_started, release_the_node = the_node
    cancel = threading.Event()
    entry = await _a_queued_run(server)
    run_the_graph = mocker.spy(processor, "_on_graph_execution")
    stop, stop_has_read, let_the_stop_write = _a_stop_held_after_its_read(
        mocker, entry, node_started, cancel
    )

    executor = None
    try:
        await asyncio.wait_for(stop_has_read.wait(), HELD_FOR)
        executor = _the_executor_runs(processor, entry, cancel)
        assert await asyncio.to_thread(
            node_started.wait, HELD_FOR
        ), "the executor never started the node"
        let_the_stop_write.set()
        await stop
        # With the second cancel delivered the graph ends TERMINATED whatever the
        # node does next; without it, the released node ends and the graph runs on.
        release_the_node.set()
        await executor
    finally:
        let_the_stop_write.set()
        release_the_node.set()
        await asyncio.gather(
            *[task for task in (stop, executor) if task], return_exceptions=True
        )

    stored = await get_graph_execution(
        entry.user_id, entry.graph_exec_id, include_node_executions=True
    )
    assert stored is not None
    assert stored.status == ExecutionStatus.TERMINATED
    assert stored.started_at is not None
    _, graph_status = run_the_graph.spy_return
    assert (
        graph_status == ExecutionStatus.TERMINATED
    ), "the workflow ran to its end although it was stopped"


async def test_a_stopped_runs_last_update_is_terminal_when_the_executors_running_one_is_late(
    server: SpinTestServer,
    processor: ExecutionProcessor,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
):
    """The executor's RUNNING update is held until the stop has published its
    TERMINATED one, so it reaches a page after it. The executor's own final write
    is then refused, since TERMINATED can't follow TERMINATED, and it broadcasts
    the stored state instead: the last update a page gets is terminal."""
    running_reached, stop_published = threading.Event(), threading.Event()
    updates: list[tuple[str, ExecutionStatus]] = []

    def the_stop_publishes(run: GraphExecutionMeta) -> None:
        updates.append(("stop", run.status))
        stop_published.set()

    def the_executor_publishes(update: Any) -> None:
        if not isinstance(update, GraphExecutionMeta):
            return
        if update.status == ExecutionStatus.RUNNING and not running_reached.is_set():
            running_reached.set()
            assert stop_published.wait(HELD_FOR), "the stop never published"
        updates.append(("executor", update.status))

    monkeypatch.setattr(manager, "send_execution_update", the_executor_publishes)
    cancel = threading.Event()
    entry = await _a_queued_run(server)
    stop, stop_has_read, let_the_stop_write = _a_stop_held_after_its_read(
        mocker, entry, running_reached, cancel, on_update=the_stop_publishes
    )

    executor = None
    try:
        await asyncio.wait_for(stop_has_read.wait(), HELD_FOR)
        executor = _the_executor_runs(processor, entry, cancel)
        assert await asyncio.to_thread(
            running_reached.wait, HELD_FOR
        ), "the executor never started the run"
        let_the_stop_write.set()
        await stop
        await executor
    finally:
        let_the_stop_write.set()
        stop_published.set()
        await asyncio.gather(
            *[task for task in (stop, executor) if task], return_exceptions=True
        )

    assert updates == [
        ("stop", ExecutionStatus.TERMINATED),
        ("executor", ExecutionStatus.RUNNING),
        ("executor", ExecutionStatus.TERMINATED),
    ]
