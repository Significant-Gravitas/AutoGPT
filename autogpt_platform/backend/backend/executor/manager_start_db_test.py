"""An executor's start of a QUEUED run, against the real database (#17178).

A stop that lands between the executor's read of the run and its start write
keeps the run stopped: the stop goes through the real database, as does the start
write that follows it, with the executor held between its read and its write by
an Event. A start that committed and lost its answer does not drop the run.
"""

import asyncio
import threading
from collections.abc import Callable, Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from backend.api.model import CreateGraph
from backend.blocks.basic import StoreValueBlock
from backend.data import graph
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionEntry,
    create_graph_execution,
    get_graph_execution,
    update_graph_execution_stats,
)
from backend.executor import manager
from backend.executor.manager import ExecutionProcessor
from backend.executor.utils import stop_graph_execution
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


async def _a_queued_run(server: SpinTestServer) -> GraphExecutionEntry:
    """A run of one Store Value node, QUEUED as `add_graph_execution` leaves it."""
    user = await create_test_user()
    workflow = await server.agent_server.test_create_graph(
        CreateGraph(
            graph=graph.Graph(
                name="Stopped as it starts",
                description="One starting node, which a stop may end before it starts",
                nodes=[
                    graph.Node(
                        block_id=StoreValueBlock().id, input_default={"input": "hello"}
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
    processor: ExecutionProcessor, entry: GraphExecutionEntry
) -> asyncio.Task[None]:
    return asyncio.create_task(
        asyncio.to_thread(
            processor.on_graph_execution, entry, threading.Event(), MagicMock()
        )
    )


def _hold_the_executor_after_its_read(
    monkeypatch: pytest.MonkeyPatch,
    read_done: threading.Event,
    release: threading.Event,
) -> None:
    """Make the executor's read of a run wait, once it has the answer, until
    ``release`` is set. Its other calls go through to the real client."""
    real_client = manager.get_db_client

    def held_client() -> MagicMock:
        client = real_client()
        read: Callable[..., Any] = client.get_graph_execution_meta

        def held_read(*args: Any, **kwargs: Any) -> Any:
            run = read(*args, **kwargs)
            read_done.set()
            assert release.wait(HELD_FOR), "the stop did not finish"
            return run

        held = MagicMock(wraps=client)
        held.get_graph_execution_meta.side_effect = held_read
        return held

    monkeypatch.setattr(manager, "get_db_client", held_client)


def _lose_the_first_answer_to_the_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the executor's start commit and then be answered by its own retry,
    as the database client does when the first answer is lost."""
    real_client = manager.get_db_client

    def client_losing_an_answer() -> MagicMock:
        client = real_client()
        start: Callable[..., Any] = client.update_graph_execution_start_time

        def start_twice(graph_exec_id: str) -> Any:
            start(graph_exec_id)
            return start(graph_exec_id)

        losing = MagicMock(wraps=client)
        losing.update_graph_execution_start_time.side_effect = start_twice
        return losing

    monkeypatch.setattr(manager, "get_db_client", client_losing_an_answer)


def _stop_without_a_queue(mocker: MockerFixture) -> None:
    """The stop's cancels and its event go to mocks; its reads and its write of
    the run go to the database."""
    mocker.patch(
        "backend.executor.utils.get_async_execution_queue",
        return_value=mocker.AsyncMock(),
    )
    event_bus = mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus"
    ).return_value
    event_bus.publish = mocker.AsyncMock()


async def test_a_stop_after_the_executors_read_keeps_the_run_stopped(
    server: SpinTestServer,
    processor: ExecutionProcessor,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
):
    _stop_without_a_queue(mocker)
    entry = await _a_queued_run(server)
    run_the_graph = mocker.patch.object(
        processor, "_on_graph_execution", wraps=processor._on_graph_execution
    )
    read_done = threading.Event()
    stop_done = threading.Event()
    _hold_the_executor_after_its_read(monkeypatch, read_done, stop_done)

    executor = _the_executor_runs(processor, entry)
    try:
        assert await asyncio.to_thread(
            read_done.wait, HELD_FOR
        ), "the executor never read the run"
        await stop_graph_execution(
            user_id=entry.user_id, graph_exec_id=entry.graph_exec_id
        )
    finally:
        stop_done.set()
        await executor

    stored = await get_graph_execution(
        entry.user_id, entry.graph_exec_id, include_node_executions=True
    )
    assert stored is not None
    assert stored.status == ExecutionStatus.TERMINATED
    assert stored.started_at is None
    assert [node.status for node in stored.node_executions] == [ExecutionStatus.QUEUED]
    run_the_graph.assert_not_called()


async def test_a_start_whose_answer_was_lost_still_runs_the_graph(
    server: SpinTestServer,
    processor: ExecutionProcessor,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
):
    """The database client retries a call that failed. A start that committed and
    lost its answer is answered nothing by its retry, with the run RUNNING, and
    the executor goes on with that run instead of dropping it as stopped."""
    entry = await _a_queued_run(server)
    run_the_graph = mocker.patch.object(
        processor, "_on_graph_execution", wraps=processor._on_graph_execution
    )
    _lose_the_first_answer_to_the_start(monkeypatch)

    await _the_executor_runs(processor, entry)

    stored = await get_graph_execution(
        entry.user_id, entry.graph_exec_id, include_node_executions=True
    )
    assert stored is not None
    assert stored.status == ExecutionStatus.COMPLETED
    assert stored.started_at is not None
    assert [node.status for node in stored.node_executions] == [
        ExecutionStatus.COMPLETED
    ]
    run_the_graph.assert_called_once()
