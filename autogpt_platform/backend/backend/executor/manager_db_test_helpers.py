"""Shared fixtures and helpers for executor DB tests (#17200)."""

import asyncio
import threading
from collections.abc import Iterator

import pytest

from backend.api.model import CreateGraph
from backend.data import graph
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionEntry,
    create_graph_execution,
    update_graph_execution_stats,
)
from backend.executor import manager
from backend.executor.manager import ExecutionProcessor
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.usecases.sample import create_test_user
from backend.util.test import SpinTestServer

HELD_FOR = 60


@pytest.fixture
def processor() -> Iterator[ExecutionProcessor]:
    """An executor worker with its node-execution and node-evaluation loops
    and its credentials manager, as `on_graph_executor_start` sets it up."""
    proc = ExecutionProcessor()
    proc.creds_manager = IntegrationCredentialsManager()
    loops = [asyncio.new_event_loop() for _ in range(2)]
    threads = [threading.Thread(target=loop.run_forever, daemon=True) for loop in loops]
    for thread in threads:
        thread.start()
    proc.node_execution_loop, proc.node_evaluation_loop = loops
    yield proc
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


async def _a_queued_run(
    server: SpinTestServer, the_graph: graph.Graph
) -> GraphExecutionEntry:
    """A run of `the_graph`, QUEUED as `add_graph_execution` leaves it."""
    user = await create_test_user()
    workflow = await server.agent_server.test_create_graph(
        CreateGraph(graph=the_graph),
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


a_queued_run = _a_queued_run
