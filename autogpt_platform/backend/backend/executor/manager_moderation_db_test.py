"""A run whose input moderation runs out of time fails, and none of its nodes
stays queued: the executor runs it against the database, as a pod would."""

import asyncio
import threading
from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest

from backend.api.model import CreateGraph
from backend.blocks.basic import StoreValueBlock
from backend.data import graph
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionEntry,
    create_graph_execution,
    get_graph_execution,
)
from backend.executor import manager
from backend.executor.manager import ExecutionProcessor
from backend.executor.manager_moderation_test import _automod_that_never_answers
from backend.usecases.sample import create_test_user
from backend.util.test import SpinTestServer


@pytest.fixture
def processor() -> Iterator[ExecutionProcessor]:
    """An executor worker with its node-execution and node-evaluation loops."""
    processor = ExecutionProcessor()
    loops = [asyncio.new_event_loop() for _ in range(2)]
    threads = [threading.Thread(target=loop.run_forever, daemon=True) for loop in loops]
    for thread in threads:
        thread.start()
    processor.node_execution_loop, processor.node_evaluation_loop = loops
    yield processor
    for loop, thread in zip(loops, threads):
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_run_whose_input_check_never_answers_fails_with_no_node_queued(
    server: SpinTestServer,
    processor: ExecutionProcessor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _automod_that_never_answers(monkeypatch, fail_open=False)
    monkeypatch.setattr(manager, "MODERATION_WAIT_SECONDS", 1.0)
    monkeypatch.setattr(manager.settings.config, "enable_credit", False)
    user = await create_test_user()
    workflow = await server.agent_server.test_create_graph(
        CreateGraph(
            graph=graph.Graph(
                name="Moderated workflow",
                description="Two starting nodes whose inputs are moderated",
                nodes=[
                    graph.Node(
                        block_id=StoreValueBlock().id, input_default={"input": text}
                    )
                    for text in ("hello", "world")
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
    assert {node.status for node in run.node_executions} == {ExecutionStatus.QUEUED}

    await asyncio.to_thread(
        processor.on_graph_execution,
        GraphExecutionEntry(
            user_id=user.id,
            graph_exec_id=run.id,
            graph_id=workflow.id,
            graph_version=workflow.version,
        ),
        threading.Event(),
        MagicMock(),
    )

    stored = await get_graph_execution(user.id, run.id, include_node_executions=True)
    assert stored is not None
    assert stored.status == ExecutionStatus.FAILED
    assert [node.status for node in stored.node_executions] == [
        ExecutionStatus.FAILED,
        ExecutionStatus.FAILED,
    ]
