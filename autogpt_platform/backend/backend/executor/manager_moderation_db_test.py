"""The executor runs moderated runs against the database, as a pod would. A run
whose input moderation runs out of time fails with none of its nodes left
queued, and a run whose outputs AutoMod refuses fails with every output masked.
"""

import asyncio
import threading
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
import sentry_sdk

from backend.api.model import CreateGraph
from backend.blocks.basic import StoreValueBlock
from backend.data import graph
from backend.data.execution import (
    ExecutionStatus,
    GraphExecutionEntry,
    create_graph_execution,
    get_graph_execution,
)
from backend.data.model import User
from backend.executor import manager
from backend.executor.manager import ExecutionProcessor
from backend.executor.manager_moderation_test import (
    _automod_that_never_answers,
    _automod_verdict,
    _real_automod,
)
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.usecases.sample import create_test_user
from backend.util.test import SpinTestServer


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
    run_id = await _run_two_node_workflow(server, processor, user)

    stored = await get_graph_execution(user.id, run_id, include_node_executions=True)
    assert stored is not None
    assert stored.status == ExecutionStatus.FAILED
    assert [node.status for node in stored.node_executions] == [
        ExecutionStatus.FAILED,
        ExecutionStatus.FAILED,
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_run_whose_outputs_automod_refuses_fails_with_every_output_masked(
    server: SpinTestServer,
    processor: ExecutionProcessor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def verdict(url: str, *, json: dict[str, Any], **kwargs: Any) -> MagicMock:
        if json["metadata"]["moderation_type"] == "execution_output":
            return _automod_verdict("rejected", "cid-out")
        return _automod_verdict("approved", "cid-in")

    _real_automod(monkeypatch, fail_open=False, post=verdict)
    monkeypatch.setattr(manager.settings.config, "enable_credit", False)
    user = await create_test_user()
    run_id = await _run_two_node_workflow(server, processor, user)

    stored = await get_graph_execution(user.id, run_id, include_node_executions=True)
    assert stored is not None
    assert stored.status == ExecutionStatus.FAILED
    moderated = "Failed due to content moderation (Moderation ID: cid-out)"
    assert [(node.status, node.output_data) for node in stored.node_executions] == [
        (ExecutionStatus.FAILED, {"output": [moderated]}),
        (ExecutionStatus.FAILED, {"output": [moderated]}),
    ]
    shown = stored.model_dump_json()
    assert "hello" not in shown and "world" not in shown


async def _run_two_node_workflow(
    server: SpinTestServer, processor: ExecutionProcessor, user: User
) -> str:
    """Execute a workflow of two Store Value nodes, holding "hello" and
    "world", the way a pod does. Returns the run's ID."""
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

    # In a pod each node run gets a Sentry scope of its own, which takes an
    # active client. Here the nodes tag the scope they inherit from this
    # process, and two at once can leave one's tags on it for later tests'
    # transactions, so the run gets its own.
    with sentry_sdk.isolation_scope():
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
    return run.id
