"""Admin requeue of a run stuck in QUEUED (#15282), on the real stack: the
database, RabbitMQ, the Redis cluster lock and executor processes.

The tests only decide when a delivery reaches the queue; what each piece of
the code under test does with it is never patched out.
"""

import asyncio
import socket
from typing import Any, Awaitable, Callable, TypeVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogpt_libs.auth.models import User as AuthUser
from fastapi import HTTPException
from prisma.enums import ReviewStatus
from prisma.models import AgentGraphExecution, PendingHumanReview

from backend.api.features.admin.diagnostics_admin_routes import (
    RequeueExecutionResponse,
    StopExecutionRequest,
    requeue_single_execution,
)
from backend.api.features.graph_executions.review.model import ReviewItem
from backend.api.features.graph_executions.review.service import process_reviews
from backend.api.model import CreateGraph
from backend.blocks.basic import StoreValueBlock
from backend.blocks.human_in_the_loop import HumanInTheLoopBlock
from backend.blocks.io import AgentInputBlock
from backend.data import execution, graph
from backend.data.execution import ExecutionStatus
from backend.data.human_review import get_pending_reviews_for_execution
from backend.data.model import User
from backend.data.onboarding import get_user_onboarding
from backend.data.rabbitmq import SyncRabbitMQ
from backend.executor import manager as executor_manager
from backend.executor import utils as executor_utils
from backend.executor.manager import ExecutionManager
from backend.executor.utils import (
    GRAPH_EXECUTION_DELAY_QUEUE_NAME,
    GRAPH_EXECUTION_QUEUE_NAME,
    create_execution_queue_config,
)
from backend.usecases.sample import create_test_graph, create_test_user
from backend.util.clients import get_async_execution_queue
from backend.util.test import SpinTestServer, wait_execution

T = TypeVar("T")

ADMIN = AuthUser(user_id="admin-user-id", email="", phone_number="", role="admin")


@pytest.fixture(autouse=True)
def _runners_belong_to_their_org():
    """Check memberships for real: the executor service checks a run's
    org/team in its own process, so a run kept in an org its user isn't in
    would be stopped there."""


@pytest.fixture(autouse=True)
def _default_team_untenanted():
    """Look up the user's own org and team for real, as production does."""


class _SecondExecutorPod(ExecutionManager):
    """Another executor pod, with its own executor id, consuming the same run
    queue. Only its metrics port differs from the real thing, so it can run
    next to the test server's executor."""

    def __init__(self):
        super().__init__()
        # A free port on the address the metrics server binds to.
        with socket.socket() as probe:
            probe.bind((executor_manager.settings.config.pyro_host, 0))
            self.metrics_port = probe.getsockname()[1]

    def run(self):
        executor_manager.settings.config.execution_manager_port = self.metrics_port
        super().run()


@pytest.mark.asyncio(loop_scope="session")
async def test_admin_requeue_runs_a_stuck_queued_execution_once(
    server: SpinTestServer,
):
    """#15282: a run whose queue message was lost sits in QUEUED for good.
    The admin's requeue must get it run, once, and must not count it again
    toward the user's runs."""
    user = await create_test_user()
    test_graph = await _create_graph(server, create_test_graph(), user)
    runs_before = (await get_user_onboarding(user.id)).agentRuns
    run = await _start_with_delivery_held(
        server, user, test_graph, {"input_1": "Hello", "input_2": "World"}, []
    )

    response = await _requeue(run.id)

    assert (response.success, response.requeued_count) == (True, 1)
    await wait_execution(user.id, run.id, 60)
    await _assert_ran_once(user, run.id, test_graph)
    assert (await get_user_onboarding(user.id)).agentRuns == runs_before + 1


@pytest.mark.asyncio(loop_scope="session")
async def test_admin_requeues_racing_the_original_delivery_run_it_once(
    server: SpinTestServer,
):
    """Three admin requeues read the run as QUEUED at the same moment, so each
    publishes it, and then the original delivery, which was only late, turns
    up: four deliveries of one run race across two executor pods. The run
    must happen once: each step runs once, so it is charged once, and the
    user's run count goes up once."""
    user = await create_test_user()
    test_graph = await _create_graph(server, create_test_graph(), user)
    runs_before = (await get_user_onboarding(user.id)).agentRuns
    held: list[dict[str, Any]] = []
    run = await _start_with_delivery_held(
        server, user, test_graph, {"input_1": "Hello", "input_2": "World"}, held
    )

    with _SecondExecutorPod():
        await _wait_for(_run_queue_has_two_consumers, "the second executor pod")
        responses = await _requeue_all_at_once(run.id, times=3)
        assert [r.requeued_count for r in responses] == [1, 1, 1]
        await _deliver(held)

        await wait_execution(user.id, run.id, 60)
        await _wait_for(_run_queues_drained, "every duplicate delivery")

    await _assert_ran_once(user, run.id, test_graph)
    assert (await get_user_onboarding(user.id)).agentRuns == runs_before + 1


@pytest.mark.asyncio(loop_scope="session")
async def test_a_late_original_delivery_does_not_get_past_a_review(
    server: SpinTestServer,
):
    """The admin's requeue ran the run up to a human review; then the original
    delivery, never lost after all, turns up and takes the run through the
    review step again. The review still waits, the step after it doesn't run
    until the user approves, and then it runs once."""
    user = await create_test_user()
    test_graph = await _create_graph(server, _graph_with_a_review(), user)
    held: list[dict[str, Any]] = []
    run = await _start_with_delivery_held(
        server, user, test_graph, {"value": "draft"}, held
    )

    assert (await _requeue(run.id)).requeued_count == 1
    parked = await _wait_for(
        lambda: _row_in(run.id, ExecutionStatus.REVIEW), "the run to reach review"
    )
    with pytest.raises(HTTPException) as refused:
        await _requeue(run.id)
    assert refused.value.status_code == 404

    await _deliver(held)
    await _wait_for(
        lambda: _row_in(run.id, ExecutionStatus.REVIEW, updated_after=parked),
        "the late delivery to be handled",
    )

    (review,) = await get_pending_reviews_for_execution(run.id, user.id)
    stored = await PendingHumanReview.prisma().find_unique(
        where={"nodeExecId": review.node_exec_id}
    )
    assert stored and stored.status == ReviewStatus.WAITING
    store_node = _node_of(test_graph, StoreValueBlock)
    assert store_node not in {ne.node_id for ne in await _node_execs(user, run.id)}

    await process_reviews(
        user.id, [ReviewItem(node_exec_id=review.node_exec_id, approved=True)]
    )
    await wait_execution(user.id, run.id, 60)
    await _assert_ran_once(user, run.id, test_graph)


async def _create_graph(s: SpinTestServer, g: graph.Graph, u: User) -> graph.Graph:
    return await s.agent_server.test_create_graph(CreateGraph(graph=g), u.id)


def _graph_with_a_review() -> graph.Graph:
    nodes = [
        graph.Node(block_id=AgentInputBlock().id, input_default={"name": "value"}),
        graph.Node(
            block_id=HumanInTheLoopBlock().id, input_default={"name": "Send it?"}
        ),
        graph.Node(block_id=StoreValueBlock().id),
    ]
    links = [
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="result",
            sink_name="data",
        ),
        graph.Link(
            source_id=nodes[1].id,
            sink_id=nodes[2].id,
            source_name="approved_data",
            sink_name="input",
        ),
    ]
    return graph.Graph(
        name="ReviewGraph", description="Input, review, store", nodes=nodes, links=links
    )


async def _start_with_delivery_held(
    server: SpinTestServer,
    user: User,
    test_graph: graph.Graph,
    inputs: dict[str, Any],
    held: list[dict[str, Any]],
) -> execution.GraphExecutionMeta:
    """Start a run whose queue message doesn't reach the queue: kept in
    ``held`` to be delivered late, or lost if the caller never delivers it."""

    def keep(**message: Any) -> None:
        held.append(message)

    queue = MagicMock(publish_message=AsyncMock(side_effect=keep))
    with patch(
        "backend.executor.utils.get_async_execution_queue",
        new=AsyncMock(return_value=queue),
    ):
        run = await server.agent_server.test_execute_graph(
            user_id=user.id,
            graph_id=test_graph.id,
            graph_version=test_graph.version,
            node_input=inputs,
        )
    stuck = await execution.get_graph_execution_meta(
        user_id=user.id, execution_id=run.id
    )
    assert stuck and stuck.status == ExecutionStatus.QUEUED
    return run


async def _requeue(graph_exec_id: str) -> RequeueExecutionResponse:
    return await requeue_single_execution(
        StopExecutionRequest(execution_id=graph_exec_id), user=ADMIN
    )


async def _requeue_all_at_once(
    graph_exec_id: str, times: int
) -> list[RequeueExecutionResponse]:
    """Requeue ``times`` times, every requeue reading the run before any of
    them publishes, so each one finds it still QUEUED."""
    read_run = executor_utils.execution_db.get_graph_execution
    all_read = asyncio.Barrier(times)

    async def read_then_wait_for_the_others(*args, **kwargs):
        row = await read_run(*args, **kwargs)
        await asyncio.wait_for(all_read.wait(), timeout=30)
        return row

    with patch.object(
        executor_utils.execution_db,
        "get_graph_execution",
        new=read_then_wait_for_the_others,
    ):
        return list(
            await asyncio.gather(*(_requeue(graph_exec_id) for _ in range(times)))
        )


async def _deliver(held: list[dict[str, Any]]) -> None:
    """Publish held messages through the real execution queue client."""
    queue = await get_async_execution_queue()
    for message in held:
        await queue.publish_message(**message)


async def _wait_for(
    check: Callable[[], Awaitable[T | None]], what: str, timeout: float = 90
) -> T:
    for _ in range(int(timeout)):
        if result := await check():
            return result
        await asyncio.sleep(1)
    raise AssertionError(f"Timed out waiting for {what}")


def _queue_state(name: str) -> tuple[int, int]:
    """Ready messages and consumers of a queue, as RabbitMQ counts them now.

    Asked over a connection of its own: the execution queue client's
    aio-pika channel answers a passive declare of a queue it declared itself
    with the counts it saw then."""
    client = SyncRabbitMQ(create_execution_queue_config())
    client.connect()
    try:
        declared = client.get_channel().queue_declare(queue=name, passive=True)
        return declared.method.message_count, declared.method.consumer_count
    finally:
        client.disconnect()


async def _run_queue_has_two_consumers() -> bool:
    _, consumers = await asyncio.to_thread(_queue_state, GRAPH_EXECUTION_QUEUE_NAME)
    return consumers >= 2


async def _run_queues_drained() -> bool:
    """No delivery waiting in the run queue or the delay queue, three checks
    in a row, a second apart: a duplicate an executor turned away comes back
    to the queue for a moment before it is handled again."""
    for attempt in range(3):
        if attempt:
            await asyncio.sleep(1)
        for name in (GRAPH_EXECUTION_QUEUE_NAME, GRAPH_EXECUTION_DELAY_QUEUE_NAME):
            ready, _ = await asyncio.to_thread(_queue_state, name)
            if ready:
                return False
    return True


async def _row_in(
    graph_exec_id: str,
    status: ExecutionStatus,
    updated_after: AgentGraphExecution | None = None,
) -> AgentGraphExecution | None:
    row = await AgentGraphExecution.prisma().find_unique(where={"id": graph_exec_id})
    if row is None or row.executionStatus != status:
        return None
    if updated_after and row.updatedAt == updated_after.updatedAt:
        return None
    return row


async def _node_execs(user: User, graph_exec_id: str):
    run = await execution.get_graph_execution(
        user.id, graph_exec_id, include_node_executions=True
    )
    assert run
    return run.node_executions


def _node_of(test_graph: graph.Graph, block: type) -> str:
    (node,) = [n for n in test_graph.nodes if n.block_id == block().id]
    return node.id


async def _assert_ran_once(
    user: User, graph_exec_id: str, test_graph: graph.Graph
) -> None:
    meta = await execution.get_graph_execution_meta(
        user_id=user.id, execution_id=graph_exec_id
    )
    assert meta and meta.status == ExecutionStatus.COMPLETED
    node_execs = await _node_execs(user, graph_exec_id)
    assert sorted(ne.node_id for ne in node_execs) == sorted(
        node.id for node in test_graph.nodes
    )
    assert all(ne.status == ExecutionStatus.COMPLETED for ne in node_execs)
