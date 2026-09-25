"""A run a copilot turn started keeps that turn's tree when it is re-queued.

The review route re-queues a paused run with a context it builds from graph
settings, which cannot know the tree, so the tree must come back off the
execution row. Driven end to end: real route, real ``add_graph_execution``,
real database; only the queue publish is captured.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from backend.api.features.library.db import create_library_agent
from backend.api.rest_api import app
from backend.blocks.autopilot import AutoPilotBlock, _spawner_envelope_from
from backend.copilot.executor.utils import _admitted_turn_envelope
from backend.copilot.sdk.session_waiter import SessionResult
from backend.copilot.tree import MAX_DEPTH, TreeRefusal, TurnEnvelope, get_tree_ledger
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    GraphExecutionEntry,
    get_graph_execution_copilot_tree,
    update_graph_execution_stats,
)
from backend.data.graph import create_graph
from backend.data.human_review import (
    get_or_create_human_review,
    get_pending_reviews_for_execution,
)
from backend.data.redis_client import get_redis_async
from backend.executor.utils import add_graph_execution
from backend.usecases.sample import create_test_graph

_INPUTS = {"input_1": "a", "input_2": "b"}


@pytest_asyncio.fixture(loop_scope="session")
async def client(server, mock_jwt_user) -> AsyncGenerator[httpx.AsyncClient, None]:
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http_client:
        yield http_client
    app.dependency_overrides.pop(get_jwt_payload, None)


@pytest.fixture
def published(mocker: MockerFixture) -> list[GraphExecutionEntry]:
    """Every entry ``add_graph_execution`` puts on the execution queue."""
    entries: list[GraphExecutionEntry] = []

    async def publish_message(*, message: str, **_):
        entries.append(GraphExecutionEntry.model_validate_json(message))

    queue = AsyncMock()
    queue.publish_message.side_effect = publish_message
    mocker.patch(
        "backend.executor.utils.get_async_execution_queue",
        new=AsyncMock(return_value=queue),
    )
    mocker.patch(
        "backend.executor.utils.get_async_execution_event_bus",
        return_value=mocker.MagicMock(publish=AsyncMock()),
    )
    mocker.patch(
        "backend.executor.utils.is_user_paywalled", new=AsyncMock(return_value=False)
    )
    return entries


def _child(tools: list[str], depth: int = 2) -> TurnEnvelope:
    return TurnEnvelope(
        tree_id=f"tree-{uuid.uuid4()}",
        depth=depth,
        tainted=True,
        tools=frozenset(tools),
        deadline_at=datetime.now(UTC) + timedelta(hours=1),
        spend_session_id=f"chat-{uuid.uuid4()}",
    )


def _unrestricted_root() -> TurnEnvelope:
    return TurnEnvelope(
        tree_id=f"tree-{uuid.uuid4()}", spend_session_id=f"chat-{uuid.uuid4()}"
    )


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "make_tree",
    [
        lambda: None,
        _unrestricted_root,
        lambda: _child(["run_sub_session", "read_workspace_file"]),
    ],
    ids=["user-started", "unrestricted-root", "child"],
)
async def test_an_approved_review_resumes_the_run_inside_its_tree(
    client: httpx.AsyncClient,
    published: list[GraphExecutionEntry],
    setup_test_user,
    test_user_id: str,
    make_tree,
) -> None:
    tree: TurnEnvelope | None = make_tree()
    graph_exec_id = await _park_a_run(test_user_id, tree, published)

    stored = await get_graph_execution_copilot_tree(test_user_id, graph_exec_id)
    if tree is None:
        assert stored is None
    else:
        # An unrestricted root is JSON null in ``tools``, never a missing row.
        assert stored is not None and "tools" in stored
        assert stored["tools"] == (None if tree.tools is None else sorted(tree.tools))

    resumed = await _approve(client, test_user_id, graph_exec_id, published)

    assert _spawner_envelope_from(resumed) == tree


@pytest.mark.asyncio(loop_scope="session")
async def test_a_resume_after_the_ledger_expired_keeps_the_trees_bounds(
    client: httpx.AsyncClient,
    published: list[GraphExecutionEntry],
    setup_test_user,
    test_user_id: str,
) -> None:
    """Expiry restarts the tree's spend and node accounting (``admit_turn``
    opens a missing ledger), but depth, tools and taint come off the row."""
    tree = _child(["run_sub_session", "read_workspace_file"])
    graph_exec_id = await _park_a_run(test_user_id, tree, published)
    resumed = await _approve(client, test_user_id, graph_exec_id, published)

    ledger = await get_tree_ledger()
    assert not await ledger.exists(tree.tree_id)
    child = await _nested_turn(resumed, test_user_id)
    try:
        assert child.tree_id == tree.tree_id
        assert child.depth == tree.depth + 1
        assert child.tainted
        assert child.spend_session_id == tree.spend_session_id
        assert tree.tools is not None and child.tools is not None
        assert child.tools <= tree.tools
    finally:
        await (await get_redis_async()).delete(ledger.key(tree.tree_id))


@pytest.mark.asyncio(loop_scope="session")
async def test_a_resume_cannot_reset_an_exhausted_depth(
    client: httpx.AsyncClient,
    published: list[GraphExecutionEntry],
    setup_test_user,
    test_user_id: str,
) -> None:
    tree = _child(["run_sub_session"], depth=MAX_DEPTH)
    graph_exec_id = await _park_a_run(test_user_id, tree, published)
    resumed = await _approve(client, test_user_id, graph_exec_id, published)

    with pytest.raises(TreeRefusal):
        await _nested_turn(resumed, test_user_id)


async def _park_a_run(
    user_id: str,
    tree: TurnEnvelope | None,
    published: list[GraphExecutionEntry],
) -> str:
    """Start the run as ``run_agent`` does, then pause it on a review."""
    graph = await create_graph(create_test_graph(), user_id)
    await create_library_agent(graph, user_id)
    graph_exec = await add_graph_execution(
        graph_id=graph.id, user_id=user_id, inputs=_INPUTS, copilot_tree=tree
    )
    assert published[-1].graph_exec_id == graph_exec.id
    for status in (ExecutionStatus.RUNNING, ExecutionStatus.REVIEW):
        await update_graph_execution_stats(graph_exec_id=graph_exec.id, status=status)
    await get_or_create_human_review(
        user_id=user_id,
        node_exec_id=graph_exec.node_executions[0].node_exec_id,
        input_data={"value": 1},
        message="Allow this?",
        editable=False,
        graph_exec_id=graph_exec.id,
        graph_id=graph.id,
        graph_version=graph.version,
    )
    return graph_exec.id


async def _approve(
    client: httpx.AsyncClient,
    user_id: str,
    graph_exec_id: str,
    published: list[GraphExecutionEntry],
) -> ExecutionContext:
    """Approve through the route and return the context the resume queued."""
    before = len(published)
    pending = await get_pending_reviews_for_execution(graph_exec_id, user_id)
    response = await client.post(
        "/api/review/action",
        json={
            "reviews": [
                {"node_exec_id": review.node_exec_id, "approved": True}
                for review in pending
            ]
        },
    )
    assert response.status_code == 200, response.text
    assert len(published) == before + 1
    entry = published[-1]
    assert entry.graph_exec_id == graph_exec_id
    return entry.execution_context


async def _nested_turn(context: ExecutionContext, user_id: str) -> TurnEnvelope:
    """The turn an AutoPilotBlock in the resumed run would get: the block's own
    dispatch kwargs through the real chokepoint and the live tree ledger."""
    turn = AsyncMock(return_value=("refused", SessionResult(refusal="stop here")))
    with patch(
        "backend.copilot.sdk.session_waiter.run_copilot_turn_via_queue", new=turn
    ):
        with pytest.raises(RuntimeError):
            await AutoPilotBlock().execute_copilot(
                prompt="continue",
                system_context="",
                session_id=f"sess-{uuid.uuid4()}",
                max_recursion_depth=3,
                user_id=user_id,
                spawner_envelope=_spawner_envelope_from(context),
            )
    assert turn.await_args is not None
    kwargs = turn.await_args.kwargs
    return await _admitted_turn_envelope(
        f"turn-{uuid.uuid4()}",
        kwargs["session_id"],
        user_id,
        kwargs["permissions"],
        kwargs["spawn"],
        kwargs["spawner_envelope"],
    )
