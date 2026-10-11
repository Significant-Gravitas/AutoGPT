"""What a resumed run does with the answer to its review.

Real graph, execution and review rows, the real ``POST /api/review/action``
and the real gate in ``Block.execute``. A node is parked the way the executor
parks it, the user answers through the route, and the resume entry the route
publishes is replayed through ``Block.execute`` with that entry's context, as
the executor would. Only the queue, the outbound GitHub call and the user's
library safe-mode settings are stubbed.
"""

from typing import Any, AsyncGenerator, Awaitable, Callable, Sequence

import httpx
import pytest
import pytest_asyncio
import pytest_mock
from pydantic import BaseModel, ConfigDict

from backend.api.rest_api import app
from backend.blocks._base import Block
from backend.blocks.github._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.github.issues import GithubMakeIssueBlock
from backend.blocks.human_in_the_loop import HumanInTheLoopBlock
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    GraphExecutionEntry,
    create_graph_execution,
    get_graph_execution_meta,
    update_graph_execution_stats,
)
from backend.data.graph import (
    Graph,
    GraphSettings,
    Node,
    create_graph,
    make_graph_model,
)
from backend.executor.utils import add_graph_execution
from backend.util.exceptions import BlockExecutionError

ISSUE_INPUT = {
    "repo_url": "https://github.com/owner/repo",
    "title": "Refund customer 4411",
    "body": "Send the refund.",
    "credentials": TEST_CREDENTIALS_INPUT,
}


@pytest_asyncio.fixture(loop_scope="session")
async def client(server, mock_jwt_user) -> AsyncGenerator[httpx.AsyncClient, None]:
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http_client:
        yield http_client
    app.dependency_overrides.pop(get_jwt_payload, None)


class Parked(BaseModel):
    """A run whose node is waiting for review, once per iteration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_exec_id: str
    node_exec_ids: list[str]
    run_node: Callable[..., Awaitable[list[tuple[str, Any]]]]

    @property
    def node_exec_id(self) -> str:
        return self.node_exec_ids[0]


@pytest.fixture
def queue(mocker: pytest_mock.MockerFixture):
    queue = mocker.AsyncMock()
    mocker.patch(
        "backend.executor.utils.get_async_execution_queue",
        new=mocker.AsyncMock(return_value=queue),
    )
    return queue


@pytest.fixture
def create_issue(mocker: pytest_mock.MockerFixture):
    """The external side effect: the issue GitHub would create."""
    return mocker.patch.object(
        GithubMakeIssueBlock,
        "create_issue",
        new=mocker.AsyncMock(
            return_value=(1, "https://github.com/owner/repo/issues/1")
        ),
    )


def settings_at_answer_time(mocker: pytest_mock.MockerFixture, **settings) -> None:
    """The user's library settings when they answer, read again by the resume."""
    mocker.patch(
        "backend.api.features.graph_executions.review.routes.get_graph_settings",
        new=mocker.AsyncMock(return_value=GraphSettings(**settings)),
    )


async def park_for_review(
    block: Block,
    input_data: dict[str, Any],
    user_id: str,
    *,
    dry_run: bool = False,
    more_iterations: Sequence[dict[str, Any]] = (),
) -> Parked:
    iterations = [input_data, *more_iterations]
    graph = make_graph_model(
        Graph(
            name=f"Review resume: {block.name}",
            description="",
            nodes=[Node(block_id=block.id)],
            links=[],
        ),
        user_id,
    )
    graph.reassign_ids(user_id=user_id, reassign_graph_id=True)
    await create_graph(graph, user_id=user_id)
    node_id = graph.nodes[0].id
    graph_exec = await create_graph_execution(
        graph_id=graph.id,
        graph_version=graph.version,
        starting_nodes_input=[(node_id, data) for data in iterations],
        inputs={},
        user_id=user_id,
        is_dry_run=dry_run,
    )
    node_exec_ids = [
        next(
            ne.node_exec_id
            for ne in graph_exec.node_executions
            if ne.input_data == data
        )
        for data in iterations
    ]

    async def run_node(
        context: ExecutionContext, iteration: int = 0
    ) -> list[tuple[str, Any]]:
        node_exec_id = node_exec_ids[iteration]
        return [
            output
            async for output in block.execute(
                iterations[iteration],
                execution_context=context.model_copy(
                    update={"node_id": node_id, "node_exec_id": node_exec_id}
                ),
                credentials=TEST_CREDENTIALS,
                user_id=user_id,
                node_id=node_id,
                node_exec_id=node_exec_id,
                graph_exec_id=graph_exec.id,
                graph_id=graph.id,
                graph_version=graph.version,
            )
        ]

    # Safe mode is on when the node first runs, so the gate holds each one.
    first_run = ExecutionContext(
        user_id=user_id,
        graph_id=graph.id,
        graph_exec_id=graph_exec.id,
        graph_version=graph.version,
        root_execution_id=graph_exec.id,
        human_in_the_loop_safe_mode=True,
        sensitive_action_safe_mode=True,
        dry_run=dry_run,
    )
    for iteration in range(len(iterations)):
        assert await run_node(first_run, iteration) == []
    assert await update_graph_execution_stats(
        graph_exec_id=graph_exec.id, status=ExecutionStatus.REVIEW
    )
    return Parked(
        graph_exec_id=graph_exec.id, node_exec_ids=node_exec_ids, run_node=run_node
    )


async def answer(
    client: httpx.AsyncClient, node_exec_id: str, **review: Any
) -> httpx.Response:
    response = await client.post(
        "/api/review/action",
        json={"reviews": [{"node_exec_id": node_exec_id, **review}]},
    )
    assert response.status_code == 200, response.text
    return response


def resumed_context(queue) -> ExecutionContext:
    queue.publish_message.assert_awaited_once()
    return GraphExecutionEntry.model_validate_json(
        queue.publish_message.call_args.kwargs["message"]
    ).execution_context


@pytest.mark.asyncio(loop_scope="session")
async def test_a_rejected_action_never_runs_after_safe_mode_is_switched_off(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    queue,
    create_issue,
    test_user_id: str,
) -> None:
    """The user turns safe mode off, then rejects what was already held.

    The resume reads safe mode as off, and the gate used to skip the review
    lookup whenever safe mode was off, so the rejected issue was created.
    """
    parked = await park_for_review(GithubMakeIssueBlock(), ISSUE_INPUT, test_user_id)
    settings_at_answer_time(mocker, sensitive_action_safe_mode=False)

    response = await answer(client, parked.node_exec_id, approved=False, message="No")
    assert response.json()["rejected_count"] == 1

    resumed = resumed_context(queue)
    assert resumed.sensitive_action_safe_mode is False
    with pytest.raises(BlockExecutionError, match="rejected by reviewer"):
        await parked.run_node(resumed)
    create_issue.assert_not_awaited()


@pytest.mark.asyncio(loop_scope="session")
async def test_an_approved_action_runs_once_as_approved_after_safe_mode_is_off(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    queue,
    create_issue,
    test_user_id: str,
) -> None:
    """The reverse: an approval runs the action once, with the reviewer's edit,
    rather than the input the reviewer was shown."""
    parked = await park_for_review(GithubMakeIssueBlock(), ISSUE_INPUT, test_user_id)
    settings_at_answer_time(mocker, sensitive_action_safe_mode=False)
    edited = {**ISSUE_INPUT, "title": "Refund customer 4411 (approved amount)"}

    response = await answer(
        client, parked.node_exec_id, approved=True, reviewed_data=edited
    )
    assert response.json()["approved_count"] == 1

    outputs = await parked.run_node(resumed_context(queue))
    assert ("number", 1) in outputs
    create_issue.assert_awaited_once()
    assert create_issue.await_args.args[1:] == (
        edited["repo_url"],
        edited["title"],
        edited["body"],
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_approving_future_runs_does_not_override_a_rejection_of_another(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    queue,
    create_issue,
    test_user_id: str,
) -> None:
    """Two iterations of one node are held. The user approves the first with
    "approve future" and rejects the second, with safe mode still on.

    The gate looked for any approval first, and the node-wide auto-approval
    matched the rejected iteration too, so it ran without its own answer ever
    being read.
    """
    approved = {**ISSUE_INPUT, "title": "Refund customer 4411"}
    rejected = {**ISSUE_INPUT, "title": "Refund customer 9999"}
    parked = await park_for_review(
        GithubMakeIssueBlock(), approved, test_user_id, more_iterations=[rejected]
    )
    settings_at_answer_time(mocker, sensitive_action_safe_mode=True)

    response = await client.post(
        "/api/review/action",
        json={
            "reviews": [
                {
                    "node_exec_id": parked.node_exec_ids[0],
                    "approved": True,
                    "auto_approve_future": True,
                },
                {
                    "node_exec_id": parked.node_exec_ids[1],
                    "approved": False,
                    "message": "Not this one",
                },
            ]
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["failed_count"] == 0

    resumed = resumed_context(queue)
    assert resumed.sensitive_action_safe_mode is True
    with pytest.raises(BlockExecutionError, match="rejected by reviewer"):
        await parked.run_node(resumed, 1)
    create_issue.assert_not_awaited()

    await parked.run_node(resumed, 0)
    create_issue.assert_awaited_once()
    assert create_issue.await_args.args[2] == approved["title"]


@pytest.mark.asyncio(loop_scope="session")
async def test_a_rejected_review_block_takes_the_rejected_branch_after_safe_mode_is_off(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    queue,
    test_user_id: str,
) -> None:
    """The Human-in-the-loop block had the same order: safe mode off meant
    ``approved_data``, even for data the human had just rejected."""
    data = {"refund": 4411}
    parked = await park_for_review(
        HumanInTheLoopBlock(),
        {"data": data, "name": "Refund", "editable": True},
        test_user_id,
    )
    settings_at_answer_time(mocker, human_in_the_loop_safe_mode=False)

    await answer(client, parked.node_exec_id, approved=False, message="No")

    resumed = resumed_context(queue)
    assert resumed.human_in_the_loop_safe_mode is False
    outputs = dict(await parked.run_node(resumed))
    assert "approved_data" not in outputs
    assert outputs["rejected_data"] == data


@pytest.mark.asyncio(loop_scope="session")
async def test_a_dry_run_stays_dry_when_its_review_is_approved(
    client: httpx.AsyncClient,
    mocker: pytest_mock.MockerFixture,
    queue,
    test_user_id: str,
) -> None:
    """The executor reads ``dry_run`` only from the published context: it
    decides whether blocks are simulated and whether ``charge_usage`` and
    ``charge_reconciled_usage`` run. The review route rebuilt that context
    without it, so the rest of a dry run ran for real and was charged.

    Blocks are simulated in a dry run, so none reaches the gate today; the node
    is parked directly to pin the resume. The requeue below is the path a dry
    run can take now.
    """
    parked = await park_for_review(
        GithubMakeIssueBlock(), ISSUE_INPUT, test_user_id, dry_run=True
    )
    settings_at_answer_time(mocker, sensitive_action_safe_mode=True)

    await answer(client, parked.node_exec_id, approved=True)

    assert resumed_context(queue).dry_run is True
    meta = await get_graph_execution_meta(
        user_id=test_user_id, execution_id=parked.graph_exec_id
    )
    assert meta and meta.is_dry_run


@pytest.mark.asyncio(loop_scope="session")
async def test_a_requeued_dry_run_stays_dry(
    server,
    queue,
    test_user_id: str,
) -> None:
    """The admin requeue of a stuck run passes no context and no ``dry_run``."""
    block = GithubMakeIssueBlock()
    graph = make_graph_model(
        Graph(
            name="Requeued dry run",
            description="",
            nodes=[Node(block_id=block.id)],
            links=[],
        ),
        test_user_id,
    )
    graph.reassign_ids(user_id=test_user_id, reassign_graph_id=True)
    await create_graph(graph, user_id=test_user_id)
    graph_exec = await create_graph_execution(
        graph_id=graph.id,
        graph_version=graph.version,
        starting_nodes_input=[(graph.nodes[0].id, ISSUE_INPUT)],
        inputs={},
        user_id=test_user_id,
        is_dry_run=True,
    )

    # Exactly the call the admin diagnostics requeue makes.
    await add_graph_execution(
        graph_id=graph.id,
        user_id=test_user_id,
        graph_version=graph.version,
        graph_exec_id=graph_exec.id,
        bypass_paywall=True,
    )

    assert resumed_context(queue).dry_run is True
