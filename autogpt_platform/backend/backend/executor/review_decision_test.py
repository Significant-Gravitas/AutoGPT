"""A person's answer to a review decides the step it was asked about, on the
real stack: the database, RabbitMQ, the Redis cluster lock and the executor.

The answer stored on a step is read before its safe mode, which only decides
whether to ask when the step has no answer. A rejection stops the step whatever
the safe mode, or the queue message that runs it, says now; an "approve future"
stands in only for a step that has no answer of its own.

The tests only decide when a delivery reaches the queue; what each piece of
the code under test does with it is never patched out.
"""

import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus
from prisma.models import AgentGraphExecution, PendingHumanReview
from pydantic import BaseModel, SecretStr

from backend.api.features.graph_executions.review.model import ReviewItem
from backend.api.features.graph_executions.review.service import process_reviews
from backend.blocks.basic import StoreValueBlock
from backend.blocks.email_block import SendEmailBlock
from backend.blocks.helpers.review import SAFE_MODE_OFF_MESSAGE
from backend.blocks.human_in_the_loop import HumanInTheLoopBlock
from backend.blocks.io import AgentInputBlock
from backend.blocks.iteration import StepThroughItemsBlock
from backend.copilot.tools._test_data import make_session
from backend.copilot.tools.models import (
    BlockOutputResponse,
    ErrorResponse,
    ReviewRequiredResponse,
)
from backend.copilot.tools.resume_capability import ResumeCapabilityTool
from backend.copilot.tools.run_block import RunBlockTool
from backend.data import graph
from backend.data.execution import ExecutionStatus
from backend.data.human_review import (
    get_pending_reviews_for_execution,
    process_all_reviews_for_execution,
)
from backend.data.model import User, UserPasswordCredentials
from backend.executor.admin_requeue_test import (
    _assert_ran_once,
    _assert_waiting_for_review,
    _create_graph,
    _deliver,
    _graph_with_a_review,
    _let_the_executor_catch_up,
    _node_execs,
    _node_of,
    _requeue,
    _row_in,
    _set_safe_mode,
    _start_with_delivery_held,
    _wait_for,
)
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.usecases.sample import create_test_user
from backend.util.test import SpinTestServer, wait_execution

HITL = "human_in_the_loop_safe_mode"
SENSITIVE = "sensitive_action_safe_mode"
SAFE_MODES = [HITL, SENSITIVE]
REJECTION = "Not this one"
# What the email step answers when it runs: it refuses the port before it
# connects anywhere, so a step that runs is seen without a message leaving.
EMAIL_STEP_RAN = "SMTP port 1 is not allowed"
EMAIL = {
    "to_email": "someone@example.com",
    "subject": "Checkout is down",
    "body": "The checkout is down.",
    "config": {"smtp_server": "smtp.example.com", "smtp_port": 1},
}
SENT_EMAIL = {**EMAIL, "config": {"smtp_server": "smtp.example.com", "smtp_port": 25}}


@pytest.fixture(autouse=True)
def _runners_belong_to_their_org():
    """Check memberships for real: the executor service checks a run's
    org/team in its own process."""


@pytest.fixture(autouse=True)
def _default_team_untenanted():
    """Look up the user's own org and team for real, as production does."""


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("safe_mode", SAFE_MODES)
async def test_a_rejected_step_stays_stopped_after_the_safe_mode_is_turned_off(
    server: SpinTestServer, safe_mode: str
):
    """The run stopped for review; the user turned the safe mode off, then
    rejected the step. The resume carries the safe mode as it is now, off, and
    the step must still not run."""
    user = await create_test_user()
    async with _an_smtp_login(user) as login:
        gated = await _graph_gated_by(server, user, safe_mode, login)
        await _set_safe_mode(user, gated.workflow, safe_mode, on=True)
        run = await _start(server, user, gated)
        await _wait_for(
            lambda: _row_in(run.id, ExecutionStatus.REVIEW), "the run to reach review"
        )
        await _set_safe_mode(user, gated.workflow, safe_mode, on=False)

        review = await _answer(user, run.id, approved=False)
        await _wait_for(lambda: _ended(run.id), "the run to end")

        await _assert_stopped_by_the_rejection(user, run.id, gated, review)


@pytest.mark.asyncio(loop_scope="session")
async def test_an_approve_future_on_another_item_does_not_run_a_rejected_one(
    server: SpinTestServer,
):
    """One step stopped for review twice, once per item. The user rejected the
    first item and approved the second with "approve future" for that step.
    The approval stands in for later items with no answer of their own; the
    first item has one, a rejection, and must not run."""
    user = await create_test_user()
    test_graph = await _create_graph(server, _graph_reviewing_each_item(), user)
    run = await server.agent_server.test_execute_graph(
        user_id=user.id,
        graph_id=test_graph.id,
        graph_version=test_graph.version,
        node_input={"items": ["first", "second"]},
    )
    await _wait_for(
        lambda: _row_in(run.id, ExecutionStatus.REVIEW), "the run to reach review"
    )
    reviews = {
        r.payload: r for r in await get_pending_reviews_for_execution(run.id, user.id)
    }
    assert set(reviews) == {"first", "second"}

    await process_reviews(
        user.id,
        [
            ReviewItem(
                node_exec_id=reviews["first"].node_exec_id,
                approved=False,
                message=REJECTION,
            ),
            ReviewItem(
                node_exec_id=reviews["second"].node_exec_id,
                approved=True,
                auto_approve_future=True,
            ),
        ],
    )
    await wait_execution(user.id, run.id, 60)

    store = _node_of(test_graph, StoreValueBlock)
    stored = [
        ne.input_data.get("input")
        for ne in await _node_execs(user, run.id)
        if ne.node_id == store and ne.status == ExecutionStatus.COMPLETED
    ]
    assert stored == ["second"], f"the rejected item ran: {stored}"
    await _assert_answered(reviews["first"].node_exec_id, ReviewStatus.REJECTED)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("safe_mode", SAFE_MODES)
async def test_a_late_original_with_a_stale_safe_mode_does_not_run_a_rejected_step(
    server: SpinTestServer, safe_mode: str
):
    """The run was queued while the safe mode was off, and its original queue
    message, only late, still carries that. The user turned the safe mode on,
    an admin requeue ran the run up to a review, and the user rejected it. The
    rejection puts the run back in the queue, and the late original reaches
    the executor before the resume's own message does. It must not run the
    rejected step on its stale safe mode, and the rejection must stand."""
    user = await create_test_user()
    async with _an_smtp_login(user) as login:
        gated = await _graph_gated_by(server, user, safe_mode, login)
        await _set_safe_mode(user, gated.workflow, safe_mode, on=False)
        original: list[dict[str, Any]] = []
        run = await _start_with_delivery_held(
            server, user, gated.workflow, gated.inputs, original
        )
        await _set_safe_mode(user, gated.workflow, safe_mode, on=True)
        assert (await _requeue(run.id)).requeued_count == 1
        await _wait_for(
            lambda: _row_in(run.id, ExecutionStatus.REVIEW), "the run to reach review"
        )

        resume: list[dict[str, Any]] = []
        with _publishing_held(resume):
            review = await _answer(user, run.id, approved=False)
        assert resume and await _row_in(run.id, ExecutionStatus.QUEUED)

        await _deliver(original)
        await _wait_for(lambda: _ended(run.id), "the late original to run the run")
        await _deliver(resume)
        await _let_the_executor_catch_up(server, user, run.id)

        await _assert_stopped_by_the_rejection(user, run.id, gated, review)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("safe_mode", SAFE_MODES)
async def test_a_stale_delivery_never_runs_a_step_whose_review_still_waits(
    server: SpinTestServer, safe_mode: str
):
    """The run was queued while the safe mode was off. The user turned it on,
    an admin requeue ran the run up to a review, and the review still waits.
    An executor pod from before #15328 resumes such a run from any delivery
    (#15328's rollout window), and the late original then runs it with the
    safe mode it was queued with, off. The step must keep waiting for its
    answer, the review must stay unanswered and the run parked."""
    user = await create_test_user()
    async with _an_smtp_login(user) as login:
        gated = await _graph_gated_by(server, user, safe_mode, login)
        await _set_safe_mode(user, gated.workflow, safe_mode, on=False)
        original: list[dict[str, Any]] = []
        run = await _start_with_delivery_held(
            server, user, gated.workflow, gated.inputs, original
        )
        await _set_safe_mode(user, gated.workflow, safe_mode, on=True)
        assert (await _requeue(run.id)).requeued_count == 1
        await _wait_for(
            lambda: _row_in(run.id, ExecutionStatus.REVIEW), "the run to reach review"
        )

        # What such a pod does with a delivery that finds the run in REVIEW.
        await AgentGraphExecution.prisma().update(
            where={"id": run.id}, data={"executionStatus": ExecutionStatus.QUEUED}
        )
        await _deliver(original)
        await _wait_for(lambda: _settled(run.id), "the late original to run the run")
        await _let_the_executor_catch_up(server, user, run.id)

        await _assert_waiting_for_review(user, run.id, gated.step)


@pytest.mark.asyncio(loop_scope="session")
async def test_an_approved_step_runs_once_with_the_reviewers_data(
    server: SpinTestServer,
):
    """The user edited the data under review and approved it, after turning
    the safe mode off. The step runs once, on the data the user approved."""
    user = await create_test_user()
    test_graph = await _create_graph(server, _graph_with_a_review(), user)
    await _set_safe_mode(user, test_graph, HITL, on=True)
    run = await server.agent_server.test_execute_graph(
        user_id=user.id,
        graph_id=test_graph.id,
        graph_version=test_graph.version,
        node_input={"value": "draft"},
    )
    await _wait_for(
        lambda: _row_in(run.id, ExecutionStatus.REVIEW), "the run to reach review"
    )
    await _set_safe_mode(user, test_graph, HITL, on=False)

    await _answer(user, run.id, approved=True, reviewed_data="edited")
    await wait_execution(user.id, run.id, 60)

    await _assert_ran_once(user, run.id, test_graph)
    store = _node_of(test_graph, StoreValueBlock)
    (stored,) = [
        ne.input_data.get("input")
        for ne in await _node_execs(user, run.id)
        if ne.node_id == store
    ]
    assert stored == "edited"


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize("safe_mode", SAFE_MODES)
@pytest.mark.parametrize("on", [True, False], ids=["safe_mode_on", "safe_mode_off"])
async def test_a_step_with_no_answer_asks_only_when_the_safe_mode_is_on(
    server: SpinTestServer, safe_mode: str, on: bool
):
    user = await create_test_user()
    async with _an_smtp_login(user) as login:
        gated = await _graph_gated_by(server, user, safe_mode, login)
        await _set_safe_mode(user, gated.workflow, safe_mode, on=on)
        run = await _start(server, user, gated)

        if on:
            await _wait_for(
                lambda: _row_in(run.id, ExecutionStatus.REVIEW),
                "the run to reach review",
            )
            await _assert_waiting_for_review(user, run.id, gated.step)
            return

        await _wait_for(lambda: _ended(run.id), "the run to end")
        assert not await PendingHumanReview.prisma().find_many(
            where={"graphExecId": run.id}
        )
        if safe_mode == HITL:
            await _assert_ran_once(user, run.id, gated.workflow)
            (asked,) = await _execs_of(user, run.id, gated.reviewed)
            assert asked.output_data.get("review_message") == [SAFE_MODE_OFF_MESSAGE]
        else:
            (email,) = await _execs_of(user, run.id, gated.step)
            assert EMAIL_STEP_RAN in str(email.output_data.get("error"))


@pytest.mark.asyncio(loop_scope="session")
async def test_an_approved_email_step_in_a_workflow_runs_with_its_login(
    server: SpinTestServer,
):
    """The safe mode is turned off before the approval, so the stored answer
    decides; the step runs with the login the run resolved."""
    user = await create_test_user()
    async with _an_smtp_login(user) as login:
        error = await _approve_and_resume_in_workflow(server, user, login)
    assert EMAIL_STEP_RAN in error, error


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "gate_decided", [False, True], ids=["no_gate_verdict", "unified_approvals_off"]
)
async def test_an_approved_email_from_a_chat_is_sent_once_with_the_users_login(
    server: SpinTestServer, gate_decided: bool
):
    """Otto asks to send an email; the chat's approval gate gives no verdict
    (auto mode off, its default), or unified approvals are off, so the user
    gets a card. The user approves it and Otto resumes it with no input
    overrides. The stored payload has no login, since a chat adds the login
    when the call runs: the email goes out once, logged in with the user's
    SMTP login, and the approval is spent, so resuming again sends nothing."""
    user = await create_test_user()
    scope = f"smtp://{SENT_EMAIL['config']['smtp_server']}:25"
    async with _an_smtp_login(user, scope=scope) as login:
        session = make_session(user_id=user.id)
        with patch(
            "backend.copilot.tools.run_block.unified_approvals",
            AsyncMock(return_value=False),
        ):
            asked = await RunBlockTool()._execute(
                user_id=user.id,
                session=session,
                block_id=SendEmailBlock().id,
                input_data=dict(SENT_EMAIL),
                gate_decided=gate_decided,
            )
        assert isinstance(asked, ReviewRequiredResponse), asked
        await process_all_reviews_for_execution(
            user.id, {asked.review_id: (ReviewStatus.APPROVED, None, None)}
        )

        with _a_mail_server() as mail:
            resume = ResumeCapabilityTool()
            sent = await resume._execute(user.id, session, review_id=asked.review_id)
            again = await resume._execute(user.id, session, review_id=asked.review_id)

    assert isinstance(sent, BlockOutputResponse), sent
    password = login.password.get_secret_value()
    assert mail.logins == [(login.username.get_secret_value(), password)]
    assert mail.sent == [SENT_EMAIL["to_email"]]
    assert isinstance(again, ErrorResponse) and "already executed" in again.message
    assert not await PendingHumanReview.prisma().find_unique(
        where={"nodeExecId": asked.review_id}
    )


class _Gated(BaseModel):
    """A workflow with a step that stops for review, and what to look at."""

    workflow: graph.Graph
    inputs: dict[str, Any]
    # The node the review is asked on, and the step that must not run without
    # the user's yes: the step after a human-in-the-loop node, or the
    # irreversible step itself.
    reviewed: str
    step: str


async def _graph_gated_by(
    server: SpinTestServer,
    user: User,
    safe_mode: str,
    login: UserPasswordCredentials,
) -> _Gated:
    if safe_mode == HITL:
        test_graph = await _create_graph(server, _graph_with_a_review(), user)
        return _Gated(
            workflow=test_graph,
            inputs={"value": "draft"},
            reviewed=_node_of(test_graph, HumanInTheLoopBlock),
            step=_node_of(test_graph, StoreValueBlock),
        )

    # An irreversible block stops for review before it runs.
    nodes = [
        graph.Node(block_id=AgentInputBlock().id, input_default={"name": "subject"}),
        graph.Node(
            block_id=SendEmailBlock().id,
            input_default={
                "to_email": EMAIL["to_email"],
                "body": EMAIL["body"],
                "config": EMAIL["config"],
                "credentials": {
                    "id": login.id,
                    "provider": login.provider,
                    "type": login.type,
                    "title": login.title,
                },
            },
        ),
    ]
    links = [
        graph.Link(
            source_id=nodes[0].id,
            sink_id=nodes[1].id,
            source_name="result",
            sink_name="subject",
        )
    ]
    test_graph = await _create_graph(
        server,
        graph.Graph(
            name="EmailGraph", description="Send an email", nodes=nodes, links=links
        ),
        user,
    )
    email = _node_of(test_graph, SendEmailBlock)
    return _Gated(
        workflow=test_graph,
        inputs={"subject": EMAIL["subject"]},
        reviewed=email,
        step=email,
    )


def _graph_reviewing_each_item() -> graph.Graph:
    nodes = [
        graph.Node(block_id=AgentInputBlock().id, input_default={"name": "items"}),
        graph.Node(block_id=StepThroughItemsBlock().id),
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
            sink_name="items",
        ),
        graph.Link(
            source_id=nodes[1].id,
            sink_id=nodes[2].id,
            source_name="item",
            sink_name="data",
        ),
        graph.Link(
            source_id=nodes[2].id,
            sink_id=nodes[3].id,
            source_name="approved_data",
            sink_name="input",
        ),
    ]
    return graph.Graph(
        name="EachItemReviewGraph",
        description="Input, each item reviewed, store",
        nodes=nodes,
        links=links,
    )


@asynccontextmanager
async def _an_smtp_login(
    user: User, scope: str | None = None
) -> AsyncIterator[UserPasswordCredentials]:
    """An SMTP login the user holds for the test only. It never reaches a
    real server: the email step refuses port 1 before connecting anywhere,
    and a chat's send goes to _a_mail_server."""
    login = UserPasswordCredentials(
        id=str(uuid.uuid4()),
        provider="smtp",
        username=SecretStr("sender@example.com"),
        password=SecretStr("not-a-real-password"),
        title="SMTP",
        scope=scope,
    )
    store = IntegrationCredentialsStore()
    await store.add_creds(user.id, login)
    try:
        yield login
    finally:
        await store.delete_creds_by_id(user.id, login.id)


class _MailServer(BaseModel):
    """What reached the mail server: logins, and each message's recipient."""

    logins: list[tuple[str, str]] = []
    sent: list[str] = []


@contextmanager
def _a_mail_server() -> Iterator[_MailServer]:
    """The mail server at the network edge, in this process: the email step,
    its login and the review gate all run for real, and only the SMTP
    connection and the server name's lookup are stood in for."""
    mail = _MailServer()

    class _Connection:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def starttls(self, **_kwargs):
            pass

        def login(self, username: str, password: str):
            mail.logins.append((username, password))

        def sendmail(self, _sender: str, to: str, _message: str):
            mail.sent.append(to)

    with (
        patch(
            "backend.blocks.email_block.resolve_and_check_blocked",
            AsyncMock(return_value=["203.0.113.10"]),
        ),
        patch("backend.blocks.email_block.PinnedSMTP", _Connection),
    ):
        yield mail


async def _approve_and_resume_in_workflow(
    server: SpinTestServer, user: User, login: UserPasswordCredentials
) -> str:
    """Run the email workflow to its review, turn the safe mode off, approve;
    returns the email step's error."""
    gated = await _graph_gated_by(server, user, SENSITIVE, login)
    await _set_safe_mode(user, gated.workflow, SENSITIVE, on=True)
    run = await _start(server, user, gated)
    await _wait_for(
        lambda: _row_in(run.id, ExecutionStatus.REVIEW), "the run to reach review"
    )
    await _set_safe_mode(user, gated.workflow, SENSITIVE, on=False)
    await _answer(user, run.id, approved=True)
    await _wait_for(lambda: _ended(run.id), "the run to end")
    (email,) = await _execs_of(user, run.id, gated.step)
    return str(email.output_data.get("error"))


async def _start(server: SpinTestServer, user: User, gated: _Gated):
    return await server.agent_server.test_execute_graph(
        user_id=user.id,
        graph_id=gated.workflow.id,
        graph_version=gated.workflow.version,
        node_input=gated.inputs,
    )


async def _answer(
    user: User, graph_exec_id: str, *, approved: bool, reviewed_data: Any = None
) -> str:
    """Answer the run's one review through the review service, as the
    review routes do; returns the step's id."""
    (review,) = await get_pending_reviews_for_execution(graph_exec_id, user.id)
    await process_reviews(
        user.id,
        [
            ReviewItem(
                node_exec_id=review.node_exec_id,
                approved=approved,
                message=None if approved else REJECTION,
                reviewed_data=reviewed_data,
            )
        ],
    )
    return review.node_exec_id


@contextmanager
def _publishing_held(held: list[dict[str, Any]]) -> Iterator[None]:
    """Keep what the run's resume publishes in ``held`` instead of the queue."""
    queue = MagicMock(publish_message=AsyncMock(side_effect=lambda **m: held.append(m)))
    with patch(
        "backend.executor.utils.get_async_execution_queue",
        new=AsyncMock(return_value=queue),
    ):
        yield


async def _ended(graph_exec_id: str) -> bool:
    for status in (
        ExecutionStatus.COMPLETED,
        ExecutionStatus.FAILED,
        ExecutionStatus.TERMINATED,
    ):
        if await _row_in(graph_exec_id, status):
            return True
    return False


async def _settled(graph_exec_id: str) -> bool:
    """The run ended or stopped for review again."""
    return bool(await _row_in(graph_exec_id, ExecutionStatus.REVIEW)) or await _ended(
        graph_exec_id
    )


async def _execs_of(user: User, graph_exec_id: str, node_id: str):
    return [
        ne for ne in await _node_execs(user, graph_exec_id) if ne.node_id == node_id
    ]


async def _assert_stopped_by_the_rejection(
    user: User, graph_exec_id: str, gated: _Gated, review: str
) -> None:
    """The step behind the rejected review never ran, the reviewed node took
    the rejection, and the review still holds the user's answer."""
    reviewed = await _execs_of(user, graph_exec_id, gated.reviewed)
    if gated.reviewed == gated.step:
        (email,) = reviewed
        error = str(email.output_data.get("error"))
        assert email.status == ExecutionStatus.FAILED
        assert REJECTION in error and EMAIL_STEP_RAN not in error, error
    else:
        (asked,) = reviewed
        assert "rejected_data" in asked.output_data, asked.output_data
        assert "approved_data" not in asked.output_data, asked.output_data
        ran = [
            ne.status
            for ne in await _execs_of(user, graph_exec_id, gated.step)
            if ne.status == ExecutionStatus.COMPLETED
        ]
        assert not ran, "the step behind the rejected review ran"
    await _assert_answered(review, ReviewStatus.REJECTED)


async def _assert_answered(node_exec_id: str, status: ReviewStatus) -> None:
    stored = await PendingHumanReview.prisma().find_unique(
        where={"nodeExecId": node_exec_id}
    )
    assert stored and stored.status == status
    if status == ReviewStatus.REJECTED:
        assert stored.reviewMessage == REJECTION
