"""A workflow step Otto started, parked on its card in the starting chat, against
Postgres, Redis and the real review route.

The step pauses through the block's own review; only the queue publish of the
resume is captured, so everything a card answer touches is what ships.
"""

import uuid
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
from prisma.enums import AgentExecutionStatus, ReviewStatus
from prisma.models import (
    AgentGraph,
    AgentGraphExecution,
    AgentNode,
    AgentNodeExecution,
    User,
)

from backend.api.rest_api import app
from backend.blocks.discord.bot_blocks import SendDiscordMessageBlock
from backend.blocks.helpers.review import HITLReviewHelper
from backend.copilot.constants import step_card_id
from backend.copilot.gate import RUNS_KEY, chat_rules, check_action, held
from backend.copilot.gate.classifier import Judgement
from backend.copilot.model import (
    ChatSession,
    get_chat_session,
    update_session_autopilot_mode,
    upsert_chat_session,
)
from backend.copilot.tools.run_capability import RunCapabilityTool
from backend.data.execution import ExecutionContext, GraphExecutionEntry
from backend.data.human_review import (
    cancel_pending_reviews_for_execution,
    get_pending_reviews_for_user,
    get_reviews_by_node_exec_ids,
    has_pending_reviews_for_graph_exec,
)
from backend.data.redis_client import get_redis_async
from backend.util.exceptions import BlockExecutionError
from backend.util.feature_flag import Flag

_ROUTES = "backend.api.features.graph_executions.review.routes"
_BLOCK = SendDiscordMessageBlock()
_INPUT = {"channel_name": "general", "message_content": "The digest is out"}
_KEY = f"block:{_BLOCK.id}"
pytestmark = pytest.mark.asyncio(loop_scope="session")


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
def flags():
    """Auto mode on for this user; unified approvals as each test sets it."""
    values = {Flag.COPILOT_AUTO_MODE: True, Flag.COPILOT_UNIFIED_APPROVALS: True}

    async def read(flag, user_id, default=False):
        return values[flag]

    with patch("backend.copilot.gate.is_feature_enabled", AsyncMock(side_effect=read)):
        yield values


@pytest.fixture
def resumed():
    with patch(f"{_ROUTES}.add_graph_execution", AsyncMock()) as add:
        yield add


class _Run:
    def __init__(self, user_id: str, chat_id: str | None) -> None:
        self.user_id = user_id
        self.graph_id = str(uuid.uuid4())
        self.node_id = str(uuid.uuid4())
        self.graph_exec_id = str(uuid.uuid4())
        self.node_exec_id = str(uuid.uuid4())
        self.chat_id = chat_id

    @property
    def card_id(self) -> str:
        return step_card_id(self.user_id, self.node_exec_id)

    async def create(
        self, status: AgentExecutionStatus = AgentExecutionStatus.REVIEW
    ) -> "_Run":
        await AgentGraph.prisma().create(
            data={
                "id": self.graph_id,
                "version": 1,
                "name": "Morning digest",
                "description": "",
                "userId": self.user_id,
                "isActive": True,
            }
        )
        await AgentNode.prisma().create(
            data={
                "id": self.node_id,
                "agentBlockId": _BLOCK.id,
                "agentGraphId": self.graph_id,
                "agentGraphVersion": 1,
            }
        )
        await AgentGraphExecution.prisma().create(
            data={
                "id": self.graph_exec_id,
                "agentGraphId": self.graph_id,
                "agentGraphVersion": 1,
                "userId": self.user_id,
                "executionStatus": status,
                "triggerSource": "copilot",
                "triggerRef": self.chat_id,
            }
        )
        await AgentNodeExecution.prisma().create(
            data={
                "id": self.node_exec_id,
                "agentGraphExecutionId": self.graph_exec_id,
                "agentNodeId": self.node_id,
                "executionStatus": AgentExecutionStatus.RUNNING,
            }
        )
        return self

    async def next_step(self) -> "_Run":
        """A later pause of the same block in this run."""
        self.node_exec_id = str(uuid.uuid4())
        await AgentNodeExecution.prisma().create(
            data={
                "id": self.node_exec_id,
                "agentGraphExecutionId": self.graph_exec_id,
                "agentNodeId": self.node_id,
                "executionStatus": AgentExecutionStatus.RUNNING,
            }
        )
        return self

    async def step(self, stamped: bool = True) -> tuple[bool, dict]:
        """The block's own review pause, as the executor reaches it."""
        ctx = ExecutionContext(
            user_id=self.user_id,
            graph_id=self.graph_id,
            graph_exec_id=self.graph_exec_id,
            graph_version=1,
            sensitive_action_safe_mode=True,
            graph_sensitive_action_safe_mode=False,
            approval_chat_id=self.chat_id if stamped else None,
        )
        return await _BLOCK.is_block_exec_need_review(
            _INPUT,
            user_id=self.user_id,
            node_id=self.node_id,
            node_exec_id=self.node_exec_id,
            graph_exec_id=self.graph_exec_id,
            graph_id=self.graph_id,
            graph_version=1,
            execution_context=ctx,
        )

    async def row(self, node_exec_id: str | None = None):
        node_exec_id = node_exec_id or self.node_exec_id
        rows = await get_reviews_by_node_exec_ids([node_exec_id], self.user_id)
        return rows.get(node_exec_id)

    async def card(self):
        return await self.row(self.card_id)


async def _chat(user_id: str) -> ChatSession:
    session = await upsert_chat_session(ChatSession.new(user_id=user_id, dry_run=False))
    await update_session_autopilot_mode(session.session_id, user_id, "ask_first")
    reloaded = await get_chat_session(session.session_id, user_id)
    assert reloaded is not None
    return reloaded


async def _parked(user_id: str) -> tuple[ChatSession, _Run]:
    chat = await _chat(user_id)
    run = await _Run(user_id, chat.session_id).create()
    paused, _ = await run.step()
    assert paused
    return chat, run


async def _answer(client: httpx.AsyncClient, node_exec_id: str, approved: bool):
    response = await client.post(
        "/api/review/action",
        json={"reviews": [{"node_exec_id": node_exec_id, "approved": approved}]},
    )
    assert response.status_code == 200, response.text


async def test_a_step_parks_on_its_card_and_the_card_resumes_the_run(
    setup_test_user, test_user_id, flags, client, resumed
):
    chat, run = await _parked(test_user_id)

    row, card = await run.row(), await run.card()
    assert row is not None and row.status == ReviewStatus.WAITING
    assert card is not None and card.status == ReviewStatus.WAITING
    assert card.session_id == chat.session_id
    assert card.payload["workflow"]["node_exec_id"] == run.node_exec_id
    assert card.payload["tool_call_id"] == ""
    node = await AgentNodeExecution.prisma().find_unique(where={"id": run.node_exec_id})
    assert node is not None and node.executionStatus == AgentExecutionStatus.REVIEW
    # One decision, one item on Home: the card.
    home = {r.node_exec_id for r in await get_pending_reviews_for_user(test_user_id)}
    assert run.card_id in home and run.node_exec_id not in home

    await _answer(client, run.card_id, approved=True)

    row = await run.row()
    assert row is not None and row.status == ReviewStatus.APPROVED
    assert await run.card() is None
    assert resumed.await_args.kwargs["graph_exec_id"] == run.graph_exec_id
    # The card finished through the run's resume, not as a held call.
    assert await held._held(chat.session_id) == {}
    assert await run.step() == (False, _INPUT)


async def test_a_run_its_card_resumed_asks_the_chat_again_at_its_next_step(
    setup_test_user, test_user_id, flags, client
):
    """The resume the card publishes, replayed as the executor would: the run's
    next irreversible step parks on a new card in the same chat (#15300)."""
    chat, run = await _parked(test_user_id)
    queue = AsyncMock()
    with patch(
        "backend.executor.utils.get_async_execution_queue",
        AsyncMock(return_value=queue),
    ):
        await _answer(client, run.card_id, approved=True)

    queue.publish_message.assert_awaited_once()
    resumed = GraphExecutionEntry.model_validate_json(
        queue.publish_message.call_args.kwargs["message"]
    ).execution_context
    assert (resumed.graph_exec_id, resumed.approval_chat_id) == (
        run.graph_exec_id,
        chat.session_id,
    )

    await run.next_step()
    outputs = [
        output
        async for output in _BLOCK.execute(
            _INPUT,
            execution_context=resumed.model_copy(
                update={"node_id": run.node_id, "node_exec_id": run.node_exec_id}
            ),
            user_id=test_user_id,
            node_id=run.node_id,
            node_exec_id=run.node_exec_id,
            graph_exec_id=run.graph_exec_id,
            graph_id=run.graph_id,
            graph_version=1,
        )
    ]

    assert outputs == []
    card = await run.card()
    assert card is not None and card.status == ReviewStatus.WAITING
    assert card.session_id == chat.session_id


async def test_a_card_declined_in_chat_rejects_the_step_and_asks_from_now_on(
    setup_test_user, test_user_id, flags, client, resumed
):
    chat, run = await _parked(test_user_id)

    await _answer(client, run.card_id, approved=False)

    row = await run.row()
    assert row is not None and row.status == ReviewStatus.REJECTED
    resumed.assert_awaited_once()
    rule = await chat_rules.rule_for(
        chat.session_id, f"block:{_BLOCK.id}", test_user_id, None
    )
    assert rule is not None and rule.rule == "ask"
    with pytest.raises(BlockExecutionError):
        await run.step()


async def test_a_page_of_home_stays_full_while_steps_wait_on_their_cards(server, flags):
    """A client that stops at a short page would never see the later cards."""
    user_id = str(uuid.uuid4())
    await User.prisma().create(data={"id": user_id, "email": f"{user_id}@example.com"})
    chat = await _chat(user_id)
    run = await _Run(user_id, chat.session_id).create()
    cards = []
    for _ in range(3):
        assert (await run.step())[0]
        cards.append(run.card_id)
        await run.next_step()

    pages = [
        [
            review.node_exec_id
            for review in await get_pending_reviews_for_user(user_id, page, 2)
        ]
        for page in (1, 2, 3)
    ]

    assert [len(page) for page in pages] == [2, 1, 0]
    assert sorted(sum(pages, [])) == sorted(cards)


async def test_a_step_answered_on_the_run_page_closes_its_card(
    setup_test_user, test_user_id, flags, client, resumed
):
    _, run = await _parked(test_user_id)

    await _answer(client, run.node_exec_id, approved=True)

    assert await run.card() is None
    resumed.assert_awaited_once()


async def test_a_stopped_run_closes_its_steps_cards(
    setup_test_user, test_user_id, flags
):
    _, run = await _parked(test_user_id)

    await cancel_pending_reviews_for_execution(run.graph_exec_id, test_user_id)

    assert await run.card() is None


async def test_an_allow_rule_runs_the_step_and_records_why(
    setup_test_user, test_user_id, flags
):
    chat = await _chat(test_user_id)
    run = await _Run(test_user_id, chat.session_id).create()
    await chat_rules.set_rule(chat.session_id, f"block:{_BLOCK.id}", "allow")

    assert await run.step() == (False, _INPUT)

    row = await run.row()
    assert row is not None and row.status == ReviewStatus.APPROVED
    assert row.review_message == (
        "Allowed by Otto: you set Send Discord Message to run without asking."
    )
    assert await run.card() is None


async def test_a_cheap_outward_step_reaches_the_supervisor_as_its_direct_call_does(
    setup_test_user, test_user_id, flags
):
    """Auto judges a cheap outward block by the lighter rubric, naming the block."""
    chat = await _chat(test_user_id)
    await update_session_autopilot_mode(chat.session_id, test_user_id, "auto")
    chat = await get_chat_session(chat.session_id, test_user_id)
    assert chat is not None
    run = await _Run(test_user_id, chat.session_id).create()
    args = {"id": _BLOCK.id, "input": _INPUT}

    async def subject():
        return await RunCapabilityTool().gate_subject(test_user_id, chat, args)

    supervise = AsyncMock(return_value=Judgement(allowed=True, reason="ok"))
    with (
        patch("backend.copilot.gate.workflow.supervise", supervise),
        patch("backend.copilot.gate.supervise", supervise),
    ):
        step = await run.step()
        direct = await check_action(
            "run_capability", args, test_user_id, chat, subject_of=subject
        )

    assert step == (False, _INPUT) and direct.allowed
    for call in supervise.await_args_list:
        assert call.kwargs.get("outward_judged") is True
        assert call.kwargs["args"].get(RUNS_KEY) == "Send Discord Message"
    assert supervise.await_count == 2


@pytest.mark.parametrize("failure", ["no stamp", "chat gone", "card fails"])
async def test_a_step_the_gate_cannot_answer_is_an_alerted_human_review(
    setup_test_user, test_user_id, flags, failure
):
    chat = await _chat(test_user_id)
    run = await _Run(
        test_user_id, str(uuid.uuid4()) if failure == "chat gone" else chat.session_id
    ).create()
    spy = AsyncMock(wraps=HITLReviewHelper.get_or_create_human_review)
    open_row = AsyncMock(return_value=failure != "card fails")
    with (
        patch.object(HITLReviewHelper, "get_or_create_human_review", spy),
        patch("backend.copilot.gate.review.open_review_row", open_row),
    ):
        paused, _ = await run.step(stamped=failure != "no stamp")

    assert paused
    row = await run.row()
    assert row is not None and row.status == ReviewStatus.WAITING
    assert await run.card() is None
    # The last write of the row is the one that alerts.
    assert spy.await_args.kwargs.get("alert", True) is True


@pytest.mark.parametrize("runs_it", ["allow rule", "unsupervised", "supervisor allow"])
async def test_a_paywalled_owners_step_parks_however_the_gate_would_run_it(
    setup_test_user, test_user_id, flags, paywalled_user, runs_it
):
    """Without the gate this step parks and its resume meets the paywall, so the
    gate must not run it instead."""
    chat = await _chat(test_user_id)
    if runs_it == "unsupervised":
        await update_session_autopilot_mode(
            chat.session_id, test_user_id, "unsupervised"
        )
    else:
        rule = "allow" if runs_it == "allow rule" else "judge"
        await chat_rules.set_rule(chat.session_id, _KEY, rule)
    run = await _Run(test_user_id, chat.session_id).create()
    supervise = AsyncMock(return_value=Judgement(allowed=True, reason="ok"))
    spy = AsyncMock(wraps=HITLReviewHelper.get_or_create_human_review)
    with (
        patch("backend.copilot.gate.workflow.supervise", supervise),
        patch.object(HITLReviewHelper, "get_or_create_human_review", spy),
    ):
        paused, _ = await run.step()

    assert paused
    row = await run.row()
    assert row is not None and row.status == ReviewStatus.WAITING
    assert await run.card() is None
    assert spy.await_args.kwargs.get("alert", True) is True
    supervise.assert_not_awaited()


async def test_a_card_opened_before_the_flag_went_off_still_resumes_its_run(
    setup_test_user, test_user_id, flags, client, resumed
):
    chat, run = await _parked(test_user_id)
    flags[Flag.COPILOT_UNIFIED_APPROVALS] = False

    await _answer(client, run.card_id, approved=True)

    resumed.assert_awaited_once()
    later = await _Run(test_user_id, chat.session_id).create()
    paused, _ = await later.step()
    assert paused and await later.card() is None


@pytest_asyncio.fixture(loop_scope="session")
async def wider_rules_cleared(test_user_id):
    """A wider rule outlives its test and would let every later step run."""
    keys = [
        chat_rules._scoped_key(scope, test_user_id, None, _KEY)
        for scope in ("expert", "team")
    ]
    redis = await get_redis_async()
    await redis.delete(*keys)
    yield
    await redis.delete(*keys)


@pytest.mark.parametrize("scope", ["chat", "expert", "team"])
@pytest.mark.parametrize("rule", ["allow", "judge"])
async def test_a_rule_from_a_step_card_covers_every_later_call_of_its_block(
    setup_test_user,
    test_user_id,
    flags,
    client,
    resumed,
    wider_rules_cleared,
    rule,
    scope,
):
    """Answered like an action card: the rule lands on the block itself, so a
    later step of the run, a later run and a direct call of the block follow it."""
    chat, run = await _parked(test_user_id)
    response = await client.post(
        "/api/review/action",
        json={
            "reviews": [
                {
                    "node_exec_id": run.card_id,
                    "approved": True,
                    "chat_rule": rule,
                    "chat_rule_scope": scope,
                }
            ]
        },
    )
    assert response.status_code == 200, response.text

    later_chat = chat if scope == "chat" else await _chat(test_user_id)
    args = {"id": _BLOCK.id, "input": _INPUT}

    async def subject():
        return await RunCapabilityTool().gate_subject(test_user_id, later_chat, args)

    supervise = AsyncMock(return_value=Judgement(allowed=True, reason="ok"))
    with (
        patch("backend.copilot.gate.workflow.supervise", supervise),
        patch("backend.copilot.gate.supervise", supervise),
    ):
        same_run = await (await run.next_step()).step()
        later = await _Run(test_user_id, later_chat.session_id).create()
        later_run = await later.step()
        direct = await check_action(
            "run_capability", args, test_user_id, later_chat, subject_of=subject
        )

    assert same_run == (False, _INPUT)
    assert later_run == (False, _INPUT)
    assert direct.allowed and direct.decided
    assert supervise.await_count == (3 if rule == "judge" else 0)
    if scope == "chat":
        elsewhere = await _Run(
            test_user_id, (await _chat(test_user_id)).session_id
        ).create()
        assert (await elsewhere.step())[0]


async def test_a_card_answered_before_its_run_pauses_waits_until_it_has(
    setup_test_user, test_user_id, flags, client, resumed
):
    """While other steps still run, the run cannot be resumed: an answer the
    step took then would let the executor finish the pass with nothing waiting
    and mark the run complete, the step never run."""
    chat = await _chat(test_user_id)
    run = await _Run(test_user_id, chat.session_id).create(
        status=AgentExecutionStatus.RUNNING
    )
    paused, _ = await run.step()
    assert paused

    early = await client.post(
        "/api/review/action",
        json={"reviews": [{"node_exec_id": run.card_id, "approved": True}]},
    )

    assert early.status_code == 409, early.text
    row, card = await run.row(), await run.card()
    assert row is not None and row.status == ReviewStatus.WAITING
    assert card is not None and card.status == ReviewStatus.WAITING
    # The executor ends its pass in REVIEW only while a row still waits.
    assert await has_pending_reviews_for_graph_exec(run.graph_exec_id)
    resumed.assert_not_awaited()

    await AgentGraphExecution.prisma().update(
        where={"id": run.graph_exec_id},
        data={"executionStatus": AgentExecutionStatus.REVIEW},
    )
    await _answer(client, run.card_id, approved=True)

    row = await run.row()
    assert row is not None and row.status == ReviewStatus.APPROVED
    assert resumed.await_args.kwargs["graph_exec_id"] == run.graph_exec_id


async def test_an_answer_the_step_could_not_take_keeps_its_card_for_a_retry(
    setup_test_user, test_user_id, flags, client, resumed
):
    _, run = await _parked(test_user_id)

    with patch(
        "backend.data.human_review.process_all_reviews_for_execution",
        AsyncMock(side_effect=OSError("database unavailable")),
    ):
        failed = await client.post(
            "/api/review/action",
            json={"reviews": [{"node_exec_id": run.card_id, "approved": True}]},
        )

    assert failed.status_code == 503, failed.text
    # Back in the chat's queue, and still the one item Home shows.
    card = await run.card()
    assert card is not None and card.status == ReviewStatus.WAITING
    home = {r.node_exec_id for r in await get_pending_reviews_for_user(test_user_id)}
    assert run.card_id in home and run.node_exec_id not in home
    row = await run.row()
    assert row is not None and row.status == ReviewStatus.WAITING
    resumed.assert_not_awaited()

    await _answer(client, run.card_id, approved=True)

    row = await run.row()
    assert row is not None and row.status == ReviewStatus.APPROVED
    assert await run.card() is None
    resumed.assert_awaited_once()


@pytest.mark.parametrize(
    "ended", [AgentExecutionStatus.FAILED, AgentExecutionStatus.TERMINATED]
)
async def test_a_card_whose_run_has_ended_can_still_be_answered_and_goes(
    setup_test_user, test_user_id, flags, client, resumed, ended
):
    _, run = await _parked(test_user_id)
    await AgentGraphExecution.prisma().update(
        where={"id": run.graph_exec_id}, data={"executionStatus": ended}
    )

    await _answer(client, run.card_id, approved=False)

    assert await run.card() is None
    resumed.assert_not_awaited()


async def test_a_card_that_could_not_be_reopened_leaves_the_step_to_the_run_page(
    setup_test_user, test_user_id, flags, client, resumed
):
    """With both writes failing, the step's own row is the way back: Home lists
    it again, and answering it there closes the card."""
    _, run = await _parked(test_user_id)

    with (
        patch(
            "backend.data.human_review.process_all_reviews_for_execution",
            AsyncMock(side_effect=OSError("database unavailable")),
        ),
        patch(
            "backend.data.human_review.reopen_review",
            AsyncMock(side_effect=OSError("database unavailable")),
        ),
    ):
        failed = await client.post(
            "/api/review/action",
            json={"reviews": [{"node_exec_id": run.card_id, "approved": True}]},
        )

    assert failed.status_code == 503, failed.text
    home = {r.node_exec_id for r in await get_pending_reviews_for_user(test_user_id)}
    assert run.node_exec_id in home and run.card_id not in home

    await _answer(client, run.node_exec_id, approved=True)

    assert await run.card() is None
    resumed.assert_awaited_once()
