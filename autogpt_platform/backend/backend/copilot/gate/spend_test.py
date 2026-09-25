"""A paid read asks once the chat has spent its ceiling.

Driven through ``check_action`` against the real tree and chat ledgers on the
in-memory Redis ``tree_test`` uses, so the ceiling read, the charges and the
raise are the ones production runs.
"""

import logging
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.blocks.agent import AgentExecutorBlock
from backend.blocks.basic import StoreValueBlock
from backend.blocks.llm import AITextGeneratorBlock
from backend.blocks.pinecone import PineconeQueryBlock
from backend.blocks.talking_head import CreateTalkingAvatarVideoBlock
from backend.copilot import tree
from backend.copilot.context import set_execution_context
from backend.copilot.gate import CEILING_UNIT_MICRODOLLARS, check_action
from backend.copilot.gate.headline import Headline
from backend.copilot.gate.policy import Effect
from backend.copilot.gate.review import open_review, review_payload
from backend.copilot.gate.subject import Subject, block_subject
from backend.copilot.model import AutopilotMode, ChatSession, ChatSessionMetadata
from backend.copilot.tools.gate_subject_test import _graph, _node, _sub
from backend.copilot.tools.helpers import _charge_block_credits
from backend.copilot.tools.run_agent import RunAgentTool
from backend.copilot.tools.run_capability import RunCapabilityTool
from backend.copilot.tree import (
    CHAT_LEDGER_TTL_SECONDS,
    SpawnRequest,
    TreeLedger,
    admit_turn,
    charge_credits,
    charge_turn,
    derive_child_envelope,
    root_envelope,
)
from backend.copilot.tree_test import BrokenRedis, FakeRedis
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.execution import ExecutionStatus
from backend.data.model import CredentialsMetaInput
from backend.data.redis_client import AsyncRedisClient

_GATE = "backend.copilot.gate"
_RUN = "backend.copilot.tools.run_agent"
_CAP = "backend.copilot.tools.run_capability"
_PAID = Subject(
    key="block:paid", name="Paid Search", effect=Effect.READ, estimate=50_000
)
_FREE = Subject(key="block:free", name="Store Value", effect=Effect.READ)
_VIDEO = Subject(
    key="block:video",
    name="Create Talking Avatar Video",
    effect=Effect.WORKSPACE,
    estimate=1_000_000,
)
_SEND = Subject(
    key="block:send",
    name="Gmail Send",
    effect=Effect.EXTERNAL,
    reason="reaches outside the platform",
    estimate=50_000,
)


_CHAT = "session-1"


def _session(mode: AutopilotMode = "auto", origin="interactive") -> ChatSession:
    return ChatSession(
        session_id=_CHAT,
        user_id="user-1",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(origin=origin, autopilot_mode=mode),
        messages=[],
    )


@pytest.fixture
def redis():
    """A metered root turn of the chat, its ledgers on an in-memory Redis."""
    fake = FakeRedis()
    _enter(root_envelope("turn-1", session_id=_CHAT))
    with patch.object(tree, "get_redis_async", AsyncMock(return_value=fake)):
        yield fake
    set_execution_context(None, None)


@pytest.fixture
def ledger(redis):
    """The running turn's tree."""
    return TreeLedger(cast(AsyncRedisClient, redis))


@pytest.fixture
def chat(redis):
    """The chat's ledger, which a paid read asks against."""
    return TreeLedger(
        cast(AsyncRedisClient, redis),
        prefix="copilot:chat-spend:",
        ttl_seconds=CHAT_LEDGER_TTL_SECONDS,
    )


@pytest.fixture
def gate(redis):
    store = SimpleNamespace(
        find_review=AsyncMock(return_value=None),
        open_review=AsyncMock(return_value=Headline(ask="Run it")),
        consume=AsyncMock(return_value=True),
    )
    with (
        patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=True)),
        patch.object(tree, "is_feature_enabled", AsyncMock(return_value=True)),
        patch(f"{_GATE}.review_store.find_review", store.find_review),
        patch(f"{_GATE}.review_store.open_review", store.open_review),
        patch(f"{_GATE}.review_store.consume", store.consume),
        patch(f"{_GATE}.held.remember", AsyncMock(return_value=True)),
        patch(f"{_GATE}.chat_rules.rule_for", AsyncMock(return_value=None)),
    ):
        yield store


def _enter(envelope):
    set_execution_context("user-1", None, envelope=envelope)


async def _open(chat: TreeLedger, ceiling: int, spent: int = 0) -> None:
    await chat.open(_CHAT, ceiling_microdollars=ceiling, max_nodes=1)
    await chat.charge(_CHAT, spent)


async def _check(
    subject: Subject | None,
    mode: AutopilotMode = "auto",
    tool="t",
    origin="interactive",
):
    async def subject_of():
        return subject

    return await check_action(
        tool,
        {"q": 1},
        "user-1",
        _session(mode, origin),
        "call-1",
        subject_of=subject_of,
    )


@pytest.mark.parametrize("mode", ["auto", "ask_first"])
async def test_at_a_zero_ceiling_every_paid_read_asks_and_a_free_one_never(
    gate, chat, mode
):
    """Kills: dropping ``estimate > 0`` (the free read asks)."""
    await _open(chat, ceiling=0)
    paid = await _check(_PAID, mode)
    assert not paid.allowed and paid.review_id
    assert paid.reason == (
        "costs about $0.05, and this chat has spent $0.00 of its $0.00 ceiling; "
        "approving adds $1.00 to it"
    )
    _, kwargs = gate.open_review.await_args
    # Microdollars: the card formats money itself.
    assert kwargs["spend"] == {
        "estimate": 50_000,
        "spent": 0,
        "ceiling": 0,
        "unit": CEILING_UNIT_MICRODOLLARS,
    }
    assert (await _check(_FREE, mode)).allowed


async def test_over_the_ceiling_a_paid_workspace_block_asks(gate, chat):
    """Kills: metering reads only (the costliest blocks never meet the ceiling)."""
    await _open(chat, ceiling=0)
    decision = await _check(_VIDEO)
    assert not decision.allowed and decision.review_id
    _, kwargs = gate.open_review.await_args
    assert kwargs["spend"]["estimate"] == 1_000_000


async def test_consulting_a_teammate_is_a_paid_read(gate, chat):
    await _open(chat, ceiling=0)
    assert not (await _check(None, tool="consult_teammate")).allowed


async def test_unsupervised_never_asks_for_money(gate, chat):
    await _open(chat, ceiling=0)
    assert (await _check(_PAID, "unsupervised")).allowed


async def test_under_the_ceiling_a_paid_read_runs(gate, chat):
    await _open(chat, ceiling=1_000_000, spent=999_999)
    assert (await _check(_PAID)).allowed


async def test_an_over_cap_external_shows_the_write_reason(gate, chat):
    """Kills: dropping the effect check (the money reason replaces it)."""
    await _open(chat, ceiling=0)
    decision = await _check(_SEND)
    assert not decision.allowed
    assert decision.reason == "reaches outside the platform"
    _, kwargs = gate.open_review.await_args
    assert kwargs["spend"] is None


async def test_the_approval_runs_next_turn_and_buys_the_chat_one_more_dollar(
    gate, ledger, chat
):
    """Turn 1 crosses the chat's ceiling and asks; the approved call runs in
    turn 2, and turn 2's next paid reads spend the dollar it bought.
    Kills: raising the running turn's tree (turn 2 asks again at once), and
    reading the tree for the ceiling (turn 1 never asks)."""
    with (
        patch.object(tree, "is_feature_enabled", AsyncMock(return_value=True)),
        patch.object(
            tree, "resolve_root_ceiling_microdollars", AsyncMock(return_value=10**7)
        ),
        patch.object(
            tree, "resolve_chat_ceiling_microdollars", AsyncMock(return_value=10**6)
        ),
    ):
        turn_1 = root_envelope("turn-1", session_id=_CHAT)
        await admit_turn(turn_1, user_id="user-1")
        await charge_turn(turn_1, 10**6)
        assert not (await _check(_PAID)).allowed

        turn_2 = root_envelope("turn-2", session_id=_CHAT)
        await admit_turn(turn_2, user_id="user-1")
        _enter(turn_2)
    gate.find_review.return_value = SimpleNamespace(
        status=ReviewStatus.APPROVED, payload={"spend": {}}
    )
    assert (await _check(_PAID)).allowed
    assert (await chat.snapshot(_CHAT))["ceiling"] == 2 * 10**6
    assert (await ledger.snapshot("turn-2"))["ceiling"] == 10**7

    gate.find_review.return_value = None
    await charge_turn(turn_2, CEILING_UNIT_MICRODOLLARS - 1)
    assert (await _check(_PAID)).allowed
    await charge_turn(turn_2, 1)
    assert not (await _check(_PAID)).allowed


async def test_an_approval_of_any_other_card_raises_nothing(gate, chat):
    await _open(chat, ceiling=0)
    gate.find_review.return_value = SimpleNamespace(
        status=ReviewStatus.APPROVED, payload={}
    )
    assert (await _check(_SEND)).allowed
    assert (await chat.snapshot(_CHAT))["ceiling"] == 0


async def test_a_block_run_charges_what_it_cost(gate, ledger, chat):
    """Kills: charging nothing at the block's charge site, or charging the chat
    twice. One credit is $0.01."""
    await _open(chat, ceiling=10_000_000)
    await ledger.open("turn-1", ceiling_microdollars=10_000_000, max_nodes=8)
    await _charge_block_credits(
        SimpleNamespace(spend_credits=AsyncMock()),
        user_id="user-1",
        block_name="Pinecone Query",
        block_id="b",
        node_exec_id="n",
        cost=7,
        cost_filter={},
        session_id="s1",
    )
    assert (await ledger.snapshot("turn-1"))["spent"] == 70_000
    assert (await chat.snapshot(_CHAT))["spent"] == 70_000


@pytest.mark.parametrize("charge_fails", [False, True])
async def test_only_a_failed_charge_is_logged_as_a_billing_leak(ledger, charge_fails):
    """Kills: a failed flag lookup after a paid charge logged as BILLING_LEAK."""
    lines: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda record: lines.append(record.getMessage())
    helpers_logger = logging.getLogger("backend.copilot.tools.helpers")
    helpers_logger.addHandler(handler)
    spend = AsyncMock(side_effect=RuntimeError("db down") if charge_fails else None)
    try:
        with patch.object(
            tree, "is_feature_enabled", AsyncMock(side_effect=RuntimeError("flags"))
        ):
            await _charge_block_credits(
                SimpleNamespace(spend_credits=spend),
                user_id="user-1",
                block_name="Pinecone Query",
                block_id="b",
                node_exec_id="n",
                cost=7,
                cost_filter={},
                session_id="s1",
            )
    finally:
        helpers_logger.removeHandler(handler)
    assert any("BILLING_LEAK" in line for line in lines) is charge_fails


async def test_a_workflow_run_charges_its_pre_flight_estimate(gate, chat):
    """Kills: charging nothing at the workflow's charge site; sub-graph nodes count."""
    await _open(chat, ceiling=10_000_000)
    graph = _graph(
        [
            _node("query", PineconeQueryBlock(), {}),
            _node("nested", AgentExecutorBlock(), {"graph_id": "sub"}),
        ],
        sub_graphs=[_sub("sub", [_node("q2", PineconeQueryBlock(), {})])],
    )
    agent = SimpleNamespace(id="lib-1", graph_id="main", name="Digest")
    execution = SimpleNamespace(id="exec-1", status=ExecutionStatus.RUNNING)
    with (
        patch(f"{_RUN}.get_or_create_library_agent", AsyncMock(return_value=agent)),
        patch(
            f"{_RUN}.execution_utils.add_graph_execution",
            AsyncMock(return_value=execution),
        ),
        patch(f"{_RUN}.track_agent_run_success"),
    ):
        await RunAgentTool()._run_agent(
            "user-1", _session(), graph, {}, {}, dry_run=False
        )
    assert (await chat.snapshot(_CHAT))["spent"] == 20_000


def test_a_paid_block_carries_its_estimate_and_pure_computation_none():
    assert block_subject(PineconeQueryBlock(), {}).estimate == 10_000
    with patch("backend.copilot.gate.subject.block_usage_cost", return_value=(5, {})):
        assert block_subject(StoreValueBlock(), {}).estimate == 0


async def test_an_llm_block_is_priced_with_the_credentials_it_will_run_with():
    """Kills: pricing the model's raw arguments (every LLM block estimates $0)."""
    cost = BLOCK_COSTS[AITextGeneratorBlock][0].cost_filter
    platform = CredentialsMetaInput.model_validate(cost["credentials"])
    session = _session()
    with patch(
        f"{_CAP}.resolve_block_credentials",
        AsyncMock(return_value=({"credentials": platform}, [])),
    ):
        subject = await RunCapabilityTool().gate_subject(
            "user-1",
            session,
            {
                "id": AITextGeneratorBlock().id,
                "input": {"prompt": "Summarise", "model": cost["model"]},
            },
        )
    assert subject is not None and subject.effect is Effect.READ
    assert subject.estimate > 0


@pytest.mark.parametrize(
    "passed",
    [{}, {"credentials": {"id": "bogus", "provider": "d_id", "type": "api_key"}}],
)
async def test_a_workspace_block_is_priced_with_the_credentials_it_will_run_with(
    passed,
):
    """Kills: pricing only reads with credentials (an avatar video estimates $0),
    and letting credentials the model passed outrank the resolved ones."""
    cost = BLOCK_COSTS[CreateTalkingAvatarVideoBlock][0].cost_filter
    platform = CredentialsMetaInput.model_validate(cost["credentials"])
    with patch(
        f"{_CAP}.resolve_block_credentials",
        AsyncMock(return_value=({"credentials": platform}, [])),
    ):
        subject = await RunCapabilityTool().gate_subject(
            "user-1",
            _session(),
            {
                "id": CreateTalkingAvatarVideoBlock().id,
                "input": {"script_input": "Hello", **passed},
            },
        )
    assert subject is not None and subject.effect is Effect.WORKSPACE
    assert subject.estimate == 1_000_000


async def test_a_held_money_card_stores_its_spend_beside_its_headline():
    """Kills: a resolution of ``open_review`` that drops ``spend`` (the card
    loses its money block and its approval raises nothing)."""
    spend = {"estimate": 50_000, "spent": 0, "ceiling": 0, "unit": 1_000_000}
    reviews = MagicMock(get_or_create_human_review=AsyncMock())
    with patch("backend.copilot.gate.review.review_db", return_value=reviews):
        headline = await open_review(
            "rid", "user-1", _session(), "t", {"q": 1}, "", _PAID, spend=spend
        )
    assert headline is not None and headline.text == "Run “Paid Search”"
    stored = reviews.get_or_create_human_review.await_args.kwargs
    assert stored["message"] == headline.text
    assert stored["input_data"]["spend"] == spend


def test_a_money_card_offers_no_chat_rule():
    """Reads never consult a rule, so the card must offer none."""
    spend = {"estimate": 50_000, "spent": 0, "ceiling": 0, "unit": 1_000_000}
    payload = review_payload("t", {}, _PAID, spend)
    assert payload["spend"] == spend
    assert payload["chat_rules_allowed"] == []
    sent = review_payload("t", {}, _SEND)
    assert sent["spend"] is None
    assert sent["chat_rules_allowed"] == ["allow", "judge"]


async def test_flag_on_a_root_turn_opens_its_tree_and_its_chat(ledger, chat):
    with (
        patch.object(tree, "is_feature_enabled", AsyncMock(return_value=True)),
        patch.object(
            tree, "resolve_root_ceiling_microdollars", AsyncMock(return_value=42)
        ),
        patch.object(
            tree, "resolve_chat_ceiling_microdollars", AsyncMock(return_value=99)
        ),
    ):
        await admit_turn(root_envelope("turn-1", session_id=_CHAT), user_id="user-1")
    snapshot = await ledger.snapshot("turn-1")
    assert (snapshot["ceiling"], snapshot["spent"], snapshot["nodes"]) == (42, 0, 1)
    assert (await chat.snapshot(_CHAT))["ceiling"] == 99


async def test_flag_off_opens_no_ledger_and_charges_nothing(ledger, chat):
    """Kills: metering roots or charging spend without the flag."""
    with patch.object(tree, "is_feature_enabled", AsyncMock(return_value=False)):
        await admit_turn(root_envelope("turn-1", session_id=_CHAT), user_id="user-1")
        assert await ledger.snapshot("turn-1") == {}
        assert await chat.snapshot(_CHAT) == {}
        # A spawned turn's tree exists with the flag off; spend must not reach it.
        await ledger.open("turn-1", ceiling_microdollars=10_000_000, max_nodes=8)
        priced = MagicMock(return_value=7)
        await charge_credits("user-1", priced)
    assert (await ledger.snapshot("turn-1"))["spent"] == 0
    priced.assert_not_called()


async def test_a_spawned_turn_charges_and_asks_against_the_chat_that_spawned_it(
    gate, chat
):
    """An expert child of the chat, and a sub-session, spend the chat's ceiling;
    only the sub-session never asks. Kills: dropping the chat id from a child."""
    await _open(chat, ceiling=10**6)
    root = root_envelope("turn-1", session_id=_CHAT)
    expert = derive_child_envelope(root, SpawnRequest(may_spawn=True))
    isolate = derive_child_envelope(expert, SpawnRequest(shares_memory=True))
    await charge_turn(expert, 600_000)
    await charge_turn(isolate, 400_000)
    assert (await chat.snapshot(_CHAT))["spent"] == 10**6

    _enter(expert)
    assert not (await _check(_PAID)).allowed
    _enter(isolate)
    assert (await _check(_PAID, origin="automation")).allowed


async def test_a_chat_ledger_that_never_opened_opens_on_the_first_paid_read(gate, chat):
    """Kills: running every paid read free when the open at admission failed."""
    with patch.object(
        tree, "resolve_chat_ceiling_microdollars", AsyncMock(return_value=0)
    ):
        assert not (await _check(_PAID)).allowed
    assert (await chat.snapshot(_CHAT))["ceiling"] == 0


async def test_an_unreachable_chat_ledger_refuses_the_paid_read(gate):
    """A9: the read raises, and ``BaseTool._gate`` refuses what raises."""
    with (
        patch.object(tree, "get_redis_async", AsyncMock(return_value=BrokenRedis())),
        pytest.raises(ConnectionError),
    ):
        await _check(_PAID)


async def test_a_chat_ledger_that_will_not_open_refuses_the_paid_read(gate, redis):
    """Kills: reading a ledger that is still missing as "under the ceiling"."""
    redis.eval = AsyncMock(return_value=0)
    with (
        patch.object(
            tree, "resolve_chat_ceiling_microdollars", AsyncMock(return_value=0)
        ),
        pytest.raises(RuntimeError, match="could not be opened"),
    ):
        await _check(_PAID)


@pytest.mark.parametrize("reset, ledgers", [("never", 1), ("daily", 2)])
async def test_a_daily_reset_gives_each_utc_day_its_own_ledger(
    gate, redis, monkeypatch, reset, ledgers
):
    """Kills: a daily key that ignores the date, and one that never resets."""
    monkeypatch.setattr(tree.config, "spend_ceiling_reset", reset)
    today = [datetime(2026, 9, 25, 23, tzinfo=UTC)]
    with (
        patch.object(tree, "datetime", SimpleNamespace(now=lambda tz: today[0])),
        patch.object(
            tree, "resolve_chat_ceiling_microdollars", AsyncMock(return_value=10**6)
        ),
    ):
        # Day one: opened at the read, then charged to its ceiling.
        assert (await _check(_PAID)).allowed
        await charge_turn(root_envelope("turn-1", session_id=_CHAT), 10**6)
        assert not (await _check(_PAID)).allowed
        today[0] = datetime(2026, 9, 26, 1, tzinfo=UTC)
        asked_next_day = not (await _check(_PAID)).allowed
    keys = [k for k in redis.hashes if k.startswith("copilot:chat-spend:")]
    assert len(keys) == ledgers
    # Under "never" the day-one spend still asks on day two.
    assert asked_next_day is (reset == "never")
    if reset == "daily":
        assert {redis.ttls[k] for k in keys} == {2 * 24 * 3600}


async def test_a_chat_ceiling_is_not_clamped_to_what_is_left_of_today():
    """Kills: carrying a late-night remainder into the days after."""
    with (
        patch.object(
            tree,
            "get_global_rate_limits",
            AsyncMock(return_value=(2_500_000, 5_000_000, None)),
        ),
        patch.object(tree, "get_remaining_usd_budget", AsyncMock(return_value=0.3)),
    ):
        assert await tree.resolve_chat_ceiling_microdollars("user-1") == 1_250_000
        assert await tree.resolve_root_ceiling_microdollars("user-1") == 300_000
