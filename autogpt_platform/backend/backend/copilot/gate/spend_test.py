"""A paid read asks once the turn's tree has spent its ceiling.

Driven through ``check_action`` against the real tree ledger on the in-memory
Redis ``tree_test`` uses, so the ceiling read, the charges and the raise are
the ones production runs.
"""

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
from backend.copilot import tree
from backend.copilot.context import set_execution_context
from backend.copilot.gate import CEILING_UNIT_MICRODOLLARS, check_action
from backend.copilot.gate.policy import Effect
from backend.copilot.gate.review import review_payload
from backend.copilot.gate.subject import Subject, block_subject
from backend.copilot.model import AutopilotMode, ChatSession, ChatSessionMetadata
from backend.copilot.tools.gate_subject_test import _graph, _node, _sub
from backend.copilot.tools.helpers import _charge_block_credits
from backend.copilot.tools.run_agent import RunAgentTool
from backend.copilot.tools.run_capability import RunCapabilityTool
from backend.copilot.tree import TreeLedger, admit_turn, charge_credits, root_envelope
from backend.copilot.tree_test import FakeRedis
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
_SEND = Subject(
    key="block:send",
    name="Gmail Send",
    effect=Effect.EXTERNAL,
    reason="reaches outside the platform",
    estimate=50_000,
)


def _session(mode: AutopilotMode = "auto") -> ChatSession:
    return ChatSession(
        session_id="session-1",
        user_id="user-1",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(origin="interactive", autopilot_mode=mode),
        messages=[],
    )


@pytest.fixture
def ledger():
    """A metered root turn, its tree open on an in-memory Redis."""
    redis = FakeRedis()
    ledger = TreeLedger(cast(AsyncRedisClient, redis))
    set_execution_context("user-1", None, envelope=root_envelope("turn-1"))
    with patch.object(tree, "get_tree_ledger", AsyncMock(return_value=ledger)):
        yield ledger
    set_execution_context(None, None)


@pytest.fixture
def gate(ledger):
    store = SimpleNamespace(
        find_review=AsyncMock(return_value=None),
        open_review=AsyncMock(return_value=True),
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


async def _open(ledger: TreeLedger, ceiling: int, spent: int = 0) -> None:
    await ledger.open("turn-1", ceiling_microdollars=ceiling, max_nodes=8)
    await ledger.charge("turn-1", spent)


async def _check(subject: Subject | None, mode: AutopilotMode = "auto", tool="t"):
    async def subject_of():
        return subject

    return await check_action(
        tool, {"q": 1}, "user-1", _session(mode), "call-1", subject_of=subject_of
    )


@pytest.mark.parametrize("mode", ["auto", "ask_first"])
async def test_at_a_zero_ceiling_every_paid_read_asks_and_a_free_one_never(
    gate, ledger, mode
):
    """Kills: dropping ``estimate > 0`` (the free read asks)."""
    await _open(ledger, ceiling=0)
    paid = await _check(_PAID, mode)
    assert not paid.allowed and paid.review_id
    assert paid.reason == (
        "costs about $0.05, and this task has spent $0.00 of its $0.00 ceiling"
    )
    _, kwargs = gate.open_review.await_args
    assert kwargs["spend"] == {
        "estimate": "$0.05",
        "spent": "$0.00",
        "ceiling": "$0.00",
    }
    assert (await _check(_FREE, mode)).allowed


async def test_consulting_a_teammate_is_a_paid_read(gate, ledger):
    await _open(ledger, ceiling=0)
    assert not (await _check(None, tool="consult_teammate")).allowed


async def test_unsupervised_never_asks_for_money(gate, ledger):
    await _open(ledger, ceiling=0)
    assert (await _check(_PAID, "unsupervised")).allowed


async def test_under_the_ceiling_a_paid_read_runs(gate, ledger):
    await _open(ledger, ceiling=1_000_000, spent=999_999)
    assert (await _check(_PAID)).allowed


async def test_an_over_cap_external_shows_the_write_reason(gate, ledger):
    """Kills: dropping the effect check (the money reason replaces it)."""
    await _open(ledger, ceiling=0)
    decision = await _check(_SEND)
    assert not decision.allowed
    assert decision.reason == "reaches outside the platform"
    _, kwargs = gate.open_review.await_args
    assert kwargs["spend"] is None


async def test_one_approval_raises_the_ceiling_by_one_unit(gate, ledger):
    """Kills: raising by zero (the next paid read asks again)."""
    await _open(ledger, ceiling=0)
    gate.find_review.return_value = SimpleNamespace(
        status=ReviewStatus.APPROVED, payload={"spend": {}}
    )
    assert (await _check(_PAID)).allowed
    assert (await ledger.snapshot("turn-1"))["ceiling"] == CEILING_UNIT_MICRODOLLARS

    gate.find_review.return_value = None
    await ledger.charge("turn-1", CEILING_UNIT_MICRODOLLARS - 1)
    assert (await _check(_PAID)).allowed
    await ledger.charge("turn-1", 1)
    assert not (await _check(_PAID)).allowed


async def test_an_approval_of_any_other_card_raises_nothing(gate, ledger):
    await _open(ledger, ceiling=0)
    gate.find_review.return_value = SimpleNamespace(
        status=ReviewStatus.APPROVED, payload={}
    )
    assert (await _check(_SEND)).allowed
    assert (await ledger.snapshot("turn-1"))["ceiling"] == 0


async def test_a_block_run_charges_what_it_cost(gate, ledger):
    """Kills: charging nothing at the block's charge site. One credit is $0.01."""
    await _open(ledger, ceiling=10_000_000)
    await _charge_block_credits(
        SimpleNamespace(spend_credits=AsyncMock()),
        user_id="user-1",
        block_name="Pinecone Query",
        block_id="b",
        node_exec_id="n",
        cost=7,
        cost_filter={},
        synthetic_graph_id="g",
        synthetic_node_id="n",
    )
    assert (await ledger.snapshot("turn-1"))["spent"] == 70_000


async def test_a_workflow_run_charges_its_pre_flight_estimate(gate, ledger):
    """Kills: charging nothing at the workflow's charge site; sub-graph nodes count."""
    await _open(ledger, ceiling=10_000_000)
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
    assert (await ledger.snapshot("turn-1"))["spent"] == 20_000


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


def test_a_money_card_offers_no_chat_rule():
    """Reads never consult a rule, so the card must offer none."""
    spend = {"estimate": "$0.05", "spent": "$0.00", "ceiling": "$0.00"}
    payload = review_payload("t", {}, _PAID, spend)
    assert payload["spend"] == spend
    assert payload["chat_rules_allowed"] == []
    sent = review_payload("t", {}, _SEND)
    assert sent["spend"] is None
    assert sent["chat_rules_allowed"] == ["allow", "judge"]


async def test_flag_on_a_root_turn_opens_its_tree(ledger):
    with (
        patch.object(tree, "is_feature_enabled", AsyncMock(return_value=True)),
        patch.object(
            tree, "resolve_root_ceiling_microdollars", AsyncMock(return_value=42)
        ),
    ):
        await admit_turn(root_envelope("turn-1"), user_id="user-1", ledger=ledger)
    snapshot = await ledger.snapshot("turn-1")
    assert (snapshot["ceiling"], snapshot["spent"], snapshot["nodes"]) == (42, 0, 1)


async def test_flag_off_opens_no_ledger_and_charges_nothing(ledger):
    """Kills: metering roots or charging spend without the flag."""
    with patch.object(tree, "is_feature_enabled", AsyncMock(return_value=False)):
        await admit_turn(root_envelope("turn-1"), user_id="user-1", ledger=ledger)
        assert await ledger.snapshot("turn-1") == {}
        # A spawned turn's tree exists with the flag off; spend must not reach it.
        await _open(ledger, ceiling=10_000_000)
        priced = MagicMock(return_value=7)
        await charge_credits("user-1", priced)
    assert (await ledger.snapshot("turn-1"))["spent"] == 0
    priced.assert_not_called()
