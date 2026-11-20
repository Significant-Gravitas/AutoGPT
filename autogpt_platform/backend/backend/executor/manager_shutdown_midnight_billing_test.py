"""A node runs, and is billed for, the model it started on, whatever day it
settles on.

A block given a model the catalog has switched off runs the model's catalog
fallback from the model's shutdown date, from 00:00 UTC. A node decides once,
when it starts. One that starts before midnight runs the model itself to the
end, so its settlement, a stop's estimate, the platform cost log and the run's
record price and name that model, though midnight passes while it runs. One
that starts on the day runs, and is billed for, the fallback, though the
dispatcher charged it the day before. A node that runs again after its pod
died decides again when it starts again.

The node is charged and runs through the executor and its billing for real,
against the real block costs. Only the provider, the database, the wallet, the
credential store and the clock the shutdown date is read against are
replaced. Midnight passes while the provider has the request.
"""

import asyncio
import math
import threading
import uuid
from contextlib import ExitStack
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.llm import AITextGeneratorBlock, LLMModel
from backend.data.block_cost_config import MODEL_COST, TOKEN_COST
from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.graph import Node
from backend.data.llm_registry import llm_models
from backend.data.model import APIKeyCredentials, GraphExecutionStats
from backend.data.platform_cost import PlatformCostEntry
from backend.executor import billing
from backend.executor.cost_tracking import drain_pending_cost_logs
from backend.executor.manager import ExecutionProcessor
from backend.executor.utils import NodeExecutionProgress
from backend.integrations.credentials_store import (
    anthropic_credentials,
    openai_credentials,
)
from backend.util.llm.providers import ProviderResponse

NODE_EXEC_ID = "node-exec-1"
# Each reply reports a million input tokens, so it costs the model's price
# per million input tokens.
INPUT_TOKENS = 1_000_000


def _utc(*parts: int) -> datetime:
    return datetime(*parts, tzinfo=timezone.utc)


# OpenAI shuts gpt-5-mini down on 11 Dec; it falls back to gpt-6.1-terra, at
# 150 credits per million input tokens against its own 38. Anthropic retires
# Sonnet 4.5 on 30 Nov; it falls back to Sonnet 5.5, at 300 against 450.
CASES = [
    pytest.param(
        LLMModel.GPT5_MINI,
        openai_credentials,
        _utc(2026, 12, 10, 23, 59, 50),
        _utc(2026, 12, 11, 0, 0, 10),
        LLMModel.GPT5_MINI,
        id="gpt-5-mini-starts-the-day-before",
    ),
    pytest.param(
        LLMModel.GPT5_MINI,
        openai_credentials,
        _utc(2026, 12, 11, 0, 0, 10),
        _utc(2026, 12, 11, 0, 0, 30),
        LLMModel.GPT6_1_TERRA,
        id="gpt-5-mini-starts-on-the-day",
    ),
    pytest.param(
        LLMModel.CLAUDE_4_5_SONNET,
        anthropic_credentials,
        _utc(2026, 11, 29, 23, 59, 50),
        _utc(2026, 11, 30, 0, 0, 10),
        LLMModel.CLAUDE_4_5_SONNET,
        id="sonnet-4-5-starts-the-day-before",
    ),
    pytest.param(
        LLMModel.CLAUDE_4_5_SONNET,
        anthropic_credentials,
        _utc(2026, 11, 30, 0, 0, 10),
        _utc(2026, 11, 30, 0, 0, 30),
        LLMModel.CLAUDE_5_5_SONNET,
        id="sonnet-4-5-starts-on-the-day",
    ),
]

# The saved model, its credentials, a moment before its shutdown midnight and
# one after it, the clock once the provider has the request, and the model a
# node that starts after midnight runs.
MIDNIGHTS = [
    pytest.param(
        LLMModel.GPT5_MINI,
        openai_credentials,
        _utc(2026, 12, 10, 23, 59, 59),
        _utc(2026, 12, 11, 0, 0, 1),
        _utc(2026, 12, 11, 0, 0, 20),
        LLMModel.GPT6_1_TERRA,
        id="gpt-5-mini",
    ),
    pytest.param(
        LLMModel.CLAUDE_4_5_SONNET,
        anthropic_credentials,
        _utc(2026, 11, 29, 23, 59, 59),
        _utc(2026, 11, 30, 0, 0, 1),
        _utc(2026, 11, 30, 0, 0, 20),
        LLMModel.CLAUDE_5_5_SONNET,
        id="sonnet-4-5",
    ),
]


class _Clock:
    """The UTC time a model's shutdown date is read against."""

    def __init__(self, now: datetime) -> None:
        self.now = now

    def today(self) -> date:
        return self.now.date()


class _Node:
    """One attempt at a node: the provider, which moves the clock while it has
    the request, and what the wallet, the cost log and the run's record saw."""

    def __init__(self, clock: _Clock, settles: datetime, *, replies: bool) -> None:
        self.clock = clock
        self.settles = settles
        self.replies = replies
        self.asked: list[str] = []
        self.spent: list[int] = []
        self.cost_log: list[PlatformCostEntry] = []
        self.recorded: list[dict[str, Any]] = []
        self.statuses: list[ExecutionStatus] = []
        self.request_sent = asyncio.Event()
        self.reconcile = AsyncMock(side_effect=billing.charge_reconciled_usage)

    async def provider(self, *, model: str, **_: Any) -> ProviderResponse:
        self.asked.append(model)
        self.clock.now = self.settles
        self.request_sent.set()
        if not self.replies:
            await asyncio.Event().wait()
        return ProviderResponse(
            content="Hello.", prompt_tokens=INPUT_TOKENS, completion_tokens=0
        )

    def spend(self, *, cost: int, **_: Any) -> int:
        self.spent.append(cost)
        return 1_000_000

    async def spend_async(self, *, cost: int, **_: Any) -> int:
        return self.spend(cost=cost)

    async def log_cost(self, entry: PlatformCostEntry) -> None:
        self.cost_log.append(entry)

    async def update_status(
        self, *, status: ExecutionStatus, stats: dict | None = None, **_: Any
    ) -> None:
        self.statuses.append(status)
        if stats is not None:
            self.recorded.append(stats)

    def patches(self, credentials: APIKeyCredentials, node: Node) -> list:
        key = credentials.model_copy(update={"api_key": SecretStr("mock-key")})
        db_client = MagicMock(
            get_node=AsyncMock(return_value=node),
            upsert_execution_output=AsyncMock(),
            get_node_execution=AsyncMock(return_value=None),
            log_platform_cost=AsyncMock(side_effect=self.log_cost),
        )
        return [
            patch.object(llm_models, "_today", self.clock.today),
            patch("backend.util.llm.providers.call_provider", self.provider),
            patch.object(
                billing,
                "get_db_client",
                return_value=MagicMock(spend_credits=MagicMock(side_effect=self.spend)),
            ),
            patch.object(
                billing,
                "get_database_manager_async_client",
                return_value=MagicMock(
                    spend_credits=AsyncMock(side_effect=self.spend_async)
                ),
            ),
            patch(
                "backend.executor.manager.get_db_async_client", return_value=db_client
            ),
            patch(
                "backend.executor.manager._get_node_credentials",
                AsyncMock(return_value=(key, None)),
            ),
            patch(
                "backend.executor.manager.acquire_auto_credentials",
                AsyncMock(return_value=({}, [])),
            ),
            patch(
                "backend.executor.manager.billing.charge_reconciled_usage",
                self.reconcile,
            ),
            patch(
                "backend.executor.manager.async_update_node_execution_status",
                self.update_status,
            ),
            patch(
                "backend.executor.manager.async_update_graph_execution_state",
                AsyncMock(),
            ),
        ]


def _node_exec(
    model: LLMModel,
    credentials: APIKeyCredentials,
    *,
    prompt: str = "Say hello.",
    graph_exec_id: str | None = None,
) -> NodeExecutionEntry:
    """The queued node, as the dispatcher builds it, a re-dispatch included:
    its ids and stored inputs."""
    graph_exec_id = graph_exec_id or str(uuid.uuid4())
    return NodeExecutionEntry(
        user_id="user-1",
        graph_exec_id=graph_exec_id,
        graph_id="graph-1",
        graph_version=1,
        node_exec_id=NODE_EXEC_ID,
        node_id="node-1",
        block_id=AITextGeneratorBlock().id,
        inputs={
            "prompt": prompt,
            "model": model.value,
            "credentials": {
                "id": credentials.id,
                "provider": credentials.provider,
                "type": credentials.type,
                "title": credentials.title,
            },
        },
        execution_context=ExecutionContext(root_execution_id=graph_exec_id),
    )


async def _attempt(
    node_exec: NodeExecutionEntry,
    credentials: APIKeyCredentials,
    *,
    charged_at: datetime | None,
    starts: datetime,
    settles: datetime,
    redispatched: bool = False,
    stopped_mid_call: bool = False,
    pod_dies_before_settling: bool = False,
) -> _Node:
    """One attempt at the node. The dispatcher charges it at *charged_at*
    (None: not again, as for a re-dispatch), and a worker starts it at
    *starts*; the clock reads *settles* from when the provider takes the
    request.

    - *redispatched*: the dispatcher sends it again after its pod died
      running it, and its settlement reads the pre-flight price again
      (`billing.redispatch_reconcile_baseline`).
    - *stopped_mid_call*: the provider never replies, and a stop request ends
      the run.
    - *pod_dies_before_settling*: the pod dies after the block ran and before
      it settled.
    """
    run = _Node(_Clock(charged_at or starts), settles, replies=not stopped_mid_call)
    if pod_dies_before_settling:
        run.reconcile = AsyncMock(return_value=(0, 0))
    node = Node(id="node-1", block_id=AITextGeneratorBlock().id)
    processor = ExecutionProcessor()
    processor.creds_manager = MagicMock()
    progress = NodeExecutionProgress()

    with ExitStack() as patches:
        for each in run.patches(credentials, node):
            patches.enter_context(each)
        if charged_at is not None:
            _, _, node_exec.pre_flight_charge = billing.charge_usage(node_exec, 0)
        if redispatched:
            node_exec.redispatched_from = ExecutionStatus.RUNNING
            node_exec.pre_flight_charge = billing.redispatch_reconcile_baseline(
                node_exec
            )
        run.clock.now = starts
        node_run = asyncio.ensure_future(
            processor.on_node_execution(
                node_exec=node_exec,
                node_exec_progress=progress,
                nodes_input_masks=None,
                graph_stats_pair=(GraphExecutionStats(), threading.Lock()),
            )
        )
        progress.add_task(NODE_EXEC_ID, node_run)
        try:
            if stopped_mid_call:
                await asyncio.wait_for(run.request_sent.wait(), timeout=60)
                progress.stop(stop_requested=True)
            finished, _ = await asyncio.wait({node_run}, timeout=60)
            assert finished, "the node didn't finish within 60 s"
        finally:
            # The node and its cost log settle before their patches come off,
            # however the wait ended.
            if not node_run.done():
                node_run.cancel()
                await asyncio.wait({node_run}, timeout=60)
            await drain_pending_cost_logs()
    return run


def _price_of_a_reply(model: LLMModel) -> int:
    return math.ceil(INPUT_TOKENS * TOKEN_COST[model].input / 1_000_000)


@pytest.mark.asyncio
@pytest.mark.parametrize(("saved", "credentials", "starts", "settles", "runs"), CASES)
async def test_a_node_is_billed_for_the_model_it_started_on(
    saved: LLMModel,
    credentials: APIKeyCredentials,
    starts: datetime,
    settles: datetime,
    runs: LLMModel,
):
    run = await _attempt(
        _node_exec(saved, credentials),
        credentials,
        charged_at=starts,
        starts=starts,
        settles=settles,
    )

    # The provider ran one model, and everything after names and prices it.
    assert run.asked == [runs.value]
    assert sum(run.spent) == _price_of_a_reply(runs)
    [logged] = run.cost_log
    assert logged.model == runs.value
    assert logged.metadata is not None
    assert logged.metadata["credit_cost"] == _price_of_a_reply(runs)
    assert run.recorded[-1]["resolved_model"] == runs.value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("saved", "credentials", "before", "after", "settles", "runs"), MIDNIGHTS
)
async def test_a_node_charged_before_midnight_and_started_after_runs_and_pays_the_fallback(
    saved: LLMModel,
    credentials: APIKeyCredentials,
    before: datetime,
    after: datetime,
    settles: datetime,
    runs: LLMModel,
):
    """The dispatcher charges the node one second before midnight, at the
    saved model's floor, and it starts one second after. Settlement charges
    the difference from what the wallet paid, so the total is the fallback's
    price for its reply: the model that ran."""
    run = await _attempt(
        _node_exec(saved, credentials),
        credentials,
        charged_at=before,
        starts=after,
        settles=settles,
    )

    assert run.asked == [runs.value]
    assert run.spent[0] == MODEL_COST[saved]
    assert sum(run.spent) == _price_of_a_reply(runs)
    [logged] = run.cost_log
    assert logged.model == runs.value
    assert run.recorded[-1]["resolved_model"] == runs.value


@pytest.mark.asyncio
async def test_a_node_rerun_after_its_pod_died_settles_on_the_model_it_started_on():
    """The node's pod died, and the dispatcher, which charged it at 23:50 on
    the eve of gpt-5-mini's shutdown date, sends it again; it starts at
    23:59:50. Its settlement reads the pre-flight price again, then prices the
    run, both for the model the rerun started on: in all, gpt-5-mini's
    price."""
    run = await _attempt(
        _node_exec(LLMModel.GPT5_MINI, openai_credentials),
        openai_credentials,
        charged_at=_utc(2026, 12, 10, 23, 50),
        starts=_utc(2026, 12, 10, 23, 59, 50),
        settles=_utc(2026, 12, 11, 0, 0, 10),
        redispatched=True,
    )

    assert run.asked == [LLMModel.GPT5_MINI.value]
    [settlement] = run.reconcile.await_args_list
    assert settlement.kwargs["pre_flight_charge"] is None
    assert sum(run.spent) == _price_of_a_reply(LLMModel.GPT5_MINI)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("saved", "credentials", "before", "after", "settles", "runs"), MIDNIGHTS
)
async def test_a_rerun_that_starts_after_midnight_runs_the_fallback_not_the_shut_down_model(
    saved: LLMModel,
    credentials: APIKeyCredentials,
    before: datetime,
    after: datetime,
    settles: datetime,
    runs: LLMModel,
):
    """The first attempt starts before midnight and calls the saved model,
    then its pod dies before it settles. The rerun starts after midnight and
    decides its model again: it calls the fallback (the provider has shut
    the saved model down), records it and bills it, not the first attempt's
    model."""
    graph_exec_id = str(uuid.uuid4())
    first = await _attempt(
        _node_exec(saved, credentials, graph_exec_id=graph_exec_id),
        credentials,
        charged_at=before,
        starts=before,
        settles=before,
        pod_dies_before_settling=True,
    )
    assert first.asked == [saved.value]

    rerun = await _attempt(
        _node_exec(saved, credentials, graph_exec_id=graph_exec_id),
        credentials,
        charged_at=None,
        starts=after,
        settles=settles,
        redispatched=True,
    )

    assert rerun.asked == [runs.value]
    [logged] = rerun.cost_log
    assert logged.model == runs.value
    assert rerun.recorded[-1]["resolved_model"] == runs.value
    # What the dead pod charged, then the rerun's settlement against the
    # pre-flight price of the model the rerun ran: the fallback's price for
    # its reply, out by the difference between the two models' floors (the
    # dead pod's pinned charge isn't kept).
    paid = sum(first.spent) + sum(rerun.spent)
    assert paid == _price_of_a_reply(runs) + MODEL_COST[saved] - MODEL_COST[runs]


@pytest.mark.asyncio
async def test_a_stop_after_midnight_bills_the_cut_off_request_at_the_started_models_rate():
    """A stop request ends the node while the provider has its request, now
    after midnight. The request's estimated input is billed at the input rate
    of the model the node started on, the one the provider was sent."""
    starts = _utc(2026, 12, 10, 23, 59, 50)
    run = await _attempt(
        # About 44,000 tokens: at gpt-5-mini's rate their price is more than
        # the pre-flight charge, and at its fallback's, more again.
        _node_exec(
            LLMModel.GPT5_MINI,
            openai_credentials,
            prompt=" ".join(str(n) for n in range(15_000)),
        ),
        openai_credentials,
        charged_at=starts,
        starts=starts,
        settles=_utc(2026, 12, 11, 0, 0, 10),
        stopped_mid_call=True,
    )

    assert run.asked == [LLMModel.GPT5_MINI.value]
    assert run.statuses[-1] == ExecutionStatus.TERMINATED
    [settlement] = run.reconcile.await_args_list
    assert settlement.kwargs["bill_stop_estimate"] is True
    estimate = settlement.kwargs["stats"].abandoned_input_token_count

    def billed_at(model: LLMModel) -> int:
        return math.ceil(estimate * TOKEN_COST[model].input / 1_000_000)

    # The pre-flight charge, then the rest of the estimate's price: a
    # settlement that charged nothing, or priced the fallback, can't pass.
    pre_flight = MODEL_COST[LLMModel.GPT5_MINI]
    assert pre_flight < billed_at(LLMModel.GPT5_MINI) < billed_at(LLMModel.GPT6_1_TERRA)
    assert run.spent == [pre_flight, billed_at(LLMModel.GPT5_MINI) - pre_flight]
    assert run.recorded[-1]["resolved_model"] == LLMModel.GPT5_MINI.value
