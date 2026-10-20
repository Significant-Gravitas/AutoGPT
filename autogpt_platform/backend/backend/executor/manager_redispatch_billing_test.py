"""A node execution is billed once however many times it is dispatched (#15272).

These drive the real ``_on_graph_execution`` → ``charge_usage`` →
``on_node_execution`` → ``charge_reconciled_usage`` chain, on real event loops,
against a wallet that records every spend. Only the block run itself
(``_on_node_execution``) is replaced, so each dispatch can park the block for
review, run it, or die mid-run the way the real one does.
"""

import asyncio
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import BlockCost, BlockCostType
from backend.blocks.basic import StoreValueBlock
from backend.blocks.helpers.review import HITLReviewHelper
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    GraphExecutionEntry,
    NodeExecutionEntry,
)
from backend.data.graph import Node
from backend.data.model import GraphExecutionStats, NodeExecutionStats
from backend.executor.manager import ExecutionProcessor
from backend.executor.utils import NodeExecutionProgress
from backend.util.decorator import TimingInfo

USER_ID = "user-1"
GRAPH_EXEC_ID = "exec-1"
PRE_FLIGHT_ESTIMATE = 10
PER_RUN_PRICE = 2
RUN_USD = 15.0  # a COST_USD block at 1 credit per dollar bills 15 for the run

PER_RUN = [BlockCost(cost_amount=PER_RUN_PRICE, cost_type=BlockCostType.RUN)]
USAGE_PRICED = [BlockCost(cost_amount=1, cost_type=BlockCostType.COST_USD)]


class Wallet:
    """Every credit spend, sync (pre-flight) and async (reconciliation)."""

    def __init__(self) -> None:
        self.spent: list[int] = []

    @property
    def total(self) -> int:
        return sum(self.spent)

    def spend(self, user_id: str, cost: int, metadata, **_) -> int:
        self.spent.append(cost)
        return 1_000 - self.total

    async def spend_async(self, user_id: str, cost: int, metadata, **_) -> int:
        return self.spend(user_id, cost, metadata)


class NodeRow:
    """The persisted node execution: the only state that survives a dispatch."""

    def __init__(self) -> None:
        self.status = ExecutionStatus.QUEUED
        self.queue_time = None
        self.entry = NodeExecutionEntry(
            user_id=USER_ID,
            graph_exec_id=GRAPH_EXEC_ID,
            graph_id="graph-1",
            graph_version=1,
            node_exec_id="node-exec-1",
            node_id="node-1",
            block_id=StoreValueBlock().id,
            inputs={"input": "post this"},
        )

    def to_node_execution_entry(self, execution_context: ExecutionContext):
        return self.entry.model_copy(update={"execution_context": execution_context})


def _parks_for_review(row: NodeRow):
    """The block writes REVIEW and returns without output, so the run reports
    COMPLETED (``_base.py`` ``_execute`` returns early on a pending review)."""

    async def run(**_):
        row.status = ExecutionStatus.REVIEW
        return TimingInfo(cpu_time=0.0, wall_time=0.0), ExecutionStatus.COMPLETED

    return run


def _runs(row: NodeRow, *, approved: bool = True):
    async def run(*, stats: NodeExecutionStats, **_):
        row.status = ExecutionStatus.RUNNING
        if not approved:
            stats.error = ValueError("Block execution rejected by reviewer")
            return TimingInfo(cpu_time=0.0, wall_time=0.0), ExecutionStatus.FAILED
        stats.provider_cost = RUN_USD
        stats.provider_cost_type = "cost_usd"
        return TimingInfo(cpu_time=0.0, wall_time=0.0), ExecutionStatus.COMPLETED

    return run


def _dies_mid_run(row: NodeRow):
    """The pod goes away after RUNNING is written, before reconciliation."""

    async def run(**_):
        row.status = ExecutionStatus.RUNNING
        return TimingInfo(cpu_time=0.0, wall_time=0.0), RuntimeError("pod died")

    return run


@pytest.fixture
def processor() -> Iterator[ExecutionProcessor]:
    processor = ExecutionProcessor()
    loops = [asyncio.new_event_loop() for _ in range(2)]
    threads = [threading.Thread(target=loop.run_forever, daemon=True) for loop in loops]
    for thread in threads:
        thread.start()
    processor.node_execution_loop, processor.node_evaluation_loop = loops
    processor._cleanup_graph_execution = MagicMock()
    yield processor
    for loop, thread in zip(loops, threads):
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


@pytest.fixture
def harness(processor: ExecutionProcessor):
    wallet = Wallet()
    row = NodeRow()
    block = StoreValueBlock()

    sync_db = MagicMock()
    sync_db.spend_credits.side_effect = wallet.spend
    sync_db.get_credits.return_value = 1_000
    sync_db.get_node_executions.side_effect = lambda _id, statuses: (
        [row] if row.status in statuses else []
    )
    sync_db.has_pending_reviews_for_graph_exec.side_effect = (
        lambda _id: row.status == ExecutionStatus.REVIEW
    )

    async_db = MagicMock()
    async_db.spend_credits = AsyncMock(side_effect=wallet.spend_async)
    async_db.get_node = AsyncMock(return_value=SimpleNamespace(block=block))

    original_costs = BLOCK_COSTS.get(StoreValueBlock)
    with (
        patch("backend.executor.manager.get_db_client", return_value=sync_db),
        patch("backend.executor.billing.get_db_client", return_value=sync_db),
        patch("backend.executor.manager.get_db_async_client", return_value=async_db),
        patch(
            "backend.executor.billing.get_database_manager_async_client",
            return_value=async_db,
        ),
        patch(
            "backend.executor.utils.get_preflight_estimate",
            return_value=PRE_FLIGHT_ESTIMATE,
        ),
        patch("backend.executor.manager.increment_execution_count", return_value=1),
        patch("backend.executor.billing.execution_usage_cost", return_value=(0, 0)),
        patch("backend.executor.billing.handle_low_balance"),
        patch("backend.executor.manager.automod_manager") as automod,
        patch("backend.executor.manager.log_system_credential_cost", AsyncMock()),
        patch(
            "backend.executor.manager.activity_events.log_node_integration_activity",
            AsyncMock(),
        ),
        patch("backend.executor.manager.async_update_node_execution_status"),
        patch("backend.executor.manager.async_update_graph_execution_state"),
    ):
        automod.moderate_graph_execution_inputs = AsyncMock(return_value=None)
        automod.moderate_graph_execution_outputs = AsyncMock(return_value=None)

        def dispatch(block_run) -> ExecutionStatus:
            processor._on_node_execution = block_run
            _, status = processor._on_graph_execution(
                graph_exec=GraphExecutionEntry(
                    user_id=USER_ID,
                    graph_exec_id=GRAPH_EXEC_ID,
                    graph_id="graph-1",
                    graph_version=1,
                ),
                cancel=threading.Event(),
                log_metadata=MagicMock(),
                execution_stats=GraphExecutionStats(),
                cluster_lock=MagicMock(),
            )
            return status

        def price(costs: list[BlockCost]) -> None:
            BLOCK_COSTS[StoreValueBlock] = costs

        yield SimpleNamespace(wallet=wallet, row=row, dispatch=dispatch, price=price)

    if original_costs is None:
        BLOCK_COSTS.pop(StoreValueBlock, None)
    else:
        BLOCK_COSTS[StoreValueBlock] = original_costs


@pytest.mark.parametrize(
    "costs, approved, billed",
    [
        pytest.param(PER_RUN, True, PER_RUN_PRICE, id="per-run, approved"),
        pytest.param(PER_RUN, False, PER_RUN_PRICE, id="per-run, rejected"),
        pytest.param(USAGE_PRICED, True, int(RUN_USD), id="usage-priced, approved"),
        pytest.param(USAGE_PRICED, False, 0, id="usage-priced, rejected"),
    ],
)
def test_a_review_resume_bills_the_block_once(harness, costs, approved, billed):
    harness.price(costs)

    assert harness.dispatch(_parks_for_review(harness.row)) == ExecutionStatus.REVIEW
    assert harness.row.status == ExecutionStatus.REVIEW

    harness.dispatch(_runs(harness.row, approved=approved))

    assert harness.wallet.total == billed, harness.wallet.spent


@pytest.mark.parametrize(
    "costs, billed",
    [
        pytest.param(PER_RUN, PER_RUN_PRICE, id="per-run"),
        pytest.param(USAGE_PRICED, int(RUN_USD), id="usage-priced"),
    ],
)
def test_a_redelivery_after_a_pod_died_bills_the_block_once(harness, costs, billed):
    harness.price(costs)

    harness.dispatch(_dies_mid_run(harness.row))
    assert harness.row.status == ExecutionStatus.RUNNING

    harness.dispatch(_runs(harness.row))

    assert harness.wallet.total == billed, harness.wallet.spent


async def test_a_parked_block_reports_a_completed_run_with_nothing_to_bill():
    """The premise ``_parks_for_review`` stubs, through the real block run: a
    block that parks for review yields nothing and its run reports COMPLETED."""
    block = StoreValueBlock()
    block.is_irreversible_action = True
    row = NodeRow()
    processor = ExecutionProcessor()
    processor.creds_manager = MagicMock()
    progress = NodeExecutionProgress()
    stats = NodeExecutionStats()

    with (
        patch("backend.data.graph.get_block", return_value=block),
        patch("backend.executor.manager.async_update_node_execution_status"),
        patch.object(
            HITLReviewHelper, "handle_review_decision", AsyncMock(return_value=None)
        ) as review,
    ):
        _, status = await processor._on_node_execution(
            node=Node(id=row.entry.node_id, block_id=block.id),
            node_exec=row.to_node_execution_entry(
                ExecutionContext(
                    graph_exec_id=GRAPH_EXEC_ID, sensitive_action_safe_mode=True
                )
            ),
            node_exec_progress=progress,
            stats=stats,
            db_client=MagicMock(
                upsert_execution_output=AsyncMock(),
                get_node_execution=AsyncMock(return_value=None),
            ),
            log_metadata=MagicMock(),
        )

    review.assert_awaited_once()
    assert status == ExecutionStatus.COMPLETED
    assert not progress.output
    assert stats.error is None
    assert stats.provider_cost is None
