"""A node's pre-flight charge lands on the ledger its run was queued on, and
only a charge to a user's own wallet can warn that user its balance is low.

These drive the real ``_on_graph_execution`` → ``charge_usage`` chain for a
one-node run, with credits on, against ledgers that record every charge. Only
the block run itself (``_on_node_execution``) is replaced.
"""

import asyncio
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import BlockCost, BlockCostType
from backend.blocks.basic import StoreValueBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.execution import (
    ExecutionContext,
    ExecutionStatus,
    GraphExecutionEntry,
    NodeExecutionEntry,
)
from backend.data.model import GraphExecutionStats, NodeExecutionStats
from backend.executor import billing, manager
from backend.executor.manager import ExecutionProcessor
from backend.util.decorator import TimingInfo

USER_ID = "user-1"
GRAPH_EXEC_ID = "exec-1"
PRICE = 7
BALANCE = 1_000


class Ledgers:
    """Every charge, as (ledger, team, payer, cost); "wallet" is the user's own."""

    def __init__(self) -> None:
        self.charges: list[tuple[str, str | None, str, int]] = []

    def spend_credits(self, user_id: str, cost: int, **_) -> int:
        self.charges.append(("wallet", None, user_id, cost))
        return BALANCE - cost

    def spend_org_credits(
        self, org_id: str, user_id: str, cost: int, team_id: str | None = None, **_
    ) -> int:
        self.charges.append((org_id, team_id, user_id, cost))
        return BALANCE - cost


class NodeRow:
    """The persisted node execution."""

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


@pytest.fixture(autouse=True)
def _credits_enabled(monkeypatch):
    for module in (billing, manager):
        monkeypatch.setattr(module.settings.config, "enable_credit", True)


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
    ledgers = Ledgers()
    row = NodeRow()
    block = StoreValueBlock()

    sync_db = MagicMock()
    sync_db.spend_credits.side_effect = ledgers.spend_credits
    sync_db.spend_org_credits.side_effect = ledgers.spend_org_credits
    sync_db.get_credits.return_value = BALANCE
    sync_db.get_org_credits.return_value = BALANCE
    sync_db.get_node_executions.side_effect = lambda _id, statuses: (
        [row] if row.status in statuses else []
    )
    sync_db.has_pending_reviews_for_graph_exec.return_value = False
    sync_db.has_live_resource_access.return_value = True
    sync_db.acquire_live_resource_lease.return_value = "dispatch-lease"
    sync_db.release_live_resource_lease.return_value = True

    async_db = MagicMock()
    async_db.get_node = AsyncMock(return_value=SimpleNamespace(block=block))
    async_db.acquire_live_resource_lease = AsyncMock(return_value="lease-1")
    async_db.is_live_resource_lease_active = AsyncMock(return_value=True)
    async_db.release_live_resource_lease = AsyncMock(return_value=True)
    async_db.has_live_resource_access = AsyncMock(return_value=True)

    async def block_run(*, stats: NodeExecutionStats, **_):
        row.status = ExecutionStatus.RUNNING
        return TimingInfo(cpu_time=0.0, wall_time=0.0), ExecutionStatus.COMPLETED

    processor._on_node_execution = block_run
    original_costs = BLOCK_COSTS.get(StoreValueBlock)
    BLOCK_COSTS[StoreValueBlock] = [
        BlockCost(cost_amount=PRICE, cost_type=BlockCostType.RUN)
    ]
    with (
        patch("backend.executor.manager.get_db_client", return_value=sync_db),
        patch("backend.executor.billing.get_db_client", return_value=sync_db),
        patch("backend.executor.manager.get_db_async_client", return_value=async_db),
        patch(
            "backend.executor.billing.get_database_manager_async_client",
            return_value=async_db,
        ),
        patch("backend.executor.manager.increment_execution_count", return_value=1),
        patch("backend.executor.billing.execution_usage_cost", return_value=(0, 0)),
        patch("backend.executor.billing.handle_low_balance") as low_balance,
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

        def run(execution_context: ExecutionContext) -> None:
            processor._on_graph_execution(
                graph_exec=GraphExecutionEntry(
                    user_id=USER_ID,
                    graph_exec_id=GRAPH_EXEC_ID,
                    graph_id="graph-1",
                    graph_version=1,
                    execution_context=execution_context,
                ),
                cancel=threading.Event(),
                log_metadata=MagicMock(),
                execution_stats=GraphExecutionStats(),
                cluster_lock=MagicMock(),
            )

        yield SimpleNamespace(
            ledgers=ledgers, low_balance=low_balance, sync_db=sync_db, run=run
        )

    if original_costs is None:
        BLOCK_COSTS.pop(StoreValueBlock, None)
    else:
        BLOCK_COSTS[StoreValueBlock] = original_costs


@pytest.mark.parametrize(
    "execution_context",
    [
        pytest.param(ExecutionContext(), id="no organization"),
        pytest.param(
            ExecutionContext(
                organization_id="org-personal", team_id="team-1", bills_own_wallet=True
            ),
            id="queued on the user's own wallet in their personal organization",
        ),
    ],
)
def test_a_charge_to_the_users_wallet_can_warn_them(harness, execution_context):
    harness.run(execution_context)

    assert harness.ledgers.charges == [("wallet", None, USER_ID, PRICE)]
    harness.low_balance.assert_called_once_with(
        db_client=harness.sync_db,
        user_id=USER_ID,
        current_balance=BALANCE - PRICE,
        transaction_cost=PRICE,
    )


def test_a_charge_to_an_organizations_ledger_warns_no_user(harness):
    harness.run(ExecutionContext(organization_id="org-1", team_id="team-1"))

    assert harness.ledgers.charges == [("org-1", "team-1", USER_ID, PRICE)]
    harness.low_balance.assert_not_called()
    # The ledger was decided when the run was queued; it is not read again.
    harness.sync_db.get_personal_org_owner.assert_not_called()
