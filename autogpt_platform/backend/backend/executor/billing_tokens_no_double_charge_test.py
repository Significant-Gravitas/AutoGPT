"""Regression: TOKENS/COST_USD/ITEMS blocks must not pay per-iteration fees.

Sentry HIGH #13735365 / #13735625: `handle_post_execution_billing` previously
applied a flat MODEL_COST per extra LLM call for every block that overrode
`extra_runtime_cost()`. For dynamic-cost blocks (TOKENS/COST_USD/ITEMS) the
reconciliation step already settles every iteration via aggregate stats, so
billing the flat per-iteration fee double-charges the user.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.blocks._base import BlockCost, BlockCostType
from backend.blocks.jina.search import SearchTheWebBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.execution import ExecutionContext, ExecutionStatus, NodeExecutionEntry
from backend.data.model import NodeExecutionStats
from backend.executor.billing import handle_post_execution_billing


@pytest.fixture
def tmp_block_costs_override():
    original = BLOCK_COSTS.get(SearchTheWebBlock)
    yield lambda costs: BLOCK_COSTS.__setitem__(SearchTheWebBlock, costs)
    if original is None:
        BLOCK_COSTS.pop(SearchTheWebBlock, None)
    else:
        BLOCK_COSTS[SearchTheWebBlock] = original


def _node_and_entry(extra_iterations: int) -> tuple[object, NodeExecutionEntry]:
    block_instance = SearchTheWebBlock()
    # Stub out extra_runtime_cost on the instance so the test controls the
    # number of extra iterations without needing an OrchestratorBlock-backed
    # Node graph fixture.
    block_instance.extra_runtime_cost = lambda stats: extra_iterations  # type: ignore[method-assign]
    node = SimpleNamespace(block=block_instance)
    entry = NodeExecutionEntry(
        user_id="u",
        graph_exec_id="g",
        graph_id="gid",
        graph_version=1,
        node_exec_id="n",
        node_id="nid",
        block_id=block_instance.id,
        inputs={},
        execution_context=ExecutionContext(),
    )
    return node, entry


@pytest.mark.asyncio
async def test_tokens_skips_extra_runtime_charge(tmp_block_costs_override):
    tmp_block_costs_override(
        [BlockCost(cost_amount=10, cost_type=BlockCostType.TOKENS)]
    )
    node, entry = _node_and_entry(extra_iterations=4)
    stats = NodeExecutionStats(input_token_count=1, output_token_count=1)
    log_metadata = MagicMock()

    with patch(
        "backend.executor.billing.charge_extra_runtime_cost",
        new=AsyncMock(return_value=(0, 0)),
    ) as charge_mock:
        await handle_post_execution_billing(
            node=node,
            node_exec=entry,
            execution_stats=stats,
            status=ExecutionStatus.COMPLETED,
            log_metadata=log_metadata,
        )

    charge_mock.assert_not_called()


@pytest.mark.asyncio
async def test_cost_usd_skips_extra_runtime_charge(tmp_block_costs_override):
    tmp_block_costs_override(
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    node, entry = _node_and_entry(extra_iterations=2)
    stats = NodeExecutionStats(provider_cost=0.01, provider_cost_type="cost_usd")

    with patch(
        "backend.executor.billing.charge_extra_runtime_cost",
        new=AsyncMock(return_value=(0, 0)),
    ) as charge_mock:
        await handle_post_execution_billing(
            node=node,
            node_exec=entry,
            execution_stats=stats,
            status=ExecutionStatus.COMPLETED,
            log_metadata=MagicMock(),
        )

    charge_mock.assert_not_called()


@pytest.mark.asyncio
async def test_items_skips_extra_runtime_charge(tmp_block_costs_override):
    tmp_block_costs_override([BlockCost(cost_amount=5, cost_type=BlockCostType.ITEMS)])
    node, entry = _node_and_entry(extra_iterations=3)
    stats = NodeExecutionStats(provider_cost=10, provider_cost_type="items")

    with patch(
        "backend.executor.billing.charge_extra_runtime_cost",
        new=AsyncMock(return_value=(0, 0)),
    ) as charge_mock:
        await handle_post_execution_billing(
            node=node,
            node_exec=entry,
            execution_stats=stats,
            status=ExecutionStatus.COMPLETED,
            log_metadata=MagicMock(),
        )

    charge_mock.assert_not_called()


@pytest.mark.asyncio
async def test_run_cost_still_charges_extra_runtime(tmp_block_costs_override):
    """Regression guard: RUN-type blocks must keep paying the flat fee."""
    tmp_block_costs_override([BlockCost(cost_amount=7, cost_type=BlockCostType.RUN)])
    node, entry = _node_and_entry(extra_iterations=3)
    stats = NodeExecutionStats()

    with patch(
        "backend.executor.billing.charge_extra_runtime_cost",
        new=AsyncMock(return_value=(21, 100)),
    ) as charge_mock:
        await handle_post_execution_billing(
            node=node,
            node_exec=entry,
            execution_stats=stats,
            status=ExecutionStatus.COMPLETED,
            log_metadata=MagicMock(),
        )

    charge_mock.assert_awaited_once()
    assert charge_mock.await_args.args[1] == 3
