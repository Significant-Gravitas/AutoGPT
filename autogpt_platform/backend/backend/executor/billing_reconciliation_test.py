"""Tests for charge_reconciled_usage post-flight dynamic-cost charging."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from backend.blocks._base import BlockCost, BlockCostType
from backend.blocks.jina.search import SearchTheWebBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.model import NodeExecutionStats
from backend.executor.billing import charge_reconciled_usage


@pytest.fixture
def tmp_block_costs_override():
    original = BLOCK_COSTS.get(SearchTheWebBlock)
    yield lambda costs: BLOCK_COSTS.__setitem__(SearchTheWebBlock, costs)
    if original is None:
        BLOCK_COSTS.pop(SearchTheWebBlock, None)
    else:
        BLOCK_COSTS[SearchTheWebBlock] = original


def _node_exec(block_id: str):
    return NodeExecutionEntry(
        user_id="test-user",
        graph_exec_id=str(uuid4()),
        graph_id=str(uuid4()),
        graph_version=1,
        node_exec_id=str(uuid4()),
        node_id=str(uuid4()),
        block_id=block_id,
        inputs={},
        execution_context=ExecutionContext(),
    )


def _async_db_client(spend_credits_return: int = 0) -> MagicMock:
    """Build an AsyncMock-backed db client stand-in for the reconciliation path.

    ``spend_credits`` is awaited in the production code, so it must be an
    AsyncMock. ``get_credits`` is only read by the sync pre-flight path.
    """
    client = MagicMock()
    client.spend_credits = AsyncMock(return_value=spend_credits_return)
    client.get_credits = MagicMock(return_value=0)
    return client


def test_dynamic_cost_block_with_zero_balance_raises_ibe_preflight(
    tmp_block_costs_override,
):
    """Sentry-flagged bug: dynamic-cost blocks (SECOND/ITEMS/COST_USD) have
    pre-flight cost 0, so without a guard a zero-balance user could run the
    block and leak the post-flight provider spend as an uncollectable
    debit. Verify charge_usage raises InsufficientBalanceError when the
    user has no balance and the block has a dynamic cost entry.
    """
    from backend.executor.billing import charge_usage
    from backend.util.exceptions import InsufficientBalanceError

    tmp_block_costs_override(
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    exec_entry = _node_exec(SearchTheWebBlock().id)

    db_client = MagicMock()
    db_client.get_credits.return_value = 0  # empty wallet

    with (
        patch("backend.executor.billing.get_db_client", return_value=db_client),
        patch("backend.executor.billing.get_block", return_value=SearchTheWebBlock()),
    ):
        with pytest.raises(InsufficientBalanceError):
            charge_usage(exec_entry, execution_count=0)

    db_client.get_credits.assert_called_once_with(user_id=exec_entry.user_id)
    db_client.spend_credits.assert_not_called()


def test_dynamic_cost_block_with_positive_balance_starts(tmp_block_costs_override):
    """The guard must only fire when balance is non-positive. A user with any
    positive balance may start a dynamic-cost block; reconciliation settles
    the actual charge afterward.
    """
    from backend.executor.billing import charge_usage

    tmp_block_costs_override(
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    exec_entry = _node_exec(SearchTheWebBlock().id)

    db_client = MagicMock()
    db_client.get_credits.return_value = 50  # has balance

    with (
        patch("backend.executor.billing.get_db_client", return_value=db_client),
        patch("backend.executor.billing.get_block", return_value=SearchTheWebBlock()),
    ):
        total_cost, remaining = charge_usage(exec_entry, execution_count=0)

    assert total_cost == 0
    assert remaining == 50
    db_client.spend_credits.assert_not_called()


@pytest.mark.asyncio
async def test_run_cost_produces_zero_delta_noop(tmp_block_costs_override):
    tmp_block_costs_override([BlockCost(cost_amount=7, cost_type=BlockCostType.RUN)])
    exec_entry = _node_exec(SearchTheWebBlock().id)
    stats = NodeExecutionStats(walltime=25.0)

    db_client = _async_db_client()
    with patch(
        "backend.executor.billing.get_database_manager_async_client",
        return_value=db_client,
    ):
        delta, _ = await charge_reconciled_usage(exec_entry, stats)

    # RUN type: pre == post == 7, so reconciliation charges nothing.
    assert delta == 0
    db_client.spend_credits.assert_not_awaited()


@pytest.mark.asyncio
async def test_cost_usd_charges_post_flight_delta(tmp_block_costs_override):
    tmp_block_costs_override(
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    exec_entry = _node_exec(SearchTheWebBlock().id)
    stats = NodeExecutionStats(provider_cost=0.05, provider_cost_type="cost_usd")

    db_client = _async_db_client(spend_credits_return=42)
    with (
        patch(
            "backend.executor.billing.get_database_manager_async_client",
            return_value=db_client,
        ),
        patch("backend.executor.billing.get_db_client", return_value=MagicMock()),
        patch("backend.executor.billing.handle_low_balance") as handle_lb,
    ):
        delta, remaining = await charge_reconciled_usage(exec_entry, stats)

    # Pre-flight COST_USD returns 0 (no stats). Post-flight: ceil(0.05 * 100) = 5.
    assert delta == 5
    assert remaining == 42
    db_client.spend_credits.assert_awaited_once()
    call_kwargs = db_client.spend_credits.await_args.kwargs
    assert call_kwargs["cost"] == 5
    # Positive delta should also fire the low-balance notification so users
    # get alerted when reconciliation crosses the threshold.
    handle_lb.assert_called_once()
    lb_args = handle_lb.call_args.args
    assert lb_args[1] == exec_entry.user_id
    assert lb_args[2] == 42
    assert lb_args[3] == 5


@pytest.mark.asyncio
async def test_positive_delta_passes_fail_insufficient_credits_false(
    tmp_block_costs_override,
):
    """Reconciliation must call spend_credits with
    fail_insufficient_credits=False on a positive delta — the wallet is
    allowed to go negative so the platform records debt instead of leaking
    the cost."""
    tmp_block_costs_override(
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    exec_entry = _node_exec(SearchTheWebBlock().id)
    stats = NodeExecutionStats(provider_cost=0.05, provider_cost_type="cost_usd")

    # Wallet went negative: the user had 1 credit, the delta is 5, the new
    # balance is -4. spend_credits returns the post-spend balance.
    db_client = _async_db_client(spend_credits_return=-4)
    with (
        patch(
            "backend.executor.billing.get_database_manager_async_client",
            return_value=db_client,
        ),
        patch("backend.executor.billing.get_db_client", return_value=MagicMock()),
        patch("backend.executor.billing.handle_low_balance"),
    ):
        delta, remaining = await charge_reconciled_usage(exec_entry, stats)

    assert delta == 5
    assert remaining == -4
    db_client.spend_credits.assert_awaited_once()
    call_kwargs = db_client.spend_credits.await_args.kwargs
    assert call_kwargs["cost"] == 5
    assert call_kwargs["fail_insufficient_credits"] is False


@pytest.mark.asyncio
async def test_missing_block_returns_zero():
    exec_entry = _node_exec("deadbeef-0000-0000-0000-000000000000")
    stats = NodeExecutionStats(walltime=10)
    with patch("backend.executor.billing.get_block", return_value=None):
        delta, _ = await charge_reconciled_usage(exec_entry, stats)
    assert delta == 0


@pytest.mark.asyncio
async def test_items_cost_scales_linearly_with_result_count(tmp_block_costs_override):
    """ITEMS with cost_divisor=2 bills 1 credit per 2 returned items.

    Apollo SearchOrganizationsBlock uses this exact config. Verifies the
    divisor path in the resolver + post-flight charge.
    """
    tmp_block_costs_override(
        [
            BlockCost(
                cost_amount=1,
                cost_type=BlockCostType.ITEMS,
                cost_divisor=2,
            )
        ]
    )
    exec_entry = _node_exec(SearchTheWebBlock().id)
    # Simulate 20 returned organizations.
    stats = NodeExecutionStats(provider_cost=20, provider_cost_type="items")

    db_client = _async_db_client(spend_credits_return=500)
    with (
        patch(
            "backend.executor.billing.get_database_manager_async_client",
            return_value=db_client,
        ),
        patch("backend.executor.billing.get_db_client", return_value=MagicMock()),
        patch("backend.executor.billing.handle_low_balance"),
    ):
        delta, _ = await charge_reconciled_usage(exec_entry, stats)

    # 20 items / cost_divisor=2 * cost_amount=1 = 10 credits.
    assert delta == 10
    call_kwargs = db_client.spend_credits.await_args.kwargs
    assert call_kwargs["cost"] == 10
    meta_input = call_kwargs["metadata"].input
    assert meta_input.get("reconciled_delta") == 10


@pytest.mark.asyncio
async def test_items_cost_bills_zero_when_no_items_returned(tmp_block_costs_override):
    """An ITEMS block that returns 0 results should bill 0, not the floor."""
    tmp_block_costs_override(
        [BlockCost(cost_amount=1, cost_type=BlockCostType.ITEMS, cost_divisor=2)]
    )
    exec_entry = _node_exec(SearchTheWebBlock().id)
    stats = NodeExecutionStats(provider_cost=0, provider_cost_type="items")

    db_client = _async_db_client()
    with patch(
        "backend.executor.billing.get_database_manager_async_client",
        return_value=db_client,
    ):
        delta, _ = await charge_reconciled_usage(exec_entry, stats)

    assert delta == 0
    db_client.spend_credits.assert_not_awaited()


@pytest.mark.asyncio
async def test_cost_usd_with_larger_spend_bills_full_delta(tmp_block_costs_override):
    """Exa deep-research: $0.20 provider spend × 100 credits/USD = 20 credits.

    Verifies ceil-semantics on fractional USD amounts and that the full
    post-flight charge lands (no refund / no clamping).
    """
    tmp_block_costs_override(
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    exec_entry = _node_exec(SearchTheWebBlock().id)
    # $0.207 spend: ceil(0.207 * 100) = 21
    stats = NodeExecutionStats(provider_cost=0.207, provider_cost_type="cost_usd")

    db_client = _async_db_client(spend_credits_return=100)
    with (
        patch(
            "backend.executor.billing.get_database_manager_async_client",
            return_value=db_client,
        ),
        patch("backend.executor.billing.get_db_client", return_value=MagicMock()),
        patch("backend.executor.billing.handle_low_balance"),
    ):
        delta, _ = await charge_reconciled_usage(exec_entry, stats)

    assert delta == 21


@pytest.mark.asyncio
async def test_tokens_cost_refunds_when_actual_below_estimate(tmp_block_costs_override):
    """TOKENS pre-flight uses MODEL_COST floor; if real token usage is cheaper,
    the user is refunded the overcharge via a negative-delta spend_credits."""
    from backend.blocks.llm import LlmModel

    tmp_block_costs_override(
        [
            BlockCost(
                cost_amount=1,
                cost_type=BlockCostType.TOKENS,
                cost_filter={"model": LlmModel.GPT5},
            )
        ]
    )
    exec_entry = _node_exec(SearchTheWebBlock().id)
    exec_entry = exec_entry.model_copy(update={"inputs": {"model": LlmModel.GPT5}})
    # Minimal real usage → post-flight < pre-flight MODEL_COST floor.
    stats = NodeExecutionStats(
        input_token_count=1,
        output_token_count=1,
    )

    db_client = _async_db_client(spend_credits_return=999)
    with (
        patch(
            "backend.executor.billing.get_database_manager_async_client",
            return_value=db_client,
        ),
        patch("backend.executor.billing.handle_low_balance") as handle_lb,
    ):
        delta, remaining = await charge_reconciled_usage(exec_entry, stats)

    assert delta < 0
    assert remaining == 999
    db_client.spend_credits.assert_awaited_once()
    call_kwargs = db_client.spend_credits.await_args.kwargs
    assert call_kwargs["cost"] == delta  # negative cost ⇒ credit back
    # Refunds can't push the user below the low-balance threshold, so
    # handle_low_balance must not fire here.
    handle_lb.assert_not_called()
