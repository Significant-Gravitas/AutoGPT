"""Tests for charge_reconciled_usage post-flight dynamic-cost charging."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from backend.blocks._base import BlockCost, BlockCostType
from backend.blocks.jina.search import SearchTheWebBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.execution import ExecutionContext, NodeExecutionEntry
from backend.data.model import NodeExecutionStats
from backend.executor.billing import _charge_reconciled_usage_sync


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


def test_run_cost_produces_zero_delta_noop(tmp_block_costs_override):
    tmp_block_costs_override([BlockCost(cost_amount=7, cost_type=BlockCostType.RUN)])
    exec_entry = _node_exec(SearchTheWebBlock().id)
    stats = NodeExecutionStats(walltime=25.0)

    db_client = MagicMock()
    with patch("backend.executor.billing.get_db_client", return_value=db_client):
        delta, _ = _charge_reconciled_usage_sync(exec_entry, stats)

    # RUN type: pre == post == 7, so reconciliation charges nothing.
    assert delta == 0
    db_client.spend_credits.assert_not_called()


def test_cost_usd_charges_post_flight_delta(tmp_block_costs_override):
    tmp_block_costs_override(
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    exec_entry = _node_exec(SearchTheWebBlock().id)
    stats = NodeExecutionStats(provider_cost=0.05, provider_cost_type="cost_usd")

    db_client = MagicMock()
    db_client.spend_credits.return_value = 42  # remaining balance
    with patch("backend.executor.billing.get_db_client", return_value=db_client):
        delta, remaining = _charge_reconciled_usage_sync(exec_entry, stats)

    # Pre-flight COST_USD returns 0 (no stats). Post-flight: ceil(0.05 * 100) = 5.
    assert delta == 5
    assert remaining == 42
    db_client.spend_credits.assert_called_once()
    call_kwargs = db_client.spend_credits.call_args.kwargs
    assert call_kwargs["cost"] == 5


def test_missing_block_returns_zero(tmp_block_costs_override):
    exec_entry = _node_exec("deadbeef-0000-0000-0000-000000000000")
    stats = NodeExecutionStats(walltime=10)
    with patch("backend.executor.billing.get_block", return_value=None):
        delta, _ = _charge_reconciled_usage_sync(exec_entry, stats)
    assert delta == 0


def test_tokens_cost_refunds_when_actual_below_estimate(tmp_block_costs_override):
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

    db_client = MagicMock()
    db_client.spend_credits.return_value = 999
    with patch("backend.executor.billing.get_db_client", return_value=db_client):
        delta, remaining = _charge_reconciled_usage_sync(exec_entry, stats)

    assert delta < 0
    assert remaining == 999
    db_client.spend_credits.assert_called_once()
    call_kwargs = db_client.spend_credits.call_args.kwargs
    assert call_kwargs["cost"] == delta  # negative cost ⇒ credit back
