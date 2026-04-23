"""Tests for the dynamic-pricing branches of block_usage_cost."""

import pytest

from backend.blocks._base import BlockCost, BlockCostType
from backend.blocks.jina.search import SearchTheWebBlock
from backend.data.block_cost_config import BLOCK_COSTS, TOKEN_COST, TokenRate
from backend.data.model import NodeExecutionStats
from backend.executor.utils import block_usage_cost


@pytest.fixture
def tmp_block_costs_override():
    """Swap out BLOCK_COSTS[SearchTheWebBlock] for the duration of a test."""
    original = BLOCK_COSTS.get(SearchTheWebBlock)
    yield lambda costs: BLOCK_COSTS.__setitem__(SearchTheWebBlock, costs)
    if original is None:
        BLOCK_COSTS.pop(SearchTheWebBlock, None)
    else:
        BLOCK_COSTS[SearchTheWebBlock] = original


def test_second_cost_type_uses_stats_walltime_with_divisor(tmp_block_costs_override):
    tmp_block_costs_override(
        [BlockCost(cost_amount=1, cost_type=BlockCostType.SECOND, cost_divisor=10)]
    )
    block = SearchTheWebBlock()
    # 25 seconds of walltime / 10 sec-per-credit = 2 credits (integer div).
    stats = NodeExecutionStats(walltime=25.0)
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 2


def test_second_cost_type_returns_zero_without_stats_or_runtime(
    tmp_block_costs_override,
):
    tmp_block_costs_override([BlockCost(cost_amount=1, cost_type=BlockCostType.SECOND)])
    block = SearchTheWebBlock()
    # Pre-flight: no stats, no run_time. SECOND should return 0 credits.
    cost, _ = block_usage_cost(block, {})
    assert cost == 0


def test_items_cost_type_multiplies_provider_cost(tmp_block_costs_override):
    tmp_block_costs_override(
        [BlockCost(cost_amount=1, cost_type=BlockCostType.ITEMS, cost_divisor=10)]
    )
    block = SearchTheWebBlock()
    stats = NodeExecutionStats(provider_cost=45, provider_cost_type="items")
    # 45 items / 10 = 4 credits.
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 4


def test_items_cost_type_ignores_non_items_provider_cost(tmp_block_costs_override):
    tmp_block_costs_override([BlockCost(cost_amount=1, cost_type=BlockCostType.ITEMS)])
    block = SearchTheWebBlock()
    # provider_cost is USD, not item count — don't misread as items.
    stats = NodeExecutionStats(provider_cost=0.05, provider_cost_type="cost_usd")
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 0


def test_cost_usd_ceils_multiplier(tmp_block_costs_override):
    tmp_block_costs_override(
        # 100 credits per USD, so $0.023 provider spend → 3 credits (ceil of 2.3).
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    block = SearchTheWebBlock()
    stats = NodeExecutionStats(provider_cost=0.023, provider_cost_type="cost_usd")
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 3


def test_cost_usd_zero_when_no_stats(tmp_block_costs_override):
    tmp_block_costs_override(
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    block = SearchTheWebBlock()
    cost, _ = block_usage_cost(block, {})
    assert cost == 0


def test_tokens_cost_type_uses_token_rate_table(tmp_block_costs_override, monkeypatch):
    from backend.blocks.llm import LlmModel

    tmp_block_costs_override([BlockCost(cost_amount=0, cost_type=BlockCostType.TOKENS)])
    # Override TOKEN_COST for a predictable rate: 1000 credits/1M input,
    # 2000 credits/1M output.
    monkeypatch.setitem(
        TOKEN_COST,
        LlmModel.GPT4O_MINI,
        TokenRate(input=1000, output=2000),
    )
    block = SearchTheWebBlock()
    stats = NodeExecutionStats(
        input_token_count=500_000,
        output_token_count=250_000,
    )
    cost, _ = block_usage_cost(
        block,
        {"model": LlmModel.GPT4O_MINI.value},
        stats=stats,
    )
    # 0.5 * 1000 + 0.25 * 2000 = 500 + 500 = 1000 credits.
    assert cost == 1000


def test_tokens_falls_back_to_flat_model_cost_when_rate_missing(
    tmp_block_costs_override,
):
    from backend.blocks.llm import LlmModel
    from backend.data.block_cost_config import MODEL_COST

    tmp_block_costs_override([BlockCost(cost_amount=0, cost_type=BlockCostType.TOKENS)])
    block = SearchTheWebBlock()
    # Ollama models aren't in TOKEN_COST but are in MODEL_COST.
    ollama_model = LlmModel.OLLAMA_LLAMA3_2
    expected = MODEL_COST[ollama_model]
    cost, _ = block_usage_cost(
        block,
        {"model": ollama_model.value},
        stats=NodeExecutionStats(input_token_count=10_000, output_token_count=10_000),
    )
    assert cost == expected


def test_run_cost_type_remains_unchanged(tmp_block_costs_override):
    tmp_block_costs_override([BlockCost(cost_amount=7, cost_type=BlockCostType.RUN)])
    block = SearchTheWebBlock()
    # Stats shouldn't affect RUN charge.
    stats = NodeExecutionStats(walltime=999, input_token_count=999)
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 7
