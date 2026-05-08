"""Tests for the dynamic-pricing branches of block_usage_cost."""

import math

import pytest

from backend.blocks._base import BlockCost, BlockCostType
from backend.blocks.code_executor import (
    ExecuteCodeBlock,
    ExecuteCodeStepBlock,
    InstantiateCodeSandboxBlock,
)
from backend.blocks.exa.search import ExaSearchBlock
from backend.blocks.fal.ai_video_generator import AIVideoGeneratorBlock
from backend.blocks.jina.search import SearchTheWebBlock
from backend.blocks.llm import AITextGeneratorBlock, LlmModel
from backend.data.block_cost_config import (
    BLOCK_COSTS,
    MODEL_COST,
    TOKEN_COST,
    TokenRate,
)
from backend.data.model import NodeExecutionStats
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import (
    anthropic_credentials,
    e2b_credentials,
    exa_credentials,
    fal_credentials,
    openai_credentials,
)


@pytest.fixture
def tmp_block_costs_override(monkeypatch):
    """Swap out BLOCK_COSTS[SearchTheWebBlock] for the duration of a test."""
    return lambda costs: monkeypatch.setitem(BLOCK_COSTS, SearchTheWebBlock, costs)


def test_second_cost_type_uses_stats_walltime_with_divisor(tmp_block_costs_override):
    tmp_block_costs_override(
        [BlockCost(cost_amount=1, cost_type=BlockCostType.SECOND, cost_divisor=10)]
    )
    block = SearchTheWebBlock()
    # 25 seconds of walltime / 10 sec-per-credit = ceil(2.5) = 3 credits.
    stats = NodeExecutionStats(walltime=25.0)
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 3


def test_second_cost_type_sub_divisor_bills_one_credit(tmp_block_costs_override):
    """Sub-divisor walltime still bills 1cr — no 0-credit leak."""
    tmp_block_costs_override(
        [BlockCost(cost_amount=1, cost_type=BlockCostType.SECOND, cost_divisor=3)]
    )
    block = SearchTheWebBlock()
    # 1s walltime on a 1cr/3s block → ceil(1/3) * 1 = 1 credit.
    stats = NodeExecutionStats(walltime=1.0)
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 1


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
    # 45 items / 10 = ceil(4.5) = 5 credits.
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 5


def test_items_cost_type_sub_divisor_bills_one_credit(tmp_block_costs_override):
    """A single item under cost_divisor=2 still bills 1cr — no 0-credit leak."""
    tmp_block_costs_override(
        [BlockCost(cost_amount=1, cost_type=BlockCostType.ITEMS, cost_divisor=2)]
    )
    block = SearchTheWebBlock()
    # Apollo SearchOrganizationsBlock shape: 1 result returned on a 1cr/2-item
    # block should bill ceil(1/2) * 1 = 1 credit (floor division would bill 0).
    stats = NodeExecutionStats(provider_cost=1, provider_cost_type="items")
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 1


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


def test_cost_usd_ignores_non_usd_provider_cost(tmp_block_costs_override):
    """provider_cost_type='items' should not be mistaken for dollars."""
    tmp_block_costs_override(
        [BlockCost(cost_amount=100, cost_type=BlockCostType.COST_USD)]
    )
    block = SearchTheWebBlock()
    # An items-typed provider_cost of 45 would otherwise be billed as $45.
    stats = NodeExecutionStats(provider_cost=45, provider_cost_type="items")
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 0


def test_tokens_cost_type_uses_token_rate_table(tmp_block_costs_override, monkeypatch):
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


def test_e2b_sandbox_blocks_bill_one_credit_per_ten_seconds():
    """End-to-end: E2B blocks use the real SECOND/divisor=10 BlockCost entry."""
    creds = {
        "credentials": {
            "id": e2b_credentials.id,
            "provider": e2b_credentials.provider,
            "type": e2b_credentials.type,
        }
    }
    for block_cls in (
        ExecuteCodeBlock,
        InstantiateCodeSandboxBlock,
        ExecuteCodeStepBlock,
    ):
        # 45s walltime → ceil(45/10) = 5 credits.
        stats = NodeExecutionStats(walltime=45.0)
        cost, _ = block_usage_cost(block_cls(), creds, stats=stats)
        assert cost == 5, f"{block_cls.__name__} expected 5 credits, got {cost}"
        # Pre-flight (no stats) → 0.
        cost, _ = block_usage_cost(block_cls(), creds)
        assert cost == 0


def test_fal_video_block_bills_fifteen_credits_per_second():
    block = AIVideoGeneratorBlock()
    creds = {
        "credentials": {
            "id": fal_credentials.id,
            "provider": fal_credentials.provider,
            "type": fal_credentials.type,
        }
    }
    # 8s clip → 15 * 8 = 120 credits.
    cost, _ = block_usage_cost(block, creds, stats=NodeExecutionStats(walltime=8.0))
    assert cost == 120
    # Pre-flight → 0.
    cost, _ = block_usage_cost(block, creds)
    assert cost == 0


def test_exa_blocks_bill_cost_usd_via_sdk_config():
    """End-to-end: Exa's ProviderBuilder.with_base_cost(100, COST_USD) is live."""
    block = ExaSearchBlock()
    creds = {
        "credentials": {
            "id": exa_credentials.id,
            "provider": exa_credentials.provider,
            "type": exa_credentials.type,
        }
    }
    # $0.05 provider spend * 100 credits/USD = 5 credits.
    stats = NodeExecutionStats(provider_cost=0.05, provider_cost_type="cost_usd")
    cost, _ = block_usage_cost(block, creds, stats=stats)
    assert cost == 5
    # Pre-flight: unknown cost → 0.
    cost, _ = block_usage_cost(block, creds)
    assert cost == 0


def test_llm_block_charges_per_token_post_flight():
    """AITextGeneratorBlock with Claude 4.6 Sonnet bills by real token counts."""
    block = AITextGeneratorBlock()
    input_data = {
        "model": LlmModel.CLAUDE_4_6_SONNET,
        "credentials": {
            "id": anthropic_credentials.id,
            "provider": anthropic_credentials.provider,
            "type": anthropic_credentials.type,
        },
    }
    rate = TOKEN_COST[LlmModel.CLAUDE_4_6_SONNET]
    stats = NodeExecutionStats(
        input_token_count=200_000,
        output_token_count=50_000,
        cache_read_token_count=100_000,
    )
    expected = math.ceil(
        (200_000 * rate.input + 50_000 * rate.output + 100_000 * rate.cache_read)
        / 1_000_000
    )
    cost, _ = block_usage_cost(block, input_data, stats=stats)
    assert cost == expected


def test_llm_block_pre_flight_falls_back_to_model_cost():
    """Pre-flight charge of an LLM block uses the flat MODEL_COST floor."""
    block = AITextGeneratorBlock()
    cost, _ = block_usage_cost(
        block,
        {
            "model": LlmModel.GPT5,
            "credentials": {
                "id": openai_credentials.id,
                "provider": openai_credentials.provider,
                "type": openai_credentials.type,
            },
        },
    )
    assert cost == MODEL_COST[LlmModel.GPT5]


def test_run_cost_type_remains_unchanged(tmp_block_costs_override):
    tmp_block_costs_override([BlockCost(cost_amount=7, cost_type=BlockCostType.RUN)])
    block = SearchTheWebBlock()
    # Stats shouldn't affect RUN charge.
    stats = NodeExecutionStats(walltime=999, input_token_count=999)
    cost, _ = block_usage_cost(block, {}, stats=stats)
    assert cost == 7
