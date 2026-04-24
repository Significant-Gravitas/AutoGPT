"""Coverage tests for the cost-leak fixes in this PR.

Each block's ``run()`` / helper emits provider_cost + cost_usd (or items)
via merge_stats so the post-flight resolver bills real provider spend.
Tests here drive that emission path directly so a regression on any one
block surfaces immediately.
"""

from unittest.mock import patch

import pytest
from pydantic import SecretStr

from backend.blocks._base import BlockCostType
from backend.blocks.ai_condition import AIConditionBlock
from backend.data.block_cost_config import BLOCK_COSTS, LLM_COST
from backend.data.model import APIKeyCredentials, NodeExecutionStats

# -------- AIConditionBlock registration --------


def test_ai_condition_registered_under_llm_cost():
    """AIConditionBlock was running wallet-free before this PR; verify it
    now resolves through the same per-model LLM_COST table as every other
    LLM block.
    """
    assert BLOCK_COSTS[AIConditionBlock] is LLM_COST


# -------- Pinecone insert ITEMS emission --------


@pytest.mark.asyncio
async def test_pinecone_insert_emits_items_provider_cost():
    from backend.blocks.pinecone import PineconeInsertBlock

    block = PineconeInsertBlock()
    captured: list[NodeExecutionStats] = []

    class _FakeIndex:
        def upsert(self, **_):
            return None

    class _FakePinecone:
        def __init__(self, *_, **__):
            pass

        def Index(self, _name):
            return _FakeIndex()

    with (
        patch("backend.blocks.pinecone.Pinecone", _FakePinecone),
        patch.object(block, "merge_stats", side_effect=captured.append),
    ):
        input_data = block.input_schema(
            credentials={
                "id": "00000000-0000-0000-0000-000000000000",
                "provider": "pinecone",
                "type": "api_key",
            },
            index="my-index",
            chunks=["alpha", "beta", "gamma"],
            embeddings=[[0.1] * 4, [0.2] * 4, [0.3] * 4],
            namespace="",
            metadata={},
        )

        creds = APIKeyCredentials(
            id="00000000-0000-0000-0000-000000000000",
            provider="pinecone",
            title="mock",
            api_key=SecretStr("mock-key"),
            expires_at=None,
        )
        outputs = [(n, v) async for n, v in block.run(input_data, credentials=creds)]

    assert any(name == "upsert_response" for name, _ in outputs)
    assert len(captured) == 1
    stats = captured[0]
    assert stats.provider_cost == pytest.approx(3.0)
    assert stats.provider_cost_type == "items"


# -------- Narration model-aware per-char rate --------


@pytest.mark.parametrize(
    "model_id, expected_rate_per_char",
    [
        ("eleven_flash_v2_5", 0.000167 * 0.5),
        ("eleven_turbo_v2_5", 0.000167 * 0.5),
        ("eleven_multilingual_v2", 0.000167 * 1.0),
        ("eleven_turbo_v2", 0.000167 * 1.0),
    ],
)
def test_narration_per_char_rate_scales_with_model(model_id, expected_rate_per_char):
    """Drive VideoNarrationBlock._record_script_cost directly so a regression
    that drops the model-aware branching (e.g. hardcoding 1.0 cr/char for
    all models) makes this test fail.
    """
    from backend.blocks.video.narration import VideoNarrationBlock

    block = VideoNarrationBlock()
    captured: list[NodeExecutionStats] = []
    with patch.object(block, "merge_stats", side_effect=captured.append):
        block._record_script_cost("x" * 5000, model_id)

    assert len(captured) == 1
    stats = captured[0]
    assert stats.provider_cost == pytest.approx(5000 * expected_rate_per_char)
    assert stats.provider_cost_type == "cost_usd"


# -------- Perplexity None-guard on x-total-cost --------


@pytest.mark.parametrize(
    "openrouter_cost, expect_type",
    [
        (0.0421, "cost_usd"),  # concrete positive USD → tagged
        (None, None),  # header missing → no tag (keeps gap observable)
        (0.0, None),  # zero → no tag (wouldn't bill anything anyway)
    ],
)
def test_perplexity_record_openrouter_cost_tags_only_on_concrete_value(
    openrouter_cost, expect_type
):
    """Drive PerplexityBlock._record_openrouter_cost directly to verify the
    None/0 guard. A regression that tags cost_usd unconditionally would
    silently floor the user's bill to 0 via the resolver — this test
    would catch it.
    """
    from backend.blocks.perplexity import PerplexityBlock

    block = PerplexityBlock()
    with patch(
        "backend.blocks.perplexity.extract_openrouter_cost",
        return_value=openrouter_cost,
    ):
        block._record_openrouter_cost(response=object())

    assert block.execution_stats.provider_cost == openrouter_cost
    assert block.execution_stats.provider_cost_type == expect_type


# -------- Codex COST_USD registration --------


def test_codex_registered_as_cost_usd_150():
    from backend.blocks.codex import CodeGenerationBlock

    entries = BLOCK_COSTS[CodeGenerationBlock]
    assert len(entries) == 1
    entry = entries[0]
    assert entry.cost_type == BlockCostType.COST_USD
    assert entry.cost_amount == 150


@pytest.mark.parametrize(
    "input_tokens, output_tokens, expected_usd",
    [
        # GPT-5.1-Codex: $1.25 / 1M input, $10 / 1M output.
        (1_000_000, 0, 1.25),
        (0, 1_000_000, 10.0),
        (100_000, 10_000, 0.225),  # 0.125 + 0.100
        (0, 0, 0.0),
    ],
)
def test_codex_computes_provider_cost_usd_from_token_counts(
    input_tokens, output_tokens, expected_usd
):
    """Drive CodeGenerationBlock._compute_token_usd directly. A regression
    to the wrong rate constants (e.g. swapping the $1.25 input rate for
    GPT-4o's $2.50) would fail this test.
    """
    from backend.blocks.codex import CodeGenerationBlock

    assert CodeGenerationBlock._compute_token_usd(
        input_tokens, output_tokens
    ) == pytest.approx(expected_usd)


# -------- ClaudeCode COST_USD registration sanity (already tested in claude_code_cost_test.py) --------


# -------- Perplexity COST_USD registration for all 3 tiers --------


def test_perplexity_sonar_all_tiers_registered_as_cost_usd_150():
    from backend.blocks.perplexity import PerplexityBlock

    entries = BLOCK_COSTS[PerplexityBlock]
    # 3 tiers (SONAR, SONAR_PRO, SONAR_DEEP_RESEARCH) all COST_USD 150.
    assert len(entries) == 3
    for entry in entries:
        assert entry.cost_type == BlockCostType.COST_USD
        assert entry.cost_amount == 150


# -------- Narration COST_USD registration --------


def test_narration_registered_as_cost_usd_150():
    from backend.blocks.video.narration import VideoNarrationBlock

    entries = BLOCK_COSTS[VideoNarrationBlock]
    assert len(entries) == 1
    assert entries[0].cost_type == BlockCostType.COST_USD
    assert entries[0].cost_amount == 150


# -------- Pinecone registrations --------


def test_pinecone_registrations():
    from backend.blocks.pinecone import (
        PineconeInitBlock,
        PineconeInsertBlock,
        PineconeQueryBlock,
    )

    assert BLOCK_COSTS[PineconeInitBlock][0].cost_type == BlockCostType.RUN
    assert BLOCK_COSTS[PineconeQueryBlock][0].cost_type == BlockCostType.RUN
    # Insert scales with item count.
    assert BLOCK_COSTS[PineconeInsertBlock][0].cost_type == BlockCostType.ITEMS
    assert BLOCK_COSTS[PineconeInsertBlock][0].cost_amount == 1
