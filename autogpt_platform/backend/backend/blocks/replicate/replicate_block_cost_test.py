"""Unit tests for ReplicateModelBlock's predictions.async_create billing path.

Verifies the refactored run_model correctly:
1. Uses predictions.async_create (version= vs model= based on ":" in model_ref)
2. Awaits async_wait() for metrics to be populated
3. Reads prediction.metrics["predict_time"] and emits provider_cost/cost_usd
4. Returns extract_result(prediction.output) with the same shape as the old
   async_run path
5. Gracefully skips merge_stats when metrics is missing (protects against a
   silent wallet-free leak on SDK quirks)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks._base import BlockCostType
from backend.blocks.replicate.replicate_block import (
    _REPLICATE_USD_PER_SEC,
    ReplicateModelBlock,
)
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.model import NodeExecutionStats


def test_registered_as_cost_usd_150():
    entries = BLOCK_COSTS[ReplicateModelBlock]
    assert len(entries) == 1
    assert entries[0].cost_type == BlockCostType.COST_USD
    assert entries[0].cost_amount == 150


def test_hardware_rate_constant_in_range():
    """$0.0014/s is Nvidia L40S tier. Sanity-check we haven't accidentally
    shipped a rate that's off by an order of magnitude (e.g. $0.014 would
    10x over-bill every run).
    """
    # Replicate's public hardware tiers: L4 $0.000275, A10G $0.000575,
    # L40S $0.000975, A100 $0.001400, A100-80GB $0.001725. L40S @
    # $0.0014/s covers most popular models with mild over-bill margin.
    assert 0.0005 <= _REPLICATE_USD_PER_SEC <= 0.002


def _make_fake_prediction(output, predict_time=None, status="succeeded", error=None):
    """Build a stand-in for replicate's Prediction with the attrs we touch."""
    pred = MagicMock()
    pred.output = output
    pred.status = status
    pred.error = error
    pred.metrics = {"predict_time": predict_time} if predict_time is not None else None
    pred.async_wait = AsyncMock(return_value=None)
    return pred


@pytest.mark.asyncio
async def test_run_model_uses_version_keyword_when_ref_has_colon():
    """`"owner/name:sha"` → predictions.async_create(version=sha, ...)."""
    block = ReplicateModelBlock()
    prediction = _make_fake_prediction(output="hello", predict_time=3.2)

    client = MagicMock()
    client.predictions.async_create = AsyncMock(return_value=prediction)

    with patch(
        "backend.blocks.replicate.replicate_block.ReplicateClient",
        return_value=client,
    ):
        await block.run_model(
            "owner/model:abc123", {"prompt": "hi"}, SecretStr("fake-key")
        )

    client.predictions.async_create.assert_awaited_once_with(
        version="abc123", input={"prompt": "hi"}
    )


@pytest.mark.asyncio
async def test_run_model_uses_model_keyword_when_ref_is_unpinned():
    """`"owner/name"` (no `:version`) → predictions.async_create(model=ref, ...)."""
    block = ReplicateModelBlock()
    prediction = _make_fake_prediction(output="hello", predict_time=1.0)

    client = MagicMock()
    client.predictions.async_create = AsyncMock(return_value=prediction)

    with patch(
        "backend.blocks.replicate.replicate_block.ReplicateClient",
        return_value=client,
    ):
        await block.run_model(
            "owner/flux-schnell", {"prompt": "cat"}, SecretStr("fake-key")
        )

    client.predictions.async_create.assert_awaited_once_with(
        model="owner/flux-schnell", input={"prompt": "cat"}
    )


@pytest.mark.asyncio
async def test_run_model_emits_provider_cost_from_predict_time():
    """Core contract: provider_cost = predict_time * $0.0014/s, cost_usd."""
    block = ReplicateModelBlock()
    # 5-second run → 5 * 0.0014 = $0.007 → 150 cr/$ * 0.007 ceil = 2 cr
    prediction = _make_fake_prediction(output="result-data", predict_time=5.0)

    client = MagicMock()
    client.predictions.async_create = AsyncMock(return_value=prediction)

    captured: list[NodeExecutionStats] = []
    with (
        patch(
            "backend.blocks.replicate.replicate_block.ReplicateClient",
            return_value=client,
        ),
        patch.object(block, "merge_stats", side_effect=captured.append),
    ):
        result = await block.run_model("owner/model", {}, SecretStr("fake-key"))

    assert len(captured) == 1
    stats = captured[0]
    assert stats.provider_cost == pytest.approx(5.0 * _REPLICATE_USD_PER_SEC)
    assert stats.provider_cost_type == "cost_usd"
    assert result == "result-data"
    # async_wait MUST be called before reading metrics — otherwise metrics
    # is None on in-flight predictions.
    prediction.async_wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_model_skips_merge_stats_when_metrics_missing():
    """Protect against the nightmare scenario: if the SDK stops populating
    metrics (or we hit a prediction that completes without metrics),
    merge_stats must NOT fire. Otherwise we'd emit a zero provider_cost
    that the resolver treats as 0 credits — a silent wallet-free leak.
    The block's run() path relies on the flat 0 fallback via
    charge_reconciled_usage's pre-flight balance guard.
    """
    block = ReplicateModelBlock()
    prediction = _make_fake_prediction(output="x", predict_time=None)

    client = MagicMock()
    client.predictions.async_create = AsyncMock(return_value=prediction)

    captured: list[NodeExecutionStats] = []
    with (
        patch(
            "backend.blocks.replicate.replicate_block.ReplicateClient",
            return_value=client,
        ),
        patch.object(block, "merge_stats", side_effect=captured.append),
    ):
        await block.run_model("owner/model", {}, SecretStr("fake-key"))

    # No merge_stats call → no provider_cost emission → COST_USD resolver
    # returns 0 → run is effectively free post-flight, but pre-flight
    # balance guard still blocks zero-balance wallets per PR #12894.
    assert captured == []


@pytest.mark.asyncio
async def test_run_model_skips_merge_when_predict_time_is_zero():
    """A 0-second predict_time would emit provider_cost=0, which is useless
    telemetry. Treat 0 same as missing (no emission)."""
    block = ReplicateModelBlock()
    prediction = _make_fake_prediction(output="x", predict_time=0)

    client = MagicMock()
    client.predictions.async_create = AsyncMock(return_value=prediction)

    captured: list[NodeExecutionStats] = []
    with (
        patch(
            "backend.blocks.replicate.replicate_block.ReplicateClient",
            return_value=client,
        ),
        patch.object(block, "merge_stats", side_effect=captured.append),
    ):
        await block.run_model("owner/model", {}, SecretStr("fake-key"))

    assert captured == []


@pytest.mark.asyncio
async def test_run_model_raises_on_failed_status_and_does_not_bill():
    """async_wait returns normally on 'failed' — without an explicit status
    check we'd bill partial compute time AND yield 'status: succeeded' with
    empty output. Verify we raise BEFORE merge_stats so the failed run is
    not billed."""
    block = ReplicateModelBlock()
    prediction = _make_fake_prediction(
        output=None, predict_time=2.5, status="failed", error="CUDA OOM"
    )

    client = MagicMock()
    client.predictions.async_create = AsyncMock(return_value=prediction)

    captured: list[NodeExecutionStats] = []
    with (
        patch(
            "backend.blocks.replicate.replicate_block.ReplicateClient",
            return_value=client,
        ),
        patch.object(block, "merge_stats", side_effect=captured.append),
    ):
        with pytest.raises(RuntimeError, match="CUDA OOM"):
            await block.run_model("owner/model", {}, SecretStr("fake-key"))

    assert captured == []


@pytest.mark.asyncio
async def test_run_model_raises_on_canceled_status_and_does_not_bill():
    """Canceled predictions — same guarantees as failed: don't bill, surface
    the cancellation."""
    block = ReplicateModelBlock()
    prediction = _make_fake_prediction(output=None, predict_time=1.0, status="canceled")

    client = MagicMock()
    client.predictions.async_create = AsyncMock(return_value=prediction)

    captured: list[NodeExecutionStats] = []
    with (
        patch(
            "backend.blocks.replicate.replicate_block.ReplicateClient",
            return_value=client,
        ),
        patch.object(block, "merge_stats", side_effect=captured.append),
    ):
        with pytest.raises(RuntimeError, match="canceled"):
            await block.run_model("owner/model", {}, SecretStr("fake-key"))

    assert captured == []
