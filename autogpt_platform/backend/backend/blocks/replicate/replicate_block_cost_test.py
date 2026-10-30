"""What ReplicateModelBlock charges for a run.

Official models are charged their published Replicate price (``_pricing.py``),
community models their run time. Runs go through the real SDK against
``conftest.FakeReplicate``, and the charge is read with the executor's own
``block_usage_cost``, as it is billed.
"""

import pytest

from backend.blocks._base import BlockCostType
from backend.blocks.replicate._auth import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.replicate._pricing import OFFICIAL_MODEL_PRICES, _units
from backend.blocks.replicate.replicate_block import (
    _REPLICATE_USD_PER_SEC,
    ReplicateModelBlock,
)
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.execution import ExecutionContext
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import replicate_credentials
from backend.util.exceptions import BlockExecutionError, BlockInputError

COMMUNITY_MODEL = "cuuupid/glm-4v-9b"
LATEST_VERSION = "69196a53d2a33ec4b1e4e3e8ee32d2e1a3a3f8e6e5d4c3b2a1f0e9d8c7b6a5f4"


def test_registered_as_cost_usd_150():
    entries = BLOCK_COSTS[ReplicateModelBlock]
    assert len(entries) == 1
    assert entries[0].cost_type == BlockCostType.COST_USD
    assert entries[0].cost_amount == 150


def test_hardware_rate_constant_in_range():
    """The flat run-time rate covers single-GPU hardware up to A100-80GB
    ($0.001875/s) without over-billing cheap hardware by an order of magnitude."""
    assert 0.001875 <= _REPLICATE_USD_PER_SEC <= 0.0025


@pytest.mark.asyncio
async def test_official_model_is_charged_its_replicate_price(fake_replicate):
    """A 4 s Veo 3.1 clip without audio costs $0.80 on Replicate, whatever the
    run time; the platform charges it at 150 credits per dollar."""
    fake_replicate.add_model("google/veo-3.1", official=True)
    fake_replicate.metrics = {
        "predict_time": 40.0,
        "billing.video_output_duration_seconds": 4,
    }

    block = await _run(
        fake_replicate, "google/veo-3.1", {"duration": 4, "generate_audio": False}
    )

    assert block.execution_stats.provider_cost == pytest.approx(0.80)
    assert _charged_credits(block) == 120


@pytest.mark.asyncio
async def test_community_model_is_charged_by_run_time(fake_replicate):
    fake_replicate.add_model(
        COMMUNITY_MODEL, official=False, latest_version=LATEST_VERSION
    )
    fake_replicate.metrics = {"predict_time": 10.0}

    block = await _run(fake_replicate, COMMUNITY_MODEL, {"prompt": "hi"})

    assert block.execution_stats.provider_cost == pytest.approx(
        10.0 * _REPLICATE_USD_PER_SEC
    )
    assert _charged_credits(block) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model, model_inputs, metrics, output, usd",
    [
        # The quantity the run reports wins over the inputs.
        pytest.param(
            "google/veo-3.1",
            {"duration": 8, "generate_audio": False},
            {"billing.video_output_duration_seconds": 4},
            "https://x/v.mp4",
            0.80,
            id="reported-seconds",
        ),
        pytest.param(
            "google/veo-3.1",
            {"duration": 6, "generate_audio": False},
            {},
            "https://x/v.mp4",
            1.20,
            id="seconds-from-inputs",
        ),
        pytest.param(
            "google/veo-3.1",
            {},
            {},
            "https://x/v.mp4",
            3.20,
            id="model-defaults",
        ),
        pytest.param(
            "prunaai/p-video",
            {"resolution": "1080p", "draft": True, "duration": 4},
            {},
            "https://x/v.mp4",
            0.04,
            id="matched-tier",
        ),
        pytest.param(
            "prunaai/p-video",
            {"resolution": "4k", "duration": 4},
            {},
            "https://x/v.mp4",
            0.16,
            id="unknown-tier-is-the-dearest",
        ),
        pytest.param(
            "prunaai/p-video-avatar",
            {"resolution": "720p"},
            {},
            "https://x/v.mp4",
            1.50,
            id="unreported-length-is-the-cap",
        ),
        pytest.param(
            "black-forest-labs/flux-schnell",
            {"num_outputs": 4},
            {},
            [
                "https://x/1.png",
                "https://x/2.png",
                "https://x/3.png",
                "https://x/4.png",
            ],
            0.012,
            id="images-counted-from-outputs",
        ),
        pytest.param(
            "meta/meta-llama-3-8b-instruct",
            {"prompt": "hi"},
            {"input_token_count": 1000, "output_token_count": 2000},
            "hello",
            1000 * 0.05e-6 + 2000 * 0.25e-6,
            id="reported-tokens",
        ),
        pytest.param(
            "minimax/speech-2.6-hd",
            {"text": "x" * 500},
            {"predict_time": 3.0},
            "https://x/a.mp3",
            0.05,
            id="speech-text-from-inputs",
        ),
    ],
)
async def test_official_run_price(
    fake_replicate, model, model_inputs, metrics, output, usd
):
    fake_replicate.add_model(model, official=True)
    fake_replicate.metrics = metrics
    fake_replicate.output = output

    block = await _run(fake_replicate, model, model_inputs)

    assert block.execution_stats.provider_cost == pytest.approx(usd)


@pytest.mark.asyncio
async def test_official_model_without_a_price_is_refused_before_it_runs(
    fake_replicate,
):
    fake_replicate.add_model(
        "acme/new-video-model", official=True, latest_version=LATEST_VERSION
    )

    with pytest.raises(BlockInputError, match="official model"):
        await _run(fake_replicate, "acme/new-video-model", {"prompt": "hi"})
    with pytest.raises(BlockInputError, match="official model"):
        await _run(
            fake_replicate,
            "acme/new-video-model",
            {"prompt": "hi"},
            version=LATEST_VERSION,
        )

    assert fake_replicate.predictions_created() == []


def test_every_price_ends_on_its_dearest_unconditional_rate():
    for model, prices in OFFICIAL_MODEL_PRICES.items():
        for price in prices:
            last = price.rates[-1]
            assert last.when == {}, model
            assert last.usd == max(rate.usd for rate in price.rates), model


def test_every_price_has_a_quantity_when_the_run_reports_none():
    for model, prices in OFFICIAL_MODEL_PRICES.items():
        for price in prices:
            units = _units(price, {"prompt": "hi", "text": "hi"}, {}, ["out"])
            assert units > 0, (model, price.metric)


@pytest.mark.asyncio
async def test_community_run_without_metrics_is_not_billed(fake_replicate):
    """Without metrics there is no run time to bill; the pre-flight balance
    guard still applies."""
    fake_replicate.add_model(
        COMMUNITY_MODEL, official=False, latest_version=LATEST_VERSION
    )
    fake_replicate.metrics = None

    block = await _run(fake_replicate, COMMUNITY_MODEL, {})

    assert block.execution_stats.provider_cost is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, error, match",
    [("failed", "CUDA OOM", "CUDA OOM"), ("canceled", None, "canceled")],
)
async def test_failed_or_canceled_run_is_not_billed(
    fake_replicate, status, error, match
):
    fake_replicate.add_model("google/veo-3.1", official=True)
    fake_replicate.status = status
    fake_replicate.error = error
    fake_replicate.output = None
    fake_replicate.metrics = {"billing.video_output_duration_seconds": 4}

    block = ReplicateModelBlock()
    with pytest.raises(BlockExecutionError, match=match):
        await _run(fake_replicate, "google/veo-3.1", {}, block=block)

    assert block.execution_stats.provider_cost is None


async def _run(
    fake_replicate,
    model_name: str,
    model_inputs: dict,
    *,
    version: str | None = None,
    block: ReplicateModelBlock | None = None,
) -> ReplicateModelBlock:
    block = block or ReplicateModelBlock()
    input_data = ReplicateModelBlock.Input(
        **{
            "credentials": TEST_CREDENTIALS_INPUT,
            "model_name": model_name,
            "model_inputs": model_inputs,
            "version": version,
        }
    )
    async for _ in block.run(
        input_data,
        credentials=TEST_CREDENTIALS,
        execution_context=ExecutionContext(user_id="user-1", graph_exec_id="exec-1"),
    ):
        pass
    return block


def _charged_credits(block: ReplicateModelBlock) -> int:
    platform_key = {
        "id": replicate_credentials.id,
        "provider": replicate_credentials.provider,
        "type": replicate_credentials.type,
    }
    credits, _ = block_usage_cost(
        block, {"credentials": platform_key}, stats=block.execution_stats
    )
    return credits
