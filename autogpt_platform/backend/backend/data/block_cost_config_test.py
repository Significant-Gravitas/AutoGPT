import pytest

from backend.blocks._base import BlockCostType
from backend.blocks.ayrshare.post_to_bluesky import PostToBlueskyBlock
from backend.blocks.ayrshare.post_to_facebook import PostToFacebookBlock
from backend.blocks.ayrshare.post_to_gmb import PostToGMBBlock
from backend.blocks.ayrshare.post_to_instagram import PostToInstagramBlock
from backend.blocks.ayrshare.post_to_linkedin import PostToLinkedInBlock
from backend.blocks.ayrshare.post_to_pinterest import PostToPinterestBlock
from backend.blocks.ayrshare.post_to_reddit import PostToRedditBlock
from backend.blocks.ayrshare.post_to_snapchat import PostToSnapchatBlock
from backend.blocks.ayrshare.post_to_telegram import PostToTelegramBlock
from backend.blocks.ayrshare.post_to_threads import PostToThreadsBlock
from backend.blocks.ayrshare.post_to_tiktok import PostToTikTokBlock
from backend.blocks.ayrshare.post_to_x import PostToXBlock
from backend.blocks.ayrshare.post_to_youtube import PostToYouTubeBlock
from backend.blocks.bannerbear.text_overlay import BannerbearTextOverlayBlock
from backend.blocks.code_executor import (
    ExecuteCodeBlock,
    ExecuteCodeStepBlock,
    InstantiateCodeSandboxBlock,
)
from backend.blocks.fal.ai_video_generator import AIVideoGeneratorBlock
from backend.blocks.jina.chunking import JinaChunkingBlock
from backend.blocks.llm import AITextGeneratorBlock, LLMModel
from backend.blocks.youtube import TranscribeYoutubeVideoBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.data.model import NodeExecutionStats
from backend.executor import utils as executor_utils
from backend.executor.utils import block_usage_cost
from backend.integrations.credentials_store import (
    e2b_credentials,
    fal_credentials,
    jina_credentials,
    open_router_credentials,
    webshare_proxy_credentials,
)


@pytest.fixture(autouse=True)
def _stub_preflight_estimate(monkeypatch):
    """Force `get_preflight_estimate` to return 0 in this module's tests so
    the dynamic-cost (E2B SECOND / FAL SECOND) pre-flight assertions don't
    couple to a populated `block_preflight_estimates.json` once the seeding
    PR registers non-zero estimates for these blocks."""
    monkeypatch.setattr(executor_utils, "get_preflight_estimate", lambda _bid: 0)


ALL_AYRSHARE_BLOCKS = [
    PostToBlueskyBlock,
    PostToFacebookBlock,
    PostToGMBBlock,
    PostToInstagramBlock,
    PostToLinkedInBlock,
    PostToPinterestBlock,
    PostToRedditBlock,
    PostToSnapchatBlock,
    PostToTelegramBlock,
    PostToThreadsBlock,
    PostToTikTokBlock,
    PostToXBlock,
    PostToYouTubeBlock,
]

# YouTube and Snapchat are video-only platforms, so their Input overrides
# is_video default to True; the @cost filter should pick the 5-credit tier.
AYRSHARE_VIDEO_ONLY_BLOCKS = [PostToYouTubeBlock, PostToSnapchatBlock]


@pytest.mark.parametrize("block_class", ALL_AYRSHARE_BLOCKS)
def test_ayrshare_block_has_video_and_default_tier(block_class):
    costs = BLOCK_COSTS.get(block_class)
    assert costs is not None and len(costs) == 2
    amounts = {c.cost_amount for c in costs}
    assert amounts == {2, 5}


def test_ayrshare_video_post_charges_video_tier():
    block = PostToXBlock()
    cost, _ = block_usage_cost(block, {"is_video": True})
    assert cost == 5


def test_ayrshare_non_video_post_charges_default_tier():
    block = PostToXBlock()
    cost, _ = block_usage_cost(block, {"is_video": False})
    assert cost == 2


def test_ayrshare_default_is_video_false_still_matches_default_tier():
    block = PostToXBlock()
    cost, _ = block_usage_cost(block, {})
    assert cost == 2


@pytest.mark.parametrize("block_class", AYRSHARE_VIDEO_ONLY_BLOCKS)
def test_ayrshare_video_only_block_defaults_to_video_tier(block_class):
    # Video-only platforms override is_video default to True so billing matches
    # the is_video=True passed into client.create_post.
    block = block_class()
    default_is_video = block.input_schema.model_fields["is_video"].default
    assert default_is_video is True
    cost, _ = block_usage_cost(block, {"is_video": default_is_video})
    assert cost == 5


def test_jina_chunking_has_flat_cost_floor():
    block = JinaChunkingBlock()
    cost, _ = block_usage_cost(
        block,
        {
            "credentials": {
                "id": jina_credentials.id,
                "provider": jina_credentials.provider,
                "type": jina_credentials.type,
            }
        },
    )
    assert cost == 1


def test_bannerbear_base_cost_is_three_credits():
    # Bannerbear is registered via the SDK ProviderBuilder with base_cost=3.
    block = BannerbearTextOverlayBlock()
    cost, _ = block_usage_cost(block, {})
    assert cost == 3


def test_e2b_sandbox_blocks_bill_per_walltime_second():
    """E2B uses SECOND cost_type with cost_divisor=10 (1 credit per 10s).

    Pre-flight (no stats) returns 0 — walltime unknown until the block runs.
    Post-flight bills at the real walltime via charge_reconciled_usage.
    """
    from backend.data.model import NodeExecutionStats

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
        # Pre-flight: unknown walltime ⇒ 0 credits (no gating on future cost).
        cost, _ = block_usage_cost(block_cls(), creds)
        assert cost == 0, f"{block_cls.__name__} pre-flight must be 0, got {cost}"
        # Post-flight: 25s ⇒ ceil(25/10) = 3 credits.
        stats = NodeExecutionStats(walltime=25.0)
        cost, _ = block_usage_cost(block_cls(), creds, stats=stats)
        assert cost == 3, f"{block_cls.__name__} @ 25s must be 3 credits, got {cost}"


def test_fal_video_generator_bills_per_walltime_second():
    """FAL AIVideoGeneratorBlock uses SECOND with cost_amount=15 (15 credits/s)."""
    from backend.data.model import NodeExecutionStats

    creds = {
        "credentials": {
            "id": fal_credentials.id,
            "provider": fal_credentials.provider,
            "type": fal_credentials.type,
        }
    }
    # Pre-flight: unknown walltime ⇒ 0 credits.
    cost, _ = block_usage_cost(AIVideoGeneratorBlock(), creds)
    assert cost == 0
    # Post-flight: 5s clip ⇒ 15 * 5 = 75 credits.
    cost, _ = block_usage_cost(
        AIVideoGeneratorBlock(), creds, stats=NodeExecutionStats(walltime=5.0)
    )
    assert cost == 75


def test_transcribe_youtube_has_one_credit_tooling_floor():
    cost, _ = block_usage_cost(
        TranscribeYoutubeVideoBlock(),
        {
            "credentials": {
                "id": webshare_proxy_credentials.id,
                "provider": webshare_proxy_credentials.provider,
                "type": webshare_proxy_credentials.type,
            }
        },
    )
    assert cost == 1


# -------- SECRT-2701: OpenRouter display rates are not billing rates --------


def _open_router_input(model: str) -> dict:
    return {
        "model": model,
        "credentials": {
            "id": open_router_credentials.id,
            "provider": open_router_credentials.provider,
            "type": open_router_credentials.type,
        },
    }


def test_open_router_bills_cost_usd_not_the_displayed_token_rate():
    """open_router models settle against the provider's own x-total-cost, so
    the catalog's per-1M credit rates only ever reach the builder's price
    label. Pinning this keeps a refactor from quietly routing these models
    down the TOKENS branch, which WOULD turn the displayed figures into
    charges — and they drift against OpenRouter by design.
    """
    entry = next(
        c
        for c in BLOCK_COSTS[AITextGeneratorBlock]
        if c.cost_filter.get("model") == LLMModel("deepseek/deepseek-chat")
    )
    assert entry.cost_type == BlockCostType.COST_USD
    assert entry.cost_amount == 150

    # A turn OpenRouter billed at $0.10 costs ceil(0.10 * 150) = 15 credits,
    # whatever the displayed per-1M rates happen to say. The token counts
    # below are deliberately huge: had this model been token-billed they
    # would have produced 48 + 133.5 credits instead.
    stats = NodeExecutionStats(
        provider_cost=0.10,
        provider_cost_type="cost_usd",
        input_token_count=1_000_000,
        output_token_count=1_000_000,
    )
    cost, _ = block_usage_cost(
        AITextGeneratorBlock(),
        _open_router_input("deepseek/deepseek-chat"),
        stats=stats,
    )
    assert cost == 15


def test_open_router_display_rate_matches_the_catalog_entry():
    """The builder's "$X in / $Y out per 1M" label is the catalog credit rate
    divided by the 150 cr/$ margin — SECRT-2701's corrected deepseek figures.
    """
    entry = next(
        c
        for c in BLOCK_COSTS[AITextGeneratorBlock]
        if c.cost_filter.get("model") == LLMModel("deepseek/deepseek-chat")
    )
    assert entry.token_rate is not None
    assert entry.token_rate.input_usd_per_1m == pytest.approx(0.32)
    assert entry.token_rate.output_usd_per_1m == pytest.approx(0.89)
