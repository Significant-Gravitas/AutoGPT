import pytest

from backend.blocks.ayrshare.post_to_x import PostToXBlock
from backend.blocks.bannerbear.text_overlay import BannerbearTextOverlayBlock
from backend.blocks.jina.chunking import JinaChunkingBlock
from backend.data.block_cost_config import BLOCK_COSTS
from backend.executor.utils import block_usage_cost


@pytest.mark.parametrize(
    "block_class",
    [
        # Spot-check one block per credential source to keep the suite fast; the
        # per-block tier tuple is built from a shared dict comprehension so
        # covering every social platform here would be redundant.
        PostToXBlock,
    ],
)
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


def test_jina_chunking_has_flat_cost_floor():
    from backend.integrations.credentials_store import jina_credentials

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
