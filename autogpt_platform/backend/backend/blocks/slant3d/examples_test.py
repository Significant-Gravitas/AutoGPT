import pytest

from backend.blocks.slant3d.filament import Slant3DFilamentBlock
from backend.blocks.slant3d.order import (
    Slant3DCreateOrderBlock,
    Slant3DEstimateOrderBlock,
    Slant3DEstimateShippingBlock,
)
from backend.blocks.slant3d.order_status import (
    Slant3DCancelOrderBlock,
    Slant3DGetOrdersBlock,
    Slant3DProcessOrderBlock,
    Slant3DTrackingBlock,
)
from backend.blocks.slant3d.slicing import Slant3DSlicerBlock
from backend.blocks.slant3d.webhook import Slant3DOrderWebhookBlock
from backend.util.test import execute_block_test


@pytest.mark.parametrize(
    "block_class",
    [
        Slant3DFilamentBlock,
        Slant3DCreateOrderBlock,
        Slant3DEstimateOrderBlock,
        Slant3DEstimateShippingBlock,
        Slant3DCancelOrderBlock,
        Slant3DGetOrdersBlock,
        Slant3DProcessOrderBlock,
        Slant3DTrackingBlock,
        Slant3DSlicerBlock,
        Slant3DOrderWebhookBlock,
    ],
)
async def test_block_examples(block_class):
    await execute_block_test(block_class())
