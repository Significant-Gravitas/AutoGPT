import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.google.docs_batch_update import GoogleDocsBatchUpdateBlock
from backend.blocks.google.sheets_batch_update import GoogleSheetsBatchUpdateBlock


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (GoogleDocsBatchUpdateBlock, BlockEffect.EXTERNAL),
        (GoogleSheetsBatchUpdateBlock, BlockEffect.EXTERNAL),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads and workspace saves without asking, asks before
    # anything that changes Google, and treats an undeclared block as the latter.
    assert block().effect is effect
