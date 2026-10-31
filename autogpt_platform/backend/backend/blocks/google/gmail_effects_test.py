import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.google.gmail_labels import (
    GmailCreateLabelBlock,
    GmailUpdateLabelsBlock,
)
from backend.blocks.google.gmail_messages import (
    GmailGetMessageBlock,
    GmailListDraftsBlock,
)
from backend.blocks.google.gmail_organize import (
    GmailMarkAsReadBlock,
    GmailSpamBlock,
    GmailTrashBlock,
)


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (GmailGetMessageBlock, BlockEffect.READ),
        (GmailListDraftsBlock, BlockEffect.READ),
        (GmailMarkAsReadBlock, BlockEffect.EXTERNAL),
        (GmailSpamBlock, BlockEffect.EXTERNAL),
        (GmailTrashBlock, BlockEffect.EXTERNAL),
        (GmailCreateLabelBlock, BlockEffect.EXTERNAL),
        (GmailUpdateLabelsBlock, BlockEffect.EXTERNAL),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads and workspace saves without asking, asks before
    # anything that changes Google, and treats an undeclared block as the latter.
    assert block().effect is effect
