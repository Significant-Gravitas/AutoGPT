import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.google.slides_create import (
    GoogleSlidesAddSlideBlock,
    GoogleSlidesCreatePresentationBlock,
)
from backend.blocks.google.slides_edit import (
    GoogleSlidesBatchUpdateBlock,
    GoogleSlidesReplaceAllTextBlock,
    GoogleSlidesSetSpeakerNotesBlock,
)
from backend.blocks.google.slides_read import (
    GoogleSlidesGetSlideBlock,
    GoogleSlidesGetSlideThumbnailBlock,
    GoogleSlidesReadPresentationBlock,
)


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (GoogleSlidesReadPresentationBlock, BlockEffect.READ),
        (GoogleSlidesGetSlideBlock, BlockEffect.READ),
        (GoogleSlidesGetSlideThumbnailBlock, BlockEffect.WORKSPACE),
        (GoogleSlidesCreatePresentationBlock, BlockEffect.EXTERNAL),
        (GoogleSlidesAddSlideBlock, BlockEffect.EXTERNAL),
        (GoogleSlidesBatchUpdateBlock, BlockEffect.EXTERNAL),
        (GoogleSlidesReplaceAllTextBlock, BlockEffect.EXTERNAL),
        (GoogleSlidesSetSpeakerNotesBlock, BlockEffect.EXTERNAL),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads and workspace saves without asking, asks before
    # anything that changes Google, and treats an undeclared block as the latter.
    assert block().effect is effect
