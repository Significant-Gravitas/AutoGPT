import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.google.developer_docs import (
    AskGoogleDeveloperDocsBlock,
    GetGoogleDeveloperDocsBlock,
    SearchGoogleDeveloperDocsBlock,
)
from backend.blocks.google.maps_platform_docs import (
    GetGoogleMapsPlatformCodingInstructionsBlock,
    SearchGoogleMapsPlatformDocsBlock,
)


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (SearchGoogleDeveloperDocsBlock, BlockEffect.READ),
        (GetGoogleDeveloperDocsBlock, BlockEffect.READ),
        (AskGoogleDeveloperDocsBlock, BlockEffect.READ),
        (SearchGoogleMapsPlatformDocsBlock, BlockEffect.READ),
        (GetGoogleMapsPlatformCodingInstructionsBlock, BlockEffect.READ),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads and workspace saves without asking, asks before
    # anything that changes Google, and treats an undeclared block as the latter.
    assert block().effect is effect
