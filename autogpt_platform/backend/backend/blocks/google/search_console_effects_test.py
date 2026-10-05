import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.google.search_console import (
    GoogleSearchConsoleGetPerformanceBlock,
    GoogleSearchConsoleListSitesBlock,
)
from backend.blocks.google.search_console_indexing import (
    GoogleSearchConsoleInspectURLBlock,
    GoogleSearchConsoleListSitemapsBlock,
)


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (GoogleSearchConsoleListSitesBlock, BlockEffect.READ),
        (GoogleSearchConsoleGetPerformanceBlock, BlockEffect.READ),
        (GoogleSearchConsoleInspectURLBlock, BlockEffect.READ),
        (GoogleSearchConsoleListSitemapsBlock, BlockEffect.READ),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads and workspace saves without asking, asks before
    # anything that changes Google, and treats an undeclared block as the latter.
    assert block().effect is effect
