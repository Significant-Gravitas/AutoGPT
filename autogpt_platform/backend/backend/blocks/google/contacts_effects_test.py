import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.google.contacts import (
    GoogleContactsGetMyProfileBlock,
    GoogleContactsSearchBlock,
)
from backend.blocks.google.contacts_directory import GoogleContactsSearchDirectoryBlock


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (GoogleContactsSearchBlock, BlockEffect.READ),
        (GoogleContactsGetMyProfileBlock, BlockEffect.READ),
        (GoogleContactsSearchDirectoryBlock, BlockEffect.READ),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads and workspace saves without asking, asks before
    # anything that changes Google, and treats an undeclared block as the latter.
    assert block().effect is effect
