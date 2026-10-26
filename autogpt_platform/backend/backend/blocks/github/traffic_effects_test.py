import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.github.traffic import GithubGetRepositoryTrafficBlock


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (GithubGetRepositoryTrafficBlock, BlockEffect.READ),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads without asking, asks before anything that changes
    # GitHub, and treats an undeclared block as the latter.
    assert block().effect is effect
