import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.hacker_news.items import HackerNewsGetItemBlock
from backend.blocks.hacker_news.search import HackerNewsSearchBlock
from backend.blocks.hacker_news.stories import HackerNewsGetStoriesBlock
from backend.blocks.hacker_news.users import HackerNewsGetUserBlock


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (HackerNewsSearchBlock, BlockEffect.READ),
        (HackerNewsGetStoriesBlock, BlockEffect.READ),
        (HackerNewsGetItemBlock, BlockEffect.READ),
        (HackerNewsGetUserBlock, BlockEffect.READ),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads without asking, and treats an undeclared block as one
    # that changes something outside the platform.
    assert block().effect is effect


@pytest.mark.parametrize(
    "block",
    [
        HackerNewsSearchBlock,
        HackerNewsGetStoriesBlock,
        HackerNewsGetItemBlock,
        HackerNewsGetUserBlock,
    ],
)
def test_block_stays_a_primitive(block: type[Block]):
    # A provider-less "service" vanishes from AutoPilot's search when the query
    # names another service ("send hacker news stories to slack"); a primitive
    # still comes back as a fallback there.
    assert block().capability_kind == "primitive"
