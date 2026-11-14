import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.exa.agent import ExaAgentRunBlock
from backend.blocks.exa.research import ExaCreateResearchBlock, ExaListResearchBlock
from backend.blocks.exa.research_queries import (
    ExaGetResearchBlock,
    ExaWaitForResearchBlock,
)


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (ExaAgentRunBlock, BlockEffect.EXTERNAL_JUDGED),
        (ExaCreateResearchBlock, BlockEffect.EXTERNAL_JUDGED),
        (ExaGetResearchBlock, BlockEffect.READ),
        (ExaWaitForResearchBlock, BlockEffect.READ),
        (ExaListResearchBlock, BlockEffect.READ),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # Starting an Exa Agent run (which Create Research now does) leaves a paid
    # run in the user's Exa account, so AutoPilot must not treat it as a read.
    assert block().effect is effect
