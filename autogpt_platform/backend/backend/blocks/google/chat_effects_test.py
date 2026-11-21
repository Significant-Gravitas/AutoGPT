import pytest

from backend.blocks._base import Block, BlockEffect
from backend.blocks.google.chat_direct_messages import (
    GoogleChatFindDirectMessageBlock,
    GoogleChatStartDirectMessageBlock,
)
from backend.blocks.google.chat_message_search import GoogleChatSearchMessagesBlock
from backend.blocks.google.chat_messages import (
    GoogleChatListMessagesBlock,
    GoogleChatSendMessageBlock,
)
from backend.blocks.google.chat_spaces import (
    GoogleChatFindGroupChatsBlock,
    GoogleChatListSpacesBlock,
    GoogleChatSearchSpacesBlock,
)


@pytest.mark.parametrize(
    ("block", "effect"),
    [
        (GoogleChatListSpacesBlock, BlockEffect.READ),
        (GoogleChatSearchSpacesBlock, BlockEffect.READ),
        (GoogleChatFindGroupChatsBlock, BlockEffect.READ),
        (GoogleChatListMessagesBlock, BlockEffect.READ),
        (GoogleChatSendMessageBlock, BlockEffect.EXTERNAL),
        (GoogleChatSearchMessagesBlock, BlockEffect.READ),
        (GoogleChatFindDirectMessageBlock, BlockEffect.READ),
        (GoogleChatStartDirectMessageBlock, BlockEffect.EXTERNAL),
    ],
)
def test_block_declares_what_it_does(block: type[Block], effect: BlockEffect):
    # AutoPilot runs reads and workspace saves without asking, asks before
    # anything that changes Google, and treats an undeclared block as the latter.
    assert block().effect is effect
