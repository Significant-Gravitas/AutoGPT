"""Who may answer an approval card posted in a linked chat channel."""

import pytest

from backend.platform_linking.cards import PostedCard, may_answer
from backend.platform_linking.models import Platform


def _card() -> PostedCard:
    return PostedCard(
        platform=Platform.DISCORD,
        server_id="server-1",
        answerer="owner",
        user_id="user-1",
        session_id="session-1",
        review_id="review-1",
        options=[],
    )


def test_only_the_linking_owner_may_answer_by_default():
    """Only the linking owner can chat with the bot in a server, so only they
    answer the cards their chats raise."""
    assert may_answer(_card(), "owner", "server-1")
    assert not may_answer(_card(), "member", "server-1")


def test_no_one_answers_from_another_conversation():
    assert not may_answer(_card(), "owner", "server-2")
    assert not may_answer(_card(), "owner", None)
