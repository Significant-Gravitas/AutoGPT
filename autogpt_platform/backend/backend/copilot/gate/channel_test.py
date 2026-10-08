"""The channel card's options: Allow once, Always allow, Deny, and what each
answer saves."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.gate import channel, chat_rules
from backend.copilot.gate.review import answer_options

_HEADLINE = {"ask": "Post", "object": "a message"}


def _row(payload: dict, status: ReviewStatus = ReviewStatus.WAITING):
    return SimpleNamespace(
        payload={"headline": _HEADLINE, **payload}, status=status, session_id="s"
    )


def test_a_card_that_can_rule_offers_allow_once_always_allow_and_deny():
    view = channel.card_for(_row({"chat_rules_allowed": ["allow", "judge"]}))
    assert [(o.choice, o.label) for o in view.options] == [
        ("approve", "Allow once"),
        ("always_allow", "Always allow"),
        ("judge", "Let Otto judge in this chat"),
        ("reject", "Deny"),
    ]
    always = view.options[1]
    assert (always.scope, always.lifetime) == ("chat", "always")
    assert view.options[0].scope is None and view.options[-1].lifetime is None


def test_a_card_that_rules_on_nothing_offers_once_and_deny():
    view = channel.card_for(_row({"chat_rules_allowed": []}))
    assert [o.choice for o in view.options] == ["approve", "reject"]


def test_a_held_read_keeps_its_own_words():
    view = channel.card_for(_row({"reason_kind": "content", "reader": "Ada"}))
    assert [o.label for o in view.options] == ["Release to Ada", "Keep it out"]


def test_the_web_and_channel_cards_share_their_options():
    options = answer_options(["allow"])
    assert [(o.id, o.approved, o.rule) for o in options] == [
        ("approve", True, None),
        ("always_allow", True, "allow"),
        ("reject", False, None),
    ]
    assert (options[1].scope, options[1].lifetime) == (
        chat_rules.DEFAULT_SCOPE,
        chat_rules.DEFAULT_LIFETIME,
    )


def test_a_card_posted_before_always_allow_still_parses():
    """Posted cards keep their options in Redis for 30 days."""
    old = channel.CardOption.model_validate(
        {"choice": "approve_chat", "label": "Approve for this chat", "receipt": "x"}
    )
    assert old.scope is None and old.lifetime is None


@pytest.mark.parametrize(
    "choice, approved, grant",
    [
        ("approve", True, None),
        ("always_allow", True, chat_rules.AllowGrant(scope="chat", lifetime="always")),
        ("approve_chat", True, None),
        ("reject", False, None),
    ],
)
async def test_each_answer_saves_its_rule_in_this_chat_only(choice, approved, grant):
    reviews = MagicMock(
        get_reviews_by_node_exec_ids=AsyncMock(
            return_value={"r": _row({"chat_rules_allowed": ["allow"]})}
        ),
        process_all_reviews_for_execution=AsyncMock(return_value={"r": object()}),
    )
    set_rules = AsyncMock()
    with (
        patch.object(channel, "review_db", return_value=reviews),
        patch.object(
            channel.held, "subject_keys", AsyncMock(return_value={"r": "mcp:h/t"})
        ),
        patch.object(channel.chat_rules, "set_answer_rules", set_rules),
    ):
        assert await channel.answer("u", "s", "r", choice) == "answered"
    decisions = reviews.process_all_reviews_for_execution.await_args.kwargs
    status = decisions["review_decisions"]["r"][0]
    assert status == (ReviewStatus.APPROVED if approved else ReviewStatus.REJECTED)
    if choice in ("approve", "reject"):
        set_rules.assert_not_awaited()
        return
    args = set_rules.await_args.args
    assert args[3] == {"r": "allow"}
    assert args[5] == {"r": "chat"}
    assert args[6] == ({"r": grant} if grant else None)
