"""What the plan step says Otto heard: the greeting's opening, without the
invitation it ends on, short enough for one line under the title."""

import pytest

from backend.api.features.onboarding_dump import plan_service
from backend.api.features.onboarding_dump.plan_service import heard_from

HEARD = "You sell stickers on Etsy. Listings eat most of your week."


def test_two_short_sentences_are_kept_whole():
    assert heard_from(HEARD) == HEARD


@pytest.mark.parametrize(
    "invitation",
    [
        "Here’s where I would start.",
        "Here's where I would start.",
        "Let’s take that off your plate.",
        "Let's take that off your plate.",
        "Pick one below to begin.",
        "Want me to start with the listings?",
        "Shall we?",
    ],
)
def test_the_invitation_the_greeting_ends_on_is_left_out(invitation: str):
    assert heard_from(f"{HEARD} {invitation}") == HEARD


def test_every_trailing_invitation_is_left_out_not_just_the_last():
    greeting = "You sell stickers on Etsy. Here’s a team for it. Let’s pick a task."

    assert heard_from(greeting) == "You sell stickers on Etsy."


def test_a_sentence_that_only_starts_like_an_invitation_is_kept():
    greeting = "Starting a shop is a lot. Whereabouts matter less than listings."

    assert heard_from(greeting) == greeting


@pytest.mark.parametrize(
    "greeting",
    ["Here’s where I would start. Let’s pick one together. Ready?", "", "   "],
    ids=["all invitation", "empty", "blank"],
)
def test_a_greeting_with_nothing_heard_in_it_gives_none(greeting: str):
    assert heard_from(greeting) is None


def test_two_sentences_too_long_together_give_the_first():
    first = (
        "You sell hand-drawn stickers and prints on Etsy and at two weekend "
        "markets, mostly to teachers and students."
    )
    second = (
        "Listings, customer emails and the books eat most of your week, and "
        "you want your evenings back."
    )
    assert len(first) <= plan_service.MAX_HEARD_CHARS
    assert len(f"{first} {second}") > plan_service.MAX_HEARD_CHARS

    assert heard_from(f"{first} {second} Here’s a start.") == first


def test_a_first_sentence_too_long_is_cut_at_a_word_with_an_ellipsis():
    first = (
        "You sell hand-drawn stickers and prints on Etsy and at two weekend "
        "markets, mostly to teachers, students, planners, journalers, and the "
        "occasional office manager, and the listings eat most of every week."
    )

    heard = heard_from(f"{first} Here’s a start.")

    assert heard == (
        "You sell hand-drawn stickers and prints on Etsy and at two weekend "
        "markets, mostly to teachers, students, planners, journalers, and the "
        "occasional office…"
    )
    assert len(heard) <= plan_service.MAX_HEARD_CHARS


def test_a_cut_that_lands_after_a_comma_drops_the_comma():
    first = "word " * 29 + "comma, extraordinarily long words continue past the cut."

    heard = heard_from(first)

    assert heard == "word " * 28 + "word comma…"


@pytest.mark.parametrize(
    "first",
    ["x" * 300, "日本語" * 80 + "。"],
    ids=["one long token", "a script written without spaces"],
)
def test_a_first_sentence_too_long_with_no_space_keeps_every_character_it_can(
    first: str,
):
    heard = heard_from(first)

    assert heard == first[: plan_service.MAX_HEARD_CHARS - 1] + "…"
    assert heard is not None and len(heard) == plan_service.MAX_HEARD_CHARS
