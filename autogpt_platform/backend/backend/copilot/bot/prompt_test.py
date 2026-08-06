"""Tests for prompt assembly helpers."""

from backend.platform_linking.models import (
    MAX_BOT_MESSAGE_CHARS,
    BotChatRequest,
    Platform,
)

from .prompt import _TRUNCATION_NOTICE, clamp_prompt


def test_clamp_prompt_leaves_a_fitting_prompt_untouched():
    message = "a" * (MAX_BOT_MESSAGE_CHARS - 1)
    assert clamp_prompt(message, MAX_BOT_MESSAGE_CHARS) == message


def test_clamp_prompt_leaves_an_exactly_capped_prompt_untouched():
    message = "a" * MAX_BOT_MESSAGE_CHARS
    assert clamp_prompt(message, MAX_BOT_MESSAGE_CHARS) == message


def test_clamp_prompt_keeps_the_tail_and_marks_the_cut():
    # The current message sits at the END of the assembled prompt, so the tail
    # (not the head) must survive.
    body = "OLD-CONTEXT" * 5000 + "THE ACTUAL QUESTION"
    clamped = clamp_prompt(body, MAX_BOT_MESSAGE_CHARS)
    assert len(clamped) == MAX_BOT_MESSAGE_CHARS
    assert clamped.startswith(_TRUNCATION_NOTICE)
    assert clamped.endswith("THE ACTUAL QUESTION")
    assert "OLD-CONTEXT" * 5000 not in clamped  # older context was dropped


def test_clamp_prompt_output_always_satisfies_bot_chat_request():
    # Regression: a long conversation's assembled prompt used to exceed the
    # request cap and raise a pydantic ValidationError mid-turn (surfaced to
    # the user as a generic "Something went wrong").
    oversized = "x" * (MAX_BOT_MESSAGE_CHARS * 3)
    request = BotChatRequest(
        platform=Platform.DISCORD,
        platform_user_id="u-1",
        message=clamp_prompt(oversized, MAX_BOT_MESSAGE_CHARS),
    )
    assert len(request.message) <= MAX_BOT_MESSAGE_CHARS


def test_clamp_prompt_hard_cuts_when_cap_below_notice_width():
    # Pathologically tiny cap can't fit the notice — still must respect it.
    tiny = len(_TRUNCATION_NOTICE) - 5
    clamped = clamp_prompt("z" * 100, tiny)
    assert len(clamped) == tiny
    assert clamped == "z" * tiny
