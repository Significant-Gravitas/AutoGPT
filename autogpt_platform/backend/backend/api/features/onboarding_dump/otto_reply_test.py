"""Tests for tidying the judge's reply before Otto says it."""

import pytest

from backend.api.features.onboarding_dump import otto_reply


@pytest.mark.parametrize(
    "raw,shown",
    [
        ("No pressure at all. So tell me,", "No pressure at all. So tell me,"),
        ("  Loud and clear!\n  So tell me, ", "Loud and clear! So tell me,"),
        ('"Hi there! Nice to meet you."', "Hi there! Nice to meet you."),
        ("“One small task is plenty.”", "One small task is plenty."),
        ("'Hi there.'", "Hi there."),
        # Quotes that don't wrap the whole reply are its own, and stay whole.
        ('I hear you loud and "clear"', 'I hear you loud and "clear"'),
        ('"Hello!" Nice to meet you.', '"Hello!" Nice to meet you.'),
        ("'Course, no pressure.", "'Course, no pressure."),
    ],
)
def test_a_usable_reply_is_tidied(raw: str, shown: str):
    assert otto_reply.clean_reply(raw) == shown


@pytest.mark.parametrize(
    "raw",
    [
        None,
        42,
        ["Hi"],
        "",
        "   ",
        '""',
        # No words: the prompt's own placeholder, punctuation or an emoji.
        "...",
        '"..."',
        "…",
        "-",
        "😀",
        # The screen's own question follows; a second one reads as a form.
        "What do you do for work?",
        "x" * (otto_reply.MAX_REPLY_CHARS + 1),
    ],
)
def test_a_reply_that_cannot_be_shown_is_dropped(raw: object):
    assert otto_reply.clean_reply(raw) is None


def test_the_longest_allowed_reply_is_kept():
    reply = "x" * otto_reply.MAX_REPLY_CHARS
    assert otto_reply.clean_reply(reply) == reply
