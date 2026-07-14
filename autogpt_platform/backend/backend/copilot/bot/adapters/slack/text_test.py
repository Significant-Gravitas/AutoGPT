"""Tests for Slack mrkdwn conversion."""

from .text import to_mrkdwn


def test_bold_is_converted():
    assert to_mrkdwn("a **bold** word") == "a *bold* word"


def test_link_is_converted():
    assert to_mrkdwn("see [docs](https://x.dev)") == "see <https://x.dev|docs>"


def test_bold_and_link_together():
    assert to_mrkdwn("**hi** [x](y)") == "*hi* <y|x>"


def test_plain_text_unchanged():
    assert to_mrkdwn("nothing to convert") == "nothing to convert"
