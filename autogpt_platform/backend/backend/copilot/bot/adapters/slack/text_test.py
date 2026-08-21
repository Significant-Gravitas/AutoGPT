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


def test_slack_control_sequences_are_escaped():
    # A raw <!channel> / <@Uid> in model output must render as text, not ping.
    assert to_mrkdwn("<!channel> hey") == "&lt;!channel&gt; hey"
    assert to_mrkdwn("ping <@U123ABC>") == "ping &lt;@U123ABC&gt;"


def test_ampersand_is_escaped_including_in_links():
    assert to_mrkdwn("a & b") == "a &amp; b"
    assert (
        to_mrkdwn("see [docs](https://x.dev?a=1&b=2)")
        == "see <https://x.dev?a=1&amp;b=2|docs>"
    )
