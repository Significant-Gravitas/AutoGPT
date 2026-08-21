"""Tests for Slack mrkdwn conversion."""

import pytest

from .text import to_mrkdwn


def test_bold_is_converted():
    assert to_mrkdwn("a **bold** word") == "a *bold* word"


def test_link_is_converted():
    assert to_mrkdwn("see [docs](https://x.dev)") == "see <https://x.dev|docs>"


def test_bold_and_link_together():
    assert to_mrkdwn("**hi** [x](https://y)") == "*hi* <https://y|x>"


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


@pytest.mark.parametrize(
    "payload",
    [
        "[click here](!channel)",
        "[x](!here)",
        "[x](@U12345678)",
        "[x](!subteam^S1)",
        "[x](#C123)",
        # A Slack filename, echoed by the attachment-problems note.
        "`[everyone](!channel).zip` is too large",
    ],
)
def test_link_rule_cannot_smuggle_a_mention(payload):
    # Only http(s)/mailto targets become <url|label>; anything else stays as
    # escaped text, so the link rule can't re-open a control sequence.
    assert "<" not in to_mrkdwn(payload)


def test_blockquotes_survive_escaping():
    assert to_mrkdwn("> quoted\n<!channel>") == "> quoted\n&lt;!channel&gt;"
    assert to_mrkdwn("a > b") == "a &gt; b"
