"""Convert the bot's canonical CommonMark output into Slack mrkdwn.

Escape Slack's control characters first (``&``, ``<``, ``>`` per its mrkdwn
rules), then re-introduce the constructs we emit — ``**bold**`` → ``*bold*``
and ``[label](url)`` → ``<url|label>``. The escaping is ping safety: a raw
``<!channel>`` or ``<@U123>`` in model output would otherwise reach
chat.postMessage as a live mention, bypassing the allowlist. User @-mention
resolution is the shared, allowlist-guarded ``text.resolve_mentions`` policy,
applied by the adapter AFTER this escaping so its ``<@Uid>`` tokens survive.
"""

import re

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def to_mrkdwn(text: str) -> str:
    """Render CommonMark bold + links in Slack's mrkdwn dialect."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _BOLD_RE.sub(r"*\1*", text)
    text = _LINK_RE.sub(r"<\2|\1>", text)
    return text
