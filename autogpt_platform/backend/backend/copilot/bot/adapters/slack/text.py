"""Convert the bot's canonical CommonMark output into Slack mrkdwn.

Only the markup dialect differs here — ``**bold**`` → ``*bold*`` and
``[label](url)`` → ``<url|label>``. User @-mention resolution is the shared,
allowlist-guarded ``text.resolve_mentions`` policy, applied separately in the
adapter's send methods.
"""

import re

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def to_mrkdwn(text: str) -> str:
    """Render CommonMark bold + links in Slack's mrkdwn dialect."""
    text = _BOLD_RE.sub(r"*\1*", text)
    text = _LINK_RE.sub(r"<\2|\1>", text)
    return text
