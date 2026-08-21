"""Convert the bot's canonical CommonMark output into Slack mrkdwn.

Escape Slack's control characters first (``&``, ``<``, ``>`` per its mrkdwn
rules), then re-introduce the constructs we emit — ``**bold**`` → ``*bold*``
and ``[label](url)`` → ``<url|label>``. The escaping is ping safety: a raw
``<!channel>`` or ``<@U123>`` in model output would otherwise reach
chat.postMessage as a live mention, bypassing the allowlist. User @-mention
resolution is the shared, allowlist-guarded ``text.resolve_mentions`` policy,
applied by the adapter around this escaping so its ``<@Uid>`` tokens survive.
"""

import re

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
# The link rule re-emits raw angle brackets, so only a real web/mail target may
# become <url|label>: ``[x](!channel)`` or ``[x](@U123)`` would otherwise turn
# back into a live mention after the escaping above.
_SAFE_URL_RE = re.compile(r"^(?:https?://|mailto:)[^\s|]+$", re.IGNORECASE)
# A line-leading ">" is a blockquote. It can't close a control sequence (every
# "<" is already escaped), so it is given back after escaping.
_QUOTE_RE = re.compile(r"^&gt;", re.MULTILINE)


def to_mrkdwn(text: str) -> str:
    """Render CommonMark bold + links in Slack's mrkdwn dialect."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _QUOTE_RE.sub(">", text)
    text = _BOLD_RE.sub(r"*\1*", text)
    text = _LINK_RE.sub(_link, text)
    return text


def _link(match: re.Match[str]) -> str:
    label, url = match.group(1), match.group(2)
    if not _SAFE_URL_RE.match(url):
        return match.group(0)
    return f"<{url}|{label}>"
