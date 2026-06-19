"""Parse Discord channel/thread references out of message text.

A user often points the bot at another thread by pasting its link
(``https://discord.com/channels/<guild>/<channel>/<message>``) or @-mentioning
it (``<#channel_id>``). This module extracts the referenced channel/thread IDs
so the adapter can fetch their content via the gateway — no parsing logic that
needs a live Discord connection lives here, so it's cheap to test.
"""

import re

# A single positional pass over either form so references are returned in the
# order they appear in the text:
#   - link group: discord.com / discordapp.com / canary. / ptb. channel link —
#     the second path segment is the channel/thread ID (third, the message ID,
#     is ignored).
#   - mention group: ``<#channel_id>``, how "#frontend" serializes on the wire.
_REFERENCE_RE = re.compile(
    r"https?://(?:\w+\.)?discord(?:app)?\.com/channels/\d+/(\d+)(?:/\d+)?" r"|<#(\d+)>",
    re.IGNORECASE,
)


def extract_referenced_channel_ids(
    text: str,
    *,
    exclude_channel_id: str,
    limit: int,
) -> list[str]:
    """Return de-duplicated channel/thread IDs referenced in ``text``.

    Order follows first appearance in the text, the current channel is skipped
    (its history is already supplied as thread context), and the result is
    capped at ``limit`` so a message full of links can't trigger an unbounded
    number of fetches.
    """
    seen: set[str] = {exclude_channel_id}
    ids: list[str] = []
    for match in _REFERENCE_RE.finditer(text):
        channel_id = match.group(1) or match.group(2)
        if channel_id in seen:
            continue
        seen.add(channel_id)
        ids.append(channel_id)
    return ids[:limit]


def replace_referenced_links(text: str, labels: dict[str, str]) -> str:
    """Rewrite each channel link / ``<#id>`` mention into a readable ``#label``.

    Only references whose channel ID is in ``labels`` (the ones we actually
    fetched) are rewritten; anything else is left untouched. This keeps the
    model from fixating on a raw URL it thinks it must open, when the linked
    content has already been supplied to it under that ``#label``.
    """

    def _sub(match: "re.Match[str]") -> str:
        channel_id = match.group(1) or match.group(2)
        label = labels.get(channel_id)
        return f"#{label}" if label else match.group(0)

    return _REFERENCE_RE.sub(_sub, text)
