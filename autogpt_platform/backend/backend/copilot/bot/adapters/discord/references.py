"""Parse Discord channel/thread references out of message text.

A user often points the bot at another conversation by pasting its link
(``https://discord.com/channels/<guild>/<channel>/<message>``) or @-mentioning
a channel (``<#channel_id>``). This module extracts the referenced
channel/thread (and the specific message, when the link names one) so the
adapter can fetch their content via the gateway — no parsing logic that needs a
live Discord connection lives here, so it's cheap to test.
"""

import re

from pydantic import BaseModel, ConfigDict

# A single positional pass over either form so references are returned in the
# order they appear in the text:
#   - link group: discord.com / discordapp.com / canary. / ptb. channel link —
#     the second path segment is the channel/thread ID, the optional third is a
#     specific message ID.
#   - mention group: ``<#channel_id>``, how "#frontend" serializes on the wire.
_REFERENCE_RE = re.compile(
    r"https?://(?:\w+\.)?discord(?:app)?\.com/channels/\d+/(\d+)(?:/(\d+))?"
    r"|<#(\d+)>",
    re.IGNORECASE,
)


class ReferenceTarget(BaseModel):
    """A channel/thread the message points at, plus the specific message ID
    when the link named one (a ``.../channel/message`` permalink)."""

    model_config = ConfigDict(frozen=True)

    channel_id: str
    message_id: str | None


def _match_target(match: "re.Match[str]") -> ReferenceTarget:
    # group(1)/group(2): channel + optional message from a permalink;
    # group(3): channel from a ``<#id>`` mention (never names a message).
    channel_id = match.group(1) or match.group(3)
    return ReferenceTarget(channel_id=channel_id, message_id=match.group(2))


def extract_referenced_targets(
    text: str,
    *,
    exclude_channel_id: str,
    limit: int,
) -> list[ReferenceTarget]:
    """Return de-duplicated reference targets in ``text``, in first-seen order.

    A bare channel reference to the *current* channel is skipped — its history
    is already supplied as context. A permalink to a *specific message* is kept
    even in the current channel: "what was said here <link>" is a request to
    read that exact message, not redundant. Capped at ``limit`` so a link-heavy
    message can't fan out into unbounded fetches.
    """
    seen: set[tuple[str, str | None]] = set()
    targets: list[ReferenceTarget] = []
    for match in _REFERENCE_RE.finditer(text):
        target = _match_target(match)
        if target.message_id is None and target.channel_id == exclude_channel_id:
            continue
        key = (target.channel_id, target.message_id)
        if key in seen:
            continue
        seen.add(key)
        targets.append(target)
    return targets[:limit]


def replace_referenced_links(text: str, labels: dict[str, str]) -> str:
    """Rewrite each channel link / ``<#id>`` mention into a readable ``#label``.

    Only references whose channel ID is in ``labels`` (the ones we actually
    fetched) are rewritten; anything else is left untouched. This keeps the
    model from fixating on a raw URL it thinks it must open, when the linked
    content has already been supplied to it under that ``#label``.
    """

    def _sub(match: "re.Match[str]") -> str:
        channel_id = match.group(1) or match.group(3)
        label = labels.get(channel_id)
        return f"#{label}" if label else match.group(0)

    return _REFERENCE_RE.sub(_sub, text)
