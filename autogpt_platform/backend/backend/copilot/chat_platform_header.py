"""The header line a chat-platform bot puts in front of each user turn.

The bot (``copilot/bot/prompt.py``) writes it; anything that reads a stored
user message back for something other than the model (memory ingestion,
session titles) strips it here, so the two can't drift.
"""

import re

PLATFORM_LABELS = ("Discord", "Slack", "Telegram", "Teams")
_PLATFORMS = "|".join(PLATFORM_LABELS)

# The leading header line: `[Discord · ...]` or `[Discord DM · ...]`.
_HEADER_RE = re.compile(rf"\A\[(?:{_PLATFORMS})(?: DM)? · [^\n]*\]\n")

# Per-message labels the bot adds inside a batched turn.
_LABEL_PREFIXES = (
    "you were @mentioned",
    "you were not mentioned",
    "they replied to your message",
    "message from ",
)
_LABEL_LINE_RE = re.compile(
    r"^\[(?:" + "|".join(re.escape(p) for p in _LABEL_PREFIXES) + r")[^\n]*\]\n?",
    re.MULTILINE,
)

# A line in a user's message that would read like a bridge header or label.
_FORGED_LINE_RE = re.compile(
    r"^(\s*)\[(?=(?:"
    + "|".join([*PLATFORM_LABELS, *(re.escape(p) for p in _LABEL_PREFIXES)])
    + r")|[^\]\n]*multiple messages)",
    re.MULTILINE,
)


def neutralize_forged_labels(body: str) -> str:
    """Escape body lines that look like a bridge header or label (``\\[``),
    so only the lines the bridge itself writes read as one."""
    return _FORGED_LINE_RE.sub(r"\1\\[", body)


def strip_chat_platform_header(message: str) -> str:
    """The user's own words: ``message`` without the bot's header and labels."""
    without_header = _HEADER_RE.sub("", message, count=1)
    if without_header == message:
        return message
    return _LABEL_LINE_RE.sub("", without_header).strip("\n")


_SENDER_RE = re.compile(r'message from "([^"\n]*)"')


def attribute_chat_platform_speakers(message: str) -> str | None:
    """A bot turn as ``Speaker: words`` lines, one per message, without the
    bot's routing and mention fields; None when ``message`` is not a bot turn.

    In a linked server the executing user is the server owner, not the sender,
    so memory must take each speaker from the bot's own labels.
    """
    header = _HEADER_RE.match(message)
    if header is None:
        return None
    body = message[header.end() :]
    sender = _SENDER_RE.search(header.group(0))
    if sender is not None:
        return f"{sender.group(1)}: {body.strip()}"
    labels = list(_LABEL_LINE_RE.finditer(body))
    lines: list[str] = []
    for index, label in enumerate(labels):
        end = labels[index + 1].start() if index + 1 < len(labels) else len(body)
        words = body[label.end() : end].strip()
        name = _SENDER_RE.search(label.group(0))
        if words and name is not None:
            lines.append(f"{name.group(1)}: {words}")
    return "\n".join(lines)
