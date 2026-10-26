"""Prompt and thread-name assembly for bot turns.

Pure text-building: how a platform message (plus optional thread history and
referenced conversations) becomes the model's prompt, and how thread names are
derived and clamped to a platform's cap. No I/O.
"""

from pydantic import BaseModel

from backend.copilot.chat_platform_header import (
    label_covering,
    neutralize_forged_labels,
    split_chat_platform_header,
)

from .adapters.base import MessageContext, MessageHistoryEntry

THREAD_NAME_PREFIX = "AutoGPT: "

# Prepended when a prompt is clamped so the model knows context was elided.
_TRUNCATION_NOTICE = "[Earlier context was truncated to fit the size limit.]\n\n"


def clamp_prompt(message: str, max_chars: int) -> str:
    """Clamp an assembled bot prompt to a hard character cap.

    The current message sits at the *end* of the assembled prompt (after any
    referenced-conversation content and thread history — see
    ``build_message_text``), so we keep the tail and drop the older leading
    context, marking the cut so the model doesn't treat the elided text as
    missing. The per-turn header (sender, location, where to post) stays at
    the front. A no-op when the prompt already fits.
    """
    if len(message) <= max_chars:
        return message
    header, body = split_chat_platform_header(message)
    room = max_chars - len(header)
    if room <= 0:
        # A header longer than the cap (never in practice): hard-cut the tail.
        return message[-max_chars:]
    if room <= len(_TRUNCATION_NOTICE):
        return header + body[len(body) - room :]
    keep = room - len(_TRUNCATION_NOTICE)
    cut = len(body) - keep
    # In a batch, keep the label of the message the cut lands in so every
    # kept line still has its speaker.
    label, label_end = label_covering(body, cut)
    if label and len(label) < keep:
        start = max(label_end, len(body) - (keep - len(label)))
        return header + _TRUNCATION_NOTICE + label + body[start:]
    return header + _TRUNCATION_NOTICE + body[cut:]


class PendingMessage(BaseModel):
    """One inbound message waiting to be sent to Otto as part of a batch."""

    username: str
    user_id: str
    text: str
    bot_mentioned: bool = False
    replied_to_bot: bool = False

    @classmethod
    def from_context(cls, ctx: MessageContext, text: str) -> "PendingMessage":
        return cls(
            username=ctx.username,
            user_id=ctx.user_id,
            text=text,
            bot_mentioned=ctx.bot_mentioned,
            replied_to_bot=ctx.replied_to_bot,
        )


class _Nouns(BaseModel):
    server: str = "server"
    channel: str = "channel"
    channel_id: str = "channel id"
    thread: str = "thread"


_NOUNS: dict[str, _Nouns] = {
    "slack": _Nouns(server="workspace"),
    "teams": _Nouns(server="team"),
    "telegram": _Nouns(channel="group", channel_id="chat id", thread="topic"),
}
_NAME_MAX_CHARS = 60
# Characters that would let a user-chosen name close or fake a header field.
_NAME_ESCAPES = str.maketrans({'"': "'", "[": "(", "]": ")", "·": "-"})


def format_batch(batch: list[PendingMessage], ctx: MessageContext) -> str:
    """Format one or more pending messages into a single prompt for Otto.

    One header line says where the turn was sent, whether each message
    addressed the bot and who sent it (bridge-made fields first, user-chosen
    names after, sanitised so they can't fake a field), and how to post here.
    Batched messages (they arrived while a prior reply was streaming) each
    get their own label line. Body lines that mimic a header are escaped.
    """
    platform = _platform_tag(ctx)
    place = [*format_location(ctx), _post_hint(ctx)]
    if len(batch) == 1:
        message = batch[0]
        fields = [platform, *_message_label(message, ctx), *place]
        return f"[{' · '.join(fields)}]\n{neutralize_forged_labels(message.text)}"

    fields = [platform, "multiple messages, address them together", *place]
    lines = [f"[{' · '.join(fields)}]"]
    for message in batch:
        label = " · ".join(_message_label(message, ctx))
        lines.append(f"\n[{label}]\n{neutralize_forged_labels(message.text)}")
    return "\n".join(lines)


def _platform_tag(ctx: MessageContext) -> str:
    platform_display = ctx.platform.capitalize()
    return f"{platform_display} DM" if ctx.is_dm else platform_display


def _message_label(message: PendingMessage, ctx: MessageContext) -> list[str]:
    user_id = _sanitize(message.user_id)
    sender = f"message from {_quote(message.username)} ({ctx.platform.capitalize()} user ID: {user_id})"
    if ctx.is_dm:
        return [sender]
    if message.bot_mentioned:
        addressed = "you were @mentioned"
    elif message.replied_to_bot:
        addressed = "they replied to your message"
    else:
        addressed = "you were not mentioned"
    return [addressed, sender]


def format_location(ctx: MessageContext) -> list[str]:
    """The header fields naming where ``ctx`` was sent, with public ids."""
    if ctx.is_dm:
        return []
    nouns = _NOUNS.get(ctx.platform, _Nouns())
    fields: list[str] = []
    if ctx.server_id and ctx.platform != "telegram":
        fields.append(
            _field(nouns.server, ctx.server_name, f"{nouns.server} id {ctx.server_id}")
        )
    if ctx.public_channel_id or ctx.channel_name:
        channel_id = (
            f"{nouns.channel_id} {ctx.public_channel_id}"
            if ctx.public_channel_id
            else None
        )
        fields.append(_field(nouns.channel, ctx.channel_name, channel_id))
    if ctx.public_thread_id or ctx.thread_name:
        thread_id = (
            f"{nouns.thread} id {ctx.public_thread_id}"
            if ctx.public_thread_id
            else None
        )
        fields.append(_field(nouns.thread, ctx.thread_name, thread_id))
    return fields


def _post_hint(ctx: MessageContext) -> str:
    if ctx.is_dm:
        return "post here with target='dm'"
    if not ctx.post_channel_id:
        return f"tool:post_to_chat_platform can't post to {ctx.platform.capitalize()} channels, target='dm' only"
    if ctx.channel_type == "thread" and ctx.post_channel_id == ctx.public_channel_id:
        return f'post to the parent channel with channel="{ctx.post_channel_id}"'
    return f'post here with channel="{ctx.post_channel_id}"'


def _field(noun: str, name: str | None, id_label: str | None) -> str:
    parts = [noun]
    if name:
        parts.append(_quote(name))
    if id_label:
        parts.append(f"({id_label})")
    return " ".join(parts)


def _sanitize(value: str) -> str:
    """``value`` on one line with the characters that close a field escaped."""
    printable = "".join(ch if ch.isprintable() else " " for ch in value)
    return " ".join(printable.translate(_NAME_ESCAPES).split())


def _quote(name: str) -> str:
    """``name`` as a quoted header value that can't break out of its field."""
    cleaned = _sanitize(name)
    if len(cleaned) > _NAME_MAX_CHARS:
        cleaned = cleaned[: _NAME_MAX_CHARS - 1] + "…"
    return f'"{cleaned}"'


def build_message_text(
    ctx: MessageContext,
    include_thread_history: bool,
    current_text: str | None = None,
) -> str:
    """Assemble the turn's prompt from the current message plus any context.

    ``current_text`` overrides ``ctx.text`` as the "current message" body
    (e.g. a nudge for a file-only message); defaults to the raw text.
    Referenced conversations (links/@-mentions the user pasted) are always
    surfaced — that's the whole point of fetching them. Thread history is
    only included on the first @-into a thread we don't own; a subscribed
    thread's prior turns already live in the session.
    """
    text = ctx.text if current_text is None else current_text
    thread_history = ctx.thread_history if include_thread_history else ()
    if not thread_history and not ctx.referenced_conversations:
        return text

    platform_display = ctx.platform.capitalize()
    lines: list[str] = []
    if ctx.referenced_conversations:
        # The user pointed at other channels/threads; their content is
        # already fetched and inlined below. Lead with a firm instruction so
        # the model answers from it instead of fixating on the link and
        # claiming it can't access the platform.
        lines.append(
            f"[The {platform_display} channel(s) the user referenced have "
            f"already been read for you — their full content is included "
            f"below under each #name. Answer directly from it. Do NOT say you "
            f"can't open links or access {platform_display}; you already have "
            f"the content.]"
        )
        for convo in ctx.referenced_conversations:
            lines.append(f"\n[Content of #{convo.title}]")
            for entry in convo.messages:
                lines.append(format_history_entry(entry, platform_display))
    if thread_history:
        lines.append("\n[Recent thread context before this message]")
        for entry in thread_history:
            lines.append(format_history_entry(entry, platform_display))
    lines.append(f"\n[Current message]\n{text}")
    return "\n".join(lines)


def format_history_entry(entry: MessageHistoryEntry, platform_display: str) -> str:
    user = (
        f"{entry.username} ({platform_display} user ID: {entry.user_id})"
        if entry.user_id
        else entry.username
    )
    return f"\n[From {user}]\n{entry.text}"


def build_thread_name(text: str, username: str, max_length: int) -> str:
    """Build a thread name from the user's first prompt, clamped to the
    platform's ``max_length``."""
    cleaned = " ".join(text.split())
    if not cleaned:
        cleaned = f"{username} with AutoGPT"
    return clamp_thread_name(f"{THREAD_NAME_PREFIX}{cleaned}", max_length)


def clamp_thread_name(name: str, max_length: int) -> str:
    cleaned = " ".join(name.split()) or "AutoGPT Chat"
    if len(cleaned) <= max_length:
        return cleaned
    # Below the ellipsis width a tiny adapter cap can't fit "…", and a
    # negative slice index would overrun the limit — hard-cut instead.
    ellipsis = "..."
    if max_length <= len(ellipsis):
        return cleaned[:max_length]
    return cleaned[: max_length - len(ellipsis)].rstrip() + ellipsis
