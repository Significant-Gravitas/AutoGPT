"""Prompt and thread-name assembly for bot turns.

Pure text-building: how a platform message (plus optional thread history and
referenced conversations) becomes the model's prompt, and how thread names are
derived and clamped to a platform's cap. No I/O.
"""

from .adapters.base import MessageContext, MessageHistoryEntry

THREAD_NAME_PREFIX = "AutoGPT: "


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
