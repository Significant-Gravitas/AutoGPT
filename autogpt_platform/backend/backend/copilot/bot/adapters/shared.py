"""Platform-agnostic helpers shared by every adapter.

These capture policy that is identical across chat platforms — inbound
attachment caps + skip bookkeeping, and the bot-loop guard — so each adapter
supplies only the thin platform-specific callback (how to fetch a file's
bytes; whether the author is us/a bot). Keeping the policy here rather than
re-implemented per adapter is what lets a new adapter stay small and keeps the
rules in one place.
"""

import logging
from typing import AsyncIterable, Awaitable, Callable, Sequence

from pydantic import BaseModel

from .base import InboundAttachment, MessageHistoryEntry

logger = logging.getLogger(__name__)

_HISTORY_TRUNCATION_MARKER = "\n… [message truncated]"


class InboundFile(BaseModel):
    """A platform file an adapter wants to ingest, normalized for
    ``collect_attachments``. ``fetch`` is the only platform-specific part — a
    zero-arg coroutine bound to this file that downloads its bytes.
    """

    filename: str | None
    size: int
    mime_type: str | None
    fetch: Callable[[], Awaitable[bytes]]


async def collect_attachments(
    files: Sequence[InboundFile],
    *,
    max_count: int,
    max_bytes: int,
) -> tuple[tuple[InboundAttachment, ...], tuple[tuple[str, str], ...]]:
    """Download the user's inbound files under shared caps.

    Returns ``(kept, skipped)`` where ``skipped`` is ``(filename, reason)`` for
    files dropped over the per-message count cap, over the per-file size cap, or
    on a failed download — so the caller can tell the user and the model rather
    than silently losing them. Only ``fetch`` touches the platform; the caps and
    skip bookkeeping are shared policy every adapter reuses.
    """
    kept: list[InboundAttachment] = []
    skipped: list[tuple[str, str]] = []
    for extra in files[max_count:]:
        skipped.append((extra.filename or "file", "too many files attached"))
    for f in files[:max_count]:
        name = f.filename or "file"
        if f.size > max_bytes:
            skipped.append((name, "too large"))
            continue
        try:
            content = await f.fetch()
        except Exception:
            # The collector can't know each platform's download exceptions, so
            # it catches broadly: a fetch failure becomes a per-file skip the
            # user is told about, never a dropped message.
            logger.warning("Could not download attachment %s", name)
            skipped.append((name, "couldn't be downloaded"))
            continue
        kept.append(
            InboundAttachment(
                filename=name,
                mime_type=f.mime_type or "application/octet-stream",
                content=content,
            )
        )
    return tuple(kept), tuple(skipped)


async def budget_history(
    entries: AsyncIterable[MessageHistoryEntry],
    *,
    char_budget: int,
) -> tuple[MessageHistoryEntry, ...]:
    """Budget a message history down to ``char_budget`` chars, returning it in
    chronological order.

    **Precondition — ``entries`` MUST be yielded newest-first.** Keeping the
    *most recent* messages under budget is only possible by consuming from the
    newest end (a `MessageHistoryEntry` has no timestamp to sort by), so the
    adapter must stream its history newest→oldest; the result is reversed back
    to chronological for the prompt. Passing oldest-first would keep the wrong
    end of the conversation.

    The adapter yields entries already normalized for its platform (its own
    messages skipped, mentions stripped, empties dropped); this owns only the
    budgeting + truncation, which is identical for every platform and is
    currently assembled in two places in the Discord adapter alone. The lone
    most-recent entry, if itself over budget, keeps a truncated head rather
    than being dropped or emitted oversized.
    """
    kept: list[MessageHistoryEntry] = []
    used = 0
    async for entry in entries:
        remaining = char_budget - used
        if remaining <= 0:
            break
        text = entry.text
        oversized = len(text) > remaining
        if oversized and kept:
            # No room for another whole message — keep what we have.
            break
        if oversized:
            # Lone most-recent message is itself over budget: keep a head.
            text = _truncate_to_budget(text, remaining)
            entry = MessageHistoryEntry(
                username=entry.username, user_id=entry.user_id, text=text
            )
        used += len(text)
        kept.append(entry)
        if oversized:
            break
    kept.reverse()  # chronological order for the prompt
    return tuple(kept)


def _truncate_to_budget(text: str, limit: int) -> str:
    """Trim ``text`` to at most ``limit`` chars, leaving a visible marker.

    Used only when a single message is itself larger than the history budget —
    keep a head of it (with a cut marker) rather than emit an oversized payload
    or drop the message entirely.
    """
    if len(text) <= limit:
        return text
    # Below the marker's own width there's no room for a head + marker; hard-cut
    # so the result still respects the budget (matters for adapters that pass a
    # small budget — the marker is ~22 chars).
    if limit <= len(_HISTORY_TRUNCATION_MARKER):
        return text[:limit]
    keep = limit - len(_HISTORY_TRUNCATION_MARKER)
    return text[:keep].rstrip() + _HISTORY_TRUNCATION_MARKER


def should_ignore(*, is_self: bool, author_is_bot: bool, bot_mentioned: bool) -> bool:
    """Whether to drop an inbound message before doing any work.

    Ignore our own messages always; ignore other bots unless they @-mention us
    — that mention gate is what stops two bots in a shared thread (a dev bot and
    a prod bot included) from replying to each other forever. The adapter
    computes the three booleans from its platform's message; the decision is
    shared so no adapter can forget the loop guard.
    """
    if is_self:
        return True
    return author_is_bot and not bot_mentioned
