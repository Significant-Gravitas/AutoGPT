"""Platform-agnostic helpers shared by every adapter.

These capture policy that is identical across chat platforms — inbound
attachment caps + skip bookkeeping, and the bot-loop guard — so each adapter
supplies only the thin platform-specific callback (how to fetch a file's
bytes; whether the author is us/a bot). Keeping the policy here rather than
re-implemented per adapter is what lets a new adapter stay small and keeps the
rules in one place.
"""

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Sequence

from .base import InboundAttachment

logger = logging.getLogger(__name__)


@dataclass
class InboundFile:
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
