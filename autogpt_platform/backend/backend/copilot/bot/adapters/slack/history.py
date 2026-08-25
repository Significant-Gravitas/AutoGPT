"""Thread history for the Slack adapter.

A first @-mention into a thread the bot doesn't own pulls the thread's recent
messages into the prompt. Slack's ``conversations.replies`` pages oldest-first
from the parent, so the fetch follows the cursor to the end and keeps only the
tail, which the shared char budget then trims newest-first.
"""

import asyncio
import logging
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Optional

from slack_sdk.web.async_client import AsyncWebClient

from backend.copilot.bot.adapters.base import MessageHistoryEntry
from backend.copilot.bot.adapters.shared import budget_history

logger = logging.getLogger(__name__)

# Slack recommends <=200 per page. The tail holds more messages than the char
# budget can keep, and the page cap bounds a pathological thread.
PAGE_SIZE = 200
TAIL_SIZE = 200
MAX_PAGES = 10
CHAR_BUDGET = 24000
# users.info is rate-limited workspace-wide; don't fan out unbounded.
NAME_LOOKUP_CONCURRENCY = 8
# Author labels are inlined into the prompt frame; a bot_message "username" is
# attacker-set and newline-capable, so flatten and cap it.
MAX_LABEL_LENGTH = 80


async def fetch_thread_history(
    client: AsyncWebClient,
    *,
    channel: str,
    thread_ts: str,
    exclude_ts: str,
    bot_user_id: str,
    display_name: Callable[[str], Awaitable[str]],
    strip_mentions: Callable[[str], Awaitable[str]],
) -> tuple[MessageHistoryEntry, ...]:
    """Budgeted, chronological history of ``thread_ts``, minus the triggering
    message and the bot's own posts (those turns already live in the session).
    Other bots' posts stay: an alert thread is often exactly what the user wants
    summarized. ``display_name`` / ``strip_mentions`` are the adapter's
    workspace-bound helpers."""
    if not bot_user_id:
        # Without our own identity the self-filter fails open and the bot's
        # prior replies would be folded back into its own prompt.
        logger.warning("Slack bot identity unknown; skipping thread history")
        return ()
    tail = await _fetch_tail(client, channel, thread_ts)
    recent, preset = _select_recent(
        tail, exclude_ts=exclude_ts, bot_user_id=bot_user_id
    )
    if not recent:
        return ()

    async def _entries() -> AsyncIterator[MessageHistoryEntry]:
        for m in recent:
            text = (m.get("text") or "").strip()
            if text:
                author = _author(m)
                # The id stands in for the name, and mention tokens stay raw,
                # until the budget has decided which entries are worth the
                # users.info lookups.
                yield MessageHistoryEntry(
                    username=preset.get(author, author), user_id=author, text=text
                )

    kept = await budget_history(_entries(), char_budget=CHAR_BUDGET)
    names = await _resolve_names(
        {e.user_id for e in kept if e.user_id and e.user_id not in preset},
        display_name,
    )
    return await _finalize(kept, names=names, strip_mentions=strip_mentions)


def _select_recent(
    tail: list[dict[str, Any]], *, exclude_ts: str, bot_user_id: str
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Newest-first messages minus the trigger and the bot's own posts, plus
    the on-message names for integration bots (no users.info entry).

    Our own posts are keyed by ``user``; a userless bot_message record of ours
    is identified by the ``bot_id`` those user-keyed posts carry."""
    own_bot_ids = {
        m["bot_id"] for m in tail if m.get("user") == bot_user_id and m.get("bot_id")
    }
    recent = [
        m
        for m in reversed(tail)
        if _author(m)
        and m.get("ts") != exclude_ts
        and m.get("user") != bot_user_id
        and m.get("bot_id") not in own_bot_ids
    ]
    preset = {
        m["bot_id"]: m.get("username")
        or (m.get("bot_profile") or {}).get("name")
        or m["bot_id"]
        for m in recent
        if not m.get("user")
    }
    return recent, preset


async def _finalize(
    kept: tuple[MessageHistoryEntry, ...],
    *,
    names: dict[str, str],
    strip_mentions: Callable[[str], Awaitable[str]],
) -> tuple[MessageHistoryEntry, ...]:
    entries = []
    for e in kept:
        text = await strip_mentions(e.text)
        if not text:
            continue
        uid = e.user_id or ""
        entries.append(
            MessageHistoryEntry(
                username=_label(names.get(uid, e.username)),
                user_id=e.user_id,
                text=text,
            )
        )
    return tuple(entries)


def _author(message: dict[str, Any]) -> str:
    return message.get("user") or message.get("bot_id") or ""


def _label(name: str) -> str:
    return " ".join(str(name).split())[:MAX_LABEL_LENGTH] or "user"


async def _resolve_names(
    user_ids: set[str], display_name: Callable[[str], Awaitable[str]]
) -> dict[str, str]:
    gate = asyncio.Semaphore(NAME_LOOKUP_CONCURRENCY)

    async def _lookup(uid: str) -> tuple[str, str]:
        async with gate:
            try:
                return uid, await display_name(uid)
            except Exception:
                # A failed lookup keeps the raw id rather than losing history.
                logger.debug("Slack name lookup failed for %s", uid, exc_info=True)
                return uid, uid

    return dict(await asyncio.gather(*(_lookup(u) for u in sorted(user_ids))))


async def _fetch_tail(
    client: AsyncWebClient, channel: str, thread_ts: str
) -> list[dict[str, Any]]:
    tail: deque[dict[str, Any]] = deque(maxlen=TAIL_SIZE)
    cursor: Optional[str] = None
    for page in range(MAX_PAGES):
        try:
            resp = await client.conversations_replies(
                channel=channel, ts=thread_ts, limit=PAGE_SIZE, cursor=cursor
            )
        except Exception:
            # slack_sdk raises SlackApiError for API errors but re-raises raw
            # aiohttp/asyncio errors for transport ones; neither may cost the
            # user their turn over optional context.
            logger.warning("Could not fetch Slack thread history", exc_info=True)
            return []
        messages = list(resp.get("messages") or [])
        if page == 0 and messages:
            # The parent's reply_count is on the first page: an over-cap thread
            # can bail after one round trip instead of paying for all pages.
            reply_count = int(messages[0].get("reply_count") or 0)
            if reply_count > PAGE_SIZE * MAX_PAGES:
                logger.warning(
                    "Slack thread %s has %d replies; skipping history",
                    thread_ts,
                    reply_count,
                )
                return []
        tail.extend(messages)
        cursor = (resp.get("response_metadata") or {}).get("next_cursor") or None
        if cursor is None:
            return list(tail)
    # Past the cap we'd be holding a stale middle window, not the recent end
    # the prompt promises — better no history than misleading history.
    logger.warning(
        "Slack thread %s ran past %d history pages; skipping history",
        thread_ts,
        MAX_PAGES,
    )
    return []
