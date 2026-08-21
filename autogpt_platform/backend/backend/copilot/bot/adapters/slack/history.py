"""Thread history for the Slack adapter.

A first @-mention into a thread the bot doesn't own pulls the thread's recent
messages into the prompt. Slack's ``conversations.replies`` pages oldest-first
from the parent, so the fetch follows the cursor to the end and keeps only the
tail, which the shared char budget then trims newest-first.
"""

import asyncio
import logging
from collections import deque
from collections.abc import Awaitable, Callable
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
    tail = await _fetch_tail(client, channel, thread_ts)
    # Newest-first: budget_history keeps the recent end.
    recent = [
        m
        for m in reversed(tail)
        if _author(m) and m.get("ts") != exclude_ts and m.get("user") != bot_user_id
    ]
    if not recent:
        return ()
    # Integration bots (subtype bot_message) have no users.info entry; their
    # name rides on the message itself.
    preset = {
        m["bot_id"]: m.get("username")
        or (m.get("bot_profile") or {}).get("name")
        or m["bot_id"]
        for m in recent
        if not m.get("user")
    }

    async def _entries():
        for m in recent:
            text = await strip_mentions(m.get("text") or "")
            if text:
                author = _author(m)
                # The id stands in for the name until the budget has decided
                # which entries are worth a users.info lookup.
                yield MessageHistoryEntry(
                    username=preset.get(author, author), user_id=author, text=text
                )

    kept = await budget_history(_entries(), char_budget=CHAR_BUDGET)
    names = await _resolve_names(
        {e.user_id for e in kept if e.user_id and e.user_id not in preset},
        display_name,
    )
    return tuple(
        MessageHistoryEntry(
            username=names.get(e.user_id or "", e.username),
            user_id=e.user_id,
            text=e.text,
        )
        for e in kept
    )


def _author(message: dict[str, Any]) -> str:
    return message.get("user") or message.get("bot_id") or ""


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
    for _ in range(MAX_PAGES):
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
        tail.extend(resp.get("messages") or [])
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
