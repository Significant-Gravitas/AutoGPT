"""Bounded paging over a Conductor session transcript.

`GET /sessions/{id}/messages` serves ascending rows in pages of at most
PAGE_SIZE, paged either from the start with `offset` or forward from a row id
with `after` (the two cannot be combined) and never from the end. These
helpers read a bounded slice either way, and every request can be capped by
the caller's remaining wall-clock budget.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ._api import PAGE_SIZE, ConductorClient

Rows = list[dict[str, Any]]
# Returns the seconds left before the caller's deadline; None means unbounded.
Remaining = Callable[[], float] | None

# A request started close to the deadline still gets this long to answer, so
# the final poll of a wait is a real one rather than an instant timeout.
REQUEST_TIMEOUT_FLOOR_SECONDS = 1.0
# Upper bound on the exponential search for the transcript end (2**40 rows).
MAX_PROBES = 40


async def fetch_after(
    client: ConductorClient,
    session_id: str,
    after: str,
    count: int,
    remaining: Remaining = None,
) -> tuple[Rows, bool]:
    """Up to `count` rows after the `after` cursor, oldest first.

    Also reports whether more rows follow the returned slice (either because
    the server has more or because `count`/the deadline stopped the read).
    """
    rows: Rows = []
    cursor = after
    while True:
        page = await bounded(
            client.list_messages(
                session_id, after=cursor, limit=min(PAGE_SIZE, count - len(rows))
            ),
            remaining,
        )
        data = list(page.get("data") or [])
        rows.extend(data)
        has_more = bool(page.get("hasMore")) and bool(data)
        cursor = str(data[-1].get("id") or "") if data else ""
        if not has_more or not cursor or len(rows) >= count or expired(remaining):
            return rows, has_more


async def fetch_tail(
    client: ConductorClient,
    session_id: str,
    count: int,
    remaining: Remaining = None,
) -> tuple[Rows, bool]:
    """The newest `count` rows, oldest first, and whether older rows exist.

    A transcript that fits in one page costs one request. Otherwise the end is
    located with `limit=1` probes (exponential then binary search, stopping
    once the bracket is no wider than `count`) and the slice is read from
    just before it.
    """
    first = await bounded(
        client.list_messages(session_id, offset=0, limit=min(count, PAGE_SIZE)),
        remaining,
    )
    rows = list(first.get("data") or [])
    if not first.get("hasMore"):
        return rows, False
    low, high = await _bracket_end(client, session_id, len(rows), count, remaining)
    start = max(0, low + 1 - count)
    rows = await _fetch_from(client, session_id, start, high - start, remaining)
    return rows[-count:], start + len(rows) > count


async def bounded(
    coro: Awaitable[dict[str, Any]], remaining: Remaining
) -> dict[str, Any]:
    """Await one request, capped by the remaining budget when there is one."""
    if remaining is None:
        return await coro
    timeout = max(remaining(), REQUEST_TIMEOUT_FLOOR_SECONDS)
    return await asyncio.wait_for(coro, timeout=timeout)


def expired(remaining: Remaining) -> bool:
    return remaining is not None and remaining() <= 0


async def _bracket_end(
    client: ConductorClient,
    session_id: str,
    low: int,
    width: int,
    remaining: Remaining,
) -> tuple[int, int]:
    """Find (low, high) with row `low` existing and row `high` not, such that
    high - low <= width. `low` must be a row index known to exist."""
    high = 0
    probe = max(2 * low, 1)
    for _ in range(MAX_PROBES):
        exists, more = await _probe(client, session_id, probe, remaining)
        if exists and not more:
            return probe, probe + 1
        if not exists:
            high = probe
            break
        low, probe = probe, probe * 2
    if high == 0:
        raise ValueError("transcript is too long to locate its end")
    while high - low > width:
        mid = (low + high) // 2
        exists, more = await _probe(client, session_id, mid, remaining)
        if exists and not more:
            return mid, mid + 1
        if exists:
            low = mid
        else:
            high = mid
    return low, high


async def _probe(
    client: ConductorClient, session_id: str, offset: int, remaining: Remaining
) -> tuple[bool, bool]:
    """Whether a row exists at `offset` and whether any row follows it."""
    page = await bounded(
        client.list_messages(session_id, offset=offset, limit=1), remaining
    )
    return bool(page.get("data")), bool(page.get("hasMore"))


async def _fetch_from(
    client: ConductorClient,
    session_id: str,
    offset: int,
    count: int,
    remaining: Remaining,
) -> Rows:
    """Up to `count` rows starting at `offset`: one offset page, then cursors."""
    first = await bounded(
        client.list_messages(session_id, offset=offset, limit=min(PAGE_SIZE, count)),
        remaining,
    )
    rows = list(first.get("data") or [])
    cursor = str(rows[-1].get("id") or "") if rows else ""
    if not first.get("hasMore") or not cursor or len(rows) >= count:
        return rows
    more, _ = await fetch_after(
        client, session_id, cursor, count - len(rows), remaining
    )
    return rows + more
