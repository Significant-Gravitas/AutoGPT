"""Whether a token's subject asked to delete their account.

Tokens are verified statelessly and outlive the sessions a close ends, so the
auth dependencies ask on every request and reject a closed account
themselves. The answers are cached briefly in this process.
"""

import asyncio
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from enum import Enum

from fastapi import HTTPException

logger = logging.getLogger(__name__)

ACCOUNT_CLOSED_DETAIL = "This account is closed."


class AccountState(Enum):
    OPEN = "open"
    CLOSED = "closed"
    # The check failed: the account is not rejected, but never healed either,
    # since it may be closed.
    UNKNOWN = "unknown"


# Recent answers to "did this user delete their account?", id -> (state,
# monotonic expiry). Tokens outlive the sessions a close ends, so every request
# asks. Every answer is kept only briefly: another worker can close the
# account at any moment, and its owner can reopen it within 14 days. A failed
# check is kept as long, so a DB outage costs one query per user per TTL.
_CLOSED_ACCOUNT_CACHE: "OrderedDict[str, tuple[AccountState, float]]" = OrderedDict()
_CLOSED_ACCOUNT_CACHE_MAX = 10_000
_ACCOUNT_STATE_TTL_SECS = 30
# Lookups still running, id -> task: a page load's burst of requests for one
# user waits on the same query instead of each running its own.
_LOOKUPS_IN_FLIGHT: "dict[str, asyncio.Task[AccountState]]" = {}


def remember_account_closed(user_id: str) -> None:
    """Reject *user_id* in this process at once.

    For the worker that just closed the account: without this it would keep
    accepting the user until its cached "open" answer expired.
    """
    _cache_account_state(user_id, AccountState.CLOSED)


def forget_account_state(user_id: str) -> None:
    """Drop this process's cached answer for *user_id*.

    For the worker that just reopened the account, so it lets the owner back
    in at once, and for test teardown.
    """
    _CLOSED_ACCOUNT_CACHE.pop(user_id, None)


async def reject_closed_account(user_id: str) -> AccountState:
    """*user_id*'s account state; HTTP 401 when the account is closed."""
    state = await get_account_state(user_id)
    if state is AccountState.CLOSED:
        raise HTTPException(status_code=401, detail=ACCOUNT_CLOSED_DETAIL)
    return state


async def get_account_state(user_id: str) -> AccountState:
    """Whether *user_id* asked to delete their account, cached briefly.

    Open without the backend package or a connected DB. On a DB error the
    account is not rejected (the request's own next query would fail anyway),
    but it is reported UNKNOWN so nothing heals an account that may be closed.
    """
    cached = _cached_account_state(user_id)
    if cached is not None:
        return cached

    lookup = _LOOKUPS_IN_FLIGHT.get(user_id)
    # A lookup left behind by another event loop can't be awaited from this one.
    if lookup is None or lookup.get_loop() is not asyncio.get_running_loop():
        query = _closed_account_query()
        if query is None:
            return AccountState.OPEN
        lookup = asyncio.ensure_future(_settle_account_state(user_id, query))
        _LOOKUPS_IN_FLIGHT[user_id] = lookup
        lookup.add_done_callback(lambda done: _forget_lookup(user_id, done))
    # Shielded: a request that is cancelled must not cancel the lookup that
    # the other requests for this user are waiting on.
    return await asyncio.shield(lookup)


def _forget_lookup(user_id: str, lookup: "asyncio.Future[AccountState]") -> None:
    if _LOOKUPS_IN_FLIGHT.get(user_id) is lookup:
        del _LOOKUPS_IN_FLIGHT[user_id]


async def _settle_account_state(
    user_id: str, query: Callable[[str], Awaitable[bool]]
) -> AccountState:
    try:
        closed = await query(user_id)
        state = AccountState.CLOSED if closed else AccountState.OPEN
    except Exception:
        logger.warning(
            f"Closed-account check failed for {user_id}; treating it as open",
            exc_info=True,
        )
        state = AccountState.UNKNOWN

    # A close remembered while the query ran wins over the query's answer.
    if _cached_account_state(user_id) is AccountState.CLOSED:
        return AccountState.CLOSED
    _cache_account_state(user_id, state)
    return state


def _closed_account_query() -> Callable[[str], Awaitable[bool]] | None:
    """The backend's closed-account query, or None without the backend
    package or a connected DB (route unit tests, tooling)."""
    try:
        from backend.data.db import prisma  # deferred -- only needed at runtime
    except ImportError:
        return None
    if not prisma.is_connected():
        return None
    try:
        from backend.data.account_deletion import is_account_closed  # deferred
    except ImportError:
        return None
    return is_account_closed


def _cached_account_state(user_id: str) -> AccountState | None:
    entry = _CLOSED_ACCOUNT_CACHE.get(user_id)
    if entry is None:
        return None
    state, expires_at = entry
    if expires_at <= time.monotonic():
        del _CLOSED_ACCOUNT_CACHE[user_id]
        return None
    return state


def _cache_account_state(user_id: str, state: AccountState) -> None:
    _CLOSED_ACCOUNT_CACHE[user_id] = (
        state,
        time.monotonic() + _ACCOUNT_STATE_TTL_SECS,
    )
    _CLOSED_ACCOUNT_CACHE.move_to_end(user_id)
    while len(_CLOSED_ACCOUNT_CACHE) > _CLOSED_ACCOUNT_CACHE_MAX:
        _CLOSED_ACCOUNT_CACHE.popitem(last=False)
