"""One subscription-trial read per request instead of one per check.

The chat route's paywall and its rate-limit checks each read the user's
trial, and every read is a database round trip (about 110 ms in prod), so a
trial user's message used to wait on four identical reads in a row. Inside
``trial_reads_shared()`` those checks share one read; outside it, every call
reads the database as before.

The sharing is scoped to one block of one request, never across requests, so
nothing can go stale: a trial's state is not expected to change between
checks a few milliseconds apart.
"""

import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager

from backend.data.subscription_trial import TrialState

_shared_reads: contextvars.ContextVar[
    "dict[str, asyncio.Future[TrialState | None]] | None"
] = contextvars.ContextVar("copilot_shared_trial_reads", default=None)


@contextmanager
def trial_reads_shared() -> Iterator[None]:
    """Let the trial lookups inside the block share one read per user."""
    token = _shared_reads.set({})
    try:
        yield
    finally:
        _shared_reads.reset(token)


async def shared_trial_read(
    user_id: str, read: Callable[[], Awaitable[TrialState | None]]
) -> TrialState | None:
    """``read()`` the user's trial, or join the read already made in this block."""
    reads = _shared_reads.get()
    if reads is None:
        return await read()
    shared = reads.get(user_id)
    if shared is None:
        shared = asyncio.ensure_future(read())
        reads[user_id] = shared
    # Shielded: one caller giving up must not cancel the read the others wait on.
    return await asyncio.shield(shared)
