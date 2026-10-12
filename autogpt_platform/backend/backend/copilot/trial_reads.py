"""One subscription-trial read per request instead of one per check.

The chat route's paywall and its rate-limit checks each read the user's
trial, and every read is a database round trip (about 110 ms in prod), so a
trial user's message used to wait on four identical reads in a row. Inside
``trial_reads_shared()`` those checks share one read; outside it, every call
reads the database as before.

The sharing is scoped to one block of one request, never across requests, so
nothing can go stale: a trial's state is not expected to change between
checks a few milliseconds apart. A task started inside the block inherits the
block, but stops sharing when the block ends.
"""

import asyncio
import contextvars
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager

from backend.data.subscription_trial import TrialState


class _SharedReads:
    """The reads a ``trial_reads_shared`` block shares while it is open."""

    def __init__(self) -> None:
        self.reads: dict[str, asyncio.Future[TrialState | None]] = {}
        self.open = True


_shared_reads: contextvars.ContextVar[_SharedReads | None] = contextvars.ContextVar(
    "copilot_shared_trial_reads", default=None
)


@contextmanager
def trial_reads_shared() -> Iterator[None]:
    """Let the trial lookups inside the block share one read per user."""
    shared = _SharedReads()
    token = _shared_reads.set(shared)
    try:
        yield
    finally:
        # Tasks started in the block copied the context and still see
        # ``shared``; closing it sends their later lookups to the database.
        shared.open = False
        _shared_reads.reset(token)


async def shared_trial_read(
    user_id: str, read: Callable[[], Awaitable[TrialState | None]]
) -> TrialState | None:
    """``read()`` the user's trial, or join the read already made in this block."""
    shared = _shared_reads.get()
    if shared is None or not shared.open:
        return await read()
    pending = shared.reads.get(user_id)
    if pending is None:
        pending = asyncio.ensure_future(read())
        shared.reads[user_id] = pending
    # Shielded: one caller giving up must not cancel the read the others wait on.
    return await asyncio.shield(pending)
