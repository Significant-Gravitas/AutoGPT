"""One trial read and one usage-state read per request, not one per check.

The chat route's paywall and its rate-limit checks each read the user's
trial and their usage state (tier, activation, trial id), and every read is a
database round trip (about 110 ms in prod). Inside ``trial_reads_shared()``
those checks share one read of each per user; outside it, every call reads
the database. The copilot executor holds a block from a turn's start until
the model's first output, so the allowance, budget and tier checks that set
up the turn share one read of each too.

The sharing is scoped to one block, never across requests or turns, so
nothing can go stale: neither is expected to change between checks a few
seconds apart, and a check that changes the usage state itself drops it from
the block first (``forget_usage_state``). Where a block waits long enough for
either to change, as a turn held for a fresh hire's setup does, it drops both
(``SharedTrialReads.forget``). A task started inside the block inherits the
block, but stops sharing when the block ends or is closed.

A tool never shares: it acts on the trial and the usage state as they are
when the tool runs, and a tool can start before the executor's block sees the
model's first output. Tools run inside ``trial_reads_fresh()``.
"""

import asyncio
import contextvars
import functools
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import ParamSpec, TypeVar

from backend.data.pro_activation import UsageActivationState
from backend.data.subscription_trial import TrialState

_P = ParamSpec("_P")
_R = TypeVar("_R")
_T = TypeVar("_T")


class SharedTrialReads:
    """The reads a ``trial_reads_shared`` block shares while it is open."""

    def __init__(self) -> None:
        self.reads: dict[str, asyncio.Future[TrialState | None]] = {}
        self.usage_states: dict[str, asyncio.Future[UsageActivationState]] = {}
        self.open = True

    def close(self) -> None:
        """Stop sharing: later lookups read the database again."""
        self.open = False

    def forget(self) -> None:
        """Drop the reads made so far: the next lookups read the database
        again, and the block shares those reads instead."""
        self.reads.clear()
        self.usage_states.clear()


_shared_reads: contextvars.ContextVar[SharedTrialReads | None] = contextvars.ContextVar(
    "copilot_shared_trial_reads", default=None
)


@contextmanager
def trial_reads_shared() -> Iterator[SharedTrialReads]:
    """Let the trial lookups inside the block share one read per user."""
    shared = SharedTrialReads()
    token = _shared_reads.set(shared)
    try:
        yield shared
    finally:
        # Tasks started in the block copied the context and still see
        # ``shared``; closing it sends their later lookups to the database.
        shared.close()
        _shared_reads.reset(token)


@contextmanager
def trial_reads_fresh() -> Iterator[None]:
    """Read the trial from the database inside the block, even within a shared one."""
    token = _shared_reads.set(None)
    try:
        yield
    finally:
        _shared_reads.reset(token)


def sharing_trial_reads(
    handler: Callable[_P, Awaitable[_R]],
) -> Callable[_P, Awaitable[_R]]:
    """Run each call of *handler* inside ``trial_reads_shared()``.

    For a request handler: the scope ends when it returns, before any response
    is streamed. ``functools.wraps`` keeps the signature FastAPI resolves.
    """

    @functools.wraps(handler)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with trial_reads_shared():
            return await handler(*args, **kwargs)

    return wrapper


async def shared_trial_read(
    user_id: str, read: Callable[[], Awaitable[TrialState | None]]
) -> TrialState | None:
    """``read()`` the user's trial, or join the read already made in this block."""
    shared = _shared_reads.get()
    if shared is None or not shared.open:
        return await read()
    return await _join(shared.reads, user_id, read)


async def shared_usage_state_read(
    user_id: str, read: Callable[[], Awaitable[UsageActivationState]]
) -> UsageActivationState:
    """``read()`` the user's usage state, or join the read already made in this
    block."""
    shared = _shared_reads.get()
    if shared is None or not shared.open:
        return await read()
    return await _join(shared.usage_states, user_id, read)


def forget_usage_state(user_id: str) -> None:
    """Drop the block's usage state for *user_id* after changing it, so the
    next lookup reads it again (and the block shares that read instead).

    A lookup already waiting on the dropped read still gets its result, as an
    unshared read made before the change would have."""
    shared = _shared_reads.get()
    if shared is not None:
        shared.usage_states.pop(user_id, None)


async def _join(
    reads: dict[str, asyncio.Future[_T]],
    user_id: str,
    read: Callable[[], Awaitable[_T]],
) -> _T:
    pending = reads.get(user_id)
    if pending is None:
        pending = asyncio.ensure_future(read())
        pending.add_done_callback(_retrieve_exception)
        reads[user_id] = pending
    # Shielded: one caller giving up must not cancel the read the others wait on.
    return await asyncio.shield(pending)


def _retrieve_exception(read: asyncio.Future[_T]) -> None:
    # If every caller gave up, nobody awaits a failed read; retrieving its
    # error here keeps asyncio from logging it as never retrieved.
    if not read.cancelled():
        read.exception()
