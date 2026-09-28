"""A dream pass's lease: its scope's Redis lock, and the expiry its DreamPass
row records for that lock, renewed together at every step.

The lock keeps two passes of a scope apart; the row's lease is what the guard
of a newer pass and the reaper read to tell a live pass from a dead one. A
renewal compare-and-extends the lock under the pass's token, which never
resurrects a lapsed key, then writes the new expiry to the row, bounded. The
row follows the lock only as far as those writes land: one that fails is
logged and the pass goes on with its row behind its lock (the reaper may then
take it for dead, ``reaper.py``), and apply's own renewal before its ingestion
drain (``apply.py``) shortens the sync lock without writing the row, which
then overstates the lock by up to twenty-two minutes.

  sync pass     written at its RUNNING insert (the token it is about to take
                the lock under, ``DEFAULT_LOCK_TTL_SECONDS`` from its start);
                renewed before each phase (``checkpoint``) and before apply
                (``admit_sync_apply``)
  batch pass    renewed at submit (``batch_handoff``, to
                ``BATCH_LOCK_TTL_SECONDS``), on every callback once its phase
                has landed (``renew_batch_lease``), and once its apply gate
                is claimed (``admit_batch_apply``)
  either        renewed by apply once more right before its first graph
                write (``ApplyLease``); emptied as the row closes, or, when
                the close marks the row for a cleanup (a stop, a batch
                pass's end), once that cleanup is done (``cleanup.py``)

A lock that is no longer the pass's ends the pass: another pass may hold the
scope now. A renewal Redis cannot answer in time is taken by what follows it.
Before a phase, or at a landed callback, the pass goes on, warned: nothing it
does before apply writes the graph, and apply's admission decides. The
renewals that admit apply fail closed: ownership unknown is not ownership,
and a pass that cannot prove it holds its lock does not write. A batch pass
renews under the token its input bundle carries, else the one its row keeps
(``batch_outcome.lock_token_of``); with none anywhere its ownership is
unknown, never "no lock": it does not apply, and its row stays marked until
the reaper can tell the lock is not its own.

What stays open: a renewal proves ownership at the instant Redis runs it and
keeps the lock one more TTL, not up to every write. A pass that stalls longer
than a TTL between its last renewal and its writes (a suspended process, a
reply held past the TTL) can still write after a newer pass took the scope;
apply's renewal right before its first write keeps that window to one TTL
from the writes. A cancel that lands after the last check is late
(``cancel.py``).
"""

import asyncio
import logging
from collections.abc import Awaitable

from pydantic import BaseModel, ConfigDict

from backend.copilot.graphiti.scope import MemoryScope

from .batch_outcome import BatchPass, fail_pass, lock_token_of
from .cancel import stop_before_apply, stop_if_stopped, stopped_error
from .locks import (
    BATCH_LOCK_TTL_SECONDS,
    DEFAULT_LOCK_TTL_SECONDS,
    LOCK_CHECK_TIMEOUT_SECONDS,
    DreamLockHandle,
    extend_dream_lock,
)
from .pass_run import DreamPassRun, PassEnded
from .store import record_lease

logger = logging.getLogger(__name__)


class ApplyLease(BaseModel):
    """The lease apply renews once more right before its first graph write
    (``apply.apply_operations``): the scope's lock compare-and-extended
    under the pass's token for its route's TTL, bounded, failing closed."""

    model_config = ConfigDict(frozen=True)

    scope: MemoryScope
    token: str
    ttl_seconds: int
    pass_id: str

    async def renew(self) -> bool:
        """Whether the lock is still the pass's and now runs ``ttl_seconds``
        more; ``False`` when it is not, and when Redis could not say in time."""
        extend = extend_dream_lock(self.scope, self.token, self.ttl_seconds)
        return await _renewed(extend, self.pass_id) is True


def lock_lost_error(step: str) -> str:
    """How a pass that lost its scope's lock before *step* ends."""
    return f"{step}: dream lock lost before {step}"


def lease_unknown_error(step: str) -> str:
    """How a pass ends whose lease could not be renewed before *step*, where
    ownership unknown fails closed."""
    return f"{step}: dream lease could not be renewed"


async def checkpoint(run: DreamPassRun, step: str) -> None:
    """The sync pass's check before each phase: its row shows no stop, then
    its lease is renewed (``renew_sync_lease``); raises ``PassEnded`` when
    either ends the pass."""
    await stop_if_stopped(run)
    await renew_sync_lease(run, step)


async def renew_sync_lease(run: DreamPassRun, step: str) -> None:
    """Renew the sync pass's lease before *step*; a renewal Redis cannot
    answer goes on, warned (a phase writes nothing; apply's admission fails
    closed). Raises ``PassEnded`` once the lock is no longer the pass's."""
    if run.lock is not None:
        await _renew_sync(run, run.lock, step, fail_closed=False)


async def admit_sync_apply(
    run: DreamPassRun, scope: MemoryScope, lock_handle: DreamLockHandle
) -> ApplyLease:
    """The sync pass's last checks before apply: its row shows no stop, its
    lock reads as its own (``stop_before_apply``) and its lease renews,
    failing closed. Returns the lease apply renews once more before it
    writes; raises ``PassEnded`` with the run's failure when it may not."""
    await stop_before_apply(run, lock_handle)
    await _renew_sync(run, lock_handle, "apply", fail_closed=True)
    return ApplyLease(
        scope=scope,
        token=lock_handle.token,
        ttl_seconds=DEFAULT_LOCK_TTL_SECONDS,
        pass_id=run.pass_id,
    )


async def renew_batch_lease(bp: BatchPass, next_step: str) -> bool:
    """On a callback whose phase has landed: stretch the batch pass's lock
    for another ``BATCH_LOCK_TTL_SECONDS`` (a whole batch lifetime) and
    record it on the row. ``False`` once the lock turned out to be no longer
    the pass's and the pass has been ended: its landed phases charged, its
    state and bundle deleted, the lock left to whoever holds it. The token
    is the one the bundle carries, else the one the row keeps
    (``lock_token_of``). A pass with no token anywhere, or whose renewal
    Redis could not answer, goes on unrenewed: apply's admission decides
    (``admit_batch_apply``)."""
    token = await lock_token_of(bp.pass_id)
    if token is None:
        return True
    scope = MemoryScope.build(bp.user_id, bp.expert_id)
    renewed = await _renewed(
        extend_dream_lock(scope, token, BATCH_LOCK_TTL_SECONDS), bp.pass_id
    )
    if renewed is False:
        error = await stopped_error(bp.pass_id) or lock_lost_error(next_step)
        logger.warning(f"Dream batch pass {bp.pass_id} stops: {error}")
        await fail_pass(bp, error, holds_lock=False)
        return False
    if renewed:
        await record_lease(bp.pass_id, token, BATCH_LOCK_TTL_SECONDS)
    return True


async def admit_batch_apply(bp: BatchPass, lock_token: str | None) -> ApplyLease | None:
    """The batch pass's fence once its apply gate is claimed, nothing durable
    awaited before it: compare-and-extend its lock under *lock_token* for a
    whole batch window. The extend is the proof: it finds the lock the
    pass's when Redis runs it and keeps it so for the window, where a read
    could return a stale "yours" whose reply crossed the lock's expiry.

    Fails closed: a lock that is no longer the pass's, one Redis could not
    answer for in time, and a pass with no token anywhere (its ownership
    unknown, not "no lock") all end the pass, its landed phases charged and
    its state and bundle cleaned. The lock is released by compare-and-delete,
    which only ever deletes the pass's own, unless it is known to be
    another's; with no token to release it by, the row keeps its mark and
    the reaper settles it. Returns the lease apply renews once more before it
    writes, or ``None`` once the pass has ended."""
    scope = MemoryScope.build(bp.user_id, bp.expert_id)
    renewed: bool | None = None
    if lock_token is not None:
        extend = extend_dream_lock(scope, lock_token, BATCH_LOCK_TTL_SECONDS)
        renewed = await _renewed(extend, bp.pass_id)
    if renewed and lock_token is not None:
        return ApplyLease(
            scope=scope,
            token=lock_token,
            ttl_seconds=BATCH_LOCK_TTL_SECONDS,
            pass_id=bp.pass_id,
        )
    error = await stopped_error(bp.pass_id) or _ended_error("apply", renewed)
    logger.warning(f"Dream batch pass {bp.pass_id} does not apply: {error}")
    await fail_pass(bp, error, holds_lock=renewed is not False)
    return None


async def _renew_sync(
    run: DreamPassRun, lock: DreamLockHandle, step: str, *, fail_closed: bool
) -> None:
    """Stretch the sync pass's lock another ``DEFAULT_LOCK_TTL_SECONDS`` and
    record it on its row. Raise ``PassEnded`` with the run's failure once
    the lock is no longer the pass's (the stop that closed its row if one
    did), and, when *fail_closed*, when Redis could not say."""
    renewed = await _renewed(lock.extend(DEFAULT_LOCK_TTL_SECONDS), run.pass_id)
    if renewed:
        await record_lease(run.pass_id, lock.token, DEFAULT_LOCK_TTL_SECONDS)
        return
    if renewed is None and not fail_closed:
        return
    error = await stopped_error(run.pass_id) or _ended_error(step, renewed)
    logger.warning(f"Dream pass {run.pass_id} stops: {error}")
    raise PassEnded(run.failure(error))


def _ended_error(step: str, renewed: bool | None) -> str:
    """Why a pass's renewal ended it: its lock lost, or ownership unknown."""
    return lock_lost_error(step) if renewed is False else lease_unknown_error(step)


async def _renewed(extend: Awaitable[bool], pass_id: str) -> bool | None:
    """Whether *extend* found the lock still the pass's and stretched it;
    ``None`` when Redis did not answer within ``LOCK_CHECK_TIMEOUT_SECONDS``
    (the caller decides whether the pass goes on)."""
    try:
        return await asyncio.wait_for(extend, timeout=LOCK_CHECK_TIMEOUT_SECONDS)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not renew its lease in time",
            exc_info=True,
        )
        return None
