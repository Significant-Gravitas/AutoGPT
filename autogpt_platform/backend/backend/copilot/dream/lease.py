"""A dream pass's lease: its scope's Redis lock, and the expiry its DreamPass
row records for that lock, renewed together at every step.

The lock keeps two passes of a scope apart; the row's lease is what the guard
of a newer pass and the reaper read to tell a live pass from a dead one. So
the lease always equals the lock's TTL from its last renewal, and a pass that
is alive never outlives it:

  sync pass     written at its RUNNING insert (the token it is about to take
                the lock under, ``DEFAULT_LOCK_TTL_SECONDS`` from its start);
                renewed before each phase (``checkpoint``) and before apply
                (``renew_sync_lease``)
  batch pass    renewed at submit (``batch_handoff``, to
                ``BATCH_LOCK_TTL_SECONDS``) and on every callback once its
                phase has landed (``renew_batch_lease``)
  either        emptied by every transition that closes the row

A renewal compare-and-extends the lock under the pass's token, which never
resurrects a lapsed key, then writes the new expiry to the row, bounded; a row
write that fails is logged and the pass goes on. A lock that is no longer the
pass's ends the pass: another pass may hold the scope now, so this one must
not start another phase or apply. A renewal Redis cannot answer in time goes
on, warned: the check right before apply fails closed (``cancel.py``).

Apply's own renewal before its ingestion drain (``apply.py``) shortens the
sync lock to ``LOCK_DRAIN_RENEWAL_SECONDS`` without writing the row, so for
that tail the row's lease can run past the lock by up to twenty-two minutes:
a newer pass's guard waits that much longer, and nothing runs twice.
"""

import asyncio
import logging
from collections.abc import Awaitable

from backend.copilot.graphiti.scope import MemoryScope

from .batch_outcome import BatchPass, fail_pass
from .batch_submit import read_lock_token
from .cancel import stop_if_stopped, stopped_error
from .locks import (
    BATCH_LOCK_TTL_SECONDS,
    DEFAULT_LOCK_TTL_SECONDS,
    LOCK_CHECK_TIMEOUT_SECONDS,
    extend_dream_lock,
)
from .pass_run import DreamPassRun, PassEnded
from .store import record_lease

logger = logging.getLogger(__name__)


def lock_lost_error(step: str) -> str:
    """How a pass that lost its scope's lock before *step* ends."""
    return f"{step}: dream lock lost before {step}"


async def checkpoint(run: DreamPassRun, step: str) -> None:
    """The sync pass's check before each phase: its row shows no stop, then
    its lease is renewed; raises ``PassEnded`` when either ends the pass."""
    await stop_if_stopped(run)
    await renew_sync_lease(run, step)


async def renew_sync_lease(run: DreamPassRun, step: str) -> None:
    """Stretch the sync pass's lock for another ``DEFAULT_LOCK_TTL_SECONDS``
    and record it on its row. Raise ``PassEnded`` with the run's failure once
    the lock is no longer the pass's: the stop that closed its row if one
    did, else ``lock_lost_error(step)``."""
    lock = run.lock
    if lock is None:
        return
    renewed = await _renewed(lock.extend(DEFAULT_LOCK_TTL_SECONDS), run.pass_id)
    if renewed is False:
        error = await stopped_error(run.pass_id) or lock_lost_error(step)
        logger.warning(f"Dream pass {run.pass_id} stops: {error}")
        raise PassEnded(run.failure(error))
    if renewed:
        await record_lease(run.pass_id, lock.token, DEFAULT_LOCK_TTL_SECONDS)


async def renew_batch_lease(bp: BatchPass, next_step: str) -> bool:
    """On a callback whose phase has landed: stretch the batch pass's lock
    for another ``BATCH_LOCK_TTL_SECONDS`` (a whole batch lifetime) and
    record it on the row. ``False`` once the lock turned out to be no longer
    the pass's and the pass has been ended: its landed phases charged, its
    state and bundle deleted, the lock left to whoever holds it. A pass with
    no token to renew under goes on unrenewed; the apply fence decides."""
    token = await _lock_token(bp.pass_id)
    if token is None:
        return True
    scope = MemoryScope.build(bp.user_id, bp.expert_id)
    extend = extend_dream_lock(scope, token, BATCH_LOCK_TTL_SECONDS)
    renewed = await _renewed(extend, bp.pass_id)
    if renewed is False:
        error = await stopped_error(bp.pass_id) or lock_lost_error(next_step)
        logger.warning(f"Dream batch pass {bp.pass_id} stops: {error}")
        await fail_pass(bp, error, holds_lock=False)
        return False
    if renewed:
        await record_lease(bp.pass_id, token, BATCH_LOCK_TTL_SECONDS)
    return True


async def _renewed(extend: Awaitable[bool], pass_id: str) -> bool | None:
    """Whether *extend* found the lock still the pass's and stretched it;
    ``None`` when Redis did not answer within ``LOCK_CHECK_TIMEOUT_SECONDS``
    and the pass goes on."""
    try:
        return await asyncio.wait_for(extend, timeout=LOCK_CHECK_TIMEOUT_SECONDS)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not renew its lease; going on",
            exc_info=True,
        )
        return None


async def _lock_token(pass_id: str) -> str | None:
    """The token the batch pass holds its lock under, kept with its input
    bundle; ``None`` when there is none or Redis cannot say."""
    try:
        return await read_lock_token(pass_id)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not read its lock token to renew it",
            exc_info=True,
        )
        return None
