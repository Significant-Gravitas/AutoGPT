"""The stop path of a dream pass: cancel it, and notice a stop while it runs.

``cancel_dream_pass`` closes the pass's row CANCELLED in one conditional
transition that also bumps its cancel generation, and only while the row is
open and the caller's own. It does not reach into the running pass; the pass
reads its row at its next check and stops itself:

  * a sync pass at each phase boundary and once more just before apply
    (``stop_if_stopped``, ``stop_before_apply``) ends with its failure result:
    the usage of the phases billed so far kept, nothing applied, its lock
    released on the way out like any other failure;
  * a pass about to submit its first batch submits nothing, and one whose row
    refuses its submit cancels that batch instead of handing it on
    (``batch_handoff``);
  * a batch pass in its callback, before it chains the next phase and before
    it claims the apply gate (``end_batch_pass_if_stopped``), cancels its
    provider batch best-effort and ends through ``fail_pass``: its landed
    phases charged, its lock released, its batch state and bundle cleaned;
  * a batch pass waiting on its provider has that batch cancelled by the
    cancel itself, and the executor drops it at its next poll without
    dispatching it; the walker whose claim takes the entry off the queue
    ends the pass the same way (``batch_deliveries``).

A cancel that lands after a pass's last check, while it claims apply or
applies, is too late: that apply runs, once, and the row stays CANCELLED.

A row a newer pass's guard expired (``guard.py``) has its generation bumped the
same way, so its pass stops at the same checks. And right before apply each
route proves it still holds its scope's lock: the sync pass reads it
(``stop_before_apply``), then both renew it by compare-and-extend, failing
closed (``lease.admit_sync_apply``, ``lease.admit_batch_apply``). A newer
pass takes the scope only once this one's lock has lapsed, so a pass the
admission finds without it never applies over the newer one; one whose lock
lapses after its last renewal is ``lease.py``'s residual. A closed row takes
no other write, so the failure each route records on the way out leaves it
CANCELLED (or EXPIRED). The stop checks read the row under the store's
deadline and never stop a pass on a read that fails: the store being down
must not end work that nobody cancelled. The lock checks fail closed: a pass
that cannot confirm its lock does not apply.
"""

import logging
from collections.abc import Awaitable

from pydantic import BaseModel

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.dream_pass_models import OPEN_STATUSES, DreamPassRecord

from .batch_outcome import BatchPass, fail_pass
from .locks import DreamLockHandle
from .pass_record import cancelled, stop_error
from .pass_run import DreamPassRun, PassEnded
from .provider_batch import cancel_provider_batch
from .store import read_dream_pass, read_open_passes, read_pass, write_stop

logger = logging.getLogger(__name__)

# How a pass that lost its scope's lock before apply ends.
LOCK_LOST_ERROR = "apply: dream lock lost before apply"


class DreamPassCancel(BaseModel):
    """What a cancel did: ``record`` is the pass's row after it (``None`` when
    the user has no such pass) and ``cancelled`` whether this call closed it;
    a row that is there but was not closed had already ended."""

    cancelled: bool
    record: DreamPassRecord | None


async def cancel_dream_pass(
    pass_id: str, *, user_id: str, reason: str
) -> DreamPassCancel:
    """Cancel *user_id*'s pass *pass_id* while it is open, then read its row
    back; a batch it closed has its provider batch in flight cancelled too,
    best-effort (the executor drops that batch at its next poll whatever the
    provider says, see ``batch_deliveries.should_dispatch``). Unlike a pass's
    own writes, a store call that fails or runs out of time raises: the caller
    is answering a request, or about to erase the memory the pass would
    write."""
    closed = await write_stop(pass_id, cancelled(reason, owner_user_id=user_id))
    record = await read_dream_pass(pass_id, user_id=user_id)
    if closed:
        logger.info(f"Dream pass {pass_id} cancelled: {reason}")
    if closed and record is not None and record.provider_batch_id:
        await cancel_provider_batch(record.provider_batch_id)
    return DreamPassCancel(cancelled=closed, record=record)


async def cancel_open_passes(
    scope: MemoryScope, *, user_id: str, reason: str
) -> list[DreamPassCancel]:
    """Cancel every open pass of *scope*, one ``cancel_dream_pass`` each, for
    the wipe to call before it erases the scope's memory. Raises when the
    store cannot list or cancel them, so the wipe does not go on unsure.

    This closes the rows; it is not a barrier. Each pass stops at its next
    check, and one already past its last check still applies, so a wipe must
    also wait for the scope's lock before it erases."""
    rows = await read_open_passes(scope)
    return [
        await cancel_dream_pass(row.id, user_id=user_id, reason=reason) for row in rows
    ]


async def stopped_error(pass_id: str) -> str | None:
    """Why the pass's row says it was stopped from outside; ``None`` when it
    was not, or when the store cannot say in time and the pass goes on."""
    row = await _read_row(pass_id)
    return stop_error(row) if row is not None else None


async def pass_closed(pass_id: str) -> bool:
    """Whether the pass's row has ended; ``False`` when there is no row or the
    store cannot say in time."""
    row = await _read_row(pass_id)
    return row is not None and row.status not in OPEN_STATUSES


async def stop_if_stopped(run: DreamPassRun) -> None:
    """Raise ``PassEnded`` with the run's failure when its row says it was
    stopped from outside; the sync pass's check before each phase."""
    error = await stopped_error(run.pass_id)
    if error is None:
        return
    logger.info(f"Dream pass {run.pass_id} stops: {error}")
    raise PassEnded(run.failure(error))


async def stop_before_apply(run: DreamPassRun, lock_handle: DreamLockHandle) -> None:
    """The sync pass's last checks before apply: its row shows no stop and it
    still holds its scope's lock; else raise ``PassEnded`` with its failure."""
    await stop_if_stopped(run)
    if not await _lock_held(lock_handle.held(), run.pass_id):
        raise PassEnded(run.failure(LOCK_LOST_ERROR))


async def end_batch_pass_if_stopped(bp: BatchPass) -> bool:
    """End the batch pass when its row says it was stopped from outside, and
    say whether it did; the callback's check before it chains the next phase
    and before it claims the apply gate.

    The batch the row names is cancelled at the provider first. By the time a
    callback runs, that is usually the batch that just ended, which refuses;
    it matters when the row names a later one still in flight."""
    row = await _read_row(bp.pass_id)
    error = stop_error(row) if row is not None else None
    if row is None or error is None:
        return False
    logger.info(f"Dream batch pass {bp.pass_id} stops: {error}")
    if row.provider_batch_id:
        await cancel_provider_batch(row.provider_batch_id)
    await fail_pass(bp, error)
    return True


async def _lock_held(check: Awaitable[bool], pass_id: str) -> bool | None:
    """*check*'s answer, or ``None`` when Redis cannot give it in time: a pass
    that cannot confirm its lock does not apply."""
    try:
        return await check
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not confirm it still holds its dream lock",
            exc_info=True,
        )
        return None


async def _read_row(pass_id: str) -> DreamPassRecord | None:
    """The pass's row for a stop check, or ``None`` when there is none or the
    store cannot say in time, and the pass goes on."""
    try:
        return await read_pass(pass_id)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not read its row to check for a stop; "
            "going on",
            exc_info=True,
        )
        return None
