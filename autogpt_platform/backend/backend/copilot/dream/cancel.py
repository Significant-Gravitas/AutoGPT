"""The stop path of a dream pass: cancel it, and notice a stop while it runs.

``cancel_dream_pass`` closes the pass's row CANCELLED in one conditional
transition that also bumps its cancel generation, and only while the row is
open and the caller's own. It does not reach into the running pass; the pass
reads its row at its next check and stops itself:

  * a sync pass at each phase boundary and once more just before apply
    (``stop_if_stopped``) ends with its failure result: the usage of the
    phases billed so far kept, nothing applied, its lock released on the way
    out like any other failure;
  * a batch pass in its callback, before it chains the next phase and before
    it claims the apply gate (``end_batch_pass_if_stopped``), cancels its
    provider batch best-effort and ends through ``fail_pass``: its landed
    phases charged, its lock released, its batch state and bundle cleaned.

A row a newer pass's guard expired (``guard.py``) has its generation bumped the
same way, so its pass stops at the same checks. A closed row takes no other
write, so the failure each route records on the way out leaves it CANCELLED
(or EXPIRED). The checks read the row under the store's deadline and never
stop a pass on a read that fails: the store being down must not end work that
nobody cancelled.
"""

import logging

from pydantic import BaseModel

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.dream_pass_models import DreamPassRecord

from .batch_outcome import BatchPass, fail_pass
from .pass_record import cancelled, stop_error
from .pass_run import DreamPassRun, PassEnded
from .provider_batch import cancel_provider_batch
from .store import read_dream_pass, read_open_passes, read_pass, write_stop

logger = logging.getLogger(__name__)


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
    back. Unlike a pass's own writes, a store call that fails or runs out of
    time raises: the caller is answering a request, or about to erase the
    memory the pass would write."""
    closed = await write_stop(pass_id, cancelled(reason, owner_user_id=user_id))
    record = await read_dream_pass(pass_id, user_id=user_id)
    if closed:
        logger.info(f"Dream pass {pass_id} cancelled: {reason}")
    return DreamPassCancel(cancelled=closed, record=record)


async def cancel_open_passes(
    scope: MemoryScope, *, user_id: str, reason: str
) -> list[DreamPassCancel]:
    """Cancel every open pass of *scope*, one ``cancel_dream_pass`` each, for
    the wipe to call before it erases the scope's memory. Raises when the
    store cannot list or cancel them, so the wipe does not go on unsure."""
    rows = await read_open_passes(scope)
    return [
        await cancel_dream_pass(row.id, user_id=user_id, reason=reason) for row in rows
    ]


async def stop_if_stopped(run: DreamPassRun) -> None:
    """Raise ``PassEnded`` with the run's failure when its row says it was
    stopped from outside; the sync pass's check before each phase and before
    apply."""
    row = await _read_row(run.pass_id)
    error = stop_error(row) if row is not None else None
    if error is None:
        return
    logger.info(f"Dream pass {run.pass_id} stops: {error}")
    raise PassEnded(run.failure(error))


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
