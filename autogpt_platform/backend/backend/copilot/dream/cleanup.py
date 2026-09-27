"""The cleanup after a dream pass whose row has closed: what a batch pass
leaves behind (its provider batch, its landed phases' state and charges, its
lock), and the lock of a sync pass that stopped without giving it back.

Four steps, each idempotent and each saying whether it finished:

  provider  the batch the pass has in flight stopped at Anthropic: the cancel
            acknowledged, or the batch found ended
            (``provider_batch.provider_batch_stopped``)
  charge    every phase landed in the pass's Redis state charged once
            (``batch_costs.charge_landed_phases``); a state that cannot be
            read is not an empty one
  unlock    the pass's dream lock released by compare-and-delete on its
            token, or found no longer the pass's; with no token anywhere
            the pass's ownership is unknown, and the lock is left held and
            the step unfinished, until the reaper finds the lock gone or
            another open pass's
  delete    the state and the input bundle deleted, once every landed phase
            is charged: a charge left unfinished keeps what it needs

Every transition that may leave this cleanup behind marks the row for it
(``cleanupPendingAt``) and keeps its lease token (``pass_record.py``): a stop
from outside, a batch pass's end. Whoever cleans up after the pass, the pass
itself as it ends (``batch_outcome.clean_up_after``) or the reaper, clears
the mark only once every step has finished; until then the row stays on the
reaper's list, and its next run does every step again, the finished ones
doing nothing (``reaper.py``).
"""

import asyncio
import logging
from typing import Literal

from pydantic import BaseModel

from backend.copilot.graphiti.scope import MemoryScope

from .batch_costs import PhaseCharges, charge_landed_phases
from .batch_state import best_effort_cleanup, read_state
from .locks import LOCK_CHECK_TIMEOUT_SECONDS, read_dream_lock_token, release_dream_lock
from .provider_batch import provider_batch_stopped
from .schemas import DreamPhase
from .store import read_open_passes, record_cleanup_finished

logger = logging.getLogger(__name__)

CleanupStep = Literal["provider", "charge", "unlock", "delete"]


class PassCleanup(BaseModel):
    """What one cleanup did: the phases it charged, and the steps it could
    not finish, none once the cleanup is done."""

    charged: list[DreamPhase] = []
    unfinished: list[CleanupStep] = []

    @property
    def finished(self) -> bool:
        return not self.unfinished


async def clean_up_pass(
    pass_id: str,
    scope: MemoryScope,
    *,
    phase_models: dict[str, str] | None,
    provider_batch_id: str | None,
    lock_token: str | None,
    release: bool,
    attribute_tokenless: bool = False,
) -> PassCleanup:
    """Every step of the cleanup after pass *pass_id* of *scope*, each run
    whatever became of the others but the delete, which waits on the charge.
    Never raises.

    *provider_batch_id* names the batch to stop, if any; *phase_models*
    prices the landed phases (``None``: they cannot be priced, and the
    charge stays unfinished while any landed); the lock is released under
    *lock_token* only when *release*. Without a token the unlock stays
    unfinished, unless *attribute_tokenless* (the reaper, for a row that kept
    no token) and the lock is found not to be the pass's."""
    unfinished: list[CleanupStep] = []
    if provider_batch_id and not await provider_batch_stopped(provider_batch_id):
        unfinished.append("provider")
    charges = await _charge(pass_id, scope, phase_models)
    if not charges.settled:
        unfinished.append("charge")
    if release and not await _unlock(
        pass_id, scope, lock_token, attribute=attribute_tokenless
    ):
        unfinished.append("unlock")
    if not charges.settled or not await best_effort_cleanup(pass_id):
        unfinished.append("delete")
    return PassCleanup(charged=charges.charged, unfinished=unfinished)


async def finish_cleanup(pass_id: str, cleanup: PassCleanup) -> None:
    """Clear the row's mark once *cleanup* has finished every step; else
    leave it for the reaper, whose next run does the cleanup again. Never
    raises: a mark left behind costs the reaper a row, nothing more."""
    if not cleanup.finished:
        logger.warning(
            f"Dream pass {pass_id}: cleanup unfinished at "
            f"{', '.join(cleanup.unfinished)}; left marked for the reaper"
        )
        return
    try:
        await record_cleanup_finished(pass_id)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not clear its cleanup mark; the "
            "reaper will",
            exc_info=True,
        )


async def _charge(
    pass_id: str, scope: MemoryScope, phase_models: dict[str, str] | None
) -> PhaseCharges:
    """Charge the phases landed in the pass's state; not settled when the
    state cannot be read, or a landed phase cannot be priced or claimed."""
    try:
        state = await read_state(pass_id)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not read its batch state to charge it",
            exc_info=True,
        )
        return PhaseCharges(settled=False)
    if not state:
        return PhaseCharges(settled=True)
    if phase_models is None:
        logger.warning(f"Dream pass {pass_id}: no phase models to price it with")
        return PhaseCharges(settled=False)
    return await charge_landed_phases(
        user_id=scope.owner_user_id,
        expert_id=scope.expert_id,
        pass_id=pass_id,
        state=state,
        phase_models=phase_models,
    )


async def _unlock(
    pass_id: str, scope: MemoryScope, lock_token: str | None, *, attribute: bool
) -> bool:
    """Release the pass's lock by compare-and-delete under *lock_token*.
    Without a token the pass's ownership is unknown: the step finishes only
    when *attribute* and the lock is found not to be the pass's
    (``_lock_is_not_the_passes``); otherwise it stays unfinished and the
    lock is left alone."""
    if lock_token is not None:
        return await release_dream_lock(scope, lock_token)
    if not attribute:
        logger.warning(
            f"Dream pass {pass_id}: no lock token to release its lock with; "
            "left for the reaper"
        )
        return False
    return await _lock_is_not_the_passes(pass_id, scope)


async def _lock_is_not_the_passes(pass_id: str, scope: MemoryScope) -> bool:
    """Whether the scope's lock is surely not the tokenless pass's: gone, or
    held under the lease token of another open pass of the scope. Never
    deletes: a lock held under a token no open pass keeps may be the pass's
    own, so it is left to lapse on its TTL (24 h 10 min at most for a batch
    lock, 30 min for a sync one), and the step stays unfinished until then."""
    try:
        holder = await asyncio.wait_for(
            read_dream_lock_token(scope), timeout=LOCK_CHECK_TIMEOUT_SECONDS
        )
        if holder is None:
            return True
        others = await read_open_passes(scope)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not tell whose its scope's lock is",
            exc_info=True,
        )
        return False
    if any(row.id != pass_id and row.lease_token == holder for row in others):
        return True
    logger.warning(
        f"Dream pass {pass_id}: its scope's lock is held under a token no open "
        "pass keeps; left to lapse on its TTL"
    )
    return False
