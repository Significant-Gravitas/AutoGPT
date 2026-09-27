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
            token, or found no longer the pass's; with no token to release
            it with, the lock is left held and the step unfinished
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

import logging
from typing import Literal

from pydantic import BaseModel

from backend.copilot.graphiti.scope import MemoryScope

from .batch_costs import PhaseCharges, charge_landed_phases
from .batch_state import best_effort_cleanup, read_state
from .locks import release_dream_lock
from .provider_batch import provider_batch_stopped
from .schemas import DreamPhase
from .store import record_cleanup_finished

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
) -> PassCleanup:
    """Every step of the cleanup after pass *pass_id* of *scope*, each run
    whatever became of the others but the delete, which waits on the charge.
    Never raises.

    *provider_batch_id* names the batch to stop, if any; *phase_models*
    prices the landed phases (``None``: they cannot be priced, and the
    charge stays unfinished while any landed); the lock is released under
    *lock_token* only when *release*."""
    unfinished: list[CleanupStep] = []
    if provider_batch_id and not await provider_batch_stopped(provider_batch_id):
        unfinished.append("provider")
    charges = await _charge(pass_id, scope, phase_models)
    if not charges.settled:
        unfinished.append("charge")
    if release and not await release_dream_lock(scope, lock_token):
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
