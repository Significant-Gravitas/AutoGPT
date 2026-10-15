"""The cleanup after a dream pass whose row has closed: what a batch pass
leaves behind (its provider batch, its landed phases' state and charges, its
lock), and the lock of a sync pass that stopped without giving it back.

Four steps, in this order, each idempotent, each bounded, and each saying
whether it finished:

  charge    every phase landed in the pass's Redis state charged once,
            under the usage attribution its row carries and never another
            (``batch_costs.charge_landed_phases``); a state that cannot be
            read is not an empty one, and one that is gone while the row
            says phases landed is usage lost, logged as an error
  unlock    the pass's dream lock released by compare-and-delete on its
            token, or found no longer the pass's; with no token anywhere
            the pass's ownership is unknown, and the lock is left held and
            the step unfinished, until the reaper finds the lock gone or
            another open pass's
  delete    the state and the input bundle deleted, once every landed phase
            is charged: a charge left unfinished keeps what it needs
  provider  the batch the pass has in flight stopped at Anthropic: the cancel
            acknowledged, or the batch found ended
            (``provider_batch.provider_batch_stopped``)

The charge goes first because the state it reads lapses. The provider goes
last because it is the slowest (two calls of up to
``PROVIDER_CANCEL_TIMEOUT_SECONDS``, spread over two cleanups when they do
not fit one, ``provider_batch.py``) and none of the other steps needs it
done first: the executor drops a closed pass's batch unpolled whether or not
Anthropic has stopped it. A provider that hangs then costs a cleanup its
last step, retried, and not the release of the user's lock.

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
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel

from backend.copilot.graphiti.scope import MemoryScope

from .batch_costs import PhaseCharges, charge_landed_phases
from .batch_state import (
    best_effort_cleanup,
    read_state,
    recorded_phase_models,
    uncharged_phases,
)
from .locks import LOCK_CHECK_TIMEOUT_SECONDS, read_dream_lock_token, release_dream_lock
from .phase_jobs import PHASE_TIERS
from .provider_batch import provider_batch_stopped
from .schemas import DreamPhase
from .store import read_open_passes, record_cleanup_finished

logger = logging.getLogger(__name__)

CleanupStep = Literal["charge", "unlock", "delete", "provider"]

# How long reading or deleting a pass's Redis state may take, the client's
# retries included: a stalled Redis costs a cleanup seconds, not its turn.
STATE_STEP_TIMEOUT_SECONDS = 5.0
# How long the cleanup waits on its landed phases' charge. The charge goes on
# without it, in a task of its own (``batch_costs.charge_landed_phases``),
# claiming for ``batch_costs.CLAIM_WINDOW_SECONDS``; what it does not reach
# waits for the next cleanup.
CHARGE_STEP_TIMEOUT_SECONDS = 10.0


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
    landed: Sequence[DreamPhase] = (),
) -> PassCleanup:
    """Every step of the cleanup after pass *pass_id* of *scope*, in order,
    each run whatever became of the others but the delete, which waits on
    the charge. Each is bounded: the state read and delete by
    ``STATE_STEP_TIMEOUT_SECONDS``, the charges by
    ``CHARGE_STEP_TIMEOUT_SECONDS``, each lock call by
    ``LOCK_CHECK_TIMEOUT_SECONDS``, each provider call by
    ``PROVIDER_CANCEL_TIMEOUT_SECONDS``. Never raises.

    *provider_batch_id* names the batch to stop, if any; *phase_models*
    prices the landed phases (``None``: they cannot be priced, and the
    charge stays unfinished while any landed); the lock is released under
    *lock_token* only when *release*. Without a token the unlock stays
    unfinished, unless *attribute_tokenless* (the reaper, for a row that kept
    no token) and the lock is found not to be the pass's. *landed* are the
    phases the pass's row says landed: should their state be gone before
    they were charged, that usage is reported lost (an error), not taken for
    nothing to charge. Each phase is charged under the usage attribution
    the pass started under, read off its row (``batch_costs``); a row
    without one is charged nothing, and the state goes all the same."""
    unfinished: list[CleanupStep] = []
    charges = await _charge(pass_id, scope, phase_models, landed)
    if not charges.settled:
        unfinished.append("charge")
    if release and not await _unlock(
        pass_id, scope, lock_token, attribute=attribute_tokenless
    ):
        unfinished.append("unlock")
    if not charges.settled or not await _deleted(pass_id):
        unfinished.append("delete")
    if provider_batch_id and not await provider_batch_stopped(provider_batch_id):
        unfinished.append("provider")
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
    pass_id: str,
    scope: MemoryScope,
    phase_models: dict[str, str] | None,
    landed: Sequence[DreamPhase],
) -> PhaseCharges:
    """Charge the phases landed in the pass's state, each on the model its
    state row recorded, else the one *phase_models* names; not settled when
    the state cannot be read, a landed phase has no model to price it with
    or cannot be claimed, or the charges outlast
    ``CHARGE_STEP_TIMEOUT_SECONDS``."""
    try:
        async with asyncio.timeout(STATE_STEP_TIMEOUT_SECONDS):
            state = await read_state(pass_id)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not read its batch state to charge it",
            exc_info=True,
        )
        return PhaseCharges(settled=False)
    if not state:
        await _report_lost_usage(pass_id, scope, landed)
        return PhaseCharges(settled=True)
    models = {**(phase_models or {}), **recorded_phase_models(state)}
    unpriced = [
        phase
        for phase in PHASE_TIERS
        if (row := state.get(phase)) and not row.get("error") and phase not in models
    ]
    if unpriced:
        logger.warning(
            f"Dream pass {pass_id}: no model to price its {', '.join(unpriced)} "
            "phase with"
        )
        return PhaseCharges(settled=False)
    try:
        async with asyncio.timeout(CHARGE_STEP_TIMEOUT_SECONDS):
            return await charge_landed_phases(
                user_id=scope.owner_user_id,
                expert_id=scope.expert_id,
                pass_id=pass_id,
                state=state,
                phase_models=models,
            )
    except TimeoutError:
        logger.warning(
            f"Dream pass {pass_id}: its charge outlasted "
            f"{CHARGE_STEP_TIMEOUT_SECONDS:g} s; it goes on on its own, and "
            "what it does not reach is left for the next cleanup"
        )
        return PhaseCharges(settled=False)


async def _report_lost_usage(
    pass_id: str, scope: MemoryScope, landed: Sequence[DreamPhase]
) -> None:
    """An error naming the landed phases whose batch state lapsed before
    anyone charged them: their usage cannot be recovered. Nothing more can
    be done for them, so the cleanup goes on as settled."""
    if not landed:
        return
    lost = await uncharged_phases(pass_id, list(landed))
    if lost:
        logger.error(
            f"Dream pass {pass_id} of user {scope.owner_user_id} "
            f"(expert {scope.expert_id or 'none'}): the batch state of landed "
            f"phase(s) {', '.join(lost)} lapsed before they were charged; their "
            "usage cannot be recovered"
        )


async def _deleted(pass_id: str) -> bool:
    """The state and input bundle deleted (``best_effort_cleanup``), within
    ``STATE_STEP_TIMEOUT_SECONDS``."""
    try:
        async with asyncio.timeout(STATE_STEP_TIMEOUT_SECONDS):
            return await best_effort_cleanup(pass_id)
    except TimeoutError:
        logger.warning(f"Dream pass {pass_id}: deleting its batch state timed out")
        return False


async def _unlock(
    pass_id: str, scope: MemoryScope, lock_token: str | None, *, attribute: bool
) -> bool:
    """Release the pass's lock by compare-and-delete under *lock_token*,
    within ``LOCK_CHECK_TIMEOUT_SECONDS``. Without a token the pass's
    ownership is unknown: the step finishes only when *attribute* and the
    lock is found not to be the pass's (``_lock_is_not_the_passes``);
    otherwise it stays unfinished and the lock is left alone."""
    if lock_token is not None:
        try:
            async with asyncio.timeout(LOCK_CHECK_TIMEOUT_SECONDS):
                return await release_dream_lock(scope, lock_token)
        except TimeoutError:
            logger.warning(f"Dream pass {pass_id}: releasing its lock timed out")
            return False
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
        async with asyncio.timeout(LOCK_CHECK_TIMEOUT_SECONDS):
            holder = await read_dream_lock_token(scope)
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
