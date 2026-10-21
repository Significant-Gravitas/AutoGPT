"""Hands a gathered dream pass to Anthropic's Message Batches API.

The sync entry point runs this when routing picked ``anthropic_batch``: it
persists the input bundle, submits phase 1 (consolidate) and returns at once.
The BatchExecutor polls; ``batch_callbacks`` submits phases 2 (recombine) and
3 (sanitize) as each prior result lands, then applies and closes the pass.
The dream lock goes with the pass: extended to the batch window here and
released by the callbacks when the pass ends.

A pass whose row was cancelled (or expired) while it gathered submits
nothing; one whose row refuses its submit (closed meanwhile, or gone), or
does not hold it (the write failed, and a read does not find the batch on
the row), takes that batch back here and keeps its lock to release on the
way out, rather than handing both to callbacks that would only stop, or
to a reaper that would take the pass for dead while its batch ran: the
batch's lease is the submit's to record.

Any caller can take this route (it depends on the flag and the deployment's
key, not the caller). ``status_id`` ties the callbacks' updates to the
JobStatus row an admin trigger created; without one the callbacks skip the
JobStatus writes, and apply and cost logging still run.
"""

import logging

from backend.copilot.config import ChatConfig
from backend.executor.batch_executor import remove_pending

from .batch_state import best_effort_cleanup
from .batch_submit import persist_input_bundle, phase_models_for_config, submit_phase
from .cancel import stopped_error
from .fetch import DreamInput
from .locks import BATCH_LOCK_TTL_SECONDS, DreamLockHandle
from .pass_errors import pass_error
from .pass_record import stop_error
from .pass_run import DreamPassRun
from .provider_batch import cancel_provider_batch
from .schemas import DreamPassResult
from .store import read_pass, record_submitted, record_taken_back

logger = logging.getLogger(__name__)

# How a pass ends whose row refused its submit and cannot say why.
REFUSED_SUBMIT_ERROR = pass_error("submit", "row_refused")
# How a pass ends whose submit write failed and whose row, read again, does
# not hold its batch, or cannot be read.
UNRECORDED_SUBMIT_ERROR = pass_error("submit", "not_on_record")


async def submit_dream_pass_batch(
    run: DreamPassRun,
    *,
    config: ChatConfig,
    input_bundle: DreamInput,
    status_id: str | None,
    lock_handle: DreamLockHandle,
) -> DreamPassResult:
    """Submit phase 1 and hand the pass, and its lock, to the callbacks."""
    api_key = config.direct_anthropic_api_key
    if not api_key:
        # Shouldn't happen — routing.py only picks anthropic_batch when
        # the key is present. Guard for type-narrowing + future safety.
        return run.failure(pass_error("submit", "no_api_key"))
    stopped = await stopped_error(run.pass_id)
    if stopped is not None:
        return run.failure(stopped)

    # Persist DreamInput so the per-phase callbacks can rebuild the
    # next phase's prompt without re-fetching from Postgres + FalkorDB.
    try:
        await persist_input_bundle(
            run.pass_id, input_bundle, lock_token=lock_handle.token
        )
    except Exception:
        logger.warning(
            f"Dream pass {run.pass_id}: could not persist its input bundle",
            exc_info=True,
        )
        return run.failure(pass_error("submit", "input_persist_failed"))

    # An empty job id tells the callbacks there is no JobStatus row to update.
    try:
        submission = await submit_phase(
            user_id=run.user_id,
            pass_id=run.pass_id,
            job_id=status_id or "",
            phase="consolidate",
            phase_models=phase_models_for_config(config),
            api_key=api_key,
            input_bundle=input_bundle,
        )
    except Exception:
        logger.warning(
            f"Dream pass {run.pass_id}: could not submit its first batch",
            exc_info=True,
        )
        return run.failure(pass_error("consolidate", "submit_failed"))
    return await _hand_off(run, lock_handle, submission.provider_batch_id, input_bundle)


async def _hand_off(
    run: DreamPassRun,
    lock_handle: DreamLockHandle,
    provider_batch_id: str,
    input_bundle: DreamInput,
) -> DreamPassResult:
    """Phase 1 is enqueued — hand the dream lock to the batch callback so it
    spans the full async lifetime (apply runs hours later). Extend the TTL to
    the batch window and record the submit first; the callback releases the
    lock on terminal/failure.

    A failed extend means the lock expired before the handoff — a newer pass
    may already own the graph, so the just-submitted batch must never be
    applied: it is taken back (``_take_back``). The lock is NOT disowned, so
    the context manager's compare-and-delete release stays a safe no-op.

    A row that refuses the submit (a stop closed it meanwhile, or it is gone),
    or does not hold it (``_record_submit``), gets the same take-back and
    keeps the lock to release on the way out. Nothing has landed yet, so
    nothing is charged.
    """
    logger.info(
        "Dream pass %s submitted via Anthropic batch=%s (phase=consolidate)",
        run.pass_id,
        provider_batch_id,
    )
    if not await lock_handle.extend(BATCH_LOCK_TTL_SECONDS):
        error = pass_error("submit", "lock_lost")
        logger.warning(
            f"Dream pass {run.pass_id}: lost its dream lock before the handoff; "
            f"taking batch {provider_batch_id} back"
        )
        await _take_back(run, lock_handle, provider_batch_id, error)
        return run.failure(error)
    stopped = await _record_submit(run, lock_handle, provider_batch_id, input_bundle)
    if stopped is not None:
        await _take_back(run, lock_handle, provider_batch_id, stopped)
        return run.failure(stopped)
    lock_handle.disown()
    return run.handed_off()


async def _record_submit(
    run: DreamPassRun,
    lock_handle: DreamLockHandle,
    provider_batch_id: str,
    input_bundle: DreamInput,
) -> str | None:
    """Record the submit on the pass's row; why the pass must not be handed
    off, else ``None``.

    The pass hands off only once its row holds the batch: the submit records
    the batch's day-long lease, without which the reaper would take the pass
    for dead within the hour while its batch ran. A refused write is
    authoritative: the row has closed or is gone, so the pass never hands
    off, its error the stop that closed the row or, when the row cannot say,
    ``REFUSED_SUBMIT_ERROR``. A write that failed or ran out of time says
    nothing about the row, which is read again (``_submit_on_record``)."""
    recorded = await record_submitted(
        run.pass_id,
        input_bundle=input_bundle,
        provider_batch_id=provider_batch_id,
        lease_token=lock_handle.token,
        lease_ttl_seconds=BATCH_LOCK_TTL_SECONDS,
    )
    if recorded:
        return None
    if recorded is None:
        return await _submit_on_record(run.pass_id, provider_batch_id)
    stopped = await stopped_error(run.pass_id)
    if stopped is not None:
        return stopped
    logger.warning(
        f"Dream pass {run.pass_id}: its row refused batch {provider_batch_id} "
        "and does not say why; not handing it off"
    )
    return REFUSED_SUBMIT_ERROR


async def _submit_on_record(pass_id: str, provider_batch_id: str) -> str | None:
    """Whether a submit whose write failed landed after all: ``None`` when
    the row, read again, holds *provider_batch_id*, else why the pass must
    not hand off (the stop that closed the row, or
    ``UNRECORDED_SUBMIT_ERROR`` when the row does not hold the batch or
    cannot be read). A write that committed late while the read missed it
    costs the pass, which the reaper would otherwise have closed mid-batch."""
    try:
        row = await read_pass(pass_id)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: its submit write failed and its row cannot "
            f"be read to find batch {provider_batch_id}; taking the batch back",
            exc_info=True,
        )
        return UNRECORDED_SUBMIT_ERROR
    stopped = stop_error(row) if row is not None else None
    if stopped is not None:
        return stopped
    if row is not None and row.provider_batch_id == provider_batch_id:
        logger.info(
            f"Dream pass {pass_id}: its submit write landed after all; handing "
            f"batch {provider_batch_id} off"
        )
        return None
    logger.warning(
        f"Dream pass {pass_id}: batch {provider_batch_id} is not on its record; "
        "taking it back"
    )
    return UNRECORDED_SUBMIT_ERROR


async def _take_back(
    run: DreamPassRun,
    lock_handle: DreamLockHandle,
    provider_batch_id: str,
    error: str,
) -> None:
    """Take back the batch a pass just submitted and must not hand off: the
    provider asked to stop it by the id the pass holds, then the batch off
    the executor's queue and the pass's bundle dropped.

    When the provider does not confirm the stop (it refused, or the call
    failed or ran out of time), the pass's open row closes ERRORED with
    *error*, the batch on it and marked (``pass_record.taken_back``), so the
    reaper's cleanup asks the provider again until the batch has stopped or
    ended. When that write does not land, the batch stays on the executor's
    queue, whose drop of the closed pass's batch
    (``batch_deliveries.drop_closed``), or the callback of a pass without a
    record, asks the provider once more to stop it by its id. The lock is
    left to the context manager's single release on the way out.

    Known limit: when the provider does not confirm the cancel and the
    take-back cannot be recorded on the open row (a store outage, a missing
    row, or a row a concurrent cancel already closed), the batch is left to
    the provider's 24-hour expiry. Its results are discarded and nothing is
    charged to the user; the provider may still bill that one phase."""
    pass_id = run.pass_id
    logger.info(
        f"Dream pass {pass_id} not handed off; taking batch {provider_batch_id} back"
    )
    if not await cancel_provider_batch(provider_batch_id):
        recorded = await record_taken_back(
            pass_id,
            error,
            provider_batch_id=provider_batch_id,
            lease_token=lock_handle.token,
        )
        if not recorded:
            logger.warning(
                f"Dream pass {pass_id}: batch {provider_batch_id} is not "
                "confirmed stopped and not on its record; left on the executor's "
                "queue, whose drop or callback stops it"
            )
            await best_effort_cleanup(pass_id)
            return
        logger.warning(
            f"Dream pass {pass_id}: batch {provider_batch_id} is not confirmed "
            "stopped; its record keeps it for the reaper's cleanup"
        )
    try:
        await remove_pending(provider_batch_id)
    except Exception:
        logger.warning(
            f"Dream pass {pass_id}: could not revoke batch {provider_batch_id}",
            exc_info=True,
        )
    await best_effort_cleanup(pass_id)
