"""Nightly batch-submit cron body — fans out one scope's batch-family work.

One APScheduler cron per memory scope (``dream_nightly_batch_{scope_key}``;
the account's scope key is its user id) fires at owner-local 03:00 daily
and calls :func:`run_nightly_batch_submit`.
The function sequentially invokes each enabled "batch-family"
submitter:

  * Dream pass (P0.2) — consolidate / recombine / sanitize / apply
  * Ratification supersession sweep (P0.4) — clean up tentatives that
    aged past the grace period without earning a warm-context hit
  * Future P2 dedup / P3 self-model refresh / P4 scenario pre-warm /
    P11 threat rehearsal land as additional submitters here

Each submitter is fast to ENQUEUE (seconds). A dream pass routed to
Anthropic's Message Batches API (``anthropic_batch``: the
``dream-pass-batch-enabled`` flag plus a direct Anthropic key) submits
its first phase and returns; the separate ``copilot_batch_executor``
poller service collects each finished batch and dispatches it by
``custom_id`` to ``batch_callbacks``, which submits the next phase and,
after the last one, applies. Every phase batch gets its own 24-hour
window (Anthropic's batch SLA, ``MAX_BATCH_LIFETIME_SECONDS``). There
is no OpenAI batch path: ``call_provider`` refuses
``execution_mode="batch"`` for OpenAI.

Otherwise the dream pass runs end-to-end via ``execute_dream_pass`` on
the sync_baseline path (30s LLM thinking + seconds-long apply). The
function shape is the same either way.

Concurrency model (per ``dream/p0-spec.md`` §5 and the architecture
plan): LLM-thinking phases stay concurrent with user writes (they're
read-only against the user's graph); only the apply / writeback step
of each submitter briefly acquires the per-user lock and queues user
writes for the seconds it holds.

One pre-flight billing check runs at the top, so a paywalled or
over-budget user costs one check for the night rather than one per
submitter. ``nightly_id`` names the fan-out in its logs and result;
cost rows carry each submitter's own id (the dream pass's pass id).
"""

from __future__ import annotations

import logging
import uuid as uuidlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .billing import check_dream_budget
from .pass_record import DreamTrigger
from .ratification import RatificationResult, run_ratification_pass
from .schemas import DreamPassResult

logger = logging.getLogger(__name__)


class NightlyBatchResult(BaseModel):
    """Structured outcome of one nightly-batch-submit pass.

    Surfaced via the scheduler wrapper + admin endpoint so we can
    audit "what fired tonight for this user" without scraping logs.
    """

    user_id: str
    # The expert whose memory scope this pass ran on; None for the account.
    expert_id: str | None = None
    nightly_id: str = Field(
        description=(
            "UUID naming this fan-out in its logs and result. Cost "
            "rows carry each submitter's own id instead: the dream "
            "pass logs under dream.pass_id."
        )
    )
    started_at: datetime
    completed_at: datetime | None = None
    elapsed_seconds: float | None = None

    # Per-submitter results. ``None`` means the submitter was skipped
    # (flag off, no input, etc.). Each Pydantic model gets its own
    # field rather than a generic dict so consumers (admin viz,
    # AgentProbe scorers) don't have to dispatch on runtime type.
    dream: DreamPassResult | None = None
    ratification: RatificationResult | None = None

    # True when the dream submitter took a provider batch path: the
    # submit pass only ENQUEUED phase 1, and the dream's apply step
    # lands asynchronously via the BatchExecutor's dream callbacks up
    # to ~24h later (correlated by ``dream.pass_id``). The fan-out
    # itself is complete — this envelope is the terminal record of the
    # SUBMIT pass, so consumers must not read the dream counts as
    # final when this flag is set.
    dream_in_flight: bool = False

    # Top-level outcome.
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None


async def run_nightly_batch_submit(
    user_id: str, *, expert_id: str | None = None, trigger: DreamTrigger = "cron"
) -> NightlyBatchResult:
    """Fan out one memory scope's nightly batch-family submissions, in order.

    ``expert_id`` picks the expert's scope instead of the account's; the
    budget check stays on the owner either way, since an expert's dreams
    are paid from the owner's allowance.

    Today the order is dream-pass → ratification-supersession-sweep.
    Future additions land as new sequential calls between these two
    (e.g. P2 dedup goes before dream so the dream pass operates on a
    deduped fact set; P11 threat rehearsal goes after ratification).

    Returns immediately after the last submitter returns. In
    sync_baseline mode that's ~30s+ (each submitter inlines its
    apply); in real-batch mode it's ~seconds (each submitter just
    enqueues to the provider's batch API).

    Never raises — top-level failures are captured in
    ``NightlyBatchResult.error`` so the scheduler wrapper can log
    without retry-storming the cron.

    ``trigger`` is recorded on the dream pass: ``cron`` from the nightly
    job, ``admin`` when an admin ran the fan-out on demand.
    """
    nightly_id = str(uuidlib.uuid4())
    started_at = datetime.now(timezone.utc)

    # One pre-flight billing check for the whole fan-out: a paywalled /
    # over-budget user costs us one LD lookup + one Redis read for the
    # night, not one per submitter.
    budget_ok, budget_skip = await check_dream_budget(user_id)
    if not budget_ok:
        return _budget_stopped(user_id, expert_id, nightly_id, started_at, budget_skip)

    result = NightlyBatchResult(
        user_id=user_id,
        expert_id=expert_id,
        nightly_id=nightly_id,
        started_at=started_at,
    )
    await _submit_dream(result, trigger)
    await _submit_ratification(result)
    return _finished(result)


def _budget_stopped(
    user_id: str,
    expert_id: str | None,
    nightly_id: str,
    started_at: datetime,
    budget_skip: str | None,
) -> NightlyBatchResult:
    """The fan-out the pre-flight billing check stopped: errored when the
    budget could not be read, else skipped."""
    completed_at = datetime.now(timezone.utc)
    elapsed = (completed_at - started_at).total_seconds()
    if budget_skip == "rate_limit_unavailable":
        return NightlyBatchResult(
            user_id=user_id,
            expert_id=expert_id,
            nightly_id=nightly_id,
            started_at=started_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed,
            error=f"billing: {budget_skip}",
        )
    return NightlyBatchResult(
        user_id=user_id,
        expert_id=expert_id,
        nightly_id=nightly_id,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed,
        skipped=True,
        skip_reason=budget_skip or "insufficient_credits",
    )


async def _submit_dream(result: NightlyBatchResult, trigger: DreamTrigger) -> None:
    """Dream pass submitter. Per-submitter failure stays isolated — a
    crashed dream pass doesn't block the ratification sweep (ratification
    operates on already-written tentatives from previous passes, not
    tonight's failed one)."""
    user_id = result.user_id
    try:
        from .orchestrator import execute_dream_pass

        result.dream = await execute_dream_pass(
            user_id, expert_id=result.expert_id, trigger=trigger
        )
        if result.dream.error:
            logger.warning(
                "Nightly batch %s: dream submitter errored for user %s: %s",
                result.nightly_id,
                user_id[:12],
                result.dream.error,
            )
        elif (
            not result.dream.skipped and result.dream.execution_path != "sync_baseline"
        ):
            result.dream_in_flight = True
    except Exception as exc:
        logger.exception(
            "Nightly batch %s: dream submitter crashed for user %s",
            result.nightly_id,
            user_id[:12],
        )
        # Capture but continue to the next submitter — sweep is
        # independent.
        result.error = f"dream: {exc}"


async def _submit_ratification(result: NightlyBatchResult) -> None:
    """Ratification supersession sweep. With the sync hit-hook landed
    (see ``ratification_hits.try_ratify_on_hit``), the nightly sweep
    primarily handles supersession of unratified tentatives past their
    grace period — promotions happen inline at retrieval-hit time."""
    user_id = result.user_id
    try:
        result.ratification = await run_ratification_pass(
            user_id, expert_id=result.expert_id
        )
        if result.ratification.error:
            logger.warning(
                "Nightly batch %s: ratification sweep errored for user %s: %s",
                result.nightly_id,
                user_id[:12],
                result.ratification.error,
            )
    except Exception as exc:
        logger.exception(
            "Nightly batch %s: ratification sweep crashed for user %s",
            result.nightly_id,
            user_id[:12],
        )
        # Append rather than overwrite so a dream failure plus a
        # ratification failure both surface.
        prev = result.error or ""
        result.error = (prev + " | " if prev else "") + f"ratification: {exc}"


def _finished(result: NightlyBatchResult) -> NightlyBatchResult:
    completed_at = datetime.now(timezone.utc)
    result.completed_at = completed_at
    result.elapsed_seconds = (completed_at - result.started_at).total_seconds()
    logger.info(
        "Nightly batch %s done for user %s in %.1fs: dream=%s ratification=%s",
        result.nightly_id,
        result.user_id[:12],
        result.elapsed_seconds,
        _summary(result.dream),
        _summary(result.ratification),
    )
    return result


def _summary(submitter_result: Any) -> str:
    """One-word summary of a submitter's outcome for the nightly log line."""
    if submitter_result is None:
        return "skipped"
    if getattr(submitter_result, "error", None):
        return "errored"
    if getattr(submitter_result, "skipped", False):
        return "no-input"
    return "ran"
