"""Nightly batch-submit cron body — fans out per-user batch-family work.

One per-user APScheduler cron (``dream_nightly_batch_{user_id}``)
fires at user-local 03:00 daily and calls :func:`run_nightly_batch_submit`.
The function sequentially invokes each enabled "batch-family"
submitter:

  * Dream pass (P0.2) — consolidate / recombine / sanitize / apply
  * Ratification supersession sweep (P0.4) — clean up tentatives that
    aged past the grace period without earning a warm-context hit
  * Future P2 dedup / P3 self-model refresh / P4 scenario pre-warm /
    P11 threat rehearsal land as additional submitters here

Each submitter is fast to ENQUEUE (seconds) but may take up to ~1h to
COMPLETE when running against a real batch provider (Anthropic /
OpenAI batch API). The submitters fire in sequence at cron time; their
batch results land asynchronously via the separate
``copilot_batch_executor`` poller service that dispatches by
``custom_id`` to per-stage apply handlers.

Until the real batch providers land (currently scaffolded with
``NotImplementedError`` stubs in ``dream/batch/``), the dream pass
runs end-to-end via ``execute_dream_pass`` (sync_baseline path —
30s LLM thinking + seconds-long apply). The function shape is the
same either way.

Concurrency model (per ``dream/p0-spec.md`` §5 and the architecture
plan): LLM-thinking phases stay concurrent with user writes (they're
read-only against the user's graph); only the apply / writeback step
of each submitter briefly acquires the per-user lock and queues user
writes for the seconds it holds.

One pre-flight billing check runs at the top so a single
``nightly_id`` correlates all per-submitter cost log rows for this
pass.
"""

from __future__ import annotations

import logging
import uuid as uuidlib
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from .billing import check_dream_budget
from .ratification import RatificationResult, run_ratification_pass
from .schemas import DreamPassResult

logger = logging.getLogger(__name__)


class NightlyBatchResult(BaseModel):
    """Structured outcome of one nightly-batch-submit pass.

    Surfaced via the scheduler wrapper + admin endpoint so we can
    audit "what fired tonight for this user" without scraping logs.
    """

    user_id: str
    nightly_id: str = Field(
        description=(
            "UUID correlating every per-submitter cost-log row for "
            "this pass. The dream pass and ratification supersession "
            "sweep all share this id."
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


async def run_nightly_batch_submit(user_id: str) -> NightlyBatchResult:
    """Fan out per-user nightly batch-family submissions, in order.

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
    """
    nightly_id = str(uuidlib.uuid4())
    started_at = datetime.now(timezone.utc)

    # One pre-flight billing check for the whole pass. Per-submitter
    # cost logs (written by each submitter's own `record_phase_cost`
    # calls) all reference the same ``nightly_id`` via the
    # ``dream_pass_id`` field on PlatformCostLog rows. This means a
    # paywalled / over-budget user costs us one LD lookup + one
    # Redis read for the night, not one per submitter.
    budget_ok, budget_skip = await check_dream_budget(user_id)
    if not budget_ok:
        completed_at = datetime.now(timezone.utc)
        elapsed = (completed_at - started_at).total_seconds()
        if budget_skip == "rate_limit_unavailable":
            return NightlyBatchResult(
                user_id=user_id,
                nightly_id=nightly_id,
                started_at=started_at,
                completed_at=completed_at,
                elapsed_seconds=elapsed,
                error=f"billing: {budget_skip}",
            )
        return NightlyBatchResult(
            user_id=user_id,
            nightly_id=nightly_id,
            started_at=started_at,
            completed_at=completed_at,
            elapsed_seconds=elapsed,
            skipped=True,
            skip_reason=budget_skip or "insufficient_credits",
        )

    result = NightlyBatchResult(
        user_id=user_id, nightly_id=nightly_id, started_at=started_at
    )

    # Dream pass submitter. Per-submitter failure stays isolated —
    # a crashed dream pass doesn't block the ratification sweep
    # (ratification operates on already-written tentatives from
    # previous passes, not tonight's failed one).
    try:
        from .orchestrator import execute_dream_pass

        result.dream = await execute_dream_pass(user_id)
        if result.dream.error:
            logger.warning(
                "Nightly batch %s: dream submitter errored for user %s: %s",
                nightly_id,
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
            nightly_id,
            user_id[:12],
        )
        # Capture but continue to the next submitter — sweep is
        # independent.
        result.error = f"dream: {exc}"

    # Ratification supersession sweep. With the sync hit-hook landed
    # (see ``ratification_hits.try_ratify_on_hit``), the nightly
    # sweep primarily handles supersession of unratified tentatives
    # past their grace period — promotions happen inline at
    # retrieval-hit time.
    try:
        result.ratification = await run_ratification_pass(user_id)
        if result.ratification.error:
            logger.warning(
                "Nightly batch %s: ratification sweep errored for user %s: %s",
                nightly_id,
                user_id[:12],
                result.ratification.error,
            )
    except Exception as exc:
        logger.exception(
            "Nightly batch %s: ratification sweep crashed for user %s",
            nightly_id,
            user_id[:12],
        )
        # Append rather than overwrite so a dream failure plus a
        # ratification failure both surface.
        prev = result.error or ""
        result.error = (prev + " | " if prev else "") + f"ratification: {exc}"

    completed_at = datetime.now(timezone.utc)
    result.completed_at = completed_at
    result.elapsed_seconds = (completed_at - started_at).total_seconds()
    logger.info(
        "Nightly batch %s done for user %s in %.1fs: dream=%s ratification=%s",
        nightly_id,
        user_id[:12],
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
