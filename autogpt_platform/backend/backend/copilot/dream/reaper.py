"""The dream pass reaper: close the passes that outlived their lease, and
finish the cleanups it could not.

A pass renews its lease at every step (``lease.py``), so an open row whose
lease lapsed a while ago most likely belongs to a pass that died (its process
killed, its batch callback lost, its last record write lost). A newer pass's
guard expires such a row when it looks at the scope; the reaper closes the
rest, every ``REAPER_INTERVAL_MINUTES``, across every user. The row's lease
is the authority: a pass whose Redis renewals kept landing while every row
write failed looks dead too, and is stopped at its next check.

Each run lists ``REAPER_ROW_LIMIT`` rows at most, oldest first: the closed
rows whose cleanup an earlier run left (``cleanupPendingAt`` set), then the
open rows whose lease lapsed more than ``REAP_GRACE_SECONDS`` (one sync lock
TTL) ago. It closes an open row EXPIRED, holding the scope's lock when it is
free, its cancel generation bumped and its error naming the phase the pass
died at, only if nothing wrote the row since it was read (else ``moved``,
left alone), and marks it for cleanup in the same statement. Then, as for a
row an earlier run left, it cancels the provider batch the row names
(best-effort), charges the landed phases once through the cost gate while
the batch state is in Redis, releases the dead pass's lock by
compare-and-delete on the token the row kept (unless the reaper holds the
scope itself), deletes the batch state and bundle, and clears the mark and
the token. Every step is idempotent, so a cleanup cut short anywhere is run
again next time. A row left APPLYING is closed the same way and never
applied again: apply is gated, and the reaper never calls it.

Bounded: a run takes at most ``REAPER_BUDGET_SECONDS``, from before its
listing to the release of its hold on a scope. Rows are worked until
``RELEASE_TIMEOUT_SECONDS`` before the end, each started only with
``ROW_RESERVE_SECONDS`` left; the release that may follow the cut has those
last seconds, and finishes even if the cut lands during it. A row the budget
does not reach, or cuts short, is listed again next run.

Not bounded here: the scheduler's ``run_async`` stops waiting on a timeout
but does not cancel the run, so the run's own budget is what stops it; rows
past ``REAPER_ROW_LIMIT`` wait for later runs; and a batch pass's state
outlives its lease by an hour (``INPUT_TTL_SECONDS``), which leaves a first
attempt, after the grace, an interval and a budget, about 1,140 s of nominal
margin to charge the landed phases, an interval less per resumed attempt.

One line per row (INFO when it expired, was cleaned or moved; WARNING when it
failed or the budget left it), then one with the counts per outcome.
"""

import asyncio
import logging
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Literal

from prisma.enums import DreamPassStatus
from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.graphiti.scope import MemoryScope
from backend.data import redis_client
from backend.data.dream_pass_models import OPEN_STATUSES, DreamPassRecord

from .batch_costs import log_all_phase_costs
from .batch_state import best_effort_cleanup, read_state_or_none
from .batch_submit import phase_models_for_config
from .locks import (
    DEFAULT_LOCK_TTL_SECONDS,
    LOCK_CHECK_TIMEOUT_SECONDS,
    release_dream_lock,
)
from .pass_record import reaped
from .provider_batch import cancel_provider_batch
from .store import (
    read_expired_passes,
    read_pending_cleanups,
    record_cleanup_finished,
    write_stop,
)

logger = logging.getLogger(__name__)

REAPER_INTERVAL_MINUTES = 10
REAPER_BUDGET_SECONDS = 60.0
REAPER_ROW_LIMIT = 100
# How long past its lease a pass is left before the reaper takes it for dead.
REAP_GRACE_SECONDS = DEFAULT_LOCK_TTL_SECONDS
# The reaper's own hold on a scope while it closes one row.
REAPER_LOCK_TTL_SECONDS = 120
# The budget a row needs left for the reaper to start it.
ROW_RESERVE_SECONDS = 15.0
# Giving back the reaper's hold on a scope, carved out of the budget.
RELEASE_TIMEOUT_SECONDS = 2.0

ReapOutcome = Literal["expired", "cleaned", "moved", "failed", "out_of_budget"]


class ReaperRun(BaseModel):
    """What one run did: how many rows it listed, and each one's outcome."""

    listed: int
    outcomes: dict[ReapOutcome, int]


async def reap_expired_passes(*, now: datetime | None = None) -> ReaperRun:
    """One reaper run, within ``REAPER_BUDGET_SECONDS``; never raises."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + REAPER_BUDGET_SECONDS - RELEASE_TIMEOUT_SECONDS
    rows: list[DreamPassRecord] = []
    done: dict[str, ReapOutcome] = {}
    try:
        async with asyncio.timeout_at(deadline):
            rows = await _list(now or datetime.now(timezone.utc))
            await _reap_rows(rows, done, deadline)
    except Exception:
        logger.warning("Dream pass reaper: the run stopped short", exc_info=True)
    return _summary(rows, done)


async def _list(now: datetime) -> list[DreamPassRecord]:
    """The rows to work, oldest first: the cleanups left pending, then the
    open rows whose lease lapsed a grace ago, ``REAPER_ROW_LIMIT`` in all."""
    pending = await read_pending_cleanups(limit=REAPER_ROW_LIMIT)
    room = REAPER_ROW_LIMIT - len(pending)
    if room <= 0:
        return pending
    lapsed_before = now - timedelta(seconds=REAP_GRACE_SECONDS)
    return pending + await read_expired_passes(lapsed_before, limit=room)


async def _reap_rows(
    rows: list[DreamPassRecord], done: dict[str, ReapOutcome], deadline: float
) -> None:
    """Work *rows* in turn while the run's deadline leaves a row's reserve."""
    loop = asyncio.get_running_loop()
    for row in rows:
        if loop.time() > deadline - ROW_RESERVE_SECONDS:
            return
        await _reap(row, done)


def _summary(rows: list[DreamPassRecord], done: dict[str, ReapOutcome]) -> ReaperRun:
    """Log each row the budget left and the counts per outcome."""
    for row in rows:
        if row.id not in done:
            logger.warning(
                f"Dream pass reaper: pass {row.id} not finished within the "
                "budget; listed again next run"
            )
    outcomes: Counter[ReapOutcome] = Counter(done.values())
    outcomes["out_of_budget"] += len(rows) - len(done)
    run = ReaperRun(listed=len(rows), outcomes=dict(+outcomes))
    logger.info(f"Dream pass reaper: {run.listed} listed; {_counts(run)}")
    return run


async def _reap(row: DreamPassRecord, done: dict[str, ReapOutcome]) -> None:
    """Close one dead pass and clean up after it, holding the scope's lock
    while it does when the lock is free, or finish the cleanup an earlier run
    left; note the outcome in *done* before the hold is given back, so a
    budget that cuts into that release keeps it. ``failed`` when the store or
    Redis would not answer, and the row is listed again next run."""
    scope: MemoryScope | None = None
    hold: str | None = None
    try:
        scope = MemoryScope.build(row.user_id, row.expert_id)
        if row.status not in OPEN_STATUSES:
            done[row.id] = await _finish_cleanup(row, scope)
            return
        hold = await _take_scope(scope)
        done[row.id] = await _close(row, scope, took_scope=hold is not None)
    except Exception:
        logger.warning(
            f"Dream pass reaper: pass {row.id} failed; listed again next run",
            exc_info=True,
        )
        done[row.id] = "failed"
    finally:
        if scope is not None and hold is not None:
            await _give_back(scope, hold)


async def _close(
    row: DreamPassRecord, scope: MemoryScope, *, took_scope: bool
) -> ReapOutcome:
    """Expire *row*, marked for cleanup, then clean up after its pass;
    ``moved`` when the row was written since it was read, and nothing is
    touched."""
    closing = reaped(_error(row), not_updated_since=row.updated_at)
    if not await write_stop(row.id, closing):
        logger.info(f"Dream pass reaper: pass {row.id} moved since it was read; left")
        return "moved"
    # A held scope may still be the dead pass's: released only if it is.
    charged = await _clean_up(row, scope, release=not took_scope)
    lock = "held by the reaper" if took_scope else "released if the dead pass's"
    logger.info(
        f"Dream pass reaper: pass {row.id} expired ({_error(row)}); "
        f"batch {row.provider_batch_id or 'none'}; charged {charged}; "
        f"scope lock {lock}"
    )
    return "expired"


async def _finish_cleanup(row: DreamPassRecord, scope: MemoryScope) -> ReapOutcome:
    """Clean up after a pass an earlier run closed and could not clean up
    after: every step again, the ones that already ran doing nothing."""
    charged = await _clean_up(row, scope, release=True)
    logger.info(
        f"Dream pass reaper: pass {row.id} ({row.status.value.lower()}) cleaned "
        f"up after an earlier run; batch {row.provider_batch_id or 'none'}; "
        f"charged {charged}"
    )
    return "cleaned"


async def _clean_up(row: DreamPassRecord, scope: MemoryScope, *, release: bool) -> str:
    """Clean up after *row*'s pass, each step idempotent, then clear the
    row's mark; say what was charged."""
    if row.provider_batch_id:
        await cancel_provider_batch(row.provider_batch_id)
    charged = await _charge_landed(row)
    if release:
        await release_dream_lock(scope, row.lease_token)
    await best_effort_cleanup(row.id)
    await record_cleanup_finished(row.id)
    return charged


async def _take_scope(scope: MemoryScope) -> str | None:
    """Take the scope's lock under a fresh token and return it; ``None`` when
    another holds the lock. Raises when Redis does not answer within
    ``LOCK_CHECK_TIMEOUT_SECONDS``."""
    token = f"reaper:{uuid.uuid4()}"
    redis = await redis_client.get_redis_async()
    taken = await asyncio.wait_for(
        redis.set(
            scope.redis_key("dream_lock"), token, nx=True, ex=REAPER_LOCK_TTL_SECONDS
        ),
        timeout=LOCK_CHECK_TIMEOUT_SECONDS,
    )
    return token if taken else None


async def _give_back(scope: MemoryScope, hold: str) -> None:
    """Release the reaper's hold on *scope*: at most
    ``RELEASE_TIMEOUT_SECONDS``, Redis client included, and finished even
    when the run's budget cuts in meanwhile. A hold not given back lapses in
    ``REAPER_LOCK_TTL_SECONDS``."""
    release = asyncio.ensure_future(_release_within_bound(scope, hold))
    try:
        await asyncio.shield(release)
    except asyncio.CancelledError:
        await release
        raise


async def _release_within_bound(scope: MemoryScope, hold: str) -> None:
    try:
        async with asyncio.timeout(RELEASE_TIMEOUT_SECONDS):
            await release_dream_lock(scope, hold)
    except TimeoutError:
        logger.warning(
            f"Dream pass reaper: could not give back scope {scope.scope_key} in "
            f"{RELEASE_TIMEOUT_SECONDS} s; the hold lapses in "
            f"{REAPER_LOCK_TTL_SECONDS} s"
        )


async def _charge_landed(row: DreamPassRecord) -> str:
    """Charge the phases of *row*'s pass that landed in its batch state, once
    through the cost gate, pricing them with the deployment's batch phase
    models; say what happened."""
    state = await read_state_or_none(row.id)
    if not state:
        return "nothing (no batch state)"
    try:
        phase_models = phase_models_for_config(ChatConfig())
    except Exception:
        logger.warning(
            f"Dream pass reaper: no batch phase models to price pass {row.id}",
            exc_info=True,
        )
        return "nothing (no phase models)"
    charged = await log_all_phase_costs(
        user_id=row.user_id,
        expert_id=row.expert_id,
        pass_id=row.id,
        state=state,
        phase_models=phase_models,
    )
    return ", ".join(sorted(state)) if charged else "nothing (already charged)"


def _error(row: DreamPassRecord) -> str:
    """The closed row's error, naming the phase its pass died at."""
    phase = row.phase.value.lower()
    lapsed = row.lease_expires_at.isoformat() if row.lease_expires_at else "unknown"
    if row.status == DreamPassStatus.APPLYING:
        return (
            f"{phase}: the lease lapsed at {lapsed} while applying; closed by the "
            "reaper without applying again, and what it wrote may have landed"
        )
    return f"{phase}: the lease lapsed at {lapsed}; closed by the reaper"


def _counts(run: ReaperRun) -> str:
    counts = ", ".join(f"{outcome}={n}" for outcome, n in sorted(run.outcomes.items()))
    return counts or "nothing to reap"
