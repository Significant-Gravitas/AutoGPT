"""The dream pass reaper: close the passes that outlived their lease, and
finish the cleanups nobody else finished.

A pass renews its lease at every step (``lease.py``), so an open row whose
lease lapsed a while ago most likely belongs to a pass that died (its process
killed, its batch callback lost, its last record write lost). A newer pass's
guard expires such a row when it looks at the scope; the reaper closes the
rest, every ``REAPER_INTERVAL_MINUTES``, across every user. The row's lease
is the authority: a pass whose Redis renewals kept landing while every row
write failed looks dead too, and is stopped at its next check.

Each run lists ``REAPER_ROW_LIMIT`` rows at most, oldest first: the closed
rows whose cleanup (``cleanup.py``) is pending and due, then the open rows
whose lease lapsed more than ``REAP_GRACE_SECONDS`` (one sync lock TTL) ago.
A cleanup is due once its lease lapsed a grace ago, or it was marked a grace
ago: time for a pass stopped while it ran to reach its next check, and for
the executor's drop to clean up after it. The grace is a bound, not a fence:
an apply that runs on longer than the grace after its pass was stopped can
still be writing when the reaper releases the lock; without the grace, the
reaper would release the lock of any pass still applying.

It closes an open row EXPIRED, holding the scope's lock when it is free, its
cancel generation bumped and its error naming the phase the pass died at,
only if nothing wrote the row since it was read (else ``moved``, left
alone), and marks it in the same statement, keeping the dead pass's lease
token and dropping its lease expiry, so its cleanup is due at once. A row
left APPLYING is closed the same way and never applied again: apply is
gated, and the reaper never calls it.

Then it cleans up after the pass (``cleanup.clean_up_pass``), releasing the
lock under the token the row kept unless the reaper holds the scope itself,
and clears the mark and the lease only once every step has finished
(``expired``, ``cleaned``); else the row stays marked (``retry``) and the
next run does every step again, the finished ones doing nothing. A row
marked longer ago than a batch can run no longer waits on the provider. A
row that kept no token (one written before leases, say) cannot have its
pass's lock released: its unlock finishes once the scope's lock is gone or
held under another open pass's token, and never by deleting a lock no one
can be matched to, so such a row stays marked until its lock lapses, 24 h
10 min at most for a batch lock and 30 min for a sync one.

Bounded: a run takes at most ``REAPER_BUDGET_SECONDS``, from before its
listing to the release of its hold on a scope. Rows are worked until
``RELEASE_TIMEOUT_SECONDS`` before the end, each started only with
``ROW_RESERVE_SECONDS`` left; the release that may follow the cut has those
last seconds, and finishes even if the cut lands during it. A row the budget
does not reach, or cuts short, is listed again next run; a phase charge it
cuts into finishes on its own (``batch_costs.py``).

Not bounded here: the scheduler's ``run_async`` stops waiting on a timeout
but does not cancel the run, so the run's own budget is what stops it; rows
past ``REAPER_ROW_LIMIT`` wait for later runs; and a batch pass's state
outlives its lease by an hour (``INPUT_TTL_SECONDS``), which leaves a first
attempt, after the grace, an interval and a budget, about 1,140 s of nominal
margin to charge the landed phases, an interval less per resumed attempt.

One line per row (INFO when it expired, was cleaned or moved; WARNING when it
failed, is to be retried, or the budget left it), then one with the counts
per outcome.
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

from .batch_submit import phase_models_for_config
from .cleanup import PassCleanup, clean_up_pass
from .locks import (
    DEFAULT_LOCK_TTL_SECONDS,
    LOCK_CHECK_TIMEOUT_SECONDS,
    release_dream_lock,
)
from .pass_record import reaped
from .provider_batch import PROVIDER_BATCH_WINDOW_SECONDS
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
# How long past its lease a pass is left before the reaper takes it for dead,
# and how long after a stop its cleanup waits for the pass to end itself.
REAP_GRACE_SECONDS = DEFAULT_LOCK_TTL_SECONDS
# The reaper's own hold on a scope while it closes one row.
REAPER_LOCK_TTL_SECONDS = 120
# The budget a row needs left for the reaper to start it.
ROW_RESERVE_SECONDS = 15.0
# Giving back the reaper's hold on a scope, carved out of the budget.
RELEASE_TIMEOUT_SECONDS = 2.0

ReapOutcome = Literal["expired", "cleaned", "moved", "retry", "failed", "out_of_budget"]


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
    """The rows to work, oldest first: the cleanups pending and due, then the
    open rows whose lease lapsed a grace ago, ``REAPER_ROW_LIMIT`` in all."""
    due_before = now - timedelta(seconds=REAP_GRACE_SECONDS)
    pending = await read_pending_cleanups(due_before=due_before, limit=REAPER_ROW_LIMIT)
    room = REAPER_ROW_LIMIT - len(pending)
    if room <= 0:
        return pending
    return pending + await read_expired_passes(due_before, limit=room)


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
    while it does when the lock is free, or finish the cleanup a closed row
    was left marked for; note the outcome in *done* before the hold is given
    back, so a budget that cuts into that release keeps it. ``failed`` when
    the store would not answer, or Redis would not let the reaper take the
    scope, and the row is listed again next run."""
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
    cleanup = await _clean_up(row, scope, release=not took_scope)
    lock = "held by the reaper" if took_scope else "released if the dead pass's"
    return _logged(
        row, cleanup, "expired", f"expired ({_error(row)})", f"; scope lock {lock}"
    )


async def _finish_cleanup(row: DreamPassRecord, scope: MemoryScope) -> ReapOutcome:
    """Clean up after a pass whose row closed marked and whose cleanup was
    not finished: every step again, the ones that already ran doing
    nothing."""
    cleanup = await _clean_up(row, scope, release=True)
    done = "cleaned" if cleanup.finished else "not cleaned up yet"
    return _logged(row, cleanup, "cleaned", f"({row.status.value.lower()}) {done}")


async def _clean_up(
    row: DreamPassRecord, scope: MemoryScope, *, release: bool
) -> PassCleanup:
    """Clean up after *row*'s pass, releasing its lock under the token the
    row kept when *release* (for a row that kept none, finding the lock not
    the pass's), then clear the row's mark and lease if every step finished.
    Raises when that write fails: the row is listed again."""
    cleanup = await clean_up_pass(
        row.id,
        scope,
        phase_models=_phase_models(row),
        provider_batch_id=_batch_to_stop(row),
        lock_token=row.lease_token,
        release=release,
        attribute_tokenless=True,
    )
    if cleanup.finished:
        await record_cleanup_finished(row.id)
    return cleanup


def _logged(
    row: DreamPassRecord,
    cleanup: PassCleanup,
    outcome: ReapOutcome,
    what: str,
    lock: str = "",
) -> ReapOutcome:
    """*outcome* once *cleanup* has finished, logged at INFO; else ``retry``,
    logged at WARNING with the steps left for the next run."""
    line = (
        f"Dream pass reaper: pass {row.id} {what}; batch "
        f"{row.provider_batch_id or 'none'}; charged "
        f"{', '.join(cleanup.charged) or 'nothing'}{lock}"
    )
    if cleanup.finished:
        logger.info(line)
        return outcome
    logger.warning(
        f"{line}; cleanup unfinished at {', '.join(cleanup.unfinished)}; "
        "listed again next run"
    )
    return "retry"


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


def _phase_models(row: DreamPassRecord) -> dict[str, str] | None:
    """The deployment's batch phase models, to price *row*'s landed phases
    with; ``None`` when they cannot be read, and its charge waits."""
    try:
        return phase_models_for_config(ChatConfig())
    except Exception:
        logger.warning(
            f"Dream pass reaper: no batch phase models to price pass {row.id}",
            exc_info=True,
        )
        return None


def _batch_to_stop(row: DreamPassRecord) -> str | None:
    """The provider batch the cleanup stops: the one the row names, unless
    the row was marked longer ago than a batch can run, so it has ended
    whatever the provider says (or cannot say)."""
    marked = row.cleanup_pending_at
    window = timedelta(seconds=PROVIDER_BATCH_WINDOW_SECONDS)
    if marked is not None and datetime.now(timezone.utc) - marked > window:
        return None
    return row.provider_batch_id


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
