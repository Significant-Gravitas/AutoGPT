"""One guard against overlapping dream passes of a scope.

Every pass runs it once, after it has taken the scope's lock and before it
gathers, so before the batch route submits anything; the admin and eval
triggers go through it like the nightly job. The lock alone does not keep two
passes apart: it lapses under a sync pass that outlives it and under a batch
pass whose last phase lands after its lease, and a row can stay open behind a
free lock when the pass's last write was lost. So the guard reads the scope's
open ``DreamPass`` rows, newest first, up to ``GUARD_ROW_LIMIT`` other than the
pass's own:

  master flag off (an authoritative answer)   skip, ``disabled``
  no other open row                           go on
  an open row, fresh                          skip, ``pass_in_progress``
  an open row, fresh, ``force``               expire it, go on
  an open row, stale                          expire it, go on

A row is fresh while its lease has not lapsed or, without a lease, while it was
written within the pass lock's TTL. Expiring is one conditional transition that
bumps the row's cancel generation, so its pass stops at its next check, and
marks the row for the cleanup after that pass (its batch, its landed phases,
its state), which the reaper finishes if the pass does not (``cleanup.py``);
it lands only while the row is still open. A stale row's expiry is also a
compare-and-set on the row's last write: it lands only if nothing has written
the row since the guard read it, and a row that moved in between is alive and
blocks after all. An admin's forced expiry of a fresh row skips that
compare-and-set: that row's pass no longer holds the lock (the forcing pass
took it), and it stops at its next check or at its lock check before apply.
Rows past the limit, and rows no newer pass looks at, are left for the reaper
(``reaper.py``).

Two passes triggered together can both skip: the one that lost the lock race
records ``lock_held`` while its row is still open, and the winner's guard sees
that fresh row and skips as ``pass_in_progress``. That is a deliberate
liveness trade-off, never two passes at once but sometimes none: an admin
retry or the scope's next trigger runs the pass.

The guard never blocks on the store or the flag service: a read or write that
fails or runs out of time is logged at warning and the pass goes on as it did
before the guard existed, and the guard as a whole gets
``GUARD_BUDGET_SECONDS``, after which the pass goes on unguarded. The master
flag is read here, once per pass; the phases never read it.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.dream_pass_models import OPEN_STATUSES, DreamPassRecord
from backend.util.feature_flag import Flag, evaluate_feature_flag

from .locks import DEFAULT_LOCK_TTL_SECONDS
from .pass_record import expired
from .pass_run import DreamPassRun, PassEnded
from .store import read_open_passes, read_pass, write_stop

logger = logging.getLogger(__name__)

# How long the master-flag read may take before the pass goes on without it.
FLAG_READ_TIMEOUT_SECONDS = 10.0
# The whole guard's budget, however many reads and expiries it makes.
GUARD_BUDGET_SECONDS = 15.0
# How many of the scope's other open passes, newest first, the guard weighs.
GUARD_ROW_LIMIT = 5


async def guard_dream_pass(run: DreamPassRun, scope: MemoryScope) -> None:
    """Raise ``PassEnded`` with the skip that keeps *run* from starting, or
    return and let it gather, the pass going on unguarded once the guard has
    used ``GUARD_BUDGET_SECONDS``."""
    try:
        async with asyncio.timeout(GUARD_BUDGET_SECONDS):
            await _guard(run, scope)
    except TimeoutError:
        logger.warning(
            f"Dream pass {run.pass_id}: the guard used up its "
            f"{GUARD_BUDGET_SECONDS:g}s budget; going on unguarded"
        )


async def _guard(run: DreamPassRun, scope: MemoryScope) -> None:
    if not await _dream_pass_enabled(run):
        raise PassEnded(run.skipped("disabled"))
    for row in await _other_open_passes(run, scope):
        if await _holds_scope(run, row):
            logger.info(
                f"Dream pass {run.pass_id} skipped: pass {row.id} of the same "
                f"scope is in progress ({row.status.value} at {row.phase.value})"
            )
            raise PassEnded(run.skipped("pass_in_progress"))


async def _dream_pass_enabled(run: DreamPassRun) -> bool:
    """The master flag. Off only on an authoritative answer: a read that
    fails, runs out of time or falls back to its default lets the pass go on,
    as the cron entry that fired it (if any) already let it through."""
    flag = Flag.DREAM_PASS_ENABLED.value
    try:
        enabled, authoritative = await asyncio.wait_for(
            evaluate_feature_flag(Flag.DREAM_PASS_ENABLED, run.user_id),
            timeout=FLAG_READ_TIMEOUT_SECONDS,
        )
    except Exception:
        logger.warning(
            f"Dream pass {run.pass_id}: could not read {flag}; going on",
            exc_info=True,
        )
        return True
    if not authoritative:
        logger.warning(f"Dream pass {run.pass_id}: {flag} unanswered; going on")
        return True
    return enabled


async def _other_open_passes(
    run: DreamPassRun, scope: MemoryScope
) -> list[DreamPassRecord]:
    """The scope's newest open passes other than *run*, at most
    ``GUARD_ROW_LIMIT``; none when the store cannot say in time, and the pass
    goes on unguarded. One more row is read for the pass's own."""
    try:
        rows = await read_open_passes(scope, limit=GUARD_ROW_LIMIT + 1)
    except Exception:
        logger.warning(
            f"Dream pass {run.pass_id}: could not read the scope's open passes; "
            "going on unguarded",
            exc_info=True,
        )
        return []
    return [row for row in rows if row.id != run.pass_id][:GUARD_ROW_LIMIT]


async def _holds_scope(run: DreamPassRun, row: DreamPassRecord) -> bool:
    """Whether *row*'s pass still holds the scope: a fresh row does, unless
    the run is forced; a stale or forced row is expired first and holds it
    only if it moved before the expiry could land."""
    stale = _is_stale(row, datetime.now(timezone.utc))
    if not stale and not run.force:
        return True
    return not await _expire(run, row, stale=stale)


def _is_stale(row: DreamPassRecord, now: datetime) -> bool:
    """A row with a lease is stale once the lease has lapsed; one without,
    once nothing has written it for the pass lock's TTL."""
    if row.lease_expires_at is not None:
        return row.lease_expires_at <= now
    return row.updated_at <= now - timedelta(seconds=DEFAULT_LOCK_TTL_SECONDS)


async def _expire(run: DreamPassRun, row: DreamPassRecord, *, stale: bool) -> bool:
    """Expire *row* and say whether it no longer holds the scope: expired now,
    closed by its own pass meanwhile, or unknown because the store did not
    answer in time (the pass goes on)."""
    reason = _expiry_reason(run, row, stale=stale)
    unchanged_since = row.updated_at if stale else None
    try:
        if await write_stop(row.id, expired(reason, not_updated_since=unchanged_since)):
            logger.warning(f"Dream pass {run.pass_id} expired pass {row.id}: {reason}")
            return True
        current = await read_pass(row.id)
    except Exception:
        logger.warning(
            f"Dream pass {run.pass_id}: could not expire pass {row.id}; going on",
            exc_info=True,
        )
        return True
    return current is None or current.status not in OPEN_STATUSES


def _expiry_reason(run: DreamPassRun, row: DreamPassRecord, *, stale: bool) -> str:
    """What the expired row's error says happened."""
    if not stale:
        return f"forced by an admin-triggered dream pass {run.pass_id}"
    if row.lease_expires_at is not None:
        lapsed = f"lease lapsed at {row.lease_expires_at.isoformat()}"
    else:
        lapsed = f"no progress since {row.updated_at.isoformat()}"
    return f"{lapsed}; expired by dream pass {run.pass_id}"
