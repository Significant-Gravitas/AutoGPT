"""The dream pass reaper: close the passes that outlived their lease, and
finish the cleanups nobody else finished.

A pass renews its lease at every step (``lease.py``), so an open row whose
lease lapsed a while ago most likely belongs to a pass that died (its process
killed, its batch callback lost, its last record write lost). A newer pass's
guard expires such a row when it looks at the scope; the reaper closes the
rest, every ``REAPER_INTERVAL_MINUTES``, across every user. The row's lease
is the authority: a pass whose Redis renewals kept landing while every row
write failed looks dead too, and is stopped at its next check.

Each run lists ``REAPER_ROW_LIMIT`` rows at most, of two kinds, each kept
to half the cap when the other has rows: the open rows whose lease lapsed
more than ``REAP_GRACE_SECONDS`` (one sync lock TTL) ago, and the closed
rows whose cleanup (``cleanup.py``) is pending and due. Each kind gets its
own share of the run's time (``_reap_both``), so neither can spend the
other's: cleanups that keep failing (a provider that hangs on every cancel)
do not keep a dead pass from being closed and charged before its batch state
lapses, and a stream of lapsing passes does not keep the pending cleanups
waiting. Within its kind, each row is tried for ``ROW_ATTEMPT_SECONDS`` at
most, so one that hangs leaves the rows behind it their turn.

Each kind is worked in scan cycles, in key order: open rows by their lapse,
pending cleanups by when they were marked, then by id. A cycle covers the
rows whose key is at most its end, the newest key when it starts, and Redis
keeps where it stands (``_SCAN_KEYS``): its end, and the key of the last row
tried, noted before that row's attempt. A run lists the rows after that one,
up to the end; once none is left, it starts the next cycle, from the first
row. So every row present when a cycle starts is tried once in it, whatever
fails, arrives or slips; a row that arrives later has a key past the end and
waits for the next cycle; and no row is written to make room. Without the
cycle's state (a Redis that cannot say) a run starts from the first row, and
fairness waits for Redis. Two reapers running at once (a rollout) can move a
cycle back, the last write winning; a row's close and its cleanup's steps
are safe to repeat, so that costs time, never a second charge.
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
(``expired``, ``cleaned``); else the row stays marked (``retry``), and a
later run does every step again, the finished ones doing nothing. A row
marked longer ago than a batch can run no longer waits on the provider. A
row that kept no token (one written before leases, say) cannot have its
pass's lock released: its unlock finishes once the scope's lock is gone or
held under another open pass's token, and never by deleting a lock no one
can be matched to, so such a row stays marked until its lock lapses, 24 h
10 min at most for a batch lock and 30 min for a sync one.

Bounded: a run takes at most ``REAPER_BUDGET_SECONDS``, from before its
listing to the release of its hold on a scope. Rows are worked until
``RELEASE_TIMEOUT_SECONDS`` before the end, each started only with
``ROW_RESERVE_SECONDS`` left in its kind's share, and all of its work, from
the note of its scan to the write that clears its mark, done by the end of
that share. Its cleanup is cut off after ``ROW_ATTEMPT_SECONDS``; the write
that clears its mark comes after, on a bound of its own, so a slow write
does not cost the cleanup its time, and a mark the share leaves no time to
clear is cleared by the next run, whose cleanup finds nothing left to do.
The release that may follow a cut has its own last seconds, and finishes
even if the cut lands during it. A cleanup cut short resumes where it
stopped: its steps are ordered and idempotent, each phase is charged under
its own claim, and a provider batch whose cancel was sent is next only
asked for its status (``provider_batch.py``). Every call a row makes is
bounded by a timeout around
a direct await, never ``asyncio.wait_for``, which on Python 3.11 returns the
result of a call that completes as a cut lands and so loses the cut. A row
the budget does not reach, or cuts short, is listed again next run; a phase
charge it cuts into finishes on its own (``batch_costs.py``).

Not bounded here: the scheduler's ``run_async`` stops waiting on a timeout
but does not cancel the run, so the run's own budget is what stops it; and
a batch pass's state outlives its lease by twelve hours
(``INPUT_TTL_SECONDS``), so
a pass's landed phases are still there to charge behind about seventy runs'
worth of cleanups that keep failing. One whose state lapsed all the same is
reported as lost usage, an error (``cleanup.py``).

With ``SWEEP_RESERVE_SECONDS`` of the budget left after the rows, a run also
sweeps the memory graphs where a dream write's derivation record failed
(``graphiti/provenance_pending.py``): it completes their citation markers
under each graph's write lock, so a dream fact's provenance is complete even
before the next forget there, which reconciles on its own anyway.

One line per row (INFO when it expired, was cleaned or moved; WARNING when it
failed, is to be retried, or the budget left it), then one with the counts
per outcome.
"""

import asyncio
import logging
import uuid
from collections import Counter
from collections.abc import Awaitable
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, cast

from prisma.enums import DreamPassPhase, DreamPassRoute
from pydantic import BaseModel

from backend.copilot.config import ChatConfig
from backend.copilot.graphiti.provenance_pending import sweep_pending
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
from .pass_errors import PassErrorStage, pass_error
from .pass_record import reaped
from .provider_batch import PROVIDER_BATCH_WINDOW_SECONDS
from .schemas import DreamPhase
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
# The most one row's attempt may take, well under a kind's share of the
# budget, so a row that hangs leaves the others behind it their turn; and
# the budget a row needs left in its kind's share for the reaper to start it.
ROW_ATTEMPT_SECONDS = 12.0
ROW_RESERVE_SECONDS = ROW_ATTEMPT_SECONDS
# Each kind's scan cycle, a Redis hash: ``end``, the key past which rows
# wait for the next cycle, and ``last``, the key of the last row tried.
ScanKind = Literal["open", "pending"]
_SCAN_KEYS: dict[ScanKind, str] = {
    "open": "dream:reaper:open_scan",
    "pending": "dream:reaper:pending_scan",
}
SCAN_TTL_SECONDS = 24 * 60 * 60
# Giving back the reaper's hold on a scope, carved out of the budget.
RELEASE_TIMEOUT_SECONDS = 2.0
# The budget a run needs left after its rows to sweep pending dream records.
SWEEP_RESERVE_SECONDS = 20.0

ReapOutcome = Literal["expired", "cleaned", "moved", "retry", "failed", "out_of_budget"]
# The outcomes of a row whose cleanup finished, its mark still to clear.
_FINISHED: frozenset[ReapOutcome] = frozenset({"expired", "cleaned"})


class ReaperRun(BaseModel):
    """What one run did: how many rows it listed, and each one's outcome;
    and how many dream records it completed in the graphs it swept."""

    listed: int
    outcomes: dict[ReapOutcome, int]
    reconciled: int = 0


async def reap_expired_passes(*, now: datetime | None = None) -> ReaperRun:
    """One reaper run, within ``REAPER_BUDGET_SECONDS``; never raises."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + REAPER_BUDGET_SECONDS - RELEASE_TIMEOUT_SECONDS
    lapsed: list[DreamPassRecord] = []
    pending: list[DreamPassRecord] = []
    done: dict[str, ReapOutcome] = {}
    reconciled = 0
    try:
        async with asyncio.timeout_at(deadline):
            lapsed, pending = await _list(now or datetime.now(timezone.utc))
            await _reap_both(lapsed, pending, done, deadline)
            if loop.time() < deadline - SWEEP_RESERVE_SECONDS:
                reconciled = (await sweep_pending()).completed
    except Exception:
        logger.warning("Dream pass reaper: the run stopped short", exc_info=True)
    return _summary([*lapsed, *pending], done, reconciled)


async def _list(now: datetime) -> tuple[list[DreamPassRecord], list[DreamPassRecord]]:
    """The rows to work, ``REAPER_ROW_LIMIT`` in all, each kind kept to half
    of it when the other has rows, each from its scan cycle (``_scan_rows``):
    the open rows whose lease lapsed a grace ago, and the cleanups pending
    and due."""
    due_before = now - timedelta(seconds=REAP_GRACE_SECONDS)
    lapsed = await _scan_rows("open", due_before, now)
    pending = await _scan_rows("pending", due_before, now)
    kept = min(len(lapsed), max(REAPER_ROW_LIMIT - len(pending), REAPER_ROW_LIMIT // 2))
    return lapsed[:kept], pending[: REAPER_ROW_LIMIT - kept]


async def _scan_rows(
    kind: ScanKind, due_before: datetime, now: datetime
) -> list[DreamPassRecord]:
    """``REAPER_ROW_LIMIT`` rows of *kind* at most, in key order: those after
    the last one tried in the scan cycle under way, up to its end; when none
    is under way, or none of its rows is left, the first rows of a new cycle,
    which ends at the newest key there is: an open row's lapse is due by
    *due_before*, a cleanup is marked by *now*."""
    scan = await _read_scan(kind)
    if scan is not None:
        end, last = scan
        rows = await _rows_of(kind, due_before, end, last)
        if rows:
            return rows
    end = due_before if kind == "open" else now
    await _start_scan(kind, end)
    return await _rows_of(kind, due_before, end, None)


async def _rows_of(
    kind: ScanKind,
    due_before: datetime,
    end: datetime,
    after: tuple[datetime, str] | None,
) -> list[DreamPassRecord]:
    if kind == "open":
        return await read_expired_passes(end, limit=REAPER_ROW_LIMIT, after=after)
    return await read_pending_cleanups(
        due_before=due_before, limit=REAPER_ROW_LIMIT, marked_through=end, after=after
    )


async def _read_scan(
    kind: ScanKind,
) -> tuple[datetime, tuple[datetime, str] | None] | None:
    """The end of *kind*'s scan cycle under way and the key of the last row
    tried in it, if any; ``None`` when none is under way or Redis cannot
    say."""
    try:
        async with asyncio.timeout(LOCK_CHECK_TIMEOUT_SECONDS):
            redis = await redis_client.get_redis_async()
            raw = await cast(
                "Awaitable[dict[Any, Any]]", redis.hgetall(_SCAN_KEYS[kind])
            )
        scan = {_text(field): _text(value) for field, value in raw.items()}
        if "end" not in scan:
            return None
        last = scan.get("last")
        return datetime.fromisoformat(scan["end"]), _parsed_key(last) if last else None
    except Exception:
        logger.warning(
            f"Dream pass reaper: could not read where its scan of {kind} passes "
            "stands; starting a new one",
            exc_info=True,
        )
        return None


async def _start_scan(kind: ScanKind, end: datetime) -> None:
    """Start a scan cycle of *kind* that ends at *end*, from its first row.
    Best-effort: without it, each run starts from the first row."""
    try:
        async with asyncio.timeout(LOCK_CHECK_TIMEOUT_SECONDS):
            redis = await redis_client.get_redis_async()
            key = _SCAN_KEYS[kind]
            await cast("Awaitable[int]", redis.hset(key, "end", end.isoformat()))
            await cast("Awaitable[int]", redis.hset(key, "last", ""))
            await redis.expire(key, SCAN_TTL_SECONDS)
    except Exception:
        logger.warning(
            f"Dream pass reaper: could not start a scan of {kind} passes",
            exc_info=True,
        )


async def _note_tried(row: DreamPassRecord) -> None:
    """Note *row* as the last of its kind tried in the scan cycle under way,
    before it is tried, so the next run goes on after it however its attempt
    ends. Best-effort."""
    kind: ScanKind = "open" if row.status in OPEN_STATUSES else "pending"
    key_time = row.lease_expires_at if kind == "open" else row.cleanup_pending_at
    if key_time is None:
        return
    try:
        async with asyncio.timeout(LOCK_CHECK_TIMEOUT_SECONDS):
            redis = await redis_client.get_redis_async()
            last = f"{key_time.isoformat()}|{row.id}"
            await cast("Awaitable[int]", redis.hset(_SCAN_KEYS[kind], "last", last))
            await redis.expire(_SCAN_KEYS[kind], SCAN_TTL_SECONDS)
    except Exception:
        logger.warning(
            f"Dream pass reaper: could not note pass {row.id} as tried",
            exc_info=True,
        )


def _parsed_key(key: str) -> tuple[datetime, str]:
    key_time, pass_id = key.split("|", 1)
    return datetime.fromisoformat(key_time), pass_id


def _text(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


async def _reap_both(
    lapsed: list[DreamPassRecord],
    pending: list[DreamPassRecord],
    done: dict[str, ReapOutcome],
    deadline: float,
) -> None:
    """Work both kinds, each in its own share of the time left: the lapsed
    passes up to half of it when cleanups are pending too, the pending
    cleanups up to the run's deadline, then the lapsed passes the first
    share did not reach, in whatever is left. A row slow enough to use a
    whole share, its calls each within their own deadlines, spends only its
    kind's time."""
    loop = asyncio.get_running_loop()
    halfway = loop.time() + (deadline - loop.time()) / 2
    await _reap_rows(lapsed, done, halfway if pending else deadline)
    await _reap_rows(pending, done, deadline)
    await _reap_rows(lapsed, done, deadline)


async def _reap_rows(
    rows: list[DreamPassRecord], done: dict[str, ReapOutcome], until: float
) -> None:
    """Work the *rows* not yet done, in turn, each started only with
    ``ROW_RESERVE_SECONDS`` left before *until* and all of its work done by
    it: noted as tried in its scan cycle (``_note_tried``), cleaned up within
    ``ROW_ATTEMPT_SECONDS`` (``_attempt``), and, its cleanup finished, its
    mark cleared (``_record_finished``). A row cut off is done for this run,
    and the next one gets its turn."""
    loop = asyncio.get_running_loop()
    for row in rows:
        if row.id in done:
            continue
        if loop.time() > until - ROW_RESERVE_SECONDS:
            return
        try:
            async with asyncio.timeout_at(until):
                await _note_tried(row)
                await _attempt(row, done, min(until, loop.time() + ROW_ATTEMPT_SECONDS))
                if done.get(row.id) in _FINISHED:
                    await _record_finished(row.id, done)
        except TimeoutError:
            logger.warning(
                f"Dream pass reaper: pass {row.id} cut short by its share of the "
                "budget; listed again next run"
            )
            done.setdefault(row.id, "out_of_budget")


async def _attempt(
    row: DreamPassRecord, done: dict[str, ReapOutcome], cutoff: float
) -> None:
    """``_reap`` *row*, cut off at *cutoff*."""
    try:
        async with asyncio.timeout_at(cutoff):
            await _reap(row, done)
    except TimeoutError:
        logger.warning(
            f"Dream pass reaper: pass {row.id} cut short after "
            f"{ROW_ATTEMPT_SECONDS:g} s; listed again next run"
        )
        done.setdefault(row.id, "out_of_budget")


def _summary(
    rows: list[DreamPassRecord], done: dict[str, ReapOutcome], reconciled: int
) -> ReaperRun:
    """Log each row the budget left and the counts per outcome."""
    for row in rows:
        if row.id not in done:
            logger.warning(
                f"Dream pass reaper: pass {row.id} not finished within the "
                "budget; listed again next run"
            )
    outcomes: Counter[ReapOutcome] = Counter(done.values())
    outcomes["out_of_budget"] += len(rows) - len(done)
    run = ReaperRun(listed=len(rows), outcomes=dict(+outcomes), reconciled=reconciled)
    logger.info(
        f"Dream pass reaper: {run.listed} listed; {_counts(run)}; "
        f"{run.reconciled} dream record(s) reconciled"
    )
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
    error = _error(row)
    if not await write_stop(row.id, reaped(error, not_updated_since=row.updated_at)):
        logger.info(f"Dream pass reaper: pass {row.id} moved since it was read; left")
        return "moved"
    # A held scope may still be the dead pass's: released only if it is.
    cleanup = await _clean_up(row, scope, release=not took_scope)
    lock = "held by the reaper" if took_scope else "released if the dead pass's"
    what = (
        f"expired as {error} ({row.status.value.lower()}, lease until "
        f"{row.lease_expires_at}, last written {row.updated_at})"
    )
    return _logged(row, cleanup, "expired", what, f"; scope lock {lock}")


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
    the pass's). Its mark is cleared after the attempt
    (``_record_finished``). Its landed phases are charged under the usage
    attribution the pass started under, read off its row, never the one the
    account has now; a row without one is charged nothing
    (``batch_costs``)."""
    return await clean_up_pass(
        row.id,
        scope,
        phase_models=_phase_models(row),
        provider_batch_id=_batch_to_stop(row),
        lock_token=row.lease_token,
        release=release,
        attribute_tokenless=True,
        landed=_landed(row),
    )


async def _record_finished(pass_id: str, done: dict[str, ReapOutcome]) -> None:
    """Clear the row's mark and lease once its cleanup has finished, on the
    write's own bound (``store.RECORD_WRITE_TIMEOUT_SECONDS``); ``failed``
    when it does not land, and the next run does the cleanup again, its
    finished steps doing nothing."""
    try:
        await record_cleanup_finished(pass_id)
    except Exception:
        logger.warning(
            f"Dream pass reaper: pass {pass_id}'s cleanup finished but its mark "
            "could not be cleared; listed again next run",
            exc_info=True,
        )
        done[pass_id] = "failed"


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
    async with asyncio.timeout(LOCK_CHECK_TIMEOUT_SECONDS):
        redis = await redis_client.get_redis_async()
        taken = await redis.set(
            scope.redis_key("dream_lock"), token, nx=True, ex=REAPER_LOCK_TTL_SECONDS
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
    """The deployment's batch phase models, to price those of *row*'s landed
    phases whose state rows recorded no model (rows written before they
    did); the others are priced on the model they ran on. ``None`` when they
    cannot be read, and such a phase's charge waits."""
    try:
        return phase_models_for_config(ChatConfig())
    except Exception:
        logger.warning(
            f"Dream pass reaper: no batch phase models to price pass {row.id}",
            exc_info=True,
        )
        return None


def _landed(row: DreamPassRecord) -> list[DreamPhase]:
    """The phases of a batch pass whose output *row* recorded: landed, so
    their usage is lost should their batch state be gone before they were
    charged. None for a sync pass, which charges each phase as it lands and
    keeps no batch state."""
    if row.route != DreamPassRoute.ANTHROPIC_BATCH:
        return []
    outputs = row.phase_outputs
    recorded: dict[DreamPhase, object] = {
        "consolidate": outputs.consolidate,
        "recombine": outputs.recombine,
        "sanitize": outputs.sanitize,
    }
    return [phase for phase, output in recorded.items() if output is not None]


def _batch_to_stop(row: DreamPassRecord) -> str | None:
    """The provider batch the cleanup stops: the one the row names, unless
    the row was marked longer ago than a batch can run, so it has ended
    whatever the provider says (or cannot say)."""
    marked = row.cleanup_pending_at
    window = timedelta(seconds=PROVIDER_BATCH_WINDOW_SECONDS)
    if marked is not None and datetime.now(timezone.utc) - marked > window:
        return None
    return row.provider_batch_id


# The step a row's phase names, as its error records it.
_PHASE_STAGES: dict[DreamPassPhase, PassErrorStage] = {
    DreamPassPhase.GATHER: "gather",
    DreamPassPhase.CONSOLIDATE: "consolidate",
    DreamPassPhase.RECOMBINE: "recombine",
    DreamPassPhase.SANITIZE: "sanitize",
    DreamPassPhase.APPLY: "apply",
    DreamPassPhase.DONE: "done",
}


def _error(row: DreamPassRecord) -> str:
    """The closed row's error, naming the step its pass died at."""
    return pass_error(_PHASE_STAGES[row.phase], "reaped")


def _counts(run: ReaperRun) -> str:
    counts = ", ".join(f"{outcome}={n}" for outcome, n in sorted(run.outcomes.items()))
    return counts or "nothing to reap"
