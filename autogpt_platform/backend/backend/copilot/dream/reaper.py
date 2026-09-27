"""The dream pass reaper: close the passes that outlived their lease.

A pass renews its lease at every step (``lease.py``), so an open row whose
lease lapsed belongs to a pass that died: its process was killed, its batch
callback never came, or its last record write was lost. A newer pass's guard
expires such a row when it looks at the scope; the reaper closes the rest,
every ``REAPER_INTERVAL_MINUTES`` from the scheduler, across every user.

Each run lists the open rows whose lease lapsed more than
``REAP_GRACE_SECONDS`` ago (one sync lock TTL, so a pass that is merely slow
to renew is never taken for dead), oldest first, at most ``REAPER_ROW_LIMIT``,
and for each, holding the scope's lock when it is free:

  * closes the row EXPIRED with its cancel generation bumped, only if nothing
    has written it since it was read (a row that moved is alive, or closed
    itself: ``moved``, left alone), its error naming the phase it died at;
  * cancels its provider batch, if it names one (best-effort);
  * charges its landed phases once, through the cost gate, when its batch
    state is still in Redis;
  * deletes its batch state and input bundle;
  * releases the lock only by compare-and-delete on the row's lease token,
    so a lock the dead pass still holds goes and another pass's stays.

A row left APPLYING by a pass that died after it claimed apply is closed the
same way and never applied again: apply is gated, and the reaper never calls
it. The reaper holds the scope's lock (``REAPER_LOCK_TTL_SECONDS``) only to
keep a new pass off the scope while it closes the old one; when another pass
holds it, the row is closed all the same and that lock left alone.

A run never blocks longer than ``REAPER_BUDGET_SECONDS``: it starts no row
once too little of the budget is left, and the budget cancels the rest. The
rows it did not finish are counted ``out_of_budget`` and listed again next
run. One info line per row, then one with the counts per outcome.
"""

import asyncio
import logging
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Literal

from prisma.enums import DreamPassStatus
from pydantic import BaseModel, Field

from backend.copilot.config import ChatConfig
from backend.copilot.graphiti.scope import MemoryScope
from backend.data import redis_client
from backend.data.dream_pass_models import DreamPassRecord

from .batch_costs import log_all_phase_costs
from .batch_state import best_effort_cleanup, read_state_or_none
from .batch_submit import phase_models_for_config
from .locks import (
    DEFAULT_LOCK_TTL_SECONDS,
    LOCK_CHECK_TIMEOUT_SECONDS,
    release_dream_lock,
)
from .pass_record import expired
from .provider_batch import cancel_provider_batch
from .store import read_expired_passes, write_stop

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

ReapOutcome = Literal["expired", "moved", "failed", "out_of_budget"]


class ReaperRun(BaseModel):
    """What one run did: how many rows it listed, and each one's outcome."""

    listed: int = 0
    outcomes: dict[ReapOutcome, int] = Field(default_factory=dict)


async def reap_expired_passes(*, now: datetime | None = None) -> ReaperRun:
    """One reaper run, within ``REAPER_BUDGET_SECONDS``; never raises."""
    expired_before = (now or datetime.now(timezone.utc)) - timedelta(
        seconds=REAP_GRACE_SECONDS
    )
    rows: list[DreamPassRecord] = []
    outcomes: Counter[ReapOutcome] = Counter()
    try:
        async with asyncio.timeout(REAPER_BUDGET_SECONDS):
            rows = await read_expired_passes(expired_before, limit=REAPER_ROW_LIMIT)
            await _reap_rows(rows, outcomes)
    except Exception:
        logger.warning("Dream pass reaper: the run stopped short", exc_info=True)
    outcomes["out_of_budget"] += len(rows) - sum(outcomes.values())
    run = ReaperRun(listed=len(rows), outcomes=dict(+outcomes))
    logger.info(f"Dream pass reaper: {run.listed} listed; {_counts(run)}")
    return run


async def _reap_rows(
    rows: list[DreamPassRecord], outcomes: Counter[ReapOutcome]
) -> None:
    """Reap *rows* in turn while the budget has room for another."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + REAPER_BUDGET_SECONDS - ROW_RESERVE_SECONDS
    for row in rows:
        if loop.time() > deadline:
            return
        outcomes[await _reap(row)] += 1


async def _reap(row: DreamPassRecord) -> ReapOutcome:
    """Close one dead pass and clean up after it, holding its scope's lock
    when it is free; ``failed`` when the store or Redis would not answer, and
    the row is listed again next run."""
    hold: str | None = None
    scope: MemoryScope | None = None
    try:
        scope = MemoryScope.build(row.user_id, row.expert_id)
        hold = await _take_scope(scope)
        return await _close(row, scope, took_scope=hold is not None)
    except Exception:
        logger.warning(f"Dream pass reaper: pass {row.id} failed", exc_info=True)
        return "failed"
    finally:
        if scope is not None and hold is not None:
            await release_dream_lock(scope, hold)


async def _close(
    row: DreamPassRecord, scope: MemoryScope, *, took_scope: bool
) -> ReapOutcome:
    """Expire *row*, then clean up after its pass; ``moved`` when the row was
    written since it was read, and nothing is touched."""
    closing = expired(_error(row), not_updated_since=row.updated_at)
    if not await write_stop(row.id, closing):
        logger.info(f"Dream pass reaper: pass {row.id} moved since it was read; left")
        return "moved"
    if row.provider_batch_id:
        await cancel_provider_batch(row.provider_batch_id)
    charged = await _charge_landed(row)
    await best_effort_cleanup(row.id)
    if not took_scope:
        # Held by someone: released only if it is still the dead pass's.
        await release_dream_lock(scope, row.lease_token)
    lock = "held by the reaper" if took_scope else "released if the dead pass's"
    logger.info(
        f"Dream pass reaper: pass {row.id} expired ({_error(row)}); "
        f"batch {row.provider_batch_id or 'none'}; charged {charged}; "
        f"scope lock {lock}"
    )
    return "expired"


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
