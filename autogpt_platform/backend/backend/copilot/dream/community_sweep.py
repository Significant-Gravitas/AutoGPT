"""The weekly community rebuild of every memory scope, from one hourly job.

The scheduler runs :func:`rebuild_due_communities` every hour. It reads the
ACTIVE scopes whose last clean rebuild (before the first, their row's
creation) is a week old, the oldest first and at most ``SWEEP_BATCH``, and
rebuilds them ``SWEEP_CONCURRENCY`` at a time. The rebuild's activity gate
skips a graph with nothing new, and the owner checks run only for a graph it
would rebuild, so the empty graph most scopes have costs two graph reads and
a stamp. A pass that ends without an error is stamped, which moves its scope
a week on; a refused or failed one stays due.

This replaced a cron per scope at owner-local Monday 04:00: every scope in a
timezone fell due at one instant, and APScheduler dispatched them all in one
pass under its jobstore lock, ahead of every other job.
"""

import asyncio
import logging
from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel

from backend.copilot.graphiti.communities import (
    REBUILD_REFUSED,
    rebuild_communities_for_user,
)
from backend.copilot.graphiti.scope import MemoryScope
from backend.data.db_accessors import memory_schedule_db
from backend.data.memory_schedule import MemoryScopeSchedule
from backend.data.user import is_user_banned

from .deadline import within_deadline
from .scope_crons import ExpertStatusLookup, memory_scope_may_fire, record_scope_run
from .scope_jobs import scope_label

logger = logging.getLogger(__name__)

REBUILD_INTERVAL = timedelta(days=7)
# A week of hourly runs covers 168 batches: 168,000 scopes rebuilt weekly.
SWEEP_BATCH = 1000
SWEEP_CONCURRENCY = 4

Outcome = Literal["built", "unchanged", "refused", "failed"]


class SweepReport(BaseModel):
    """How many scopes one sweep found due, and how many ended each way."""

    due: int = 0
    outcomes: dict[Outcome, int] = {}


async def rebuild_due_communities(
    expert_status: ExpertStatusLookup, rebuild_timeout: float
) -> SweepReport:
    """One sweep (above), each rebuild bounded by ``rebuild_timeout``. Never
    raises: a listing that fails rebuilds nothing until the next hour."""
    try:
        rows = await within_deadline(
            memory_schedule_db().list_due_community_scopes(
                due_before=datetime.now(UTC) - REBUILD_INTERVAL,
                limit=SWEEP_BATCH,
            )
        )
    except Exception:
        logger.warning("Community sweep skipped: no due scopes listed", exc_info=True)
        return SweepReport()
    turns = asyncio.Semaphore(SWEEP_CONCURRENCY)

    async def take_turn(row: MemoryScopeSchedule) -> Outcome:
        async with turns:
            return await _rebuild(row, expert_status, rebuild_timeout)

    outcomes = await asyncio.gather(*(take_turn(row) for row in rows))
    report = SweepReport(due=len(rows), outcomes=dict(Counter(outcomes)))
    logger.info(f"Community sweep: {report.due} due, {report.outcomes}")
    return report


async def _rebuild(
    row: MemoryScopeSchedule, expert_status: ExpertStatusLookup, timeout: float
) -> Outcome:
    try:
        scope = MemoryScope.build(row.user_id, row.expert_id)
        rebuild = rebuild_communities_for_user(
            scope.owner_user_id,
            expert_id=scope.expert_id,
            may_rebuild=lambda: _owner_allows(scope, expert_status),
        )
        # Cut off between its delete and its build, a rebuild would leave the
        # graph half built: a timeout only stops the sweep waiting for it.
        result = await asyncio.wait_for(asyncio.shield(rebuild), timeout)
    except Exception:
        logger.warning(
            f"Community rebuild of scope {row.scope_key[:12]} failed", exc_info=True
        )
        return "failed"
    if result.get("error"):
        return "failed"
    if result.get("skip_reason") == REBUILD_REFUSED:
        return "refused"
    await record_scope_run(scope, "community")
    return "unchanged" if result.get("skipped") else "built"


async def _owner_allows(scope: MemoryScope, expert_status: ExpertStatusLookup) -> bool:
    """What a dream-system cron body asks before it runs: the owner is not
    banned, and the registry row, and an expert's lifecycle, let it run."""
    try:
        banned = await within_deadline(is_user_banned(scope.owner_user_id))
    except Exception:
        logger.warning(
            f"Community rebuild of {scope_label(scope)} skipped: ban unreadable",
            exc_info=True,
        )
        return False
    if banned:
        logger.info(f"Community rebuild of {scope_label(scope)} skipped: owner banned")
        return False
    return await memory_scope_may_fire(scope, expert_status)
