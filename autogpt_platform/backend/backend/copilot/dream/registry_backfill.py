"""Backfill the memory-scope schedule registry.

    poetry run memory-schedule-backfill [--dry-run] [--force]

Schedules the dream-system crons of every memory scope that should have them
and pauses the ones that should not. Scopes come from three places: every
account with a FalkorDB graph (``GRAPH.LIST``), every hired, unarchived
expert in Postgres, and every ACTIVE registry row. Accounts go through
``ensure_scope_scheduled``; live experts are resumed or, when their schedules
are paused, paused; an ACTIVE row whose expert is archived or gone is paused.

Safe to re-run: an account or live expert already registered in its owner's
timezone costs a few reads and no scheduler call, but a paused expert is
paused again (a state write and a remove call) on every run. The report
lists every scope whose registration, recording or pause failed, with the
cron and the reason, and the command exits 1 when there is any.

Needs what the backend services need: Postgres (Prisma connects directly),
Redis (registration markers), FalkorDB, the scheduler service for the
registration RPCs, and the feature flag backend, since crons are only
registered for owners whose flags are on.
"""

import argparse
import asyncio
import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.copilot.graphiti.graphs import list_account_graph_owners
from backend.copilot.graphiti.scope import MemoryScope
from backend.data import db, memory_schedule
from backend.data.memory_schedule import LiveExpertScope, MemoryScopeSchedule
from backend.util.feature_flag import initialize_feature_flags, shutdown_feature_flags
from backend.util.settings import AppEnvironment, Config

from .registry import ensure_scope_scheduled, pause_scope, resume_scope

logger = logging.getLogger(__name__)

Outcome = Literal["registered", "already_scheduled", "skipped", "failed"]
FAILURE_REASONS = frozenset(
    {
        "registration_failed",
        "record_failed",
        "registry_unavailable",
        "timezone_lookup_failed",
    }
)
_PAGE_SIZE = 500
_CONNECT_TIMEOUT_SECONDS = 60


class BackfillFailure(BaseModel):
    """One cron (or the pause) of one scope that the backfill could not settle."""

    scope_key: str
    user_id: str
    expert_id: str | None
    job: str
    reason: str


class BackfillReport(BaseModel):
    """What one backfill run found, what it did, and what failed."""

    dry_run: bool
    accounts: int = 0
    experts: int = 0
    experts_to_pause: int = 0
    graphs_without_user: int = 0

    registered: int = 0
    already_scheduled: int = 0
    skipped: int = 0
    paused: int = 0
    failed: int = 0
    failures: list[BackfillFailure] = Field(default_factory=list)


async def backfill_schedules(
    *, force: bool = False, dry_run: bool = False
) -> BackfillReport:
    """Schedule or pause every memory scope; ``dry_run`` only counts them.
    ``force`` re-registers every enabled cron of every scope it schedules."""
    report = BackfillReport(dry_run=dry_run)
    active_rows = await _active_rows()
    accounts = await _account_owners(report, active_rows)
    live = await _live_experts()
    live_ids = {expert.expert_id for expert in live}
    stale = [
        MemoryScope.for_expert(row.user_id, row.expert_id)
        for row in active_rows
        if row.expert_id is not None and row.expert_id not in live_ids
    ]
    report.accounts, report.experts = len(accounts), len(live)
    report.experts_to_pause = len(stale) + sum(expert.paused for expert in live)
    if dry_run:
        return report
    for user_id in accounts:
        scope = MemoryScope.for_user(user_id)
        _tally(report, scope, await ensure_scope_scheduled(scope, force_refresh=force))
    for expert in live:
        await _backfill_expert(report, expert, force)
    for scope in stale:
        await _pause(report, scope)
    return report


def main() -> None:
    """Entry point of ``poetry run memory-schedule-backfill``."""
    parser = argparse.ArgumentParser(
        description="Schedule the dream-system crons of every memory scope "
        "that should have them, and pause the rest. Safe to re-run."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count the scopes it would schedule or pause, and change nothing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-register every enabled cron even when the registry says it "
        "is current (e.g. after jobs were lost from the scheduler)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    try:
        report = asyncio.run(_run(force=args.force, dry_run=args.dry_run))
    except Exception:
        # Per-scope failures are listed in the report; this is Postgres or
        # FalkorDB being unreachable, or a listing failing outright.
        logger.exception("memory-schedule-backfill failed")
        raise SystemExit(1)
    print(report.model_dump_json(indent=2))
    if report.failures:
        # The report names each one; a re-run retries them.
        raise SystemExit(1)


async def _run(*, force: bool, dry_run: bool) -> BackfillReport:
    # One bounded attempt, not ``db.connect()``'s retry loop: an operator who
    # runs this where the database is unreachable wants an error, not an hour
    # of retries.
    await asyncio.wait_for(db.prisma.connect(), timeout=_CONNECT_TIMEOUT_SECONDS)
    # Eager, like the scheduler: a lazily started flag client answers
    # "off" until it connects, which would skip every registration.
    flags = not dry_run and Config().app_env != AppEnvironment.LOCAL
    try:
        if flags:
            initialize_feature_flags()
        return await backfill_schedules(force=force, dry_run=dry_run)
    finally:
        if flags:
            shutdown_feature_flags()
        await db.disconnect()


async def _backfill_expert(
    report: BackfillReport, expert: LiveExpertScope, force: bool
) -> None:
    scope = MemoryScope.for_expert(expert.user_id, expert.expert_id)
    if expert.paused:
        await _pause(report, scope)
        return
    _tally(report, scope, await resume_scope(scope, force_refresh=force))


async def _pause(report: BackfillReport, scope: MemoryScope) -> None:
    if await pause_scope(scope):
        report.paused += 1
        return
    report.failed += 1
    report.failures.append(_failure(scope, "pause", "pause_failed"))


def _tally(report: BackfillReport, scope: MemoryScope, results: dict[str, Any]) -> None:
    """Count one scope by its per-cron ensure results, listing each failed
    cron."""
    match _classify(results):
        case "failed":
            report.failed += 1
            report.failures.extend(
                _failure(scope, job, outcome["reason"])
                for job, outcome in results.items()
                if _failed(outcome)
            )
        case "registered":
            report.registered += 1
        case "already_scheduled":
            report.already_scheduled += 1
        case "skipped":
            report.skipped += 1


def _classify(results: dict[str, Any]) -> Outcome:
    """One scope's outcome from its per-cron results. A failed cron makes the
    scope failed whatever the other crons did."""
    outcomes = list(results.values())
    if any(_failed(outcome) for outcome in outcomes):
        return "failed"
    if any(outcome and not outcome.get("skipped") for outcome in outcomes):
        return "registered"
    if any(outcome is None for outcome in outcomes):
        return "already_scheduled"
    return "skipped"


def _failed(outcome: Any) -> bool:
    return bool(outcome) and outcome.get("reason") in FAILURE_REASONS


def _failure(scope: MemoryScope, job: str, reason: str) -> BackfillFailure:
    return BackfillFailure(
        scope_key=scope.scope_key,
        user_id=scope.owner_user_id,
        expert_id=scope.expert_id,
        job=job,
        reason=reason,
    )


async def _active_rows() -> list[MemoryScopeSchedule]:
    rows: list[MemoryScopeSchedule] = []
    while True:
        after = rows[-1].scope_key if rows else None
        page = await memory_schedule.list_active_scope_schedules(
            after=after, limit=_PAGE_SIZE
        )
        rows += page
        if len(page) < _PAGE_SIZE:
            return rows


async def _live_experts() -> list[LiveExpertScope]:
    experts: list[LiveExpertScope] = []
    while True:
        after = experts[-1].expert_id if experts else None
        page = await memory_schedule.list_live_expert_scopes(
            after=after, limit=_PAGE_SIZE
        )
        experts += page
        if len(page) < _PAGE_SIZE:
            return experts


async def _account_owners(
    report: BackfillReport, active_rows: list[MemoryScopeSchedule]
) -> list[str]:
    """Owners of an account graph that still have a user row, plus every
    account with an ACTIVE registry row."""
    owners = await list_account_graph_owners()
    existing: set[str] = set()
    for start in range(0, len(owners), _PAGE_SIZE):
        chunk = owners[start : start + _PAGE_SIZE]
        existing |= await memory_schedule.existing_user_ids(chunk)
    report.graphs_without_user = len(owners) - len(existing)
    registered = {row.user_id for row in active_rows if row.expert_id is None}
    return sorted(existing | registered)
