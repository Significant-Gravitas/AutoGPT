"""Erase every closed account in turn: the periodic job and `erase-account
--pending` both run `erase_pending`.

The job runs inside the database manager, which has the Prisma connection
the scheduler lacks. The scheduler only starts it over RPC
(`start_account_erasure_sweep`), and only while
`account_erasure_sweep_enabled` is on (off by default).
"""

import asyncio
import logging
from datetime import timedelta

from backend.data.account_erasure.models import (
    AccountErasureBlocked,
    ErasureReport,
    StepStatus,
)
from backend.data.account_erasure.request import PENDING_GRACE, pending_user_ids
from backend.data.account_erasure.routine import erase_account

logger = logging.getLogger(__name__)


async def erase_pending(
    *, dry_run: bool = True, grace: timedelta = PENDING_GRACE
) -> list[ErasureReport]:
    """Erase each account closed at least `grace` ago and not erased yet,
    oldest first. One that is blocked or fails is reported and left for the
    next run; the rest still go."""
    return await erase_each(await pending_user_ids(grace=grace), dry_run=dry_run)


async def erase_each(user_ids: list[str], *, dry_run: bool) -> list[ErasureReport]:
    """Erase these accounts in turn; a blocked one is reported, not raised."""
    return [await _erase_one(user_id, dry_run) for user_id in user_ids]


async def _erase_one(user_id: str, dry_run: bool) -> ErasureReport:
    try:
        report = await erase_account(user_id, dry_run=dry_run)
    except AccountErasureBlocked as e:
        report = e.report
    if not dry_run:
        _log(report)
    return report


def _log(report: ErasureReport) -> None:
    """Names and counts only: the report never carries an email or a path."""
    if report.blockers:
        names = ", ".join(blocker.name for blocker in report.blockers)
        logger.error(
            f"Account erasure of {report.user_id} is blocked ({names}); it "
            "needs a person: see docs/engineering/account-deletion.md"
        )
        return
    failed = [s.name for s in report.steps if s.status == StepStatus.FAILED]
    if failed:
        logger.error(
            f"Account erasure of {report.user_id} stopped at {failed[0]}; the "
            "next run retries it"
        )
        return
    skipped = [s.name for s in report.steps if s.status == StepStatus.SKIPPED]
    if skipped:
        logger.warning(
            f"Account erasure of {report.user_id} skipped {', '.join(skipped)}, "
            "not configured on this service"
        )


async def start_account_erasure_sweep() -> bool:
    """Start a sweep in the background and return at once.

    The scheduler reaches this over RPC, whose call timeout is shorter than a
    sweep of many accounts can take and which retries on a timeout; waiting
    for the sweep there would start a second one alongside the first. Returns
    False when a sweep is already running in this process.
    """
    if any(not task.done() for task in _sweeps):
        return False
    task = asyncio.get_running_loop().create_task(_run_sweep())
    _sweeps.add(task)
    task.add_done_callback(_sweeps.discard)
    return True


_sweeps: set[asyncio.Task] = set()


async def _run_sweep() -> None:
    try:
        reports = await erase_pending(dry_run=False)
    except Exception:
        logger.exception("Account erasure sweep failed")
        return
    if reports:
        done = sum(1 for report in reports if report.complete)
        logger.info(f"Account erasure sweep erased {done} of {len(reports)}")
