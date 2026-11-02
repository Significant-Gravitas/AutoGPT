"""Erase every closed account in turn: the periodic job and `erase-account
--pending` both run `erase_pending`.

The job runs inside the database manager, which has the Prisma connection
the scheduler lacks. The scheduler only starts it over RPC
(`start_account_erasure_sweep`), and only while
`account_erasure_sweep_enabled` is on (off by default). Each sweep also
purges the ban records past their date (`ban_record.py`).
"""

import asyncio
import logging

from backend.data.account_erasure.ban_record import purge_expired_ban_records
from backend.data.account_erasure.models import (
    AccountErasureBlocked,
    ErasureReport,
    StepReport,
    StepStatus,
)
from backend.data.account_erasure.request import pending_user_ids
from backend.data.account_erasure.routine import erase_account

logger = logging.getLogger(__name__)


async def erase_pending(*, dry_run: bool = True) -> list[ErasureReport]:
    """Erase each account whose 14 days to keep it are over and that is not
    erased yet, oldest first. One that is blocked or fails is reported and
    left for the next run; the rest still go."""
    return await erase_each(await pending_user_ids(), dry_run=dry_run)


async def erase_each(user_ids: list[str], *, dry_run: bool) -> list[ErasureReport]:
    """Erase these accounts in turn; a blocked one is reported, not raised."""
    return [await _erase_one(user_id, dry_run) for user_id in user_ids]


async def _erase_one(user_id: str, dry_run: bool) -> ErasureReport:
    try:
        report = await erase_account(user_id, dry_run=dry_run)
    except AccountErasureBlocked as e:
        report = e.report
    except Exception as e:
        # Outside any step (reading the account, the lease, the claim): this
        # account waits for the next run, and the rest still go. The type
        # only, as a message can quote the address.
        logger.error(f"Account erasure of {user_id} failed: {type(e).__name__}")
        report = ErasureReport(
            user_id=user_id,
            dry_run=dry_run,
            steps=[
                StepReport(
                    name="erasure", status=StepStatus.FAILED, note=type(e).__name__
                )
            ],
        )
    if not dry_run:
        _log(report)
    return report


def _log(report: ErasureReport) -> None:
    """Names and counts only: the report never carries an email or a path."""
    if report.not_claimed:
        logger.info(
            f"Account erasure of {report.user_id} skipped: {report.not_claimed}"
        )
        return
    if any(blocker.name == "not_closed" for blocker in report.blockers):
        # Listed as pending at the start of the sweep, then kept by its owner.
        logger.info(f"Account erasure of {report.user_id} skipped: kept by its owner")
        return
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


async def purge_ban_records() -> int:
    """Purge the ban records past their date, on its own: a failure here is
    logged and never stops an erasure."""
    try:
        purged = await purge_expired_ban_records()
    except Exception as e:
        logger.error(f"Purging expired ban records failed: {type(e).__name__}")
        return 0
    if purged:
        logger.info(f"Purged {purged} expired ban record(s)")
    return purged


async def _run_sweep() -> None:
    await purge_ban_records()
    try:
        reports = await erase_pending(dry_run=False)
    except Exception as e:
        logger.error(f"Account erasure sweep failed: {type(e).__name__}")
        return
    if reports:
        done = sum(1 for report in reports if report.complete)
        logger.info(f"Account erasure sweep erased {done} of {len(reports)}")
