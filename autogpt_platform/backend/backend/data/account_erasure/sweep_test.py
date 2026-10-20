"""The sweep erases every pending account in turn: a blocked or failed one is
logged and left for the next run, and the rest still go. Only one sweep runs
at a time in a process."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.data.account_erasure import sweep
from backend.data.account_erasure.models import (
    AccountErasureBlocked,
    Blocker,
    ErasureReport,
    StepReport,
    StepStatus,
)


def _report(user_id: str, *, dry_run: bool, status=StepStatus.DONE, blocked=False):
    return ErasureReport(
        user_id=user_id,
        dry_run=dry_run,
        blockers=(
            [Blocker(name="shared_orgs_without_another_owner", count=1, resolution="")]
            if blocked
            else []
        ),
        steps=[StepReport(name="files", status=status)],
    )


@pytest.mark.asyncio
async def test_each_pending_account_is_erased_and_a_blocked_one_skipped(
    monkeypatch,
):
    monkeypatch.setattr(
        sweep, "pending_user_ids", AsyncMock(return_value=["a", "b", "c"])
    )

    async def erase(user_id, *, dry_run):
        if user_id == "b":
            raise AccountErasureBlocked(_report("b", dry_run=dry_run, blocked=True))
        status = StepStatus.FAILED if user_id == "c" else StepStatus.DONE
        return _report(user_id, dry_run=dry_run, status=status)

    monkeypatch.setattr(sweep, "erase_account", erase)

    reports = await sweep.erase_pending(dry_run=False)

    assert [r.user_id for r in reports] == ["a", "b", "c"]
    assert [r.complete for r in reports] == [True, False, False]
    assert reports[1].blockers[0].name == "shared_orgs_without_another_owner"


@pytest.mark.asyncio
async def test_only_one_sweep_runs_at_a_time(monkeypatch):
    release = asyncio.Event()

    async def erase_pending(*, dry_run):
        await release.wait()
        return []

    monkeypatch.setattr(sweep, "erase_pending", erase_pending)

    assert await sweep.start_account_erasure_sweep() is True
    assert await sweep.start_account_erasure_sweep() is False
    release.set()
    await asyncio.gather(*sweep._sweeps)
    assert await sweep.start_account_erasure_sweep() is True
    await asyncio.gather(*sweep._sweeps)


@pytest.mark.asyncio
async def test_a_sweep_that_fails_is_logged_not_raised(monkeypatch):
    monkeypatch.setattr(
        sweep, "erase_pending", AsyncMock(side_effect=RuntimeError("db down"))
    )

    await sweep._run_sweep()
