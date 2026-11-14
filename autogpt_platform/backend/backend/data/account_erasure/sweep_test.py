"""The sweep erases every pending account in turn: a blocked or failed one is
logged and left for the next run, and the rest still go. Each sweep purges
the ban records past their date. Only one sweep runs at a time in a process,
and the scheduler's job does nothing while the switch is off."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.data.account_erasure import sweep
from backend.data.account_erasure.models import (
    AccountErasureBlocked,
    Blocker,
    ErasureReport,
    StepReport,
    StepStatus,
)
from backend.executor import scheduler


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
async def test_a_sweep_purges_the_ban_records_past_their_date(monkeypatch):
    monkeypatch.setattr(sweep, "erase_pending", AsyncMock(return_value=[]))
    purge = AsyncMock(return_value=2)
    monkeypatch.setattr(sweep, "purge_expired_ban_records", purge)

    await sweep._run_sweep()

    purge.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_an_unclaimed_account_is_logged_as_skipped_not_as_an_error(
    monkeypatch, caplog
):
    """The owner kept it after the sweep listed it: nothing needs a person."""
    report = ErasureReport(user_id="a", dry_run=False, not_claimed="kept")
    monkeypatch.setattr(sweep, "erase_account", AsyncMock(return_value=report))

    with caplog.at_level("INFO", logger=sweep.logger.name):
        await sweep.erase_each(["a"], dry_run=False)

    assert [r.levelname for r in caplog.records] == ["INFO"]


@pytest.mark.asyncio
async def test_an_account_kept_after_the_sweep_listed_it_is_not_an_error(
    monkeypatch, caplog
):
    """Its request is gone by the time its turn comes, so it reads as not
    closed: that needs no person."""
    blocked = ErasureReport(
        user_id="a",
        dry_run=False,
        blockers=[Blocker(name="not_closed", count=0, resolution="")],
    )
    monkeypatch.setattr(
        sweep, "erase_account", AsyncMock(side_effect=AccountErasureBlocked(blocked))
    )

    with caplog.at_level("INFO", logger=sweep.logger.name):
        await sweep.erase_each(["a"], dry_run=False)

    assert [r.levelname for r in caplog.records] == ["INFO"]


@pytest.mark.asyncio
async def test_only_one_sweep_runs_at_a_time(monkeypatch):
    monkeypatch.setattr(sweep, "purge_expired_ban_records", AsyncMock(return_value=0))
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


@pytest.mark.parametrize("enabled", [False, True])
def test_the_scheduler_job_starts_the_sweep_only_when_switched_on(monkeypatch, enabled):
    """The switch is checked in the job too, so turning it off also stops a
    job that was persisted while it was on."""
    monkeypatch.setattr(scheduler.config, "account_erasure_sweep_enabled", enabled)
    client = MagicMock()
    client.start_account_erasure_sweep = AsyncMock(return_value=True)
    monkeypatch.setattr(scheduler, "get_database_manager_async_client", lambda: client)
    started: list = []

    def run_async(coro, *args, **kwargs):
        # Records the RPC without running an event loop of its own.
        started.append(coro)
        coro.close()

    monkeypatch.setattr(scheduler, "run_async", run_async)

    scheduler.erase_closed_accounts()

    assert len(started) == int(enabled)


@pytest.mark.asyncio
async def test_an_error_outside_the_steps_skips_that_account_only(monkeypatch, caplog):
    """Reading the account, the lease or the claim failing: logged by type,
    never by message (it can quote the address), and the rest still go."""

    async def erase(user_id, *, dry_run):
        if user_id == "b":
            raise RuntimeError("sam@example.com is unreachable")
        return _report(user_id, dry_run=dry_run)

    monkeypatch.setattr(sweep, "erase_account", erase)

    with caplog.at_level("ERROR", logger=sweep.logger.name):
        reports = await sweep.erase_each(["a", "b", "c"], dry_run=False)

    assert [r.user_id for r in reports] == ["a", "b", "c"]
    assert [r.complete for r in reports] == [True, False, True]
    assert "RuntimeError" in caplog.text
    assert "sam@example.com" not in caplog.text


@pytest.mark.asyncio
async def test_a_failing_purge_does_not_stop_the_sweep(monkeypatch):
    erase = AsyncMock(return_value=[])
    monkeypatch.setattr(sweep, "erase_pending", erase)
    monkeypatch.setattr(
        sweep, "purge_expired_ban_records", AsyncMock(side_effect=RuntimeError())
    )

    await sweep._run_sweep()

    erase.assert_awaited_once()
