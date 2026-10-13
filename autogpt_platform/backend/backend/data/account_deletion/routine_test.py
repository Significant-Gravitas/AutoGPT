"""The routine runs every step in order, deletes nothing on a dry run or when
something is blocked, stops at the first failed step so a rerun can resume,
and a rerun of a finished deletion is a no-op."""

from unittest.mock import AsyncMock

import pytest

from backend.data.account_deletion import routine
from backend.data.account_deletion.models import (
    AccountDeletionBlocked,
    Blocker,
    StepReport,
    StepStatus,
    Subject,
)
from backend.data.account_deletion.preflight import SharedAgentCounts

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
STEP_NAMES = [
    "sign_in",
    "schedules",
    "executions",
    "subscriptions",
    "integrations",
    "marketplace",
    "files",
    "memory",
    "posthog",
    "mailerlite",
    "database",
]


class Steps:
    """Every step replaced by a mock that records the dry_run it got."""

    def __init__(self, monkeypatch, *, failing: str | None = None):
        self.calls: list[tuple[str, bool]] = []
        self.failing = failing
        targets = {
            "sign_in": "disable_sign_in",
            "schedules": "stop_schedules",
            "executions": "stop_executions",
            "subscriptions": "cancel_subscriptions",
            "integrations": "release_integrations",
            "marketplace": "delete_marketplace",
            "files": "delete_files",
            "memory": "delete_memory",
            "posthog": "delete_posthog_person",
            "mailerlite": "forget_mailerlite_subscriber",
            "database": "delete_rows",
        }
        for name, attribute in targets.items():
            monkeypatch.setattr(routine, attribute, self._step(name))

    def _step(self, name: str):
        async def step(subject: Subject, *, dry_run: bool, **kwargs) -> StepReport:
            self.calls.append((name, dry_run))
            if name == self.failing:
                raise RuntimeError("sam@example.com is down")
            status = StepStatus.WOULD_DO if dry_run else StepStatus.DONE
            return StepReport(name=name, status=status, counts={"rows": 0})

        return step


@pytest.fixture
def account(monkeypatch):
    subject = Subject(user_id=USER_ID, email="sam@example.com", has_user_row=True)
    monkeypatch.setattr(routine, "load_subject", AsyncMock(return_value=subject))
    monkeypatch.setattr(
        routine, "shared_agent_counts", AsyncMock(return_value=SharedAgentCounts())
    )
    monkeypatch.setattr(routine, "find_blockers", AsyncMock(return_value=[]))
    return subject


@pytest.mark.asyncio
async def test_a_dry_run_runs_every_step_as_a_dry_run(monkeypatch, account):
    steps = Steps(monkeypatch)

    report = await routine.delete_account(USER_ID)

    assert steps.calls == [(name, True) for name in STEP_NAMES]
    assert report.dry_run and not report.complete
    assert [s.status for s in report.steps] == [StepStatus.WOULD_DO] * len(STEP_NAMES)


@pytest.mark.asyncio
async def test_an_apply_runs_every_step_in_order_and_completes(monkeypatch, account):
    steps = Steps(monkeypatch)

    report = await routine.delete_account(USER_ID, dry_run=False)

    assert steps.calls == [(name, False) for name in STEP_NAMES]
    assert report.complete


@pytest.mark.asyncio
async def test_a_blocker_stops_an_apply_before_any_step(monkeypatch, account):
    steps = Steps(monkeypatch)
    blocker = Blocker(name="agents_used_by_other_accounts", count=2, resolution="x")
    monkeypatch.setattr(routine, "find_blockers", AsyncMock(return_value=[blocker]))

    with pytest.raises(AccountDeletionBlocked) as raised:
        await routine.delete_account(USER_ID, dry_run=False)

    assert steps.calls == []
    assert raised.value.report.blockers == [blocker]


@pytest.mark.asyncio
async def test_a_dry_run_reports_blockers_instead_of_raising(monkeypatch, account):
    Steps(monkeypatch)
    blocker = Blocker(name="shared_orgs_without_another_owner", count=1, resolution="x")
    monkeypatch.setattr(routine, "find_blockers", AsyncMock(return_value=[blocker]))

    report = await routine.delete_account(USER_ID)

    assert report.blockers == [blocker]
    assert len(report.steps) == len(STEP_NAMES)


@pytest.mark.asyncio
async def test_a_failed_step_stops_the_apply_and_keeps_the_rows(monkeypatch, account):
    steps = Steps(monkeypatch, failing="mailerlite")

    report = await routine.delete_account(USER_ID, dry_run=False)

    ran = [name for name, _ in steps.calls]
    assert ran == STEP_NAMES[: STEP_NAMES.index("mailerlite") + 1]
    assert "database" not in ran
    assert report.failed and not report.complete
    failed = report.steps[-1]
    assert failed.status == StepStatus.FAILED
    # The exception type only: its message carried an email.
    assert "sam@example.com" not in failed.note
    assert "RuntimeError" in failed.note


@pytest.mark.asyncio
async def test_a_failed_step_does_not_stop_a_dry_run(monkeypatch, account):
    steps = Steps(monkeypatch, failing="files")

    report = await routine.delete_account(USER_ID)

    assert [name for name, _ in steps.calls] == STEP_NAMES
    assert report.failed


@pytest.mark.asyncio
async def test_a_second_run_finds_nothing_and_still_completes(monkeypatch):
    gone = Subject(user_id=USER_ID)
    monkeypatch.setattr(routine, "load_subject", AsyncMock(return_value=gone))
    monkeypatch.setattr(
        routine, "shared_agent_counts", AsyncMock(return_value=SharedAgentCounts())
    )
    monkeypatch.setattr(routine, "find_blockers", AsyncMock(return_value=[]))
    Steps(monkeypatch)

    report = await routine.delete_account(USER_ID, dry_run=False)

    assert not report.account_found
    assert report.complete
    assert all(set(s.counts.values()) <= {0} for s in report.steps)


@pytest.mark.asyncio
async def test_shared_agent_removal_reaches_the_marketplace_step(monkeypatch, account):
    monkeypatch.setattr(
        routine,
        "shared_agent_counts",
        AsyncMock(return_value=SharedAgentCounts(library_entries=3, presets=1)),
    )
    Steps(monkeypatch)
    marketplace = AsyncMock(
        return_value=StepReport(name="marketplace", status=StepStatus.DONE)
    )
    monkeypatch.setattr(routine, "delete_marketplace", marketplace)

    await routine.delete_account(USER_ID, dry_run=False, remove_shared_agents=True)

    kwargs = marketplace.await_args.kwargs
    assert kwargs["remove_shared_agents"] is True
    assert kwargs["shared_agent_rows"] == 4
    assert kwargs["dry_run"] is False
