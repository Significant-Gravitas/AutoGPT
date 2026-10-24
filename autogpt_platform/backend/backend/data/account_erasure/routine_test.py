"""The routine runs every step in order, deletes nothing on a dry run or when
something is blocked, stops at the first failed step so a rerun can resume,
and a rerun of a finished erasure is a no-op."""

from unittest.mock import AsyncMock

import pytest
from prisma.enums import AccountDeletionSource

from backend.data.account_erasure import routine
from backend.data.account_erasure.models import (
    AccountErasureBlocked,
    Blocker,
    StepReport,
    StepStatus,
    Subject,
)
from backend.data.account_erasure.shared_agents import TransferPlan

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
STEP_NAMES = [
    "stop_work",
    "confirmation",
    "org_subscriptions",
    "shared_agents",
    "marketplace",
    "files",
    "memory",
    "posthog",
    "mailerlite",
    "database",
    "request",
]


class Steps:
    """Every step replaced by a mock that records the dry_run it got."""

    def __init__(self, monkeypatch, *, failing: str | None = None):
        self.calls: list[tuple[str, bool]] = []
        self.subjects: list[Subject] = []
        self.failing = failing
        targets = {
            "stop_work": "stop_work",
            "confirmation": "send_confirmation",
            "org_subscriptions": "cancel_org_subscriptions",
            "shared_agents": "transfer_shared_agents",
            "marketplace": "delete_marketplace",
            "files": "delete_files",
            "memory": "delete_memory",
            "posthog": "delete_posthog_person",
            "mailerlite": "forget_mailerlite_subscriber",
            "database": "delete_rows",
            "request": "complete_request",
        }
        for name, attribute in targets.items():
            monkeypatch.setattr(routine, attribute, self._step(name))

    def _step(self, name: str):
        async def step(subject: Subject, *, dry_run: bool) -> StepReport:
            self.calls.append((name, dry_run))
            self.subjects.append(subject)
            if name == self.failing:
                raise RuntimeError("sam@example.com is down")
            status = StepStatus.WOULD_DO if dry_run else StepStatus.DONE
            return StepReport(name=name, status=status, counts={"rows": 0})

        return step


@pytest.fixture
def account(monkeypatch):
    subject = Subject(
        user_id=USER_ID,
        email="sam@example.com",
        has_user_row=True,
        request_source=AccountDeletionSource.SETTINGS,
    )
    monkeypatch.setattr(routine, "load_subject", AsyncMock(return_value=subject))
    monkeypatch.setattr(
        routine, "plan_transfer", AsyncMock(return_value=TransferPlan())
    )
    monkeypatch.setattr(routine, "find_blockers", AsyncMock(return_value=[]))
    return subject


@pytest.mark.asyncio
async def test_a_dry_run_runs_every_step_as_a_dry_run(monkeypatch, account):
    steps = Steps(monkeypatch)

    report = await routine.erase_account(USER_ID)

    assert steps.calls == [(name, True) for name in STEP_NAMES]
    assert report.dry_run and not report.complete
    assert [s.status for s in report.steps] == [StepStatus.WOULD_DO] * len(STEP_NAMES)


@pytest.mark.asyncio
async def test_an_apply_runs_every_step_in_order_and_completes(monkeypatch, account):
    steps = Steps(monkeypatch)

    report = await routine.erase_account(USER_ID, dry_run=False)

    assert steps.calls == [(name, False) for name in STEP_NAMES]
    assert report.complete


@pytest.mark.asyncio
async def test_a_blocker_stops_an_apply_before_any_step(monkeypatch, account):
    steps = Steps(monkeypatch)
    blocker = Blocker(name="not_closed", count=0, resolution="x")
    monkeypatch.setattr(routine, "find_blockers", AsyncMock(return_value=[blocker]))

    with pytest.raises(AccountErasureBlocked) as raised:
        await routine.erase_account(USER_ID, dry_run=False)

    assert steps.calls == []
    assert raised.value.report.blockers == [blocker]


@pytest.mark.asyncio
async def test_a_dry_run_reports_blockers_instead_of_raising(monkeypatch, account):
    Steps(monkeypatch)
    blocker = Blocker(name="shared_orgs_without_another_owner", count=1, resolution="x")
    monkeypatch.setattr(routine, "find_blockers", AsyncMock(return_value=[blocker]))

    report = await routine.erase_account(USER_ID)

    assert report.blockers == [blocker]
    assert len(report.steps) == len(STEP_NAMES)


@pytest.mark.asyncio
async def test_a_failed_step_stops_the_apply_and_keeps_the_rows(monkeypatch, account):
    steps = Steps(monkeypatch, failing="mailerlite")

    report = await routine.erase_account(USER_ID, dry_run=False)

    ran = [name for name, _ in steps.calls]
    assert ran == STEP_NAMES[: STEP_NAMES.index("mailerlite") + 1]
    assert "database" not in ran
    # The request stays pending, for the next run to finish.
    assert "request" not in ran
    assert report.failed and not report.complete
    failed = report.steps[-1]
    assert failed.status == StepStatus.FAILED
    # The exception type only: its message carried an email.
    assert "sam@example.com" not in failed.note
    assert "RuntimeError" in failed.note


@pytest.mark.asyncio
async def test_a_failed_step_does_not_stop_a_dry_run(monkeypatch, account):
    steps = Steps(monkeypatch, failing="files")

    report = await routine.erase_account(USER_ID)

    assert [name for name, _ in steps.calls] == STEP_NAMES
    assert report.failed


@pytest.mark.asyncio
async def test_a_second_run_finds_nothing_and_still_completes(monkeypatch):
    gone = Subject(
        user_id=USER_ID,
        request_source=AccountDeletionSource.SETTINGS,
        request_completed=True,
    )
    monkeypatch.setattr(routine, "load_subject", AsyncMock(return_value=gone))
    monkeypatch.setattr(
        routine, "plan_transfer", AsyncMock(return_value=TransferPlan())
    )
    monkeypatch.setattr(routine, "find_blockers", AsyncMock(return_value=[]))
    Steps(monkeypatch)

    report = await routine.erase_account(USER_ID, dry_run=False)

    assert not report.account_found
    assert report.complete
    assert all(set(s.counts.values()) <= {0} for s in report.steps)


@pytest.mark.asyncio
async def test_every_step_sees_the_agents_that_go_to_the_placeholder(
    monkeypatch, account
):
    plan = TransferPlan(graph_ids=["g1"], listing_ids=["l1"])
    monkeypatch.setattr(routine, "plan_transfer", AsyncMock(return_value=plan))
    steps = Steps(monkeypatch)

    await routine.erase_account(USER_ID)

    assert {tuple(s.shared_graph_ids) for s in steps.subjects} == {("g1",)}
    assert {tuple(s.kept_listing_ids) for s in steps.subjects} == {("l1",)}


@pytest.mark.asyncio
@pytest.mark.parametrize("closing", [False, True])
async def test_closing_reaches_the_blocker_check(monkeypatch, account, closing):
    Steps(monkeypatch)
    find_blockers = AsyncMock(return_value=[])
    monkeypatch.setattr(routine, "find_blockers", find_blockers)

    await routine.erase_account(USER_ID, closing=closing)

    assert find_blockers.await_args.kwargs["closing"] is closing


@pytest.mark.asyncio
async def test_a_malformed_id_runs_no_step(monkeypatch):
    """The ID check is in the preflight, so not even the work stop (which
    reaches the scheduler) sees a malformed ID."""
    bad = Subject(user_id="../everyone", request_source=AccountDeletionSource.SETTINGS)
    monkeypatch.setattr(routine, "load_subject", AsyncMock(return_value=bad))
    monkeypatch.setattr(
        routine, "plan_transfer", AsyncMock(return_value=TransferPlan())
    )
    steps = Steps(monkeypatch)

    with pytest.raises(AccountErasureBlocked) as raised:
        await routine.erase_account("../everyone", dry_run=False)

    assert steps.calls == []
    assert [b.name for b in raised.value.report.blockers] == ["invalid_account_id"]
