"""The routine claims the request before any step, then runs every step in
order; it deletes nothing on a dry run, when something is blocked or when the
request can't be claimed (kept, not due, or erased already), and stops at the
first failed step so a rerun can resume."""

from types import SimpleNamespace
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
REQUEST_ID = "9d4f6a2b-1c3e-4b5a-8f7d-2e6c0a9b1d34"
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
    "ban_record",
    "database",
    "request",
]


class Steps:
    """Every step replaced by a mock that records the dry_run it got."""

    def __init__(
        self,
        monkeypatch,
        *,
        failing: str | None = None,
        skipping: str | None = None,
    ):
        self.calls: list[tuple[str, bool]] = []
        self.subjects: list[Subject] = []
        self.failing = failing
        self.skipping = skipping
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
            "ban_record": "keep_ban_record",
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
            if name == self.skipping:
                return StepReport(name=name, status=StepStatus.SKIPPED, note="unset")
            status = StepStatus.WOULD_DO if dry_run else StepStatus.DONE
            return StepReport(name=name, status=status, counts={"rows": 0})

        return step


@pytest.fixture(autouse=True)
def lease(monkeypatch):
    """The per-account lease, free unless a test takes it."""
    held = SimpleNamespace(
        acquire=AsyncMock(return_value=object()), release=AsyncMock()
    )
    monkeypatch.setattr(routine, "acquire_erasure_lease", held.acquire)
    monkeypatch.setattr(routine, "release_erasure_lease", held.release)
    held.keep = AsyncMock(return_value=True)
    monkeypatch.setattr(routine, "keep_erasure_lease", held.keep)
    return held


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
    monkeypatch.setattr(routine, "claim_request", AsyncMock(return_value=REQUEST_ID))
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
async def test_an_apply_claims_the_request_before_any_step(monkeypatch, account):
    steps = Steps(monkeypatch)
    order: list[str] = []

    async def claim(user_id):
        order.append(f"claim {user_id}")
        assert steps.calls == []
        return REQUEST_ID

    monkeypatch.setattr(routine, "claim_request", claim)

    await routine.erase_account(USER_ID, dry_run=False)

    assert order == [f"claim {USER_ID}"]
    assert {s.request_id for s in steps.subjects} == {REQUEST_ID}


@pytest.mark.asyncio
async def test_a_dry_run_claims_nothing(monkeypatch, account):
    Steps(monkeypatch)
    claim = AsyncMock()
    monkeypatch.setattr(routine, "claim_request", claim)

    await routine.erase_account(USER_ID)

    claim.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "completed, reason",
    [
        (False, "kept by its owner, or its 14 days are not over"),
        (True, "erased already"),
    ],
)
async def test_a_request_that_cannot_be_claimed_runs_no_step(
    monkeypatch, account, completed, reason
):
    """The owner kept the account after the dry run read it, or an earlier
    run finished: nothing is deleted, and nothing reads as done."""
    account.request_completed = completed
    monkeypatch.setattr(routine, "claim_request", AsyncMock(return_value=None))
    steps = Steps(monkeypatch)

    report = await routine.erase_account(USER_ID, dry_run=False)

    assert steps.calls == []
    assert report.not_claimed == reason
    assert not report.complete


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


@pytest.mark.asyncio
async def test_a_skipped_step_does_not_stop_the_apply(monkeypatch, account):
    """A setting this stack lacks: logged by the sweep, not a stop."""
    steps = Steps(monkeypatch, skipping="posthog")

    report = await routine.erase_account(USER_ID, dry_run=False)

    assert [name for name, _ in steps.calls] == STEP_NAMES
    assert report.complete


@pytest.mark.asyncio
async def test_an_account_another_run_is_erasing_is_left_alone(
    monkeypatch, account, lease
):
    lease.acquire.return_value = None
    steps = Steps(monkeypatch)
    load = AsyncMock()
    monkeypatch.setattr(routine, "load_subject", load)

    report = await routine.erase_account(USER_ID, dry_run=False)

    load.assert_not_awaited()
    assert steps.calls == []
    assert report.not_claimed == "another run is erasing it"
    lease.release.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_lease_is_released_after_an_apply_and_never_taken_by_a_dry_run(
    monkeypatch, account, lease
):
    Steps(monkeypatch, failing="files")

    await routine.erase_account(USER_ID)
    lease.acquire.assert_not_awaited()

    await routine.erase_account(USER_ID, dry_run=False)
    lease.acquire.assert_awaited_once_with(USER_ID)
    lease.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_lease_is_renewed_before_each_step(monkeypatch, account, lease):
    steps = Steps(monkeypatch)

    await routine.erase_account(USER_ID, dry_run=False)

    assert lease.keep.await_count == len(steps.calls) == len(STEP_NAMES)


@pytest.mark.asyncio
async def test_a_lost_lease_stops_the_erasure_before_the_next_step(
    monkeypatch, account, lease
):
    """A long erasure whose lease lapsed may have another run on it now."""
    renewals = iter([True, True, False])
    lease.keep.side_effect = lambda *_: next(renewals)
    steps = Steps(monkeypatch)

    report = await routine.erase_account(USER_ID, dry_run=False)

    assert [name for name, _ in steps.calls] == STEP_NAMES[:2]
    assert report.steps[-1].name == STEP_NAMES[2]
    assert report.steps[-1].status == StepStatus.FAILED
    assert "lease" in report.steps[-1].note
    assert not report.complete
    lease.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_failure_logs_no_traceback_for_steps_that_handle_the_address(
    monkeypatch, account, caplog
):
    Steps(monkeypatch, failing="mailerlite")

    with caplog.at_level("ERROR", logger=routine.logger.name):
        await routine.erase_account(USER_ID, dry_run=False)

    (record,) = caplog.records
    assert not record.exc_info
    assert "sam@example.com" not in caplog.text


@pytest.mark.asyncio
async def test_a_traced_step_logs_its_trace_as_text_without_frame_locals(
    monkeypatch, account, caplog
):
    """Sentry attaches frame locals to a record with exc_info, and the step's
    subject holds the account's address."""
    Steps(monkeypatch, failing="files")

    with caplog.at_level("ERROR", logger=routine.logger.name):
        await routine.erase_account(USER_ID, dry_run=False)

    (record,) = caplog.records
    assert not record.exc_info
    assert "Traceback" in record.getMessage()
    assert "RuntimeError" in record.getMessage()
