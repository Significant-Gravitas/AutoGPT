"""The routine claims the request before any step, then runs every step in
order; it deletes nothing on a dry run, when something is blocked or when the
request can't be claimed (kept, not due, or erased already), and stops at the
first failed step so a rerun can resume."""

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from prisma.enums import AccountDeletionSource

from backend.data.account_erasure import routine
from backend.data.account_erasure.deleted_user import DELETED_USER_ID
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
    held.beating = []

    @contextlib.asynccontextmanager
    async def renew_in_background(lock):
        held.beating.append(True)
        try:
            yield
        finally:
            held.beating[-1] = False

    monkeypatch.setattr(routine, "renew_in_background", renew_in_background)
    return held


@pytest.fixture
def unused_services(monkeypatch):
    services: list[str] = []
    monkeypatch.setattr(
        routine,
        "Settings",
        lambda: SimpleNamespace(
            config=SimpleNamespace(account_erasure_unused_services=services)
        ),
    )
    return services


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
@pytest.mark.parametrize(
    "user_id, blocker",
    [("", "invalid_account_id"), (DELETED_USER_ID, "system_account")],
)
async def test_a_refused_id_runs_no_step_even_in_a_dry_run(
    monkeypatch, user_id, blocker
):
    """An empty ID would list every user's runs, and the placeholder's
    counts mean nothing, so the dry run stops at the blocker too."""
    subject = Subject(user_id=user_id, request_source=AccountDeletionSource.SETTINGS)
    monkeypatch.setattr(routine, "load_subject", AsyncMock(return_value=subject))
    monkeypatch.setattr(
        routine, "plan_transfer", AsyncMock(return_value=TransferPlan())
    )
    steps = Steps(monkeypatch)

    report = await routine.erase_account(user_id)

    assert steps.calls == []
    assert [b.name for b in report.blockers] == [blocker]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "step", ["org_subscriptions", "files", "memory", "posthog", "mailerlite"]
)
async def test_a_step_skipped_for_want_of_its_service_stops_before_the_rows(
    monkeypatch, account, unused_services, step
):
    """The request stays open, and the rows stay for a rerun once the setting
    is in: they hold the address MailerLite needs and the expert IDs that name
    the memory graphs."""
    steps = Steps(monkeypatch, skipping=step)

    report = await routine.erase_account(USER_ID, dry_run=False)

    ran = [name for name, _ in steps.calls]
    assert ran[-1] == step
    assert "database" not in ran and "request" not in ran
    assert report.steps[-1].status == StepStatus.FAILED
    assert "ACCOUNT_ERASURE_UNUSED_SERVICES" in report.steps[-1].note
    assert not report.complete


@pytest.mark.asyncio
async def test_a_service_the_stack_does_not_use_may_be_skipped(
    monkeypatch, account, unused_services
):
    unused_services.append("posthog")
    steps = Steps(monkeypatch, skipping="posthog")

    report = await routine.erase_account(USER_ID, dry_run=False)

    assert [name for name, _ in steps.calls] == STEP_NAMES
    assert report.complete


@pytest.mark.asyncio
async def test_a_dry_run_reports_a_skip_as_a_skip(
    monkeypatch, account, unused_services
):
    Steps(monkeypatch, skipping="memory")

    report = await routine.erase_account(USER_ID)

    assert len(report.steps) == len(STEP_NAMES)
    memory = next(s for s in report.steps if s.name == "memory")
    assert memory.status == StepStatus.SKIPPED


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
async def test_a_heartbeat_keeps_the_lease_for_the_whole_apply(
    monkeypatch, account, lease
):
    """A step can outlast the lease's few minutes: the heartbeat renews it
    while every step runs, and stops once the erasure ends. A dry run takes
    no lease, so it has no heartbeat."""
    seen: list[bool] = []

    async def step(subject, *, dry_run):
        seen.append(bool(lease.beating) and lease.beating[-1])
        return StepReport(name="files", status=StepStatus.DONE)

    monkeypatch.setattr(routine, "_erasure_steps", lambda: [("files", step)])

    await routine.erase_account(USER_ID)
    assert lease.beating == []

    await routine.erase_account(USER_ID, dry_run=False)
    assert seen == [False, True]
    assert lease.beating == [False]


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


@pytest.mark.parametrize(
    "status, step, unused, stops",
    [
        (StepStatus.WOULD_DO, "posthog", [], False),
        (StepStatus.FAILED, "ban_record", [], True),
        (StepStatus.SKIPPED, "posthog", [], True),
        (StepStatus.SKIPPED, "posthog", ["posthog"], False),
        # A skip with no service behind it is not a missing setting.
        (StepStatus.SKIPPED, "ban_record", [], False),
    ],
)
def test_a_dry_run_step_shows_whether_the_apply_would_stop_there(
    unused_services, status, step, unused, stops
):
    unused_services.extend(unused)

    assert routine.stops_an_apply(StepReport(name=step, status=status)) is stops
