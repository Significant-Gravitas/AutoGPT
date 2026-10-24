"""The memory-scope schedule registry against a real Postgres: concurrent
first claims of one scope, a pause racing them, a wiped expert's scope
going through the real pause, resume, archive and revive functions, a
deleted user's legacy cron (with a real scheduler and jobstore), and the
backfill's listing of the accounts that learn.

Needs the database (``server`` starts the test stack). The unit tests in
``memory_schedule_test.py`` cover the same code with Prisma mocked.
"""

import asyncio
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import prisma.models
import pytest
from prisma.enums import MemoryScopeScheduleState

from backend.api.features.experts import experts_db
from backend.api.features.experts import scheduling as expert_scheduling
from backend.copilot.dream import registry, scope_crons, scope_jobs, scope_reconcile
from backend.copilot.dream.registry_fakes_test import FakeScheduler
from backend.copilot.dream.scheduling import SKILL_LEARNING_JOB_PREFIX, dream_system_job
from backend.copilot.dream.scope_reconcile_jobstore_test import (
    _job,
    _scheduler,
    _sqlite,
)
from backend.copilot.graphiti.scope import MemoryScope
from backend.data import memory_schedule
from backend.data.memory_schedule import MemoryScopeSchedule
from backend.executor import scheduler as scheduler_module
from backend.util.test import SpinTestServer

SCOPES = 10
CONTENDERS = 20
WIPED = MemoryScopeScheduleState.WIPED


async def _make_users(prefix: str) -> list[str]:
    user_ids = [f"{prefix}-{uuid.uuid4().hex[:16]}" for _ in range(SCOPES)]
    await prisma.models.User.prisma().create_many(
        data=[{"id": uid, "email": f"{uid}@example.invalid"} for uid in user_ids]
    )
    return user_ids


async def _drop_users(user_ids: list[str]) -> None:
    await prisma.models.User.prisma().delete_many(where={"id": {"in": user_ids}})


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_first_claims_all_return_the_one_row(server: SpinTestServer):
    user_ids = await _make_users("claim-race")
    try:
        for user_id in user_ids:
            results = await asyncio.gather(
                *(
                    memory_schedule.claim_scope_schedule(user_id, user_id, None, "UTC")
                    for _ in range(CONTENDERS)
                ),
                return_exceptions=True,
            )
            failures = [r for r in results if not isinstance(r, MemoryScopeSchedule)]
            assert failures == []
            rows = await prisma.models.MemoryScopeSchedule.prisma().count(
                where={"userId": user_id}
            )
            assert rows == 1
    finally:
        await _drop_users(user_ids)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_pause_racing_first_registrations_always_ends_paused(
    server: SpinTestServer,
):
    """A pause of a scope with no row claims it PAUSED while registrations
    claim it ACTIVE. In every other round a registration's claim lands
    first, so the pause's own claim loses and gets the ACTIVE row back;
    either way the pause ends PAUSED with the crons removed."""
    user_ids = await _make_users("pause-race")
    try:
        for round_number, user_id in enumerate(user_ids):
            await _race_pause_against_claims(
                user_id, registration_first=round_number % 2 == 1
            )
    finally:
        await _drop_users(user_ids)


async def _race_pause_against_claims(user_id: str, registration_first: bool) -> None:
    barrier = asyncio.Barrier(CONTENDERS + 1)
    registered = asyncio.Event()

    async def register(*args: Any) -> MemoryScopeSchedule:
        await barrier.wait()
        row = await memory_schedule.claim_scope_schedule(*args)
        registered.set()
        return row

    async def pause_claim(*args: Any) -> MemoryScopeSchedule:
        await barrier.wait()
        if registration_first:
            await registered.wait()
        return await memory_schedule.claim_scope_schedule(*args)

    db = SimpleNamespace(
        set_scope_state=memory_schedule.set_scope_state,
        claim_scope_schedule=pause_claim,
    )
    remove = AsyncMock(return_value=True)
    with (
        patch.object(registry, "memory_schedule_db", return_value=db),
        patch.object(registry, "resolve_user_timezone", AsyncMock(return_value="UTC")),
        patch.object(registry, "remove_scope_jobs", remove),
    ):
        paused, *_ = await asyncio.gather(
            registry.pause_scope(MemoryScope.for_user(user_id)),
            *(register(user_id, user_id, None, "UTC") for _ in range(CONTENDERS)),
        )
    row = await memory_schedule.get_scope_schedule(user_id, user_id)
    assert paused is True
    assert row is not None and row.state == MemoryScopeScheduleState.PAUSED
    remove.assert_awaited_once()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_wiped_expert_stays_wiped_through_its_lifecycle(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    """Codex's round-2 reproduction: pause then resume, and archive then
    revive, used to bring a wiped expert's scope back ACTIVE with both
    crons, because the pause wrote PAUSED over WIPED. A direct resume of a
    wiped scope was already refused."""
    scheduler = _lifecycle_boundaries(monkeypatch)
    user_id, scope = await _make_expert("wiped-lifecycle")
    try:
        await registry.ensure_scope_scheduled(scope)
        assert await registry.mark_wiped(scope)
        await registry.resume_scope(scope)
        assert await _stopped(scope, scheduler) == WIPED

        assert await expert_scheduling.pause_expert_schedules(
            user_id, scope.expert_id or "", "test"
        )
        assert await expert_scheduling.resume_expert_schedules(
            user_id, scope.expert_id or ""
        )
        assert await _stopped(scope, scheduler) == WIPED

        await experts_db.archive_expert(user_id, scope.expert_id or "")
        revived = await prisma.models.Expert.prisma().update(
            where={"id": scope.expert_id or ""}, data={"isArchived": False}
        )
        assert revived is not None
        await experts_db._resume_revived_hire(revived)
        assert await _stopped(scope, scheduler) == WIPED
    finally:
        await _drop_users([user_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_a_paused_expert_still_resumes(
    server: SpinTestServer, monkeypatch: pytest.MonkeyPatch
):
    scheduler = _lifecycle_boundaries(monkeypatch)
    user_id, scope = await _make_expert("paused-lifecycle")
    try:
        await registry.ensure_scope_scheduled(scope)
        await expert_scheduling.pause_expert_schedules(
            user_id, scope.expert_id or "", "test"
        )
        assert await _stopped(scope, scheduler) == MemoryScopeScheduleState.PAUSED

        await expert_scheduling.resume_expert_schedules(user_id, scope.expert_id or "")

        row = await memory_schedule.get_scope_schedule(user_id, scope.scope_key)
        assert row is not None and row.state == MemoryScopeScheduleState.ACTIVE
        assert len(scheduler.jobs) == 2
    finally:
        await _drop_users([user_id])


def _lifecycle_boundaries(monkeypatch: pytest.MonkeyPatch) -> FakeScheduler:
    """Everything but Postgres: the scheduler, flags, the timezone lookup
    and the triggers the lifecycle functions touch."""
    scheduler = FakeScheduler()
    monkeypatch.setattr(
        scope_jobs, "evaluate_feature_flag", AsyncMock(return_value=(True, True))
    )
    monkeypatch.setattr(scope_jobs, "get_scheduler_client", lambda: scheduler)
    monkeypatch.setattr(
        registry, "resolve_user_timezone", AsyncMock(return_value="UTC")
    )
    for name in (
        "reset_weekly_spend",
        "detach_expert_triggers",
        "reattach_expert_triggers",
    ):
        monkeypatch.setattr(expert_scheduling, name, AsyncMock())
    monkeypatch.setattr(experts_db, "emit_funnel_event", MagicMock())
    monkeypatch.setattr(
        experts_db, "get_user_default_team", AsyncMock(return_value=("org", "team"))
    )
    return scheduler


async def _make_expert(prefix: str) -> tuple[str, MemoryScope]:
    user_id = f"{prefix}-{uuid.uuid4().hex[:16]}"
    await prisma.models.User.prisma().create(
        data={"id": user_id, "email": f"{user_id}@example.invalid", "timezone": "UTC"}
    )
    expert = await prisma.models.Expert.prisma().create(
        data={
            "ownerUserId": user_id,
            "name": "Lifecycle",
            "role": "test",
            "identity": "test",
            "weeklyBudget": 10,
        }
    )
    return user_id, MemoryScope.for_expert(user_id, expert.id)


async def _stopped(
    scope: MemoryScope, scheduler: FakeScheduler
) -> MemoryScopeScheduleState | None:
    """The scope's state, asserting it has no crons left."""
    assert scheduler.jobs == {}
    row = await memory_schedule.get_scope_schedule(scope.owner_user_id, scope.scope_key)
    return row.state if row is not None else None


@pytest.mark.asyncio(loop_scope="session")
async def test_a_deleted_accounts_legacy_cron_goes_and_a_live_ones_stays(
    server: SpinTestServer, tmp_path
):
    """The learning fit review's probe: an untagged learning cron from
    before the registry, of an account with no row, in a real scheduler and
    jobstore. Once the user is deleted, the reconcile removes it and its
    gate skips it; a live account's cron stays and its gate lets it run."""
    user_ids = [f"legacy-cron-{uuid.uuid4().hex[:16]}" for _ in range(2)]
    await prisma.models.User.prisma().create_many(
        data=[{"id": uid, "email": f"{uid}@example.invalid"} for uid in user_ids]
    )
    live, gone = (MemoryScope.for_user(uid) for uid in user_ids)
    job = dream_system_job(SKILL_LEARNING_JOB_PREFIX)
    body = scheduler_module._MEMORY_CRONS[SKILL_LEARNING_JOB_PREFIX]
    try:
        with _scheduler(_sqlite(tmp_path)) as service:
            for scope in (live, gone):
                service._add_memory_job(*body, job, scope, "UTC")
            await prisma.models.User.prisma().delete(where={"id": gone.owner_user_id})

            crons = scheduler_module._SchedulerCrons(service)
            report = await asyncio.wait_for(scope_reconcile.reconcile(crons), 30)

            assert gone.scope_key in report.removed
            assert live.scope_key not in report.removed
            assert _job(service, job.job_id(gone)) is None
            assert _job(service, job.job_id(live)) is not None
            assert not await scope_crons._schedule_active(gone)
            assert await scope_crons._schedule_active(live)
    finally:
        await _drop_users(user_ids)


@pytest.mark.asyncio(loop_scope="session")
async def test_the_accounts_that_learn_are_listed_once_each_page_by_page(
    server: SpinTestServer,
):
    """One entry per user with learning sources, however many it has, in
    id order; a user without any is not listed."""
    stem = f"learning-list-{uuid.uuid4().hex[:16]}"
    user_ids = [f"{stem}-a", f"{stem}-b", f"{stem}-c"]
    await prisma.models.User.prisma().create_many(
        data=[{"id": uid, "email": f"{uid}@example.invalid"} for uid in user_ids]
    )
    try:
        for user_id, sources in zip(user_ids, (2, 1, 0)):
            for n in range(sources):
                await prisma.models.SkillLearningSource.prisma().create(
                    data={
                        "userId": user_id,
                        "ownerKey": "personal",
                        "sourceKind": "chat_session",
                        "sourceId": f"session-{n}",
                        "revision": "1",
                    }
                )

        first = await memory_schedule.list_learning_account_ids(after=stem, limit=1)
        second = await memory_schedule.list_learning_account_ids(
            after=first[-1], limit=1
        )
        third = await memory_schedule.list_learning_account_ids(
            after=second[-1], limit=1
        )
        together = await memory_schedule.list_learning_account_ids(after=stem, limit=10)

        assert (first, second) == ([user_ids[0]], [user_ids[1]])
        assert user_ids[2] not in third
        assert together[:2] == user_ids[:2] and user_ids[2] not in together
    finally:
        await _drop_users(user_ids)
