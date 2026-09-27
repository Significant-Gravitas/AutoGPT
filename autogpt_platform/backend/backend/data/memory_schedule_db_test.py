"""The memory-scope schedule registry against a real Postgres: concurrent
first claims of one scope, a pause racing them, and a wiped expert's scope
going through the real pause, resume, archive and revive functions.

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
from backend.copilot.dream import registry, scope_jobs
from backend.copilot.dream.registry_fakes_test import FakeScheduler
from backend.copilot.graphiti.scope import MemoryScope
from backend.data import memory_schedule
from backend.data.memory_schedule import MemoryScopeSchedule
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
    """Everything but Postgres: the scheduler, flags, Redis markers, the
    timezone lookup and the triggers the lifecycle functions touch."""
    scheduler = FakeScheduler()
    monkeypatch.setattr(scope_jobs, "is_feature_enabled", AsyncMock(return_value=True))
    monkeypatch.setattr(scope_jobs, "get_scheduler_client", lambda: scheduler)
    monkeypatch.setattr(scope_jobs, "write_registration_marker", AsyncMock())
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
