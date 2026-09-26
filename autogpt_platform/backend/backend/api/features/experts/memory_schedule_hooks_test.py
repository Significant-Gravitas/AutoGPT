"""The expert lifecycle keeps the expert's memory crons in step with it: a new
hire or raise registers them in the background (so the hire returns at once)
and never resumes a scope paused meanwhile, archiving pauses them (also for an
expert whose schedules were already paused), and none of the hooks can hold an
API request past the registry deadline. Pause and resume themselves are
pinned in ``scheduling_test.py``; the registry in ``copilot/dream/``."""

import asyncio
import time
from collections.abc import Callable, Coroutine, Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import prisma.models
import pytest

from backend.api.features.experts import experts_db, scheduling
from backend.copilot.dream import registry
from backend.copilot.dream.registry_fakes_test import (
    EXPERT,
    PAUSED,
    USER,
    WIPED,
    Hang,
    install_registry_env,
    use_short_deadline,
)

TEMPLATE = SimpleNamespace(
    id="template-1",
    isTemplate=True,
    name="Maria",
    avatarUrl=None,
    color="",
    role="Marketing Specialist",
    jobTitle=None,
    tagline=None,
    bio=None,
    skills=[],
    categories=[],
    identity="You are Maria.",
    voicePreferences=None,
    boundaries=None,
    toolProfile=None,
    Workflows=[],
)


@pytest.fixture
def sync(mocker) -> AsyncMock:
    return mocker.patch.object(experts_db, "sync_expert_scope", new=AsyncMock())


@pytest.fixture
def register(mocker) -> AsyncMock:
    return mocker.patch.object(experts_db, "ensure_expert_scheduled", new=AsyncMock())


@pytest.fixture
def spawned(mocker) -> Iterator[dict[str, Any]]:
    """The background tasks the code under test spawned, by name, kept for
    the test to await; whatever it leaves is closed afterwards."""
    tasks: dict[str, Any] = {}

    def spawn(coro: Any, *, name: str) -> None:
        tasks[name] = coro

    mocker.patch.object(experts_db, "spawn_background_task", side_effect=spawn)
    yield tasks
    for coro in tasks.values():
        coro.close()


def _hire(mocker, state: str) -> None:
    expert_client = SimpleNamespace(find_first=AsyncMock(return_value=TEMPLATE))
    mocker.patch.object(prisma.models.Expert, "prisma", return_value=expert_client)
    mocker.patch.object(
        experts_db,
        "_reserve_hired_expert",
        new=AsyncMock(return_value=(SimpleNamespace(id="expert-1"), state)),
    )
    mocker.patch.object(
        experts_db,
        "_resume_revived_hire",
        new=AsyncMock(return_value=SimpleNamespace(id="expert-1")),
    )
    mocker.patch.object(experts_db, "_claim_setup", new=AsyncMock(return_value=False))
    mocker.patch.object(experts_db, "_run_hire_setup", new=MagicMock())
    mocker.patch.object(experts_db, "emit_funnel_event")
    mocker.patch.object(experts_db, "_to_model")
    mocker.patch.object(experts_db, "HireResult")


@pytest.mark.asyncio
async def test_a_new_hire_registers_its_memory_scope_in_the_background(
    mocker, sync, register, spawned
):
    _hire(mocker, "created")

    await experts_db._hire_expert_impl("owner-1", "template-1", None)

    register.assert_not_awaited()  # the hire did not wait on it
    await spawned["expert-memory-schedule-expert-1"]
    register.assert_awaited_once_with("owner-1", "expert-1")
    sync.assert_not_awaited()  # registering, not resuming


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["revived", "existing"])
async def test_revive_and_idempotent_rehire_leave_it_to_their_own_paths(
    mocker, sync, register, spawned, state: str
):
    """A revive resumes the memory crons through ``resume_expert_schedules``;
    re-hiring an active expert changes nothing."""
    _hire(mocker, state)

    await experts_db._hire_expert_impl("owner-1", "template-1", None)

    sync.assert_not_awaited()
    register.assert_not_awaited()
    assert "expert-memory-schedule-expert-1" not in spawned


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["pause", "wipe"])
async def test_a_hire_registration_landing_after_a_pause_leaves_it_stopped(
    monkeypatch, stop: str
):
    """Codex's reproduction: whatever the hire spawns for the memory crons is
    held back until the scope was paused (or wiped), then run. It must leave
    the scope as it found it, with no crons, rather than resume it."""
    env = install_registry_env(monkeypatch)
    held: list[Coroutine[Any, Any, Any]] = []

    def hold(coro: Coroutine[Any, Any, Any], *, name: str) -> None:
        held.append(coro)

    monkeypatch.setattr(experts_db, "spawn_background_task", hold)
    experts_db._schedule_expert_memory(USER, "expert-1")
    stop_scope = registry.pause_scope if stop == "pause" else registry.mark_wiped
    assert await stop_scope(EXPERT)

    await asyncio.wait_for(held[0], 1)  # the hire's registration lands late

    assert env.db.rows[EXPERT.scope_key].state == (PAUSED if stop == "pause" else WIPED)
    assert env.scheduler.jobs == {}
    assert env.adds() == 0


@pytest.mark.asyncio
async def test_a_raised_expert_registers_its_memory_scope(mocker, register, spawned):
    resolved = SimpleNamespace(
        skill_names=[], skills=[], library_skill_names=[], workflows=[]
    )
    attachments = experts_db.raise_attachments
    mocker.patch.object(
        attachments, "resolve_attachments", new=AsyncMock(return_value=resolved)
    )
    mocker.patch.object(
        attachments, "install_marketplace_skills", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(
        attachments, "install_workflows", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(
        experts_db,
        "_create_raised_expert_row",
        new=AsyncMock(return_value=SimpleNamespace(id="expert-1", skills=[])),
    )
    mocker.patch.object(
        experts_db, "_copy_library_skills", new=AsyncMock(return_value=[])
    )
    mocker.patch.object(experts_db, "_to_model")
    mocker.patch.object(experts_db, "RaiseResult")

    await experts_db.create_raised_expert("owner-1", "Nova", None, None)

    await spawned["expert-memory-schedule-expert-1"]
    register.assert_awaited_once_with("owner-1", "expert-1")


def _archive(mocker, *, paused: bool, archived_rows: int) -> None:
    mocker.patch.object(
        scheduling, "pause_expert_schedules", new=AsyncMock(return_value=paused)
    )
    expert_client = SimpleNamespace(
        update_many=AsyncMock(return_value=archived_rows),
        find_first=AsyncMock(return_value=SimpleNamespace(id="expert-1")),
    )
    mocker.patch.object(prisma.models.Expert, "prisma", return_value=expert_client)
    mocker.patch.object(scheduling, "detach_expert_triggers", new=AsyncMock())
    mocker.patch.object(experts_db, "emit_funnel_event")


@pytest.mark.asyncio
async def test_archiving_an_already_paused_expert_still_pauses_its_memory(mocker, sync):
    """A budget breach paused the schedules earlier, so the archive's own
    pause is a no-op — the memory crons must stop anyway."""
    _archive(mocker, paused=False, archived_rows=1)

    await experts_db.archive_expert("owner-1", "expert-1")

    sync.assert_awaited_once_with("owner-1", "expert-1", active=False)


@pytest.mark.asyncio
async def test_archive_leaves_the_memory_pause_to_the_schedule_pause(mocker, sync):
    """``pause_expert_schedules`` pauses the memory crons itself when it
    pauses anything; the archive does not pause them twice."""
    _archive(mocker, paused=True, archived_rows=1)

    await experts_db.archive_expert("owner-1", "expert-1")

    sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_re_archiving_changes_nothing(mocker, sync):
    _archive(mocker, paused=False, archived_rows=0)

    await experts_db.archive_expert("owner-1", "expert-1")

    sync.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("hook", ["pause", "resume", "archive"])
async def test_a_lifecycle_request_is_not_held_up_by_a_hanging_scheduler(
    mocker, monkeypatch, hook: str
):
    """The API routes behind pause, resume and archive (and revive, which
    resumes) finish their own writes and return within the registry
    deadline; the scheduler calls they gave up on are cancelled."""
    env = install_registry_env(monkeypatch)
    use_short_deadline(monkeypatch)
    hang = Hang()
    for rpc in (
        "remove_scope_memory_jobs",
        "add_scope_community_rebuild_schedule",
        "add_scope_nightly_batch_schedule",
    ):
        monkeypatch.setattr(env.scheduler, rpc, hang.forever)
    expert_table = SimpleNamespace(
        update_many=AsyncMock(return_value=1),
        find_first=AsyncMock(return_value=SimpleNamespace(id="expert-1")),
    )
    mocker.patch.object(prisma.models.Expert, "prisma", return_value=expert_table)
    mocker.patch.object(
        prisma.models.ExpertPauseEvent,
        "prisma",
        return_value=SimpleNamespace(create=AsyncMock(), update_many=AsyncMock()),
    )
    mocker.patch.object(scheduling, "reset_weekly_spend", new=AsyncMock())
    mocker.patch.object(scheduling, "detach_expert_triggers", new=AsyncMock())
    mocker.patch.object(experts_db, "emit_funnel_event")
    hooks: dict[str, Callable[[], Coroutine[Any, Any, Any]]] = {
        "pause": lambda: scheduling.pause_expert_schedules(USER, "expert-1", "test"),
        "resume": lambda: scheduling.resume_expert_schedules(USER, "expert-1"),
        "archive": lambda: experts_db.archive_expert(USER, "expert-1"),
    }
    started = time.monotonic()

    await hooks[hook]()

    assert time.monotonic() - started < 1.0
    expert_table.update_many.assert_awaited()  # the lifecycle write landed
    assert hang.started >= 1
    assert hang.cancelled == hang.started
