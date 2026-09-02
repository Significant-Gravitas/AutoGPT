"""Unit tests for the overseer pass — no DB, frozen clock via ``now=``."""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.tasks.models import (
    DelegatedTask,
    TaskAmendment,
    TaskExpertRef,
)
from backend.copilot.overseer.service import run_overseer_pass

_MODULE = "backend.copilot.overseer.service"

_NOW = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)


def _task(
    *,
    task_id: str = "task-1",
    status: str = "WORKING",
    updated_minutes_ago: float = 0,
    amendments: list[TaskAmendment] | None = None,
    stale_at: datetime | None = None,
    origin_session_id: str | None = "sess-1",
    owner: TaskExpertRef | None = None,
    created_by_type: str = "USER",
    naive_updated: bool = False,
) -> DelegatedTask:
    updated = _NOW - timedelta(minutes=updated_minutes_ago)
    if naive_updated:
        updated = updated.replace(tzinfo=None)
    return DelegatedTask(
        id=task_id,
        title="Draft the weekly report",
        spec="spec",
        status=status,  # type: ignore[arg-type]
        acceptance="PENDING",
        created_by_type=created_by_type,  # type: ignore[arg-type]
        created_by_id="user-1",
        owner=owner,
        parent_task_id=None,
        root_task_id=task_id,
        origin_session_id=origin_session_id,
        ancestor_expert_ids=[],
        handoff_count=0,
        revision_count=0,
        spend_total=0,
        outcome_summary=None,
        amendments=amendments or [],
        stale_at=stale_at,
        created_at=updated,
        updated_at=updated,
    )


def _retry_amendment() -> TaskAmendment:
    return TaskAmendment(
        at=_NOW - timedelta(minutes=20), by="overseer", note="retried", kind="retry"
    )


def _client(
    tasks: list[DelegatedTask],
    *,
    running: dict[str, bool] | None = None,
    failed_counts: dict[str, int] | None = None,
) -> MagicMock:
    client = MagicMock()
    client.list_open_tasks = AsyncMock(return_value=tasks)
    client.has_running_executions = AsyncMock(return_value=running or {})
    client.close_delegated_task = AsyncMock()
    client.append_task_amendment = AsyncMock()
    client.mark_task_stale = AsyncMock(return_value=True)
    client.count_recent_failed_tasks_by_expert = AsyncMock(
        return_value=failed_counts or {}
    )
    client.pause_expert_schedules = AsyncMock(return_value=True)
    return client


@contextmanager
def _patched(client: MagicMock, *, flag: bool = True):
    """Every outbound edge of the pass stubbed; yields the two dispatch mocks
    a nudge can land on."""
    queued = MagicMock(turn_in_flight=False)
    with patch(
        f"{_MODULE}.get_database_manager_async_client", return_value=client
    ), patch(f"{_MODULE}.is_feature_enabled", AsyncMock(return_value=flag)), patch(
        f"{_MODULE}.queue_user_message", AsyncMock(return_value=queued)
    ), patch(
        f"{_MODULE}.schedule_chat_turn", AsyncMock()
    ) as schedule, patch(
        f"{_MODULE}.start_task_in_new_session", AsyncMock(return_value="sess-new")
    ) as kickoff:
        yield {"schedule": schedule, "kickoff": kickoff}


@pytest.mark.asyncio
async def test_flag_off_does_nothing():
    client = _client([_task(updated_minutes_ago=60)])
    with _patched(client, flag=False):
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary == {"retried": 0, "failed": 0, "stale": 0, "paused_experts": 0}
    client.list_open_tasks.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_stall_records_retry_and_nudges_the_session():
    client = _client([_task(updated_minutes_ago=20)], running={"task-1": False})
    with _patched(client) as mocks:
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["retried"] == 1
    kwargs = client.append_task_amendment.call_args.kwargs
    assert kwargs["kind"] == "retry"
    assert kwargs["by"] == "overseer"
    client.close_delegated_task.assert_not_awaited()
    # queue reported no turn in flight, so the nudge schedules a fresh turn.
    mocks["schedule"].assert_awaited_once()
    assert mocks["schedule"].call_args.kwargs["session_id"] == "sess-1"


@pytest.mark.asyncio
async def test_task_closed_under_the_pass_is_not_dispatched():
    """The task closed between list_open_tasks and the retry — the amendment
    returns None, so nothing gets nudged or kicked off and nothing counts."""
    client = _client([_task(updated_minutes_ago=20)], running={"task-1": False})
    client.append_task_amendment = AsyncMock(return_value=None)
    with _patched(client) as mocks:
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["retried"] == 0
    mocks["schedule"].assert_not_awaited()
    mocks["kickoff"].assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_task_with_no_session_gets_a_worker():
    """An intro task from a hire lands QUEUED with nothing driving it. The
    retry has no session to nudge, so it opens one for the owner instead."""
    owner = TaskExpertRef(id="expert-1", name="Alex", avatar_url=None, role="Ops")
    client = _client(
        [
            _task(
                status="QUEUED",
                updated_minutes_ago=20,
                origin_session_id=None,
                owner=owner,
            )
        ],
        running={"task-1": False},
    )
    with _patched(client) as mocks:
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["retried"] == 1
    mocks["kickoff"].assert_awaited_once_with(
        "user-1",
        task_id="task-1",
        title="Draft the weekly report",
        expert_id="expert-1",
    )
    mocks["schedule"].assert_not_awaited()


@pytest.mark.asyncio
async def test_fresh_queued_task_is_left_to_start_on_its_own():
    """A receipt opened seconds ago is mid-kickoff, not stalled."""
    client = _client([_task(status="QUEUED", updated_minutes_ago=2)])
    with _patched(client) as mocks:
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["retried"] == 0
    mocks["kickoff"].assert_not_awaited()
    client.has_running_executions.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_dream_proposal_is_never_auto_started():
    """The dream pass parks sessionless proposals on purpose — starting one
    here would do work the user never accepted."""
    client = _client(
        [
            _task(
                status="QUEUED",
                updated_minutes_ago=6 * 60,
                origin_session_id=None,
                created_by_type="DREAM",
            )
        ]
    )
    with _patched(client) as mocks:
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["retried"] == 0
    mocks["kickoff"].assert_not_awaited()
    client.append_task_amendment.assert_not_awaited()


@pytest.mark.asyncio
async def test_second_stall_fails_the_task():
    client = _client(
        [_task(updated_minutes_ago=20, amendments=[_retry_amendment()])],
        running={"task-1": False},
    )
    with _patched(client):
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["failed"] == 1
    kwargs = client.close_delegated_task.call_args.kwargs
    assert kwargs["succeeded"] is False
    client.append_task_amendment.assert_not_awaited()


@pytest.mark.asyncio
async def test_recently_updated_working_task_is_left_alone():
    client = _client([_task(updated_minutes_ago=5)])
    with _patched(client):
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["retried"] == 0
    client.has_running_executions.assert_not_awaited()


@pytest.mark.asyncio
async def test_stalled_task_with_live_execution_is_not_retried():
    client = _client([_task(updated_minutes_ago=45)], running={"task-1": True})
    with _patched(client):
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["retried"] == 0
    client.append_task_amendment.assert_not_awaited()
    client.close_delegated_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_week_old_waiting_task_is_stamped_stale():
    client = _client([_task(status="WAITING_USER", updated_minutes_ago=8 * 24 * 60)])
    with _patched(client):
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["stale"] == 1
    client.mark_task_stale.assert_awaited_once_with("user-1", "task-1", stale_at=_NOW)


@pytest.mark.asyncio
async def test_naive_updated_at_does_not_crash_the_pass():
    # Prisma can hand back a timezone-naive ``updated_at``; the pass must
    # normalize it rather than raise TypeError on the comparison.
    client = _client(
        [
            _task(updated_minutes_ago=20, naive_updated=True),
            _task(
                task_id="task-2",
                status="WAITING_USER",
                updated_minutes_ago=8 * 24 * 60,
                naive_updated=True,
            ),
        ],
        running={"task-1": False},
    )
    with _patched(client):
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["retried"] == 1
    assert summary["stale"] == 1


@pytest.mark.asyncio
async def test_day_old_waiting_task_is_not_stale_and_never_cancelled():
    client = _client([_task(status="WAITING_USER", updated_minutes_ago=25 * 60)])
    with _patched(client):
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["stale"] == 0
    client.mark_task_stale.assert_not_awaited()
    client.close_delegated_task.assert_not_awaited()


@pytest.mark.asyncio
async def test_already_stale_task_is_not_restamped():
    client = _client(
        [
            _task(
                status="WAITING_USER",
                updated_minutes_ago=9 * 24 * 60,
                stale_at=_NOW - timedelta(days=1),
            )
        ]
    )
    with _patched(client):
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["stale"] == 0
    client.mark_task_stale.assert_not_awaited()


@pytest.mark.asyncio
async def test_three_failures_in_a_week_pause_the_expert():
    client = _client([], failed_counts={"expert-1": 3, "expert-2": 2})
    with _patched(client):
        summary = await run_overseer_pass("user-1", now=_NOW)

    assert summary["paused_experts"] == 1
    client.pause_expert_schedules.assert_awaited_once()
    args = client.pause_expert_schedules.call_args.args
    assert args[:2] == ("user-1", "expert-1")
