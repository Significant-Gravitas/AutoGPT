import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from backend.api.features.library.model import LibraryAgentRef
from backend.data.activity_event import ActivityEvent
from backend.executor.scheduler import GraphExecutionJobInfo

from .service import build_home_dashboard
from .work_events import get_work_events


def _file_event(event_id: str, name: str, file_id: str) -> ActivityEvent:
    return ActivityEvent(
        id=event_id,
        user_id="user-1",
        created_at=datetime(2026, 10, 23, 11, 0, tzinfo=timezone.utc),
        category="FILE",
        event_type="file.created",
        title=name,
        object_id=file_id,
    )


def _schedule_created(
    event_id: str, schedule_id: str, *, minutes_ago: int = 1
) -> ActivityEvent:
    return ActivityEvent(
        id=event_id,
        user_id="user-1",
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        category="SCHEDULE",
        event_type="schedule.created",
        title=event_id,
        object_id="scheduled-graph",
        schedule_id=schedule_id,
    )


@pytest.mark.asyncio
async def test_schedule_and_activity_reads_can_start_together(
    mocker, home_dependencies
):
    settings = mocker.patch("backend.api.features.home.service.settings")
    settings.config.schedule_read_timeout_seconds = 120
    both_started = asyncio.Barrier(2)

    async def read(*args, **kwargs):
        await both_started.wait()
        return []

    mocker.patch(
        "backend.api.features.home.work_events.activity_db.list_activity_events",
        side_effect=read,
    )
    home_dependencies.get_execution_schedules.side_effect = read
    await asyncio.wait_for(build_home_dashboard(user_id="user-1"), timeout=60)


@pytest.mark.asyncio
async def test_recreated_schedule_has_one_item_for_its_library_workflow(
    mocker, home_dependencies
):
    paused = GraphExecutionJobInfo(
        id="schedule-b",
        name="Paused",
        next_run_time="",
        user_id="user-1",
        graph_id="scheduled-graph",
        graph_version=1,
        cron="0 9 * * *",
        input_data={},
    )
    home_dependencies.get_execution_schedules.return_value = [paused]
    mocker.patch(
        "backend.api.features.home.service.execution_db.get_graph_executions",
        return_value=[],
    )
    lookup = mocker.patch(
        "backend.api.features.home.service.library_db.get_library_agent_refs_by_graph_ids",
        return_value=[
            LibraryAgentRef(id="lib-b", graph_id=paused.graph_id, name="Digest")
        ],
    )
    mocker.patch(
        "backend.api.features.home.work_events.activity_db.list_activity_events",
        return_value=[
            _schedule_created("recreated-b", paused.id),
            _schedule_created("deleted-a", "schedule-a", minutes_ago=2),
            _schedule_created("original-b", paused.id, minutes_ago=3),
        ],
    )

    home = await build_home_dashboard(user_id="user-1")

    lookup.assert_awaited_once_with("user-1", [paused.graph_id], include_deleted=True)
    (group,) = home.recent_work.groups
    assert (group.actor.kind, group.actor.name, group.actor.link) == (
        "workflow",
        "Digest",
        "/library/agents/lib-b",
    )
    assert [item.id for item in group.items] == ["recreated-b"]
    assert home.recent_work.total_count == group.schedule_count == 1
    assert home.upcoming_tasks == []
    home_dependencies.get_execution_schedules.assert_awaited_once_with(
        user_id="user-1", include_paused=True
    )


@pytest.mark.asyncio
async def test_deleted_schedules_cannot_hide_the_next_page_of_recent_work(
    mocker, home_dependencies
):
    first_page = [
        _schedule_created(f"event-{i}", f"deleted-{i}", minutes_ago=i + 1)
        for i in range(200)
    ]
    file = _file_event("older", "Report", "report-id")
    file.created_at = first_page[-1].created_at - timedelta(seconds=1)
    read = mocker.patch(
        "backend.api.features.home.work_events.activity_db.list_activity_events",
        side_effect=[first_page, [file]],
    )
    mocker.patch(
        "backend.api.features.home.work_events.workspace_db.list_internal_file_ids",
        return_value=set(),
    )

    home = await build_home_dashboard(user_id="user-1")

    assert [item.id for group in home.recent_work.groups for item in group.items] == [
        "older"
    ]
    assert home.recent_work.total_count == 1
    assert read.await_count == 2
    assert read.await_args.kwargs["after_id"] == first_page[-1].id


@pytest.mark.asyncio
async def test_recent_work_drops_files_marked_as_ottos_own(mocker: MockerFixture):
    """FILE rows logged before working-file writes stopped being logged: the
    migration marks the file, and the row is dropped through its file id."""
    sticker = _file_event("e1", "sticker-01.png", "file-sticker")
    scratchpad = _file_event("e2", "build_state.json", "file-state")
    mocker.patch(
        "backend.api.features.home.work_events.activity_db.list_activity_events",
        AsyncMock(return_value=[sticker, scratchpad]),
    )
    internal = mocker.patch(
        "backend.api.features.home.work_events.workspace_db.list_internal_file_ids",
        AsyncMock(return_value={"file-state"}),
    )

    events = await get_work_events(
        user_id="user-1", since=datetime(2026, 10, 16, tzinfo=timezone.utc)
    )

    assert [event.title for event in events] == ["sticker-01.png"]
    internal.assert_awaited_once_with("user-1", ["file-sticker", "file-state"])


@pytest.mark.asyncio
async def test_recent_work_keeps_actions_when_the_file_check_fails(
    mocker: MockerFixture,
):
    action = _file_event("e1", "Airtable Create Table", "block-1").model_copy(
        update={"category": "INTEGRATION", "event_type": "integration.action"}
    )
    mocker.patch(
        "backend.api.features.home.work_events.activity_db.list_activity_events",
        AsyncMock(return_value=[action, _file_event("e2", "sticker-01.png", "f1")]),
    )
    mocker.patch(
        "backend.api.features.home.work_events.workspace_db.list_internal_file_ids",
        AsyncMock(side_effect=RuntimeError("database unavailable")),
    )

    events = await get_work_events(
        user_id="user-1", since=datetime(2026, 10, 16, tzinfo=timezone.utc)
    )

    assert [event.title for event in events] == ["Airtable Create Table"]
