import asyncio
import logging
from collections.abc import AsyncIterator
from datetime import datetime

from backend.data import activity_event as activity_db
from backend.data import workspace as workspace_db
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

from .helpers import LOG_ID_CHARS
from .recent_work import is_model_call, scheduled_workflow_id

logger = logging.getLogger(__name__)

_WORK_EVENT_LIMIT = 200
_MAX_WORK_EVENT_PAGES = 3


async def get_work_events(
    *,
    user_id: str,
    since: datetime,
    until: datetime | None = None,
    schedules_task: (
        asyncio.Task[list[GraphExecutionJobInfo | CopilotTurnJobInfo] | None] | None
    ) = None,
) -> list[activity_db.ActivityEvent]:
    visible: list[activity_db.ActivityEvent] = []
    seen_schedule_ids: set[str] = set()
    graph_by_schedule: dict[str, str] | None = None
    async for page in _work_event_pages(user_id, since, until):
        if schedules_task is not None:
            schedules = await schedules_task
            if schedules is not None:
                graph_by_schedule = {
                    job.id: job.graph_id for job in schedules if job.kind == "graph"
                }
            schedules_task = None
        for event in page:
            if is_model_call(event):
                continue
            schedule_id = event.schedule_id
            if schedule_id and (graph_id := scheduled_workflow_id(event)):
                if schedule_id in seen_schedule_ids:
                    continue
                if (
                    graph_by_schedule is not None
                    and graph_by_schedule.get(schedule_id) != graph_id
                ):
                    continue
                seen_schedule_ids.add(schedule_id)
            visible.append(event)
            if len(visible) == _WORK_EVENT_LIMIT:
                return visible
    return visible


async def _work_event_pages(
    user_id: str, since: datetime, until: datetime | None
) -> AsyncIterator[list[activity_db.ActivityEvent]]:
    after_id = None
    for _ in range(_MAX_WORK_EVENT_PAGES):
        try:
            events = await activity_db.list_activity_events(
                user_id,
                since=since,
                until=until,
                after_id=after_id,
                categories=["FILE", "INTEGRATION", "SCHEDULE"],
                limit=_WORK_EVENT_LIMIT,
            )
        except Exception:
            logger.warning(
                "Home could not load work events for user %s", user_id[:LOG_ID_CHARS]
            )
            return
        try:
            page = await _without_working_files(user_id, events)
        except Exception:
            logger.warning(
                "Home could not check work files for user %s", user_id[:LOG_ID_CHARS]
            )
            page = [event for event in events if event.category != "FILE"]
        yield page
        if len(events) < _WORK_EVENT_LIMIT:
            return
        after_id = events[-1].id


async def _without_working_files(
    user_id: str, events: list[activity_db.ActivityEvent]
) -> list[activity_db.ActivityEvent]:
    """Drop the FILE rows logged for Otto's own working files before writes
    stopped logging them. The migration marks the files (``isInternal``), not
    these log rows, so the mark is read through each row's file id."""
    file_ids = [e.object_id for e in events if e.category == "FILE" and e.object_id]
    internal = await workspace_db.list_internal_file_ids(user_id, file_ids)
    return [
        event
        for event in events
        if not (event.category == "FILE" and event.object_id in internal)
    ]
