"""The coming run times of the user's schedules, for the Schedules page: each
repeating task's (its crons, in its timezone) and each workflow schedule's.

Tasks that start once keep their one time in ``Task.start_at``, and a task that
watches for events has no clock times, so neither is listed here. Every read
is the caller's own: tasks by user id, workflow schedules through the same
listing ``GET /schedules`` uses.
"""

import logging
from datetime import UTC, datetime
from typing import Annotated, Literal

from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from fastapi import APIRouter, Query, Security
from pydantic import BaseModel, Field

from backend.api.features.schedules.routes import list_all_graphs_execution_schedules
from backend.copilot.scheduled_chats.cadence import next_fire
from backend.data.task import LIST_LIMIT_MAX, list_tasks

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Security(requires_user)])

DEFAULT_COUNT = 6
MAX_COUNT = 24


class UpcomingRuns(BaseModel):
    kind: Literal["task", "workflow"]
    id: str = Field(description="The task's id, or the workflow schedule's id.")
    runs: list[datetime] = Field(
        default_factory=list, description="The next run times, soonest first."
    )


class UpcomingRunsResponse(BaseModel):
    schedules: list[UpcomingRuns]


@router.get(
    "/schedules/upcoming",
    summary="Coming run times of my schedules",
    operation_id="listUpcomingScheduleRuns",
)
async def list_upcoming_schedule_runs(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    count: Annotated[int, Query(ge=1, le=MAX_COUNT)] = DEFAULT_COUNT,
) -> UpcomingRunsResponse:
    now = datetime.now(UTC)
    tasks = await list_tasks(
        user_id=user_id, include_subtasks=False, limit=LIST_LIMIT_MAX
    )
    workflows = await list_all_graphs_execution_schedules(user_id=user_id, ctx=ctx)
    entries = [
        UpcomingRuns(
            kind="task",
            id=task.id,
            runs=coming_runs(task.repeat.crons, task.repeat.timezone, now, count),
        )
        for task in tasks
        if task.repeat and task.repeat.crons and not task.repeat.watch
    ]
    entries += [
        UpcomingRuns(
            kind="workflow",
            id=schedule.id,
            runs=coming_runs([schedule.cron], schedule.timezone, now, count),
        )
        for schedule in workflows
    ]
    return UpcomingRunsResponse(schedules=entries)


def coming_runs(
    crons: list[str], timezone: str, after: datetime, count: int
) -> list[datetime]:
    """The next *count* fire times of *crons*, soonest first; fewer when the
    crons stop firing within a year or do not parse."""
    runs: list[datetime] = []
    at = after
    for _ in range(count):
        fire = next_fire(crons, timezone, at)
        if fire is None:
            break
        runs.append(fire.astimezone(UTC))
        at = fire
    return runs
