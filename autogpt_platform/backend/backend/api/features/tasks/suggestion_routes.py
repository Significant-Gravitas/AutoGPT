"""Tasks Otto drafted for the caller: list them, set one aside, or draft more
now. Starting one is the task run route, which takes it onto the board.

Included in the app before the task routes: ``GET /tasks/suggestions`` would
otherwise be read as ``GET /tasks/{task_id}``.
"""

from typing import Annotated

from autogpt_libs.auth import get_user_id, requires_user
from fastapi import APIRouter, Depends, Path, Security

from backend.api.utils.rate_limit import RateLimiter
from backend.copilot.briefing.suggestion_prompt import MAX_SUGGESTIONS
from backend.copilot.briefing.suggestions import draft_task_suggestions
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.data.task import Task, list_tasks
from backend.data.task_suggestion import dismiss_task_suggestion

from .model import TaskListResponse

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    dependencies=[Security(requires_user)],
)

# Each refresh is a model call. Documented in docs/platform/rate-limits.md:
# update it with any change here.
REFRESH_MAX_REQUESTS = 10
REFRESH_WINDOW_SECONDS = 3600
_refresh_limiter = RateLimiter(
    "tasks:suggestions:refresh",
    max_requests=REFRESH_MAX_REQUESTS,
    window_seconds=REFRESH_WINDOW_SECONDS,
)

_TASK_ID = Path(min_length=1, max_length=128, description="A task the caller owns")
UserId = Annotated[str, Security(get_user_id)]


@router.get("/suggestions", operation_id="list_task_suggestions")
async def list_my_task_suggestions(user_id: UserId) -> TaskListResponse:
    """The tasks Otto drafted that the caller has neither started nor set
    aside, newest first."""
    tasks = await list_tasks(user_id=user_id, suggested=True, limit=MAX_SUGGESTIONS)
    return TaskListResponse(tasks=tasks)


@router.post(
    "/suggestions/refresh",
    operation_id="refresh_task_suggestions",
    # Drafting makes a model call the platform pays for.
    dependencies=[Depends(enforce_payment_paywall)],
)
async def refresh_my_task_suggestions(user_id: UserId) -> TaskListResponse:
    """Draft next tasks now, up to the cap, and return the live ones."""
    await _refresh_limiter.check(user_id)
    return TaskListResponse(tasks=await draft_task_suggestions(user_id))


@router.post("/{task_id}/dismiss", operation_id="dismiss_task_suggestion")
async def dismiss_my_task_suggestion(
    user_id: UserId, task_id: Annotated[str, _TASK_ID]
) -> Task:
    """Set a suggested task aside; Otto will not suggest it again."""
    return await dismiss_task_suggestion(task_id, user_id)
