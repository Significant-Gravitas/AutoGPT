"""The tasks part of the morning brief: what finished and failed since the
last brief, and what is waiting on the user."""

from datetime import datetime

from prisma.enums import TaskStatus

from backend.api.features.experts.models import Expert
from backend.data.task import Task

from .models import BriefingTaskItem, BriefingTasks

# Rows listed per group; the totals carry the rest.
MAX_TASK_ITEMS = 5
_MAX_LINE_CHARS = 160


def compose_task_summary(
    *, tasks: list[Task], experts: list[Expert], since: datetime
) -> BriefingTasks:
    """``tasks`` are the user's board tasks (no suggestions); subtasks are
    left out so a split task is reported once."""
    experts_by_id = {expert.id: expert for expert in experts}
    own = [task for task in tasks if not task.parent_task_id and not task.is_archived]
    finished = _newest(
        [t for t in own if t.status == TaskStatus.DONE and _settled_since(t, since)]
    )
    failed = _newest(
        [t for t in own if t.status == TaskStatus.FAILED and _settled_since(t, since)]
    )
    waiting = _newest([t for t in own if t.status == TaskStatus.NEEDS_INPUT])
    return BriefingTasks(
        finished=[_item(t, experts_by_id) for t in finished[:MAX_TASK_ITEMS]],
        failed=[_item(t, experts_by_id) for t in failed[:MAX_TASK_ITEMS]],
        waiting=[_item(t, experts_by_id) for t in waiting[:MAX_TASK_ITEMS]],
        finished_total=len(finished),
        failed_total=len(failed),
        waiting_total=len(waiting),
        in_progress_total=sum(1 for t in own if t.status == TaskStatus.IN_PROGRESS),
    )


def _settled_since(task: Task, since: datetime) -> bool:
    return (task.completed_at or task.updated_at) >= since


def _newest(tasks: list[Task]) -> list[Task]:
    return sorted(tasks, key=lambda t: t.completed_at or t.updated_at, reverse=True)


def _item(task: Task, experts_by_id: dict[str, Expert]) -> BriefingTaskItem:
    owner = experts_by_id.get(task.owner_expert_id or "")
    return BriefingTaskItem(
        task_id=task.id,
        title=task.title,
        status=task.status.value,
        owner_expert_id=task.owner_expert_id,
        owner_name=owner.name if owner else None,
        owner_avatar_url=owner.avatar_url if owner else None,
        line=_line(task),
        occurred_at=task.completed_at or task.updated_at,
    )


def _line(task: Task) -> str:
    """The one line a reader needs: the result for a finished task, why it
    stopped for a failed one, the question for one waiting on the user."""
    if task.status == TaskStatus.DONE:
        text = task.result or task.status_text or ""
    elif task.status == TaskStatus.FAILED:
        text = task.failure_reason or task.status_text or ""
    else:
        text = task.status_text or task.next_step or ""
    words = " ".join(text.split())
    if len(words) <= _MAX_LINE_CHARS:
        return words
    return words[: _MAX_LINE_CHARS - 1].rstrip() + "…"
