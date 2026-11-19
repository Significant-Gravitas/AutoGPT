"""The user's word on a task's result: accepted, or changes asked for. Kept in
``Task.metadata`` (``task_state_meta``); scoped to the owning user."""

from datetime import UTC, datetime

import prisma.models
from prisma.enums import TaskStatus

from backend.data.task import Task, get_task
from backend.data.task_state_meta import (
    REVIEW_KEY,
    REVIEW_NOTE_MAX_CHARS,
    ReviewState,
    TaskReview,
    with_state,
)
from backend.util.exceptions import ConflictError, NotFoundError

# The results a user can review: one that was delivered, or one that was not.
REVIEWABLE_STATUSES = {
    "accepted": [TaskStatus.DONE],
    "changes_requested": [TaskStatus.DONE, TaskStatus.FAILED],
}


async def set_task_review(
    task_id: str, user_id: str, state: ReviewState, note: str | None = None
) -> Task:
    """Record the user's review of the task's latest result. Accepting needs a
    finished task; asking for changes, a finished or failed one."""
    row = await prisma.models.Task.prisma().find_first(
        where={"id": task_id, "userId": user_id, "isArchived": False}
    )
    if row is None:
        raise NotFoundError(f"Task {task_id} not found")
    if row.status not in REVIEWABLE_STATUSES[state]:
        raise ConflictError(
            "Only a finished task's result can be accepted."
            if state == "accepted"
            else "Changes can be asked for once the task has finished or failed."
        )
    note = " ".join((note or "").split())[:REVIEW_NOTE_MAX_CHARS] or None
    review = TaskReview(state=state, note=note, at=datetime.now(UTC))
    moved = await prisma.models.Task.prisma().update_many(
        where={"id": task_id, "userId": user_id, "status": row.status},
        data={
            "metadata": with_state(
                row.metadata, REVIEW_KEY, review.model_dump(mode="json")
            )
        },
    )
    if moved == 0:
        raise ConflictError(f"'{row.title}' changed meanwhile. Try again.")
    task = await get_task(task_id, user_id)
    if task is None:
        raise NotFoundError(f"Task {task_id} not found")
    return task
