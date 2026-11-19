"""What a user may change on a task by hand (``PATCH /tasks/{id}``), and the
activity lines those changes leave.

A user moves a task to To do, Done or Cancelled only: Working, Needs you and
Failed come from the work itself. Reopening, and giving the task to someone
else, wait until the work going on it stops.
"""

from fastapi import HTTPException
from prisma.enums import TaskStatus
from starlette.status import HTTP_409_CONFLICT, HTTP_422_UNPROCESSABLE_CONTENT

from backend.data.task import Task

from .model import TaskUpdateRequest

# The statuses a run sets; a user stops the run instead.
WORKED_STATUSES = (TaskStatus.IN_PROGRESS, TaskStatus.NEEDS_INPUT)
USER_STATUSES = (TaskStatus.TODO, TaskStatus.DONE, TaskStatus.CANCELLED)
STATUS_REFUSED = (
    "You can move a task to To do, Done or Cancelled. To work on it, start it."
)
REOPEN_REFUSED = "It is being worked on. Stop it first, then move it back to To do."
OWNER_REFUSED = "It is being worked on. Stop it before giving it to someone else."
# The words the task UI shows for each status.
STATUS_WORDS = {
    TaskStatus.TODO: "To do",
    TaskStatus.IN_PROGRESS: "Working",
    TaskStatus.NEEDS_INPUT: "Needs you",
    TaskStatus.DONE: "Done",
    TaskStatus.FAILED: "Failed",
    TaskStatus.CANCELLED: "Cancelled",
}


def check_change(before: Task, body: TaskUpdateRequest) -> None:
    """Refuse a change the user may not make, before anything is written."""
    status = body.status
    if status is not None and status != before.status:
        if status not in USER_STATUSES:
            raise HTTPException(HTTP_422_UNPROCESSABLE_CONTENT, STATUS_REFUSED)
        if status == TaskStatus.TODO and before.status in WORKED_STATUSES:
            raise HTTPException(HTTP_409_CONFLICT, REOPEN_REFUSED)
    if changes_owner(before, body) and before.status in WORKED_STATUSES:
        raise HTTPException(HTTP_409_CONFLICT, OWNER_REFUSED)


def changes_owner(before: Task, body: TaskUpdateRequest) -> bool:
    if body.assign_to_otto:
        return before.owner_expert_id is not None
    return bool(body.owner_expert_id) and body.owner_expert_id != (
        before.owner_expert_id
    )


def stops_work(before: Task, body: TaskUpdateRequest) -> bool:
    """Done or Cancelled on a task being worked on: its run stops first."""
    return (
        body.status in (TaskStatus.DONE, TaskStatus.CANCELLED)
        and before.status in WORKED_STATUSES
    )


def user_change_lines(before: Task, after: Task) -> list[str]:
    """One activity line for each thing the user just changed, in the words
    the task UI uses."""
    lines: list[str] = []
    if before.status != after.status:
        lines.append(f"You moved it to {STATUS_WORDS[after.status]}")
    if before.title != after.title:
        lines.append(f"You renamed it to {after.title}")
    if before.description != after.description:
        lines.append("You changed the description")
    if before.goal != after.goal:
        lines.append("You changed the goal")
    if before.done_when != after.done_when:
        lines.append("You changed what done looks like")
    if before.owner_expert_id != after.owner_expert_id:
        lines.append(
            "You gave it to Otto"
            if after.owner_expert_id is None
            else "You changed who owns it"
        )
    return lines
