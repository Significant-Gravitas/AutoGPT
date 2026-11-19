"""Build the task row a case starts from, the way production stores it
(``data.task``).

Ids look like production's (uuids) and are derived from the case id, so a
rerun builds the same task.
"""

import uuid
from datetime import UTC, datetime, timedelta

from prisma.enums import TaskCreator, TaskStatus

from backend.data.task import Task

from .models import TaskSpec

T0 = datetime(2026, 11, 9, 9, 0, tzinfo=UTC)


def make_task(case_id: str, spec: TaskSpec, minutes: float = 30) -> Task:
    """The task row: started at ``T0``, settled ``minutes`` later."""
    status = TaskStatus(spec.status)
    settled = status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)
    return Task(
        id=_id(case_id, "task"),
        user_id=_id(case_id, "user"),
        title=spec.title,
        description=_description(spec),
        status=status,
        owner_expert_id=spec.owner,
        created_by=TaskCreator.USER,
        created_at=T0,
        updated_at=T0 + timedelta(minutes=minutes),
        started_at=T0,
        completed_at=T0 + timedelta(minutes=minutes) if settled else None,
        goal=spec.goal,
        done_when=spec.done_when,
        result=spec.result,
        failure_reason=spec.failure_reason,
        attempt=spec.attempt,
        work_session_id=_id(case_id, "session"),
        session_ids=[_id(case_id, "session")],
    )


def _description(spec: TaskSpec) -> str | None:
    text = spec.description
    if not text or not spec.pad_description_to:
        return text
    repeats = spec.pad_description_to // (len(text) + 1) + 1
    return " ".join([text] * repeats)[: spec.pad_description_to]


def _id(case_id: str, kind: str, n: int = 0) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"eval/{case_id}/{kind}/{n}"))
