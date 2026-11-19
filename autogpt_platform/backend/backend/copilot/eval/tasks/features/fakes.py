"""Fakes the feature eval's tests play cases on: a model backend with a fixed
answer, and production's task controls kept in memory."""

from datetime import timedelta
from typing import Any

from prisma.enums import TaskActivityKind, TaskActor, TaskStatus

from backend.copilot.eval.tasks.llm import Reply
from backend.data.task import OPEN_STATUSES, Task
from backend.data.task_record import TaskActivity, TaskRecord

from .live import POLL_SECONDS
from .models import TaskSpec
from .records import make_task


class FakeBackend:
    """A backend whose text answer is fixed; it records what it was sent."""

    def __init__(self, answer: str = "") -> None:
        self.answer, self.sent = answer, []

    async def chat(
        self, model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> Reply:
        return Reply()

    async def text(self, model: str, system: str, user: str, json_out: bool) -> str:
        self.sent.append(user)
        return self.answer


class FakeApi:
    """Production's task controls in memory: a run settles DONE on the next poll."""

    def __init__(self, keep_working_after_stop: bool = False) -> None:
        self.tasks: dict[str, Task] = {}
        self.activity: list[TaskActivity] = []
        self.archived: list[str] = []
        self.keep_working = keep_working_after_stop
        self.clock = 0.0

    def _line(
        self,
        task_id: str,
        text: str,
        kind: TaskActivityKind,
        actor: TaskActor,
        **data: Any,
    ) -> None:
        base = next(iter(self.tasks.values())).created_at
        self.activity.append(
            TaskActivity(
                id=f"l{len(self.activity)}",
                task_id=task_id,
                created_at=base + timedelta(seconds=self.clock + len(self.activity)),
                actor_type=actor,
                kind=kind,
                text=text,
                data=data,
            )
        )

    async def create(self, user_id: str, spec: TaskSpec) -> Task:
        task = make_task("live", spec).model_copy(
            update={"status": TaskStatus.TODO, "completed_at": None}
        )
        self.tasks[task.id] = task
        return task

    async def suggest(self, user_id: str, spec: TaskSpec) -> Task:
        task = await self.create(user_id, spec)
        self.tasks[task.id] = task.model_copy(update={"is_suggested": True})
        return self.tasks[task.id]

    async def fail(self, user_id: str, task_id: str, reason: str) -> Task:
        self.tasks[task_id] = self.tasks[task_id].model_copy(
            update={"status": TaskStatus.FAILED}
        )
        return self.tasks[task_id]

    async def run(self, user_id: str, task_id: str, note: str | None) -> Task:
        before = self.tasks[task_id]
        retry = before.status == TaskStatus.FAILED
        self.tasks[task_id] = before.model_copy(
            update={
                "status": TaskStatus.IN_PROGRESS,
                "is_suggested": False,
                "attempt": before.attempt + retry,
            }
        )
        kind = TaskActivityKind.RETRIED if retry else TaskActivityKind.STARTED
        self._line(
            task_id,
            "You started it",
            kind,
            TaskActor.USER,
            **({"instructions": note} if note else {}),
        )
        return self.tasks[task_id]

    async def cancel(self, user_id: str, task_id: str) -> Task:
        task = self.tasks[task_id]
        if task.status not in OPEN_STATUSES:
            return task  # as production: a settled task is returned as it is
        self.tasks[task_id] = task.model_copy(update={"status": TaskStatus.CANCELLED})
        self._line(task_id, "You stopped it", TaskActivityKind.STATUS, TaskActor.USER)
        return self.tasks[task_id]

    async def get(self, user_id: str, task_id: str) -> Task | None:
        return self.tasks.get(task_id)

    async def record(self, user_id: str, task_id: str) -> TaskRecord | None:
        return TaskRecord(
            task=self.tasks[task_id], activity=list(reversed(self.activity))
        )

    async def archive(self, user_id: str, task_id: str) -> None:
        self.archived.append(task_id)

    async def sleep(self, seconds: float) -> None:
        """A poll finds the work done; the stop's waits find it still going."""
        self.clock += seconds
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.IN_PROGRESS and seconds == POLL_SECONDS:
                self._line(
                    task_id, "Writing", TaskActivityKind.PROGRESS, TaskActor.OTTO
                )
                update = {"status": TaskStatus.DONE, "result": "Saved."}
                self.tasks[task_id] = task.model_copy(update=update)
            elif self.keep_working and task.status == TaskStatus.CANCELLED:
                self._line(
                    task_id, "Searched the web", TaskActivityKind.TOOL, TaskActor.OTTO
                )
