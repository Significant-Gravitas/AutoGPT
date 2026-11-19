"""Task controls against the live database and services.

A live case makes a real task for an existing user (``--live-user-id``),
starts, retries or stops it through production's own functions
(``copilot.task_run``), waits for the work chat's turn to end on the running
copilot executor, and checks what the database then holds. It needs
Postgres, Redis, RabbitMQ and a running copilot executor, and each start is
a paid worker turn on the platform's models. Every task it makes is archived
when the case ends.
"""

import asyncio
from typing import Protocol

from prisma.enums import TaskActivityKind, TaskActor, TaskCreator, TaskStatus

from backend.copilot.eval.tasks.models import CheckResult
from backend.copilot.task_run import cancel_task_run, start_task_run
from backend.data import task as task_data
from backend.data.task import OPEN_STATUSES, Task, TaskPatch
from backend.data.task_record import TaskRecord, get_task_record
from backend.data.task_suggestion import SuggestedTaskInput, create_suggested_task

from .models import FeatureCase, TaskSpec

POLL_SECONDS = 5.0
SETTLE_TIMEOUT_SECONDS = 900.0
STOP_AFTER_SECONDS = 20.0
QUIET_AFTER_STOP_SECONDS = 30.0


class TaskApi(Protocol):
    """What a live case does to a task; the real one is production's code."""

    async def create(self, user_id: str, spec: TaskSpec) -> Task: ...
    async def suggest(self, user_id: str, spec: TaskSpec) -> Task: ...
    async def fail(self, user_id: str, task_id: str, reason: str) -> Task: ...
    async def run(self, user_id: str, task_id: str, note: str | None) -> Task: ...
    async def cancel(self, user_id: str, task_id: str) -> Task: ...
    async def get(self, user_id: str, task_id: str) -> Task | None: ...
    async def record(self, user_id: str, task_id: str) -> TaskRecord | None: ...
    async def archive(self, user_id: str, task_id: str) -> None: ...
    async def sleep(self, seconds: float) -> None: ...


class RealTaskApi:
    async def create(self, user_id: str, spec: TaskSpec) -> Task:
        return await task_data.create_task(
            user_id=user_id,
            title=spec.title,
            created_by=TaskCreator.USER,
            description=spec.description,
            goal=spec.goal,
            done_when=spec.done_when,
        )

    async def suggest(self, user_id: str, spec: TaskSpec) -> Task:
        draft = SuggestedTaskInput(
            title=spec.title,
            goal=spec.goal or "",
            brief=spec.description or "",
            reason="Follows from the eval's simulated history.",
        )
        return await create_suggested_task(user_id, draft)

    async def fail(self, user_id: str, task_id: str, reason: str) -> Task:
        patch = TaskPatch(status=TaskStatus.FAILED, failure_reason=reason)
        return await task_data.update_task(task_id, user_id, patch)

    async def run(self, user_id: str, task_id: str, note: str | None) -> Task:
        return await start_task_run(user_id, task_id, note)

    async def cancel(self, user_id: str, task_id: str) -> Task:
        return await cancel_task_run(user_id, task_id)

    async def get(self, user_id: str, task_id: str) -> Task | None:
        return await task_data.get_task(task_id, user_id)

    async def record(self, user_id: str, task_id: str) -> TaskRecord | None:
        return await get_task_record(task_id, user_id)

    async def archive(self, user_id: str, task_id: str) -> None:
        await task_data.archive_task(task_id, user_id)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


async def run_live(
    case: FeatureCase, api: TaskApi, user_id: str
) -> tuple[list[CheckResult], TaskRecord | None]:
    """Play the case's action and check the task it leaves."""
    spec = case.task
    if spec is None or case.live is None:
        raise ValueError(f"{case.id}: a live case needs a task and an action")
    task = await (api.suggest if case.live == "start_suggestion" else api.create)(
        user_id, spec
    )
    try:
        checks = await _ACTIONS[case.live](case, api, user_id, task)
        return checks, await api.record(user_id, task.id)
    finally:
        await api.archive(user_id, task.id)


async def _start(
    case: FeatureCase, api: TaskApi, user_id: str, task: Task
) -> list[CheckResult]:
    await api.run(user_id, task.id, case.note)
    return await _settled_checks(api, user_id, task.id)


async def _retry(
    case: FeatureCase, api: TaskApi, user_id: str, task: Task
) -> list[CheckResult]:
    before = await api.fail(
        user_id, task.id, "The first attempt used the wrong sources."
    )
    after = await api.run(user_id, task.id, case.note)
    record = await api.record(user_id, task.id)
    noted = [
        a
        for a in (record.activity if record else [])
        if a.kind == TaskActivityKind.RETRIED
        and a.data.get("instructions") == case.note
    ]
    checks = [
        _check(
            "attempt_up",
            after.attempt == before.attempt + 1,
            f"{before.attempt} -> {after.attempt}",
        ),
        _check("note_on_record", bool(noted)),
    ]
    return checks + await _settled_checks(api, user_id, task.id)


async def _stop(
    case: FeatureCase, api: TaskApi, user_id: str, task: Task
) -> list[CheckResult]:
    await api.run(user_id, task.id, case.note)
    await api.sleep(STOP_AFTER_SECONDS)
    stopped = await api.cancel(user_id, task.id)
    await api.sleep(QUIET_AFTER_STOP_SECONDS)
    final = await api.get(user_id, task.id)
    record = await api.record(user_id, task.id)
    lines = sorted(record.activity if record else [], key=lambda a: a.created_at)
    stop = next(
        (
            a
            for a in lines
            if a.actor_type == TaskActor.USER and a.text == "You stopped it"
        ),
        None,
    )
    later = [a.text for a in lines if stop and a.created_at > stop.created_at]
    return [
        _check("cancelled", stopped.status == TaskStatus.CANCELLED and final is not None and final.status == TaskStatus.CANCELLED, final.status.value if final else "gone"),
        _check("nothing_after_stop", stop is not None and not later, "; ".join(later[:3]) or ("no stop line" if stop is None else "")),
    ]  # fmt: skip


async def _start_suggestion(
    case: FeatureCase, api: TaskApi, user_id: str, task: Task
) -> list[CheckResult]:
    started = await api.run(user_id, task.id, None)
    checks = [
        _check("was_suggested", task.is_suggested),
        _check("suggested_cleared", not started.is_suggested),
        _check(
            "runs_like_any_task",
            started.status == TaskStatus.IN_PROGRESS and bool(started.work_session_id),
            started.status.value,
        ),
    ]
    await api.cancel(user_id, task.id)  # the start is what is checked
    return checks


async def _settled_checks(
    api: TaskApi, user_id: str, task_id: str
) -> list[CheckResult]:
    task = await _wait_settled(api, user_id, task_id)
    if task is None:
        return [_check("settled", False, "task gone")]
    status = task.status
    checks = [
        _check(
            "settled",
            status not in OPEN_STATUSES or status == TaskStatus.NEEDS_INPUT,
            status.value,
        )
    ]
    if status == TaskStatus.DONE:
        checks.append(_check("done_has_result", bool((task.result or "").strip())))
    if status == TaskStatus.FAILED:
        checks.append(
            _check(
                "failed_has_reason",
                bool((task.failure_reason or "").strip()),
                task.failure_reason or "",
            )
        )
    return checks  # fmt: skip


async def _wait_settled(api: TaskApi, user_id: str, task_id: str) -> Task | None:
    waited = 0.0
    task = await api.get(user_id, task_id)
    while task is not None and task.status in (TaskStatus.TODO, TaskStatus.IN_PROGRESS):
        if waited >= SETTLE_TIMEOUT_SECONDS:
            break
        await api.sleep(POLL_SECONDS)
        waited += POLL_SECONDS
        task = await api.get(user_id, task_id)
    return task


def _check(name: str, passed: bool, detail: str = "") -> CheckResult:
    return CheckResult(name=name, passed=passed, detail=detail)


_ACTIONS = {
    "start": _start,
    "retry": _retry,
    "stop": _stop,
    "start_suggestion": _start_suggestion,
}
