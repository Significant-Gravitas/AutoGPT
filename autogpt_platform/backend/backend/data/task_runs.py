"""Each run of a task, read back from its activity.

A run starts with a STARTED or RETRIED line naming the chat it runs in, and
ends with the line that settles it in that chat (DONE, FAILED, or a STATUS
line marked with ``run_end``). No table of its own: a repeating task's runs
are its activity grouped by chat.
"""

from datetime import datetime
from typing import Any

import prisma.models
from prisma.enums import TaskActivityKind
from pydantic import BaseModel, Field

from backend.data.task_schedule_meta import RunEnd
from backend.data.task_text import clip

RUNS_LIMIT = 30
RUN_RESULT_MAX_CHARS = 300
# Enough lines for the runs shown, read newest first.
_LINES_READ = 400
_START_KINDS = (TaskActivityKind.STARTED, TaskActivityKind.RETRIED)
_END_BY_KIND: dict[TaskActivityKind, RunEnd] = {
    TaskActivityKind.DONE: "done",
    TaskActivityKind.FAILED: "failed",
}
# Lines written before runs were marked, read as the end they say.
_CANCELLED_TEXTS = ("Cancelled", "You stopped it", "You stopped this run")
_FAILED_PREFIXES = ("Did not finish: ", "Failed: ")


class TaskRun(BaseModel):
    """One run of a task, in the chat it ran in."""

    chat_session_id: str
    started_at: datetime
    ended_at: datetime | None = None
    outcome: RunEnd | None = Field(
        default=None, description="How it ended; None while it runs."
    )
    result: str | None = Field(default=None, description="Its result, cut short.")
    failure_reason: str | None = None
    trigger: str | None = Field(
        default=None, description="What started it, in a few words: an event."
    )


def run_end_data(ended: RunEnd, result: str | None = None) -> dict[str, Any]:
    """What the line that ends a run carries, so the run reads back."""
    data: dict[str, Any] = {"run_end": ended}
    short = clip(result, RUN_RESULT_MAX_CHARS) if result else None
    if short:
        data["result"] = short
    return data


async def list_task_runs(task_id: str) -> list[TaskRun]:
    """The task's runs, newest first. The caller has checked the owner."""
    rows = await prisma.models.TaskActivity.prisma().find_many(
        where={
            "taskId": task_id,
            "chatSessionId": {"not": None},
            "kind": {"in": [*_START_KINDS, *_END_BY_KIND, TaskActivityKind.STATUS]},
        },
        order={"createdAt": "desc"},
        take=_LINES_READ,
    )
    return runs_from_activity(list(reversed(rows)))[:RUNS_LIMIT]


def runs_from_activity(rows: list[prisma.models.TaskActivity]) -> list[TaskRun]:
    """Runs from activity lines oldest first; returned newest first."""
    runs: list[TaskRun] = []
    open_runs: dict[str, int] = {}
    for row in rows:
        chat = row.chatSessionId
        if not chat:
            continue
        if row.kind in _START_KINDS:
            if chat not in open_runs:
                open_runs[chat] = len(runs)
                runs.append(
                    TaskRun(
                        chat_session_id=chat,
                        started_at=row.createdAt,
                        trigger=_trigger(row),
                    )
                )
            continue
        ended = _run_end(row)
        if ended is None or chat not in open_runs:
            continue
        index = open_runs.pop(chat)
        runs[index] = _ended(runs[index], row, ended)
    return runs[::-1]


def _trigger(row: prisma.models.TaskActivity) -> str | None:
    data = row.data if isinstance(row.data, dict) else {}
    trigger = data.get("trigger")
    return trigger if isinstance(trigger, str) and trigger else None


def _run_end(row: prisma.models.TaskActivity) -> RunEnd | None:
    data = row.data if isinstance(row.data, dict) else {}
    marked = data.get("run_end")
    if marked in ("done", "failed", "cancelled"):
        return marked
    if row.kind in _END_BY_KIND:
        return _END_BY_KIND[row.kind]
    if row.kind == TaskActivityKind.STATUS and row.text in _CANCELLED_TEXTS:
        return "cancelled"
    return None


def _ended(run: TaskRun, row: prisma.models.TaskActivity, ended: RunEnd) -> TaskRun:
    data = row.data if isinstance(row.data, dict) else {}
    result = data.get("result")
    prefix = next((p for p in _FAILED_PREFIXES if row.text.startswith(p)), None)
    reason = row.text.removeprefix(prefix) if ended == "failed" and prefix else None
    return run.model_copy(
        update={
            "ended_at": row.createdAt,
            "outcome": ended,
            "result": result if isinstance(result, str) else None,
            "failure_reason": reason,
        }
    )
