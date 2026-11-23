"""A task's runs read back from its activity: one per start, ended by the line
that settled it in that run's chat."""

import json
from datetime import UTC, datetime, timedelta

import prisma.models
from prisma.enums import TaskActivityKind, TaskActor

from backend.data.task_runs import run_end_data, runs_from_activity

_AT = datetime(2026, 11, 17, 8, 0, tzinfo=UTC)


def _line(
    minute: int,
    kind: TaskActivityKind,
    chat: str | None,
    text: str = "",
    data: dict | None = None,
) -> prisma.models.TaskActivity:
    return prisma.models.TaskActivity(
        id=f"line-{minute}-{kind}",
        createdAt=_AT + timedelta(minutes=minute),
        taskId="task-1",
        actorType=TaskActor.SYSTEM,
        chatSessionId=chat,
        kind=kind,
        text=text or kind.value,
        isMilestone=True,
        data=json.dumps(data or {}),
    )


def test_each_run_reads_back_newest_first_with_how_it_ended():
    rows = [
        _line(0, TaskActivityKind.CREATED, "origin", "Repeats: Every 3 minutes"),
        _line(3, TaskActivityKind.STARTED, "run-1"),
        _line(4, TaskActivityKind.TOOL, "run-1", "Searched the web"),
        _line(
            5,
            TaskActivityKind.DONE,
            "run-1",
            "Finished",
            run_end_data("done", "Found 3 posts."),
        ),
        _line(6, TaskActivityKind.STARTED, "run-2"),
        _line(
            7,
            TaskActivityKind.FAILED,
            "run-2",
            "Failed: The work ran out of time.",
            run_end_data("failed"),
        ),
        _line(9, TaskActivityKind.STARTED, "run-3"),
        _line(
            10,
            TaskActivityKind.STATUS,
            "run-3",
            "You stopped this run",
            run_end_data("cancelled"),
        ),
        _line(12, TaskActivityKind.STARTED, "run-4"),
    ]

    runs = runs_from_activity(rows)

    assert [run.chat_session_id for run in runs] == ["run-4", "run-3", "run-2", "run-1"]
    assert [run.outcome for run in runs] == [None, "cancelled", "failed", "done"]
    assert runs[0].ended_at is None
    assert runs[2].failure_reason == "The work ran out of time."
    assert runs[3].result == "Found 3 posts."
    assert runs[3].started_at == _AT + timedelta(minutes=3)
    assert runs[3].ended_at == _AT + timedelta(minutes=5)


def test_lines_from_before_runs_were_marked_still_read():
    rows = [
        _line(0, TaskActivityKind.STARTED, "run-1"),
        _line(1, TaskActivityKind.STATUS, "run-1", "Cancelled"),
        _line(2, TaskActivityKind.RETRIED, "run-2"),
        _line(3, TaskActivityKind.DONE, "run-2", "Finished"),
    ]

    runs = runs_from_activity(rows)

    assert [run.outcome for run in runs] == ["done", "cancelled"]


def test_a_long_result_is_cut_short():
    data = run_end_data("done", "word " * 200)

    assert len(data["result"]) <= 300
