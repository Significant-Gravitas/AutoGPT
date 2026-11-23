"""Task controls without the database: what the worker is sent when a task
is started or tried again (``task_run.run_prompt``), how a run settles for
every way its turn can end (``task_run_settle``, ``data.task_run``), and a
worker's real turn on that message against the simulated world.

The worker turn reuses the task eval's engines: its first message is
production's ``run_prompt`` for the case's task, sent as a worker brief, so
its tool calls land on a simulated task record.
"""

import prisma.models
from prisma.enums import TaskStatus

from backend.copilot.background_task import BACKGROUND_TIMEOUT_MESSAGE
from backend.copilot.eval.tasks.models import (
    Case,
    CheckResult,
    Expected,
    Run,
    ToolCall,
    Turn,
    WorkerBrief,
)
from backend.copilot.stream_registry import CANCELLED_MESSAGE
from backend.copilot.task_run import run_brief, run_prompt
from backend.copilot.task_run_settle import turn_ending
from backend.copilot.tools.start_task import worker_prompt_text
from backend.data.task import OPEN_STATUSES, Task
from backend.data.task_run import (
    NO_RESULT_REASON,
    TurnEnding,
    _settled_status,
    reply_reason,
)

from .models import FeatureCase, FeatureChecks

_PARTS = ("title", "goal", "done_when", "description", "note")


def brief_checks(
    checks: FeatureChecks, task: Task, note: str | None
) -> list[CheckResult]:
    """Each part of the task the worker must be told is in its first message."""
    prompt = " ".join(run_prompt(task, note).split())
    values = {
        "title": task.title,
        "goal": task.goal,
        "done_when": task.done_when,
        "description": task.description,
        "note": note,
    }
    results = []
    for part in checks.brief_has:
        if part not in _PARTS:
            raise ValueError(f"Unknown brief part {part}")
        value = " ".join((values[part] or "").split())
        results.append(
            CheckResult(
                name=f"brief_has:{part}",
                passed=bool(value) and value in prompt,
                detail="" if value in prompt else f"missing: '{value[:60]}'",
            )
        )
    return results


def settle_table() -> list[CheckResult]:
    """Every way a run's turn can end, with and without a result and with
    and without open subtasks: the task must leave IN_PROGRESS unless its
    subtasks are still working, and DONE needs a result, FAILED a reason."""
    endings: dict[str | None, TurnEnding] = {
        None: "completed",
        "boom": "failed",
        BACKGROUND_TIMEOUT_MESSAGE: "timed_out",
        CANCELLED_MESSAGE: "cancelled",
    }
    results = [
        CheckResult(
            name="turn_ending_mapping",
            passed=all(turn_ending(error) == want for error, want in endings.items()),
        )
    ]
    for ending in endings.values():
        for result in ("Saved the binder.", None):
            for open_subtask in (False, True):
                status, reason = _settled_status(_row(result, open_subtask), ending)
                name = f"settle:{ending}:{'result' if result else 'no_result'}"
                name += ":open_subtask" if open_subtask else ""
                passed, detail = _settled_ok(
                    ending, result, open_subtask, status, reason
                )
                results.append(CheckResult(name=name, passed=passed, detail=detail))
    return results + [_worker_words_check()]


_LAST_WORDS = "Shopify is not connected, so I could not pull the orders. Connect it and run me again."


def _worker_words_check() -> CheckResult:
    """A clean turn that left no result fails with what the worker last
    said, not a generic line."""
    status, reason = _settled_status(_row(None, False), "completed", _LAST_WORDS)
    passed = status == TaskStatus.FAILED and "Shopify is not connected" in (
        reason or ""
    )
    return CheckResult(
        name="settle:completed:no_result:worker_words",
        passed=passed,
        detail=reason or "",
    )


def _row(result: str | None, open_subtask: bool) -> prisma.models.Task:
    child = prisma.models.Task.model_construct(status=TaskStatus.IN_PROGRESS)
    return prisma.models.Task.model_construct(
        result=result, Subtasks=[child] if open_subtask else []
    )


def _settled_ok(
    ending: TurnEnding,
    result: str | None,
    open_subtask: bool,
    status: TaskStatus | None,
    reason: str | None,
) -> tuple[bool, str]:
    detail = f"{status.value if status else 'left open'} ({reason or 'no reason'})"
    if status is None:
        return ending == "completed" and open_subtask, detail
    if status in OPEN_STATUSES:
        return False, detail
    if status == TaskStatus.DONE:
        return bool(result), detail
    if status == TaskStatus.FAILED:
        return bool(reason), detail
    return ending == "cancelled", detail


def worker_case(case: FeatureCase, task: Task) -> Case:
    """The case as the task eval runs it: one worker turn whose message is
    production's ``run_prompt`` for the task."""
    brief = run_brief(task, case.note)
    sent = worker_prompt_text(task.title, task.id, brief, task.done_when)
    if sent != run_prompt(task, case.note):
        raise RuntimeError(
            "run_prompt no longer wraps run_brief: update controls.worker_case"
        )
    return Case(
        id=case.id,
        category="controls",
        world=case.world,
        title=case.title,
        turns=[
            Turn(
                worker=WorkerBrief(
                    title=task.title,
                    brief=brief,
                    task_id=task.id,
                    done_when=task.done_when,
                )
            )
        ],
        expected=Expected(
            behaviour="work",
            must=case.expected.must,
            must_not=case.expected.must_not,
            checks=case.expected.checks.worker,
        ),
        human_example=case.human_example,
    )


def settled_as(run: Run) -> str:
    """How production would settle the run when the worker's turn ended:
    DONE when it left a result, FAILED when that result says it was not done
    (``finish_task`` outcome blocked or failed), NEEDS_INPUT when it ended on
    a question, FAILED otherwise."""
    finish = _last_finish(run)
    if finish is not None:
        done = _finish_outcome(finish) == "done"
        return (TaskStatus.DONE if done else TaskStatus.FAILED).value
    if run.turns and run.turns[-1].ended_by == "question":
        return TaskStatus.NEEDS_INPUT.value
    return TaskStatus.FAILED.value


def settled_reason(run: Run) -> str | None:
    """The failure reason production would store: a not-done finish's
    summary, or for a run that left no result the worker's last words
    (``task_run.reply_reason``)."""
    if settled_as(run) != TaskStatus.FAILED.value or not run.turns:
        return None
    finish = _last_finish(run)
    if finish is not None:
        return str((finish.arguments or {}).get("summary") or "") or None
    last = run.turns[-1]
    said = last.final_reply or (last.reply.split("\n\n")[-1] if last.reply else "")
    return reply_reason(said) or NO_RESULT_REASON


def worker_extra_checks(checks: FeatureChecks, run: Run) -> list[CheckResult]:
    results = []
    status = settled_as(run)
    if checks.settles_as:
        results.append(
            CheckResult(
                name="settles_as", passed=status in checks.settles_as, detail=status
            )
        )
    if checks.fails_with_reason is not None:
        reason = settled_reason(run)
        own = status != TaskStatus.FAILED.value or (
            bool(reason) and reason != NO_RESULT_REASON
        )
        results.append(
            CheckResult(name="fails_with_reason", passed=own, detail=reason or "")
        )
    if checks.mentions_any:
        said = _result_text(run).lower()
        hit = [w for w in checks.mentions_any if w.lower() in said]
        results.append(
            CheckResult(
                name="mentions_any", passed=bool(hit), detail=", ".join(hit) or "none"
            )
        )
    return results


def _result_text(run: Run) -> str:
    """What the worker left: its summary, its files' names and contents."""
    calls = [c for t in run.turns for c in t.tool_calls if not c.refused]
    kept = [c for c in calls if c.name in ("finish_task", "write_workspace_file")]
    return " ".join(str(v) for c in kept for v in c.arguments.values())


def _last_finish(run: Run) -> ToolCall | None:
    calls = [c for t in run.turns for c in t.tool_calls if not c.refused]
    finishes = [c for c in calls if c.name == "finish_task" and not c.failed]
    return finishes[-1] if finishes else None


def _finish_outcome(call: ToolCall) -> str:
    return str((call.arguments or {}).get("outcome") or "done").strip().lower()
