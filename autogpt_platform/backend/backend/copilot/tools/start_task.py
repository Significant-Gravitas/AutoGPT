"""StartTaskTool: hand a piece of work to a background worker and wrap up.

Otto answers simple requests itself. Work that takes many steps ("fix this
issue", "plan the marketing funnel") becomes a task: a record the user can
follow, worked on by a background subtask in a chat of its own. This turn
wraps up right after (``turn_end``): more tasks and one closing line, nothing
else, so the user can keep talking here; when the
worker finishes, this chat is woken to carry on with the result
(``background_delivery``), which also settles the task's status.
"""

import logging
from typing import Any

from prisma.enums import TaskActivityKind, TaskActor, TaskCreator, TaskStatus

from backend.copilot import task_recorder, turn_end
from backend.copilot.background_subtasks import start_background_subtask
from backend.copilot.model import ChatSession
from backend.copilot.tree import TreeRefusal
from backend.data.db_accessors import task_db, task_record_db
from backend.data.task import MAX_SUBTASKS, Task, TaskPatch
from backend.data.task_record import ActivityInput
from backend.data.task_text import DONE_WHEN_MAX_CHARS, GOAL_MAX_CHARS, clip

from .base import BaseTool
from .external_scope import error_detail
from .models import ErrorResponse, ToolResponseBase
from .task_models import TaskInfo, TaskStartedResponse

logger = logging.getLogger(__name__)

BRIEF_MAX_CHARS = 6_000
# What the worker is told to aim for; the tools refuse past the real caps
# (a model counts characters loosely, so the aim sits well under the cap).
PROGRESS_HINT_CHARS = 60
SUMMARY_HINT_CHARS = 300


class StartTaskTool(BaseTool):
    """Open a task for multi-step work and run it in the background."""

    @property
    def name(self) -> str:
        return "start_task"

    @property
    def description(self) -> str:
        return (
            "Start multi-step work as a task that runs in the background, so "
            "the user can keep chatting; its result comes back to this chat. "
            "Use it for anything bigger than a quick answer. After it, only "
            "more start_task calls and one closing line. Inside a task, it "
            "splits off a subtask for one part of the work."
        )

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short task title."},
                "brief": {
                    "type": "string",
                    "description": (
                        "Everything the worker needs: the goal, what done "
                        "looks like, and facts from this chat."
                    ),
                },
                "goal": {
                    "type": "string",
                    "description": (
                        "What the task is for, in one short plain sentence the "
                        f"user skims. At most {GOAL_MAX_CHARS} characters."
                    ),
                },
                "done_when": {
                    "type": "string",
                    "description": (
                        "What done looks like, so the result can be checked. "
                        f"At most {DONE_WHEN_MAX_CHARS} characters."
                    ),
                },
                "task_id": {
                    "type": "string",
                    "description": (
                        "Set to try a failed task again or carry a finished "
                        "one on, instead of opening a new task."
                    ),
                    "entity": None,
                },
                "expert_id": {
                    "type": "string",
                    "description": (
                        "Set when the work is in a hired expert's area: "
                        "they own it. Omit when you do it."
                    ),
                    "entity": "expert",
                },
            },
            "required": ["title", "brief"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        title: str = "",
        brief: str = "",
        expert_id: str | None = None,
        goal: str | None = None,
        done_when: str | None = None,
        task_id: str | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        assert user_id is not None  # guaranteed by requires_auth
        brief = brief.strip()[:BRIEF_MAX_CHARS]
        if not brief:
            return _error("A task needs a brief for the worker.", session)
        try:
            if task_id:
                task = await _reopen_task(user_id, session, task_id, brief)
            else:
                duplicate = await task_record_db().find_open_task_by_title(
                    user_id, session.session_id, title
                )
                if duplicate is not None:
                    return _already_running(duplicate, session)
                task = await _open_task(
                    user_id, session, title, brief, expert_id, goal, done_when
                )
        except ValueError as e:  # includes NotFoundError
            return _error(str(e), session)
        except Exception as e:
            return _error(error_detail(e, session), session)

        try:
            background = await start_background_subtask(
                user_id, session, worker_prompt(task, brief)
            )
        except TreeRefusal as refused:
            logger.info(
                "start_task ran in the foreground for %s: %s",
                session.session_id,
                refused.message,
            )
            # The task is worked on in this chat, so this chat's tool calls
            # are what its record shows.
            task = await _set_work_session(task, session.session_id, user_id)
            return TaskStartedResponse(
                message=(
                    f"Created task '{task.title}', but it could not run in the "
                    f"background ({refused.message}). Work on it here now."
                ),
                task=TaskInfo.from_task(task),
                in_background=False,
                session_id=session.session_id,
            )

        task = await _link_worker(task, background.child_session_id, user_id)
        turn_end.request_wrap_up(session.session_id)
        return TaskStartedResponse(
            message=(
                f"Started '{task.title}' in the background; the result comes back "
                "to this chat. If you have not told the user yet, tell them in one "
                "short line for everything you started (what, and when it comes "
                "back; no findings and nothing from the brief); if you have, write "
                "nothing more. Do none of the task here."
            ),
            task=TaskInfo.from_task(task),
            in_background=True,
            worker_session_id=background.child_session_id,
            worker_session_link=f"/copilot?sessionId={background.child_session_id}",
            background_task_id=background.id,
            session_id=session.session_id,
        )


def worker_prompt(task: Task, brief: str) -> str:
    """What the background worker is asked to do."""
    return worker_prompt_text(task.title, task.id, brief, task.done_when)


def worker_prompt_text(
    title: str, task_id: str, brief: str, done_when: str | None = None
) -> str:
    """The worker's instructions. The user follows the task on its page and
    skims it, so the worker is told what to write there and how short."""
    return (
        f"You are working on the task '{title}' (task id {task_id}) for "
        f"the user, in the background.\n\n{brief}\n\n"
        "Do the whole task with your tools. Decide anything you can "
        "reasonably decide and say the assumption in your summary: a missing "
        "name, scope or format is yours to choose. The user is not watching "
        "this chat, so ask with ask_question only when a wrong guess could "
        "not be undone (money, sending, deleting) or you have no workable "
        "default at all.\n\n"
        "The user follows this task on its page and only skims it, so keep "
        "what you write there short. Three tools write to it, each called as "
        'run_capability(id="tool:<name>", input={...}):\n'
        '- tool:report_progress {"status_text": "...", "next_step": "..."}: '
        "each time you start a new part of the work (finding, comparing, "
        "writing), what you are doing now, in "
        f"under {PROGRESS_HINT_CHARS} characters ('Comparing pricing pages'). "
        "No findings, numbers or tool names.\n"
        '- tool:add_evidence {"claim": "...", "file" or "tool" or "url": '
        '"..."}: when something happened (sent, booked, published, saved), '
        "one short plain claim ('Reminder sent to Acme') and what shows it: "
        "the file you saved, the one tool you ran, or the page you read.\n"
        '- tool:finish_task {"summary": "...", "file": "..."}: last. The '
        "summary is two or three short plain sentences, under "
        f"{SUMMARY_HINT_CHARS} characters: what was done and the one thing "
        "the user should know, with any assumption that matters.\n\n"
        "Anything longer than that summary (a report, a comparison, drafts, "
        "a plan) is a document: save it with write_workspace_file under a "
        "name that says what it is ('cursor-pricing-brief.md') and pass that "
        "name as finish_task's file. Never put the document in your reply. "
        "finish_task ends your turn, so call it only when everything else, "
        "proof included, is done.\n\n"
        "If the work has parts that teammates should do, split them off with "
        f"start_task (at most {MAX_SUBTASKS}) and end with one short line; a "
        "subtask does its part itself and cannot split further."
        + (f"\n\nDone when: {done_when}" if done_when else "")
    )


async def _open_task(
    user_id: str,
    session: ChatSession,
    title: str,
    brief: str,
    expert_id: str | None,
    goal: str | None = None,
    done_when: str | None = None,
) -> Task:
    """Create the task; started from a chat that is itself working on a task,
    it becomes that task's subtask."""
    working_on = await task_record_db().work_tasks(session.session_id, user_id)
    parent = working_on[0] if working_on else None
    fields: dict[str, Any] = {
        "user_id": user_id,
        "title": title,
        "description": brief,
        "created_by": TaskCreator.EXPERT if session.expert_id else TaskCreator.OTTO,
        "created_by_expert_id": session.expert_id,
        "session_id": session.session_id,
        "organization_id": session.organization_id,
        "team_id": session.team_id,
        "goal": clip(goal, GOAL_MAX_CHARS),
        "done_when": clip(done_when, DONE_WHEN_MAX_CHARS),
        "parent_task_id": parent.id if parent else None,
    }
    try:
        # Work an expert starts from their own chat is theirs unless they
        # hand it to someone else.
        created = await task_db().create_task(
            **fields, owner_expert_id=expert_id or session.expert_id or None
        )
    except ValueError:
        if expert_id or not session.expert_id:
            raise
        # The chat's expert is no longer on the team: Otto keeps the task.
        created = await task_db().create_task(**fields, owner_expert_id=None)
    actor = TaskActor.EXPERT if session.expert_id else TaskActor.OTTO
    await _note(
        created.id,
        user_id,
        ActivityInput(
            kind=TaskActivityKind.CREATED,
            text="Task started",
            actor_type=actor,
            actor_expert_id=session.expert_id,
            chat_session_id=session.session_id,
            is_milestone=True,
        ),
    )
    if parent:
        await _note(
            parent.id,
            user_id,
            ActivityInput(
                kind=TaskActivityKind.DELEGATED,
                text=f"Split off a part: {created.title}",
                actor_type=actor,
                actor_expert_id=session.expert_id,
                chat_session_id=session.session_id,
                is_milestone=True,
                data={"subtask_id": created.id, "owner_expert_id": expert_id},
            ),
        )
    return await task_db().update_task(
        created.id, user_id, TaskPatch(status=TaskStatus.IN_PROGRESS)
    )


async def _reopen_task(
    user_id: str, session: ChatSession, task_id: str, brief: str
) -> Task:
    """Run an existing task again: a failed one counts one more attempt, a
    finished one carries on with the follow-up."""
    current = await task_db().get_task(task_id, user_id)
    if current is None:
        raise ValueError(f"Task {task_id} not found.")
    if current.status in (TaskStatus.IN_PROGRESS, TaskStatus.NEEDS_INPUT):
        raise ValueError(f"'{current.title}' is already being worked on.")
    retry = current.status == TaskStatus.FAILED
    task = await task_db().update_task(
        task_id,
        user_id,
        # The last run's result goes: this run leaves its own.
        TaskPatch(
            status=TaskStatus.IN_PROGRESS, description=brief, retry=retry, result=""
        ),
    )
    await _note(
        task_id,
        user_id,
        ActivityInput(
            kind=TaskActivityKind.RETRIED,
            text=(
                f"Trying again (attempt {task.attempt})" if retry else "Picked up again"
            ),
            actor_type=TaskActor.EXPERT if session.expert_id else TaskActor.OTTO,
            actor_expert_id=session.expert_id,
            chat_session_id=session.session_id,
            is_milestone=True,
        ),
    )
    return task


def _already_running(task: Task, session: ChatSession) -> TaskStartedResponse:
    return TaskStartedResponse(
        message=(
            f"'{task.title}' is already running as a task from this chat, so no "
            "second one was started. Tell the user it is under way."
        ),
        task=TaskInfo.from_task(task),
        in_background=task.work_session_id != session.session_id,
        worker_session_id=task.work_session_id,
        worker_session_link=(
            f"/copilot?sessionId={task.work_session_id}"
            if task.work_session_id
            else None
        ),
        session_id=session.session_id,
    )


async def _note(task_id: str, user_id: str, entry: ActivityInput) -> None:
    """Add a line to the task's activity; the work matters more than the note."""
    try:
        await task_record_db().add_activity(task_id, user_id, entry)
    except Exception as e:
        logger.warning("Could not note activity on task %s: %s", task_id, e)


async def _set_work_session(task: Task, session_id: str, user_id: str) -> Task:
    """Name the chat doing the work, so its tool calls land on this task."""
    try:
        task = await task_db().update_task(
            task.id, user_id, TaskPatch(work_session_id=session_id)
        )
        task_recorder.forget(session_id)
    except Exception as e:
        logger.warning("Could not set the work chat of task %s: %s", task.id, e)
    return task


async def _link_worker(task: Task, worker_session_id: str, user_id: str) -> Task:
    """Link the worker's chat to the task; best-effort, the work has started."""
    try:
        await task_db().attach_task_session(task.id, worker_session_id, user_id)
    except Exception as e:
        logger.warning("Could not link worker chat to task %s: %s", task.id, e)
    return await _set_work_session(task, worker_session_id, user_id)


def _error(message: str, session: ChatSession) -> ErrorResponse:
    return ErrorResponse(
        message=message, error="start_task_failed", session_id=session.session_id
    )
