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

from prisma.enums import TaskCreator, TaskStatus

from backend.copilot import turn_end
from backend.copilot.background_subtasks import start_background_subtask
from backend.copilot.model import ChatSession
from backend.copilot.tree import TreeRefusal
from backend.data.db_accessors import task_db
from backend.data.task import Task, TaskPatch

from .base import BaseTool
from .external_scope import error_detail
from .models import ErrorResponse, ToolResponseBase
from .task_models import TaskInfo, TaskStartedResponse

logger = logging.getLogger(__name__)

BRIEF_MAX_CHARS = 6_000


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
            "more start_task calls and one closing line."
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
        **kwargs,
    ) -> ToolResponseBase:
        assert user_id is not None  # guaranteed by requires_auth
        brief = brief.strip()[:BRIEF_MAX_CHARS]
        if not brief:
            return _error("A task needs a brief for the worker.", session)
        try:
            task = await _open_task(user_id, session, title, brief, expert_id)
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
                "short line; if you have, write nothing more. Do none of the task "
                "here."
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
    return (
        f"You are working on the task '{task.title}' (task id {task.id}) for "
        f"the user, in the background.\n\n{brief}\n\n"
        "Do the whole task with your tools. Decide anything you can "
        "reasonably decide and note it as an assumption. If only the user "
        "can unblock you, ask with ask_question. Finish with the result and "
        "the assumptions you made."
    )


async def _open_task(
    user_id: str,
    session: ChatSession,
    title: str,
    brief: str,
    expert_id: str | None,
) -> Task:
    created = await task_db().create_task(
        user_id=user_id,
        title=title,
        description=brief,
        owner_expert_id=expert_id or None,
        created_by=TaskCreator.EXPERT if session.expert_id else TaskCreator.OTTO,
        created_by_expert_id=session.expert_id,
        session_id=session.session_id,
        organization_id=session.organization_id,
        team_id=session.team_id,
    )
    return await task_db().update_task(
        created.id, user_id, TaskPatch(status=TaskStatus.IN_PROGRESS)
    )


async def _link_worker(task: Task, worker_session_id: str, user_id: str) -> Task:
    """Link the worker's chat to the task; best-effort, the work has started."""
    try:
        return await task_db().attach_task_session(task.id, worker_session_id, user_id)
    except Exception as e:
        logger.warning("Could not link worker chat to task %s: %s", task.id, e)
        return task


def _error(message: str, session: ChatSession) -> ErrorResponse:
    return ErrorResponse(
        message=message, error="start_task_failed", session_id=session.session_id
    )
