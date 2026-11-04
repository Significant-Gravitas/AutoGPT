"""Shared plumbing for the Tasks v1 copilot tools."""

import logging
from typing import Any

from backend.api.features.tasks.access import tasks_store
from backend.api.features.tasks.models import (
    Actor,
    Assignee,
    AssigneeType,
    Task,
    TaskFilters,
    TaskStatus,
)
from backend.copilot.model import ChatSession
from backend.data.db_accessors import experts_db
from backend.util.feature_flag import Flag, is_feature_enabled

from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)


class TaskToolError(Exception):
    """A refusal the tool turns into an ``ErrorResponse``."""


class TaskToolBase(BaseTool):
    """Flag gate, auth and error handling shared by every task tool."""

    @property
    def requires_auth(self) -> bool:
        return True

    async def _execute(
        self, user_id: str | None, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        if not user_id:
            return _error("Sign in to use tasks.", session)
        if not await is_feature_enabled(Flag.TASKS_V1, user_id, default=False):
            return _error("Tasks are not enabled for this account.", session)
        try:
            return await self._run(user_id, session, **kwargs)
        except TaskToolError as e:
            return _error(str(e), session)
        except Exception as e:
            logger.exception("Task tool %s failed", self.name)
            return _error(f"The task update failed: {e}", session)

    async def _run(
        self, user_id: str, session: ChatSession, **kwargs: Any
    ) -> ToolResponseBase:
        raise NotImplementedError


def _error(message: str, session: ChatSession) -> ErrorResponse:
    return ErrorResponse(message=message, session_id=session.session_id)


async def resolve_task(user_id: str, session: ChatSession, task_id: str | None) -> Task:
    """The named task, or the one this work chat is working."""
    if task_id:
        task = await tasks_store().get_task(user_id, task_id)
        if task is None:
            raise TaskToolError(f"No task with id {task_id}.")
        return task
    task = await find_work_task(user_id, session)
    if task is None:
        raise TaskToolError(
            "This chat is not working a task. Pass task_id (see list_my_tasks)."
        )
    return task


async def find_work_task(user_id: str, session: ChatSession) -> Task | None:
    """The task this chat is working, if it is a task's work chat."""
    if not session.session_id:
        return None
    found = await tasks_store().get_task_for_work_session(session.session_id)
    if found is None or found[0] != user_id:
        return None
    return found[1]


async def find_open_duplicate(
    user_id: str, title: str, assignee: Assignee | None
) -> Task | None:
    """An open top-level task with the same title (ignoring case and spacing)
    and the same assignee: the same outcome, already tracked."""
    wanted = _normalize_title(title)
    candidates = await tasks_store().list_tasks(
        user_id, TaskFilters(top_level=True, q=title.strip(), limit=20)
    )
    return next(
        (
            t
            for t in candidates
            if t.status != TaskStatus.DONE
            and _normalize_title(t.title) == wanted
            and _same_assignee(t.assignee, assignee)
        ),
        None,
    )


def _same_assignee(a: Assignee | None, b: Assignee | None) -> bool:
    if a is None or b is None:
        return a is b
    if a.type != b.type:
        return False
    return a.type != AssigneeType.EXPERT or a.id == b.id


def distinct_titles(titles: list[str]) -> int:
    return len({_normalize_title(t) for t in titles})


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


async def caller_actor(user_id: str, session: ChatSession) -> Actor:
    """Who is speaking: the session's expert, or Otto."""
    if not session.expert_id:
        return Actor.otto()
    try:
        expert = await experts_db().get_expert(
            user_id, session.expert_id, include_workflows=False
        )
    except Exception:
        expert = None
    if expert is None:
        return Actor.expert(session.expert_id, "Expert")
    return Actor.expert(expert.id, expert.name, expert.avatar_url, expert.color)


async def resolve_assignee_name(user_id: str, name: str | None) -> Assignee | None:
    """Map 'otto' or an expert's name/id to an ``Assignee``. The user never
    holds a task, so 'me' or 'user' means Otto too."""
    store = tasks_store()
    if not name:
        return None
    key = name.strip().lower()
    if key in ("otto", "autopilot", "me", "user", "you"):
        return await store.resolve_assignee(user_id, AssigneeType.OTTO, None)
    experts = await store.list_experts_for_user(user_id)
    match = next(
        (e for e in experts if e.id == name or e.name.strip().lower() == key), None
    )
    if match is None:
        names = ", ".join(e.name for e in experts) or "none"
        raise TaskToolError(f"No expert named {name}. Your team: {names}.")
    return await store.resolve_assignee(user_id, AssigneeType.EXPERT, match.id)


def dump(value: Any) -> dict[str, Any] | None:
    return value.model_dump(mode="json") if value is not None else None
