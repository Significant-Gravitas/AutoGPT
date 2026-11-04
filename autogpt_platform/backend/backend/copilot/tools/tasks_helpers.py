"""Shared plumbing for the Tasks v1 copilot tools."""

import logging
from typing import Any

from backend.api.features.tasks.access import tasks_store
from backend.api.features.tasks.models import Actor, Assignee, AssigneeType, Task
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
    store = tasks_store()
    if task_id:
        task = await store.get_task(user_id, task_id)
        if task is None:
            raise TaskToolError(f"No task with id {task_id}.")
        return task
    found = (
        await store.get_task_for_work_session(session.session_id)
        if session.session_id
        else None
    )
    if found is None or found[0] != user_id:
        raise TaskToolError(
            "This chat is not working a task. Pass task_id (see list_my_tasks)."
        )
    return found[1]


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
    """Map 'otto', 'me' or an expert's name/id to an ``Assignee``."""
    store = tasks_store()
    if not name:
        return None
    key = name.strip().lower()
    if key in ("otto", "autopilot"):
        return await store.resolve_assignee(user_id, AssigneeType.OTTO, None)
    if key in ("me", "user", "you"):
        return await store.resolve_assignee(user_id, AssigneeType.USER, None)
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
