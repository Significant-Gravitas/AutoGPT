"""Helpers shared by the Todoist blocks for working with todoist-api-python."""

from collections.abc import Iterable
from datetime import datetime
from typing import Any, Literal, TypeVar

from todoist_api_python.api import TodoistAPI
from todoist_api_python.models import Project, Task

from backend.blocks.todoist._auth import TodoistCredentials

T = TypeVar("T")


def get_api(credentials: TodoistCredentials) -> TodoistAPI:
    return TodoistAPI(credentials.access_token.get_secret_value())


def flatten_pages(pages: Iterable[list[T]]) -> list[T]:
    """The SDK's list methods return paginated results; collect every page."""
    return [item for page in pages for item in page]


def parse_duration_unit(value: str | None) -> Literal["minute", "day"] | None:
    if value is None:
        return None
    if value == "minute":
        return "minute"
    if value == "day":
        return "day"
    raise ValueError(f"Invalid duration unit {value!r}, expected 'minute' or 'day'")


def task_to_dict(task: Task) -> dict[str, Any]:
    """
    Serialize a task, keeping the keys the blocks returned before
    todoist-api-python 4.0 (``url``, ``is_completed``, ``comment_count``,
    ``sync_id``, ``due.datetime``). The API no longer returns a comment count
    or sync ID, so those are filled with ``0`` and ``None``.
    """
    data = task.to_dict()
    data["url"] = task.url
    data["is_completed"] = task.is_completed
    data["comment_count"] = 0
    data["sync_id"] = None

    due = data.get("due")
    if task.due is not None and isinstance(due, dict):
        due_date = task.due.date
        if isinstance(due_date, datetime):
            due["datetime"] = due.get("date")
            due["date"] = due_date.date().isoformat()
        else:
            due["datetime"] = None

    return data


def project_to_dict(project: Project) -> dict[str, Any]:
    """
    Serialize a project, keeping the keys the blocks returned before
    todoist-api-python 4.0. The API no longer returns a comment count or team
    inbox flag, so those are filled with ``0`` and ``None``.
    """
    data = project.to_dict()
    data["url"] = project.url
    data["comment_count"] = 0
    data["is_team_inbox"] = None
    return data
