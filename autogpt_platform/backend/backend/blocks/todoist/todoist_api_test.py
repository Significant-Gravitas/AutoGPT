import json

import pytest
from todoist_api_python.models import Project, Task

from backend.blocks.todoist._api import (
    flatten_pages,
    parse_duration_unit,
    project_to_dict,
    task_to_dict,
)


def _task_payload(due: dict | None) -> dict:
    return {
        "id": "6X7rM8997g3RQmvh",
        "content": "Buy groceries",
        "description": "",
        "project_id": "6Jf8VQXxpwv56VQ7",
        "section_id": None,
        "parent_id": None,
        "labels": ["shopping"],
        "priority": 4,
        "due": due,
        "deadline": None,
        "duration": None,
        "collapsed": False,
        "child_order": 1,
        "responsible_uid": None,
        "assigned_by_uid": None,
        "completed_at": None,
        "added_by_uid": "2671355",
        "added_at": "2026-10-01T10:00:00.000000Z",
        "updated_at": "2026-10-02T11:30:00.000000Z",
    }


def test_flatten_pages_collects_every_page():
    assert flatten_pages(iter([[1, 2], [], [3]])) == [1, 2, 3]


def test_parse_duration_unit():
    assert parse_duration_unit(None) is None
    assert parse_duration_unit("minute") == "minute"
    assert parse_duration_unit("day") == "day"
    with pytest.raises(ValueError):
        parse_duration_unit("hour")


def test_task_to_dict_keeps_legacy_keys():
    task = Task.from_dict(
        _task_payload(
            {
                "date": "2026-10-05",
                "string": "Oct 5",
                "lang": "en",
                "is_recurring": False,
            }
        )
    )

    data = task_to_dict(task)

    assert data["id"] == "6X7rM8997g3RQmvh"
    assert data["project_id"] == "6Jf8VQXxpwv56VQ7"
    assert data["order"] == 1
    assert data["creator_id"] == "2671355"
    assert data["url"] == task.url
    assert data["is_completed"] is False
    assert data["comment_count"] == 0
    assert data["sync_id"] is None
    assert data["due"]["date"] == "2026-10-05"
    assert data["due"]["datetime"] is None
    json.dumps(data)


def test_task_to_dict_splits_due_datetime():
    task = Task.from_dict(
        _task_payload(
            {
                "date": "2026-10-05T15:00:00Z",
                "string": "Oct 5 3pm",
                "lang": "en",
                "is_recurring": False,
                "timezone": "Europe/Warsaw",
            }
        )
    )

    data = task_to_dict(task)

    assert data["due"]["date"] == "2026-10-05"
    assert data["due"]["datetime"].startswith("2026-10-05T15:00:00")
    json.dumps(data)


def test_project_to_dict_keeps_legacy_keys():
    project = Project.from_dict(
        {
            "id": "6Jf8VQXxpwv56VQ7",
            "name": "Shopping List",
            "description": "",
            "child_order": 1,
            "color": "charcoal",
            "is_collapsed": False,
            "is_shared": False,
            "is_favorite": False,
            "is_archived": False,
            "can_assign_tasks": False,
            "view_style": "list",
            "created_at": "2026-10-01T10:00:00.000000Z",
            "updated_at": "2026-10-02T11:30:00.000000Z",
        }
    )

    data = project_to_dict(project)

    assert data["id"] == "6Jf8VQXxpwv56VQ7"
    assert data["name"] == "Shopping List"
    assert data["order"] == 1
    assert data["url"] == project.url
    assert data["comment_count"] == 0
    assert data["is_team_inbox"] is None
    json.dumps(data)
