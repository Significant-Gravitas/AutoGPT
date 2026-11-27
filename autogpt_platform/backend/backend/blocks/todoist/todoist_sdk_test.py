import json
from datetime import date
from unittest.mock import patch

import httpx
import pytest
from todoist_api_python.api import TodoistAPI

import backend.blocks.todoist._api as todoist_api
from backend.blocks.todoist._auth import TEST_CREDENTIALS
from backend.blocks.todoist.comments import (
    TodoistCreateCommentBlock,
    TodoistDeleteCommentBlock,
    TodoistGetCommentBlock,
    TodoistGetCommentsBlock,
    TodoistUpdateCommentBlock,
)
from backend.blocks.todoist.labels import (
    TodoistCreateLabelBlock,
    TodoistDeleteLabelBlock,
    TodoistGetLabelBlock,
    TodoistGetSharedLabelsBlock,
    TodoistListLabelsBlock,
    TodoistRemoveSharedLabelsBlock,
    TodoistRenameSharedLabelsBlock,
    TodoistUpdateLabelBlock,
)
from backend.blocks.todoist.projects import (
    TodoistCreateProjectBlock,
    TodoistDeleteProjectBlock,
    TodoistGetProjectBlock,
    TodoistListCollaboratorsBlock,
    TodoistListProjectsBlock,
    TodoistUpdateProjectBlock,
)
from backend.blocks.todoist.sections import (
    TodoistDeleteSectionBlock,
    TodoistGetSectionBlock,
    TodoistListSectionsBlock,
)
from backend.blocks.todoist.tasks import (
    TodoistCloseTaskBlock,
    TodoistCreateTaskBlock,
    TodoistDeleteTaskBlock,
    TodoistGetTaskBlock,
    TodoistGetTasksBlock,
    TodoistReopenTaskBlock,
    TodoistUpdateTaskBlock,
)

CREDS = TEST_CREDENTIALS
TS = "2026-10-01T10:00:00.000000Z"


def _task(task_id: str) -> dict:
    return {
        "id": task_id,
        "content": f"task {task_id}",
        "description": "",
        "project_id": "P",
        "section_id": None,
        "parent_id": None,
        "labels": [],
        "priority": 1,
        "due": None,
        "deadline": None,
        "duration": None,
        "collapsed": False,
        "child_order": 1,
        "responsible_uid": None,
        "assigned_by_uid": None,
        "completed_at": None,
        "added_by_uid": "u",
        "added_at": TS,
        "updated_at": TS,
    }


def _project(project_id: str) -> dict:
    return {
        "id": project_id,
        "name": f"project {project_id}",
        "description": "",
        "child_order": 0,
        "color": "charcoal",
        "is_collapsed": False,
        "is_shared": False,
        "is_favorite": False,
        "is_archived": False,
        "can_assign_tasks": False,
        "view_style": "list",
        "created_at": TS,
        "updated_at": TS,
    }


def _section(section_id: str) -> dict:
    return {
        "id": section_id,
        "name": "section",
        "project_id": "P",
        "is_collapsed": False,
        "section_order": 1,
    }


def _label(label_id: str) -> dict:
    return {
        "id": label_id,
        "name": f"label {label_id}",
        "color": "red",
        "order": 1,
        "is_favorite": False,
    }


def _comment(comment_id: str) -> dict:
    return {
        "id": comment_id,
        "item_id": "T",
        "content": "hello",
        "posted_uid": "u",
        "posted_at": TS,
        "file_attachment": None,
    }


def _collaborator(collaborator_id: str) -> dict:
    return {"id": collaborator_id, "name": "Name", "email": "e@example.com"}


def _paged(make, request: httpx.Request) -> dict:
    """Two pages: two items with a cursor, then one more and no cursor."""
    if request.url.params.get("cursor") is None:
        return {"results": [make("a"), make("b")], "next_cursor": "c1"}
    return {"results": [make("c")], "next_cursor": None}


def _respond(request: httpx.Request) -> httpx.Response:
    path = request.url.path.removeprefix("/api/v1/")
    get = request.method == "GET"
    if request.method == "DELETE":
        return httpx.Response(204)
    if path in ("tasks", "tasks/filter") and get:
        return httpx.Response(200, json=_paged(_task, request))
    if path.startswith("tasks/") and path.endswith(("/close", "/reopen")):
        return httpx.Response(204)
    if path.startswith("tasks"):
        return httpx.Response(200, json=_task("T"))
    if path.endswith("/collaborators"):
        return httpx.Response(200, json=_paged(_collaborator, request))
    if path == "projects" and get:
        return httpx.Response(200, json=_paged(_project, request))
    if path.startswith("projects"):
        return httpx.Response(200, json=_project("P"))
    if path == "sections" and get:
        return httpx.Response(200, json=_paged(_section, request))
    if path.startswith("sections"):
        return httpx.Response(200, json=_section("S"))
    if path == "labels/shared" and get:
        return httpx.Response(200, json=_paged(lambda i: f"shared {i}", request))
    if path.startswith("labels/shared"):
        return httpx.Response(204)
    if path == "labels" and get:
        return httpx.Response(200, json=_paged(_label, request))
    if path.startswith("labels"):
        return httpx.Response(200, json=_label("L"))
    if path == "comments" and get:
        return httpx.Response(200, json=_paged(_comment, request))
    if path.startswith("comments"):
        return httpx.Response(200, json=_comment("C"))
    return httpx.Response(404, json={"path": path})


@pytest.fixture
def requests_sent():
    """Run the blocks against the real SDK with a fake Todoist API v1."""
    sent: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(
            {
                "method": request.method,
                "path": request.url.path,
                "params": dict(request.url.params),
                "body": json.loads(request.content) if request.content else None,
            }
        )
        return _respond(request)

    def make_api(token: str) -> TodoistAPI:
        return TodoistAPI(
            token, client=httpx.Client(transport=httpx.MockTransport(handler))
        )

    with patch.object(todoist_api, "TodoistAPI", make_api):
        yield sent


def _paths(sent: list[dict]) -> list[str]:
    return [request["path"] for request in sent]


def test_get_tasks_reads_every_page(requests_sent):
    tasks = TodoistGetTasksBlock.get_tasks(CREDS, project_id="P")

    assert [task["id"] for task in tasks] == ["a", "b", "c"]
    assert requests_sent[0]["params"]["project_id"] == "P"
    assert requests_sent[0]["params"]["limit"] == str(todoist_api.PAGE_SIZE)
    assert requests_sent[1]["params"]["cursor"] == "c1"


def test_get_tasks_ids_take_precedence_over_project(requests_sent):
    TodoistGetTasksBlock.get_tasks(CREDS, project_id="P", ids=["a", "b"])

    params = requests_sent[0]["params"]
    assert "a" in params["ids"] and "b" in params["ids"]
    assert "project_id" not in params


def test_get_tasks_filter_uses_the_filter_endpoint(requests_sent):
    tasks = TodoistGetTasksBlock.get_tasks(
        CREDS, filter="today", lang="en", project_id="P"
    )

    assert len(tasks) == 3
    assert requests_sent[0]["path"].endswith("/tasks/filter")
    assert requests_sent[0]["params"]["query"] == "today"
    assert requests_sent[0]["params"]["lang"] == "en"


def test_create_task_sends_due_date_and_duration(requests_sent):
    task_id, url, data = TodoistCreateTaskBlock.create_task(
        CREDS,
        "Buy milk",
        project_id="P",
        due_date=date(2026, 10, 5),
        duration=15,
        duration_unit="minute",
    )

    assert task_id == "T"
    assert url == data["url"]
    body = requests_sent[0]["body"]
    assert body["due_date"] == "2026-10-05"
    assert body["duration"] == 15
    assert body["duration_unit"] == "minute"


def test_update_task_does_not_move_the_task(requests_sent):
    assert (
        TodoistUpdateTaskBlock.update_task(
            CREDS, "T", content="new", project_id="P2", section_id="S", order=3
        )
        is True
    )

    body = requests_sent[0]["body"]
    assert body["content"] == "new"
    assert "project_id" not in body
    assert "section_id" not in body


def test_task_lifecycle_calls(requests_sent):
    assert TodoistGetTaskBlock.get_task(CREDS, "T")["id"] == "T"
    TodoistCloseTaskBlock.close_task(CREDS, "T")
    TodoistReopenTaskBlock.reopen_task(CREDS, "T")
    TodoistDeleteTaskBlock.delete_task(CREDS, "T")

    paths = _paths(requests_sent)
    assert paths[1].endswith("/tasks/T/close")
    assert paths[2].endswith("/tasks/T/reopen")
    assert requests_sent[3]["method"] == "DELETE"


def test_project_blocks(requests_sent):
    names, ids, urls, complete_data, error = TodoistListProjectsBlock.get_project_lists(
        CREDS
    )
    assert ids == ["a", "b", "c"]
    assert len(names) == len(urls) == len(complete_data) == 3
    assert error is None

    collaborator_ids, _, emails, _ = TodoistListCollaboratorsBlock.get_collaborators(
        CREDS, "P"
    )
    assert collaborator_ids == ["a", "b", "c"]
    assert emails[0] == "e@example.com"

    assert (
        TodoistCreateProjectBlock.create_project(CREDS, "new", None, None, False, None)
        is True
    )
    assert TodoistGetProjectBlock.get_project(CREDS, "P")[0] == "P"
    assert (
        TodoistUpdateProjectBlock.update_project(
            CREDS, "P", "renamed", None, None, None
        )
        is True
    )
    TodoistDeleteProjectBlock.delete_project(CREDS, "P")
    assert requests_sent[-1]["method"] == "DELETE"


def test_section_blocks(requests_sent):
    names, ids, complete_data = TodoistListSectionsBlock.get_section_lists(CREDS, "P")
    assert ids == ["a", "b", "c"]
    assert len(names) == len(complete_data) == 3

    assert TodoistGetSectionBlock.get_section(CREDS, "S")["id"] == "S"
    TodoistDeleteSectionBlock.delete_section(CREDS, "S")
    assert requests_sent[-1]["method"] == "DELETE"


def test_label_blocks(requests_sent):
    assert [label["id"] for label in TodoistListLabelsBlock.get_labels(CREDS)] == [
        "a",
        "b",
        "c",
    ]

    TodoistCreateLabelBlock.create_label(
        CREDS, "new", order=2, color="red", is_favorite=False
    )
    assert requests_sent[-1]["body"]["item_order"] == 2

    assert TodoistGetLabelBlock.get_label(CREDS, "L")["id"] == "L"
    assert TodoistUpdateLabelBlock.update_label(CREDS, "L", name="m", order=4) is True
    assert requests_sent[-1]["body"]["item_order"] == 4
    TodoistDeleteLabelBlock.delete_label(CREDS, "L")
    assert requests_sent[-1]["method"] == "DELETE"


def test_shared_label_blocks(requests_sent):
    assert len(TodoistGetSharedLabelsBlock.get_shared_labels(CREDS)) == 3

    TodoistRenameSharedLabelsBlock.rename_shared_labels(CREDS, "old", "new")
    TodoistRemoveSharedLabelsBlock.remove_shared_label(CREDS, "new")

    rename, remove = requests_sent[-2], requests_sent[-1]
    assert rename["path"].endswith("/labels/shared/rename")
    assert rename["params"]["name"] == "old"
    assert rename["body"]["new_name"] == "new"
    assert remove["path"].endswith("/labels/shared/remove")
    assert remove["body"]["name"] == "new"


def test_comment_blocks(requests_sent):
    comments = TodoistGetCommentsBlock.get_comments(CREDS, task_id="T")
    assert [comment["id"] for comment in comments] == ["a", "b", "c"]

    attachment = {
        "file_url": "https://example.com/a.pdf",
        "file_name": "a.pdf",
        "file_type": "application/pdf",
        "resource_type": "file",
    }
    created = TodoistCreateCommentBlock.create_comment(
        CREDS, "hello", task_id="T", attachment=attachment
    )
    assert created["id"] == "C"
    json.dumps(created)
    body = requests_sent[-1]["body"]
    assert any(
        isinstance(value, dict) and value.get("file_url") == attachment["file_url"]
        for value in body.values()
    )

    assert TodoistGetCommentBlock.get_comment(CREDS, "C")["id"] == "C"
    assert TodoistUpdateCommentBlock.update_comment(CREDS, "C", "edited") is True
    TodoistDeleteCommentBlock.delete_comment(CREDS, "C")
    assert requests_sent[-1]["method"] == "DELETE"


def test_create_comment_rejects_an_attachment_without_a_url(requests_sent):
    with pytest.raises(ValueError, match="file_url or url"):
        TodoistCreateCommentBlock.create_comment(
            CREDS, "hello", task_id="T", attachment={"file_name": "a.pdf"}
        )

    assert requests_sent == []
