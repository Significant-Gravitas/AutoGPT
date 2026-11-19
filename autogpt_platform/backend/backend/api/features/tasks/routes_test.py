"""Tests for the task routes: request shape, user scoping and error mapping."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import TaskActivityKind, TaskActor, TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.api.utils.exceptions import add_exception_handlers
from backend.data.task import Task, TaskPatch
from backend.data.task_activity_meta import TaskActivityAccount
from backend.data.task_list import TaskFilter, TaskPage
from backend.data.task_record import TaskActivity, TaskRecord
from backend.util.exceptions import NotFoundError

from .routes import router

app = fastapi.FastAPI()
app.include_router(router)
add_exception_handlers(app)
client = fastapi.testclient.TestClient(app)

_ROUTES = "backend.api.features.tasks.routes"
_NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _task(**overrides) -> Task:
    fields = {
        "id": "task-1",
        "user_id": "test-user-id",
        "title": "Build feature X",
        "status": TaskStatus.TODO,
        "created_by": TaskCreator.USER,
        "session_ids": ["chat-1"],
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return Task(**(fields | overrides))


def _mock(mocker: MockerFixture, name: str, result) -> AsyncMock:
    return mocker.patch(
        f"{_ROUTES}.{name}", new_callable=AsyncMock, return_value=result
    )


def test_create_records_the_caller_as_creator(mocker: MockerFixture, test_user_id):
    create = _mock(mocker, "create_task", _task())

    response = client.post(
        "/tasks", json={"title": "Build feature X", "session_id": "chat-1"}
    )

    assert response.status_code == 201
    assert response.json()["id"] == "task-1"
    create.assert_awaited_once_with(
        user_id=test_user_id,
        title="Build feature X",
        description=None,
        owner_expert_id=None,
        session_id="chat-1",
        created_by=TaskCreator.USER,
        goal=None,
        done_when=None,
    )


@pytest.mark.parametrize("body", [{}, {"title": ""}, {"title": "x" * 201}])
def test_create_rejects_a_missing_or_oversized_title(mocker: MockerFixture, body):
    create = _mock(mocker, "create_task", _task())

    assert client.post("/tasks", json=body).status_code == 422
    create.assert_not_awaited()


def test_list_passes_filters_for_the_caller(mocker: MockerFixture, test_user_id):
    page = _mock(mocker, "list_task_page", TaskPage(tasks=[_task()]))

    response = client.get(
        "/tasks",
        params={"status": "IN_PROGRESS", "session_id": "chat-1", "limit": 10},
    )

    assert response.status_code == 200
    assert [t["id"] for t in response.json()["tasks"]] == ["task-1"]
    page.assert_awaited_once_with(
        test_user_id,
        TaskFilter(statuses=[TaskStatus.IN_PROGRESS], session_id="chat-1"),
        10,
        0,
    )


def test_the_list_pages_searches_and_filters_by_owner_and_lane(
    mocker: MockerFixture, test_user_id
):
    """QA B5: past the first 100, a title search, Otto's tasks, a lane."""
    page = _mock(
        mocker,
        "list_task_page",
        TaskPage(tasks=[_task()], has_more=True, next_offset=200),
    )

    response = client.get(
        "/tasks",
        params=[
            ("q", "launch plan"),
            ("owned_by_otto", "true"),
            ("statuses", "DONE"),
            ("statuses", "FAILED"),
            ("limit", "100"),
            ("offset", "100"),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert (body["has_more"], body["next_offset"]) == (True, 200)
    page.assert_awaited_once_with(
        test_user_id,
        TaskFilter(
            statuses=[TaskStatus.DONE, TaskStatus.FAILED],
            owned_by_otto=True,
            search="launch plan",
        ),
        100,
        100,
    )


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}])
def test_list_rejects_out_of_range_paging(mocker: MockerFixture, params):
    _mock(mocker, "list_task_page", TaskPage(tasks=[]))
    assert client.get("/tasks", params=params).status_code == 422


def test_get_returns_404_for_a_task_the_caller_cannot_see(
    mocker: MockerFixture, test_user_id
):
    get = _mock(mocker, "get_task", None)

    assert client.get("/tasks/task-9").status_code == 404
    get.assert_awaited_once_with("task-9", test_user_id)


def test_update_sends_only_what_the_body_sets(mocker: MockerFixture, test_user_id):
    _mock(mocker, "get_task", _task())
    update = _mock(mocker, "update_task", _task(status=TaskStatus.DONE))
    noted = _mock(mocker, "add_activity", None)

    response = client.patch("/tasks/task-1", json={"status": "DONE"})

    assert response.status_code == 200
    assert response.json()["status"] == "DONE"
    update.assert_awaited_once_with(
        "task-1", test_user_id, TaskPatch(status=TaskStatus.DONE)
    )
    # What the user changed lands on the task's activity, as theirs.
    entry = noted.await_args.args[2]
    assert (entry.text, entry.actor_type) == ("You moved it to Done", TaskActor.USER)


@pytest.mark.parametrize(
    "body, after, line",
    [
        (
            {"goal": "Ship by Friday"},
            {"goal": "Ship by Friday"},
            "You changed the goal",
        ),
        ({"title": "New name"}, {"title": "New name"}, "You renamed it to New name"),
        ({"description": "same task"}, {}, None),
    ],
)
def test_an_edit_is_noted_only_when_it_changes_something_the_user_sees(
    mocker: MockerFixture, body, after, line
):
    _mock(mocker, "get_task", _task())
    _mock(mocker, "update_task", _task(**after))
    noted = _mock(mocker, "add_activity", None)

    assert client.patch("/tasks/task-1", json=body).status_code == 200
    if line is None:
        noted.assert_not_awaited()
    else:
        assert noted.await_args.args[2].text == line


def test_the_record_is_returned_whole_and_404s_for_a_stranger(
    mocker: MockerFixture, test_user_id
):
    record = TaskRecord(task=_task(), so_far=["Task started"], is_unverified=False)
    get = _mock(mocker, "get_task_record", record)

    response = client.get("/tasks/task-1/record")

    assert response.status_code == 200
    assert response.json()["task"]["id"] == "task-1"
    assert response.json()["so_far"] == ["Task started"]
    get.assert_awaited_once_with("task-1", test_user_id)

    _mock(mocker, "get_task_record", None)
    assert client.get("/tasks/task-9/record").status_code == 404


def test_each_activity_line_says_which_account_and_skill_it_used(
    mocker: MockerFixture,
):
    def line(id: str, **meta) -> TaskActivity:
        return TaskActivity(
            id=id,
            task_id="task-1",
            created_at=_NOW,
            actor_type=TaskActor.EXPERT,
            kind=TaskActivityKind.TOOL,
            text="Used Gmail to send email",
            **meta,
        )

    gmail = TaskActivityAccount(service="google", name="Gmail", label="work@acme.com")
    record = TaskRecord(
        task=_task(),
        activity=[line("a-1", account=gmail, skill="Cold email"), line("a-2")],
    )
    _mock(mocker, "get_task_record", record)

    sent, plain = client.get("/tasks/task-1/record").json()["activity"]

    assert sent["account"] == {
        "service": "google",
        "name": "Gmail",
        "label": "work@acme.com",
    }
    assert sent["skill"] == "Cold email"
    assert plain["account"] is None and plain["skill"] is None


def test_subtasks_are_listed_only_when_asked_for(mocker: MockerFixture):
    page = _mock(mocker, "list_task_page", TaskPage(tasks=[]))

    client.get("/tasks", params={"include_subtasks": "true"})

    assert page.await_args.args[1].include_subtasks is True


@pytest.mark.parametrize(
    "error, status", [(NotFoundError("gone"), 404), (ValueError("bad"), 400)]
)
def test_update_maps_data_layer_errors(mocker: MockerFixture, error, status):
    _mock(mocker, "get_task", _task())
    mocker.patch(f"{_ROUTES}.update_task", new_callable=AsyncMock, side_effect=error)

    assert client.patch("/tasks/task-1", json={"title": "x"}).status_code == status


@pytest.mark.parametrize("archived, status", [(True, 204), (False, 404)])
def test_archive(mocker: MockerFixture, test_user_id, archived, status):
    _mock(mocker, "get_task", _task())
    archive = _mock(mocker, "archive_task", archived)
    clear = _mock(mocker, "clear_task_repeat", False)

    assert client.delete("/tasks/task-1").status_code == status
    archive.assert_awaited_once_with("task-1", test_user_id)
    # An archived task stops repeating or watching, its webhook let go of.
    assert clear.await_count == (1 if archived else 0)


def test_attach_and_detach_a_chat(mocker: MockerFixture, test_user_id):
    attach = _mock(mocker, "attach_task_session", _task(session_ids=["c1", "c2"]))
    detach = _mock(mocker, "detach_task_session", _task(session_ids=["c1"]))

    attached = client.post("/tasks/task-1/sessions/c2")
    detached = client.delete("/tasks/task-1/sessions/c2")

    assert attached.json()["session_ids"] == ["c1", "c2"]
    assert detached.json()["session_ids"] == ["c1"]
    attach.assert_awaited_once_with("task-1", "c2", test_user_id)
    detach.assert_awaited_once_with("task-1", "c2", test_user_id)


# ---------------------------------------------------------------- start time

_LATER = datetime(2099, 1, 2, 9, 0, tzinfo=UTC)


def _start_time(mocker: MockerFixture, timezone: str = "Europe/London") -> None:
    mocker.patch(
        "backend.api.features.tasks.start_time.user_timezone",
        new_callable=AsyncMock,
        return_value=(timezone, True),
    )


def test_a_future_start_time_schedules_a_todo_task(mocker: MockerFixture, test_user_id):
    _start_time(mocker)
    _mock(mocker, "get_task", _task())
    update = _mock(mocker, "update_task", _task())
    schedule = _mock(mocker, "set_task_start", _task(start_at=_LATER))
    noted = _mock(mocker, "add_activity", None)

    response = client.patch("/tasks/task-1", json={"start_at": "2099-01-02T09:00:00Z"})

    assert response.status_code == 200
    assert response.json()["start_at"].startswith("2099-01-02T09:00:00")
    assert response.json()["missed_start"] is False
    update.assert_awaited_once_with("task-1", test_user_id, TaskPatch())
    schedule.assert_awaited_once_with("task-1", test_user_id, _LATER)
    assert noted.await_args.args[2].text == (
        "You set it to start Fri 2 Jan 2099, 09:00 Europe/London"
    )


def test_a_null_start_time_unschedules_the_task(mocker: MockerFixture, test_user_id):
    _mock(mocker, "get_task", _task(start_at=_LATER))
    _mock(mocker, "update_task", _task(start_at=_LATER))
    schedule = _mock(mocker, "set_task_start", _task())
    noted = _mock(mocker, "add_activity", None)

    response = client.patch("/tasks/task-1", json={"start_at": None})

    assert response.status_code == 200
    schedule.assert_awaited_once_with("task-1", test_user_id, None)
    assert noted.await_args.args[2].text == "You removed the start time"


def test_an_edit_without_a_start_time_leaves_it_alone(mocker: MockerFixture):
    _mock(mocker, "get_task", _task(start_at=_LATER))
    _mock(mocker, "update_task", _task(start_at=_LATER, title="New"))
    schedule = _mock(mocker, "set_task_start", _task())
    _mock(mocker, "add_activity", None)

    assert client.patch("/tasks/task-1", json={"title": "New"}).status_code == 200
    schedule.assert_not_awaited()


@pytest.mark.parametrize(
    "task, body, status",
    [
        (
            _task(status=TaskStatus.IN_PROGRESS),
            {"start_at": "2099-01-02T09:00:00Z"},
            409,
        ),
        (_task(), {"start_at": "2099-01-02T09:00:00Z", "status": "DONE"}, 409),
        (_task(), {"start_at": "2020-01-02T09:00:00Z"}, 422),
        (None, {"start_at": "2099-01-02T09:00:00Z"}, 404),
    ],
)
def test_a_start_time_is_refused_before_anything_is_written(
    mocker: MockerFixture, task, body, status
):
    _mock(mocker, "get_task", task)
    update = _mock(mocker, "update_task", _task())
    schedule = _mock(mocker, "set_task_start", _task())

    assert client.patch("/tasks/task-1", json=body).status_code == status
    update.assert_not_awaited()
    schedule.assert_not_awaited()


def test_scheduled_tasks_can_be_listed_on_their_own(mocker: MockerFixture):
    page = _mock(mocker, "list_task_page", TaskPage(tasks=[_task(start_at=_LATER)]))

    response = client.get("/tasks", params={"scheduled": "true"})

    assert response.json()["tasks"][0]["start_at"].startswith("2099-01-02")
    assert page.await_args.args[1].scheduled is True
