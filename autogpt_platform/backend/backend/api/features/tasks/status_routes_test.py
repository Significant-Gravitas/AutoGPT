"""Tests for what a user may do to a task's status by hand (QA A2, A3, A7,
H5): only To do, Done or Cancelled; Done or Cancelled on work going stops the
work first; archiving a running task stops it; a missing task is 404 before
anything reads it; every change gets its own activity line in the UI's words."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.api.utils.exceptions import add_exception_handlers
from backend.data.task import Task

from .routes import router

app = fastapi.FastAPI()
app.include_router(router)
add_exception_handlers(app)
client = fastapi.testclient.TestClient(app)

_ROUTES = "backend.api.features.tasks.routes"
_NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)


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
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return Task(**(fields | overrides))


def _mock(mocker: MockerFixture, name: str, **kwargs) -> AsyncMock:
    return mocker.patch(f"{_ROUTES}.{name}", new_callable=AsyncMock, **kwargs)


@pytest.mark.parametrize("status", ["IN_PROGRESS", "NEEDS_INPUT", "FAILED"])
def test_a_status_with_no_work_behind_it_is_refused(mocker: MockerFixture, status):
    _mock(mocker, "get_task", return_value=_task())
    update = _mock(mocker, "update_task", return_value=_task())

    response = client.patch("/tasks/task-1", json={"status": status})

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "You can move a task to To do, Done or Cancelled. To work on it, start it."
    )
    update.assert_not_awaited()


@pytest.mark.parametrize("status", ["DONE", "CANCELLED"])
def test_done_or_cancelled_on_work_going_stops_the_work_first(
    mocker: MockerFixture, test_user_id, status
):
    _mock(
        mocker,
        "get_task",
        return_value=_task(status=TaskStatus.IN_PROGRESS, work_session_id="chat-w"),
    )
    settle = _mock(
        mocker, "settle_by_hand", return_value=_task(status=TaskStatus(status))
    )
    update = _mock(mocker, "update_task", return_value=_task())
    _mock(mocker, "add_activity")

    response = client.patch("/tasks/task-1", json={"status": status})

    assert response.status_code == 200
    assert response.json()["status"] == status
    settle.assert_awaited_once_with(test_user_id, "task-1", TaskStatus(status))
    update.assert_not_awaited()


def test_reopening_work_going_is_refused(mocker: MockerFixture):
    _mock(mocker, "get_task", return_value=_task(status=TaskStatus.IN_PROGRESS))
    update = _mock(mocker, "update_task", return_value=_task())

    response = client.patch("/tasks/task-1", json={"status": "TODO"})

    assert response.status_code == 409
    update.assert_not_awaited()


def test_the_owner_cannot_change_while_the_task_is_worked_on(mocker: MockerFixture):
    _mock(
        mocker,
        "get_task",
        return_value=_task(status=TaskStatus.IN_PROGRESS, owner_expert_id="exp-1"),
    )
    update = _mock(mocker, "update_task", return_value=_task())

    response = client.patch("/tasks/task-1", json={"assign_to_otto": True})

    assert response.status_code == 409
    assert "being worked on" in response.json()["detail"]
    update.assert_not_awaited()


@pytest.mark.parametrize(
    "body", [{"title": "New"}, {"start_at": None}, {"status": "DONE"}]
)
def test_a_missing_task_is_404_before_anything_reads_it(mocker: MockerFixture, body):
    _mock(mocker, "get_task", return_value=None)
    update = _mock(mocker, "update_task", return_value=_task())

    response = client.patch("/tasks/task-1", json=body)

    assert response.status_code == 404
    update.assert_not_awaited()


def test_each_change_gets_its_own_line_in_the_uis_words(
    mocker: MockerFixture, test_user_id
):
    _mock(
        mocker,
        "get_task",
        side_effect=[_task(), _task(status=TaskStatus.DONE, title="Ship it")],
    )
    _mock(
        mocker,
        "update_task",
        return_value=_task(status=TaskStatus.DONE, title="Ship it"),
    )
    note = _mock(mocker, "add_activity")

    response = client.patch(
        "/tasks/task-1", json={"status": "DONE", "title": "Ship it"}
    )

    assert response.status_code == 200
    lines = [call.args[2].text for call in note.await_args_list]
    assert lines == ["You moved it to Done", "You renamed it to Ship it"]


def test_archiving_a_running_task_stops_it_first(mocker: MockerFixture, test_user_id):
    calls: list[str] = []
    _mock(mocker, "get_task", return_value=_task(status=TaskStatus.IN_PROGRESS))
    _mock(
        mocker,
        "cancel_task_run",
        side_effect=lambda *a, **k: calls.append("stop") or _task(),
    )
    _mock(
        mocker,
        "archive_task",
        side_effect=lambda *a, **k: calls.append("archive") or True,
    )
    _mock(mocker, "clear_task_repeat")

    response = client.delete("/tasks/task-1")

    assert response.status_code == 204
    assert calls == ["stop", "archive"]


def test_archiving_a_settled_task_stops_nothing(mocker: MockerFixture):
    _mock(mocker, "get_task", return_value=_task(status=TaskStatus.DONE))
    stop = _mock(mocker, "cancel_task_run")
    _mock(mocker, "archive_task", return_value=True)
    _mock(mocker, "clear_task_repeat")

    response = client.delete("/tasks/task-1")

    assert response.status_code == 204
    stop.assert_not_awaited()
