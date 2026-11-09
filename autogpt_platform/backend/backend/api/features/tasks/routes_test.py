"""Tests for the task routes: request shape, user scoping and error mapping."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.api.utils.exceptions import add_exception_handlers
from backend.data.task import Task, TaskPatch
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
    )


@pytest.mark.parametrize("body", [{}, {"title": ""}, {"title": "x" * 201}])
def test_create_rejects_a_missing_or_oversized_title(mocker: MockerFixture, body):
    create = _mock(mocker, "create_task", _task())

    assert client.post("/tasks", json=body).status_code == 422
    create.assert_not_awaited()


def test_list_passes_filters_for_the_caller(mocker: MockerFixture, test_user_id):
    list_tasks = _mock(mocker, "list_tasks", [_task()])

    response = client.get(
        "/tasks",
        params={"status": "IN_PROGRESS", "session_id": "chat-1", "limit": 10},
    )

    assert response.status_code == 200
    assert [t["id"] for t in response.json()["tasks"]] == ["task-1"]
    list_tasks.assert_awaited_once_with(
        user_id=test_user_id,
        status=TaskStatus.IN_PROGRESS,
        owner_expert_id=None,
        session_id="chat-1",
        limit=10,
        offset=0,
    )


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}])
def test_list_rejects_out_of_range_paging(mocker: MockerFixture, params):
    _mock(mocker, "list_tasks", [])
    assert client.get("/tasks", params=params).status_code == 422


def test_get_returns_404_for_a_task_the_caller_cannot_see(
    mocker: MockerFixture, test_user_id
):
    get = _mock(mocker, "get_task", None)

    assert client.get("/tasks/task-9").status_code == 404
    get.assert_awaited_once_with("task-9", test_user_id)


def test_update_sends_only_what_the_body_sets(mocker: MockerFixture, test_user_id):
    update = _mock(mocker, "update_task", _task(status=TaskStatus.DONE))

    response = client.patch("/tasks/task-1", json={"status": "DONE"})

    assert response.status_code == 200
    assert response.json()["status"] == "DONE"
    update.assert_awaited_once_with(
        "task-1", test_user_id, TaskPatch(status=TaskStatus.DONE)
    )


@pytest.mark.parametrize(
    "error, status", [(NotFoundError("gone"), 404), (ValueError("bad"), 400)]
)
def test_update_maps_data_layer_errors(mocker: MockerFixture, error, status):
    mocker.patch(f"{_ROUTES}.update_task", new_callable=AsyncMock, side_effect=error)

    assert client.patch("/tasks/task-1", json={"title": "x"}).status_code == status


@pytest.mark.parametrize("archived, status", [(True, 204), (False, 404)])
def test_archive(mocker: MockerFixture, test_user_id, archived, status):
    archive = _mock(mocker, "archive_task", archived)

    assert client.delete("/tasks/task-1").status_code == status
    archive.assert_awaited_once_with("task-1", test_user_id)


def test_attach_and_detach_a_chat(mocker: MockerFixture, test_user_id):
    attach = _mock(mocker, "attach_task_session", _task(session_ids=["c1", "c2"]))
    detach = _mock(mocker, "detach_task_session", _task(session_ids=["c1"]))

    attached = client.post("/tasks/task-1/sessions/c2")
    detached = client.delete("/tasks/task-1/sessions/c2")

    assert attached.json()["session_ids"] == ["c1", "c2"]
    assert detached.json()["session_ids"] == ["c1"]
    attach.assert_awaited_once_with("task-1", "c2", test_user_id)
    detach.assert_awaited_once_with("task-1", "c2", test_user_id)
