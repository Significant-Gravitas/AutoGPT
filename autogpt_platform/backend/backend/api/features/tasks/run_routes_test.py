"""Tests for running and stopping a task from its page: request shape, user
scoping and error mapping."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from autogpt_libs.auth.models import RequestContext
from prisma.enums import TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.api.utils.exceptions import add_exception_handlers
from backend.copilot.turn_queue import InflightCapExceeded
from backend.data.task import Task
from backend.util.exceptions import ConflictError, NotFoundError

from .routes import router

app = fastapi.FastAPI()
app.include_router(router)
add_exception_handlers(app)
client = fastapi.testclient.TestClient(app)

_ROUTES = "backend.api.features.tasks.routes"
_NOW = datetime(2026, 11, 14, 9, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, test_user_id):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = lambda: RequestContext(
        user_id=test_user_id,
        org_id="org-1",
        team_id="team-1",
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=True,
        is_team_admin=True,
        is_team_billing_manager=True,
        seat_status="ACTIVE",
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def admitted(mocker: MockerFixture):
    mocker.patch(f"{_ROUTES}.enforce_payment_paywall", new_callable=AsyncMock)
    mocker.patch(f"{_ROUTES}.enforce_platform_allowance", new_callable=AsyncMock)


def _task(**overrides) -> Task:
    fields = {
        "id": "task-1",
        "user_id": "test-user-id",
        "title": "Plan the launch",
        "status": TaskStatus.IN_PROGRESS,
        "created_by": TaskCreator.USER,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return Task(**(fields | overrides))


def _mock(mocker: MockerFixture, name: str, **kwargs) -> AsyncMock:
    return mocker.patch(f"{_ROUTES}.{name}", new_callable=AsyncMock, **kwargs)


def test_run_starts_the_task_for_the_caller(mocker: MockerFixture, test_user_id):
    start = _mock(
        mocker, "start_task_run", return_value=_task(work_session_id="chat-w")
    )

    response = client.post(
        "/tasks/task-1/run", json={"instructions": "Use the Q3 numbers."}
    )

    assert response.status_code == 200
    assert response.json()["work_session_id"] == "chat-w"
    start.assert_awaited_once_with(
        test_user_id,
        "task-1",
        "Use the Q3 numbers.",
        organization_id="org-1",
        team_id="team-1",
        impersonated=False,
    )


def test_run_takes_no_body(mocker: MockerFixture, test_user_id):
    start = _mock(mocker, "start_task_run", return_value=_task())

    response = client.post("/tasks/task-1/run")

    assert response.status_code == 200
    assert start.await_args.args == (test_user_id, "task-1", None)


@pytest.mark.parametrize(
    "error,status",
    [
        (ConflictError("'Plan the launch' is already being worked on."), 409),
        (NotFoundError("Task task-1 not found"), 404),
        (InflightCapExceeded(used=15, cap=15), 429),
    ],
)
def test_run_maps_refusals(mocker: MockerFixture, error, status):
    _mock(mocker, "start_task_run", side_effect=error)

    response = client.post("/tasks/task-1/run", json={})

    assert response.status_code == status


def test_run_rejects_oversized_instructions(mocker: MockerFixture):
    start = _mock(mocker, "start_task_run", return_value=_task())

    response = client.post("/tasks/task-1/run", json={"instructions": "x" * 2001})

    assert response.status_code == 422
    start.assert_not_awaited()


def test_cancel_stops_the_callers_task(mocker: MockerFixture, test_user_id):
    cancel = _mock(
        mocker, "cancel_task_run", return_value=_task(status=TaskStatus.CANCELLED)
    )

    response = client.post("/tasks/task-1/cancel")

    assert response.status_code == 200
    assert response.json()["status"] == "CANCELLED"
    cancel.assert_awaited_once_with(test_user_id, "task-1", this_run_only=False)


def test_cancel_of_another_users_task_is_404(mocker: MockerFixture):
    _mock(mocker, "cancel_task_run", side_effect=NotFoundError("Task task-1"))

    response = client.post("/tasks/task-1/cancel")

    assert response.status_code == 404


def test_cancel_can_stop_only_the_run_going_now(mocker: MockerFixture, test_user_id):
    cancel = _mock(mocker, "cancel_task_run", return_value=_task())

    response = client.post("/tasks/task-1/cancel", json={"this_run_only": True})

    assert response.status_code == 200
    cancel.assert_awaited_once_with(test_user_id, "task-1", this_run_only=True)


@pytest.mark.parametrize(
    "method, path, store, line",
    [
        ("post", "/tasks/task-1/repeat/pause", "pause_task_repeat", "You paused it"),
        ("post", "/tasks/task-1/repeat/resume", "resume_task_repeat", "You resumed it"),
        (
            "delete",
            "/tasks/task-1/repeat",
            "stop_task_repeat",
            "You stopped it repeating",
        ),
    ],
)
def test_pause_resume_and_stop_repeating_are_the_callers_and_noted(
    mocker: MockerFixture, test_user_id, method, path, store, line
):
    change = _mock(mocker, store, return_value=_task())
    note = _mock(mocker, "add_activity")
    _mock(mocker, "get_task", return_value=_task())

    response = getattr(client, method)(path)

    assert response.status_code == 200
    change.assert_awaited_once_with("task-1", test_user_id)
    assert note.await_args.args[2].text == line


def test_pausing_a_task_that_does_not_repeat_is_409(mocker: MockerFixture):
    _mock(mocker, "pause_task_repeat", side_effect=ConflictError("It does not repeat."))

    response = client.post("/tasks/task-1/repeat/pause")

    assert response.status_code == 409
