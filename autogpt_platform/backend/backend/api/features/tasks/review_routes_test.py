"""Creating a task with a start or a rhythm (QA C1), and accepting a result
or asking for changes (QA D1)."""

from datetime import UTC, datetime, timedelta
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
from backend.data.task import Task
from backend.util.exceptions import ConflictError

from .routes import router

app = fastapi.FastAPI()
app.include_router(router)
add_exception_handlers(app)
client = fastapi.testclient.TestClient(app)

_ROUTES = "backend.api.features.tasks.routes"
_CREATION = "backend.api.features.tasks.creation"
_NOW = datetime(2026, 11, 19, 12, 0, tzinfo=UTC)


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
    mocker.patch(
        f"{_CREATION}.user_timezone",
        new_callable=AsyncMock,
        return_value=("Europe/London", True),
    )


def _task(**overrides) -> Task:
    fields = {
        "id": "task-1",
        "user_id": "test-user-id",
        "title": "Brief me",
        "status": TaskStatus.TODO,
        "created_by": TaskCreator.USER,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return Task(**(fields | overrides))


def _mock(mocker: MockerFixture, where: str, name: str, **kwargs) -> AsyncMock:
    return mocker.patch(f"{where}.{name}", new_callable=AsyncMock, **kwargs)


def test_a_task_created_with_a_rhythm_repeats(mocker: MockerFixture, test_user_id):
    _mock(mocker, _ROUTES, "create_task", return_value=_task())
    schedule = _mock(mocker, _CREATION, "set_task_start", return_value=_task())
    note = _mock(mocker, _CREATION, "add_activity")

    response = client.post(
        "/tasks", json={"title": "Brief me", "repeat": {"crons": ["0 8 * * *"]}}
    )

    assert response.status_code == 201
    task_id, user_id, at, origin = schedule.await_args.args
    repeat = schedule.await_args.kwargs["repeat"]
    assert (task_id, user_id, origin) == ("task-1", test_user_id, None)
    assert at > datetime.now(UTC)
    assert repeat.crons == ["0 8 * * *"] and repeat.timezone == "Europe/London"
    assert repeat.label == "Every day at 08:00"
    assert note.await_args.args[2].text.startswith("Repeats: Every day at 08:00.")


def test_a_task_created_with_a_start_time_waits_for_it(mocker: MockerFixture):
    _mock(mocker, _ROUTES, "create_task", return_value=_task())
    schedule = _mock(mocker, _CREATION, "set_task_start", return_value=_task())
    _mock(mocker, _CREATION, "add_activity")
    start = datetime.now(UTC) + timedelta(hours=2)

    response = client.post(
        "/tasks", json={"title": "Brief me", "start_at": start.isoformat()}
    )

    assert response.status_code == 201
    assert schedule.await_args.args[2] == start
    assert schedule.await_args.kwargs["repeat"] is None


@pytest.mark.parametrize(
    "extra, detail",
    [
        ({"repeat": {"crons": ["* * * * * *"]}}, "5 fields"),
        ({"repeat": {"crons": ["0 8 * * *"], "timezone": "Mars/Base"}}, "time zone"),
        (
            {"repeat": {"crons": ["0 8 * * *"]}, "start_at": "2099-01-01T09:00:00Z"},
            "not both",
        ),
        ({"start_at": "2020-01-01T09:00:00Z"}, "in the future"),
    ],
)
def test_a_start_or_rhythm_that_cannot_be_used_creates_nothing(
    mocker: MockerFixture, extra, detail
):
    create = _mock(mocker, _ROUTES, "create_task", return_value=_task())

    response = client.post("/tasks", json={"title": "Brief me", **extra})

    assert response.status_code == 422
    assert detail in response.json()["detail"]
    create.assert_not_awaited()


def test_accepting_a_result_records_it(mocker: MockerFixture, test_user_id):
    review = _mock(
        mocker, _ROUTES, "set_task_review", return_value=_task(status=TaskStatus.DONE)
    )
    note = _mock(mocker, _ROUTES, "add_activity")
    run = _mock(mocker, _ROUTES, "start_task_run")

    response = client.post("/tasks/task-1/review", json={"decision": "accept"})

    assert response.status_code == 200
    review.assert_awaited_once_with("task-1", test_user_id, "accepted")
    assert note.await_args.args[2].text == "You accepted the result"
    run.assert_not_awaited()


def test_asking_for_changes_runs_the_task_again_with_the_note(
    mocker: MockerFixture, test_user_id
):
    review = _mock(mocker, _ROUTES, "set_task_review", return_value=_task())
    _mock(mocker, _ROUTES, "add_activity")
    run = _mock(
        mocker,
        _ROUTES,
        "start_task_run",
        return_value=_task(status=TaskStatus.IN_PROGRESS),
    )

    response = client.post(
        "/tasks/task-1/review",
        json={"decision": "request_changes", "note": "Use the Q3 numbers."},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "IN_PROGRESS"
    review.assert_awaited_once_with(
        "task-1", test_user_id, "changes_requested", "Use the Q3 numbers."
    )
    assert run.await_args.args == (test_user_id, "task-1", "Use the Q3 numbers.")


def test_asking_for_changes_needs_a_note(mocker: MockerFixture):
    review = _mock(mocker, _ROUTES, "set_task_review", return_value=_task())

    response = client.post(
        "/tasks/task-1/review", json={"decision": "request_changes", "note": " "}
    )

    assert response.status_code == 422
    review.assert_not_awaited()


def test_a_result_that_cannot_be_reviewed_now_is_409(mocker: MockerFixture):
    _mock(
        mocker,
        _ROUTES,
        "set_task_review",
        side_effect=ConflictError("Only a finished task's result can be accepted."),
    )

    response = client.post("/tasks/task-1/review", json={"decision": "accept"})

    assert response.status_code == 409
