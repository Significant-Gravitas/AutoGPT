"""QA pass 3 on the task routes: sorting (M3), undoing an archive (M5), a
title of spaces and unknown fields (P14), the one answer to a second start
(P4), and 404s without ids (P5/U8)."""

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
from backend.data.task import TASK_GONE, Task
from backend.data.task_list import TaskPage
from backend.data.task_run import ALREADY_STARTED
from backend.util.exceptions import ConflictError, NotFoundError

from .routes import router

app = fastapi.FastAPI()
app.include_router(router)
add_exception_handlers(app)
client = fastapi.testclient.TestClient(app)

_ROUTES = "backend.api.features.tasks.routes"
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
        f"{_ROUTES}.start_scheduled_now", new_callable=AsyncMock, return_value=None
    )


def _task(**overrides) -> Task:
    fields = {
        "id": "task-1",
        "user_id": "test-user-id",
        "title": "Call supplier",
        "status": TaskStatus.TODO,
        "created_by": TaskCreator.USER,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    return Task(**(fields | overrides))


def _mock(mocker: MockerFixture, name: str, **kwargs) -> AsyncMock:
    return mocker.patch(f"{_ROUTES}.{name}", new_callable=AsyncMock, **kwargs)


@pytest.mark.parametrize("sort", ["updated", "created", "title"])
def test_the_list_is_sorted_as_asked(mocker: MockerFixture, sort):
    page = _mock(mocker, "list_task_page", return_value=TaskPage(tasks=[]))

    response = client.get("/tasks", params={"sort": sort})

    assert response.status_code == 200
    assert page.await_args.args[1].sort == sort


def test_the_list_sorts_by_last_update_unless_asked(mocker: MockerFixture):
    page = _mock(mocker, "list_task_page", return_value=TaskPage(tasks=[]))

    client.get("/tasks")

    assert page.await_args.args[1].sort == "updated"


def test_an_unknown_sort_is_refused(mocker: MockerFixture):
    page = _mock(mocker, "list_task_page", return_value=TaskPage(tasks=[]))

    assert client.get("/tasks", params={"sort": "due"}).status_code == 422
    page.assert_not_awaited()


def test_an_archived_task_comes_back(mocker: MockerFixture, test_user_id):
    restore = _mock(mocker, "unarchive_task", return_value=True)
    _mock(mocker, "get_task", return_value=_task())
    note = _mock(mocker, "add_activity")

    response = client.post("/tasks/task-1/unarchive")

    assert response.status_code == 200
    assert response.json()["id"] == "task-1"
    restore.assert_awaited_once_with("task-1", test_user_id)
    assert note.await_args.args[2].text == "You brought it back from the archive"


def test_unarchiving_a_task_that_is_not_archived_is_404(mocker: MockerFixture):
    _mock(mocker, "unarchive_task", return_value=False)

    response = client.post("/tasks/task-1/unarchive")

    assert response.status_code == 404
    assert response.json()["detail"] == "This task no longer exists."


@pytest.mark.parametrize("title", ["   ", "\n\t"])
def test_a_title_of_spaces_is_a_422_by_the_field(mocker: MockerFixture, title):
    create = _mock(mocker, "create_task", return_value=_task())

    response = client.post("/tasks", json={"title": title})

    assert response.status_code == 422
    assert response.json()["detail"] == "A task needs a title."
    create.assert_not_awaited()


def test_a_rename_to_spaces_is_a_422(mocker: MockerFixture):
    _mock(mocker, "get_task", return_value=_task())
    update = _mock(mocker, "update_task", return_value=_task())

    response = client.patch("/tasks/task-1", json={"title": "  "})

    assert response.status_code == 422
    assert response.json()["detail"] == "A task needs a title."
    update.assert_not_awaited()


@pytest.mark.parametrize(
    "method, body",
    [
        ("post", {"title": "Call supplier", "colour": "red"}),
        # A watch is set up through Otto (on_event), not on create.
        (
            "post",
            {
                "title": "Call supplier",
                "repeat": {"crons": ["0 8 * * *"], "watch": {"provider": "github"}},
            },
        ),
        ("patch", {"title": "Call supplier", "priority": 1}),
    ],
)
def test_unknown_fields_are_refused_not_dropped(mocker: MockerFixture, method, body):
    create = _mock(mocker, "create_task", return_value=_task())
    update = _mock(mocker, "update_task", return_value=_task())
    _mock(mocker, "get_task", return_value=_task())

    path = "/tasks" if method == "post" else "/tasks/task-1"
    response = getattr(client, method)(path, json=body)

    assert response.status_code == 422
    create.assert_not_awaited()
    update.assert_not_awaited()


def test_a_second_start_gets_the_already_started_answer(mocker: MockerFixture):
    _mock(mocker, "start_task_run", side_effect=ConflictError(ALREADY_STARTED))

    response = client.post("/tasks/task-1/run")

    assert response.status_code == 409
    assert response.json()["detail"] == "It is already being worked on."


@pytest.mark.parametrize(
    "method, path, store, body",
    [
        ("get", "/tasks/task-1", "get_task", None),
        ("get", "/tasks/task-1/record", "get_task_record", None),
        ("patch", "/tasks/task-1", "get_task", {"title": "x"}),
        ("delete", "/tasks/task-1", "get_task", None),
        ("post", "/tasks/task-1/run", "start_task_run", None),
        ("post", "/tasks/task-1/cancel", "cancel_task_run", None),
        ("post", "/tasks/task-1/review", "set_task_review", {"decision": "accept"}),
        ("post", "/tasks/task-1/repeat/pause", "pause_task_repeat", None),
        ("post", "/tasks/task-1/repeat/resume", "resume_task_repeat", None),
    ],
)
def test_a_task_that_is_gone_is_404_without_its_id(
    mocker: MockerFixture, method, path, store, body
):
    if store in ("get_task", "get_task_record"):
        _mock(mocker, store, return_value=None)
    else:
        _mock(mocker, store, side_effect=NotFoundError(TASK_GONE))

    response = getattr(client, method)(
        path, **({"json": body} if body is not None else {})
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "This task no longer exists."
