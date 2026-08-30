"""Tests for the task-spine API routes.

Pattern mirrors backend/api/features/experts/routes_test.py: a local FastAPI
app, the global `mock_jwt_user` auth override fixture, and `tasks_db` mocked
with AsyncMock at the route module's import site.

The tenancy tests here assert that every handler *forwards the caller's own
user id* into the DB layer — that forwarding is what makes user A's tasks
unreachable to user B, and it is the thing a refactor is most likely to drop.
The end-to-end proof against a real database lives in ``tasks_db_test.py``.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.jwt_utils import get_jwt_payload

from backend.api.features.tasks.errors import DelegatedTaskNotFoundError
from backend.api.features.tasks.models import (
    DelegatedTask,
    DelegatedTaskDetail,
    TaskExpertRef,
)
from backend.api.features.tasks.routes import router

app = fastapi.FastAPI()
app.include_router(router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Setup auth overrides for all tests in this module"""
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    app.dependency_overrides[get_request_context] = mock_jwt_user["get_request_context"]
    yield
    app.dependency_overrides.clear()


def _make_task(**overrides) -> DelegatedTask:
    now = datetime(2026, 8, 30, 9, 0, tzinfo=timezone.utc)
    values = {
        "id": "task-1",
        "title": "Draft the weekly report",
        "spec": "Run Weekly Report with:\n- week: current",
        "status": "WORKING",
        "acceptance": "PENDING",
        "created_by_type": "USER",
        "created_by_id": "user-1",
        "owner": TaskExpertRef(
            id="expert-maria",
            name="Maria",
            avatar_url=None,
            role="Marketing Strategist",
        ),
        "parent_task_id": None,
        "root_task_id": "task-1",
        "origin_session_id": "session-1",
        "ancestor_expert_ids": ["expert-maria"],
        "handoff_count": 0,
        "revision_count": 0,
        "spend_total": 250,
        "outcome_summary": None,
        "amendments": [],
        "created_at": now,
        "updated_at": now,
        "runs": [],
    }
    values.update(overrides)
    return DelegatedTask(**values)


# ─── list ──────────────────────────────────────────────────────────────


def test_list_tasks_returns_the_callers_tasks(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_list = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.list_tasks",
        new_callable=AsyncMock,
        return_value=[_make_task()],
    )

    response = client.get("/tasks")

    assert response.status_code == 200
    data = response.json()
    assert [task["id"] for task in data] == ["task-1"]
    assert data[0]["owner"]["name"] == "Maria"
    mock_list.assert_awaited_once_with(
        test_user_id, expert_id=None, status=None, limit=50
    )


def test_list_tasks_forwards_expert_and_status_filters(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_list = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.list_tasks",
        new_callable=AsyncMock,
        return_value=[],
    )

    response = client.get("/tasks?expert_id=expert-maria&status=DONE&limit=5")

    assert response.status_code == 200
    mock_list.assert_awaited_once_with(
        test_user_id, expert_id="expert-maria", status="DONE", limit=5
    )


def test_list_tasks_rejects_an_unknown_status() -> None:
    response = client.get("/tasks?status=NOPE")

    assert response.status_code == 422


def test_list_tasks_rejects_an_oversized_limit() -> None:
    """The cap exists so one user's history can't turn into an unbounded
    scan; a client asking past it must be refused, not silently clamped."""
    response = client.get("/tasks?limit=500")

    assert response.status_code == 422


# ─── detail ────────────────────────────────────────────────────────────


def test_get_task_returns_the_task_and_its_children(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_get = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.get_task",
        new_callable=AsyncMock,
        return_value=DelegatedTaskDetail(
            task=_make_task(),
            children=[_make_task(id="task-2", parent_task_id="task-1")],
        ),
    )

    response = client.get("/tasks/task-1")

    assert response.status_code == 200
    data = response.json()
    assert data["task"]["id"] == "task-1"
    assert [child["id"] for child in data["children"]] == ["task-2"]
    mock_get.assert_awaited_once_with(test_user_id, "task-1")


def test_get_task_returns_404_for_another_users_task(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """``get_task`` answers None for both "missing" and "not yours", and the
    route must not distinguish them — otherwise a 404-vs-403 split would
    confirm the existence of another user's task id."""
    mock_get = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.get_task",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.get("/tasks/someone-elses-task")

    assert response.status_code == 404
    assert response.json()["detail"] == "Task not found"
    mock_get.assert_awaited_once_with(test_user_id, "someone-elses-task")


# ─── cancel ────────────────────────────────────────────────────────────


def test_cancel_task_cancels_and_returns_the_tree(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_cancel = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.cancel_task",
        new_callable=AsyncMock,
        return_value=DelegatedTaskDetail(
            task=_make_task(status="CANCELLED"),
            children=[_make_task(id="task-2", status="CANCELLED")],
        ),
    )

    response = client.post("/tasks/task-1/cancel")

    assert response.status_code == 200
    data = response.json()
    assert data["task"]["status"] == "CANCELLED"
    assert data["children"][0]["status"] == "CANCELLED"
    mock_cancel.assert_awaited_once_with(test_user_id, "task-1")


def test_cancel_task_returns_404_for_another_users_task(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_cancel = mocker.patch(
        "backend.api.features.tasks.routes.tasks_db.cancel_task",
        new_callable=AsyncMock,
        side_effect=DelegatedTaskNotFoundError("someone-elses-task"),
    )

    response = client.post("/tasks/someone-elses-task/cancel")

    assert response.status_code == 404
    mock_cancel.assert_awaited_once_with(test_user_id, "someone-elses-task")
