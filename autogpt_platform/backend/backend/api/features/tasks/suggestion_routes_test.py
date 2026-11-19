"""Tests for the suggested-task routes: listing, dismissing and refreshing."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi import HTTPException
from prisma.enums import TaskCreator, TaskStatus
from pytest_mock import MockerFixture

from backend.api.utils.exceptions import add_exception_handlers
from backend.data.task import Task
from backend.util.exceptions import NotFoundError

from .routes import router as task_router
from .suggestion_routes import router

app = fastapi.FastAPI()
# The order the app includes them in: the suggestion routes first.
app.include_router(router)
app.include_router(task_router)
add_exception_handlers(app)
client = fastapi.testclient.TestClient(app)

_ROUTES = "backend.api.features.tasks.suggestion_routes"
_NOW = datetime(2026, 11, 14, 8, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def limiter(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        f"{_ROUTES}._refresh_limiter.check", new_callable=AsyncMock, return_value=None
    )


def _suggestion(**overrides) -> Task:
    fields = {
        "id": "task-1",
        "user_id": "test-user-id",
        "title": "Pricing deck",
        "status": TaskStatus.TODO,
        "created_by": TaskCreator.OTTO,
        "created_at": _NOW,
        "updated_at": _NOW,
        "is_suggested": True,
        "suggested_reason": "The pricing comparison finished yesterday.",
    }
    return Task(**(fields | overrides))


def _mock(mocker: MockerFixture, name: str, **kwargs) -> AsyncMock:
    return mocker.patch(f"{_ROUTES}.{name}", new_callable=AsyncMock, **kwargs)


def test_lists_only_live_suggestions(mocker: MockerFixture, test_user_id):
    list_tasks = _mock(mocker, "list_tasks", return_value=[_suggestion()])

    response = client.get("/tasks/suggestions")

    assert response.status_code == 200
    [task] = response.json()["tasks"]
    assert task["is_suggested"] is True
    assert task["suggested_reason"] == "The pricing comparison finished yesterday."
    list_tasks.assert_awaited_once_with(user_id=test_user_id, suggested=True, limit=3)


def test_dismiss_sets_the_suggestion_aside(mocker: MockerFixture, test_user_id):
    dismiss = _mock(mocker, "dismiss_task_suggestion", return_value=_suggestion())

    response = client.post("/tasks/task-1/dismiss")

    assert response.status_code == 200
    dismiss.assert_awaited_once_with("task-1", test_user_id)


@pytest.mark.parametrize(
    "error, status",
    [
        (NotFoundError("Task task-1 not found"), 404),
        (ValueError("Only a suggested task can be dismissed."), 400),
    ],
)
def test_dismiss_refuses_a_missing_or_unsuggested_task(
    mocker: MockerFixture, error, status
):
    _mock(mocker, "dismiss_task_suggestion", side_effect=error)
    assert client.post("/tasks/task-1/dismiss").status_code == status


def test_refresh_drafts_for_the_caller(
    mocker: MockerFixture, limiter: AsyncMock, test_user_id
):
    draft = _mock(mocker, "draft_task_suggestions", return_value=[_suggestion()])

    response = client.post("/tasks/suggestions/refresh")

    assert response.status_code == 200
    assert [t["title"] for t in response.json()["tasks"]] == ["Pricing deck"]
    draft.assert_awaited_once_with(test_user_id)
    limiter.assert_awaited_once_with(test_user_id)


def test_refresh_past_the_limit_drafts_nothing(
    mocker: MockerFixture, limiter: AsyncMock
):
    limiter.side_effect = HTTPException(status_code=429, detail="Rate limit exceeded")
    draft = _mock(mocker, "draft_task_suggestions", return_value=[])

    assert client.post("/tasks/suggestions/refresh").status_code == 429
    draft.assert_not_awaited()
