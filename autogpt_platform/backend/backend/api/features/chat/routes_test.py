from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.copilot.model import ChatSession
from backend.util.exceptions import NotFoundError

from .routes import router as chat_router

app = fastapi.FastAPI()
app.include_router(chat_router)

client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user):
    """Setup auth overrides for all tests in this module"""
    from autogpt_libs.auth.jwt_utils import get_jwt_payload

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    yield
    app.dependency_overrides.clear()


def _mock_validate_and_get_session(mocker: pytest_mock.MockerFixture):
    """Mock _validate_and_get_session to succeed (session exists & belongs to user)."""
    return mocker.patch(
        "backend.api.features.chat.routes._validate_and_get_session",
        new_callable=AsyncMock,
    )


def _mock_validate_not_found(mocker: pytest_mock.MockerFixture):
    """Mock _validate_and_get_session to raise NotFoundError."""
    mock = mocker.patch(
        "backend.api.features.chat.routes._validate_and_get_session",
        new_callable=AsyncMock,
    )
    mock.side_effect = NotFoundError("Session not found.")
    return mock


def _mock_update_session_title(
    mocker: pytest_mock.MockerFixture, *, success: bool = True
):
    """Mock update_session_title."""
    return mocker.patch(
        "backend.api.features.chat.routes.update_session_title",
        new_callable=AsyncMock,
        return_value=success,
    )


def _mock_get_chat_session(mocker: pytest_mock.MockerFixture, *, exists: bool = True):
    """Mock get_chat_session for the error-handling re-check path."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    value = (
        ChatSession(
            session_id="sess-1",
            user_id="user-1",
            messages=[],
            usage=[],
            started_at=now,
            updated_at=now,
        )
        if exists
        else None
    )
    return mocker.patch(
        "backend.api.features.chat.routes.get_chat_session",
        new_callable=AsyncMock,
        return_value=value,
    )


# ─── Update title: success ─────────────────────────────────────────────


def test_update_title_success(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    _mock_validate_and_get_session(mocker)
    mock_update = _mock_update_session_title(mocker, success=True)

    response = client.patch(
        "/sessions/sess-1/title",
        json={"title": "My project"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_update.assert_called_once_with("sess-1", "My project")


def test_update_title_trims_whitespace(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    _mock_validate_and_get_session(mocker)
    mock_update = _mock_update_session_title(mocker, success=True)

    response = client.patch(
        "/sessions/sess-1/title",
        json={"title": "  trimmed  "},
    )

    assert response.status_code == 200
    mock_update.assert_called_once_with("sess-1", "trimmed")


# ─── Update title: blank / whitespace-only → 422 ──────────────────────


def test_update_title_blank_rejected(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """Whitespace-only titles must be rejected before hitting the DB."""
    _mock_validate_and_get_session(mocker)

    response = client.patch(
        "/sessions/sess-1/title",
        json={"title": "   "},
    )

    assert response.status_code == 422


def test_update_title_empty_rejected(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    _mock_validate_and_get_session(mocker)

    response = client.patch(
        "/sessions/sess-1/title",
        json={"title": ""},
    )

    assert response.status_code == 422


# ─── Update title: session not found → 404 ────────────────────────────


def test_update_title_session_not_found(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    _mock_validate_not_found(mocker)

    response = client.patch(
        "/sessions/sess-1/title",
        json={"title": "New name"},
    )

    assert response.status_code == 404


# ─── Update title: update_session_title fails → 500 ───────────────────


def test_update_title_internal_failure(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """When update_session_title returns False but the session still exists,
    report a 500 rather than a misleading 404."""
    _mock_validate_and_get_session(mocker)
    _mock_update_session_title(mocker, success=False)
    _mock_get_chat_session(mocker, exists=True)

    response = client.patch(
        "/sessions/sess-1/title",
        json={"title": "New name"},
    )

    assert response.status_code == 500


def test_update_title_disappeared_after_validate(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """Session disappeared between validate and update → 404."""
    _mock_validate_and_get_session(mocker)
    _mock_update_session_title(mocker, success=False)
    _mock_get_chat_session(mocker, exists=False)

    response = client.patch(
        "/sessions/sess-1/title",
        json={"title": "New name"},
    )

    assert response.status_code == 404
