"""Tests for chat API routes: session title update and file attachment validation."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.api.features.chat import routes as chat_routes
from backend.copilot.model import ChatSession
from backend.util.exceptions import NotFoundError

app = fastapi.FastAPI()
app.include_router(chat_routes.router)

client = fastapi.testclient.TestClient(app)

TEST_USER_ID = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"


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


# ─── file_ids Pydantic validation ─────────────────────────────────────


def test_stream_chat_rejects_too_many_file_ids():
    """More than 20 file_ids should be rejected by Pydantic validation (422)."""
    response = client.post(
        "/sessions/sess-1/stream",
        json={
            "message": "hello",
            "file_ids": [f"00000000-0000-0000-0000-{i:012d}" for i in range(21)],
        },
    )
    assert response.status_code == 422


def _mock_stream_internals(mocker: pytest_mock.MockFixture):
    """Mock the async internals of stream_chat_post so tests can exercise
    validation and enrichment logic without needing Redis/RabbitMQ."""
    mocker.patch(
        "backend.api.features.chat.routes._validate_and_get_session",
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.chat.routes.append_and_save_message",
        return_value=None,
    )
    mock_registry = mocker.MagicMock()
    mock_registry.create_session = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "backend.api.features.chat.routes.stream_registry",
        mock_registry,
    )
    mocker.patch(
        "backend.api.features.chat.routes.enqueue_copilot_turn",
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.chat.routes.track_user_message",
        return_value=None,
    )


def test_stream_chat_accepts_20_file_ids(mocker: pytest_mock.MockFixture):
    """Exactly 20 file_ids should be accepted (not rejected by validation)."""
    _mock_stream_internals(mocker)
    # Patch workspace lookup as imported by the routes module
    mocker.patch(
        "backend.api.features.chat.routes.get_or_create_workspace",
        return_value=type("W", (), {"id": "ws-1"})(),
    )
    mock_prisma = mocker.MagicMock()
    mock_prisma.find_many = mocker.AsyncMock(return_value=[])
    mocker.patch(
        "prisma.models.UserWorkspaceFile.prisma",
        return_value=mock_prisma,
    )

    response = client.post(
        "/sessions/sess-1/stream",
        json={
            "message": "hello",
            "file_ids": [f"00000000-0000-0000-0000-{i:012d}" for i in range(20)],
        },
    )
    # Should get past validation — 200 streaming response expected
    assert response.status_code == 200


# ─── UUID format filtering ─────────────────────────────────────────────


def test_file_ids_filters_invalid_uuids(mocker: pytest_mock.MockFixture):
    """Non-UUID strings in file_ids should be silently filtered out
    and NOT passed to the database query."""
    _mock_stream_internals(mocker)
    mocker.patch(
        "backend.api.features.chat.routes.get_or_create_workspace",
        return_value=type("W", (), {"id": "ws-1"})(),
    )

    mock_prisma = mocker.MagicMock()
    mock_prisma.find_many = mocker.AsyncMock(return_value=[])
    mocker.patch(
        "prisma.models.UserWorkspaceFile.prisma",
        return_value=mock_prisma,
    )

    valid_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    client.post(
        "/sessions/sess-1/stream",
        json={
            "message": "hello",
            "file_ids": [
                valid_id,
                "not-a-uuid",
                "../../../etc/passwd",
                "",
            ],
        },
    )

    # The find_many call should only receive the one valid UUID
    mock_prisma.find_many.assert_called_once()
    call_kwargs = mock_prisma.find_many.call_args[1]
    assert call_kwargs["where"]["id"]["in"] == [valid_id]


# ─── Cross-workspace file_ids ─────────────────────────────────────────


def test_file_ids_scoped_to_workspace(mocker: pytest_mock.MockFixture):
    """The batch query should scope to the user's workspace."""
    _mock_stream_internals(mocker)
    mocker.patch(
        "backend.api.features.chat.routes.get_or_create_workspace",
        return_value=type("W", (), {"id": "my-workspace-id"})(),
    )

    mock_prisma = mocker.MagicMock()
    mock_prisma.find_many = mocker.AsyncMock(return_value=[])
    mocker.patch(
        "prisma.models.UserWorkspaceFile.prisma",
        return_value=mock_prisma,
    )

    fid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    client.post(
        "/sessions/sess-1/stream",
        json={"message": "hi", "file_ids": [fid]},
    )

    call_kwargs = mock_prisma.find_many.call_args[1]
    assert call_kwargs["where"]["workspaceId"] == "my-workspace-id"
    assert call_kwargs["where"]["isDeleted"] is False
