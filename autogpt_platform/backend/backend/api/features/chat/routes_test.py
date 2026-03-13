"""Tests for chat API routes: session title update, file attachment validation, and suggested prompts."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.api.features.chat import routes as chat_routes
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.response_model import StreamFinish
from backend.copilot.session_types import ChatSessionStartType

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


def _mock_update_session_title(
    mocker: pytest_mock.MockerFixture, *, success: bool = True
):
    """Mock update_session_title."""
    return mocker.patch(
        "backend.api.features.chat.routes.update_session_title",
        new_callable=AsyncMock,
        return_value=success,
    )


# ─── Update title: success ─────────────────────────────────────────────


def test_update_title_success(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_update = _mock_update_session_title(mocker, success=True)

    response = client.patch(
        "/sessions/sess-1/title",
        json={"title": "My project"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_update.assert_called_once_with("sess-1", test_user_id, "My project")


def test_update_title_trims_whitespace(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_update = _mock_update_session_title(mocker, success=True)

    response = client.patch(
        "/sessions/sess-1/title",
        json={"title": "  trimmed  "},
    )

    assert response.status_code == 200
    mock_update.assert_called_once_with("sess-1", test_user_id, "trimmed")


# ─── Update title: blank / whitespace-only → 422 ──────────────────────


def test_update_title_blank_rejected(
    test_user_id: str,
) -> None:
    """Whitespace-only titles must be rejected before hitting the DB."""
    response = client.patch(
        "/sessions/sess-1/title",
        json={"title": "   "},
    )

    assert response.status_code == 422


def test_update_title_empty_rejected(
    test_user_id: str,
) -> None:
    response = client.patch(
        "/sessions/sess-1/title",
        json={"title": ""},
    )

    assert response.status_code == 422


# ─── Update title: session not found or wrong user → 404 ──────────────


def test_update_title_not_found(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    _mock_update_session_title(mocker, success=False)

    response = client.patch(
        "/sessions/sess-1/title",
        json={"title": "New name"},
    )

    assert response.status_code == 404


def test_list_sessions_defaults_to_manual_only(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    started_at = datetime.now(timezone.utc)
    mock_get_user_sessions = mocker.patch(
        "backend.api.features.chat.routes.get_user_sessions",
        new_callable=AsyncMock,
        return_value=(
            [
                SimpleNamespace(
                    session_id="sess-1",
                    started_at=started_at,
                    updated_at=started_at,
                    title="Nightly check-in",
                    start_type=chat_routes.ChatSessionStartType.AUTOPILOT_NIGHTLY,
                    execution_tag="autopilot-nightly:2026-03-13",
                )
            ],
            1,
        ),
    )

    pipe = MagicMock()
    pipe.hget = MagicMock()
    pipe.execute = AsyncMock(return_value=["running"])
    redis = MagicMock()
    redis.pipeline = MagicMock(return_value=pipe)
    mocker.patch(
        "backend.api.features.chat.routes.get_redis_async",
        new_callable=AsyncMock,
        return_value=redis,
    )

    response = client.get("/sessions")

    assert response.status_code == 200
    assert response.json() == {
        "sessions": [
            {
                "id": "sess-1",
                "created_at": started_at.isoformat(),
                "updated_at": started_at.isoformat(),
                "title": "Nightly check-in",
                "start_type": "AUTOPILOT_NIGHTLY",
                "execution_tag": "autopilot-nightly:2026-03-13",
                "is_processing": True,
            }
        ],
        "total": 1,
    }
    mock_get_user_sessions.assert_awaited_once_with(
        test_user_id,
        50,
        0,
        with_auto=False,
    )


def test_list_sessions_can_include_auto_sessions(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_get_user_sessions = mocker.patch(
        "backend.api.features.chat.routes.get_user_sessions",
        new_callable=AsyncMock,
        return_value=([], 0),
    )

    response = client.get("/sessions?with_auto=true")

    assert response.status_code == 200
    assert response.json() == {"sessions": [], "total": 0}
    mock_get_user_sessions.assert_awaited_once_with(
        test_user_id,
        50,
        0,
        with_auto=True,
    )


def test_consume_callback_token_route_returns_session_id(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mock_consume = mocker.patch(
        "backend.api.features.chat.routes.consume_callback_token",
        new_callable=AsyncMock,
        return_value=SimpleNamespace(session_id="sess-2"),
    )

    response = client.post(
        "/sessions/callback-token/consume",
        json={"token": "token-123"},
    )

    assert response.status_code == 200
    assert response.json() == {"session_id": "sess-2"}
    mock_consume.assert_awaited_once_with("token-123", TEST_USER_ID)


def test_consume_callback_token_route_returns_404_on_invalid_token(
    mocker: pytest_mock.MockerFixture,
) -> None:
    mocker.patch(
        "backend.api.features.chat.routes.consume_callback_token",
        new_callable=AsyncMock,
        side_effect=ValueError("Callback token not found"),
    )

    response = client.post(
        "/sessions/callback-token/consume",
        json={"token": "token-123"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Callback token not found"}


def test_get_session_hides_internal_only_messages_and_strips_internal_content(
    mocker: pytest_mock.MockerFixture,
) -> None:
    session = ChatSession.new(
        TEST_USER_ID,
        start_type=ChatSessionStartType.AUTOPILOT_NIGHTLY,
        execution_tag="autopilot-nightly:2026-03-13",
    )
    session.messages = [
        ChatMessage(role="user", content="<internal>hidden</internal>"),
        ChatMessage(
            role="user",
            content="Visible<internal>hidden</internal> text",
        ),
        ChatMessage(role="assistant", content="Public response"),
    ]

    mocker.patch(
        "backend.api.features.chat.routes.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.api.features.chat.routes.stream_registry.get_active_session",
        new_callable=AsyncMock,
        return_value=(None, None),
    )

    response = client.get(f"/sessions/{session.session_id}")

    assert response.status_code == 200
    assert response.json()["messages"] == [
        {
            "role": "user",
            "content": "Visible text",
            "name": None,
            "tool_call_id": None,
            "refusal": None,
            "tool_calls": None,
            "function_call": None,
        },
        {
            "role": "assistant",
            "content": "Public response",
            "name": None,
            "tool_call_id": None,
            "refusal": None,
            "tool_calls": None,
            "function_call": None,
        },
    ]


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
    subscriber_queue = asyncio.Queue()
    subscriber_queue.put_nowait(StreamFinish())
    mock_registry.create_session = mocker.AsyncMock(return_value=None)
    mock_registry.subscribe_to_session = mocker.AsyncMock(return_value=subscriber_queue)
    mock_registry.unsubscribe_from_session = mocker.AsyncMock(return_value=None)
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


# ─── Suggested prompts endpoint ──────────────────────────────────────


def _mock_get_business_understanding(
    mocker: pytest_mock.MockerFixture,
    *,
    return_value=None,
):
    """Mock get_business_understanding."""
    return mocker.patch(
        "backend.api.features.chat.routes.get_business_understanding",
        new_callable=AsyncMock,
        return_value=return_value,
    )


def test_suggested_prompts_returns_prompts(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """User with understanding and prompts gets them back."""
    mock_understanding = MagicMock()
    mock_understanding.suggested_prompts = ["Do X", "Do Y", "Do Z"]
    _mock_get_business_understanding(mocker, return_value=mock_understanding)

    response = client.get("/suggested-prompts")

    assert response.status_code == 200
    assert response.json() == {"prompts": ["Do X", "Do Y", "Do Z"]}


def test_suggested_prompts_no_understanding(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """User with no understanding gets empty list."""
    _mock_get_business_understanding(mocker, return_value=None)

    response = client.get("/suggested-prompts")

    assert response.status_code == 200
    assert response.json() == {"prompts": []}


def test_suggested_prompts_empty_prompts(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """User with understanding but no prompts gets empty list."""
    mock_understanding = MagicMock()
    mock_understanding.suggested_prompts = []
    _mock_get_business_understanding(mocker, return_value=mock_understanding)

    response = client.get("/suggested-prompts")

    assert response.status_code == 200
    assert response.json() == {"prompts": []}
