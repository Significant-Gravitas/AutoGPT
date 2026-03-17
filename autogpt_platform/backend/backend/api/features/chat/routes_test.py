"""Tests for chat API routes: session title update, file attachment validation, usage, rate limiting, and suggested prompts."""

import asyncio
from datetime import UTC, datetime, timedelta, timezone
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


def test_get_session_hides_internal_only_messages_for_manual_sessions(
    mocker: pytest_mock.MockerFixture,
) -> None:
    session = ChatSession.new(
        TEST_USER_ID,
        start_type=ChatSessionStartType.MANUAL,
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


def test_get_session_shows_cleaned_internal_kickoff_for_autopilot_sessions(
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
            "content": "hidden",
            "name": None,
            "tool_call_id": None,
            "refusal": None,
            "tool_calls": None,
            "function_call": None,
        },
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
    workspace_store = mocker.MagicMock()
    workspace_store.get_workspace_files_by_ids = mocker.AsyncMock(return_value=[])
    mocker.patch(
        "backend.api.features.chat.routes.workspace_db",
        return_value=workspace_store,
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

    workspace_store = mocker.MagicMock()
    workspace_store.get_workspace_files_by_ids = mocker.AsyncMock(return_value=[])
    mocker.patch(
        "backend.api.features.chat.routes.workspace_db",
        return_value=workspace_store,
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
    workspace_store.get_workspace_files_by_ids.assert_called_once_with(
        workspace_id="ws-1",
        file_ids=[valid_id],
    )


# ─── Cross-workspace file_ids ─────────────────────────────────────────


def test_file_ids_scoped_to_workspace(mocker: pytest_mock.MockFixture):
    """The batch query should scope to the user's workspace."""
    _mock_stream_internals(mocker)
    mocker.patch(
        "backend.api.features.chat.routes.get_or_create_workspace",
        return_value=type("W", (), {"id": "my-workspace-id"})(),
    )

    workspace_store = mocker.MagicMock()
    workspace_store.get_workspace_files_by_ids = mocker.AsyncMock(return_value=[])
    mocker.patch(
        "backend.api.features.chat.routes.workspace_db",
        return_value=workspace_store,
    )

    fid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    client.post(
        "/sessions/sess-1/stream",
        json={"message": "hi", "file_ids": [fid]},
    )

    workspace_store.get_workspace_files_by_ids.assert_called_once_with(
        workspace_id="my-workspace-id",
        file_ids=[fid],
    )


# ─── Rate limit → 429 ─────────────────────────────────────────────────


def test_stream_chat_returns_429_on_daily_rate_limit(mocker: pytest_mock.MockFixture):
    """When check_rate_limit raises RateLimitExceeded for daily limit the endpoint returns 429."""
    from backend.copilot.rate_limit import RateLimitExceeded

    _mock_stream_internals(mocker)
    # Ensure the rate-limit branch is entered by setting a non-zero limit.
    mocker.patch.object(chat_routes.config, "daily_token_limit", 10000)
    mocker.patch.object(chat_routes.config, "weekly_token_limit", 50000)
    mocker.patch(
        "backend.api.features.chat.routes.check_rate_limit",
        side_effect=RateLimitExceeded("daily", datetime.now(UTC) + timedelta(hours=1)),
    )

    response = client.post(
        "/sessions/sess-1/stream",
        json={"message": "hello"},
    )
    assert response.status_code == 429
    assert "daily" in response.json()["detail"].lower()


def test_stream_chat_returns_429_on_weekly_rate_limit(mocker: pytest_mock.MockFixture):
    """When check_rate_limit raises RateLimitExceeded for weekly limit the endpoint returns 429."""
    from backend.copilot.rate_limit import RateLimitExceeded

    _mock_stream_internals(mocker)
    mocker.patch.object(chat_routes.config, "daily_token_limit", 10000)
    mocker.patch.object(chat_routes.config, "weekly_token_limit", 50000)
    resets_at = datetime.now(UTC) + timedelta(days=3)
    mocker.patch(
        "backend.api.features.chat.routes.check_rate_limit",
        side_effect=RateLimitExceeded("weekly", resets_at),
    )

    response = client.post(
        "/sessions/sess-1/stream",
        json={"message": "hello"},
    )
    assert response.status_code == 429
    detail = response.json()["detail"].lower()
    assert "weekly" in detail
    assert "resets in" in detail


def test_stream_chat_429_includes_reset_time(mocker: pytest_mock.MockFixture):
    """The 429 response detail should include the human-readable reset time."""
    from backend.copilot.rate_limit import RateLimitExceeded

    _mock_stream_internals(mocker)
    mocker.patch.object(chat_routes.config, "daily_token_limit", 10000)
    mocker.patch.object(chat_routes.config, "weekly_token_limit", 50000)
    mocker.patch(
        "backend.api.features.chat.routes.check_rate_limit",
        side_effect=RateLimitExceeded(
            "daily", datetime.now(UTC) + timedelta(hours=2, minutes=30)
        ),
    )

    response = client.post(
        "/sessions/sess-1/stream",
        json={"message": "hello"},
    )
    assert response.status_code == 429
    detail = response.json()["detail"]
    assert "2h" in detail
    assert "Resets in" in detail


# ─── Usage endpoint ───────────────────────────────────────────────────


def _mock_usage(
    mocker: pytest_mock.MockerFixture,
    *,
    daily_used: int = 500,
    weekly_used: int = 2000,
) -> AsyncMock:
    """Mock get_usage_status to return a predictable CoPilotUsageStatus."""
    from backend.copilot.rate_limit import CoPilotUsageStatus, UsageWindow

    resets_at = datetime.now(UTC) + timedelta(days=1)
    status = CoPilotUsageStatus(
        daily=UsageWindow(used=daily_used, limit=10000, resets_at=resets_at),
        weekly=UsageWindow(used=weekly_used, limit=50000, resets_at=resets_at),
    )
    return mocker.patch(
        "backend.api.features.chat.routes.get_usage_status",
        new_callable=AsyncMock,
        return_value=status,
    )


def test_usage_returns_daily_and_weekly(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """GET /usage returns daily and weekly usage."""
    mock_get = _mock_usage(mocker, daily_used=500, weekly_used=2000)

    mocker.patch.object(chat_routes.config, "daily_token_limit", 10000)
    mocker.patch.object(chat_routes.config, "weekly_token_limit", 50000)

    response = client.get("/usage")

    assert response.status_code == 200
    data = response.json()
    assert data["daily"]["used"] == 500
    assert data["weekly"]["used"] == 2000

    mock_get.assert_called_once_with(
        user_id=test_user_id,
        daily_token_limit=10000,
        weekly_token_limit=50000,
    )


def test_usage_uses_config_limits(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """The endpoint forwards daily_token_limit and weekly_token_limit from config."""
    mock_get = _mock_usage(mocker)

    mocker.patch.object(chat_routes.config, "daily_token_limit", 99999)
    mocker.patch.object(chat_routes.config, "weekly_token_limit", 77777)

    response = client.get("/usage")

    assert response.status_code == 200
    mock_get.assert_called_once_with(
        user_id=test_user_id,
        daily_token_limit=99999,
        weekly_token_limit=77777,
    )


def test_usage_rejects_unauthenticated_request() -> None:
    """GET /usage should return 401 when no valid JWT is provided."""
    unauthenticated_app = fastapi.FastAPI()
    unauthenticated_app.include_router(chat_routes.router)
    unauthenticated_client = fastapi.testclient.TestClient(unauthenticated_app)

    response = unauthenticated_client.get("/usage")

    assert response.status_code == 401


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
