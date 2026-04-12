"""Tests for chat API routes: session title update, file attachment validation, usage, and rate limiting."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.api.features.chat import routes as chat_routes
from backend.copilot.rate_limit import SubscriptionTier

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
    daily_limit: int = 10000,
    weekly_limit: int = 50000,
    tier: "SubscriptionTier" = SubscriptionTier.FREE,
) -> AsyncMock:
    """Mock get_usage_status and get_global_rate_limits for usage endpoint tests.

    Mocks both ``get_global_rate_limits`` (returns the given limits + tier) and
    ``get_usage_status`` so that tests exercise the endpoint without hitting
    LaunchDarkly or Prisma.
    """
    from backend.copilot.rate_limit import CoPilotUsageStatus, UsageWindow

    mocker.patch(
        "backend.api.features.chat.routes.get_global_rate_limits",
        new_callable=AsyncMock,
        return_value=(daily_limit, weekly_limit, tier),
    )

    resets_at = datetime.now(UTC) + timedelta(days=1)
    status = CoPilotUsageStatus(
        daily=UsageWindow(used=daily_used, limit=daily_limit, resets_at=resets_at),
        weekly=UsageWindow(used=weekly_used, limit=weekly_limit, resets_at=resets_at),
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
        rate_limit_reset_cost=chat_routes.config.rate_limit_reset_cost,
        tier=SubscriptionTier.FREE,
    )


def test_usage_uses_config_limits(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """The endpoint forwards resolved limits from get_global_rate_limits to get_usage_status."""
    mock_get = _mock_usage(mocker, daily_limit=99999, weekly_limit=77777)

    mocker.patch.object(chat_routes.config, "rate_limit_reset_cost", 500)

    response = client.get("/usage")

    assert response.status_code == 200
    mock_get.assert_called_once_with(
        user_id=test_user_id,
        daily_token_limit=99999,
        weekly_token_limit=77777,
        rate_limit_reset_cost=500,
        tier=SubscriptionTier.FREE,
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


def test_suggested_prompts_returns_themes(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """User with themed prompts gets them back as themes list."""
    mock_understanding = MagicMock()
    mock_understanding.suggested_prompts = {
        "Learn": ["L1", "L2"],
        "Create": ["C1"],
    }
    _mock_get_business_understanding(mocker, return_value=mock_understanding)

    response = client.get("/suggested-prompts")

    assert response.status_code == 200
    data = response.json()
    assert "themes" in data
    themes_by_name = {t["name"]: t["prompts"] for t in data["themes"]}
    assert themes_by_name["Learn"] == ["L1", "L2"]
    assert themes_by_name["Create"] == ["C1"]


def test_suggested_prompts_no_understanding(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """User with no understanding gets empty themes list."""
    _mock_get_business_understanding(mocker, return_value=None)

    response = client.get("/suggested-prompts")

    assert response.status_code == 200
    assert response.json() == {"themes": []}


def test_suggested_prompts_empty_prompts(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """User with understanding but empty prompts gets empty themes list."""
    mock_understanding = MagicMock()
    mock_understanding.suggested_prompts = {}
    _mock_get_business_understanding(mocker, return_value=mock_understanding)

    response = client.get("/suggested-prompts")

    assert response.status_code == 200
    assert response.json() == {"themes": []}


# ─── Create session: dry_run contract ─────────────────────────────────


def _mock_create_chat_session(mocker: pytest_mock.MockerFixture):
    """Mock create_chat_session to return a fake session."""
    from backend.copilot.model import ChatSession

    async def _fake_create(user_id: str, *, dry_run: bool):
        return ChatSession.new(user_id, dry_run=dry_run)

    return mocker.patch(
        "backend.api.features.chat.routes.create_chat_session",
        new_callable=AsyncMock,
        side_effect=_fake_create,
    )


def test_create_session_dry_run_true(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """Sending ``{"dry_run": true}`` sets metadata.dry_run to True."""
    _mock_create_chat_session(mocker)

    response = client.post("/sessions", json={"dry_run": True})

    assert response.status_code == 200
    assert response.json()["metadata"]["dry_run"] is True


def test_create_session_dry_run_default_false(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """Empty body defaults dry_run to False."""
    _mock_create_chat_session(mocker)

    response = client.post("/sessions")

    assert response.status_code == 200
    assert response.json()["metadata"]["dry_run"] is False


def test_create_session_rejects_nested_metadata(
    test_user_id: str,
) -> None:
    """Sending ``{"metadata": {"dry_run": true}}`` must return 422, not silently
    default to ``dry_run=False``. This guards against the common mistake of
    nesting dry_run inside metadata instead of providing it at the top level."""
    response = client.post(
        "/sessions",
        json={"metadata": {"dry_run": True}},
    )

    assert response.status_code == 422


class TestStreamChatRequestModeValidation:
    """Pydantic-level validation of the ``mode`` field on StreamChatRequest."""

    def test_rejects_invalid_mode_value(self) -> None:
        """Any string outside the Literal set must raise ValidationError."""
        from pydantic import ValidationError

        from backend.api.features.chat.routes import StreamChatRequest

        with pytest.raises(ValidationError):
            StreamChatRequest(message="hi", mode="turbo")  # type: ignore[arg-type]

    def test_accepts_fast_mode(self) -> None:
        from backend.api.features.chat.routes import StreamChatRequest

        req = StreamChatRequest(message="hi", mode="fast")
        assert req.mode == "fast"

    def test_accepts_extended_thinking_mode(self) -> None:
        from backend.api.features.chat.routes import StreamChatRequest

        req = StreamChatRequest(message="hi", mode="extended_thinking")
        assert req.mode == "extended_thinking"

    def test_accepts_none_mode(self) -> None:
        """``mode=None`` is valid (server decides via feature flags)."""
        from backend.api.features.chat.routes import StreamChatRequest

        req = StreamChatRequest(message="hi", mode=None)
        assert req.mode is None

    def test_mode_defaults_to_none_when_omitted(self) -> None:
        from backend.api.features.chat.routes import StreamChatRequest

        req = StreamChatRequest(message="hi")
        assert req.mode is None


# ─── QueuePendingMessageRequest validation ────────────────────────────


class TestQueuePendingMessageRequest:
    """Unit tests for QueuePendingMessageRequest field validation."""

    def test_accepts_valid_message(self) -> None:
        from backend.api.features.chat.routes import QueuePendingMessageRequest

        req = QueuePendingMessageRequest(message="hello")
        assert req.message == "hello"

    def test_rejects_empty_message(self) -> None:
        import pydantic

        from backend.api.features.chat.routes import QueuePendingMessageRequest

        with pytest.raises(pydantic.ValidationError):
            QueuePendingMessageRequest(message="")

    def test_rejects_message_over_limit(self) -> None:
        import pydantic

        from backend.api.features.chat.routes import QueuePendingMessageRequest

        with pytest.raises(pydantic.ValidationError):
            QueuePendingMessageRequest(message="x" * 16_001)

    def test_accepts_valid_context(self) -> None:
        from backend.api.features.chat.routes import QueuePendingMessageRequest

        req = QueuePendingMessageRequest(
            message="hi",
            context={"url": "https://example.com", "content": "page text"},
        )
        assert req.context is not None
        assert req.context.url == "https://example.com"

    def test_rejects_context_url_over_limit(self) -> None:
        import pydantic

        from backend.api.features.chat.routes import QueuePendingMessageRequest

        with pytest.raises(pydantic.ValidationError, match="url"):
            QueuePendingMessageRequest(
                message="hi",
                context={"url": "https://example.com/" + "x" * 2_000},
            )

    def test_rejects_context_content_over_limit(self) -> None:
        import pydantic

        from backend.api.features.chat.routes import QueuePendingMessageRequest

        with pytest.raises(pydantic.ValidationError, match="content"):
            QueuePendingMessageRequest(
                message="hi",
                context={"content": "x" * 32_001},
            )

    def test_rejects_extra_fields(self) -> None:
        """extra='forbid' should reject unknown fields."""
        import pydantic

        from backend.api.features.chat.routes import QueuePendingMessageRequest

        with pytest.raises(pydantic.ValidationError):
            QueuePendingMessageRequest(message="hi", unknown_field="bad")  # type: ignore[call-arg]

    def test_accepts_up_to_20_file_ids(self) -> None:
        from backend.api.features.chat.routes import QueuePendingMessageRequest

        req = QueuePendingMessageRequest(
            message="hi",
            file_ids=[f"00000000-0000-0000-0000-{i:012d}" for i in range(20)],
        )
        assert req.file_ids is not None
        assert len(req.file_ids) == 20

    def test_rejects_more_than_20_file_ids(self) -> None:
        import pydantic

        from backend.api.features.chat.routes import QueuePendingMessageRequest

        with pytest.raises(pydantic.ValidationError):
            QueuePendingMessageRequest(
                message="hi",
                file_ids=[f"00000000-0000-0000-0000-{i:012d}" for i in range(21)],
            )


# ─── queue_pending_message endpoint ──────────────────────────────────


def _mock_pending_internals(
    mocker: pytest_mock.MockerFixture,
    *,
    session_exists: bool = True,
    call_count: int = 1,
):
    """Mock all async dependencies for the pending-message endpoint."""
    if session_exists:
        mock_session = mocker.MagicMock()
        mock_session.id = "sess-1"
        mocker.patch(
            "backend.api.features.chat.routes._validate_and_get_session",
            new_callable=AsyncMock,
            return_value=mock_session,
        )
    else:
        mocker.patch(
            "backend.api.features.chat.routes._validate_and_get_session",
            side_effect=fastapi.HTTPException(
                status_code=404, detail="Session not found."
            ),
        )
    mocker.patch(
        "backend.api.features.chat.routes.get_global_rate_limits",
        new_callable=AsyncMock,
        return_value=(0, 0, None),
    )
    mocker.patch(
        "backend.api.features.chat.routes.check_rate_limit",
        new_callable=AsyncMock,
        return_value=None,
    )
    # Mock Redis for per-user call-frequency rate limit (atomic Lua EVAL)
    mock_redis = mocker.MagicMock()
    mock_redis.eval = mocker.AsyncMock(return_value=call_count)
    mocker.patch(
        "backend.api.features.chat.routes.get_redis_async",
        new_callable=AsyncMock,
        return_value=mock_redis,
    )
    mocker.patch(
        "backend.api.features.chat.routes.track_user_message",
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.chat.routes.push_pending_message",
        new_callable=AsyncMock,
        return_value=1,
    )
    mock_registry = mocker.MagicMock()
    mock_registry.get_session = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "backend.api.features.chat.routes.stream_registry",
        mock_registry,
    )


def test_queue_pending_message_returns_202(mocker: pytest_mock.MockerFixture) -> None:
    """Happy path: valid message returns 202 with buffer_length."""
    _mock_pending_internals(mocker)

    response = client.post(
        "/sessions/sess-1/messages/pending",
        json={"message": "follow-up"},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["buffer_length"] == 1
    assert data["turn_in_flight"] is False


def test_queue_pending_message_empty_body_returns_422() -> None:
    """Empty message must be rejected by Pydantic before hitting any route logic."""
    response = client.post(
        "/sessions/sess-1/messages/pending",
        json={"message": ""},
    )
    assert response.status_code == 422


def test_queue_pending_message_missing_message_returns_422() -> None:
    """Missing 'message' field returns 422."""
    response = client.post(
        "/sessions/sess-1/messages/pending",
        json={},
    )
    assert response.status_code == 422


def test_queue_pending_message_session_not_found_returns_404(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """If the session doesn't exist or belong to the user, returns 404."""
    _mock_pending_internals(mocker, session_exists=False)

    response = client.post(
        "/sessions/bad-sess/messages/pending",
        json={"message": "hi"},
    )
    assert response.status_code == 404


def test_queue_pending_message_rate_limited_returns_429(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """When rate limit is exceeded, endpoint returns 429."""
    from backend.copilot.rate_limit import RateLimitExceeded

    _mock_pending_internals(mocker)
    mocker.patch(
        "backend.api.features.chat.routes.check_rate_limit",
        side_effect=RateLimitExceeded("daily", datetime.now(UTC) + timedelta(hours=1)),
    )

    response = client.post(
        "/sessions/sess-1/messages/pending",
        json={"message": "hi"},
    )
    assert response.status_code == 429


def test_queue_pending_message_call_frequency_limit_returns_429(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """When per-user call frequency limit is exceeded, endpoint returns 429."""
    from backend.api.features.chat.routes import _PENDING_CALL_LIMIT

    _mock_pending_internals(mocker, call_count=_PENDING_CALL_LIMIT + 1)

    response = client.post(
        "/sessions/sess-1/messages/pending",
        json={"message": "hi"},
    )
    assert response.status_code == 429
    assert "Too many pending messages" in response.json()["detail"]


def test_queue_pending_message_context_url_too_long_returns_422() -> None:
    """context.url over 2 KB is rejected."""
    response = client.post(
        "/sessions/sess-1/messages/pending",
        json={
            "message": "hi",
            "context": {"url": "https://example.com/" + "x" * 2_000},
        },
    )
    assert response.status_code == 422


def test_queue_pending_message_context_content_too_long_returns_422() -> None:
    """context.content over 32 KB is rejected."""
    response = client.post(
        "/sessions/sess-1/messages/pending",
        json={
            "message": "hi",
            "context": {"content": "x" * 32_001},
        },
    )
    assert response.status_code == 422


def test_queue_pending_message_too_many_file_ids_returns_422() -> None:
    """More than 20 file_ids should be rejected."""
    response = client.post(
        "/sessions/sess-1/messages/pending",
        json={
            "message": "hi",
            "file_ids": [f"00000000-0000-0000-0000-{i:012d}" for i in range(21)],
        },
    )
    assert response.status_code == 422


def test_queue_pending_message_file_ids_scoped_to_workspace(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """File IDs must be sanitized to the user's workspace before push."""
    _mock_pending_internals(mocker)
    mocker.patch(
        "backend.api.features.chat.routes.get_or_create_workspace",
        new_callable=AsyncMock,
        return_value=type("W", (), {"id": "ws-1"})(),
    )
    mock_prisma = mocker.MagicMock()
    mock_prisma.find_many = mocker.AsyncMock(return_value=[])
    mocker.patch(
        "prisma.models.UserWorkspaceFile.prisma",
        return_value=mock_prisma,
    )
    fid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    client.post(
        "/sessions/sess-1/messages/pending",
        json={"message": "hi", "file_ids": [fid, "not-a-uuid"]},
    )

    call_kwargs = mock_prisma.find_many.call_args[1]
    assert call_kwargs["where"]["id"]["in"] == [fid]
    assert call_kwargs["where"]["workspaceId"] == "ws-1"
    assert call_kwargs["where"]["isDeleted"] is False
