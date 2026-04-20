"""Tests for chat API routes: session title update, file attachment validation, usage, and rate limiting."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock

from backend.api.features.chat import routes as chat_routes
from backend.api.features.chat.routes import _strip_injected_context
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


def _mock_stream_internals(mocker: pytest_mock.MockerFixture):
    """Mock the async internals of stream_chat_post so tests can exercise
    validation and enrichment logic without needing RabbitMQ.

    Returns:
        A namespace with ``save`` and ``enqueue`` mock objects so
        callers can make additional assertions about side-effects.
    """
    import types

    mocker.patch(
        "backend.api.features.chat.routes._validate_and_get_session",
        return_value=None,
    )
    mock_save = mocker.patch(
        "backend.api.features.chat.routes.append_and_save_message",
        return_value=MagicMock(),  # non-None = message was saved (not a duplicate)
    )
    mock_registry = mocker.MagicMock()
    mock_registry.create_session = mocker.AsyncMock(return_value=None)
    mocker.patch(
        "backend.api.features.chat.routes.stream_registry",
        mock_registry,
    )
    mock_enqueue = mocker.patch(
        "backend.api.features.chat.routes.enqueue_copilot_turn",
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.chat.routes.track_user_message",
        return_value=None,
    )
    return types.SimpleNamespace(
        save=mock_save, enqueue=mock_enqueue, registry=mock_registry
    )


def test_stream_chat_accepts_20_file_ids(mocker: pytest_mock.MockerFixture):
    """Exactly 20 file_ids should be accepted (not rejected by validation)."""
    _mock_stream_internals(mocker)
    # Patch workspace lookup as imported by the routes module
    mocker.patch(
        "backend.data.workspace.get_or_create_workspace",
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


# ─── Duplicate message dedup ──────────────────────────────────────────


def test_stream_chat_skips_enqueue_for_duplicate_message(
    mocker: pytest_mock.MockerFixture,
):
    """When append_and_save_message returns None (duplicate detected),
    enqueue_copilot_turn and stream_registry.create_session must NOT be called
    to avoid double-processing and to prevent overwriting the active stream's
    turn_id in Redis (which would cause reconnecting clients to miss the response)."""
    mocks = _mock_stream_internals(mocker)
    # Override save to return None — signalling a duplicate
    mocks.save.return_value = None

    response = client.post(
        "/sessions/sess-1/stream",
        json={"message": "hello"},
    )
    assert response.status_code == 200
    mocks.enqueue.assert_not_called()
    mocks.registry.create_session.assert_not_called()


# ─── UUID format filtering ─────────────────────────────────────────────


def test_file_ids_filters_invalid_uuids(mocker: pytest_mock.MockerFixture):
    """Non-UUID strings in file_ids should be silently filtered out
    and NOT passed to the database query."""
    _mock_stream_internals(mocker)
    mocker.patch(
        "backend.data.workspace.get_or_create_workspace",
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


def test_file_ids_scoped_to_workspace(mocker: pytest_mock.MockerFixture):
    """The batch query should scope to the user's workspace."""
    _mock_stream_internals(mocker)
    mocker.patch(
        "backend.data.workspace.get_or_create_workspace",
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


def test_stream_chat_returns_429_on_daily_rate_limit(mocker: pytest_mock.MockerFixture):
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


def test_stream_chat_returns_429_on_weekly_rate_limit(
    mocker: pytest_mock.MockerFixture,
):
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


def test_stream_chat_429_includes_reset_time(mocker: pytest_mock.MockerFixture):
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


# ─── POST /stream queue-fallback (when a turn is already in flight) ──


def _mock_stream_queue_internals(
    mocker: pytest_mock.MockerFixture,
    *,
    session_exists: bool = True,
    turn_in_flight: bool = True,
    call_count: int = 1,
):
    """Mock dependencies for the POST /stream queue-fallback path.

    When ``turn_in_flight`` is True the handler takes the 202 queue branch.
    """
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
        "backend.api.features.chat.routes.is_turn_in_flight",
        new_callable=AsyncMock,
        return_value=turn_in_flight,
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
    mocker.patch(
        "backend.copilot.pending_message_helpers.get_redis_async",
        new_callable=AsyncMock,
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "backend.copilot.pending_message_helpers.incr_with_ttl",
        new_callable=AsyncMock,
        return_value=call_count,
    )
    mocker.patch(
        "backend.copilot.pending_message_helpers.push_pending_message",
        new_callable=AsyncMock,
        return_value=1,
    )
    # queue_user_message re-runs is_turn_in_flight via the helper module —
    # stub that path out too so we don't need a fake stream_registry.
    mocker.patch(
        "backend.copilot.pending_message_helpers.get_active_session_meta",
        new_callable=AsyncMock,
        return_value=None,
    )


def test_stream_queue_returns_202_when_turn_in_flight(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Happy path: POST /stream to a session with a live turn → 202 queue."""
    _mock_stream_queue_internals(mocker)

    response = client.post(
        "/sessions/sess-1/stream",
        json={"message": "follow-up", "is_user_message": True},
    )

    assert response.status_code == 202
    data = response.json()
    assert data["buffer_length"] == 1
    assert "turn_in_flight" in data


def test_stream_queue_session_not_found_returns_404(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """If the session doesn't exist or belong to the user, returns 404."""
    _mock_stream_queue_internals(mocker, session_exists=False)

    response = client.post(
        "/sessions/bad-sess/stream",
        json={"message": "hi", "is_user_message": True},
    )
    assert response.status_code == 404


def test_stream_queue_call_frequency_limit_returns_429(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Per-user call-frequency cap rejects rapid-fire queued pushes."""
    from backend.copilot.pending_message_helpers import PENDING_CALL_LIMIT

    _mock_stream_queue_internals(mocker, call_count=PENDING_CALL_LIMIT + 1)

    response = client.post(
        "/sessions/sess-1/stream",
        json={"message": "hi", "is_user_message": True},
    )
    assert response.status_code == 429
    assert "Too many queued message requests this minute" in response.json()["detail"]


def test_stream_queue_converts_context_dict_to_pending_context(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """StreamChatRequest.context is a raw dict; must be coerced to the
    typed PendingMessageContext before being pushed onto the buffer."""
    _mock_stream_queue_internals(mocker)
    queue_spy = mocker.patch(
        "backend.copilot.pending_message_helpers.queue_user_message",
        new_callable=AsyncMock,
    )
    from backend.copilot.pending_message_helpers import QueuePendingMessageResponse

    queue_spy.return_value = QueuePendingMessageResponse(
        buffer_length=1, max_buffer_length=10, turn_in_flight=True
    )

    response = client.post(
        "/sessions/sess-1/stream",
        json={
            "message": "hi",
            "is_user_message": True,
            "context": {"url": "https://example.test", "content": "body"},
        },
    )

    assert response.status_code == 202
    queue_spy.assert_awaited_once()
    kwargs = queue_spy.await_args.kwargs
    from backend.copilot.pending_messages import PendingMessageContext

    assert isinstance(kwargs["context"], PendingMessageContext)
    assert kwargs["context"].url == "https://example.test"
    assert kwargs["context"].content == "body"


def test_stream_queue_passes_none_context_when_omitted(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """When request.context is omitted, the queue call receives context=None."""
    _mock_stream_queue_internals(mocker)
    queue_spy = mocker.patch(
        "backend.copilot.pending_message_helpers.queue_user_message",
        new_callable=AsyncMock,
    )
    from backend.copilot.pending_message_helpers import QueuePendingMessageResponse

    queue_spy.return_value = QueuePendingMessageResponse(
        buffer_length=1, max_buffer_length=10, turn_in_flight=True
    )

    response = client.post(
        "/sessions/sess-1/stream",
        json={"message": "hi", "is_user_message": True},
    )

    assert response.status_code == 202
    queue_spy.assert_awaited_once()
    assert queue_spy.await_args.kwargs["context"] is None


# ─── get_pending_messages (GET /sessions/{session_id}/messages/pending) ─────


def test_get_pending_messages_returns_200_with_empty_buffer(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Happy path: no pending messages returns 200 with empty list."""
    mocker.patch(
        "backend.api.features.chat.routes._validate_and_get_session",
        new_callable=AsyncMock,
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "backend.api.features.chat.routes.peek_pending_messages",
        new_callable=AsyncMock,
        return_value=[],
    )

    response = client.get("/sessions/sess-1/messages/pending")

    assert response.status_code == 200
    data = response.json()
    assert data["messages"] == []
    assert data["count"] == 0


def test_get_pending_messages_returns_queued_messages(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Returns pending messages from buffer without consuming them."""
    mocker.patch(
        "backend.api.features.chat.routes._validate_and_get_session",
        new_callable=AsyncMock,
        return_value=mocker.MagicMock(),
    )
    mocker.patch(
        "backend.api.features.chat.routes.peek_pending_messages",
        new_callable=AsyncMock,
        return_value=[
            MagicMock(content="first message"),
            MagicMock(content="second message"),
        ],
    )

    response = client.get("/sessions/sess-1/messages/pending")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert data["messages"] == ["first message", "second message"]


def test_get_pending_messages_session_not_found_returns_404(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """If session does not exist or belongs to another user, returns 404."""
    mocker.patch(
        "backend.api.features.chat.routes._validate_and_get_session",
        side_effect=fastapi.HTTPException(status_code=404, detail="Session not found."),
    )

    response = client.get("/sessions/bad-sess/messages/pending")

    assert response.status_code == 404


class TestStripInjectedContext:
    """Unit tests for `_strip_injected_context` — the GET-side helper that
    hides the server-injected `<user_context>` block from API responses.

    The strip is intentionally exact-match: it only removes the prefix the
    inject helper writes (`<user_context>...</user_context>\\n\\n` at the very
    start of the message). Any drift between writer and reader leaves the raw
    block visible in the chat history, which is the failure mode this suite
    documents.
    """

    @staticmethod
    def _msg(role: str, content):
        return {"role": role, "content": content}

    def test_strips_well_formed_prefix(self) -> None:

        original = "<user_context>\nbiz ctx\n</user_context>\n\nhello world"
        result = _strip_injected_context(self._msg("user", original))
        assert result["content"] == "hello world"

    def test_passes_through_message_without_prefix(self) -> None:

        result = _strip_injected_context(self._msg("user", "just a question"))
        assert result["content"] == "just a question"

    def test_only_strips_when_prefix_is_at_start(self) -> None:
        """An embedded `<user_context>` block later in the message must NOT
        be stripped — only the leading prefix is server-injected."""

        content = (
            "I copied this from somewhere: <user_context>\nfoo\n</user_context>\n\n"
        )
        result = _strip_injected_context(self._msg("user", content))
        assert result["content"] == content

    def test_does_not_strip_with_only_single_newline_separator(self) -> None:
        """The strip regex requires `\\n\\n` after the closing tag — a single
        newline indicates a different format and must not be touched."""

        content = "<user_context>\nfoo\n</user_context>\nhello"
        result = _strip_injected_context(self._msg("user", content))
        assert result["content"] == content

    def test_assistant_messages_pass_through(self) -> None:

        original = "<user_context>\nfoo\n</user_context>\n\nhi"
        result = _strip_injected_context(self._msg("assistant", original))
        assert result["content"] == original

    def test_non_string_content_passes_through(self) -> None:
        """Multimodal / structured content (e.g. list of blocks) is not a
        string and must not be touched by the strip helper."""

        blocks = [{"type": "text", "text": "hello"}]
        result = _strip_injected_context(self._msg("user", blocks))
        assert result["content"] is blocks

    def test_strip_with_multiline_understanding(self) -> None:
        """The understanding payload spans multiple lines (markdown headings,
        bullet points). `re.DOTALL` must allow the regex to span them."""

        original = (
            "<user_context>\n"
            "# User Business Context\n\n"
            "## User\nName: Alice\n\n"
            "## Business\nCompany: Acme\n"
            "</user_context>\n\nactual question"
        )
        result = _strip_injected_context(self._msg("user", original))
        assert result["content"] == "actual question"

    def test_strip_when_message_is_only_the_prefix(self) -> None:
        """An empty user message gets injected with just the prefix; the
        strip should yield an empty string."""

        original = "<user_context>\nctx\n</user_context>\n\n"
        result = _strip_injected_context(self._msg("user", original))
        assert result["content"] == ""

    def test_does_not_mutate_original_dict(self) -> None:
        """The helper must return a copy — the original dict stays intact."""
        original_content = "<user_context>\nctx\n</user_context>\n\nhello"
        msg = self._msg("user", original_content)
        result = _strip_injected_context(msg)
        assert result["content"] == "hello"
        assert msg["content"] == original_content
        assert result is not msg

    def test_no_role_field_does_not_crash(self) -> None:

        msg = {"content": "hello"}
        result = _strip_injected_context(msg)
        # Without a role, the helper short-circuits without touching content.
        assert result["content"] == "hello"


# ─── DELETE /sessions/{id}/stream — disconnect listeners ──────────────


def test_disconnect_stream_returns_204_and_awaits_registry(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mock_session = MagicMock()
    mocker.patch(
        "backend.api.features.chat.routes.get_chat_session",
        new_callable=AsyncMock,
        return_value=mock_session,
    )
    mock_disconnect = mocker.patch(
        "backend.api.features.chat.routes.stream_registry.disconnect_all_listeners",
        new_callable=AsyncMock,
        return_value=2,
    )

    response = client.delete("/sessions/sess-1/stream")

    assert response.status_code == 204
    mock_disconnect.assert_awaited_once_with("sess-1")


def test_disconnect_stream_returns_404_when_session_missing(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    mocker.patch(
        "backend.api.features.chat.routes.get_chat_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    mock_disconnect = mocker.patch(
        "backend.api.features.chat.routes.stream_registry.disconnect_all_listeners",
        new_callable=AsyncMock,
    )

    response = client.delete("/sessions/unknown-session/stream")

    assert response.status_code == 404
    mock_disconnect.assert_not_awaited()


# ─── GET /sessions/{session_id} — backward pagination ─────────────────────────


def _make_paginated_messages(
    mocker: pytest_mock.MockerFixture, *, has_more: bool = False
):
    """Return a mock PaginatedMessages and configure the DB patch."""
    from datetime import UTC, datetime

    from backend.copilot.db import PaginatedMessages
    from backend.copilot.model import ChatMessage, ChatSessionInfo, ChatSessionMetadata

    now = datetime.now(UTC)
    session_info = ChatSessionInfo(
        session_id="sess-1",
        user_id=TEST_USER_ID,
        usage=[],
        started_at=now,
        updated_at=now,
        metadata=ChatSessionMetadata(),
    )
    page = PaginatedMessages(
        messages=[ChatMessage(role="user", content="hello", sequence=0)],
        has_more=has_more,
        oldest_sequence=0,
        session=session_info,
    )
    mock_paginate = mocker.patch(
        "backend.api.features.chat.routes.get_chat_messages_paginated",
        new_callable=AsyncMock,
        return_value=page,
    )
    return page, mock_paginate


def test_get_session_returns_backward_paginated(
    mocker: pytest_mock.MockerFixture,
    test_user_id: str,
) -> None:
    """All sessions use backward (newest-first) pagination."""
    _make_paginated_messages(mocker)
    mocker.patch(
        "backend.api.features.chat.routes.stream_registry.get_active_session",
        new_callable=AsyncMock,
        return_value=(None, None),
    )

    response = client.get("/sessions/sess-1")

    assert response.status_code == 200
    data = response.json()
    assert data["oldest_sequence"] == 0
    assert "forward_paginated" not in data
    assert "newest_sequence" not in data
