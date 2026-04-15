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
from backend.util.exceptions import NotFoundError

app = fastapi.FastAPI()
app.include_router(chat_routes.router)


@app.exception_handler(NotFoundError)
async def _not_found_handler(
    request: fastapi.Request, exc: NotFoundError
) -> fastapi.responses.JSONResponse:
    """Mirror the production NotFoundError → 404 mapping from the REST app."""
    return fastapi.responses.JSONResponse(status_code=404, content={"detail": str(exc)})


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


# ─── message max_length validation ───────────────────────────────────


def test_stream_chat_rejects_too_long_message():
    """A message exceeding max_length=64_000 must be rejected (422)."""
    response = client.post(
        "/sessions/sess-1/stream",
        json={
            "message": "x" * 64_001,
        },
    )
    assert response.status_code == 422


def test_stream_chat_accepts_exactly_max_length_message(
    mocker: pytest_mock.MockFixture,
):
    """A message exactly at max_length=64_000 must be accepted."""
    _mock_stream_internals(mocker)
    mocker.patch(
        "backend.api.features.chat.routes.get_global_rate_limits",
        new_callable=AsyncMock,
        return_value=(0, 0, SubscriptionTier.FREE),
    )

    response = client.post(
        "/sessions/sess-1/stream",
        json={
            "message": "x" * 64_000,
        },
    )
    assert response.status_code == 200


# ─── list_sessions ────────────────────────────────────────────────────


def _make_session_info(session_id: str = "sess-1", title: str | None = "Test"):
    """Build a minimal ChatSessionInfo-like mock."""
    from backend.copilot.model import ChatSessionInfo, ChatSessionMetadata

    return ChatSessionInfo(
        session_id=session_id,
        user_id=TEST_USER_ID,
        title=title,
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(),
    )


def test_list_sessions_returns_sessions(mocker: pytest_mock.MockerFixture) -> None:
    """GET /sessions returns list of sessions with is_processing=False when Redis OK."""
    session = _make_session_info("sess-abc")
    mocker.patch(
        "backend.api.features.chat.routes.get_user_sessions",
        new_callable=AsyncMock,
        return_value=([session], 1),
    )
    # Redis pipeline returns "done" (not "running") for this session
    mock_redis = MagicMock()
    mock_pipe = MagicMock()
    mock_pipe.hget = MagicMock(return_value=None)
    mock_pipe.execute = AsyncMock(return_value=["done"])
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)
    mocker.patch(
        "backend.api.features.chat.routes.get_redis_async",
        new_callable=AsyncMock,
        return_value=mock_redis,
    )

    response = client.get("/sessions")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    assert len(data["sessions"]) == 1
    assert data["sessions"][0]["id"] == "sess-abc"
    assert data["sessions"][0]["is_processing"] is False


def test_list_sessions_marks_running_as_processing(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Sessions with Redis status='running' should have is_processing=True."""
    session = _make_session_info("sess-xyz")
    mocker.patch(
        "backend.api.features.chat.routes.get_user_sessions",
        new_callable=AsyncMock,
        return_value=([session], 1),
    )
    mock_redis = MagicMock()
    mock_pipe = MagicMock()
    mock_pipe.hget = MagicMock(return_value=None)
    mock_pipe.execute = AsyncMock(return_value=["running"])
    mock_redis.pipeline = MagicMock(return_value=mock_pipe)
    mocker.patch(
        "backend.api.features.chat.routes.get_redis_async",
        new_callable=AsyncMock,
        return_value=mock_redis,
    )

    response = client.get("/sessions")

    assert response.status_code == 200
    assert response.json()["sessions"][0]["is_processing"] is True


def test_list_sessions_redis_failure_defaults_to_not_processing(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Redis failures must be swallowed and sessions default to is_processing=False."""
    session = _make_session_info("sess-fallback")
    mocker.patch(
        "backend.api.features.chat.routes.get_user_sessions",
        new_callable=AsyncMock,
        return_value=([session], 1),
    )
    mocker.patch(
        "backend.api.features.chat.routes.get_redis_async",
        side_effect=Exception("Redis down"),
    )

    response = client.get("/sessions")

    assert response.status_code == 200
    assert response.json()["sessions"][0]["is_processing"] is False


def test_list_sessions_empty(mocker: pytest_mock.MockerFixture) -> None:
    """GET /sessions with no sessions returns empty list without hitting Redis."""
    mocker.patch(
        "backend.api.features.chat.routes.get_user_sessions",
        new_callable=AsyncMock,
        return_value=([], 0),
    )

    response = client.get("/sessions")

    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 0
    assert data["sessions"] == []


# ─── delete_session ───────────────────────────────────────────────────


def test_delete_session_success(mocker: pytest_mock.MockerFixture) -> None:
    """DELETE /sessions/{id} returns 204 when deleted successfully."""
    mocker.patch(
        "backend.api.features.chat.routes.delete_chat_session",
        new_callable=AsyncMock,
        return_value=True,
    )
    # Patch use_e2b_sandbox env-var to disable E2B so the route skips sandbox cleanup.
    # Patching the Pydantic property directly doesn't work (Pydantic v2 intercepts
    # attribute setting on BaseSettings instances and raises AttributeError).
    mocker.patch.dict("os.environ", {"USE_E2B_SANDBOX": "false"})

    response = client.delete("/sessions/sess-1")

    assert response.status_code == 204


def test_delete_session_not_found(mocker: pytest_mock.MockerFixture) -> None:
    """DELETE /sessions/{id} returns 404 when session not found or not owned."""
    mocker.patch(
        "backend.api.features.chat.routes.delete_chat_session",
        new_callable=AsyncMock,
        return_value=False,
    )

    response = client.delete("/sessions/sess-missing")

    assert response.status_code == 404


# ─── get_session ──────────────────────────────────────────────────────


def _make_paginated_messages(
    session_id: str = "sess-1",
    messages=None,
    has_more: bool = False,
    oldest_sequence: int | None = None,
):
    """Build a minimal PaginatedMessages-like object."""
    from backend.copilot.db import PaginatedMessages
    from backend.copilot.model import ChatSessionInfo, ChatSessionMetadata

    session_info = ChatSessionInfo(
        session_id=session_id,
        user_id=TEST_USER_ID,
        title=None,
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(),
    )
    return PaginatedMessages(
        messages=messages or [],
        has_more=has_more,
        oldest_sequence=oldest_sequence,
        session=session_info,
    )


def test_get_session_not_found(mocker: pytest_mock.MockerFixture) -> None:
    """GET /sessions/{id} returns 404 when session is not found."""
    mocker.patch(
        "backend.api.features.chat.routes.get_chat_messages_paginated",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.get("/sessions/sess-missing")

    assert response.status_code == 404


def test_get_session_initial_load(mocker: pytest_mock.MockerFixture) -> None:
    """GET /sessions/{id} without pagination returns full session details."""
    page = _make_paginated_messages("sess-1")
    mocker.patch(
        "backend.api.features.chat.routes.get_chat_messages_paginated",
        new_callable=AsyncMock,
        return_value=page,
    )
    # No active stream
    mock_registry = MagicMock()
    mock_registry.get_active_session = AsyncMock(return_value=(None, None))
    mocker.patch("backend.api.features.chat.routes.stream_registry", mock_registry)

    response = client.get("/sessions/sess-1")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "sess-1"
    assert data["active_stream"] is None
    assert data["has_more_messages"] is False


def test_get_session_with_active_stream(mocker: pytest_mock.MockerFixture) -> None:
    """GET /sessions/{id} with active stream populates active_stream field."""
    from backend.copilot.stream_registry import ActiveSession

    page = _make_paginated_messages("sess-1")
    mocker.patch(
        "backend.api.features.chat.routes.get_chat_messages_paginated",
        new_callable=AsyncMock,
        return_value=page,
    )
    active_session = ActiveSession(
        session_id="sess-1",
        user_id=TEST_USER_ID,
        tool_call_id="chat_stream",
        tool_name="chat",
        turn_id="turn-abc",
        status="running",
    )
    mock_registry = MagicMock()
    mock_registry.get_active_session = AsyncMock(return_value=(active_session, "1-1"))
    mocker.patch("backend.api.features.chat.routes.stream_registry", mock_registry)

    response = client.get("/sessions/sess-1")

    assert response.status_code == 200
    data = response.json()
    assert data["active_stream"] is not None
    assert data["active_stream"]["turn_id"] == "turn-abc"


def test_get_session_load_more_skips_active_stream(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """GET /sessions/{id}?before_sequence=10 skips active_stream check."""
    page = _make_paginated_messages("sess-1", has_more=True, oldest_sequence=5)
    mocker.patch(
        "backend.api.features.chat.routes.get_chat_messages_paginated",
        new_callable=AsyncMock,
        return_value=page,
    )
    mock_registry = MagicMock()
    mocker.patch("backend.api.features.chat.routes.stream_registry", mock_registry)

    response = client.get("/sessions/sess-1?before_sequence=10")

    assert response.status_code == 200
    data = response.json()
    assert data["active_stream"] is None
    assert data["has_more_messages"] is True
    assert data["oldest_sequence"] == 5
    # get_active_session should NOT be called for load-more requests
    mock_registry.get_active_session.assert_not_called()


# ─── cancel_session_task ──────────────────────────────────────────────


def _mock_validate_session(
    mocker: pytest_mock.MockerFixture, *, session_id: str = "sess-1"
):
    """Mock _validate_and_get_session to return a dummy session."""
    from backend.copilot.model import ChatSession

    dummy = ChatSession.new(TEST_USER_ID, dry_run=False)
    mocker.patch(
        "backend.api.features.chat.routes._validate_and_get_session",
        new_callable=AsyncMock,
        return_value=dummy,
    )


def test_cancel_session_no_active_task(mocker: pytest_mock.MockerFixture) -> None:
    """Cancel returns cancelled=True with reason when no stream is active."""
    _mock_validate_session(mocker)
    mock_registry = MagicMock()
    mock_registry.get_active_session = AsyncMock(return_value=(None, None))
    mocker.patch("backend.api.features.chat.routes.stream_registry", mock_registry)

    response = client.post("/sessions/sess-1/cancel")

    assert response.status_code == 200
    data = response.json()
    assert data["cancelled"] is True
    assert data["reason"] == "no_active_session"


def test_cancel_session_enqueues_cancel_and_confirms(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Cancel enqueues cancel task and returns cancelled=True once stream stops."""
    from backend.copilot.stream_registry import ActiveSession

    _mock_validate_session(mocker)
    active_session = ActiveSession(
        session_id="sess-1",
        user_id=TEST_USER_ID,
        tool_call_id="chat_stream",
        tool_name="chat",
        turn_id="turn-1",
        status="running",
    )
    stopped_session = ActiveSession(
        session_id="sess-1",
        user_id=TEST_USER_ID,
        tool_call_id="chat_stream",
        tool_name="chat",
        turn_id="turn-1",
        status="completed",
    )
    mock_registry = MagicMock()
    mock_registry.get_active_session = AsyncMock(return_value=(active_session, "1-0"))
    mock_registry.get_session = AsyncMock(return_value=stopped_session)
    mocker.patch("backend.api.features.chat.routes.stream_registry", mock_registry)
    mock_enqueue = mocker.patch(
        "backend.api.features.chat.routes.enqueue_cancel_task",
        new_callable=AsyncMock,
    )

    response = client.post("/sessions/sess-1/cancel")

    assert response.status_code == 200
    assert response.json()["cancelled"] is True
    mock_enqueue.assert_called_once_with("sess-1")


# ─── session_assign_user ──────────────────────────────────────────────


def test_session_assign_user(mocker: pytest_mock.MockerFixture) -> None:
    """PATCH /sessions/{id}/assign-user calls assign_user_to_session and returns ok."""
    mock_assign = mocker.patch(
        "backend.api.features.chat.routes.chat_service.assign_user_to_session",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.patch("/sessions/sess-1/assign-user")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    mock_assign.assert_called_once_with("sess-1", TEST_USER_ID)


# ─── get_ttl_config ──────────────────────────────────────────────────


def test_get_ttl_config(mocker: pytest_mock.MockerFixture) -> None:
    """GET /config/ttl returns correct TTL values derived from config."""
    mocker.patch.object(chat_routes.config, "stream_ttl", 300)

    response = client.get("/config/ttl")

    assert response.status_code == 200
    data = response.json()
    assert data["stream_ttl_seconds"] == 300
    assert data["stream_ttl_ms"] == 300_000


# ─── reset_copilot_usage ──────────────────────────────────────────────


def _mock_reset_internals(
    mocker: pytest_mock.MockerFixture,
    *,
    cost: int = 100,
    enable_credit: bool = True,
    daily_limit: int = 10_000,
    weekly_limit: int = 50_000,
    tier: "SubscriptionTier" = SubscriptionTier.FREE,
    daily_used: int = 10_001,
    weekly_used: int = 1_000,
    reset_count: int | None = 0,
    acquire_lock: bool = True,
    reset_daily: bool = True,
    remaining_balance: int = 9_000,
):
    """Set up all dependencies for reset_copilot_usage tests."""
    from backend.copilot.rate_limit import CoPilotUsageStatus, UsageWindow

    mocker.patch.object(chat_routes.config, "rate_limit_reset_cost", cost)
    mocker.patch.object(chat_routes.config, "max_daily_resets", 3)
    mocker.patch.object(chat_routes.settings.config, "enable_credit", enable_credit)

    mocker.patch(
        "backend.api.features.chat.routes.get_global_rate_limits",
        new_callable=AsyncMock,
        return_value=(daily_limit, weekly_limit, tier),
    )
    resets_at = datetime.now(UTC) + timedelta(hours=1)
    status = CoPilotUsageStatus(
        daily=UsageWindow(used=daily_used, limit=daily_limit, resets_at=resets_at),
        weekly=UsageWindow(used=weekly_used, limit=weekly_limit, resets_at=resets_at),
    )
    mocker.patch(
        "backend.api.features.chat.routes.get_usage_status",
        new_callable=AsyncMock,
        return_value=status,
    )
    mocker.patch(
        "backend.api.features.chat.routes.get_daily_reset_count",
        new_callable=AsyncMock,
        return_value=reset_count,
    )
    mocker.patch(
        "backend.api.features.chat.routes.acquire_reset_lock",
        new_callable=AsyncMock,
        return_value=acquire_lock,
    )
    mocker.patch(
        "backend.api.features.chat.routes.release_reset_lock",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "backend.api.features.chat.routes.reset_daily_usage",
        new_callable=AsyncMock,
        return_value=reset_daily,
    )
    mocker.patch(
        "backend.api.features.chat.routes.increment_daily_reset_count",
        new_callable=AsyncMock,
    )

    mock_credit_model = MagicMock()
    mock_credit_model.spend_credits = AsyncMock(return_value=remaining_balance)
    mock_credit_model.top_up_credits = AsyncMock(return_value=None)
    mocker.patch(
        "backend.api.features.chat.routes.get_user_credit_model",
        new_callable=AsyncMock,
        return_value=mock_credit_model,
    )
    return mock_credit_model


def test_reset_usage_returns_400_when_cost_is_zero(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """POST /usage/reset returns 400 when rate_limit_reset_cost <= 0."""
    mocker.patch.object(chat_routes.config, "rate_limit_reset_cost", 0)

    response = client.post("/usage/reset")

    assert response.status_code == 400
    assert "not available" in response.json()["detail"].lower()


def test_reset_usage_returns_400_when_credits_disabled(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """POST /usage/reset returns 400 when credit system is disabled."""
    mocker.patch.object(chat_routes.config, "rate_limit_reset_cost", 100)
    mocker.patch.object(chat_routes.settings.config, "enable_credit", False)

    response = client.post("/usage/reset")

    assert response.status_code == 400
    assert "disabled" in response.json()["detail"].lower()


def test_reset_usage_returns_400_when_no_daily_limit(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """POST /usage/reset returns 400 when daily_limit is 0."""
    mocker.patch.object(chat_routes.config, "rate_limit_reset_cost", 100)
    mocker.patch.object(chat_routes.settings.config, "enable_credit", True)
    mocker.patch(
        "backend.api.features.chat.routes.get_global_rate_limits",
        new_callable=AsyncMock,
        return_value=(0, 50_000, SubscriptionTier.FREE),
    )
    mocker.patch(
        "backend.api.features.chat.routes.get_daily_reset_count",
        new_callable=AsyncMock,
        return_value=0,
    )

    response = client.post("/usage/reset")

    assert response.status_code == 400
    assert "nothing to reset" in response.json()["detail"].lower()


def test_reset_usage_returns_503_when_redis_unavailable(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """POST /usage/reset returns 503 when Redis is unavailable for reset count."""
    mocker.patch.object(chat_routes.config, "rate_limit_reset_cost", 100)
    mocker.patch.object(chat_routes.settings.config, "enable_credit", True)
    mocker.patch(
        "backend.api.features.chat.routes.get_global_rate_limits",
        new_callable=AsyncMock,
        return_value=(10_000, 50_000, SubscriptionTier.FREE),
    )
    mocker.patch(
        "backend.api.features.chat.routes.get_daily_reset_count",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = client.post("/usage/reset")

    assert response.status_code == 503


def test_reset_usage_returns_429_when_max_resets_reached(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """POST /usage/reset returns 429 when max daily resets exceeded."""
    mocker.patch.object(chat_routes.config, "rate_limit_reset_cost", 100)
    mocker.patch.object(chat_routes.config, "max_daily_resets", 2)
    mocker.patch.object(chat_routes.settings.config, "enable_credit", True)
    mocker.patch(
        "backend.api.features.chat.routes.get_global_rate_limits",
        new_callable=AsyncMock,
        return_value=(10_000, 50_000, SubscriptionTier.FREE),
    )
    mocker.patch(
        "backend.api.features.chat.routes.get_daily_reset_count",
        new_callable=AsyncMock,
        return_value=2,
    )

    response = client.post("/usage/reset")

    assert response.status_code == 429
    assert "resets" in response.json()["detail"].lower()


def test_reset_usage_returns_429_when_lock_not_acquired(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """POST /usage/reset returns 429 when a concurrent reset is in progress."""
    mocker.patch.object(chat_routes.config, "rate_limit_reset_cost", 100)
    mocker.patch.object(chat_routes.config, "max_daily_resets", 3)
    mocker.patch.object(chat_routes.settings.config, "enable_credit", True)
    mocker.patch(
        "backend.api.features.chat.routes.get_global_rate_limits",
        new_callable=AsyncMock,
        return_value=(10_000, 50_000, SubscriptionTier.FREE),
    )
    mocker.patch(
        "backend.api.features.chat.routes.get_daily_reset_count",
        new_callable=AsyncMock,
        return_value=0,
    )
    mocker.patch(
        "backend.api.features.chat.routes.acquire_reset_lock",
        new_callable=AsyncMock,
        return_value=False,
    )

    response = client.post("/usage/reset")

    assert response.status_code == 429
    assert "in progress" in response.json()["detail"].lower()


def test_reset_usage_returns_400_when_limit_not_reached(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """POST /usage/reset returns 400 when daily limit has not been reached."""
    _mock_reset_internals(mocker, daily_used=500, daily_limit=10_000)
    mocker.patch(
        "backend.api.features.chat.routes.release_reset_lock",
        new_callable=AsyncMock,
    )

    response = client.post("/usage/reset")

    assert response.status_code == 400
    assert "not reached" in response.json()["detail"].lower()


def test_reset_usage_returns_400_when_weekly_also_exhausted(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """POST /usage/reset returns 400 when weekly limit is also exhausted."""
    _mock_reset_internals(
        mocker,
        daily_used=10_001,
        daily_limit=10_000,
        weekly_used=50_001,
        weekly_limit=50_000,
    )
    mocker.patch(
        "backend.api.features.chat.routes.release_reset_lock",
        new_callable=AsyncMock,
    )

    response = client.post("/usage/reset")

    assert response.status_code == 400
    assert "weekly" in response.json()["detail"].lower()


def test_reset_usage_returns_402_when_insufficient_credits(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """POST /usage/reset returns 402 when credits are insufficient."""
    from backend.util.exceptions import InsufficientBalanceError

    mock_credit = _mock_reset_internals(mocker)
    mock_credit.spend_credits = AsyncMock(
        side_effect=InsufficientBalanceError(
            message="Insufficient balance",
            user_id=TEST_USER_ID,
            balance=0.0,
            amount=100.0,
        )
    )
    mocker.patch(
        "backend.api.features.chat.routes.release_reset_lock",
        new_callable=AsyncMock,
    )

    response = client.post("/usage/reset")

    assert response.status_code == 402


def test_reset_usage_success(mocker: pytest_mock.MockerFixture) -> None:
    """POST /usage/reset returns 200 with updated usage on success."""
    _mock_reset_internals(mocker, remaining_balance=8_900)

    response = client.post("/usage/reset")

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["credits_charged"] == 100
    assert data["remaining_balance"] == 8_900
    assert "daily" in data["usage"]
    assert "weekly" in data["usage"]


def test_reset_usage_refunds_on_redis_failure(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """POST /usage/reset returns 503 and refunds credits when Redis reset fails."""
    mock_credit = _mock_reset_internals(mocker, reset_daily=False)

    response = client.post("/usage/reset")

    assert response.status_code == 503
    # Credits should be refunded via top_up_credits
    mock_credit.top_up_credits.assert_called_once()


# ─── resume_session_stream ───────────────────────────────────────────


def test_resume_session_stream_no_active_session(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """GET /sessions/{id}/stream returns 204 when no active session."""
    mock_registry = MagicMock()
    mock_registry.get_active_session = AsyncMock(return_value=(None, None))
    mocker.patch("backend.api.features.chat.routes.stream_registry", mock_registry)

    response = client.get("/sessions/sess-1/stream")

    assert response.status_code == 204


def test_resume_session_stream_no_subscriber_queue(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """GET /sessions/{id}/stream returns 204 when subscribe_to_session returns None."""
    from backend.copilot.stream_registry import ActiveSession

    active_session = ActiveSession(
        session_id="sess-1",
        user_id=TEST_USER_ID,
        tool_call_id="chat_stream",
        tool_name="chat",
        turn_id="turn-1",
        status="running",
    )
    mock_registry = MagicMock()
    mock_registry.get_active_session = AsyncMock(return_value=(active_session, "1-0"))
    mock_registry.subscribe_to_session = AsyncMock(return_value=None)
    mocker.patch("backend.api.features.chat.routes.stream_registry", mock_registry)

    response = client.get("/sessions/sess-1/stream")

    assert response.status_code == 204
