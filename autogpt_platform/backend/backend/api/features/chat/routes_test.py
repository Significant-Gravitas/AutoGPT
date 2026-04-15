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


def _mock_stream_internals(
    mocker: pytest_mock.MockerFixture,
    *,
    redis_set_returns: object = True,
):
    """Mock the async internals of stream_chat_post so tests can exercise
    validation and enrichment logic without needing Redis/RabbitMQ.

    Args:
        redis_set_returns: Value returned by the mocked Redis ``set`` call.
            ``True`` (default) simulates a fresh key (new message);
            ``None`` simulates a collision (duplicate blocked).

    Returns:
        A namespace with ``redis``, ``save``, and ``enqueue`` mock objects so
        callers can make additional assertions about side-effects.
    """
    import types

    mocker.patch(
        "backend.api.features.chat.routes._validate_and_get_session",
        return_value=None,
    )
    mock_save = mocker.patch(
        "backend.api.features.chat.routes.append_and_save_message",
        return_value=None,
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
    mock_redis = AsyncMock()
    mock_redis.set = AsyncMock(return_value=redis_set_returns)
    mocker.patch(
        "backend.copilot.message_dedup.get_redis_async",
        new_callable=AsyncMock,
        return_value=mock_redis,
    )
    ns = types.SimpleNamespace(redis=mock_redis, save=mock_save, enqueue=mock_enqueue)
    return ns


def test_stream_chat_accepts_20_file_ids(mocker: pytest_mock.MockerFixture):
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


def test_file_ids_filters_invalid_uuids(mocker: pytest_mock.MockerFixture):
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


def test_file_ids_scoped_to_workspace(mocker: pytest_mock.MockerFixture):
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


# ─── Idempotency / duplicate-POST guard ──────────────────────────────


def test_stream_chat_blocks_duplicate_post_returns_empty_sse(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """A second POST with the same message within the 30-s window must return
    an empty SSE stream (StreamFinish + [DONE]) so the frontend marks the
    turn complete without creating a ghost response."""
    # redis_set_returns=None simulates a collision: the NX key already exists.
    ns = _mock_stream_internals(mocker, redis_set_returns=None)

    response = client.post(
        "/sessions/sess-dup/stream",
        json={"message": "duplicate message", "is_user_message": True},
    )

    assert response.status_code == 200
    body = response.text
    # The response must contain StreamFinish (type=finish) and the SSE [DONE] terminator.
    assert '"finish"' in body
    assert "[DONE]" in body
    # The empty SSE response must include the AI SDK protocol header so the
    # frontend treats it as a valid stream and marks the turn complete.
    assert response.headers.get("x-vercel-ai-ui-message-stream") == "v1"
    # The duplicate guard must prevent save/enqueue side effects.
    ns.save.assert_not_called()
    ns.enqueue.assert_not_called()


def test_stream_chat_first_post_proceeds_normally(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """The first POST (Redis NX key set successfully) must proceed through the
    normal streaming path — no early return."""
    ns = _mock_stream_internals(mocker, redis_set_returns=True)

    response = client.post(
        "/sessions/sess-new/stream",
        json={"message": "first message", "is_user_message": True},
    )

    assert response.status_code == 200
    # Redis set must have been called once with the NX flag.
    ns.redis.set.assert_called_once()
    call_kwargs = ns.redis.set.call_args
    assert call_kwargs.kwargs.get("nx") is True


def test_stream_chat_dedup_skipped_for_non_user_messages(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """System/assistant messages (is_user_message=False) bypass the dedup
    guard — they are injected programmatically and must always be processed."""
    ns = _mock_stream_internals(mocker, redis_set_returns=None)

    response = client.post(
        "/sessions/sess-sys/stream",
        json={"message": "system context", "is_user_message": False},
    )

    # Even though redis_set_returns=None (would block a user message),
    # the endpoint must proceed because is_user_message=False.
    assert response.status_code == 200
    ns.redis.set.assert_not_called()


def test_stream_chat_dedup_hash_uses_original_message_not_mutated(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """The dedup hash must be computed from the original request message,
    not the mutated version that has the [Attached files] block appended.
    A file_id is sent so the route actually appends the [Attached files] block,
    exercising the mutation path — the hash must still match the original text."""
    import hashlib

    ns = _mock_stream_internals(mocker, redis_set_returns=True)

    file_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    # Mock workspace + prisma so the attachment block is actually appended.
    mocker.patch(
        "backend.api.features.chat.routes.get_or_create_workspace",
        return_value=type("W", (), {"id": "ws-1"})(),
    )
    fake_file = type(
        "F",
        (),
        {
            "id": file_id,
            "name": "doc.pdf",
            "mimeType": "application/pdf",
            "sizeBytes": 1024,
        },
    )()
    mock_prisma = mocker.MagicMock()
    mock_prisma.find_many = mocker.AsyncMock(return_value=[fake_file])
    mocker.patch(
        "prisma.models.UserWorkspaceFile.prisma",
        return_value=mock_prisma,
    )

    response = client.post(
        "/sessions/sess-hash/stream",
        json={
            "message": "plain message",
            "is_user_message": True,
            "file_ids": [file_id],
        },
    )

    assert response.status_code == 200
    ns.redis.set.assert_called_once()
    call_args = ns.redis.set.call_args
    dedup_key = call_args.args[0]

    # Hash must use the original message + sorted file IDs, not the mutated text.
    expected_hash = hashlib.sha256(
        f"sess-hash:plain message:{file_id}".encode()
    ).hexdigest()[:16]
    expected_key = f"chat:msg_dedup:sess-hash:{expected_hash}"
    assert dedup_key == expected_key, (
        f"Dedup key {dedup_key!r} does not match expected {expected_key!r} — "
        "hash may be using mutated message or wrong inputs"
    )


def test_stream_chat_dedup_key_released_after_stream_finish(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """The dedup Redis key must be deleted after the turn completes (when
    subscriber_queue is None the route yields StreamFinish immediately and
    should release the key so the user can re-send the same message)."""
    from unittest.mock import AsyncMock as _AsyncMock

    # Set up all internals manually so we can control subscribe_to_session.
    mocker.patch(
        "backend.api.features.chat.routes._validate_and_get_session",
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.chat.routes.append_and_save_message",
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.chat.routes.enqueue_copilot_turn",
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.chat.routes.track_user_message",
        return_value=None,
    )
    mock_registry = mocker.MagicMock()
    mock_registry.create_session = _AsyncMock(return_value=None)
    # None → early-finish path: StreamFinish yielded immediately, dedup key released.
    mock_registry.subscribe_to_session = _AsyncMock(return_value=None)
    mocker.patch(
        "backend.api.features.chat.routes.stream_registry",
        mock_registry,
    )
    mock_redis = mocker.AsyncMock()
    mock_redis.set = _AsyncMock(return_value=True)
    mocker.patch(
        "backend.copilot.message_dedup.get_redis_async",
        new_callable=_AsyncMock,
        return_value=mock_redis,
    )

    response = client.post(
        "/sessions/sess-finish/stream",
        json={"message": "hello", "is_user_message": True},
    )

    assert response.status_code == 200
    body = response.text
    assert '"finish"' in body
    # The dedup key must be released so intentional re-sends are allowed.
    mock_redis.delete.assert_called_once()


def test_stream_chat_dedup_key_released_even_when_redis_delete_raises(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """The route must not crash when the dedup Redis delete fails on the
    subscriber_queue-is-None early-finish path (except Exception: pass)."""
    from unittest.mock import AsyncMock as _AsyncMock

    mocker.patch(
        "backend.api.features.chat.routes._validate_and_get_session",
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.chat.routes.append_and_save_message",
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.chat.routes.enqueue_copilot_turn",
        return_value=None,
    )
    mocker.patch(
        "backend.api.features.chat.routes.track_user_message",
        return_value=None,
    )
    mock_registry = mocker.MagicMock()
    mock_registry.create_session = _AsyncMock(return_value=None)
    mock_registry.subscribe_to_session = _AsyncMock(return_value=None)
    mocker.patch(
        "backend.api.features.chat.routes.stream_registry",
        mock_registry,
    )
    mock_redis = mocker.AsyncMock()
    mock_redis.set = _AsyncMock(return_value=True)
    # Make the delete raise so the except-pass branch is exercised.
    mock_redis.delete = _AsyncMock(side_effect=RuntimeError("redis gone"))
    mocker.patch(
        "backend.copilot.message_dedup.get_redis_async",
        new_callable=_AsyncMock,
        return_value=mock_redis,
    )

    # Should not raise even though delete fails.
    response = client.post(
        "/sessions/sess-finish-err/stream",
        json={"message": "hello", "is_user_message": True},
    )

    assert response.status_code == 200
    assert '"finish"' in response.text
    # delete must have been attempted — the except-pass branch silenced the error.
    mock_redis.delete.assert_called_once()


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
