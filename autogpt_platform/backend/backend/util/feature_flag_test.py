import datetime
import logging
import uuid

import pytest
from fastapi import HTTPException
from ldclient import LDClient

from backend.util.feature_flag import (
    Flag,
    _env_flag_override,
    _fetch_user_context_data,
    feature_flag,
    is_feature_enabled,
    mock_flag_variation,
)


@pytest.fixture
def ld_client(mocker):
    client = mocker.Mock(spec=LDClient)
    mocker.patch("ldclient.get", return_value=client)
    client.is_initialized.return_value = True
    return client


@pytest.mark.asyncio
async def test_feature_flag_enabled(ld_client):
    ld_client.variation.return_value = True

    @feature_flag("test-flag")
    async def test_function(user_id: str):
        return "success"

    result = await test_function(user_id="test-user")
    assert result == "success"
    ld_client.variation.assert_called_once()


@pytest.mark.asyncio
async def test_feature_flag_unauthorized_response(ld_client):
    ld_client.variation.return_value = False

    @feature_flag("test-flag")
    async def test_function(user_id: str):
        return "success"

    with pytest.raises(HTTPException) as exc_info:
        await test_function(user_id="test-user")
    assert exc_info.value.status_code == 404


def test_mock_flag_variation(ld_client):
    with mock_flag_variation("test-flag", True):
        assert ld_client.variation("test-flag", None, False) is True

    with mock_flag_variation("test-flag", False):
        assert ld_client.variation("test-flag", None, True) is False


@pytest.mark.asyncio
async def test_is_feature_enabled(ld_client):
    """Test the is_feature_enabled helper function."""
    ld_client.is_initialized.return_value = True
    ld_client.variation.return_value = True

    result = await is_feature_enabled(Flag.AUTOMOD, "user123", default=False)
    assert result is True

    ld_client.variation.assert_called_once()
    call_args = ld_client.variation.call_args
    assert call_args[0][0] == "AutoMod"  # flag_key
    assert call_args[0][2] is False  # default value


@pytest.mark.asyncio
async def test_is_feature_enabled_not_initialized(ld_client):
    """Test is_feature_enabled when LaunchDarkly is not initialized."""
    ld_client.is_initialized.return_value = False

    result = await is_feature_enabled(Flag.AGENT_ACTIVITY, "user123", default=True)
    assert result is True  # Should return default

    ld_client.variation.assert_not_called()


@pytest.mark.asyncio
async def test_is_feature_enabled_exception(mocker):
    """Test is_feature_enabled when get_client() raises an exception."""
    mocker.patch(
        "backend.util.feature_flag.get_client",
        side_effect=Exception("Client error"),
    )

    result = await is_feature_enabled(Flag.AGENT_ACTIVITY, "user123", default=True)
    assert result is True  # Should return default


def test_flag_enum_values():
    """Test that Flag enum has expected values."""
    assert Flag.AUTOMOD == "AutoMod"
    assert Flag.AI_ACTIVITY_STATUS == "ai-agent-execution-summary"
    assert Flag.BETA_BLOCKS == "beta-blocks"
    assert Flag.AGENT_ACTIVITY == "agent-activity"


@pytest.mark.asyncio
async def test_is_feature_enabled_with_flag_enum(mocker):
    """Test is_feature_enabled function with Flag enum."""
    mock_get_feature_flag_value = mocker.patch(
        "backend.util.feature_flag.get_feature_flag_value"
    )
    mock_get_feature_flag_value.return_value = True

    result = await is_feature_enabled(Flag.AUTOMOD, "user123")

    assert result is True
    # Should call with the flag's string value
    mock_get_feature_flag_value.assert_called_once()


class TestEnvFlagOverride:
    def test_force_flag_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_CHAT", "true")
        assert _env_flag_override(Flag.CHAT) is True

    def test_force_flag_false(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_CHAT", "false")
        assert _env_flag_override(Flag.CHAT) is False

    def test_next_public_prefix_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NEXT_PUBLIC_FORCE_FLAG_CHAT", "true")
        assert _env_flag_override(Flag.CHAT) is True

    def test_unset_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("FORCE_FLAG_CHAT", raising=False)
        monkeypatch.delenv("NEXT_PUBLIC_FORCE_FLAG_CHAT", raising=False)
        assert _env_flag_override(Flag.CHAT) is None

    def test_invalid_value_returns_false(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_CHAT", "notaboolean")
        assert _env_flag_override(Flag.CHAT) is False

    def test_numeric_one_returns_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_CHAT", "1")
        assert _env_flag_override(Flag.CHAT) is True

    def test_yes_returns_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_CHAT", "yes")
        assert _env_flag_override(Flag.CHAT) is True

    def test_on_returns_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_CHAT", "on")
        assert _env_flag_override(Flag.CHAT) is True

    def test_hyphenated_flag_converts_to_underscore(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("FORCE_FLAG_CHAT_MODE_OPTION", "true")
        assert _env_flag_override(Flag.CHAT_MODE_OPTION) is True

    def test_force_flag_takes_precedence_over_next_public(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("FORCE_FLAG_CHAT", "false")
        monkeypatch.setenv("NEXT_PUBLIC_FORCE_FLAG_CHAT", "true")
        assert _env_flag_override(Flag.CHAT) is False

    def test_whitespace_is_stripped(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_CHAT", "  true  ")
        assert _env_flag_override(Flag.CHAT) is True

    def test_case_insensitive_value(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_CHAT", "TRUE")
        assert _env_flag_override(Flag.CHAT) is True


class TestUserContext:
    @staticmethod
    def _stub_supabase(mocker, *, created_at, role="authenticated", email="x@y.com"):
        user = mocker.MagicMock(role=role, email=email, created_at=created_at)
        response = mocker.MagicMock(user=user)
        supabase = mocker.MagicMock()
        supabase.auth.admin.get_user_by_id.return_value = response
        mocker.patch("backend.util.clients.get_supabase", return_value=supabase)
        return supabase

    @pytest.mark.asyncio
    async def test_context_includes_created_at_iso_string(self, mocker):
        created = datetime.datetime(2026, 5, 7, 12, 0, 0, tzinfo=datetime.timezone.utc)
        supabase = self._stub_supabase(mocker, created_at=created)
        user_id = str(uuid.uuid4())

        ctx = await _fetch_user_context_data(user_id)

        assert ctx.get("created_at") == created.isoformat()
        assert ctx.get("email") == "x@y.com"
        supabase.auth.admin.get_user_by_id.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_context_skips_created_at_when_missing(self, mocker):
        supabase = self._stub_supabase(mocker, created_at=None)
        user_id = str(uuid.uuid4())

        ctx = await _fetch_user_context_data(user_id)

        assert "created_at" not in ctx.custom_attributes
        assert ctx.get("email") == "x@y.com"
        supabase.auth.admin.get_user_by_id.assert_called_once_with(user_id)


class TestUserContextCacheDegradation:
    """A failed Supabase lookup must not poison the 24h context cache.

    If the degraded anonymous (email-less) context were cached, one
    Supabase blip would make this process evaluate email/role-targeted
    flags differently from its peers for a full day, silently.
    """

    @staticmethod
    def _stub_failing_supabase(mocker):
        supabase = mocker.MagicMock()
        supabase.auth.admin.get_user_by_id.side_effect = ConnectionError(
            "supabase unreachable"
        )
        mocker.patch("backend.util.clients.get_supabase", return_value=supabase)
        return supabase

    @pytest.mark.asyncio
    async def test_degraded_anonymous_context_is_not_cached(self, mocker):
        supabase = self._stub_failing_supabase(mocker)
        user_id = str(uuid.uuid4())

        first = await _fetch_user_context_data(user_id)
        second = await _fetch_user_context_data(user_id)

        assert first.anonymous is True
        assert second.anonymous is True
        assert supabase.auth.admin.get_user_by_id.call_count == 2

    @pytest.mark.asyncio
    async def test_successful_context_is_cached_across_calls(self, mocker):
        supabase = TestUserContext._stub_supabase(mocker, created_at=None)
        user_id = str(uuid.uuid4())

        first = await _fetch_user_context_data(user_id)
        second = await _fetch_user_context_data(user_id)

        assert first.get("email") == "x@y.com"
        assert second.get("email") == "x@y.com"
        assert supabase.auth.admin.get_user_by_id.call_count == 1

    @pytest.mark.asyncio
    async def test_context_lookup_recovers_after_transient_failure(self, mocker):
        user = mocker.MagicMock(role="authenticated", email="x@y.com", created_at=None)
        response = mocker.MagicMock(user=user)
        supabase = mocker.MagicMock()
        supabase.auth.admin.get_user_by_id.side_effect = [
            ConnectionError("supabase blip"),
            response,
        ]
        mocker.patch("backend.util.clients.get_supabase", return_value=supabase)
        user_id = str(uuid.uuid4())

        degraded = await _fetch_user_context_data(user_id)
        recovered = await _fetch_user_context_data(user_id)

        assert degraded.anonymous is True
        assert degraded.get("email") is None
        assert recovered.anonymous is False
        assert recovered.get("email") == "x@y.com"

    @pytest.mark.asyncio
    async def test_degraded_lookup_logs_degradation_warning(self, mocker, caplog):
        self._stub_failing_supabase(mocker)
        user_id = str(uuid.uuid4())

        with caplog.at_level(logging.WARNING, logger="backend.util.feature_flag"):
            await _fetch_user_context_data(user_id)

        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ]
        assert any(user_id in message and "degraded" in message for message in warnings)

    @pytest.mark.asyncio
    async def test_non_uuid_key_skips_supabase_lookup(self, mocker):
        get_supabase = mocker.patch("backend.util.clients.get_supabase")

        ctx = await _fetch_user_context_data("system")

        assert ctx.anonymous is True
        get_supabase.assert_not_called()
