import ast
import datetime
import logging
import uuid
from pathlib import Path

import pytest
from fastapi import HTTPException
from posthog.types import FeatureFlagResult

import backend
import backend.util.feature_flag as feature_flag_module
import backend.util.feature_flag.posthog as ph
from backend.util.feature_flag import (
    _NON_BOOLEAN_FLAG_VALUES,
    Flag,
    _env_flag_override,
    _force_all_flags_enabled,
    _UserContext,
    create_feature_flag_dependency,
    evaluate_feature_flag,
    feature_flag,
    is_feature_enabled,
)
from backend.util.settings import AppEnvironment


@pytest.fixture
def posthog_client(mocker):
    """A ready PostHog flag client whose answer each test sets with `served`."""
    client = mocker.Mock()
    mocker.patch.object(ph, "get_flag_client", return_value=client)
    mocker.patch.object(ph, "is_configured", return_value=True)
    return client


async def _context(user_id: str) -> _UserContext:
    context, _ = await feature_flag_module._fetch_user_context_status(user_id)
    return context


@pytest.mark.asyncio
async def test_feature_flag_enabled(posthog_client):
    posthog_client.get_feature_flag_result.return_value = served(True)

    @feature_flag("test-flag")
    async def test_function(user_id: str):
        return "success"

    result = await test_function(user_id="test-user")
    assert result == "success"
    posthog_client.get_feature_flag_result.assert_called_once()


@pytest.mark.asyncio
async def test_feature_flag_unauthorized_response(posthog_client):
    posthog_client.get_feature_flag_result.return_value = served(False)

    @feature_flag("test-flag")
    async def test_function(user_id: str):
        return "success"

    with pytest.raises(HTTPException) as exc_info:
        await test_function(user_id="test-user")
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_is_feature_enabled(posthog_client):
    """Test the is_feature_enabled helper function."""
    posthog_client.get_feature_flag_result.return_value = served(True)

    result = await is_feature_enabled(Flag.AUTOMOD, "user123", default=False)
    assert result is True

    posthog_client.get_feature_flag_result.assert_called_once()
    call = posthog_client.get_feature_flag_result.call_args
    assert call.args == ("AutoMod", "user123")


@pytest.mark.asyncio
async def test_is_feature_enabled_without_a_client(mocker):
    """No PostHog client (no key) serves the default."""
    mocker.patch.object(ph, "get_flag_client", return_value=None)

    result = await is_feature_enabled(Flag.AUTOMOD, "user123", default=True)
    assert result is True  # Should return default


@pytest.mark.asyncio
async def test_is_feature_enabled_exception(posthog_client):
    """A read that raises serves the default."""
    posthog_client.get_feature_flag_result.side_effect = Exception("Client error")

    result = await is_feature_enabled(Flag.AUTOMOD, "user123", default=True)
    assert result is True  # Should return default


def test_flag_enum_values():
    """Test that Flag enum has expected values."""
    assert Flag.AUTOMOD == "AutoMod"
    assert Flag.AI_ACTIVITY_STATUS == "ai-agent-execution-summary"


@pytest.mark.asyncio
async def test_is_feature_enabled_with_flag_enum(mocker):
    """Test is_feature_enabled function with Flag enum."""
    mock_evaluate = mocker.patch("backend.util.feature_flag._evaluate_flag_value")
    mock_evaluate.return_value = (True, True)

    result = await is_feature_enabled(Flag.AUTOMOD, "user123")

    assert result is True
    # Should call with the flag's string value
    mock_evaluate.assert_called_once()
    assert mock_evaluate.call_args.args[0] == Flag.AUTOMOD.value


class TestEnvFlagOverride:
    def test_force_flag_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "true")
        assert _env_flag_override(Flag.AUTOMOD) is True

    def test_force_flag_false(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "false")
        assert _env_flag_override(Flag.AUTOMOD) is False

    def test_next_public_prefix_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("NEXT_PUBLIC_FORCE_FLAG_AUTOMOD", "true")
        assert _env_flag_override(Flag.AUTOMOD) is True

    def test_unset_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("FORCE_FLAG_AUTOMOD", raising=False)
        monkeypatch.delenv("NEXT_PUBLIC_FORCE_FLAG_AUTOMOD", raising=False)
        assert _env_flag_override(Flag.AUTOMOD) is None

    def test_invalid_value_returns_false(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "notaboolean")
        assert _env_flag_override(Flag.AUTOMOD) is False

    def test_numeric_one_returns_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "1")
        assert _env_flag_override(Flag.AUTOMOD) is True

    def test_yes_returns_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "yes")
        assert _env_flag_override(Flag.AUTOMOD) is True

    def test_on_returns_true(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "on")
        assert _env_flag_override(Flag.AUTOMOD) is True

    def test_hyphenated_flag_converts_to_underscore(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("FORCE_FLAG_CHAT_MODE_OPTION", "true")
        assert _env_flag_override(Flag.CHAT_MODE_OPTION) is True

    def test_force_flag_takes_precedence_over_next_public(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "false")
        monkeypatch.setenv("NEXT_PUBLIC_FORCE_FLAG_AUTOMOD", "true")
        assert _env_flag_override(Flag.AUTOMOD) is False

    def test_whitespace_is_stripped(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "  true  ")
        assert _env_flag_override(Flag.AUTOMOD) is True

    def test_case_insensitive_value(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "TRUE")
        assert _env_flag_override(Flag.AUTOMOD) is True


def _flags_read_via_get_feature_flag_value() -> set[str]:
    """Flag values passed to ``get_feature_flag_value`` across the backend.

    Walks the AST of every module that mentions the function and collects
    the ``Flag.X`` / ``Flag.X.value`` first arguments. Call sites that pass
    a non-literal (``feature_flag``'s own raw-string key) are skipped —
    there is no flag identity to check there.
    """
    backend_root = Path(backend.__file__).parent
    found: set[str] = set()
    for path in backend_root.rglob("*.py"):
        if path.name.endswith("_test.py"):
            continue
        source = path.read_text(encoding="utf-8")
        if "get_feature_flag_value" not in source:
            continue
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else getattr(func, "id", None)
            )
            if name != "get_feature_flag_value":
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Attribute) and arg.attr == "value":
                arg = arg.value
            if not isinstance(arg, ast.Attribute):
                continue
            if isinstance(arg.value, ast.Name) and arg.value.id == "Flag":
                found.add(Flag[arg.attr].value)
    return found


class TestForceAllFlags:
    def _clear(self, monkeypatch: pytest.MonkeyPatch):
        for name in (
            "FORCE_ALL_FLAGS",
            "NEXT_PUBLIC_FORCE_ALL_FLAGS",
            "FORCE_FLAG_AUTOMOD",
            "NEXT_PUBLIC_FORCE_FLAG_AUTOMOD",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_off_by_default(self, monkeypatch: pytest.MonkeyPatch):
        self._clear(monkeypatch)
        assert _force_all_flags_enabled() is False
        assert _env_flag_override(Flag.AUTOMOD) is None

    def test_forces_boolean_flag_on(self, monkeypatch: pytest.MonkeyPatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")
        assert _force_all_flags_enabled() is True
        assert _env_flag_override(Flag.AUTOMOD) is True

    def test_next_public_master_switch(self, monkeypatch: pytest.MonkeyPatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("NEXT_PUBLIC_FORCE_ALL_FLAGS", "1")
        assert _env_flag_override(Flag.AUTOMOD) is True

    def test_skips_non_boolean_flags(self, monkeypatch: pytest.MonkeyPatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")
        assert _env_flag_override(Flag.STRIPE_PRODUCT_ID_TOPUP) is None
        assert _env_flag_override(Flag.COPILOT_MODEL_ROUTING) is None
        for value in _NON_BOOLEAN_FLAG_VALUES:
            assert _env_flag_override(value) is None, value

    def test_non_boolean_set_covers_every_json_valued_flag(self):
        """Every flag read through ``get_feature_flag_value`` as a payload.

        Cross-checked against the real call sites rather than against the
        set's own literals: a newly-added JSON-valued flag that someone
        reads via ``get_feature_flag_value`` but forgets to list in
        ``_NON_BOOLEAN_FLAG_VALUES`` would be force-all'd to ``True`` and
        handed to a caller expecting a payload, so this test must fail on
        that omission rather than restate the set.
        """
        read_as_payload = _flags_read_via_get_feature_flag_value()
        assert read_as_payload, "no call sites found - the scanner is broken"
        missing = read_as_payload - _NON_BOOLEAN_FLAG_VALUES
        assert not missing, (
            f"{sorted(missing)} are read through get_feature_flag_value as a "
            "payload but are missing from _NON_BOOLEAN_FLAG_VALUES, so "
            "FORCE_ALL_FLAGS would force them to True"
        )

    def test_per_flag_false_wins_over_force_all(self, monkeypatch: pytest.MonkeyPatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "false")
        assert _env_flag_override(Flag.AUTOMOD) is False

    def test_accepts_raw_string_key(self, monkeypatch: pytest.MonkeyPatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")
        assert _env_flag_override(Flag.AUTOMOD.value) is True

    def test_falsey_value_does_not_force(self, monkeypatch: pytest.MonkeyPatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("FORCE_ALL_FLAGS", "false")
        assert _force_all_flags_enabled() is False
        assert _env_flag_override(Flag.AUTOMOD) is None

    def test_ignored_in_production(
        self, monkeypatch: pytest.MonkeyPatch, mocker, caplog: pytest.LogCaptureFixture
    ):
        self._clear(monkeypatch)
        monkeypatch.setattr(feature_flag_module, "_force_all_logged", False)
        mocker.patch.object(
            feature_flag_module.settings.config, "app_env", AppEnvironment.PRODUCTION
        )
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")
        with caplog.at_level(logging.ERROR, logger="backend.util.feature_flag"):
            assert _force_all_flags_enabled() is False
            assert _env_flag_override(Flag.AUTOMOD) is None
        assert "FORCE_ALL_FLAGS is set but app_env is prod, not local" in caplog.text

    def test_ignored_in_shared_development(
        self, monkeypatch: pytest.MonkeyPatch, mocker, caplog: pytest.LogCaptureFixture
    ):
        """``dev`` is a shared deployment, not a developer's machine.

        The single-container entrypoint exports ``APP_ENV=dev``, so a
        production-only guard would leave every fail-closed gate open for
        every user of a publicly reachable dev instance.
        """
        self._clear(monkeypatch)
        monkeypatch.setattr(feature_flag_module, "_force_all_logged", False)
        mocker.patch.object(
            feature_flag_module.settings.config, "app_env", AppEnvironment.DEVELOPMENT
        )
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")
        with caplog.at_level(logging.ERROR, logger="backend.util.feature_flag"):
            assert _force_all_flags_enabled() is False
            assert _env_flag_override(Flag.AUTOMOD) is None
        assert "FORCE_ALL_FLAGS is set but app_env is dev, not local" in caplog.text

    def test_per_flag_override_still_works_in_development(
        self, monkeypatch: pytest.MonkeyPatch, mocker
    ):
        """The per-flag escape hatch is what single-container users keep."""
        self._clear(monkeypatch)
        mocker.patch.object(
            feature_flag_module.settings.config, "app_env", AppEnvironment.DEVELOPMENT
        )
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "true")
        assert _env_flag_override(Flag.AUTOMOD) is True

    def test_per_flag_override_still_works_in_production(
        self, monkeypatch: pytest.MonkeyPatch, mocker
    ):
        self._clear(monkeypatch)
        mocker.patch.object(
            feature_flag_module.settings.config, "app_env", AppEnvironment.PRODUCTION
        )
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "true")
        assert _env_flag_override(Flag.AUTOMOD) is True

    def test_honoured_in_local(self, monkeypatch: pytest.MonkeyPatch, mocker):
        self._clear(monkeypatch)
        mocker.patch.object(
            feature_flag_module.settings.config, "app_env", AppEnvironment.LOCAL
        )
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")
        assert _env_flag_override(Flag.AUTOMOD) is True

    def test_warns_once_when_honoured(
        self, monkeypatch: pytest.MonkeyPatch, mocker, caplog: pytest.LogCaptureFixture
    ):
        self._clear(monkeypatch)
        monkeypatch.setattr(feature_flag_module, "_force_all_logged", False)
        mocker.patch.object(
            feature_flag_module.settings.config, "app_env", AppEnvironment.LOCAL
        )
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")
        with caplog.at_level(logging.WARNING, logger="backend.util.feature_flag"):
            assert _force_all_flags_enabled() is True
            assert _force_all_flags_enabled() is True
        warnings = [
            r for r in caplog.records if "FORCE_ALL_FLAGS is on" in r.getMessage()
        ]
        assert len(warnings) == 1
        assert warnings[0].levelno == logging.WARNING


class TestEnvOverrideWiring:
    """The decorator and the router dependency must honour the env override
    before their "PostHog not initialised / not configured" bail-out,
    which is the normal local-dev state."""

    @pytest.fixture(autouse=True)
    def clear_env(self, monkeypatch: pytest.MonkeyPatch):
        for name in (
            "FORCE_ALL_FLAGS",
            "NEXT_PUBLIC_FORCE_ALL_FLAGS",
            "FORCE_FLAG_TEST_FLAG",
            "NEXT_PUBLIC_FORCE_FLAG_TEST_FLAG",
            "FORCE_FLAG_AUTOMOD",
            "NEXT_PUBLIC_FORCE_FLAG_AUTOMOD",
        ):
            monkeypatch.delenv(name, raising=False)

    @pytest.mark.asyncio
    async def test_decorator_force_all_wins_over_uninitialised_client(
        self, mocker, monkeypatch: pytest.MonkeyPatch
    ):
        mocker.patch.object(ph, "get_flag_client", return_value=None)
        evaluate = mocker.patch.object(ph, "evaluate_flag")
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")

        @feature_flag("test-flag")
        async def test_function(user_id: str):
            return "success"

        assert await test_function(user_id="test-user") == "success"
        evaluate.assert_not_called()

    @pytest.mark.asyncio
    async def test_decorator_per_flag_false_wins_over_posthog(
        self, posthog_client, monkeypatch: pytest.MonkeyPatch
    ):
        posthog_client.get_feature_flag_result.return_value = served(True)
        monkeypatch.setenv("FORCE_FLAG_TEST_FLAG", "false")

        @feature_flag("test-flag")
        async def test_function(user_id: str):
            return "success"

        with pytest.raises(HTTPException) as exc_info:
            await test_function(user_id="test-user")
        assert exc_info.value.status_code == 404
        posthog_client.get_feature_flag_result.assert_not_called()

    @pytest.mark.asyncio
    async def test_dependency_force_all_wins_over_unconfigured_posthog(
        self, mocker, monkeypatch: pytest.MonkeyPatch
    ):
        mocker.patch.object(ph, "is_configured", return_value=False)
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")

        check_feature_flag = create_feature_flag_dependency(Flag.AUTOMOD)
        assert await check_feature_flag(user_id=None) is None

    @pytest.mark.asyncio
    async def test_dependency_per_flag_false_returns_404(
        self, mocker, monkeypatch: pytest.MonkeyPatch
    ):
        mocker.patch.object(ph, "is_configured", return_value=False)
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "false")

        # default=True so the 404 can only come from the override, not the
        # unconfigured fallback.
        check_feature_flag = create_feature_flag_dependency(Flag.AUTOMOD, default=True)
        with pytest.raises(HTTPException) as exc_info:
            await check_feature_flag(user_id=None)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_decorator_disabled_flag_is_not_logged_as_an_error(
        self, posthog_client, caplog: pytest.LogCaptureFixture
    ):
        """A gated-off route is an expected outcome, not an evaluation error.

        The 404 is raised inside the decorator's try block, so without an
        explicit passthrough every request to a disabled route files an
        ERROR and buries real flag-evaluation failures.
        """
        posthog_client.get_feature_flag_result.return_value = served(False)

        @feature_flag("test-flag")
        async def test_function(user_id: str):
            return "success"

        with caplog.at_level(logging.ERROR, logger="backend.util.feature_flag"):
            with pytest.raises(HTTPException) as exc_info:
                await test_function(user_id="test-user")
        assert exc_info.value.status_code == 404
        assert "Error evaluating feature flag" not in caplog.text

    @pytest.mark.asyncio
    async def test_decorator_still_logs_real_evaluation_errors(
        self, mocker, caplog: pytest.LogCaptureFixture
    ):
        """The passthrough must not silence genuine failures."""
        mocker.patch.object(
            ph, "get_flag_client", side_effect=RuntimeError("PostHog exploded")
        )

        @feature_flag("test-flag")
        async def test_function(user_id: str):
            return "success"

        with caplog.at_level(logging.ERROR, logger="backend.util.feature_flag"):
            with pytest.raises(RuntimeError):
                await test_function(user_id="test-user")
        assert "Error evaluating feature flag" in caplog.text

    @pytest.mark.asyncio
    async def test_dependency_disabled_flag_returns_404_not_500(self, posthog_client):
        """A disabled flag is an answer, not a PostHog failure.

        The 404 is raised inside the try block that catches evaluation
        errors, so without an explicit re-raise the generic handler turns
        every gated-off route into a 500.
        """
        posthog_client.get_feature_flag_result.return_value = served(False)

        check_feature_flag = create_feature_flag_dependency(Flag.AUTOMOD)
        with pytest.raises(HTTPException) as exc_info:
            await check_feature_flag(user_id="test-user")
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Feature not available"

    @pytest.mark.asyncio
    async def test_dependency_uninitialised_client_returns_404_not_500(self, mocker):
        """Same for the 404 raised on the uninitialised-client branch."""
        mocker.patch.object(ph, "is_configured", return_value=True)
        mocker.patch.object(ph, "get_flag_client", return_value=None)

        check_feature_flag = create_feature_flag_dependency(Flag.AUTOMOD)
        with pytest.raises(HTTPException) as exc_info:
            await check_feature_flag(user_id="test-user")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_dependency_evaluation_error_still_returns_500(
        self, posthog_client, mocker
    ):
        """The re-raise must not swallow genuine evaluation failures."""
        mocker.patch(
            "backend.util.feature_flag.is_feature_enabled",
            new=mocker.AsyncMock(side_effect=RuntimeError("PostHog exploded")),
        )

        check_feature_flag = create_feature_flag_dependency(Flag.AUTOMOD)
        with pytest.raises(HTTPException) as exc_info:
            await check_feature_flag(user_id="test-user")
        assert exc_info.value.status_code == 500


class TestUserContext:
    @staticmethod
    def _stub_flag_fields(mocker, *, created_at, role=None, email="x@y.com"):
        """Stub the direct data-layer accessor with Prisma reported connected.

        Returns the AsyncMock standing in for ``get_auth_user_flag_fields`` so
        callers can assert on how it was invoked.
        """
        from backend.data.user import AuthUserFlagFields

        mock_prisma = mocker.patch("backend.data.db.prisma")
        mock_prisma.is_connected.return_value = True
        fields = AuthUserFlagFields(role=role, email=email, created_at=created_at)
        return mocker.patch(
            "backend.data.user.get_auth_user_flag_fields",
            new=mocker.AsyncMock(return_value=fields),
        )

    @pytest.mark.asyncio
    async def test_context_includes_created_at_iso_string(self, mocker):
        created = datetime.datetime(2026, 5, 7, 12, 0, 0, tzinfo=datetime.timezone.utc)
        accessor = self._stub_flag_fields(mocker, created_at=created)
        user_id = str(uuid.uuid4())

        ctx = await _context(user_id)

        assert (ctx.key, ctx.anonymous) == (user_id, False)
        assert ctx.attributes["created_at"] == created.isoformat()
        assert ctx.attributes["email"] == "x@y.com"
        assert ctx.attributes["email_domain"] == "y.com"
        accessor.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_context_skips_created_at_when_missing(self, mocker):
        accessor = self._stub_flag_fields(mocker, created_at=None)
        user_id = str(uuid.uuid4())

        ctx = await _context(user_id)

        assert "created_at" not in ctx.attributes
        assert ctx.attributes["email"] == "x@y.com"
        accessor.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_context_maps_admin_role_through(self, mocker):
        self._stub_flag_fields(mocker, created_at=None, role="admin")

        ctx = await _context(str(uuid.uuid4()))

        assert ctx.attributes["role"] == "admin"

    @pytest.mark.asyncio
    async def test_context_normalizes_non_admin_role_to_authenticated(self, mocker):
        self._stub_flag_fields(mocker, created_at=None, role="user")

        ctx = await _context(str(uuid.uuid4()))

        assert ctx.attributes["role"] == "authenticated"

    @pytest.mark.asyncio
    async def test_missing_user_falls_back_to_uncached_anonymous(self, mocker):
        # A not-found user (e.g. mid auth-migration bridge window) must NOT be
        # cached as anonymous — the inner lookup raises so @cached skips it and
        # the caller returns an uncached anonymous context.
        from backend.util import feature_flag as ff

        mock_prisma = mocker.patch("backend.data.db.prisma")
        mock_prisma.is_connected.return_value = True
        mocker.patch(
            "backend.data.user.get_auth_user_flag_fields",
            new=mocker.AsyncMock(return_value=None),
        )
        user_id = str(uuid.uuid4())

        with pytest.raises(LookupError):
            await ff._fetch_user_context(user_id)

        ctx = await _context(user_id)
        assert ctx.anonymous is True
        assert ctx.attributes == {}


class TestUserContextConnectionRouting:
    """The context lookup must reach the DB directly when Prisma is locally
    connected, and through the DatabaseManager RPC client otherwise.

    Prisma-less workers (scheduler, copilot-executor) previously hit a direct
    Prisma call that raised ClientNotConnectedError and silently degraded the
    flag context to anonymous.
    """

    @pytest.mark.asyncio
    async def test_connected_uses_direct_accessor(self, mocker):
        from backend.data.user import AuthUserFlagFields

        mock_prisma = mocker.patch("backend.data.db.prisma")
        mock_prisma.is_connected.return_value = True

        fields = AuthUserFlagFields(role="admin", email="a@b.com", created_at=None)
        direct = mocker.patch(
            "backend.data.user.get_auth_user_flag_fields",
            new=mocker.AsyncMock(return_value=fields),
        )
        rpc_client = mocker.MagicMock()
        rpc_client.get_auth_user_flag_fields = mocker.AsyncMock(return_value=fields)
        get_rpc_client = mocker.patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=rpc_client,
        )
        user_id = str(uuid.uuid4())

        ctx = await _context(user_id)

        assert ctx.attributes["role"] == "admin"
        direct.assert_called_once_with(user_id)
        get_rpc_client.assert_not_called()
        rpc_client.get_auth_user_flag_fields.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnected_routes_through_rpc_client(self, mocker):
        from backend.data.user import AuthUserFlagFields

        mock_prisma = mocker.patch("backend.data.db.prisma")
        mock_prisma.is_connected.return_value = False

        fields = AuthUserFlagFields(role="user", email="c@d.com", created_at=None)
        direct = mocker.patch(
            "backend.data.user.get_auth_user_flag_fields",
            new=mocker.AsyncMock(return_value=fields),
        )
        rpc_client = mocker.MagicMock()
        rpc_client.get_auth_user_flag_fields = mocker.AsyncMock(return_value=fields)
        mocker.patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=rpc_client,
        )
        user_id = str(uuid.uuid4())

        ctx = await _context(user_id)

        assert ctx.attributes["role"] == "authenticated"
        assert ctx.attributes["email"] == "c@d.com"
        rpc_client.get_auth_user_flag_fields.assert_called_once_with(user_id)
        direct.assert_not_called()


class TestUserContextCacheDegradation:
    """A failed user lookup must not poison the 24h context cache.

    If the degraded anonymous (email-less) context were cached, one
    database blip would make this process evaluate email/role-targeted
    flags differently from its peers for a full day, silently.
    """

    @staticmethod
    def _stub_failing_lookup(mocker):
        mock_prisma = mocker.patch("backend.data.db.prisma")
        mock_prisma.is_connected.return_value = True
        return mocker.patch(
            "backend.data.user.get_auth_user_flag_fields",
            new=mocker.AsyncMock(side_effect=ConnectionError("database unreachable")),
        )

    @pytest.mark.asyncio
    async def test_degraded_anonymous_context_is_not_cached(self, mocker):
        accessor = self._stub_failing_lookup(mocker)
        user_id = str(uuid.uuid4())

        first = await _context(user_id)
        second = await _context(user_id)

        assert first.anonymous is True
        assert second.anonymous is True
        assert accessor.call_count == 2

    @pytest.mark.asyncio
    async def test_successful_context_is_cached_across_calls(self, mocker):
        accessor = TestUserContext._stub_flag_fields(mocker, created_at=None)
        user_id = str(uuid.uuid4())

        first = await _context(user_id)
        second = await _context(user_id)

        assert first.attributes["email"] == "x@y.com"
        assert second.attributes["email"] == "x@y.com"
        assert accessor.call_count == 1

    @pytest.mark.asyncio
    async def test_context_lookup_recovers_after_transient_failure(self, mocker):
        from backend.data.user import AuthUserFlagFields

        fields = AuthUserFlagFields(
            role="authenticated", email="x@y.com", created_at=None
        )
        mock_prisma = mocker.patch("backend.data.db.prisma")
        mock_prisma.is_connected.return_value = True
        mocker.patch(
            "backend.data.user.get_auth_user_flag_fields",
            new=mocker.AsyncMock(
                side_effect=[ConnectionError("database blip"), fields]
            ),
        )
        user_id = str(uuid.uuid4())

        degraded = await _context(user_id)
        recovered = await _context(user_id)

        assert degraded.anonymous is True
        assert "email" not in degraded.attributes
        assert recovered.anonymous is False
        assert recovered.attributes["email"] == "x@y.com"

    @pytest.mark.asyncio
    async def test_degraded_lookup_logs_degradation_warning(self, mocker, caplog):
        self._stub_failing_lookup(mocker)
        user_id = str(uuid.uuid4())

        with caplog.at_level(logging.WARNING, logger="backend.util.feature_flag"):
            await _context(user_id)

        warnings = [
            record.getMessage()
            for record in caplog.records
            if record.levelno >= logging.WARNING
        ]
        assert any(user_id in message and "degraded" in message for message in warnings)

    @pytest.mark.asyncio
    async def test_non_uuid_key_skips_user_lookup(self, mocker):
        accessor = mocker.patch(
            "backend.data.user.get_auth_user_flag_fields",
            new=mocker.AsyncMock(),
        )

        ctx = await _context("system")

        assert ctx.anonymous is True
        accessor.assert_not_called()


class TestEvaluateFeatureFlag:
    """Callers that delete state on a False need to know it was a real answer."""

    @pytest.fixture(autouse=True)
    def no_env_override(self, monkeypatch: pytest.MonkeyPatch):
        """`.env` may force unrelated flags; pin this one to PostHog."""
        monkeypatch.delenv("FORCE_FLAG_HIRE_EXPERTS", raising=False)
        monkeypatch.delenv("NEXT_PUBLIC_FORCE_FLAG_HIRE_EXPERTS", raising=False)

    @pytest.fixture
    def user_context(self, mocker):
        """A resolved context, so `authoritative` turns purely on evaluation."""
        return mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(_UserContext(key="u-1", anonymous=False), True),
        )

    @pytest.mark.asyncio
    async def test_a_successful_evaluation_is_authoritative(
        self, posthog_client, user_context
    ):
        posthog_client.get_feature_flag_result.return_value = served(False)
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, True)

    @pytest.mark.asyncio
    async def test_no_client_is_not(self, mocker, user_context):
        mocker.patch.object(ph, "get_flag_client", return_value=None)
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, False)

    @pytest.mark.asyncio
    async def test_an_unresolved_flag_is_not(self, posthog_client, user_context):
        posthog_client.get_feature_flag_result.return_value = None
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, False)

    @pytest.mark.asyncio
    async def test_an_evaluation_that_raises_is_not(self, posthog_client, user_context):
        """The regression this class exists for: a LIVE client can still fail
        to produce a value, and that must not read as a real "off"."""
        posthog_client.get_feature_flag_result.side_effect = Exception(
            "evaluation exploded"
        )
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, False)

    @pytest.mark.asyncio
    async def test_a_degraded_user_context_is_not(self, posthog_client, mocker):
        """The context lookup is a database read that SWALLOWS its failure and
        returns an anonymous context, so evaluation still succeeds — against
        the wrong user. That value must not be trusted as an answer."""
        mock_prisma = mocker.patch("backend.data.db.prisma")
        mock_prisma.is_connected.return_value = True
        mocker.patch(
            "backend.data.user.get_auth_user_flag_fields",
            new=mocker.AsyncMock(side_effect=ConnectionError("database unreachable")),
        )
        posthog_client.get_feature_flag_result.return_value = served(False)

        result = await evaluate_feature_flag(Flag.HIRE_EXPERTS, str(uuid.uuid4()))

        assert result == (False, False)
        posthog_client.get_feature_flag_result.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_non_boolean_flag_value_is_not(self, posthog_client, user_context):
        posthog_client.get_feature_flag_result.return_value = served(
            True, payload={"some": "object"}
        )
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, False)

    @pytest.mark.asyncio
    async def test_an_env_override_answers_without_posthog(
        self, mocker, monkeypatch: pytest.MonkeyPatch
    ):
        """A forced flag is a real answer, so acting on its False is safe."""
        mocker.patch.object(
            ph, "get_flag_client", side_effect=Exception("PostHog unavailable")
        )
        monkeypatch.setenv("FORCE_FLAG_HIRE_EXPERTS", "false")
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, True)

    @pytest.mark.asyncio
    async def test_is_feature_enabled_still_returns_the_bare_value(
        self, posthog_client, user_context
    ):
        """The refactor must not change what existing callers see."""
        posthog_client.get_feature_flag_result.return_value = served(True)
        assert await is_feature_enabled(Flag.HIRE_EXPERTS, "u-1") is True
        posthog_client.get_feature_flag_result.side_effect = Exception("boom")
        assert await is_feature_enabled(Flag.HIRE_EXPERTS, "u-1") is False


class TestRequestAttributes:
    """Facts known only per request (the visitor's country) reach targeting."""

    @staticmethod
    def _evaluated(posthog_client) -> tuple[str, dict]:
        call = posthog_client.get_feature_flag_result.call_args
        return call.args[1], call.kwargs["person_properties"]

    @pytest.mark.asyncio
    async def test_attributes_are_layered_on_without_touching_the_cache(
        self, posthog_client, mocker
    ):
        cached = _UserContext(
            key="user-1", anonymous=False, attributes={"email_domain": "agpt.co"}
        )
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(cached, True),
        )
        posthog_client.get_feature_flag_result.return_value = served(
            True, payload={"version": "v1"}
        )

        await feature_flag_module.get_feature_flag_value(
            "card-required-trial-offer", "user-1", None, attributes={"country": "IN"}
        )

        _, properties = self._evaluated(posthog_client)
        assert properties["country"] == "IN"
        assert properties["email_domain"] == "agpt.co"
        assert "country" not in cached.attributes

    @pytest.mark.asyncio
    async def test_attributes_cannot_change_who_is_evaluated(
        self, posthog_client, mocker
    ):
        cached = _UserContext(key="user-1", anonymous=False)
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(cached, True),
        )
        posthog_client.get_feature_flag_result.return_value = served(False)

        await feature_flag_module.get_feature_flag_value(
            "f",
            "user-1",
            None,
            attributes={"distinct_id": "someone-else", "is_anonymous": "true"},
        )

        distinct_id, properties = self._evaluated(posthog_client)
        assert distinct_id == "user-1"
        assert "distinct_id" not in properties
        assert properties["is_anonymous"] is False

    @pytest.mark.asyncio
    async def test_an_anonymous_context_sends_no_country(self, posthog_client, mocker):
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(_UserContext(key="user-1"), False),
        )
        posthog_client.get_feature_flag_result.return_value = served(False)

        await feature_flag_module.get_feature_flag_value(
            "card-required-trial-offer", "user-1", None, attributes={"country": "US"}
        )

        _, properties = self._evaluated(posthog_client)
        assert "country" not in properties
        assert properties["is_anonymous"] is False


def served(value, payload=None):
    """PostHog's answer for a flag it resolved."""
    return FeatureFlagResult.from_value_and_payload("flag", value, payload)
