import ast
import datetime
import logging
import uuid
from pathlib import Path

import pytest
from fastapi import HTTPException
from ldclient import Context, LDClient
from ldclient.config import Config
from ldclient.integrations.test_data import TestData

import backend
import backend.util.feature_flag as feature_flag_module
from backend.util.feature_flag import (
    _NON_BOOLEAN_FLAG_VALUES,
    Flag,
    _env_flag_override,
    _fetch_user_context_data,
    _force_all_flags_enabled,
    create_feature_flag_dependency,
    evaluate_feature_flag,
    feature_flag,
    get_client,
    is_feature_enabled,
    mock_flag_variation,
    shutdown_launchdarkly,
)
from backend.util.settings import AppEnvironment


@pytest.fixture
def ld_client(mocker):
    client = mocker.Mock(spec=LDClient)
    mocker.patch("backend.util.feature_flag.ldclient.get", return_value=client)
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

    result = await is_feature_enabled(Flag.AUTOMOD, "user123", default=True)
    assert result is True  # Should return default

    ld_client.variation.assert_not_called()


@pytest.mark.asyncio
async def test_is_feature_enabled_exception(mocker):
    """Test is_feature_enabled when get_client() raises an exception."""
    mocker.patch(
        "backend.util.feature_flag.get_client",
        side_effect=Exception("Client error"),
    )

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
    before their "LaunchDarkly not initialised / not configured" bail-out,
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
        self, ld_client, monkeypatch: pytest.MonkeyPatch
    ):
        ld_client.is_initialized.return_value = False
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")

        @feature_flag("test-flag")
        async def test_function(user_id: str):
            return "success"

        assert await test_function(user_id="test-user") == "success"
        ld_client.variation.assert_not_called()

    @pytest.mark.asyncio
    async def test_decorator_per_flag_false_wins_over_launchdarkly(
        self, ld_client, monkeypatch: pytest.MonkeyPatch
    ):
        ld_client.variation.return_value = True
        monkeypatch.setenv("FORCE_FLAG_TEST_FLAG", "false")

        @feature_flag("test-flag")
        async def test_function(user_id: str):
            return "success"

        with pytest.raises(HTTPException) as exc_info:
            await test_function(user_id="test-user")
        assert exc_info.value.status_code == 404
        ld_client.variation.assert_not_called()

    @pytest.mark.asyncio
    async def test_dependency_force_all_wins_over_unconfigured_sdk(
        self, mocker, monkeypatch: pytest.MonkeyPatch
    ):
        mocker.patch("backend.util.feature_flag.is_configured", return_value=False)
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")

        check_feature_flag = create_feature_flag_dependency(Flag.AUTOMOD)
        assert await check_feature_flag(user_id=None) is None

    @pytest.mark.asyncio
    async def test_dependency_per_flag_false_returns_404(
        self, mocker, monkeypatch: pytest.MonkeyPatch
    ):
        mocker.patch("backend.util.feature_flag.is_configured", return_value=False)
        monkeypatch.setenv("FORCE_ALL_FLAGS", "true")
        monkeypatch.setenv("FORCE_FLAG_AUTOMOD", "false")

        # default=True so the 404 can only come from the override, not the
        # unconfigured-SDK fallback.
        check_feature_flag = create_feature_flag_dependency(Flag.AUTOMOD, default=True)
        with pytest.raises(HTTPException) as exc_info:
            await check_feature_flag(user_id=None)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_decorator_disabled_flag_is_not_logged_as_an_error(
        self, ld_client, caplog: pytest.LogCaptureFixture
    ):
        """A gated-off route is an expected outcome, not an evaluation error.

        The 404 is raised inside the decorator's try block, so without an
        explicit passthrough every request to a disabled route files an
        ERROR and buries real flag-evaluation failures.
        """
        ld_client.variation.return_value = False

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
        self, ld_client, caplog: pytest.LogCaptureFixture
    ):
        """The passthrough must not silence genuine failures."""
        ld_client.is_initialized.side_effect = RuntimeError("LD exploded")

        @feature_flag("test-flag")
        async def test_function(user_id: str):
            return "success"

        with caplog.at_level(logging.ERROR, logger="backend.util.feature_flag"):
            with pytest.raises(RuntimeError):
                await test_function(user_id="test-user")
        assert "Error evaluating feature flag" in caplog.text

    @pytest.mark.asyncio
    async def test_dependency_disabled_flag_returns_404_not_500(
        self, ld_client, mocker
    ):
        """A disabled flag is an answer, not a LaunchDarkly failure.

        The 404 is raised inside the try block that catches LaunchDarkly
        errors, so without an explicit re-raise the generic handler turns
        every gated-off route into a 500.
        """
        mocker.patch("backend.util.feature_flag.is_configured", return_value=True)
        ld_client.is_initialized.return_value = True
        ld_client.variation.return_value = False

        check_feature_flag = create_feature_flag_dependency(Flag.AUTOMOD)
        with pytest.raises(HTTPException) as exc_info:
            await check_feature_flag(user_id="test-user")
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == "Feature not available"

    @pytest.mark.asyncio
    async def test_dependency_uninitialised_client_returns_404_not_500(
        self, ld_client, mocker
    ):
        """Same for the 404 raised on the uninitialised-client branch."""
        mocker.patch("backend.util.feature_flag.is_configured", return_value=True)
        ld_client.is_initialized.return_value = False

        check_feature_flag = create_feature_flag_dependency(Flag.AUTOMOD)
        with pytest.raises(HTTPException) as exc_info:
            await check_feature_flag(user_id="test-user")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_dependency_launchdarkly_error_still_returns_500(
        self, ld_client, mocker
    ):
        """The re-raise must not swallow genuine LaunchDarkly failures."""
        mocker.patch("backend.util.feature_flag.is_configured", return_value=True)
        ld_client.is_initialized.return_value = True
        mocker.patch(
            "backend.util.feature_flag.is_feature_enabled",
            new=mocker.AsyncMock(side_effect=RuntimeError("LD exploded")),
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

        ctx = await _fetch_user_context_data(user_id)

        assert ctx.get("created_at") == created.isoformat()
        assert ctx.get("email") == "x@y.com"
        accessor.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_context_skips_created_at_when_missing(self, mocker):
        accessor = self._stub_flag_fields(mocker, created_at=None)
        user_id = str(uuid.uuid4())

        ctx = await _fetch_user_context_data(user_id)

        assert "created_at" not in ctx.custom_attributes
        assert ctx.get("email") == "x@y.com"
        accessor.assert_called_once_with(user_id)

    @pytest.mark.asyncio
    async def test_context_maps_admin_role_through(self, mocker):
        self._stub_flag_fields(mocker, created_at=None, role="admin")

        ctx = await _fetch_user_context_data(str(uuid.uuid4()))

        assert ctx.get("role") == "admin"

    @pytest.mark.asyncio
    async def test_context_normalizes_non_admin_role_to_authenticated(self, mocker):
        self._stub_flag_fields(mocker, created_at=None, role="user")

        ctx = await _fetch_user_context_data(str(uuid.uuid4()))

        assert ctx.get("role") == "authenticated"

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

        ctx = await _fetch_user_context_data(user_id)
        assert ctx.get("email") is None
        assert ctx.get("role") is None


class TestUserContextConnectionRouting:
    """The context lookup must reach the DB directly when Prisma is locally
    connected, and through the DatabaseManager RPC client otherwise.

    Prisma-less workers (scheduler, copilot-executor) previously hit a direct
    Prisma call that raised ClientNotConnectedError and silently degraded the
    LaunchDarkly context to anonymous.
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

        ctx = await _fetch_user_context_data(user_id)

        assert ctx.get("role") == "admin"
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

        ctx = await _fetch_user_context_data(user_id)

        assert ctx.get("role") == "authenticated"
        assert ctx.get("email") == "c@d.com"
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

        first = await _fetch_user_context_data(user_id)
        second = await _fetch_user_context_data(user_id)

        assert first.anonymous is True
        assert second.anonymous is True
        assert accessor.call_count == 2

    @pytest.mark.asyncio
    async def test_successful_context_is_cached_across_calls(self, mocker):
        accessor = TestUserContext._stub_flag_fields(mocker, created_at=None)
        user_id = str(uuid.uuid4())

        first = await _fetch_user_context_data(user_id)
        second = await _fetch_user_context_data(user_id)

        assert first.get("email") == "x@y.com"
        assert second.get("email") == "x@y.com"
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

        degraded = await _fetch_user_context_data(user_id)
        recovered = await _fetch_user_context_data(user_id)

        assert degraded.anonymous is True
        assert degraded.get("email") is None
        assert recovered.anonymous is False
        assert recovered.get("email") == "x@y.com"

    @pytest.mark.asyncio
    async def test_degraded_lookup_logs_degradation_warning(self, mocker, caplog):
        self._stub_failing_lookup(mocker)
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
    async def test_non_uuid_key_skips_user_lookup(self, mocker):
        accessor = mocker.patch(
            "backend.data.user.get_auth_user_flag_fields",
            new=mocker.AsyncMock(),
        )

        ctx = await _fetch_user_context_data("system")

        assert ctx.anonymous is True
        accessor.assert_not_called()


class TestShutdown:
    @pytest.fixture(autouse=True)
    def reset_module_state(self):
        initialized = feature_flag_module._is_initialized
        attempted = feature_flag_module._init_attempted
        yield
        feature_flag_module._is_initialized = initialized
        feature_flag_module._init_attempted = attempted

    @pytest.fixture
    def sdk_key(self, mocker):
        return mocker.patch.object(
            feature_flag_module.settings.secrets,
            "launch_darkly_sdk_key",
            "sdk-key",
        )

    def test_shutdown_is_a_noop_when_never_initialized(self, mocker):
        # `initialize_launchdarkly` returns early when no SDK key is set, so
        # `ldclient.set_config` was never called and `ldclient.get()` raises
        # "set_config was not called". Callers pair init/shutdown on app_env
        # alone, so this ran on every unconfigured non-LOCAL deployment and
        # took the exception out through service teardown, leaving the process
        # alive until it was killed.
        feature_flag_module._is_initialized = False
        get_ldclient = mocker.patch("backend.util.feature_flag.ldclient.get")

        shutdown_launchdarkly()

        get_ldclient.assert_not_called()

    def test_shutdown_closes_an_initialized_client(self, ld_client):
        feature_flag_module._is_initialized = True

        shutdown_launchdarkly()

        ld_client.close.assert_called_once()

    def test_shutdown_closes_a_client_that_never_connected(self, ld_client):
        # A configured client that never reached LaunchDarkly still has
        # streaming and event threads running. Skipping close() there leaves
        # exactly the kind of live thread that holds a process open past its
        # stop deadline.
        feature_flag_module._is_initialized = True
        ld_client.is_initialized.return_value = False

        shutdown_launchdarkly()

        ld_client.close.assert_called_once()

    def test_shutdown_does_not_rearm_lazy_initialization(
        self, mocker, ld_client, sdk_key
    ):
        # `_is_initialized` is never cleared, which is what keeps a flag
        # evaluation arriving after teardown from rebuilding the client. This
        # pins that property; it is not a regression test for the gate swap.
        feature_flag_module._is_initialized = True
        feature_flag_module._init_attempted = True
        set_config = mocker.patch("backend.util.feature_flag.ldclient.set_config")

        shutdown_launchdarkly()
        get_client()

        set_config.assert_not_called()

    def test_unconfigured_deployment_only_attempts_initialization_once(
        self, mocker, ld_client
    ):
        # Without a key `_is_initialized` never becomes True, so gating the
        # lazy init on it re-entered initialize_launchdarkly on every flag
        # evaluation: a warning plus a raise per call, several per request.
        feature_flag_module._is_initialized = False
        feature_flag_module._init_attempted = False
        mocker.patch.object(
            feature_flag_module.settings.secrets, "launch_darkly_sdk_key", ""
        )
        warn = mocker.patch.object(feature_flag_module.logger, "warning")

        get_client()
        get_client()
        get_client()

        assert warn.call_count == 1


class TestEvaluateFeatureFlag:
    """Callers that delete state on a False need to know it was a real answer."""

    @pytest.fixture(autouse=True)
    def no_env_override(self, monkeypatch: pytest.MonkeyPatch):
        """`.env` may force unrelated flags; pin this one to LaunchDarkly."""
        monkeypatch.delenv("FORCE_FLAG_HIRE_EXPERTS", raising=False)
        monkeypatch.delenv("NEXT_PUBLIC_FORCE_FLAG_HIRE_EXPERTS", raising=False)

    @pytest.fixture
    def user_context(self, mocker):
        """A resolved context, so `authoritative` turns purely on evaluation."""
        return mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(Context.create("u-1"), True),
        )

    @pytest.mark.asyncio
    async def test_a_successful_evaluation_is_authoritative(
        self, ld_client, user_context
    ):
        ld_client.variation.return_value = False
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, True)

    @pytest.mark.asyncio
    async def test_an_uninitialised_client_is_not(self, ld_client, user_context):
        ld_client.is_initialized.return_value = False
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, False)

    @pytest.mark.asyncio
    async def test_an_evaluation_that_raises_is_not(self, ld_client, user_context):
        """The regression this class exists for: a LIVE client can still fail
        to produce a value, and that must not read as a real "off"."""
        ld_client.variation.side_effect = Exception("evaluation exploded")
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, False)

    @pytest.mark.asyncio
    async def test_a_degraded_user_context_is_not(self, ld_client, mocker):
        """The context lookup is a database read that SWALLOWS its failure and
        returns an anonymous context, so evaluation still succeeds — against
        the wrong user. That value must not be trusted as an answer."""
        mock_prisma = mocker.patch("backend.data.db.prisma")
        mock_prisma.is_connected.return_value = True
        mocker.patch(
            "backend.data.user.get_auth_user_flag_fields",
            new=mocker.AsyncMock(side_effect=ConnectionError("database unreachable")),
        )
        ld_client.variation.return_value = False

        result = await evaluate_feature_flag(Flag.HIRE_EXPERTS, str(uuid.uuid4()))

        assert result == (False, False)
        ld_client.variation.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_non_boolean_flag_value_is_not(self, ld_client, user_context):
        ld_client.variation.return_value = {"some": "object"}
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, False)

    @pytest.mark.asyncio
    async def test_an_env_override_answers_without_launchdarkly(
        self, mocker, monkeypatch: pytest.MonkeyPatch
    ):
        """A forced flag is a real answer, so acting on its False is safe."""
        mocker.patch(
            "backend.util.feature_flag.get_client",
            side_effect=Exception("set_config was not called"),
        )
        monkeypatch.setenv("FORCE_FLAG_HIRE_EXPERTS", "false")
        assert await evaluate_feature_flag(Flag.HIRE_EXPERTS, "u-1") == (False, True)

    @pytest.mark.asyncio
    async def test_is_feature_enabled_still_returns_the_bare_value(
        self, ld_client, user_context
    ):
        """The refactor must not change what existing callers see."""
        ld_client.variation.return_value = True
        assert await is_feature_enabled(Flag.HIRE_EXPERTS, "u-1") is True
        ld_client.variation.side_effect = Exception("boom")
        assert await is_feature_enabled(Flag.HIRE_EXPERTS, "u-1") is False


class TestRequestAttributes:
    """Facts known only per request (the visitor's country) reach targeting."""

    @pytest.mark.asyncio
    async def test_attributes_are_layered_on_without_touching_the_cache(
        self, ld_client, mocker
    ):
        cached = (
            Context.builder("user-1")
            .kind("user")
            .set("email_domain", "agpt.co")
            .build()
        )
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(cached, True),
        )
        ld_client.variation.return_value = {"version": "v1"}

        await feature_flag_module.get_feature_flag_value(
            "card-required-trial-offer", "user-1", None, attributes={"country": "IN"}
        )

        evaluated = ld_client.variation.call_args[0][1]
        assert evaluated.get("country") == "IN"
        assert evaluated.get("email_domain") == "agpt.co"
        assert cached.get("country") is None

    @pytest.mark.asyncio
    async def test_attributes_cannot_change_who_is_evaluated(self, ld_client, mocker):
        cached = Context.builder("user-1").kind("user").build()
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(cached, True),
        )
        await feature_flag_module.get_feature_flag_value(
            "f", "user-1", None, attributes={"key": "someone-else", "kind": "org"}
        )
        evaluated = ld_client.variation.call_args[0][1]
        assert (evaluated.key, evaluated.kind) == ("user-1", "user")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "country,offered", [("US", True), ("BR", True), ("IN", False), (None, False)]
    )
    async def test_a_real_country_rule_decides_the_offer(
        self, mocker, country, offered
    ):
        """End to end through the real SDK evaluator, not a mocked variation.

        The recommended production shape: serve the offer when country is not
        one of the excluded list; otherwise, including an unknown country,
        fall through to off.
        """
        td = TestData.data_source()
        td.update(
            td.flag("card-required-trial-offer")
            .variations({"enabled": False}, {"version": "v1"})
            .fallthrough_variation(0)
            .if_not_match("country", "IN")
            .then_return(1)
        )
        client = LDClient(
            Config("sdk-test", update_processor_class=td, send_events=False)
        )
        mocker.patch("backend.util.feature_flag.ldclient.get", return_value=client)
        mocker.patch(
            "backend.util.feature_flag._fetch_user_context_status",
            return_value=(Context.builder("user-1").kind("user").build(), True),
        )
        try:
            value = await feature_flag_module.get_feature_flag_value(
                "card-required-trial-offer",
                "user-1",
                None,
                attributes={"country": country} if country else None,
            )
        finally:
            client.close()
        assert (value == {"version": "v1"}) is offered
