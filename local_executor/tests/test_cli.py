"""CLI safety and capability opt-in tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from autogpt_local_executor import cli
from autogpt_local_executor.config import ShimConfig


def test_platform_oauth_url_cli_override(tmp_path: Path) -> None:
    args = cli._build_parser().parse_args(
        [
            "--config",
            str(tmp_path / "missing.toml"),
            "--platform-url",
            "https://api.example.com",
            "--platform-oauth-url",
            "https://app.example.com/auth",
            "auth",
        ]
    )

    config = cli._build_config(args)

    assert config.derived_oauth_url == "https://app.example.com/auth"
    assert config.derived_oauth_token_url == "https://api.example.com/api/oauth/token"


@pytest.mark.parametrize(
    "url",
    [
        "http://platform.example.com",
        "https://user:secret-value@platform.example.com",
        "https://platform.example.com:secret-value",
        "https://[secret-value]",
    ],
)
def test_invalid_endpoint_exits_cleanly_without_echoing_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    url: str,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "autogpt-shim",
            "--config",
            str(tmp_path / "missing.toml"),
            "--platform-url",
            url,
            "start",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 78
    output = capsys.readouterr()
    assert "Configuration error" in output.err
    assert "secret-value" not in output.err + output.out
    assert "Traceback" not in output.err + output.out


@pytest.mark.parametrize("session_after_start", [False, True])
def test_start_session_id_is_accepted_before_or_after_subcommand_without_overwrite(
    tmp_path: Path, session_after_start: bool
) -> None:
    parser = cli._build_parser()
    argv = [
        "--config",
        str(tmp_path / "missing.toml"),
        "--allowed-root",
        str(tmp_path),
    ]
    if not session_after_start:
        argv.extend(["--session-id", "session-123"])
    argv.append("start")
    if session_after_start:
        argv.extend(["--session-id", "session-123"])
    argv.extend(["--enable-shell", "--enable-recording"])
    args = parser.parse_args(argv)

    config = cli._build_config(args)

    assert config.session_id == "session-123"
    assert config.allowed_root == tmp_path
    assert config.enable_shell is True
    assert config.enable_recording is True


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", [None, "", "   "])
async def test_start_without_session_uses_persistent_control_mode(
    tmp_path: Path,
    session_id: str | None,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import autogpt_local_executor.daemon as daemon_module

    ran = False

    class ControlDaemon:
        def __init__(self, **_kwargs) -> None:
            pass

        async def run(self) -> None:
            nonlocal ran
            ran = True

    monkeypatch.setattr(daemon_module, "ShimDaemon", ControlDaemon)
    config = ShimConfig(
        session_id=session_id,
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )

    assert await cli._cmd_start(config) == 0
    assert ran is True
    assert "persistent machine control" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_recording_preview_fails_startup_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import autogpt_local_executor.daemon as daemon_module

    monkeypatch.setattr(daemon_module, "get_or_create_audit_key", lambda: b"a" * 32)
    config = ShimConfig(
        session_id="session-1",
        enable_recording=True,
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )

    assert await cli._cmd_start(config) == 78
    output = capsys.readouterr().out
    assert "design preview" in output
    assert "capture and interpretation are incomplete" in output


@pytest.mark.asyncio
async def test_audit_initialization_failure_returns_ex_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import autogpt_local_executor.daemon as daemon_module

    def fail_audit(_config):
        raise daemon_module.DaemonPreflightError("audit unavailable")

    monkeypatch.setattr(
        daemon_module.ShimDaemon,
        "_build_audit_writer",
        staticmethod(fail_audit),
    )
    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )

    assert await cli._cmd_start(config) == 78
    assert "Preflight failed: audit unavailable" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_terminal_authentication_failure_returns_ex_noperm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import autogpt_local_executor.daemon as daemon_module

    class RejectedDaemon:
        def __init__(self, **_kwargs) -> None:
            pass

        async def run(self) -> None:
            raise daemon_module.DaemonAuthenticationError("session denied")

    monkeypatch.setattr(daemon_module, "ShimDaemon", RejectedDaemon)
    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )

    assert await cli._cmd_start(config) == 77
    assert "Authentication failed: session denied" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["auth", "start", "revoke"])
async def test_all_auth_commands_construct_a_deployment_scoped_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    from autogpt_local_executor.auth import CredentialScope, OAuthFlow
    from autogpt_local_executor.daemon import ShimDaemon

    config = ShimConfig(
        platform_url="https://api.example.com/backend",
        platform_oauth_url="https://app.example.com/auth",
        platform_ws_url="wss://relay.example.com/ws/local-executor",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )
    scopes = []

    def auth_run(flow):
        scopes.append(flow.token_store.scope)

    async def revoke(flow):
        scopes.append(flow.token_store.scope)

    async def daemon_run(daemon):
        scopes.append(daemon.token_store.scope)

    monkeypatch.setattr(OAuthFlow, "run", auth_run)
    monkeypatch.setattr(OAuthFlow, "revoke_tokens", revoke)
    monkeypatch.setattr(ShimDaemon, "run", daemon_run)
    monkeypatch.setattr(ShimDaemon, "_build_audit_writer", staticmethod(lambda _config: None))
    if command == "auth":
        cli._cmd_auth(config)
    elif command == "start":
        assert await cli._cmd_start(config) == 0
    else:
        assert await cli._cmd_revoke(config) == 0

    assert scopes == [CredentialScope.from_config(config)]


@pytest.mark.asyncio
async def test_legacy_credentials_have_actionable_start_failure_without_secret_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from autogpt_local_executor.daemon import ShimDaemon

    config = ShimConfig(allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log")
    monkeypatch.setattr(
        "autogpt_local_executor.auth.keyring.get_password",
        lambda _service, key: "legacy-secret" if key == "access_token" else None,
    )
    monkeypatch.setattr(ShimDaemon, "_build_audit_writer", staticmethod(lambda _config: None))
    monkeypatch.setattr(ShimDaemon, "run", ShimDaemon._session)

    assert await cli._cmd_start(config) == 77
    output = capsys.readouterr().out
    assert "legacy" in output
    assert "autogpt-shim auth" in output
    assert "left untouched" in output
    assert "legacy-secret" not in output


@pytest.mark.asyncio
async def test_revoke_failure_does_not_log_token_bearing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from autogpt_local_executor.auth import OAuthFlow

    async def fail_revoke(_flow):
        raise RuntimeError("transport failed: access-secret refresh-secret")

    monkeypatch.setattr(OAuthFlow, "revoke_tokens", fail_revoke)
    assert await cli._cmd_revoke(ShimConfig(allowed_root=tmp_path)) == 1
    output = capsys.readouterr().out
    assert "revocation did not complete" in output
    assert "access-secret" not in output
    assert "refresh-secret" not in output


def test_auth_failure_does_not_print_token_bearing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from autogpt_local_executor.auth import OAuthFlow

    def fail_auth(_flow):
        raise RuntimeError("keychain failed: access-secret refresh-secret")

    monkeypatch.setattr(OAuthFlow, "run", fail_auth)

    assert cli._cmd_auth(ShimConfig(allowed_root=tmp_path)) == 1
    output = capsys.readouterr().out
    assert "Authentication did not complete" in output
    assert "access-secret" not in output
    assert "refresh-secret" not in output
