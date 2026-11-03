"""CLI safety and capability opt-in tests."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from websockets.asyncio.server import serve

from autogpt_local_executor import cli
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.daemon import ShimDaemon


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


@contextlib.contextmanager
def _logging_set_up_as_main_does(level: str) -> Iterator[None]:
    """main() configures logging before anything else has. Under pytest the root
    logger already has handlers, so they come off while this runs, or
    basicConfig would do nothing."""
    root = logging.getLogger()
    websockets_logger = logging.getLogger("websockets")
    saved = (root.handlers[:], root.level, websockets_logger.level)
    root.handlers = []
    try:
        cli._configure_logging(level)
        yield
    finally:
        root.handlers = saved[0]
        root.setLevel(saved[1])
        websockets_logger.setLevel(saved[2])


@pytest.mark.asyncio
async def test_debug_logging_never_prints_the_bearer_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """websockets logs every handshake header at DEBUG, Authorization included."""
    sent: list[str | None] = []

    async def record(_connection, request) -> None:
        sent.append(request.headers.get("Authorization"))

    async def handler(ws) -> None:
        await ws.close()

    with _logging_set_up_as_main_does("debug"):
        async with serve(handler, "127.0.0.1", 0, process_request=record) as server:
            config = ShimConfig(
                platform_url=f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}",
                allowed_root=tmp_path,
                audit_log_path=tmp_path / "audit.log",
            )
            store = SimpleNamespace(get_access_token=AsyncMock(return_value="access-secret"))
            daemon = ShimDaemon(config, token_store=store, audit=MagicMock())
            monkeypatch.setattr(daemon, "_run_session", AsyncMock())
            await asyncio.wait_for(daemon._session(), timeout=30)
        debug_on = logging.getLogger("autogpt_local_executor").isEnabledFor(logging.DEBUG)

    assert debug_on
    assert sent == ["Bearer access-secret"]
    assert "access-secret" not in capsys.readouterr().err


def test_a_quieter_log_level_also_quiets_websockets() -> None:
    """A record passes its own logger's level, then every handler above it:
    capping websockets at INFO under a WARNING root would let its INFO through."""
    with _logging_set_up_as_main_does("WARNING"):
        assert not logging.getLogger("websockets.client").isEnabledFor(logging.INFO)


@pytest.mark.asyncio
async def test_ctrl_c_stops_the_daemon_and_exits_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """On Ctrl+C, asyncio.run cancels the main task rather than raising
    KeyboardInterrupt inside it."""
    running = asyncio.Event()
    stopped: list[bool] = []

    class FakeDaemon:
        def __init__(self, **_kwargs: object) -> None:
            pass

        async def run(self) -> None:
            running.set()
            await asyncio.Event().wait()

        async def stop(self) -> None:
            stopped.append(True)

    monkeypatch.setattr("autogpt_local_executor.daemon.ShimDaemon", FakeDaemon)
    monkeypatch.setattr("autogpt_local_executor.auth.KeychainTokenStore", lambda _config: object())
    start = asyncio.create_task(cli._cmd_start(ShimConfig(allowed_root=tmp_path)))
    await asyncio.wait_for(running.wait(), timeout=30)
    start.cancel()

    assert await start == 0
    assert stopped == [True]
    assert "Shim stopped." in capsys.readouterr().out


# ── audit subcommands and --version ──────────────────────────────────────────

AUDIT_KEY = b"\x01" * 32


@pytest.fixture
def audit_key(monkeypatch: pytest.MonkeyPatch) -> bytes:
    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: AUDIT_KEY)
    return AUDIT_KEY


@pytest.mark.asyncio
async def test_audit_show_prints_every_record_with_that_seq(
    tmp_path: Path, audit_key: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    """seq restarts at 1 in every file, so the current log must not hide a
    rotated record with the same seq."""
    from autogpt_local_executor.audit import AuditWriter

    writer = AuditWriter(path=tmp_path / "audit.log", audit_key=audit_key)
    await writer.write("EXECUTE_COMMAND", request_id="before-rotation", details={})
    rotated = writer.force_rotate()
    await writer.write("EXECUTE_COMMAND", request_id="after-rotation", details={})
    assert rotated is not None

    assert cli._audit_show(writer.path, 1) == 0

    output = capsys.readouterr().out
    assert f"# from {writer.path.name}\n" in output
    assert f"# from {rotated.name}\n" in output
    assert "before-rotation" in output
    assert "after-rotation" in output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory permissions")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
@pytest.mark.asyncio
async def test_a_failed_rotation_is_reported_as_a_failure(
    tmp_path: Path, audit_key: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    from autogpt_local_executor.audit import AuditWriter

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    writer = AuditWriter(path=log_dir / "audit.log", audit_key=audit_key)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={})
    log_dir.chmod(0o500)
    try:
        exit_code = cli._audit_rotate(writer.path)
    finally:
        log_dir.chmod(0o700)

    assert exit_code == 1
    assert "Could not rotate" in capsys.readouterr().out
    assert [p.name for p in log_dir.iterdir() if not p.name.startswith(".")] == ["audit.log"]


def test_audit_prune_keeps_a_file_whose_newest_record_has_no_timestamp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    current = tmp_path / "audit.log"
    current.write_text("")
    undated = tmp_path / "audit.log.20240101-000000"
    undated.write_text('{"seq": 1, "op": "SHIM_START"}\n')
    old = tmp_path / "audit.log.20240102-000000"
    old.write_text('{"seq": 1, "op": "SHIM_START", "ts": 1}\n')

    assert cli._audit_prune(current, "1d") == 0

    assert undated.exists()
    assert not old.exists()
    assert "no timestamp" in capsys.readouterr().out


def test_version_prints_the_installed_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from importlib.metadata import version

    monkeypatch.setattr(sys, "argv", ["autogpt-shim", "--version"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"autogpt-shim {version('autogpt-local-executor')}"
