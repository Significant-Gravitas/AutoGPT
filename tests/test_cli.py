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


def _empty_config(tmp_path: Path) -> Path:
    """A config file with nothing in it, so the user's own stays out."""
    path = tmp_path / "config.toml"
    path.write_text("")
    return path


def test_platform_oauth_url_cli_override(tmp_path: Path) -> None:
    args = cli._build_parser().parse_args(
        [
            "--config",
            str(_empty_config(tmp_path)),
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
            str(_empty_config(tmp_path)),
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
        str(_empty_config(tmp_path)),
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
    captured = capsys.readouterr()
    output = captured.out + captured.err
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
    assert "Preflight failed: audit unavailable" in capsys.readouterr().err


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

    monkeypatch.delenv("AUTOGPT_SHIM_SUPERVISOR", raising=False)

    assert await cli._cmd_start(config) == 77
    assert "Authentication failed: session denied" in capsys.readouterr().err


@pytest.mark.asyncio
async def test_an_authentication_failure_under_launchd_stops_the_restarts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """launchd's KeepAlive restarts any non-zero exit, so 77 would retry the
    rejected credentials every few seconds; the LaunchAgent sets the marker."""
    import autogpt_local_executor.daemon as daemon_module

    class RejectedDaemon:
        def __init__(self, **_kwargs) -> None:
            pass

        async def run(self) -> None:
            raise daemon_module.DaemonAuthenticationError("session denied")

    monkeypatch.setattr(daemon_module, "ShimDaemon", RejectedDaemon)
    monkeypatch.setenv("AUTOGPT_SHIM_SUPERVISOR", "launchd")
    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )

    assert await cli._cmd_start(config) == 0
    err = capsys.readouterr().err
    assert "Authentication failed: session denied" in err
    assert "launchctl kickstart" in err


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


@pytest.mark.parametrize(
    "ts",
    ["-Infinity", '"-Infinity"', "NaN", "true", '"1"'],
    ids=["infinity", "infinity-string", "nan", "boolean", "number-string"],
)
def test_audit_prune_keeps_a_file_with_a_timestamp_that_is_not_a_finite_number(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], ts: str
) -> None:
    """float() turned -Infinity, as a JSON token or a string, into -inf,
    older than any cutoff, so prune deleted the file. The writer only ever
    writes a finite number, and a file prune can't date is kept."""
    current = tmp_path / "audit.log"
    current.write_text("")
    rotated = tmp_path / "audit.log.20240101-000000"
    rotated.write_text(f'{{"seq": 1, "op": "SHIM_START", "ts": {ts}}}\n')

    assert cli._audit_prune(current, "1d") == 0

    assert rotated.exists()
    assert "Kept" in capsys.readouterr().out


def test_version_prints_the_installed_version(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from importlib.metadata import version

    monkeypatch.setattr(sys, "argv", ["autogpt-shim", "--version"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"autogpt-shim {version('autogpt-local-executor')}"


def test_an_unknown_log_level_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli._build_parser().parse_args(["--log-level", "trace", "status"])

    assert exit_info.value.code == 2
    assert cli._build_parser().parse_args(["--log-level", "debug", "status"]).log_level == "DEBUG"


@contextlib.contextmanager
def _embedder_logging_at_debug() -> Iterator[None]:
    """A program that imports ShimDaemon sets up its own logging; main()'s
    cap on the websockets loggers never runs there."""
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    root.handlers = [logging.StreamHandler()]
    root.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        root.handlers = saved[0]
        root.setLevel(saved[1])


@pytest.mark.asyncio
async def test_an_embedder_logging_at_debug_never_sees_the_bearer_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[str | None] = []

    async def record(_connection, request) -> None:
        sent.append(request.headers.get("Authorization"))

    async def handler(ws) -> None:
        await ws.close()

    with _embedder_logging_at_debug():
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

    assert sent == ["Bearer access-secret"]
    assert "access-secret" not in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("supervisor", "permanent", "code"),
    [
        ("launchd", True, 0),
        (None, True, 78),
        ("launchd", False, 78),
    ],
)
async def test_a_preflight_failure_a_restart_cannot_fix_stops_a_supervised_shim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    supervisor: str | None,
    permanent: bool,
    code: int,
) -> None:
    """An enabled preview fails preflight every time, so a supervisor would
    restart it every few seconds; a missing macOS permission (the TCC chain)
    must still exit 78 so launchd relaunches it once it is granted."""
    import autogpt_local_executor.daemon as daemon_module

    class FailingDaemon:
        def __init__(self, **_kwargs) -> None:
            pass

        async def run(self) -> None:
            raise daemon_module.DaemonPreflightError("preview is disabled", permanent=permanent)

    monkeypatch.setattr(daemon_module, "ShimDaemon", FailingDaemon)
    if supervisor is None:
        monkeypatch.delenv("AUTOGPT_SHIM_SUPERVISOR", raising=False)
    else:
        monkeypatch.setenv("AUTOGPT_SHIM_SUPERVISOR", supervisor)
    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )

    assert await cli._cmd_start(config) == code
    assert "Preflight failed: preview is disabled" in capsys.readouterr().err


@pytest.mark.parametrize(
    "content",
    [None, b"allowed_root = [unclosed\n", b"max_concurrent = 0\n"],
    ids=["missing", "malformed", "invalid"],
)
def test_install_refuses_a_config_the_service_could_not_start_with(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    content: bytes | None,
) -> None:
    """install wrote the autostart entry for any --config, and the service
    then failed at every start with a config error nobody saw."""
    from autogpt_local_executor import install

    config = tmp_path / "config.toml"
    if content is not None:
        config.write_bytes(content)
    install_autostart = MagicMock()
    monkeypatch.setattr(install, "install_autostart", install_autostart)
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", "--config", str(config), "install"])

    with pytest.raises(SystemExit) as exited:
        cli.main()

    assert exited.value.code == 78
    assert "Configuration error" in capsys.readouterr().err
    install_autostart.assert_not_called()


def test_install_takes_a_config_the_service_can_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autogpt_local_executor import install

    config = tmp_path / "config.toml"
    config.write_text(f'allowed_root = "{(tmp_path / "files").as_posix()}"\n')
    install_autostart = MagicMock()
    install_autostart.return_value = SimpleNamespace(
        target_path=tmp_path / "unit",
        platform="linux",
        enable_command=None,
        disable_command=None,
        notes=[],
    )
    monkeypatch.setattr(install, "install_autostart", install_autostart)
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", "--config", str(config), "install"])

    cli.main()

    assert install_autostart.call_args.kwargs["config_path"] == config


def test_install_carries_the_deployment_flags_into_the_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """install dropped --platform-url and --platform-oauth-url, so the
    service talked to the default platform with credentials scoped to
    another one, and failed at every start."""
    from autogpt_local_executor import install

    root = tmp_path / "files"
    install_autostart = MagicMock()
    install_autostart.return_value = SimpleNamespace(
        target_path=tmp_path / "unit",
        platform="linux",
        enable_command=None,
        disable_command=None,
        notes=[],
    )
    monkeypatch.setattr(install, "install_autostart", install_autostart)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autogpt-shim",
            "--config",
            str(_empty_config(tmp_path)),
            "--platform-url",
            "https://platform.example.com",
            "--platform-oauth-url",
            "https://login.example.com/auth",
            "--allowed-root",
            "files",
            "install",
        ],
    )

    cli.main()

    assert install_autostart.call_args.kwargs["global_args"] == (
        "--platform-url",
        "https://platform.example.com",
        "--platform-oauth-url",
        "https://login.example.com/auth",
        "--allowed-root",
        str(root),
    )


def _capture_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    from autogpt_local_executor import install

    install_autostart = MagicMock()
    install_autostart.return_value = SimpleNamespace(
        target_path=tmp_path / "unit",
        platform="linux",
        enable_command=None,
        disable_command=None,
        notes=[],
    )
    monkeypatch.setattr(install, "install_autostart", install_autostart)
    return install_autostart


def test_install_carries_the_deployment_from_the_environment_into_the_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only flags reached the service. With the platform URL set in the
    environment, auth signed in to that deployment and the service, which
    doesn't get this shell's environment, looked for the default one's
    credentials and failed at every start."""
    install_autostart = _capture_install(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTOGPT_SHIM_PLATFORM_URL", "https://platform.example.com")
    monkeypatch.setenv("AUTOGPT_SHIM_PLATFORM_OAUTH_URL", "https://login.example.com/auth")
    monkeypatch.setenv("AUTOGPT_SHIM_ALLOWED_ROOT", "files")
    monkeypatch.setattr(
        sys, "argv", ["autogpt-shim", "--config", str(_empty_config(tmp_path)), "install"]
    )

    cli.main()

    assert install_autostart.call_args.kwargs["global_args"] == (
        "--platform-url",
        "https://platform.example.com",
        "--platform-oauth-url",
        "https://login.example.com/auth",
        "--allowed-root",
        str(tmp_path / "files"),
    )


def test_an_install_flag_outranks_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_autostart = _capture_install(tmp_path, monkeypatch)
    monkeypatch.setenv("AUTOGPT_SHIM_PLATFORM_URL", "https://other.example.com")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "autogpt-shim",
            "--config",
            str(_empty_config(tmp_path)),
            "--platform-url",
            "https://platform.example.com",
            "install",
        ],
    )

    cli.main()

    assert install_autostart.call_args.kwargs["global_args"] == (
        "--platform-url",
        "https://platform.example.com",
    )


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("AUTOGPT_SHIM_PLATFORM_WS_URL", "wss://platform.example.com/ws/local-executor"),
        ("AUTOGPT_SHIM_OAUTH_CLIENT_ID", "another-client"),
    ],
)
def test_install_refuses_a_deployment_setting_only_the_environment_has(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    variable: str,
    value: str,
) -> None:
    """These pick the saved credentials too, and no option carries them to
    the service, so it would fail at every start."""
    install_autostart = _capture_install(tmp_path, monkeypatch)
    monkeypatch.setenv(variable, value)
    monkeypatch.setattr(
        sys, "argv", ["autogpt-shim", "--config", str(_empty_config(tmp_path)), "install"]
    )

    with pytest.raises(SystemExit) as exited:
        cli.main()

    assert exited.value.code == 78
    assert variable in capsys.readouterr().err
    install_autostart.assert_not_called()


def test_install_names_settings_from_the_environment_the_service_will_not_get(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_autostart = _capture_install(tmp_path, monkeypatch)
    monkeypatch.setenv("AUTOGPT_SHIM_ENABLE_SHELL", "true")
    monkeypatch.setattr(
        sys, "argv", ["autogpt-shim", "--config", str(_empty_config(tmp_path)), "install"]
    )

    cli.main()

    install_autostart.assert_called_once()
    assert "AUTOGPT_SHIM_ENABLE_SHELL" in capsys.readouterr().out


def test_a_bad_config_stops_the_launchagent_instead_of_looping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """launchd restarts any non-zero exit, and a bad config fails the same
    way every time; exit 78 looped every ten seconds."""
    import sys

    config = tmp_path / "config.toml"
    config.write_text("allowed_root = [unclosed\n")
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", "--config", str(config), "start"])
    monkeypatch.setenv("AUTOGPT_SHIM_SUPERVISOR", "launchd")

    with pytest.raises(SystemExit) as exited:
        cli.main()

    assert exited.value.code == 0
    assert "launchctl kickstart" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize("supervisor", [None, "launchd", "systemd"])
async def test_a_temporary_preflight_failure_exits_75_for_a_restart(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    supervisor: str | None,
) -> None:
    """Audit storage that was full, or not mounted yet, failed preflight with
    78, which the systemd unit never restarts, so the shim stayed down once
    the storage was back. 75 (EX_TEMPFAIL) is restarted by systemd and
    launchd alike."""
    import autogpt_local_executor.daemon as daemon_module

    class FailingDaemon:
        def __init__(self, **_kwargs) -> None:
            pass

        async def run(self) -> None:
            raise daemon_module.DaemonPreflightError("the disk is full", transient=True)

    monkeypatch.setattr(daemon_module, "ShimDaemon", FailingDaemon)
    if supervisor is None:
        monkeypatch.delenv("AUTOGPT_SHIM_SUPERVISOR", raising=False)
    else:
        monkeypatch.setenv("AUTOGPT_SHIM_SUPERVISOR", supervisor)
    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )

    assert await cli._cmd_start(config) == 75
    assert "Preflight failed: the disk is full" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize(("code", "status"), [("ENOSPC", 75), ("EIO", 78)])
async def test_start_exits_by_how_the_audit_storage_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str, status: int
) -> None:
    """The exit-status test raised a preflight error already marked
    transient, so a full disk taken for a permanent failure where the
    storage error is classified would still have passed it."""
    import errno

    import autogpt_local_executor.daemon as daemon_module

    number = getattr(errno, code)

    def refuse(**_kwargs) -> None:
        raise OSError(number, os.strerror(number))

    monkeypatch.setattr(daemon_module, "get_or_create_audit_key", lambda: b"k" * 32)
    monkeypatch.setattr(daemon_module, "AuditWriter", refuse)
    monkeypatch.delenv("AUTOGPT_SHIM_SUPERVISOR", raising=False)
    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )

    assert await cli._cmd_start(config) == status


def test_the_systemd_unit_restarts_a_temporary_failure() -> None:
    template = Path(cli.__file__).parent / "templates" / "systemd.service.template"
    [line] = [
        line
        for line in template.read_text().splitlines()
        if line.startswith("RestartPreventExitStatus=")
    ]
    assert "75" not in line.split("=", 1)[1].split()


@pytest.mark.skipif(sys.platform == "win32", reason="asyncio signal handlers are POSIX-only")
@pytest.mark.asyncio
async def test_sigterm_stops_the_daemon_as_ctrl_c_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """systemd and launchd stop the service with SIGTERM, whose default
    action ended the process at once: no stop, no SHIM_STOP record, and a
    running command's process group left behind."""
    import signal

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
    loop = asyncio.get_running_loop()
    handlers: dict[int, tuple] = {}
    monkeypatch.setattr(
        loop,
        "add_signal_handler",
        lambda sig, callback, *args: handlers.update({sig: (callback, args)}),
    )
    monkeypatch.setattr(
        loop, "remove_signal_handler", lambda sig: handlers.pop(sig, None) is not None
    )
    start = asyncio.create_task(cli._cmd_start(ShimConfig(allowed_root=tmp_path)))
    await asyncio.wait_for(running.wait(), timeout=30)

    callback, args = handlers[signal.SIGTERM]
    callback(*args)

    assert await start == 0
    assert stopped == [True]
    assert signal.SIGTERM not in handlers
    assert "Shim stopped." in capsys.readouterr().out
