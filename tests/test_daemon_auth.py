"""WebSocket authentication retry and session binding tests."""

from __future__ import annotations

from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from websockets.asyncio.server import serve
from websockets.exceptions import InvalidStatus

import autogpt_local_executor.daemon as daemon_module
from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.auth import KeychainTokenStore
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.daemon import (
    MAX_WEBSOCKET_MESSAGE_BYTES,
    DaemonAuthenticationError,
    ShimDaemon,
)


def _daemon(tmp_path: Path, *, session_id: str | None = "session-1") -> ShimDaemon:
    config = ShimConfig(
        session_id=session_id,
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )
    return ShimDaemon(
        config,
        token_store=object(),
        audit=MagicMock(spec=AuditWriter),
    )


class _ScriptedConnect:
    def __init__(self, statuses: list[int | None]) -> None:
        self.statuses = list(statuses)
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, *, additional_headers: dict[str, str], **kwargs):
        status = self.statuses.pop(0)
        self.calls.append({"url": url, "headers": additional_headers, **kwargs})

        class Connection:
            async def __aenter__(self):
                if status is not None:
                    response = SimpleNamespace(status_code=status)
                    raise InvalidStatus(response)  # type: ignore[arg-type]
                return object()

            async def __aexit__(self, *args):
                return None

        return Connection()


def _install_auth_doubles(
    daemon: ShimDaemon,
    monkeypatch: pytest.MonkeyPatch,
    statuses: list[int | None],
) -> tuple[list[bool], _ScriptedConnect]:
    token_calls: list[bool] = []

    async def get_token(*, refresh_on_fail: bool, rejected_token: str | None = None) -> str:
        token_calls.append(refresh_on_fail)
        if refresh_on_fail:
            assert rejected_token == "stale"
        return "fresh" if refresh_on_fail else "stale"

    async def run_session(_ws: object) -> None:
        return None

    connect = _ScriptedConnect(statuses)
    monkeypatch.setattr(daemon, "_get_access_token", get_token)
    monkeypatch.setattr(daemon, "_run_session", run_session)
    monkeypatch.setattr(daemon_module, "CredentialBoundConnect", connect)
    return token_calls, connect


@pytest.mark.asyncio
async def test_initial_401_refreshes_once_and_sets_relay_receive_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = _daemon(tmp_path)
    token_calls, connect = _install_auth_doubles(daemon, monkeypatch, [401, None])

    await daemon._session()

    assert token_calls == [False, True]
    assert [call["headers"] for call in connect.calls] == [
        {"Authorization": "Bearer stale"},
        {"Authorization": "Bearer fresh"},
    ]
    assert all(call["max_size"] == MAX_WEBSOCKET_MESSAGE_BYTES for call in connect.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["platform_url", "platform_ws_url"])
async def test_insecure_websocket_is_rejected_before_loading_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    daemon = _daemon(tmp_path)
    daemon.config = daemon.config.model_copy(update={field: "ws://platform.example.com"})
    token_calls, connect = _install_auth_doubles(daemon, monkeypatch, [None])

    with pytest.raises(ValueError, match="TLS"):
        await daemon._session()

    assert token_calls == []
    assert connect.calls == []


@pytest.mark.asyncio
async def test_initial_403_is_terminal_without_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = _daemon(tmp_path)
    token_calls, connect = _install_auth_doubles(daemon, monkeypatch, [403])

    with pytest.raises(DaemonAuthenticationError, match="403"):
        await daemon._session()

    assert token_calls == [False]
    assert len(connect.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["platform_url", "platform_ws_url", "platform_oauth_url"])
async def test_changed_destination_does_not_send_saved_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    daemon = _daemon(tmp_path)
    value = (
        "wss://other.example.com/ws" if field == "platform_ws_url" else "https://other.example.com"
    )
    setattr(daemon.config, field, value)
    _, connect = _install_auth_doubles(daemon, monkeypatch, [None])

    with pytest.raises(DaemonAuthenticationError, match="configuration.*changed"):
        await daemon._session()

    assert connect.calls == []


@pytest.mark.asyncio
async def test_legacy_credentials_stop_before_websocket_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[str] = []

    def get_password(_service: str, key: str) -> str | None:
        reads.append(key)
        return "legacy-secret" if key == "access_token" else None

    monkeypatch.setattr("autogpt_local_executor.auth.keyring.get_password", get_password)
    daemon = _daemon(tmp_path)
    daemon.token_store = KeychainTokenStore(daemon.config)
    connect = _ScriptedConnect([None])
    monkeypatch.setattr(daemon_module, "CredentialBoundConnect", connect)

    with pytest.raises(DaemonAuthenticationError, match="legacy.*autogpt-shim.*auth") as exc:
        await daemon._session()

    assert "secret" not in str(exc.value)
    assert "access_token" in reads
    assert connect.calls == []


@pytest.mark.asyncio
async def test_missing_current_deployment_credentials_require_auth_before_connect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = _daemon(tmp_path)
    daemon.token_store = KeychainTokenStore(daemon.config)
    monkeypatch.setattr("autogpt_local_executor.auth.keyring.get_password", lambda *_args: None)
    connect = _ScriptedConnect([None])
    monkeypatch.setattr(daemon_module, "CredentialBoundConnect", connect)

    with pytest.raises(DaemonAuthenticationError, match="this deployment.*autogpt-shim.*auth"):
        await daemon._session()

    assert connect.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("same_origin", [True, False])
async def test_websocket_redirect_never_reaches_an_unapproved_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_origin: bool
) -> None:
    received: list[str | None] = []
    original_requests: list[str | None] = []

    async def handler(ws) -> None:
        await ws.close()

    async def destination_request(_connection, request):
        received.append(request.headers.get("Authorization"))

    async with serve(handler, "127.0.0.1", 0, process_request=destination_request) as destination:
        target_port = destination.sockets[0].getsockname()[1]

        async def origin_request(connection, request):
            if request.path != "/ws/local-executor":
                received.append(request.headers.get("Authorization"))
                return None
            original_requests.append(request.headers.get("Authorization"))
            response = connection.respond(HTTPStatus.FOUND, "redirect")
            response.headers["Location"] = (
                "/unapproved" if same_origin else f"ws://127.0.0.1:{target_port}/unapproved"
            )
            return response

        async with serve(handler, "127.0.0.1", 0, process_request=origin_request) as origin:
            config = ShimConfig(
                platform_url=f"http://127.0.0.1:{origin.sockets[0].getsockname()[1]}",
                allowed_root=tmp_path,
                audit_log_path=tmp_path / "audit.log",
            )
            store = SimpleNamespace(get_access_token=AsyncMock(return_value="access-secret"))
            daemon = ShimDaemon(config, token_store=store, audit=MagicMock())
            monkeypatch.setattr(daemon, "_run_session", AsyncMock())

            with pytest.raises(DaemonAuthenticationError, match="redirect"):
                await daemon_module.asyncio.wait_for(daemon._session(), timeout=5)

    assert original_requests == ["Bearer access-secret"]
    assert received == []


@pytest.mark.asyncio
async def test_keychain_read_error_is_retried_and_does_not_disclose_tokens(tmp_path: Path) -> None:
    """A read error can clear (a keychain still locked at login), so it is
    retried rather than terminal; its text never reaches the message."""
    from autogpt_local_executor.daemon import KeychainUnavailableError

    daemon = _daemon(tmp_path)
    daemon.token_store = SimpleNamespace(
        get_access_token=AsyncMock(side_effect=RuntimeError("access-secret"))
    )

    with pytest.raises(KeychainUnavailableError, match="OS keychain") as exc:
        await daemon._get_access_token(refresh_on_fail=False)

    assert "access-secret" not in str(exc.value)


@pytest.mark.asyncio
async def test_refresh_failure_does_not_log_token_bearing_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from autogpt_local_executor.auth import OAuthFlow

    daemon = _daemon(tmp_path)
    daemon.token_store = SimpleNamespace(get_access_token=AsyncMock(return_value="access-secret"))

    async def failed_refresh(_flow):
        raise RuntimeError("refresh failed: refresh-secret")

    monkeypatch.setattr(OAuthFlow, "refresh_token", failed_refresh)

    assert await daemon._get_access_token(refresh_on_fail=True) is None
    assert "Token refresh failed" in caplog.text
    assert "access-secret" not in caplog.text
    assert "refresh-secret" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_status", [401, 403])
async def test_denial_after_single_refresh_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry_status: int
) -> None:
    daemon = _daemon(tmp_path)
    token_calls, connect = _install_auth_doubles(daemon, monkeypatch, [401, retry_status])

    with pytest.raises(DaemonAuthenticationError, match="after one refresh"):
        await daemon._session()

    assert token_calls == [False, True]
    assert len(connect.calls) == 2


@pytest.mark.asyncio
async def test_terminal_auth_error_never_enters_reconnect_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = _daemon(tmp_path)
    calls = 0

    async def rejected_session() -> None:
        nonlocal calls
        calls += 1
        raise DaemonAuthenticationError("denied")

    async def noop(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(daemon, "_session", rejected_session)
    monkeypatch.setattr(daemon, "_audit_shim_start", noop)
    monkeypatch.setattr(daemon, "_audit_shim_stop", noop)

    with pytest.raises(DaemonAuthenticationError, match="denied"):
        await daemon.run()

    assert calls == 1
    assert daemon._disable_reconnect is True
    assert daemon._running is False


@pytest.mark.asyncio
async def test_forced_refresh_does_not_reuse_rejected_keychain_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autogpt_local_executor.auth import OAuthFlow

    class Store:
        access_token = "stale"

        async def get_access_token(self) -> str:
            return self.access_token

    store = Store()
    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )
    daemon = ShimDaemon(config, token_store=store, audit=MagicMock())  # type: ignore[arg-type]
    refreshed = False

    async def refresh_token(_flow: OAuthFlow) -> None:
        nonlocal refreshed
        refreshed = True
        store.access_token = "fresh"

    monkeypatch.setattr(OAuthFlow, "refresh_token", refresh_token)

    assert await daemon._get_access_token(refresh_on_fail=True) == "fresh"
    assert refreshed is True


@pytest.mark.asyncio
async def test_shared_refresh_lock_rotates_token_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autogpt_local_executor.auth import OAuthFlow

    class Store:
        access_token = "stale"

        async def get_access_token(self) -> str:
            return self.access_token

    store = Store()
    refresh_lock = daemon_module.asyncio.Lock()
    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )
    first = ShimDaemon(
        config.model_copy(deep=True),
        token_store=store,
        audit=MagicMock(),
        token_refresh_lock=refresh_lock,
    )
    second = ShimDaemon(
        config.model_copy(deep=True),
        token_store=store,
        audit=MagicMock(),
        token_refresh_lock=refresh_lock,
    )
    refresh_calls = 0

    async def refresh_token(_flow: OAuthFlow) -> None:
        nonlocal refresh_calls
        refresh_calls += 1
        await daemon_module.asyncio.sleep(0)
        store.access_token = "fresh"

    monkeypatch.setattr(OAuthFlow, "refresh_token", refresh_token)

    tokens = await daemon_module.asyncio.gather(
        first._get_access_token(
            refresh_on_fail=True,
            rejected_token="stale",
        ),
        second._get_access_token(
            refresh_on_fail=True,
            rejected_token="stale",
        ),
    )

    assert tokens == ["fresh", "fresh"]
    assert refresh_calls == 1


def test_connect_url_without_session_uses_control_endpoint(tmp_path: Path) -> None:
    for session_id in (None, "", "   "):
        daemon = _daemon(tmp_path, session_id=session_id)
        assert daemon._connect_url().endswith("/ws/local-executor")


def test_audit_initialization_failure_is_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from autogpt_local_executor.daemon import DaemonPreflightError

    def fail_key() -> bytes:
        raise RuntimeError("keychain unavailable")

    monkeypatch.setattr(daemon_module, "get_or_create_audit_key", fail_key)
    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )

    with pytest.raises(DaemonPreflightError, match="mandatory audit log"):
        ShimDaemon(config, token_store=object())

    injected_audit = MagicMock()
    daemon = ShimDaemon(config, token_store=object(), audit=injected_audit)
    assert daemon.audit is injected_audit


@pytest.mark.asyncio
async def test_first_audit_write_failure_prevents_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )
    audit = MagicMock(spec=AuditWriter)
    audit.shim_start = AsyncMock(side_effect=OSError("disk full"))
    daemon = ShimDaemon(config, token_store=object(), audit=audit)
    session = AsyncMock()
    monkeypatch.setattr(daemon, "_session", session)

    with pytest.raises(daemon_module.DaemonPreflightError, match="initial SHIM_START"):
        await daemon.run()

    session.assert_not_awaited()
    assert daemon._running is False


@pytest.mark.asyncio
async def test_clean_disconnect_uses_backoff_before_reconnect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = _daemon(tmp_path)
    calls = 0
    delays: list[float] = []

    async def clean_session() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            daemon._running = False

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(daemon, "_session", clean_session)
    monkeypatch.setattr(daemon, "_backoff_delay", lambda _attempt: 1.5)
    monkeypatch.setattr(daemon_module.asyncio, "sleep", fake_sleep)

    await daemon.run()

    assert calls == 2
    assert delays == [1.5]


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [True, False])
async def test_a_legacy_session_creates_its_allowed_root_on_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, legacy: bool
) -> None:
    """CROSS_PLATFORM.md: the shim creates the default root on first run. A
    persistent control session picks folders per chat and has none."""
    root = tmp_path / "autogpt-workspace"
    config = ShimConfig(
        session_id="session-1" if legacy else None,
        allowed_root=root,
        audit_log_path=tmp_path / "audit.log",
    )
    daemon = ShimDaemon(
        config,
        token_store=object(),
        audit=MagicMock(spec=AuditWriter),
        manage_audit_lifecycle=False,
    )
    monkeypatch.setattr(
        daemon, "_session", AsyncMock(side_effect=DaemonAuthenticationError("stop here"))
    )

    with pytest.raises(DaemonAuthenticationError):
        await daemon.run()

    assert root.is_dir() is legacy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "terminal"),
    [("KeyringLocked", False), ("KeyringError", False), ("NoKeyringError", True)],
)
async def test_a_keychain_read_failure_is_retried_unless_there_is_no_keychain(
    tmp_path: Path, error: str, terminal: bool
) -> None:
    """A keychain that is still locked at login used to stop the LaunchAgent
    for good; only a missing keychain backend is a setup problem."""
    import keyring.errors

    from autogpt_local_executor.daemon import KeychainUnavailableError

    failure = getattr(keyring.errors, error)

    class Store:
        async def get_access_token(self) -> str:
            raise failure("keychain says no")

    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )
    daemon = ShimDaemon(config, token_store=Store(), audit=MagicMock())  # type: ignore[arg-type]

    expected = DaemonAuthenticationError if terminal else KeychainUnavailableError
    with pytest.raises(expected):
        await daemon._read_access_token()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "transient"),
    [
        ("connect", True),
        ("timeout", True),
        (503, True),
        (429, True),
        (400, False),
    ],
)
async def test_a_refresh_that_could_not_reach_the_server_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure, transient: bool
) -> None:
    """A network error or a 5xx during the refresh used to end in a terminal
    authentication failure; only a refused refresh token should."""
    import httpx

    from autogpt_local_executor.auth import OAuthFlow
    from autogpt_local_executor.daemon import TokenRefreshUnavailableError

    daemon = _daemon(tmp_path)
    daemon.token_store = SimpleNamespace(get_access_token=AsyncMock(return_value="stale"))
    request = httpx.Request("POST", "https://platform.example/api/oauth/token")

    async def refresh(_flow):
        if failure == "connect":
            raise httpx.ConnectError("unreachable", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("slow", request=request)
        raise httpx.HTTPStatusError(
            "refused", request=request, response=httpx.Response(failure, request=request)
        )

    monkeypatch.setattr(OAuthFlow, "refresh_token", refresh)

    if transient:
        with pytest.raises(TokenRefreshUnavailableError):
            await daemon._get_access_token(refresh_on_fail=True, rejected_token="stale")
    else:
        assert await daemon._get_access_token(refresh_on_fail=True, rejected_token="stale") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("where", "error", "terminal"),
    [
        ("refresh", "KeyringLocked", False),
        ("refresh", "PasswordSetError", False),
        ("refresh", "NoKeyringError", True),
        ("reread", "KeyringLocked", False),
    ],
)
async def test_a_keychain_failure_during_a_refresh_is_retried_unless_there_is_no_keychain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, error: str, terminal: bool
) -> None:
    """The refresh's catch-all took a keychain that locked during the refresh
    for a refused refresh token, which ended the session for good."""
    import keyring.errors

    from autogpt_local_executor.auth import OAuthFlow
    from autogpt_local_executor.daemon import KeychainUnavailableError

    failure = getattr(keyring.errors, error)
    reads = 0

    async def get_access_token() -> str:
        nonlocal reads
        reads += 1
        if where == "reread" and reads > 1:
            raise failure("keychain says no")
        return "stale"

    async def refresh(_flow) -> None:
        if where == "refresh":
            raise failure("keychain says no")

    daemon = _daemon(tmp_path)
    daemon.token_store = SimpleNamespace(get_access_token=get_access_token)
    monkeypatch.setattr(OAuthFlow, "refresh_token", refresh)

    expected = DaemonAuthenticationError if terminal else KeychainUnavailableError
    with pytest.raises(expected):
        await daemon._get_access_token(refresh_on_fail=True, rejected_token="stale")


@pytest.mark.parametrize(
    ("failure", "permanent", "transient"),
    [
        ("corrupt", True, False),
        ("disk-full", False, True),
        ("read-only", True, False),
        ("keychain-locked", False, True),
    ],
)
def test_an_audit_failure_at_start_says_whether_a_restart_can_fix_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    permanent: bool,
    transient: bool,
) -> None:
    """Every audit failure at start was the same preflight error, exit 78.
    Under launchd a log that fails verification restarted every few seconds,
    and under systemd storage that recovered was never retried."""
    import errno
    import os

    import keyring
    import keyring.errors

    from autogpt_local_executor.daemon import DaemonPreflightError

    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )
    monkeypatch.delenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", raising=False)
    if failure == "keychain-locked":

        def locked(*_args: object) -> None:
            raise keyring.errors.KeyringLocked("the keychain is locked")

        monkeypatch.setattr(keyring, "get_password", locked)
    else:
        monkeypatch.setattr(daemon_module, "get_or_create_audit_key", lambda: b"k" * 32)
    if failure == "corrupt":
        config.audit_log_path.write_text('{"seq": 1, "ts": 1.0, "op": "X", "hmac": "00"}\n')
    elif failure in ("disk-full", "read-only"):
        code = errno.ENOSPC if failure == "disk-full" else errno.EROFS

        def refuse(**_kwargs: object) -> None:
            raise OSError(code, os.strerror(code))

        monkeypatch.setattr(daemon_module, "AuditWriter", refuse)

    with pytest.raises(DaemonPreflightError) as raised:
        ShimDaemon._build_audit_writer(config)

    assert raised.value.permanent is permanent
    assert getattr(raised.value, "transient", False) is transient
