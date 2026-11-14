"""WebSocket authentication retry and session binding tests."""

from __future__ import annotations

import asyncio
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Any
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

    def fail_key(_audit_log: Path | None = None) -> bytes:
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
async def test_a_legacy_allowed_root_is_made_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session's allowed_root on a stalled network mount held the event
    loop in mkdir, which every session and the machine's connection share:
    none of them answered a request or a keepalive until the mount did."""
    import threading

    root = tmp_path / "root"
    config = ShimConfig(
        session_id="session-1",
        allowed_root=root,
        audit_log_path=tmp_path / "audit.log",
    )
    daemon = ShimDaemon(
        config,
        token_store=object(),
        audit=MagicMock(spec=AuditWriter),
        manage_audit_lifecycle=False,
    )
    monkeypatch.setattr(daemon, "_preflight_or_raise", AsyncMock())
    monkeypatch.setattr(
        daemon, "_session", AsyncMock(side_effect=DaemonAuthenticationError("stop here"))
    )
    stalled = threading.Event()
    answered = threading.Event()
    loop_ran_meanwhile: list[bool] = []
    real_mkdir = Path.mkdir

    def mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == root:
            stalled.set()
            # The mount answers once the event loop has run something else.
            loop_ran_meanwhile.append(answered.wait(timeout=30))
        real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)
    run = asyncio.ensure_future(daemon.run())
    try:
        for _ in range(3000):
            if stalled.is_set():
                break
            await asyncio.sleep(0.01)
        answered.set()
        with pytest.raises(DaemonAuthenticationError):
            await asyncio.wait_for(run, timeout=60)
    finally:
        answered.set()
        if not run.done():
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)

    assert loop_ran_meanwhile == [True]
    assert root.is_dir()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "restart"),
    [
        ("KeyringLocked", False),
        ("KeyringError", False),
        ("NoKeyringError", True),
        ("InitError", True),
    ],
)
async def test_a_keychain_read_failure_is_retried(
    tmp_path: Path, error: str, restart: bool
) -> None:
    """A keychain that is still locked at login used to stop the LaunchAgent
    for good. So did one that hadn't started yet (a Secret Service the
    desktop session starts after the autostart service), whose audit key
    the shim waits for, or reads from the file fallback: the sign-in read
    then exited 77, and the service stayed stopped once it came up. The
    keyring package picks its backend once per process, so that one is
    waited for by a new process (exit 75), not a retry in this one."""
    import keyring.errors

    from autogpt_local_executor.daemon import KeychainNotUpError, KeychainUnavailableError

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

    expected = KeychainNotUpError if restart else KeychainUnavailableError
    with pytest.raises(expected) as raised:
        await daemon._read_access_token()
    assert not isinstance(raised.value, DaemonAuthenticationError)
    if restart:
        assert raised.value.transient is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "transient"),
    [
        ("connect", True),
        ("timeout", True),
        (503, True),
        (429, True),
        # The server gave up waiting for the request, so it never read the
        # refresh token: sending it again is safe.
        (408, True),
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
    ("where", "error", "restart"),
    [
        ("refresh", "KeyringLocked", False),
        ("refresh", "PasswordSetError", False),
        ("refresh", "NoKeyringError", True),
        ("refresh", "InitError", True),
        ("reread", "KeyringLocked", False),
        ("reread", "NoKeyringError", True),
        ("reread", "InitError", True),
    ],
)
async def test_a_keychain_failure_during_a_refresh_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str, error: str, restart: bool
) -> None:
    """The refresh's catch-all took a keychain that locked during the refresh
    for a refused refresh token, which ended the session for good. So it
    did a keychain gone by the read after a refresh that worked: the shim
    said to sign in again, where a restart finds the keychain."""
    import keyring.errors

    from autogpt_local_executor.auth import OAuthFlow
    from autogpt_local_executor.daemon import KeychainNotUpError, KeychainUnavailableError

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

    expected = KeychainNotUpError if restart else KeychainUnavailableError
    with pytest.raises(expected):
        await daemon._get_access_token(refresh_on_fail=True, rejected_token="stale")


def _rotating_token_endpoint(
    monkeypatch: pytest.MonkeyPatch, *, before_answering: Any = None
) -> list[str]:
    """A token endpoint that rotates the refresh token as the platform's
    does: each one works once, and one used already gets 400. Returns the
    refresh tokens it was sent, in order. `before_answering`, a coroutine
    function, runs once the first exchange has gone through, before the
    answer comes back."""
    import urllib.parse

    import httpx

    sent: list[str] = []

    async def token_endpoint(request: httpx.Request) -> httpx.Response:
        token = urllib.parse.parse_qs(request.content.decode())["refresh_token"][0]
        used = token in sent
        sent.append(token)
        if used:
            return httpx.Response(400, json={"detail": "refresh token has been revoked"})
        if before_answering is not None and len(sent) == 1:
            await before_answering()
        tokens = {"access_token": f"access-{len(sent)}", "refresh_token": f"refresh-{len(sent)}"}
        return httpx.Response(200, json=tokens)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        "autogpt_local_executor.auth.httpx.AsyncClient",
        lambda: real_client(transport=httpx.MockTransport(token_endpoint)),
    )
    return sent


def _keychain_failing_writes(monkeypatch: pytest.MonkeyPatch) -> list[Exception]:
    """An in-memory keychain whose next writes raise what the returned list
    holds, one each, in order."""
    passwords: dict[tuple[str, str], str] = {}
    failures: list[Exception] = []

    def set_password(service: str, user: str, value: str) -> None:
        if failures:
            raise failures.pop(0)
        passwords[(service, user)] = value

    def delete_password(service: str, user: str) -> None:
        import keyring.errors

        if passwords.pop((service, user), None) is None:
            raise keyring.errors.PasswordDeleteError("not found")

    monkeypatch.setattr(
        "autogpt_local_executor.auth.keyring.get_password",
        lambda service, user: passwords.get((service, user)),
    )
    monkeypatch.setattr("autogpt_local_executor.auth.keyring.set_password", set_password)
    monkeypatch.setattr("autogpt_local_executor.auth.keyring.delete_password", delete_password)
    return failures


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", ["PasswordSetError", "KeyringLocked", "InitError", "RuntimeError"]
)
async def test_a_refreshed_sign_in_the_keychain_failed_to_store_is_stored_on_the_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    """The platform rotates the refresh token on every exchange and refuses a
    used one. A refresh whose new pair the keychain failed to store (locked
    meanwhile, say) left the used-up token in the keychain: the retry sent
    it again, the platform refused it, and the shim stopped, asking for
    `autogpt-shim auth`. An InitError exited for a restart, which lost the
    pair just the same, and another error (Windows' CredWrite raises its
    own) asked for the sign-in at once."""
    import keyring.errors

    from autogpt_local_executor.daemon import KeychainUnavailableError

    sent = _rotating_token_endpoint(monkeypatch)
    failures = _keychain_failing_writes(monkeypatch)
    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )
    store = KeychainTokenStore(config)
    await store.store_tokens("access-0", "refresh-0")
    daemon = ShimDaemon(config, token_store=store, audit=MagicMock(spec=AuditWriter))
    failure = RuntimeError if error == "RuntimeError" else getattr(keyring.errors, error)
    failures.append(failure("the keychain did not store it"))

    with pytest.raises(KeychainUnavailableError):
        await daemon._get_access_token(refresh_on_fail=True, rejected_token="access-0")
    token = await daemon._get_access_token(refresh_on_fail=True, rejected_token="access-0")

    assert token == "access-1"
    assert sent == ["refresh-0"]
    assert await store.get_refresh_token() == "refresh-1"
    assert store.unsaved_refresh is None


@pytest.mark.asyncio
async def test_a_sign_in_stored_after_a_failed_refresh_goes_ahead_of_the_kept_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pair kept from a refresh the keychain failed to store is stored only
    while the keychain still holds the refresh token that exchange used up.
    A sign-in stored since (`autogpt-shim auth`) is newer: the next refresh
    exchanges its token, and the kept pair is dropped."""
    import keyring.errors

    from autogpt_local_executor.daemon import KeychainUnavailableError

    sent = _rotating_token_endpoint(monkeypatch)
    failures = _keychain_failing_writes(monkeypatch)
    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )
    store = KeychainTokenStore(config)
    await store.store_tokens("access-0", "refresh-0")
    daemon = ShimDaemon(config, token_store=store, audit=MagicMock(spec=AuditWriter))
    failures.append(keyring.errors.PasswordSetError("the keychain is locked"))
    with pytest.raises(KeychainUnavailableError):
        await daemon._get_access_token(refresh_on_fail=True, rejected_token="access-0")
    await store.store_tokens("access-auth", "refresh-auth")  # `autogpt-shim auth`, meanwhile

    token = await daemon._get_access_token(refresh_on_fail=True, rejected_token="access-auth")

    assert sent == ["refresh-0", "refresh-auth"]
    assert token == "access-2"
    assert await store.get_refresh_token() == "refresh-2"


def _refresh_config(tmp_path: Path) -> ShimConfig:
    return ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("meanwhile", ["auth", "revoke"])
async def test_a_refresh_keeps_a_sign_in_change_made_during_its_exchange(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, meanwhile: str
) -> None:
    """A refresh stored its new pair over whatever the keychain held once the
    exchange came back: a sign-in `autogpt-shim auth` stored meanwhile was
    replaced by the older sign-in's refreshed pair, and a computer
    `autogpt-shim revoke` had just signed out was signed back in."""
    _keychain_failing_writes(monkeypatch)
    config = _refresh_config(tmp_path)

    async def another_process() -> None:
        other = KeychainTokenStore(config)
        if meanwhile == "auth":
            await other.store_tokens("access-auth", "refresh-auth")
        else:
            other.clear_tokens()

    sent = _rotating_token_endpoint(monkeypatch, before_answering=another_process)
    store = KeychainTokenStore(config)
    await store.store_tokens("access-0", "refresh-0")
    daemon = ShimDaemon(config, token_store=store, audit=MagicMock(spec=AuditWriter))

    token = await daemon._get_access_token(refresh_on_fail=True, rejected_token="access-0")

    assert sent == ["refresh-0"]
    if meanwhile == "auth":
        assert token == "access-auth"
        assert await store.get_refresh_token() == "refresh-auth"
    else:
        assert token is None
        assert await store.get_refresh_token() is None
    assert store.unsaved_refresh is None


@pytest.mark.asyncio
async def test_a_kept_pair_does_not_replace_a_sign_in_stored_after_the_retry_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry storing a kept pair compared the keychain's refresh token with
    the one its exchange used up, then wrote: a sign-in stored after that
    read was replaced by the kept pair."""
    import keyring.errors

    from autogpt_local_executor.daemon import KeychainUnavailableError

    sent = _rotating_token_endpoint(monkeypatch)
    failures = _keychain_failing_writes(monkeypatch)
    config = _refresh_config(tmp_path)
    store = KeychainTokenStore(config)
    await store.store_tokens("access-0", "refresh-0")
    daemon = ShimDaemon(config, token_store=store, audit=MagicMock(spec=AuditWriter))
    failures.append(keyring.errors.PasswordSetError("the keychain is locked"))
    with pytest.raises(KeychainUnavailableError):
        await daemon._get_access_token(refresh_on_fail=True, rejected_token="access-0")
    assert store.unsaved_refresh is not None
    read_refresh_token = store.get_refresh_token

    async def read_then_sign_in() -> str | None:
        refresh = await read_refresh_token()
        await KeychainTokenStore(config).store_tokens("access-auth", "refresh-auth")
        return refresh

    monkeypatch.setattr(store, "get_refresh_token", read_then_sign_in)
    token = await daemon._get_access_token(refresh_on_fail=True, rejected_token="access-0")

    assert sent == ["refresh-0"]
    assert token == "access-auth"
    assert await read_refresh_token() == "refresh-auth"
    assert store.unsaved_refresh is None


@pytest.mark.asyncio
async def test_a_refresh_outlives_the_session_stopped_while_it_waited_for_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chat's session stopped (its task cancelled, _stop_child_session)
    while its refresh waited for the token endpoint dropped the answer, though
    the platform had rotated the refresh token already: the keychain kept the
    used-up one, and the next session's refresh sent it again, was refused,
    and ended the sign-in."""
    _keychain_failing_writes(monkeypatch)
    config = _refresh_config(tmp_path)
    exchanged = asyncio.Event()
    answer = asyncio.Event()

    async def hold_the_answer() -> None:
        exchanged.set()
        await answer.wait()

    sent = _rotating_token_endpoint(monkeypatch, before_answering=hold_the_answer)
    store = KeychainTokenStore(config)
    await store.store_tokens("access-0", "refresh-0")
    lock = asyncio.Lock()
    stopped = ShimDaemon(
        config, token_store=store, audit=MagicMock(spec=AuditWriter), token_refresh_lock=lock
    )
    other = ShimDaemon(
        config, token_store=store, audit=MagicMock(spec=AuditWriter), token_refresh_lock=lock
    )
    refresh = asyncio.create_task(
        stopped._get_access_token(refresh_on_fail=True, rejected_token="access-0")
    )
    await asyncio.wait_for(exchanged.wait(), timeout=10)
    refresh.cancel()
    with pytest.raises(asyncio.CancelledError):
        await refresh
    answer.set()

    token = await asyncio.wait_for(
        other._get_access_token(refresh_on_fail=True, rejected_token="access-0"), timeout=10
    )

    assert sent == ["refresh-0"]
    assert token == "access-1"
    assert await store.get_refresh_token() == "refresh-1"


@pytest.mark.parametrize(
    ("failure", "permanent", "transient"),
    [
        ("corrupt", True, False),
        ("disk-full", False, True),
        ("read-only", True, False),
        ("keychain-locked", False, True),
        # No keychain at all yet, and no file key: one the session starts
        # after the service.
        ("no-keychain-yet", False, True),
        # The same with the passphrase set, and a log the keychain's key
        # signed, which a new file key would never verify.
        ("no-keychain-yet-signed-log", False, True),
        # PYTHON_KEYRING_BACKEND names a backend that isn't there: a
        # setting, which no restart fixes.
        ("keyring-backend-missing", True, False),
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
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path / "fallback"))
    if failure == "keychain-locked":

        def locked(*_args: object) -> None:
            raise keyring.errors.KeyringLocked("the keychain is locked")

        monkeypatch.setattr(keyring, "get_password", locked)
    elif failure.startswith("no-keychain-yet"):

        def missing(*_args: object) -> None:
            raise keyring.errors.NoKeyringError("no recommended backend")

        monkeypatch.setattr(keyring, "get_password", missing)
        if failure == "no-keychain-yet-signed-log":
            monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
            config.audit_log_path.write_text(
                '{"seq": 1, "ts": 1.0, "op": "X", "hmac": "00"}\n', encoding="utf-8"
            )
    elif failure == "keyring-backend-missing":
        import keyring.core

        monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.no_such_backend.Keyring")
        # keyring picks its backend once per process; have it pick again.
        monkeypatch.setattr(keyring.core, "_keyring_backend", None)
    else:
        monkeypatch.setattr(daemon_module, "get_or_create_audit_key", lambda _log=None: b"k" * 32)
    if failure == "corrupt":
        config.audit_log_path.write_text(
            '{"seq": 1, "ts": 1.0, "op": "X", "hmac": "00"}\n', encoding="utf-8"
        )
    elif failure in ("disk-full", "read-only"):
        code = errno.ENOSPC if failure == "disk-full" else errno.EROFS

        def refuse(**_kwargs: object) -> None:
            raise OSError(code, os.strerror(code))

        monkeypatch.setattr(daemon_module, "AuditWriter", refuse)

    with pytest.raises(DaemonPreflightError) as raised:
        ShimDaemon._build_audit_writer(config)

    assert raised.value.permanent is permanent
    assert getattr(raised.value, "transient", False) is transient
    # No stray file key for a later start to pick up instead of the keychain.
    assert not (tmp_path / "fallback" / "audit-key.enc").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "permanent", "transient"),
    [
        ("ENOSPC", False, True),
        ("EROFS", True, False),
        ("EIO", True, False),
        ("ENXIO", True, False),
        ("ENODEV", True, False),
    ],
)
async def test_a_failed_first_audit_write_says_whether_a_restart_can_fix_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str, permanent: bool, transient: bool
) -> None:
    """A full disk at the first audit write exited 78 like a bad config, so
    the systemd unit never restarted once space was freed. A failed or
    missing device counted as transient, so the unit restarted every five
    seconds for as long as the device stayed broken."""
    import errno
    import os

    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )
    number = getattr(errno, code)
    audit = MagicMock(spec=AuditWriter)
    audit.shim_start = AsyncMock(side_effect=OSError(number, os.strerror(number)))
    daemon = ShimDaemon(config, token_store=object(), audit=audit)
    monkeypatch.setattr(daemon, "_session", AsyncMock())

    with pytest.raises(daemon_module.DaemonPreflightError, match="initial SHIM_START") as raised:
        await daemon.run()

    assert raised.value.permanent is permanent
    assert raised.value.transient is transient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "platform", "permanent", "transient"),
    [
        ("EACCES", "linux", True, False),
        ("EROFS", "linux", True, False),
        ("ENOTDIR", "linux", True, False),
        # A file where the folder belongs.
        ("EEXIST", "linux", True, False),
        ("ENOSPC", "linux", False, True),
        ("EPERM", "linux", True, False),
        # A folder the app may not open yet (Files and Folders): a missing
        # permission, which launchd relaunches until it is granted.
        ("EPERM", "darwin", False, False),
    ],
)
async def test_a_legacy_allowed_root_that_cannot_be_created_says_whether_a_restart_can_fix_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    platform: str,
    permanent: bool,
    transient: bool,
) -> None:
    """A legacy session's allowed_root the shim couldn't create was a
    preflight error neither permanent nor transient, so `start` exited 78,
    which the LaunchAgent relaunches every few seconds for as long as the
    folder stays unwritable, and the systemd unit never retried a full disk."""
    import errno
    import os

    root = tmp_path / "root"
    config = ShimConfig(
        session_id="session-1",
        allowed_root=root,
        audit_log_path=tmp_path / "audit.log",
    )
    daemon = ShimDaemon(config, token_store=object(), audit=MagicMock(spec=AuditWriter))
    monkeypatch.setattr(daemon, "_preflight_or_raise", AsyncMock())
    monkeypatch.setattr(daemon_module.platform_info, "detect_platform", lambda: platform)
    number = getattr(errno, code)
    real_mkdir = Path.mkdir

    def mkdir(self: Path, *args: object, **kwargs: object) -> None:
        if self == root:
            raise OSError(number, os.strerror(number), str(self))
        real_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", mkdir)

    with pytest.raises(daemon_module.DaemonPreflightError, match="allowed_root") as raised:
        await daemon.run()

    assert raised.value.permanent is permanent
    assert raised.value.transient is transient


@pytest.mark.asyncio
async def test_run_retries_a_keychain_that_is_still_locked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keychain tests called _read_access_token alone, so a run() that
    stopped on KeychainUnavailableError, instead of retrying with backoff
    until the keychain unlocks, would still have passed them."""
    import keyring.errors

    reads = 0

    class Store:
        async def get_access_token(self) -> str:
            nonlocal reads
            reads += 1
            if reads == 1:
                raise keyring.errors.KeyringError("the keychain is still locked")
            return "the-token"

    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
        reconnect_base_delay=0.01,
        reconnect_jitter_seconds=0,
    )
    audit = MagicMock()
    for name in ("write", "shim_start", "shim_stop", "ws_connected", "ws_disconnected"):
        setattr(audit, name, AsyncMock())
    daemon = ShimDaemon(config, token_store=Store(), audit=audit)  # type: ignore[arg-type]
    tokens: list[str | None] = []

    async def session() -> None:
        tokens.append(await daemon._read_access_token())
        daemon._running = False

    monkeypatch.setattr(daemon, "_session", session)

    await asyncio.wait_for(daemon.run(), timeout=30)

    assert reads == 2
    assert tokens == ["the-token"]


@pytest.mark.asyncio
async def test_run_retries_a_refresh_the_token_endpoint_could_not_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refresh tests called _get_access_token alone, so a run() that
    stopped on TokenRefreshUnavailableError, instead of retrying with backoff
    until the endpoint answers, would still have passed them."""
    import httpx

    from autogpt_local_executor.auth import OAuthFlow

    refreshes = 0
    store_token = "stale"

    async def get_access_token() -> str:
        return store_token

    async def refresh(_flow) -> None:
        nonlocal refreshes, store_token
        refreshes += 1
        if refreshes == 1:
            request = httpx.Request("POST", "https://platform.example/api/oauth/token")
            raise httpx.ConnectError("unreachable", request=request)
        store_token = "fresh"

    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
        reconnect_base_delay=0.01,
        reconnect_jitter_seconds=0,
    )
    audit = MagicMock()
    for name in (
        "write",
        "shim_start",
        "shim_stop",
        "ws_connected",
        "ws_disconnected",
        "token_refreshed",
    ):
        setattr(audit, name, AsyncMock())
    daemon = ShimDaemon(
        config,
        token_store=SimpleNamespace(get_access_token=get_access_token),  # type: ignore[arg-type]
        audit=audit,
    )
    monkeypatch.setattr(OAuthFlow, "refresh_token", refresh)
    tokens: list[str | None] = []

    async def session() -> None:
        tokens.append(await daemon._get_access_token(refresh_on_fail=True, rejected_token="stale"))
        daemon._running = False

    monkeypatch.setattr(daemon, "_session", session)

    await asyncio.wait_for(daemon.run(), timeout=30)

    assert refreshes == 2
    assert tokens == ["fresh"]


@pytest.mark.asyncio
async def test_run_ends_for_a_restart_when_no_keychain_is_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A keychain that wasn't up when the sign-in was read was retried inside
    the process, but the keyring package picks its backend once per process,
    so a Secret Service that started later was never found. run() now ends
    with a transient preflight error, which `start` turns into 75 for the
    service manager to start a new process."""
    import keyring.errors

    from autogpt_local_executor.daemon import DaemonPreflightError

    reads = 0

    class Store:
        async def get_access_token(self) -> str:
            nonlocal reads
            reads += 1
            if reads == 1:
                raise keyring.errors.NoKeyringError("no recommended backend")
            return "the-token"

    config = ShimConfig(
        session_id="session-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
        reconnect_base_delay=0.01,
        reconnect_jitter_seconds=0,
    )
    audit = MagicMock()
    for name in ("write", "shim_start", "shim_stop", "ws_connected", "ws_disconnected"):
        setattr(audit, name, AsyncMock())
    daemon = ShimDaemon(config, token_store=Store(), audit=audit)  # type: ignore[arg-type]

    async def session() -> None:
        await daemon._read_access_token()
        daemon._running = False

    monkeypatch.setattr(daemon, "_session", session)

    with pytest.raises(DaemonPreflightError) as raised:
        await asyncio.wait_for(daemon.run(), timeout=30)

    assert raised.value.transient is True
    assert reads == 1


@pytest.mark.parametrize(
    ("code", "permanent", "transient"), [("ENOSPC", False, True), ("EROFS", True, False)]
)
def test_an_audit_key_lock_that_cannot_be_taken_says_whether_a_restart_can_fix_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str, permanent: bool, transient: bool
) -> None:
    """The lock a new audit key is made under failed as AuditKeyError, which
    startup takes for permanent, so a full disk stopped a supervised shim for
    good instead of exiting for a restart once space was freed."""
    import contextlib
    import errno
    import os

    import keyring

    from autogpt_local_executor import audit as audit_module

    config = ShimConfig(
        session_id="session-1", allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log"
    )
    monkeypatch.delenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", raising=False)
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path / "fallback"))
    monkeypatch.setattr(keyring, "get_password", lambda *_args: None)
    number = getattr(errno, code)

    @contextlib.contextmanager
    def refuse(_path: Path, **_kwargs: object):
        raise OSError(number, os.strerror(number))
        yield

    monkeypatch.setattr(audit_module, "_file_lock", refuse)

    with pytest.raises(daemon_module.DaemonPreflightError) as raised:
        ShimDaemon._build_audit_writer(config)

    assert raised.value.permanent is permanent
    assert getattr(raised.value, "transient", False) is transient


@pytest.mark.parametrize(
    ("code", "permanent", "transient"),
    [
        ("ENOSPC", False, True),
        ("EROFS", True, False),
        # A file where the folder belongs.
        ("EEXIST", True, False),
    ],
)
def test_a_fallback_audit_key_folder_that_cannot_be_made_says_whether_a_restart_can_fix_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: str, permanent: bool, transient: bool
) -> None:
    """With no keychain and the file fallback's passphrase set, a full disk
    making the folder for the fallback's audit key raised a bare OSError,
    which startup takes for permanent, so a supervised shim stopped for good
    instead of exiting for a restart once space was freed."""
    import errno
    import os

    import keyring
    import keyring.errors

    config = ShimConfig(
        session_id="session-1", allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log"
    )
    fallback = tmp_path / "fallback"
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(fallback))

    def missing(*_args: object) -> None:
        raise keyring.errors.NoKeyringError("no recommended backend")

    monkeypatch.setattr(keyring, "get_password", missing)
    number = getattr(errno, code)
    real_mkdir = Path.mkdir

    def mkdir(self: Path, *args: object, **kwargs: object) -> None:
        if self == fallback:
            raise OSError(number, os.strerror(number), str(self))
        real_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "mkdir", mkdir)

    with pytest.raises(daemon_module.DaemonPreflightError) as raised:
        ShimDaemon._build_audit_writer(config)

    assert raised.value.permanent is permanent
    assert raised.value.transient is transient


@pytest.mark.parametrize(
    ("failure", "permanent", "transient"),
    [
        # A network mount timing out, which can come back.
        ("ETIMEDOUT", False, True),
        ("EACCES", True, False),
        # Contents that aren't a key, which no restart changes.
        ("malformed", True, False),
    ],
)
def test_a_fallback_audit_key_that_cannot_be_read_says_whether_a_restart_can_fix_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, permanent: bool, transient: bool
) -> None:
    """With no keychain, a storage error reading the file fallback's audit
    key was reported as a malformed key, which startup takes for permanent,
    so a supervised shim stopped for good when its storage timed out
    instead of exiting for a restart once it came back."""
    import errno
    import os

    import keyring
    import keyring.errors

    config = ShimConfig(
        session_id="session-1", allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log"
    )
    fallback = tmp_path / "fallback"
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(fallback))

    def missing(*_args: object) -> None:
        raise keyring.errors.NoKeyringError("no recommended backend")

    monkeypatch.setattr(keyring, "get_password", missing)
    fallback.mkdir()
    key_file = fallback / "audit-key.enc"
    if failure == "malformed":
        key_file.write_text("not-an-encrypted-key", encoding="utf-8")
    else:
        key_file.write_text("00" * 16 + ":" + "00" * 32, encoding="utf-8")
        number = getattr(errno, failure)
        real_read_text = Path.read_text

        def read_text(self: Path, *args: Any, **kwargs: Any) -> str:
            if self == key_file:
                raise OSError(number, os.strerror(number), str(self))
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", read_text)

    with pytest.raises(daemon_module.DaemonPreflightError) as raised:
        ShimDaemon._build_audit_writer(config)

    assert raised.value.permanent is permanent
    assert raised.value.transient is transient
    assert key_file.exists()


def test_a_full_disk_writing_the_fallback_audit_key_leaves_no_key_for_later_starts_to_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk part way through writing a new file fallback key raised a
    bare OSError, which startup takes for permanent, and left the key file
    cut short, which every later start refused as malformed. The start exits
    for a restart instead, leaving no key file behind, and the next start,
    with space freed, makes the key."""
    import errno
    import io

    import keyring
    import keyring.errors

    from autogpt_local_executor import audit as audit_module

    config = ShimConfig(
        session_id="session-1", allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log"
    )
    fallback = tmp_path / "fallback"
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(fallback))

    def missing(*_args: object) -> None:
        raise keyring.errors.NoKeyringError("no recommended backend")

    monkeypatch.setattr(keyring, "get_password", missing)

    class FullDisk:
        """A file whose writes fail the way a full disk does: part of the
        data lands, then the disk is full."""

        def __init__(self, file: Any) -> None:
            self._file = file

        def write(self, data: Any) -> int:
            self._file.write(data[:1])
            self._file.flush()
            raise OSError(errno.ENOSPC, "No space left on device")

        def __getattr__(self, name: str) -> Any:
            return getattr(self._file, name)

        def __enter__(self) -> FullDisk:
            return self

        def __exit__(self, *_exc: object) -> bool:
            self._file.close()
            return False

    real_open = io.open

    def full_disk_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = real_open(file, mode, *args, **kwargs)
        return FullDisk(handle) if "w" in mode else handle

    with monkeypatch.context() as full_disk:
        full_disk.setattr(io, "open", full_disk_open)
        with pytest.raises(daemon_module.DaemonPreflightError) as raised:
            ShimDaemon._build_audit_writer(config)

    assert raised.value.transient is True
    # Nothing a later start would read as a malformed key, or leave lying about.
    assert [path.name for path in fallback.iterdir()] == [".audit-key.lock"]
    writer = ShimDaemon._build_audit_writer(config)
    assert (fallback / "audit-key.enc").is_file()
    assert writer.audit_key == audit_module.get_or_create_audit_key(config.audit_log_path)
