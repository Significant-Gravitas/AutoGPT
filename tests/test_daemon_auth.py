"""WebSocket authentication retry and session binding tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from websockets.exceptions import InvalidStatus

import autogpt_local_executor.daemon as daemon_module
from autogpt_local_executor.audit import AuditWriter
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
    monkeypatch.setattr(daemon_module.websockets, "connect", connect)
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
