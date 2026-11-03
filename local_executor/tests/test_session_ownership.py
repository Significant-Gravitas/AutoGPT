"""Tests for shim-side session ownership (#36).

Covers:
  * SESSION_REVOKED frame: shim audits, closes WS, disables auto-reconnect,
    and does NOT send further frames after receipt.
  * WS close-code translation (4426/4427/4428/4429): each fatal application
    code sets _disable_reconnect and emits a SESSION_REVOKED audit row.
  * Non-fatal WS close codes (e.g. 1011) do NOT disable reconnect.

All tests mock the WebSocket — no live network.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import websockets

from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.daemon import (
    WS_CLOSE_PLATFORM_SHUTDOWN,
    WS_CLOSE_SESSION_REVOKED,
    WS_CLOSE_SESSION_TAKEN_OVER,
    ShimDaemon,
)
from autogpt_local_executor.protocol import (
    SESSION_REVOKED_REASONS,
    HelloAckMessage,
    HelloAckPayload,
    MessageType,
    SessionRevokedMessage,
    SessionRevokedPayload,
    dump_message,
    new_id,
    now_ts,
)

# ── Fakes ────────────────────────────────────────────────────────────────────


class _FakeWebSocketCloseCode:
    """websockets.ConnectionClosed-compatible carrier for our tests.

    We construct a real ConnectionClosed with an Rcvd close frame so the
    daemon's `getattr(exc, "code", None)` finds the expected wire code.
    """


def _make_connection_closed(code: int, reason: str = "") -> websockets.exceptions.ConnectionClosed:
    from websockets.frames import Close

    rcvd = Close(code=code, reason=reason)
    return websockets.exceptions.ConnectionClosed(rcvd=rcvd, sent=None)


class _FakeWebSocket:
    """WS double that supports both async-iteration (for _run_session) and
    direct recv()/close(). Inbound items can be raw strings (yielded as
    frames) OR a websockets.exceptions.ConnectionClosed exception (raised
    when reached, simulating server-initiated close)."""

    def __init__(self, inbound: list[Any]) -> None:
        self._inbound = list(inbound)
        self.sent: list[str] = []
        self.close_calls: list[dict[str, Any]] = []
        self._closed = False

    async def send(self, raw: str) -> None:
        if self._closed:
            raise RuntimeError("send after close")
        self.sent.append(raw)

    async def recv(self) -> str:
        if not self._inbound:
            await asyncio.sleep(3600)
            raise AssertionError("recv blocked unexpectedly")
        nxt = self._inbound.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        payload = json.loads(nxt)
        if payload.get("type") == "HELLO_ACK" and payload.get("id") == "__HELLO_ID__":
            payload["id"] = json.loads(self.sent[0])["id"]
            return json.dumps(payload)
        return nxt

    def __aiter__(self) -> _FakeWebSocket:
        return self

    async def __anext__(self) -> str:
        if not self._inbound:
            raise StopAsyncIteration
        nxt = self._inbound.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    async def close(self, *, code: int = 1000, reason: str = "") -> None:
        self.close_calls.append({"code": code, "reason": reason})
        self._closed = True


@pytest.fixture()
def shim_config(tmp_path: Path) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path / "workspace",
        audit_log_path=tmp_path / "audit.log",
        session_id="test-session",
        machine_id="test-machine",
    )


class _RecordingAudit:
    """Audit double — captures (op, details) tuples."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def set_machine_id(self, _mid: str) -> None:
        pass

    def set_session_id(self, _sid: str) -> None:
        pass

    async def write(
        self, op: str, *, request_id: Any = None, details: dict[str, Any], result: Any = None
    ) -> None:
        self.events.append((op, details))

    async def shim_start(self, *_a: Any, **_k: Any) -> None:
        pass

    async def shim_stop(self, *_a: Any, **_k: Any) -> None:
        pass

    async def ws_connected(self, *_a: Any, **_k: Any) -> None:
        pass

    async def ws_disconnected(self, *_a: Any, **_k: Any) -> None:
        pass

    async def config_reloaded(self, *_a: Any, **_k: Any) -> None:
        pass

    async def token_refreshed(self, *_a: Any, **_k: Any) -> None:
        pass

    async def jail_violation(self, *_a: Any, **_k: Any) -> None:
        pass


@pytest.fixture()
def audit() -> _RecordingAudit:
    return _RecordingAudit()


@pytest.fixture()
def daemon(shim_config: ShimConfig, audit: _RecordingAudit) -> ShimDaemon:
    return ShimDaemon(shim_config, token_store=None, audit=audit)


def _hello_ack() -> str:
    payload = HelloAckPayload(
        session_id="test-session",
        granted_capabilities=["files"],
        max_file_size_bytes=1024,
        command_timeout_seconds=30,
        max_concurrent=4,
    )
    return dump_message(HelloAckMessage(id="__HELLO_ID__", ts=now_ts(), payload=payload))


def _session_revoked_frame(
    reason: str = "another_shim_connected", new_mid: str | None = None
) -> str:
    return dump_message(
        SessionRevokedMessage(
            id=new_id(),
            ts=now_ts(),
            payload=SessionRevokedPayload(reason=reason, new_shim_machine_id=new_mid),
        )
    )


# ── SESSION_REVOKED frame handling ───────────────────────────────────────────


async def test_session_revoked_audits_closes_disables_reconnect(
    daemon: ShimDaemon, audit: _RecordingAudit
) -> None:
    """Receipt of SESSION_REVOKED → audit row, ws.close called, no further sends."""
    ws = _FakeWebSocket([_session_revoked_frame(new_mid="other-machine-1")])
    # Bypass handshake by pre-seeding the semaphore.
    daemon._semaphore = asyncio.Semaphore(4)

    raw = await ws.__anext__()
    await daemon._on_frame(ws, raw)

    assert daemon._disable_reconnect is True
    # Exactly one close call, with the SESSION_REVOKED close code.
    assert len(ws.close_calls) == 1
    assert ws.close_calls[0]["code"] == WS_CLOSE_SESSION_REVOKED
    # Nothing was sent (no PONG, no ack — silent close).
    assert ws.sent == []
    # SESSION_REVOKED audit row exists with the right reason + machine.
    revoked = [(op, d) for op, d in audit.events if op == "SESSION_REVOKED"]
    assert len(revoked) == 1
    _, details = revoked[0]
    assert details["reason"] == "another_shim_connected"
    assert details["source"] == "frame"
    assert details["new_shim_machine_id"] == "other-machine-1"


@pytest.mark.parametrize("reason", SESSION_REVOKED_REASONS)
async def test_session_revoked_accepts_each_spec_reason(
    daemon: ShimDaemon, audit: _RecordingAudit, reason: str
) -> None:
    """The three reasons in the spec all flow through unchanged."""
    ws = _FakeWebSocket([_session_revoked_frame(reason=reason)])
    daemon._semaphore = asyncio.Semaphore(4)
    await daemon._on_frame(ws, await ws.__anext__())
    revoked = [d for op, d in audit.events if op == "SESSION_REVOKED"]
    assert revoked[-1]["reason"] == reason


async def test_session_revoked_tolerates_unknown_reason(
    daemon: ShimDaemon, audit: _RecordingAudit
) -> None:
    """Forward-compat: an unknown future reason still parses and revokes."""
    ws = _FakeWebSocket([_session_revoked_frame(reason="future_reason_xyz")])
    daemon._semaphore = asyncio.Semaphore(4)
    await daemon._on_frame(ws, await ws.__anext__())
    assert daemon._disable_reconnect is True
    revoked = [d for op, d in audit.events if op == "SESSION_REVOKED"]
    assert revoked[-1]["reason"] == "future_reason_xyz"


async def test_session_revoked_is_a_recognized_message_type() -> None:
    """The new MessageType enum value round-trips through parse_message."""
    from autogpt_local_executor.protocol import parse_message

    raw = _session_revoked_frame()
    msg = parse_message(raw)
    assert isinstance(msg, SessionRevokedMessage)
    assert msg.type == MessageType.SESSION_REVOKED


# ── WS close-code semantics ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "close_code,expected_reason_label",
    [
        (4427, "session_taken_over"),
        (4428, "session_revoked"),
        (4429, "platform_shutdown"),
    ],
)
async def test_fatal_close_codes_disable_reconnect(
    daemon: ShimDaemon,
    audit: _RecordingAudit,
    close_code: int,
    expected_reason_label: str,
) -> None:
    """When the server closes with one of the fatal app codes, the shim
    sets _disable_reconnect and writes a SESSION_REVOKED audit row tagged
    with source=ws_close_code."""
    ws = _FakeWebSocket(
        inbound=[
            _hello_ack(),
            _make_connection_closed(code=close_code, reason="bye"),
        ]
    )
    with pytest.raises(websockets.exceptions.ConnectionClosed):
        await daemon._run_session(ws)
    assert daemon._disable_reconnect is True
    revoked = [d for op, d in audit.events if op == "SESSION_REVOKED"]
    assert len(revoked) == 1
    assert revoked[0]["source"] == "ws_close_code"
    assert revoked[0]["close_code"] == close_code
    assert revoked[0]["reason"] == expected_reason_label


@pytest.mark.parametrize("close_code", [1011, 4401])
async def test_non_fatal_close_code_does_NOT_disable_reconnect(
    daemon: ShimDaemon, audit: _RecordingAudit, close_code: int
) -> None:
    """Transient and access-token-expiry closes allow reconnect and refresh."""
    ws = _FakeWebSocket(
        inbound=[
            _hello_ack(),
            _make_connection_closed(code=close_code, reason="retryable"),
        ]
    )
    with pytest.raises(websockets.exceptions.ConnectionClosed):
        await daemon._run_session(ws)
    assert daemon._disable_reconnect is False
    # And no SESSION_REVOKED audit row — that's reserved for fatal codes.
    revoked = [d for op, d in audit.events if op == "SESSION_REVOKED"]
    assert revoked == []


async def test_fatal_close_code_4426_disables_reconnect(
    daemon: ShimDaemon, audit: _RecordingAudit
) -> None:
    """Cross-cutting: a 4426 close (already covered by Task A's version
    mismatch path) hitting us mid-session ALSO maps through the close-code
    table and disables reconnect — defense in depth."""
    ws = _FakeWebSocket(
        inbound=[
            _hello_ack(),
            _make_connection_closed(code=4426, reason="version drift"),
        ]
    )
    with pytest.raises(websockets.exceptions.ConnectionClosed):
        await daemon._run_session(ws)
    assert daemon._disable_reconnect is True
    revoked = [d for op, d in audit.events if op == "SESSION_REVOKED"]
    assert revoked[0]["reason"] == "protocol_version_mismatch"


async def test_close_code_constants_match_spec() -> None:
    """Lock the wire numbers — these are spec'd and platform code reads them."""
    assert WS_CLOSE_SESSION_TAKEN_OVER == 4427
    assert WS_CLOSE_SESSION_REVOKED == 4428
    assert WS_CLOSE_PLATFORM_SHUTDOWN == 4429


# ── Cross-cutting: SESSION_REVOKED frame, then attempt to send ──────────────


async def test_session_revoked_blocks_further_sends(
    daemon: ShimDaemon, audit: _RecordingAudit
) -> None:
    """Once SESSION_REVOKED is handled the WS is closed; any subsequent
    attempt to send a frame raises — proving the shim isn't going to
    accidentally emit one more response in flight."""
    ws = _FakeWebSocket([_session_revoked_frame()])
    daemon._semaphore = asyncio.Semaphore(4)
    await daemon._on_frame(ws, await ws.__anext__())

    with pytest.raises(RuntimeError):
        await ws.send(json.dumps({"type": "PONG"}))
