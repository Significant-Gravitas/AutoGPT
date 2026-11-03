"""Tests for shim backpressure signaling (#38).

Covers:
  * StatusMessage / StatusPayload wire shape (round-trip via parse_message).
  * `pending_capacity` envelope field on every response.
  * _build_status_message snapshots the right counters.
  * _dispatch stamps response.pending_capacity correctly under various
    fake in-flight states.
  * Full → not-full capacity edge emits an unsolicited STATUS frame
    immediately after the response, without waiting for the periodic tick.

All tests mock the WebSocket / handlers — no live network, no real handlers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.daemon import (
    STATUS_INTERVAL_SECONDS,
    ShimDaemon,
)
from autogpt_local_executor.protocol import (
    MAX_WEBSOCKET_MESSAGE_BYTES,
    AckMessage,
    AckPayload,
    CommandResultMessage,
    CommandResultPayload,
    MessageType,
    StatusMessage,
    StatusPayload,
    dump_message,
    make_ack,
    new_id,
    now_ts,
    parse_message,
)


class _FakeWS:
    """Minimal WS — captures sent frames."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send(self, raw: str) -> None:
        if self.closed:
            raise RuntimeError("send after close")
        self.sent.append(raw)

    async def close(self, **_kw: Any) -> None:
        self.closed = True


@pytest.fixture()
def shim_config(tmp_path: Path) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path / "workspace",
        audit_log_path=tmp_path / "audit.log",
        session_id="test-session",
        machine_id="test-machine",
        max_concurrent=4,
    )


@pytest.fixture()
def daemon(shim_config: ShimConfig) -> ShimDaemon:
    audit = AuditWriter(path=shim_config.audit_log_path, audit_key=b"a" * 32)
    return ShimDaemon(shim_config, token_store=object(), audit=audit)


# ── Wire shape ───────────────────────────────────────────────────────────────


def test_status_frame_round_trips() -> None:
    msg = StatusMessage(
        id=new_id(),
        ts=now_ts(),
        payload=StatusPayload(
            in_flight=2,
            max_concurrent=4,
            queue_depth=1,
            audit_log_bytes=1024,
            uptime_seconds=12.5,
        ),
    )
    msg.pending_capacity = 2
    raw = dump_message(msg)
    again = parse_message(raw)
    assert isinstance(again, StatusMessage)
    assert again.type == MessageType.STATUS
    assert again.payload.in_flight == 2
    assert again.payload.max_concurrent == 4
    assert again.pending_capacity == 2


def test_pending_capacity_on_ack_envelope_serializes_when_set() -> None:
    ack = make_ack("x")
    ack.pending_capacity = 3
    payload = json.loads(dump_message(ack))
    assert payload["pending_capacity"] == 3


def test_pending_capacity_defaults_to_null() -> None:
    ack = make_ack("x")
    payload = json.loads(dump_message(ack))
    assert payload["pending_capacity"] is None


def test_legacy_response_without_pending_capacity_still_parses() -> None:
    """Forward-compat for receivers: a response missing pending_capacity
    parses fine (the field defaults to None)."""
    raw = json.dumps(
        {
            "type": "ACK",
            "id": "x",
            "ts": 1.0,
            "payload": {"ok": True},
        }
    )
    msg = parse_message(raw)
    assert isinstance(msg, AckMessage)
    assert msg.pending_capacity is None


# ── _build_status_message ────────────────────────────────────────────────────


def test_build_status_message_snapshots_counters(daemon: ShimDaemon) -> None:
    daemon._in_flight = 2
    daemon._queue_depth = 1
    daemon._session_started_at = now_ts() - 10.0
    msg = daemon._build_status_message()
    assert isinstance(msg, StatusMessage)
    assert msg.payload.in_flight == 2
    assert msg.payload.queue_depth == 1
    assert msg.payload.max_concurrent == daemon.config.max_concurrent
    # Queued work consumes admission capacity too.
    assert msg.pending_capacity == daemon.config.max_concurrent - 3
    # Uptime is positive (we set started_at 10s ago).
    assert msg.payload.uptime_seconds >= 10.0


def test_build_status_message_handles_missing_session_start(daemon: ShimDaemon) -> None:
    """Before the WS session begins (or after it ends), uptime is 0 rather
    than blowing up on None arithmetic."""
    daemon._session_started_at = None
    msg = daemon._build_status_message()
    assert msg.payload.uptime_seconds == 0.0


def test_build_status_message_includes_audit_log_size(daemon: ShimDaemon, tmp_path: Path) -> None:
    """audit_log_bytes reflects the on-disk size if the file exists."""
    daemon.config.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    daemon.config.audit_log_path.write_bytes(b"hello world\n")
    msg = daemon._build_status_message()
    assert msg.payload.audit_log_bytes == len(b"hello world\n")


def test_build_status_message_missing_audit_log_reports_zero(daemon: ShimDaemon) -> None:
    """A missing audit log file reports 0 bytes rather than raising."""
    if daemon.config.audit_log_path.exists():
        daemon.config.audit_log_path.unlink()
    msg = daemon._build_status_message()
    assert msg.payload.audit_log_bytes == 0


# ── _dispatch stamps pending_capacity ───────────────────────────────────────


async def test_dispatch_stamps_pending_capacity_on_response(
    daemon: ShimDaemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After _dispatch runs a handler the response carries
    pending_capacity = max_concurrent (we just released our slot)."""
    daemon._semaphore = asyncio.Semaphore(daemon.config.max_concurrent)

    response = AckMessage(id="reqid", ts=now_ts(), payload=AckPayload(ok=True))

    async def fake_handle(_msg: Any, *, send: Any = None) -> AckMessage:
        return response

    monkeypatch.setattr(daemon, "_handle", fake_handle)

    ws = _FakeWS()
    # Fake inbound message (any type, _handle is mocked).
    dummy_in = make_ack("ignored")
    await daemon._dispatch(ws, dummy_in)

    assert len(ws.sent) == 1
    sent = json.loads(ws.sent[0])
    # We held + released one slot — so max_concurrent slots are free again.
    assert sent["pending_capacity"] == daemon.config.max_concurrent
    assert daemon._in_flight == 0
    assert daemon._queue_depth == 0


async def test_dispatch_emits_status_on_full_to_not_full_edge(
    daemon: ShimDaemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If our task was the one occupying the LAST free slot at finish,
    a STATUS frame follows the response on the wire."""
    daemon._semaphore = asyncio.Semaphore(daemon.config.max_concurrent)
    # Pretend max_concurrent - 1 other tasks were in flight when ours
    # entered. The fake handle leaves them in flight (no decrement on
    # the way out), so when our finally block computes the "was at cap?"
    # check, _in_flight is still at max_concurrent.
    other_in_flight = daemon.config.max_concurrent - 1
    daemon._in_flight = other_in_flight

    response = AckMessage(id="reqid", ts=now_ts(), payload=AckPayload(ok=True))

    async def fake_handle(_msg: Any, *, send: Any = None) -> AckMessage:
        return response

    monkeypatch.setattr(daemon, "_handle", fake_handle)

    ws = _FakeWS()
    dummy_in = make_ack("ignored")
    await daemon._dispatch(ws, dummy_in)

    # Two frames went out: the response THEN a STATUS.
    assert len(ws.sent) == 2
    response_frame = json.loads(ws.sent[0])
    status_frame = json.loads(ws.sent[1])
    assert response_frame["type"] == "ACK"
    # pending_capacity on the response reflects free slots = 1 (our slot,
    # the other 3 still in flight in the fake setup).
    assert response_frame["pending_capacity"] == 1
    assert status_frame["type"] == "STATUS"
    assert status_frame["payload"]["max_concurrent"] == daemon.config.max_concurrent
    assert status_frame["payload"]["in_flight"] == other_in_flight
    assert status_frame["pending_capacity"] == 1


async def test_dispatch_does_not_emit_status_when_not_at_edge(
    daemon: ShimDaemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Normal under-cap finish: only the response goes out (no STATUS)."""
    daemon._semaphore = asyncio.Semaphore(daemon.config.max_concurrent)

    response = AckMessage(id="reqid", ts=now_ts(), payload=AckPayload(ok=True))

    async def fake_handle(_msg: Any, *, send: Any = None) -> AckMessage:
        return response

    monkeypatch.setattr(daemon, "_handle", fake_handle)
    ws = _FakeWS()
    await daemon._dispatch(ws, make_ack("ignored"))
    assert len(ws.sent) == 1
    assert json.loads(ws.sent[0])["type"] == "ACK"


async def test_on_frame_rejects_overload_without_unbounded_task_queue(
    daemon: ShimDaemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon.config.max_concurrent = 1
    daemon._semaphore = asyncio.Semaphore(1)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow_handle(_msg: Any, *, send: Any = None) -> AckMessage:
        entered.set()
        await release.wait()
        return make_ack("first")

    monkeypatch.setattr(daemon, "_handle", slow_handle)
    ws = _FakeWS()

    await daemon._on_frame(ws, dump_message(make_ack("first")))
    await entered.wait()
    await daemon._on_frame(ws, dump_message(make_ack("second")))

    assert daemon._pending_requests == 1
    assert len(daemon._dispatch_tasks) == 1
    assert len(ws.sent) == 1
    overloaded = parse_message(ws.sent[0])
    assert overloaded.type == MessageType.ERROR
    assert overloaded.payload.code == "SHIM_OVERLOADED"

    release.set()
    await asyncio.gather(*tuple(daemon._dispatch_tasks))


async def test_cancel_dispatch_tasks_resets_capacity_and_cancels_handlers(
    daemon: ShimDaemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon.config.max_concurrent = 1
    daemon._semaphore = asyncio.Semaphore(1)
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_handle(_msg: Any, *, send: Any = None) -> AckMessage:
        entered.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    monkeypatch.setattr(daemon, "_handle", slow_handle)
    ws = _FakeWS()
    await daemon._on_frame(ws, dump_message(make_ack("first")))
    await entered.wait()

    await daemon._cancel_dispatch_tasks()

    assert cancelled.is_set()
    assert daemon._pending_requests == 0
    assert daemon._queue_depth == 0
    assert daemon._in_flight == 0
    assert not daemon._dispatch_tasks
    assert daemon._available_capacity() == 1


async def test_send_frame_replaces_oversized_response_with_correlated_error(
    daemon: ShimDaemon,
) -> None:
    ws = _FakeWS()
    response = CommandResultMessage(
        id="oversized",
        ts=now_ts(),
        payload=CommandResultPayload(
            stdout="\0" * (MAX_WEBSOCKET_MESSAGE_BYTES // 6),
            stderr="",
            exit_code=0,
            timed_out=False,
            duration_seconds=0,
        ),
    )

    sent_original = await daemon._send_frame(ws, response)

    assert sent_original is False
    assert len(ws.sent) == 1
    error = parse_message(ws.sent[0])
    assert error.type == MessageType.ERROR
    assert error.id == "oversized"
    assert len(ws.sent[0].encode("utf-8")) < MAX_WEBSOCKET_MESSAGE_BYTES


# ── Periodic ticker ─────────────────────────────────────────────────────────


async def test_status_interval_constant_is_30s() -> None:
    """Lock the spec'd cadence — platform-side consumer relies on it."""
    assert STATUS_INTERVAL_SECONDS == 30.0


async def test_status_ticker_emits_on_interval(
    daemon: ShimDaemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Speed up the cadence and verify the ticker emits STATUS frames."""
    daemon._running = True
    monkeypatch.setattr("autogpt_local_executor.daemon.STATUS_INTERVAL_SECONDS", 0.01)
    daemon._session_started_at = now_ts()
    ws = _FakeWS()
    daemon._ws = ws

    task = asyncio.create_task(daemon._status_ticker(ws))
    # Let the ticker fire a couple of times.
    await asyncio.sleep(0.05)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # We should have several STATUS frames; assert at least 2 to avoid flakes.
    status_frames = [f for f in ws.sent if json.loads(f)["type"] == "STATUS"]
    assert len(status_frames) >= 2
    for raw in status_frames:
        msg = parse_message(raw)
        assert isinstance(msg, StatusMessage)


async def test_status_ticker_exits_when_ws_swapped(
    daemon: ShimDaemon, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ticker for the old WS quits cleanly once daemon._ws is replaced
    (mirrors reconnect)."""
    daemon._running = True
    monkeypatch.setattr("autogpt_local_executor.daemon.STATUS_INTERVAL_SECONDS", 0.01)
    daemon._session_started_at = now_ts()
    old_ws = _FakeWS()
    daemon._ws = old_ws
    task = asyncio.create_task(daemon._status_ticker(old_ws))
    await asyncio.sleep(0.02)
    # Simulate reconnect: swap the daemon's WS to a new one.
    daemon._ws = _FakeWS()
    # Wait for the ticker to notice and exit.
    await asyncio.wait_for(task, timeout=1.0)
    # And the old WS should not receive new frames after the swap.
    pre_swap_count = len(old_ws.sent)
    await asyncio.sleep(0.05)
    assert len(old_ws.sent) == pre_swap_count
