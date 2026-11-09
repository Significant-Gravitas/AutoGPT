from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from autogpt_local_executor.audit import AuditWriter, SessionAuditWriter
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.daemon import ShimDaemon, _ActiveChild
from autogpt_local_executor.directory_browser import DirectoryBrowser
from autogpt_local_executor.protocol import (
    ActivateSessionMessage,
    ActivateSessionPayload,
    AttachSessionMessage,
    AttachSessionPayload,
    DetachSessionMessage,
    DetachSessionPayload,
    DirectoryListRequestMessage,
    DirectoryListRequestPayload,
    ErrorCode,
    ErrorMessage,
    FileStatMessage,
    FileStatPayload,
    HelloAckMessage,
    HelloAckPayload,
    MessageType,
    RestoreSessionMessage,
    RestoreSessionPayload,
    SessionActivatedMessage,
    SessionAttachedMessage,
    SessionRestoredMessage,
    dump_message,
    make_ack,
    now_ts,
)


def _control_daemon(tmp_path: Path, *, audit_key: bytes = b"a" * 32) -> ShimDaemon:
    config = ShimConfig(
        session_id=None,
        machine_id="machine-1",
        allowed_root=tmp_path / "unused-global-root",
        audit_log_path=tmp_path / "audit.log",
    )
    audit = AuditWriter(config.audit_log_path, audit_key, machine_id=config.machine_id)
    daemon = ShimDaemon(config, token_store=object(), audit=audit)
    daemon._directory_browser = DirectoryBrowser(platform_name="linux", home=tmp_path)
    daemon._granted_capabilities = frozenset({"directory_browse"})
    return daemon


async def _browse_to_child(daemon: ShimDaemon, child_name: str):
    start = DirectoryListRequestMessage(
        id="list-roots",
        ts=now_ts(),
        payload=DirectoryListRequestPayload(),
    )
    roots = await daemon._handle(start)
    assert roots is not None and roots.type == MessageType.DIRECTORY_LIST_RESPONSE
    home = roots.payload.entries[0]
    listing = await daemon._handle(
        DirectoryListRequestMessage(
            id="list-home",
            ts=now_ts(),
            payload=DirectoryListRequestPayload(
                browse_id=roots.payload.browse_id,
                directory_ref=home.directory_ref,
            ),
        )
    )
    assert listing is not None and listing.type == MessageType.DIRECTORY_LIST_RESPONSE
    child = next(entry for entry in listing.payload.entries if entry.name == child_name)
    return listing.payload.browse_id, child.directory_ref


@pytest.mark.asyncio
async def test_control_attach_uses_host_ref_and_does_not_mutate_global_root(tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    daemon = _control_daemon(tmp_path)
    browse_id, directory_ref = await _browse_to_child(daemon, "selected")
    original_global_root = daemon.config.allowed_root

    response = await daemon._handle(
        AttachSessionMessage(
            id="attach",
            ts=now_ts(),
            payload=AttachSessionPayload(
                session_id="session-1",
                browse_id=browse_id,
                directory_ref=directory_ref,
                expected_connection_id="platform-generation",
            ),
        )
    )

    assert isinstance(response, SessionAttachedMessage)
    assert response.payload.allowed_root == str(selected.resolve())
    assert response.payload.revision == 1
    assert daemon.config.allowed_root == original_global_root
    assert daemon._root_grants.verify(response.payload.root_grant, "session-1").allowed_root == (
        selected.resolve()
    )
    with pytest.raises(Exception):
        assert daemon._directory_browser is not None
        daemon._directory_browser.resolve(browse_id, directory_ref)

    records = [json.loads(line) for line in daemon.audit.path.read_text().splitlines()]
    list_records = [record for record in records if record["op"] == "DIRECTORY_LIST"]
    assert list_records
    assert all("entries" not in record["details"] for record in list_records)


@pytest.mark.asyncio
async def test_control_connection_rejects_data_plane_operations(tmp_path: Path) -> None:
    daemon = _control_daemon(tmp_path)
    daemon._granted_capabilities = frozenset({"directory_browse", "files"})

    response = await daemon._handle(
        FileStatMessage(
            id="stat",
            ts=now_ts(),
            payload=FileStatPayload(path=str(tmp_path)),
        )
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.FEATURE_NOT_SUPPORTED


@pytest.mark.asyncio
async def test_reattach_increments_root_revision(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    daemon = _control_daemon(tmp_path)

    first_browse, first_ref = await _browse_to_child(daemon, "first")
    first = await daemon._handle(
        AttachSessionMessage(
            id="attach-1",
            ts=now_ts(),
            payload=AttachSessionPayload(
                session_id="session-1",
                browse_id=first_browse,
                directory_ref=first_ref,
            ),
        )
    )
    second_browse, second_ref = await _browse_to_child(daemon, "second")
    second = await daemon._handle(
        AttachSessionMessage(
            id="attach-2",
            ts=now_ts(),
            payload=AttachSessionPayload(
                session_id="session-1",
                browse_id=second_browse,
                directory_ref=second_ref,
            ),
        )
    )

    assert isinstance(first, SessionAttachedMessage)
    assert isinstance(second, SessionAttachedMessage)
    assert first.payload.revision == 1
    assert second.payload.revision == 2
    assert second.payload.allowed_root == str(second_root.resolve())


@pytest.mark.asyncio
async def test_activate_revalidates_directory_fingerprint(tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    daemon = _control_daemon(tmp_path)
    browse_id, directory_ref = await _browse_to_child(daemon, "selected")
    attached = await daemon._handle(
        AttachSessionMessage(
            id="attach",
            ts=now_ts(),
            payload=AttachSessionPayload(
                session_id="session-1",
                browse_id=browse_id,
                directory_ref=directory_ref,
            ),
        )
    )
    assert isinstance(attached, SessionAttachedMessage)
    selected.rename(tmp_path / "selected-old")
    selected.mkdir()

    response = await daemon._handle(
        ActivateSessionMessage(
            id="activate",
            ts=now_ts(),
            payload=ActivateSessionPayload(session_id="session-1", revision=1),
        )
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.ROOT_GRANT_INVALID
    assert "session-1" not in daemon._child_sessions


@pytest.mark.asyncio
async def test_activate_isolated_child_and_detach(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    daemon = _control_daemon(tmp_path)
    browse_id, directory_ref = await _browse_to_child(daemon, "selected")
    attached = await daemon._handle(
        AttachSessionMessage(
            id="attach",
            ts=now_ts(),
            payload=AttachSessionPayload(
                session_id="session-1",
                browse_id=browse_id,
                directory_ref=directory_ref,
            ),
        )
    )
    assert isinstance(attached, SessionAttachedMessage)
    running = asyncio.Event()

    async def parked_run(child: ShimDaemon) -> None:
        running.set()
        await asyncio.Future()

    monkeypatch.setattr(ShimDaemon, "run", parked_run)
    activated = await daemon._handle(
        ActivateSessionMessage(
            id="activate",
            ts=now_ts(),
            payload=ActivateSessionPayload(session_id="session-1", revision=1),
        )
    )
    await running.wait()

    assert isinstance(activated, SessionActivatedMessage)
    child = daemon._child_sessions["session-1"].daemon
    assert child.config is not daemon.config
    assert child.config.session_id == "session-1"
    assert child.config.allowed_root == selected.resolve()
    assert isinstance(child.audit, SessionAuditWriter)
    assert daemon.config.allowed_root != selected.resolve()

    detached = await daemon._handle(
        DetachSessionMessage(
            id="detach",
            ts=now_ts(),
            payload=DetachSessionPayload(session_id="session-1"),
        )
    )
    assert detached is not None and detached.type == MessageType.ACK
    assert "session-1" not in daemon._child_sessions
    assert "session-1" not in daemon._session_bindings


@pytest.mark.asyncio
async def test_restore_verifies_grant_before_starting_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selected = tmp_path / "selected"
    selected.mkdir()
    daemon = _control_daemon(tmp_path)
    binding = daemon._root_grants.issue("session-1", selected, 4)
    running = asyncio.Event()

    async def parked_run(child: ShimDaemon) -> None:
        running.set()
        await asyncio.Future()

    monkeypatch.setattr(ShimDaemon, "run", parked_run)
    response = await daemon._handle(
        RestoreSessionMessage(
            id="restore",
            ts=now_ts(),
            payload=RestoreSessionPayload(
                session_id="session-1",
                root_grant=binding.root_grant,
            ),
        )
    )
    await running.wait()

    assert isinstance(response, SessionRestoredMessage)
    assert response.payload.revision == 4
    await daemon._shutdown_children()

    bad = await daemon._handle(
        RestoreSessionMessage(
            id="bad-restore",
            ts=now_ts(),
            payload=RestoreSessionPayload(
                session_id="session-1",
                root_grant=binding.root_grant[:-1] + "x",
            ),
        )
    )
    assert isinstance(bad, ErrorMessage)
    assert bad.payload.code == ErrorCode.ROOT_GRANT_INVALID


@pytest.mark.asyncio
async def test_session_bound_audits_share_chain_without_context_leak(tmp_path: Path) -> None:
    writer = AuditWriter(tmp_path / "audit.log", b"a" * 32, machine_id="machine-1")
    first = SessionAuditWriter(writer, "session-1")
    second = SessionAuditWriter(writer, "session-2")

    await asyncio.gather(
        first.write("TEST", request_id="one", details={}),
        second.write("TEST", request_id="two", details={}),
    )

    records = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert {record["session_id"] for record in records} == {"session-1", "session-2"}
    assert AuditWriter.verify(writer.path, b"a" * 32) == []


def test_control_and_legacy_connect_urls_and_hello(tmp_path: Path) -> None:
    control = _control_daemon(tmp_path)
    legacy_config = control.config.model_copy(deep=True)
    legacy_config.session_id = "session-1"
    legacy = ShimDaemon(
        legacy_config,
        token_store=object(),
        audit=SessionAuditWriter(control.audit, "session-1"),  # type: ignore[arg-type]
    )

    assert control._connect_url().endswith("/ws/local-executor")
    assert legacy._connect_url().endswith("/ws/local-executor/session-1")


@pytest.mark.asyncio
async def test_control_hello_and_ack_are_not_session_bound(tmp_path: Path) -> None:
    daemon = _control_daemon(tmp_path)
    hello = await daemon._build_hello()
    assert hello.payload.allowed_root is None
    assert "directory_browse" in hello.payload.capabilities

    class WebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, raw: str) -> None:
            self.sent.append(raw)

        async def recv(self) -> str:
            hello_id = json.loads(self.sent[0])["id"]
            return dump_message(
                HelloAckMessage(
                    id=hello_id,
                    ts=now_ts(),
                    payload=HelloAckPayload(
                        session_id=None,
                        connection_id="connection-1",
                        granted_capabilities=["files", "directory_browse"],
                    ),
                )
            )

    await daemon._handshake(WebSocket())
    assert daemon._connection_id == "connection-1"


@pytest.mark.asyncio
async def test_control_handshake_rejects_missing_directory_capability(
    tmp_path: Path,
) -> None:
    daemon = _control_daemon(tmp_path)

    class WebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.closed: tuple[int, str] | None = None

        async def send(self, raw: str) -> None:
            self.sent.append(raw)

        async def recv(self) -> str:
            hello_id = json.loads(self.sent[0])["id"]
            return dump_message(
                HelloAckMessage(
                    id=hello_id,
                    ts=now_ts(),
                    payload=HelloAckPayload(
                        session_id=None,
                        connection_id="connection-1",
                        granted_capabilities=["files"],
                    ),
                )
            )

        async def close(self, *, code: int, reason: str) -> None:
            self.closed = (code, reason)

    websocket = WebSocket()
    with pytest.raises(RuntimeError, match="machine-control protocol"):
        await daemon._handshake(websocket)

    assert websocket.closed is not None and websocket.closed[0] == 4426
    assert daemon._disable_reconnect is True


class _AckSocket:
    """Answers HELLO with a HELLO_ACK carrying `payload` as given, so a test
    can leave fields out; `close_fails` makes close() raise."""

    def __init__(self, payload: dict[str, Any], *, close_fails: bool = False) -> None:
        self.sent: list[str] = []
        self.closed: tuple[int, str] | None = None
        self._payload = payload
        self._close_fails = close_fails

    async def send(self, raw: str) -> None:
        self.sent.append(raw)

    async def recv(self) -> str:
        hello_id = json.loads(self.sent[0])["id"]
        return json.dumps(
            {"type": "HELLO_ACK", "id": hello_id, "ts": now_ts(), "payload": self._payload}
        )

    async def close(self, *, code: int, reason: str) -> None:
        self.closed = (code, reason)
        if self._close_fails:
            raise OSError("the connection was reset")


# As the platform sends it: a control ACK names its null session.
_CONTROL_ACK: dict[str, Any] = {
    "session_id": None,
    "connection_id": "connection-1",
    "granted_capabilities": ["files", "directory_browse"],
    "protocol_version": "2.0",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("close_fails", [False, True], ids=["closes", "close-fails"])
@pytest.mark.parametrize(
    ("missing", "close_code"),
    [
        ("granted_capabilities", 4400),
        ("protocol_version", 4426),
        ("connection_id", 4426),
        ("session_id", 4426),
    ],
)
async def test_a_control_hello_ack_missing_a_required_field_stops_the_daemon(
    tmp_path: Path, missing: str, close_code: int, close_fails: bool
) -> None:
    """PROTOCOL.md: a control daemon fails terminally when its HELLO_ACK
    lacks a required field. One without granted_capabilities failed to parse
    before the flag was set, so run() reconnected with backoff; one without
    protocol_version was taken for the shim's own; one without session_id
    was taken for a null one; and a close that raised skipped the flag on
    every path."""
    daemon = _control_daemon(tmp_path)
    payload = {key: value for key, value in _CONTROL_ACK.items() if key != missing}
    websocket = _AckSocket(payload, close_fails=close_fails)

    with pytest.raises(RuntimeError):
        await daemon._handshake(websocket)

    assert websocket.closed is not None and websocket.closed[0] == close_code
    assert daemon._disable_reconnect is True


@pytest.mark.asyncio
async def test_a_complete_control_hello_ack_connects(tmp_path: Path) -> None:
    daemon = _control_daemon(tmp_path)
    websocket = _AckSocket(dict(_CONTROL_ACK))

    await daemon._handshake(websocket)

    assert websocket.closed is None
    assert daemon._disable_reconnect is False
    assert daemon._connection_id == "connection-1"


def _idle_child(tmp_path: Path) -> tuple[ShimDaemon, list[str], list[str]]:
    """A child session idle past the reaper's limit, with its sends and the
    paths of the requests it starts recorded."""
    config = ShimConfig(
        session_id="session-1",
        machine_id="machine-1",
        allowed_root=tmp_path / "child-root",
        audit_log_path=tmp_path / "child-audit.log",
    )
    audit = AuditWriter(config.audit_log_path, b"a" * 32, machine_id=config.machine_id)
    child = ShimDaemon(config, token_store=object(), audit=audit)
    child._semaphore = asyncio.Semaphore(4)
    child._last_request_at = 1000.0
    started: list[str] = []
    sent: list[str] = []

    async def handle(msg, *, send):
        started.append(msg.payload.path)
        return make_ack(msg.id)

    child._handle = handle  # type: ignore[method-assign]
    return child, started, sent


class _ChildSocket:
    def __init__(self, sent: list[str]) -> None:
        self._sent = sent

    async def send(self, raw: str) -> None:
        self._sent.append(raw)


def _stat_request(path: str) -> str:
    return dump_message(
        FileStatMessage(id=f"stat-{path}", ts=now_ts(), payload=FileStatPayload(path=path))
    )


@pytest.mark.asyncio
async def test_a_child_with_a_request_still_running_is_not_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reaper looked only at when the last request arrived, so a command
    running longer than the idle limit was stopped part-way, and its reply
    never went out."""
    daemon = _control_daemon(tmp_path)
    child, _started, _sent = _idle_child(tmp_path)
    still_running = asyncio.create_task(asyncio.sleep(60))
    child._dispatch_tasks.add(still_running)
    child._pending_requests = 1
    session = asyncio.create_task(asyncio.sleep(60))
    daemon._child_sessions["session-1"] = _ActiveChild(daemon=child, task=session)
    stop = AsyncMock()
    monkeypatch.setattr(daemon, "_stop_child_session", stop)
    try:
        await daemon._reap_idle_children()
    finally:
        still_running.cancel()
        session.cancel()

    stop.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_request_while_an_idle_detach_is_audited_is_refused_not_cut_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audit write runs in a thread, and a request that reached the child
    meanwhile started, then the stop that followed cut it off. Once the
    reaper decides, the child refuses requests as busy, and the platform
    retries once the session is attached again."""
    daemon = _control_daemon(tmp_path)
    child, started, sent = _idle_child(tmp_path)
    session = asyncio.create_task(asyncio.sleep(60))
    daemon._child_sessions["session-1"] = _ActiveChild(daemon=child, task=session)
    stop = AsyncMock()
    monkeypatch.setattr(daemon, "_stop_child_session", stop)

    async def audit_while_a_request_arrives(op: str, **kwargs: Any) -> None:
        await child._on_frame(_ChildSocket(sent), _stat_request("/tmp/late"))

    monkeypatch.setattr(daemon, "_write_control_audit", audit_while_a_request_arrives)
    try:
        await daemon._reap_idle_children()
        for _ in range(10):
            await asyncio.sleep(0)
    finally:
        session.cancel()

    stop.assert_awaited_once_with("session-1")
    assert started == []
    (refusal,) = (json.loads(raw) for raw in sent)
    assert refusal["type"] == "ERROR" and refusal["payload"]["code"] == "SHIM_OVERLOADED"


@pytest.mark.asyncio
async def test_a_child_whose_detach_cannot_be_audited_keeps_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = _control_daemon(tmp_path)
    child, started, sent = _idle_child(tmp_path)
    session = asyncio.create_task(asyncio.sleep(60))
    daemon._child_sessions["session-1"] = _ActiveChild(daemon=child, task=session)
    stop = AsyncMock()
    monkeypatch.setattr(daemon, "_stop_child_session", stop)
    monkeypatch.setattr(daemon, "_write_control_audit", AsyncMock(side_effect=OSError("full")))
    try:
        await daemon._reap_idle_children()
        await child._on_frame(_ChildSocket(sent), _stat_request("/tmp/next"))
        await asyncio.wait_for(asyncio.gather(*child._dispatch_tasks), timeout=30)
    finally:
        session.cancel()

    stop.assert_not_awaited()
    assert started == ["/tmp/next"]


@pytest.mark.asyncio
async def test_restore_restarts_child_when_root_binding_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    daemon = _control_daemon(tmp_path)
    first = daemon._root_grants.issue("session-1", first_root, 1)
    second = daemon._root_grants.issue("session-1", second_root, 2)
    daemon._remember_binding(first)
    old_task = asyncio.create_task(asyncio.sleep(60))
    daemon._child_sessions["session-1"] = SimpleNamespace(
        daemon=SimpleNamespace(),
        task=old_task,
    )
    stop = AsyncMock(side_effect=lambda session_id: daemon._child_sessions.pop(session_id, None))
    start = AsyncMock()
    monkeypatch.setattr(daemon, "_stop_child_session", stop)
    monkeypatch.setattr(daemon, "_start_child_session", start)

    response = await daemon._handle_restore_session_locked(
        RestoreSessionMessage(
            id="restore",
            ts=now_ts(),
            payload=RestoreSessionPayload(
                session_id="session-1",
                root_grant=second.root_grant,
            ),
        )
    )

    assert isinstance(response, SessionRestoredMessage)
    stop.assert_awaited_once_with("session-1")
    start.assert_awaited_once_with(second)
    assert daemon._session_bindings["session-1"] == second
    old_task.cancel()
    await asyncio.gather(old_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_restore_start_failure_rolls_back_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "selected"
    root.mkdir()
    daemon = _control_daemon(tmp_path)
    binding = daemon._root_grants.issue("session-1", root, 1)
    monkeypatch.setattr(
        daemon,
        "_start_child_session",
        AsyncMock(side_effect=RuntimeError("at capacity")),
    )
    monkeypatch.setattr(daemon, "_stop_child_session", AsyncMock())

    with pytest.raises(RuntimeError, match="at capacity"):
        await daemon._handle_restore_session_locked(
            RestoreSessionMessage(
                id="restore",
                ts=now_ts(),
                payload=RestoreSessionPayload(
                    session_id="session-1",
                    root_grant=binding.root_grant,
                ),
            )
        )

    assert "session-1" not in daemon._session_bindings


@pytest.mark.asyncio
async def test_session_locks_are_released_after_invalid_requests(tmp_path: Path) -> None:
    daemon = _control_daemon(tmp_path)

    for index in range(100):
        response = await daemon._handle(
            RestoreSessionMessage(
                id=f"restore-{index}",
                ts=now_ts(),
                payload=RestoreSessionPayload(
                    session_id=f"session-{index}",
                    root_grant="invalid-grant",
                ),
            )
        )
        assert isinstance(response, ErrorMessage)

    assert daemon._session_locks == {}


@pytest.mark.asyncio
async def test_control_reconnect_invalidates_browse_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    daemon = _control_daemon(tmp_path)
    assert daemon._directory_browser is not None
    listing = daemon._directory_browser.list_directories(None, None)
    old_ref = listing.entries[0]

    async def fail_handshake(_ws: Any) -> None:
        raise RuntimeError("stop after reset")

    monkeypatch.setattr(daemon, "_handshake", fail_handshake)
    with pytest.raises(RuntimeError, match="stop after reset"):
        await daemon._run_session(object())

    with pytest.raises(Exception):
        daemon._directory_browser.resolve(listing.browse_id, old_ref.directory_ref)


@pytest.mark.asyncio
async def test_hello_detects_capabilities_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The X display probe can take seconds, and HELLO ran it on the event
    loop, stalling every other session the machine connection carries."""
    import threading

    from autogpt_local_executor import platform_info

    daemon = _control_daemon(tmp_path)
    ran_on: list[int] = []
    real = platform_info.detect_capabilities

    def detect(**kwargs: Any) -> list[str]:
        ran_on.append(threading.get_ident())
        return real(**kwargs)

    monkeypatch.setattr(platform_info, "detect_capabilities", detect)

    await daemon._build_hello()

    assert ran_on and ran_on[0] != threading.get_ident()
