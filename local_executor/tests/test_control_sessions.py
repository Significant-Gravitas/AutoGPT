from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import websockets

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
    SessionRevokedMessage,
    SessionRevokedPayload,
    dump_message,
    make_ack,
    new_id,
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

    records = [
        json.loads(line) for line in daemon.audit.path.read_text(encoding="utf-8").splitlines()
    ]
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

    records = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]
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
@pytest.mark.parametrize(
    ("payload", "where"),
    [
        (
            {**_CONTROL_ACK, "granted_capabilities": "hunter2-held-value"},
            "HELLO_ACK.payload.granted_capabilities: list_type",
        ),
        ({**_CONTROL_ACK, "hunter2-held-value": 1}, "HELLO_ACK.payload.*: extra_forbidden"),
    ],
    ids=["a-value", "a-field-name"],
)
async def test_a_malformed_control_hello_ack_is_reported_without_what_it_held(
    tmp_path: Path, payload: dict[str, Any], where: str
) -> None:
    """The error a control daemon stops with carried pydantic's, whose text
    quotes the values the HELLO_ACK held, in its traceback, and then the
    name of a field it shouldn't have, which its location holds."""
    daemon = _control_daemon(tmp_path)
    websocket = _AckSocket(payload)

    with pytest.raises(RuntimeError) as raised:
        await daemon._handshake(websocket)

    told = "".join(traceback.format_exception(raised.value))
    assert "hunter2" not in told
    assert where in told
    assert websocket.closed is not None and websocket.closed[0] == 4400


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", [None, ""], ids=["null", "empty"])
async def test_a_complete_control_hello_ack_connects(
    tmp_path: Path, session_id: str | None
) -> None:
    """PROTOCOL.md lets a control ACK name a null or an empty session. The
    model refused an empty one before the daemon's own check, which allows
    it, so the daemon closed with 4400 and stopped reconnecting."""
    daemon = _control_daemon(tmp_path)
    websocket = _AckSocket({**_CONTROL_ACK, "session_id": session_id})

    await daemon._handshake(websocket)

    assert websocket.closed is None
    assert daemon._disable_reconnect is False
    assert daemon._connection_id == "connection-1"


@pytest.mark.asyncio
async def test_a_data_connection_ack_with_an_empty_session_is_refused(tmp_path: Path) -> None:
    """An empty session_id is the control connection's; a chat's session must
    see its own named back."""
    config = ShimConfig(
        session_id="session-1",
        machine_id="machine-1",
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
    )
    audit = AuditWriter(config.audit_log_path, b"a" * 32, machine_id=config.machine_id)
    daemon = ShimDaemon(config, token_store=object(), audit=audit)
    websocket = _AckSocket({**_CONTROL_ACK, "session_id": ""})

    with pytest.raises(RuntimeError):
        await daemon._handshake(websocket)

    assert websocket.closed is not None


@pytest.mark.asyncio
async def test_a_control_reconnect_keeps_the_machine_slots_old_work_holds(tmp_path: Path) -> None:
    """Each control handshake made a new semaphore for the machine's request
    slots. Work a session started can outlive it (a thread can't be
    stopped), so after a reconnect it no longer counted, and the chats'
    new requests could run past the negotiated limit beside it."""
    daemon = _control_daemon(tmp_path)
    await daemon._handshake(_AckSocket({**_CONTROL_ACK, "max_concurrent": 1}))
    old_work = daemon._machine_work_semaphore
    assert old_work is not None
    await old_work.acquire()  # still running in a thread after its session ended

    await daemon._handshake(_AckSocket({**_CONTROL_ACK, "max_concurrent": 1}))
    slots = daemon._machine_work_semaphore
    assert slots is not None

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.2):
            await slots.acquire()
    old_work.release()  # the old work ends
    async with asyncio.timeout(10):
        await slots.acquire()


@pytest.mark.asyncio
async def test_a_lower_limit_after_a_reconnect_counts_the_old_work(tmp_path: Path) -> None:
    daemon = _control_daemon(tmp_path)
    await daemon._handshake(_AckSocket({**_CONTROL_ACK, "max_concurrent": 2}))
    slots = daemon._machine_work_semaphore
    assert slots is not None
    await slots.acquire()
    await slots.acquire()

    await daemon._handshake(_AckSocket({**_CONTROL_ACK, "max_concurrent": 1}))
    slots.release()

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.2):
            await daemon._machine_work_semaphore.acquire()  # type: ignore[union-attr]
    slots.release()
    async with asyncio.timeout(10):
        await daemon._machine_work_semaphore.acquire()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_a_machine_slot_handed_to_a_waiter_that_gives_up_goes_to_the_next() -> None:
    from autogpt_local_executor.daemon import MachineSlots

    slots = MachineSlots(1)
    await slots.acquire()
    first = asyncio.ensure_future(slots.acquire())
    second = asyncio.ensure_future(slots.acquire())
    await asyncio.sleep(0)

    slots.release()  # hands the slot to `first`, which gives up before it runs
    first.cancel()

    with pytest.raises(asyncio.CancelledError):
        await first
    async with asyncio.timeout(10):
        assert await second
    assert slots.locked()


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
async def test_a_child_whose_detach_is_cancelled_while_audited_keeps_serving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The control connection dropping while the idle detach was audited
    cancelled the reaper there, leaving the child running and refusing
    every request as being detached until control reconnected."""
    daemon = _control_daemon(tmp_path)
    child, started, sent = _idle_child(tmp_path)
    session = asyncio.create_task(asyncio.sleep(60))
    daemon._child_sessions["session-1"] = _ActiveChild(daemon=child, task=session)
    stop = AsyncMock()
    monkeypatch.setattr(daemon, "_stop_child_session", stop)
    auditing = asyncio.Event()

    async def audit_until_cancelled(op: str, **kwargs: Any) -> None:
        auditing.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(daemon, "_write_control_audit", audit_until_cancelled)
    reaper = asyncio.create_task(daemon._reap_idle_children())
    try:
        await asyncio.wait_for(auditing.wait(), timeout=30)
        reaper.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reaper
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


# ── A HELLO said before the desktop's display was up ─────────────────────────


class _ConnectedSocket:
    """Answers HELLO with a HELLO_ACK, then carries nothing until the shim
    closes it."""

    def __init__(self, session_id: str | None) -> None:
        self.sent: list[str] = []
        self.closed_with: tuple[int, str] | None = None
        self._closed = asyncio.Event()
        self._session_id = session_id

    async def send(self, raw: str) -> None:
        self.sent.append(raw)

    async def recv(self) -> str:
        hello_id = json.loads(self.sent[0])["id"]
        granted = ["files"] if self._session_id else ["files", "directory_browse"]
        return dump_message(
            HelloAckMessage(
                id=hello_id,
                ts=now_ts(),
                payload=HelloAckPayload(
                    session_id=self._session_id,
                    connection_id="connection-1",
                    granted_capabilities=granted,
                ),
            )
        )

    def __aiter__(self) -> _ConnectedSocket:
        return self

    async def __anext__(self) -> str:
        await self._closed.wait()
        raise StopAsyncIteration

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed_with = (code, reason)
        self._closed.set()

    async def wait_closed(self) -> None:
        await asyncio.wait_for(self._closed.wait(), timeout=10)


class _Desktop:
    """The desktop's display, exported to the systemd manager or not yet,
    counting the times the daemon looks for it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import autogpt_local_executor.daemon as daemon_module
        from autogpt_local_executor import platform_info
        from autogpt_local_executor.computer_use import driver as driver_module

        self.up = False
        self.looks = 0
        monkeypatch.setattr(daemon_module, "DISPLAY_WATCH_SECONDS", 0.01, raising=False)
        monkeypatch.setattr(platform_info, "detect_capabilities", lambda **_kwargs: ["files"])
        monkeypatch.setattr(
            platform_info, "computer_use_waits_for_a_display", lambda: not self.up, raising=False
        )
        monkeypatch.setattr(platform_info, "can_do_computer_use", self._can_do_computer_use)
        # The driver is installed, as the bridge's look finds.
        monkeypatch.setattr(
            driver_module, "find_driver_binary", lambda _explicit=None: Path("cua-driver")
        )

    def _can_do_computer_use(
        self, _driver_path: str | None = None, *, driver_found: bool | None = None
    ) -> bool:
        self.looks += 1
        return self.up

    async def looked(self, times: int) -> None:
        loop = asyncio.get_running_loop()
        until, deadline = self.looks + times, loop.time() + 10
        while self.looks < until:
            assert loop.time() < deadline, "the session never looked for the display"
            await asyncio.sleep(0.01)


def _computer_use_session(tmp_path: Path, mode: str) -> ShimDaemon:
    config = ShimConfig(
        session_id="session-1" if mode == "chat" else None,
        machine_id="machine-1",
        allowed_root=tmp_path / "root",
        audit_log_path=tmp_path / "audit.log",
        enable_computer_use=True,
    )
    audit = AuditWriter(config.audit_log_path, b"a" * 32, machine_id=config.machine_id)
    daemon = ShimDaemon(config, token_store=object(), audit=audit)
    daemon._running = True
    return daemon


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["chat", "machine"])
async def test_a_session_says_hello_again_once_the_desktops_display_is_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """The systemd service starts at login, before the desktop exports its
    display to the user manager, so a session's HELLO left computer use out,
    and nothing offered it afterwards: HELLO is the only place it is offered,
    and the connection could last for days. Once the display is up, the
    session closes its connection, and the reconnect's HELLO offers it."""
    desktop = _Desktop(monkeypatch)
    daemon = _computer_use_session(tmp_path, mode)
    ws = _ConnectedSocket("session-1" if mode == "chat" else None)
    session = asyncio.create_task(daemon._run_session(ws))
    try:
        await desktop.looked(2)
        assert ws.closed_with is None
        desktop.up = True
        await ws.wait_closed()
        await asyncio.wait_for(session, timeout=10)
    finally:
        session.cancel()

    assert ws.closed_with is not None and ws.closed_with[0] == 1000


@pytest.mark.asyncio
@pytest.mark.parametrize("busy", ["request", "chat"])
async def test_a_session_says_hello_again_once_nothing_is_under_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, busy: str
) -> None:
    """Closing a connection fails what it carries, and the machine's next
    session stops every chat's, so the new HELLO waits for the session's own
    requests, and the machine's for its chats' too."""
    desktop = _Desktop(monkeypatch)
    daemon = _computer_use_session(tmp_path, "chat" if busy == "request" else "machine")
    ws = _ConnectedSocket("session-1" if busy == "request" else None)
    session = asyncio.create_task(daemon._run_session(ws))
    chat = asyncio.create_task(asyncio.sleep(60))
    try:
        await desktop.looked(1)
        if busy == "request":
            working = daemon
        else:
            working, _started, _sent = _idle_child(tmp_path)
            daemon._child_sessions["session-1"] = _ActiveChild(daemon=working, task=chat)
        working._pending_requests = 1
        desktop.up = True
        await desktop.looked(3)
        assert ws.closed_with is None

        working._pending_requests = 0
        await ws.wait_closed()
        await asyncio.wait_for(session, timeout=10)
    finally:
        session.cancel()
        chat.cancel()

    assert ws.closed_with is not None and ws.closed_with[0] == 1000


class _ClosingSocket(_ConnectedSocket):
    """A _ConnectedSocket whose close lets `meanwhile` run first: frames the
    platform sent before it saw the close still arrive, on this connection
    and on the chats' own."""

    def __init__(self, session_id: str | None, meanwhile: Any) -> None:
        super().__init__(session_id)
        self._meanwhile = meanwhile

    async def close(self, code: int = 1000, reason: str = "") -> None:
        await self._meanwhile(self)
        await super().close(code, reason)


async def _until(condition: Any) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10
    while not condition():
        assert loop.time() < deadline, "the condition never came true"
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_a_request_while_the_machine_reconnects_for_the_display_is_refused_not_cut_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the machine's connection had decided to say HELLO again, its
    close and the reconnect's backoff took seconds, and its chats' sessions
    went on taking requests meanwhile: one started then was stopped with
    every chat's session when the next connection began. From the decision
    on, the machine's session and its chats' refuse requests as busy, and
    the next connection serves them again."""
    desktop = _Desktop(monkeypatch)
    daemon = _computer_use_session(tmp_path, "machine")
    child, started, sent = _idle_child(tmp_path)
    chat = asyncio.create_task(asyncio.sleep(60))
    served: list[str] = []

    async def handle(msg, *, send):
        served.append(msg.payload.path)
        return make_ack(msg.id)

    daemon._handle = handle  # type: ignore[method-assign]

    async def requests_arrive(ws: _ConnectedSocket) -> None:
        await child._on_frame(_ChildSocket(sent), _stat_request("/tmp/chat"))
        await daemon._on_frame(ws, _stat_request("/tmp/machine"))
        for _ in range(10):
            await asyncio.sleep(0)

    ws = _ClosingSocket(None, requests_arrive)
    session = asyncio.create_task(daemon._run_session(ws))
    try:
        await desktop.looked(1)
        daemon._child_sessions["session-1"] = _ActiveChild(daemon=child, task=chat)
        desktop.up = True
        await ws.wait_closed()
        await asyncio.wait_for(session, timeout=10)
    finally:
        session.cancel()

    assert started == [] and served == []
    (chat_refusal,) = (json.loads(raw) for raw in sent)
    (machine_refusal,) = (
        frame for frame in map(json.loads, ws.sent) if frame["id"] == "stat-/tmp/machine"
    )
    for refusal in (chat_refusal, machine_refusal):
        assert refusal["type"] == "ERROR"
        assert refusal["payload"]["code"] == "SHIM_OVERLOADED"
        assert refusal["payload"]["details"] == {"reason": "reconnecting"}

    # A chat that reconnects on its own meanwhile (its connection dropped
    # too) still refuses: the machine's next session stops it all the same.
    chat_again = _ConnectedSocket("session-1")
    chat_session = asyncio.create_task(child._run_session(chat_again))
    try:
        await _until(lambda: child._status_task is not None)
        await child._on_frame(chat_again, _stat_request("/tmp/chat-again"))
        for _ in range(10):
            await asyncio.sleep(0)
    finally:
        await chat_again.close()
        await asyncio.wait_for(chat_session, timeout=10)
    assert started == []
    (chat_again_refusal,) = (
        frame for frame in map(json.loads, chat_again.sent) if frame["id"] == "stat-/tmp/chat-again"
    )
    assert chat_again_refusal["payload"]["details"] == {"reason": "reconnecting"}

    # The next connection stops the chat it would have cut off, and serves.
    again = _ConnectedSocket(None)
    session = asyncio.create_task(daemon._run_session(again))
    try:
        await _until(lambda: daemon._status_task is not None)
        assert chat.cancelled() and daemon._child_sessions == {}
        await daemon._on_frame(again, _stat_request("/tmp/after"))
        await _until(lambda: served == ["/tmp/after"])
    finally:
        await again.close()
        await asyncio.wait_for(session, timeout=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("why", ["computer-use-off", "display-there"])
async def test_a_hello_without_computer_use_for_another_reason_isnt_said_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, why: str
) -> None:
    """A HELLO that left computer use out for another reason (the person
    didn't turn it on; a driver missing, with the display there) would leave
    it out again: no new HELLO, every ten seconds, for it."""
    from autogpt_local_executor import platform_info

    monkeypatch.setattr(platform_info, "detect_capabilities", lambda **_kwargs: ["files"])
    monkeypatch.setattr(
        platform_info, "computer_use_waits_for_a_display", lambda: why != "display-there"
    )
    daemon = _computer_use_session(tmp_path, "chat")
    daemon.config.enable_computer_use = why != "computer-use-off"

    await daemon._build_hello()

    assert daemon._hello_waits_for_display is False


class _NoManager:
    """platform_info.session_display, with the display already followed."""

    def follow(self, timeout: float | None = None) -> bool:
        return False


@pytest.mark.asyncio
async def test_the_display_watch_looks_for_the_driver_through_the_bridges_one_look(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While a session waits for the desktop's display, its watch asked
    every ten seconds whether computer use could start, which looked for the
    driver in a worker thread the watch waited for to the end. A stalled
    mount on sys.path or PATH held that look, and the reconnect, cancelling
    the watch, left its thread held: each connection one more, until the
    threads every session's file and audit work shares were all held. The
    watch now looks through the bridge's look, one at a time and bounded,
    and tells the display check what it found."""
    import threading

    import autogpt_local_executor.daemon as daemon_module
    from autogpt_local_executor import handlers, platform_info
    from autogpt_local_executor.computer_use import driver as driver_module

    released = threading.Event()
    counting = threading.Lock()
    lookups = 0

    def stalled_lookup(_explicit: str | None = None) -> Path:
        nonlocal lookups
        with counting:
            lookups += 1
        released.wait(timeout=60)  # a mount on PATH, stalled until it clears
        return tmp_path / "cua-driver"

    up = False
    # What each display check was told of the driver: None, it looked itself.
    told: list[bool | None] = []
    real_can_do_computer_use = platform_info.can_do_computer_use

    def can_do_computer_use(
        cua_driver_path: str | None = None, *, driver_found: bool | None = None
    ) -> bool:
        told.append(driver_found)
        return real_can_do_computer_use(cua_driver_path, driver_found=driver_found)

    monkeypatch.setattr(driver_module, "find_driver_binary", stalled_lookup)
    monkeypatch.setattr(handlers, "DRIVER_LOOKUP_SECONDS", 0.05)
    monkeypatch.setattr(daemon_module, "DISPLAY_WATCH_SECONDS", 0.01)
    monkeypatch.setattr(platform_info, "detect_capabilities", lambda **_kwargs: ["files"])
    monkeypatch.setattr(platform_info, "computer_use_waits_for_a_display", lambda: not up)
    monkeypatch.setattr(platform_info, "can_do_computer_use", can_do_computer_use)
    monkeypatch.setattr(platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setattr(platform_info, "session_display", _NoManager())
    monkeypatch.setattr(
        platform_info,
        "linux_display_problem",
        lambda _environ=None: None if up else "no display yet",
    )
    daemon = _computer_use_session(tmp_path, "machine")
    bridge = daemon._computer_handler.driver
    real_find_binary = bridge.find_binary
    watch_looks = 0

    async def find_binary() -> Path | None:
        nonlocal watch_looks
        if asyncio.current_task() is daemon._display_watch_task:
            watch_looks += 1
        return await real_find_binary()

    monkeypatch.setattr(bridge, "find_binary", find_binary)

    def watch_asked() -> int:
        return watch_looks + len(told)

    try:
        for _ in range(3):
            ws = _ConnectedSocket(None)
            asked_before = watch_asked()
            session = asyncio.create_task(daemon._run_session(ws))
            try:
                await _until(lambda: watch_asked() > asked_before)
            finally:
                await ws.close()
                await asyncio.wait_for(session, timeout=10)
        # HELLO's look, which the watches waited for in turn.
        assert lookups == 1

        released.set()
        ws = _ConnectedSocket(None)
        told_before = len(told)
        session = asyncio.create_task(daemon._run_session(ws))
        try:
            await _until(lambda: len(told) > told_before)
            assert ws.closed_with is None
            up = True
            await ws.wait_closed()
            await asyncio.wait_for(session, timeout=10)
        finally:
            session.cancel()
    finally:
        released.set()

    assert ws.closed_with == (1000, "Computer use is available now")
    assert told and all(found is True for found in told)


# ── A revoked session's work stops before its record is written ──────────────


class _PlatformClosesSocket(_ConnectedSocket):
    """A _ConnectedSocket the platform closes, with a code of its choosing."""

    def __init__(self, session_id: str | None) -> None:
        super().__init__(session_id)
        self._platform_code: int | None = None
        self._platform_closed = asyncio.Event()

    def close_from_platform(self, code: int) -> None:
        self._platform_code = code
        self._platform_closed.set()

    async def __anext__(self) -> str:
        from websockets.frames import Close

        await self._platform_closed.wait()
        assert self._platform_code is not None
        raise websockets.exceptions.ConnectionClosed(
            rcvd=Close(code=self._platform_code, reason=""), sent=None
        )


def _running_command(session: ShimDaemon) -> tuple[asyncio.Event, asyncio.Event]:
    """Have `session` run every request as a command that goes on until it
    is stopped. Returns events set once one runs and once it has stopped."""
    running = asyncio.Event()
    stopped = asyncio.Event()

    async def command(msg, *, send):
        running.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    session._handle = command  # type: ignore[method-assign]
    return running, stopped


async def _chat_running_a_command(
    daemon: ShimDaemon, tmp_path: Path, *, session_id: str = "session-1"
) -> tuple[asyncio.Task[None], asyncio.Event]:
    """One of `daemon`'s chats, writing to its audit log as those
    _start_child_session starts do, with a command running. Returns the
    chat's session and an event set once the command has stopped."""
    config = ShimConfig(
        session_id=session_id,
        machine_id="machine-1",
        allowed_root=tmp_path / f"{session_id}-root",
        audit_log_path=daemon.config.audit_log_path,
    )
    chat = ShimDaemon(
        config,
        token_store=object(),
        audit=SessionAuditWriter(daemon.audit, session_id),
        manage_audit_lifecycle=False,
    )
    running, stopped = _running_command(chat)
    chat_ws = _ConnectedSocket(session_id)
    session = asyncio.create_task(chat._run_session(chat_ws))
    daemon._child_sessions[session_id] = _ActiveChild(daemon=chat, task=session)
    await _until(lambda: chat._status_task is not None)
    await chat._on_frame(chat_ws, _stat_request("/tmp/command"))
    await asyncio.wait_for(running.wait(), timeout=10)
    return session, stopped


def _hold_the_audit_writer(daemon: ShimDaemon, monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    """Hold every write to `daemon`'s audit log, its chats' too, until the
    event returned is set, as `audit verify` holding the log's lock while it
    reads a large log does."""
    free = asyncio.Event()
    write = daemon.audit.write

    async def held(op: str, **kwargs: Any) -> None:
        await free.wait()
        await write(op, **kwargs)

    monkeypatch.setattr(daemon.audit, "write", held)
    return free


def _revocations(daemon: ShimDaemon) -> list[dict[str, Any]]:
    records = (
        json.loads(line)
        for line in daemon.config.audit_log_path.read_text(encoding="utf-8").splitlines()
    )
    return [record["details"] for record in records if record["op"] == "SESSION_REVOKED"]


@pytest.mark.asyncio
async def test_a_revoked_machine_stops_its_chats_while_the_revocation_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SESSION_REVOKED stopped the machine's own requests at once, but its
    chats' sessions only once its audit record was written, and an audit
    writer held up (its lock taken by `audit verify` on a large log, say)
    left their commands running on a machine whose session had been revoked.
    They stop at once, and the record is still written."""
    daemon = _control_daemon(tmp_path)
    chat, stopped = await _chat_running_a_command(daemon, tmp_path)
    writer_free = _hold_the_audit_writer(daemon, monkeypatch)
    revoked = dump_message(
        SessionRevokedMessage(
            id=new_id(),
            ts=now_ts(),
            payload=SessionRevokedPayload(reason="revoked_by_user"),
        )
    )
    handling = asyncio.ensure_future(daemon._on_frame(_ConnectedSocket(None), revoked))
    try:
        await asyncio.wait_for(stopped.wait(), timeout=10)
        assert not handling.done()
    finally:
        writer_free.set()
    await asyncio.wait_for(handling, timeout=10)

    assert chat.done() and daemon._child_sessions == {}
    assert [details["source"] for details in _revocations(daemon)] == ["frame"]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", [4427, 4428])
@pytest.mark.parametrize("working", ["chat", "session"])
async def test_a_session_closed_as_revoked_stops_its_work_while_that_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, code: int, working: str
) -> None:
    """A revocation the platform sends as the close code (taken over by
    another computer, or revoked) was recorded before anything stopped: the
    session's own requests were cancelled after the record, and the
    machine's chats were stopped after that, so a held-up audit writer left
    their commands running. They stop at once; the record is still written."""
    if working == "chat":
        daemon = _control_daemon(tmp_path)
        ws = _PlatformClosesSocket(None)
    else:
        config = ShimConfig(
            session_id="session-1",
            machine_id="machine-1",
            allowed_root=tmp_path / "root",
            audit_log_path=tmp_path / "audit.log",
        )
        audit = AuditWriter(config.audit_log_path, b"a" * 32, machine_id=config.machine_id)
        daemon = ShimDaemon(config, token_store=object(), audit=audit)
        ws = _PlatformClosesSocket("session-1")
    session = asyncio.create_task(daemon._run_session(ws))
    try:
        await _until(lambda: daemon._status_task is not None)
        if working == "chat":
            chat, stopped = await _chat_running_a_command(daemon, tmp_path)
        else:
            running, stopped = _running_command(daemon)
            await daemon._on_frame(ws, _stat_request("/tmp/command"))
            await asyncio.wait_for(running.wait(), timeout=10)
        writer_free = _hold_the_audit_writer(daemon, monkeypatch)
        ws.close_from_platform(code)
        try:
            await asyncio.wait_for(stopped.wait(), timeout=10)
            assert not session.done()
        finally:
            writer_free.set()
        with pytest.raises(websockets.exceptions.ConnectionClosed):
            await asyncio.wait_for(session, timeout=10)
    finally:
        session.cancel()

    if working == "chat":
        assert chat.done() and daemon._child_sessions == {}
    assert [(details["source"], details["close_code"]) for details in _revocations(daemon)] == [
        ("ws_close_code", code)
    ]


# ── Every chat stops before any of them is waited for ────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("how", ["revoked", "closed_as_revoked", "shut_down"])
async def test_every_chat_stops_before_one_held_up_stopping_is_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    """The machine stopped its chats one after another, waiting for each in
    full, and a chat's stop waits for its WS_DISCONNECTED record: with the
    audit writer held up (its lock taken by `audit verify` on a large log,
    say), the first chat's stop waited, and the chats after it went on
    running their commands, after the machine's session was revoked too.
    Every chat's work stops, and its connection closes, before any of them
    is waited for: on a revocation, on a close as revoked, and when the
    machine stops them all (`stop`, the end of `run`, its next connection)."""
    daemon = _control_daemon(tmp_path)
    ws = _PlatformClosesSocket(None)
    session = asyncio.create_task(daemon._run_session(ws))
    try:
        await _until(lambda: daemon._status_task is not None)
        chats = [
            await _chat_running_a_command(daemon, tmp_path, session_id=session_id)
            for session_id in ("session-1", "session-2")
        ]
        chat_sockets = [active.daemon._ws for active in daemon._child_sessions.values()]
        writer_free = _hold_the_audit_writer(daemon, monkeypatch)
        stopping: asyncio.Future[None]
        if how == "revoked":
            revoked = dump_message(
                SessionRevokedMessage(
                    id=new_id(),
                    ts=now_ts(),
                    payload=SessionRevokedPayload(reason="revoked_by_user"),
                )
            )
            stopping = asyncio.ensure_future(daemon._on_frame(ws, revoked))
        elif how == "closed_as_revoked":
            ws.close_from_platform(4428)
            stopping = session
        else:
            stopping = asyncio.ensure_future(daemon._shutdown_children())
        try:
            for _chat, stopped in chats:
                await asyncio.wait_for(stopped.wait(), timeout=10)
            await _until(lambda: all(chat_ws.closed_with is not None for chat_ws in chat_sockets))
            assert not stopping.done()
        finally:
            writer_free.set()
        if how == "closed_as_revoked":
            with pytest.raises(websockets.exceptions.ConnectionClosed):
                await asyncio.wait_for(stopping, timeout=10)
        else:
            await asyncio.wait_for(stopping, timeout=10)
    finally:
        session.cancel()
        await asyncio.gather(session, return_exceptions=True)

    assert all(chat.done() for chat, _stopped in chats)
    assert daemon._child_sessions == {}


# ── What commands leave running ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_chat_leaves_what_its_commands_left_running_to_the_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chat ends long before a server one of its commands started, so the
    reads that keep such a program's output from filling are the machine's
    command handler's, which a chat's shares, as it shares the command
    limits: the chat's close leaves them, and the machine's ends them."""
    from autogpt_local_executor.session_grants import RootBinding

    daemon = _control_daemon(tmp_path)

    async def parked_run(child: ShimDaemon) -> None:
        await asyncio.Future()

    monkeypatch.setattr(ShimDaemon, "run", parked_run)
    binding = RootBinding(
        session_id="session-1",
        allowed_root=tmp_path,
        fingerprint="0" * 64,
        revision=1,
        root_grant="grant",
    )
    try:
        await daemon._start_child_session(binding)
        chat = daemon._child_sessions["session-1"].daemon._command_handler
        assert chat._command_rate is daemon._command_handler._command_rate
        assert chat._concurrency is daemon._command_handler._concurrency
        # A command whose stdout a program it left running still holds.
        chat._keep_draining(SimpleNamespace(stdout=asyncio.StreamReader(), stderr=None))
        await chat.close()
        [reading] = daemon._command_handler._output_drains
        assert not reading.done()
    finally:
        await daemon._shutdown_children()
        await daemon._command_handler.close()

    assert reading.cancelled()


@pytest.mark.asyncio
async def test_the_shim_stops_reading_what_commands_left_running_as_it_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a command left running is read until that program closes its
    output, which a server never does, so the shim's stop ends the reads."""
    daemon = _control_daemon(tmp_path)
    # A command whose stdout a program it left running still holds.
    daemon._command_handler._keep_draining(
        SimpleNamespace(stdout=asyncio.StreamReader(), stderr=None)
    )
    [reading] = daemon._command_handler._output_drains

    async def one_session() -> None:
        daemon._running = False

    monkeypatch.setattr(daemon, "_preflight_or_raise", AsyncMock())
    monkeypatch.setattr(daemon, "_audit_shim_start", AsyncMock())
    monkeypatch.setattr(daemon, "_audit_shim_stop", AsyncMock())
    monkeypatch.setattr(daemon, "_session", one_session)

    await asyncio.wait_for(daemon.run(), timeout=30)

    assert reading.cancelled()
    assert not daemon._command_handler._output_drains
