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
from autogpt_local_executor.daemon import ShimDaemon
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
