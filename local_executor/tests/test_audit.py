"""Tests for the audit log writer + verifier (docs/AUDIT_LOG.md)."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import hashlib
import hmac
import json
import os
import stat
import sys
import threading
import types
from pathlib import Path
from unittest.mock import patch

import pytest

import autogpt_local_executor.audit as audit_module
from autogpt_local_executor import cli
from autogpt_local_executor.audit import (
    AuditWriter,
    canonical_bytes,
    get_or_create_audit_key,
    list_rotated_files,
)

KEY = b"\x00" * 32  # deterministic test key


def _writer(tmp_path: Path) -> AuditWriter:
    return AuditWriter(
        path=tmp_path / "audit.log",
        audit_key=KEY,
        shim_version="test-0.0",
        machine_id="test-machine",
        session_id="test-session",
    )


# ── Canonicalization spot-checks ─────────────────────────────────────────────


def test_canonical_sorts_keys_and_strips_whitespace() -> None:
    out = canonical_bytes({"b": 1, "a": "x", "c": [1, 2, {"y": True, "x": False}]})
    assert out == b'{"a":"x","b":1,"c":[1,2,{"x":false,"y":true}]}'


def test_canonical_handles_unicode() -> None:
    out = canonical_bytes({"k": "café"})
    # JSON spec keeps non-ASCII as-is when ensure_ascii=False.
    assert out == '{"k":"café"}'.encode()


def test_canonical_rejects_nan_and_inf() -> None:
    with pytest.raises(ValueError):
        canonical_bytes({"k": float("nan")})
    with pytest.raises(ValueError):
        canonical_bytes({"k": float("inf")})


# ── Round-trip + verify ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_round_trip_100_records_verifies_clean(tmp_path: Path) -> None:
    w = _writer(tmp_path)
    for i in range(100):
        await w.write(
            "EXECUTE_COMMAND",
            request_id=f"req-{i}",
            details={
                "command": f"echo {i}",
                "argv": None,
                "shell": "auto",
                "cwd": "/tmp",
                "env_keys": [],
                "timeout_seconds": 30,
            },
            result={"ok": True, "exit_code": 0, "duration_ms": 1, "error_code": None},
        )
    violations = AuditWriter.verify(w.path, KEY)
    assert violations == []
    # Last record has seq=100.
    lines = w.path.read_text().splitlines()
    assert len(lines) == 100
    assert json.loads(lines[-1])["seq"] == 100


@pytest.mark.asyncio
async def test_verify_clean_for_empty_file_returns_no_violations(tmp_path: Path) -> None:
    path = tmp_path / "audit.log"
    path.touch()
    assert AuditWriter.verify(path, KEY) == []


# ── Tamper detection ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_byte_flip_in_middle_detected_at_right_seq(tmp_path: Path) -> None:
    w = _writer(tmp_path)
    for i in range(20):
        await w.write("EXECUTE_COMMAND", request_id=str(i), details={"i": i}, result={})
    # Corrupt one byte inside line 10 (seq 10). Pick the `i` field value so
    # the hmac mismatches without breaking JSON parsing.
    raw = w.path.read_bytes().splitlines(keepends=True)
    target = raw[9]
    rec = json.loads(target)
    rec["details"]["i"] = 999  # tamper
    raw[9] = (json.dumps(rec) + "\n").encode("utf-8")
    w.path.write_bytes(b"".join(raw))

    violations = AuditWriter.verify(w.path, KEY)
    # First violation is the hmac mismatch on seq=10. The chain break then
    # cascades to subsequent records' prev_hmac check, which is fine.
    assert violations
    first = violations[0]
    assert first.kind == "hmac_mismatch"
    assert first.seq == 10


@pytest.mark.asyncio
async def test_truncation_mid_chain_detected(tmp_path: Path) -> None:
    w = _writer(tmp_path)
    for i in range(10):
        await w.write("EXECUTE_COMMAND", request_id=str(i), details={"i": i}, result={})
    # Drop the trailing newline of the file: simulates a torn write.
    data = w.path.read_bytes()
    w.path.write_bytes(data[:-1])
    violations = AuditWriter.verify(w.path, KEY)
    assert any(v.kind == "truncated" for v in violations)


@pytest.mark.asyncio
async def test_sequence_gap_detected_when_record_deleted(tmp_path: Path) -> None:
    w = _writer(tmp_path)
    for i in range(5):
        await w.write("EXECUTE_COMMAND", request_id=str(i), details={"i": i}, result={})
    lines = w.path.read_text().splitlines(keepends=True)
    # Delete the third line — verifier should report a sequence gap on the
    # fourth line (which now sits where seq=3 should be).
    del lines[2]
    w.path.write_text("".join(lines))
    violations = AuditWriter.verify(w.path, KEY)
    kinds = {v.kind for v in violations}
    assert "sequence_gap" in kinds


# ── Rotation ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rotation_when_size_threshold_exceeded(tmp_path: Path, monkeypatch) -> None:
    """Use a small threshold so we don't have to write 64 MiB in a test."""
    monkeypatch.setattr("autogpt_local_executor.audit.ROTATE_MAX_BYTES", 4096)
    w = _writer(tmp_path)
    # Write enough records to cross the threshold.
    for i in range(50):
        await w.write(
            "EXECUTE_COMMAND",
            request_id=str(i),
            details={"command": "x" * 200, "i": i},
            result={},
        )
    rotated = list_rotated_files(w.path)
    assert rotated, "Expected at least one rotated file"
    # The rotated file should chain-verify cleanly on its own.
    for path in rotated:
        violations = AuditWriter.verify(path, KEY)
        assert violations == [], f"{path.name} should verify"
    # New file starts at seq=1.
    if w.path.read_text().strip():
        first_line = w.path.read_text().splitlines()[0]
        assert json.loads(first_line)["seq"] == 1


@pytest.mark.asyncio
async def test_force_rotate_returns_path_and_resets_seq(tmp_path: Path) -> None:
    w = _writer(tmp_path)
    await w.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    rotated = w.force_rotate()
    assert rotated is not None and rotated.exists()
    # Next write goes to a fresh file at seq=1.
    await w.write("EXECUTE_COMMAND", request_id="b", details={}, result={})
    rec = json.loads(w.path.read_text().splitlines()[-1])
    assert rec["seq"] == 1
    assert rec["prev_hmac"] is None


@pytest.mark.asyncio
async def test_cli_rotation_while_the_daemon_runs_keeps_both_chains_valid(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """`autogpt-shim audit rotate` has its own writer; the daemon's must notice."""
    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    daemon = _writer(tmp_path)
    await daemon.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    await daemon.write("EXECUTE_COMMAND", request_id="b", details={}, result={})

    assert cli._audit_rotate(daemon.path) == 0
    await daemon.write("EXECUTE_COMMAND", request_id="c", details={}, result={})

    [rotated] = list_rotated_files(daemon.path)
    assert AuditWriter.verify(rotated, KEY) == []
    assert AuditWriter.verify(daemon.path, KEY) == []
    first = json.loads(daemon.path.read_text().splitlines()[0])
    assert (first["request_id"], first["seq"], first["prev_hmac"]) == ("c", 1, None)
    AuditWriter(path=daemon.path, audit_key=KEY)  # a restarted daemon accepts the log


@pytest.mark.asyncio
async def test_a_rotation_waits_for_an_append_in_progress(tmp_path: Path, monkeypatch) -> None:
    daemon = _writer(tmp_path)
    await daemon.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    appending = threading.Event()
    release = threading.Event()
    append = daemon._append

    def held_append(line: str) -> None:
        appending.set()
        release.wait(timeout=30)
        append(line)

    monkeypatch.setattr(daemon, "_append", held_append)
    # Built before the append, so the only thing left to wait on the lock is
    # the rotation itself, not the second writer's startup.
    other = AuditWriter(path=daemon.path, audit_key=KEY)
    write = asyncio.create_task(
        daemon.write("EXECUTE_COMMAND", request_id="b", details={}, result={})
    )
    assert await asyncio.to_thread(appending.wait, 30)
    rotation = asyncio.create_task(asyncio.to_thread(other.force_rotate))
    await asyncio.sleep(0.2)
    rotated_during_the_append = rotation.done()
    release.set()
    await write
    rotated = await rotation

    assert not rotated_during_the_append
    assert rotated is not None
    assert [json.loads(line)["request_id"] for line in rotated.read_text().splitlines()] == [
        "a",
        "b",
    ]
    assert AuditWriter.verify(rotated, KEY) == []


@pytest.mark.asyncio
async def test_force_rotate_collision_safe(tmp_path: Path) -> None:
    """If two rotations happen in the same second, the second gets a -N suffix."""
    w = _writer(tmp_path)
    await w.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    first = w.force_rotate()
    assert first is not None
    await w.write("EXECUTE_COMMAND", request_id="b", details={}, result={})
    with patch("autogpt_local_executor.audit._dt.datetime") as mock_dt:
        # Pin both rotations to the same timestamp so the collision branch fires.
        mock_dt.now.return_value.strftime.return_value = first.name.split(".")[-1]
        second = w.force_rotate()
    assert second is not None and second.exists()
    assert second.name != first.name


# ── Redaction ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_env_values_never_leak_into_audit_log(tmp_path: Path) -> None:
    """The writer hands the caller's `details` straight through; the
    contract is that callers only pass env_keys, never env values. This test
    is the seam check: if any handler regresses and starts passing values,
    the secret would show up here."""
    from autogpt_local_executor.config import ShimConfig
    from autogpt_local_executor.handlers import CommandHandler
    from autogpt_local_executor.protocol import (
        ExecuteCommandMessage,
        ExecuteCommandPayload,
        new_id,
        now_ts,
    )

    config = ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "audit.log",
        platform_url="http://localhost:9999",
        machine_id="m",
        enable_shell=True,
    )
    audit = AuditWriter(path=config.audit_log_path, audit_key=KEY)
    h = CommandHandler(config, audit=audit)

    class _FakeProc:
        def __init__(self) -> None:
            self.returncode = 0
            self.pid = 123

        async def communicate(self):
            return b"", b""

    async def fake_exec(*args, **kwargs):
        return _FakeProc()

    msg = ExecuteCommandMessage(
        id=new_id(),
        ts=now_ts(),
        payload=ExecuteCommandPayload(
            argv=["echo", "x"],
            cwd=str(tmp_path),
            env={"SECRET": "shouldnotappear"},
        ),
    )
    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await h.handle(msg)

    contents = config.audit_log_path.read_text()
    assert "shouldnotappear" not in contents
    # But the env key is logged.
    assert "SECRET" in contents


@pytest.mark.asyncio
async def test_input_action_text_redacted_to_length(tmp_path: Path) -> None:
    audit = AuditWriter(path=tmp_path / "audit.log", audit_key=KEY)
    # Mimic the ComputerUseHandler details shape (text → text_length only).
    await audit.write(
        "INPUT_ACTION",
        request_id="r",
        details={
            "action": "type",
            "coordinate": None,
            "key": None,
            "direction": None,
            "clicks": None,
            "text_length": len("my-password"),
        },
        result={"ok": True, "exit_code": None, "duration_ms": 0, "error_code": None},
    )
    contents = (tmp_path / "audit.log").read_text()
    assert "my-password" not in contents
    assert '"text_length":11' in contents.replace(" ", "")


# ── verify-all across multiple rotated files ─────────────────────────────────


@pytest.mark.asyncio
async def test_verify_all_reports_only_tampered_file(tmp_path: Path) -> None:
    w = _writer(tmp_path)
    await w.write("EXECUTE_COMMAND", request_id="a", details={"i": 1}, result={})
    clean_rotated = w.force_rotate()
    assert clean_rotated is not None

    await w.write("EXECUTE_COMMAND", request_id="b", details={"i": 2}, result={})
    tampered_rotated = w.force_rotate()
    assert tampered_rotated is not None
    # Tamper: flip a byte in the rotated file.
    raw = tampered_rotated.read_text()
    rec = json.loads(raw.strip().splitlines()[0])
    rec["details"]["i"] = 999
    tampered_rotated.write_text(json.dumps(rec) + "\n")

    # Verify each file individually — only the tampered one should report.
    assert AuditWriter.verify(clean_rotated, KEY) == []
    violations = AuditWriter.verify(tampered_rotated, KEY)
    assert violations and violations[0].kind == "hmac_mismatch"


# ── Keyring round-trip (mocked) ──────────────────────────────────────────────


def test_get_or_create_audit_key_round_trips_via_keyring() -> None:
    store: dict[tuple[str, str], str] = {}

    def fake_get(svc: str, user: str) -> str | None:
        return store.get((svc, user))

    def fake_set(svc: str, user: str, value: str) -> None:
        store[(svc, user)] = value

    import keyring

    with (
        patch.object(keyring, "get_password", side_effect=fake_get),
        patch.object(keyring, "set_password", side_effect=fake_set),
    ):
        first = get_or_create_audit_key()
        second = get_or_create_audit_key()
    assert first == second
    assert len(first) == 32


@pytest.mark.parametrize("stored", ["not-hex!", "", "ab", "ab" * 31])
def test_get_or_create_audit_key_preserves_corrupt_stored_value(stored: str) -> None:
    store: dict[tuple[str, str], str] = {("autogpt-local-executor", "audit_key"): stored}

    def fake_get(svc: str, user: str) -> str | None:
        return store.get((svc, user))

    def fake_set(svc: str, user: str, value: str) -> None:
        store[(svc, user)] = value

    import keyring

    with (
        patch.object(keyring, "get_password", side_effect=fake_get),
        patch.object(keyring, "set_password", side_effect=fake_set),
        patch("autogpt_local_executor.audit._fallback_audit_key") as fallback,
    ):
        with pytest.raises(RuntimeError, match="audit key"):
            get_or_create_audit_key()
        fallback.assert_not_called()
    assert store[("autogpt-local-executor", "audit_key")] == stored


@pytest.mark.parametrize("blob", ["not-an-encrypted-key", "ab:cd", "00" * 16 + ":" + "ab" * 31])
def test_corrupt_fallback_key_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blob: str
) -> None:
    from autogpt_local_executor.audit import _fallback_audit_key

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "test-only-passphrase")
    path = tmp_path / "audit-key.enc"
    path.write_text(blob)

    with pytest.raises(RuntimeError, match="audit key"):
        _fallback_audit_key()

    assert path.read_text() == blob


@pytest.mark.asyncio
async def test_writer_rejects_changed_key_without_modifying_existing_chain(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    await writer.write("SHIM_START", request_id=None, details={})
    previous = writer.path.read_bytes()

    with pytest.raises(RuntimeError, match="audit log"):
        AuditWriter(path=writer.path, audit_key=b"x" * 32)

    assert writer.path.read_bytes() == previous


def test_writer_rejects_corrupt_existing_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.log"
    path.write_bytes(b"{truncated")

    with pytest.raises(RuntimeError, match="audit log"):
        _writer(tmp_path)

    assert path.read_bytes() == b"{truncated"


# ── Tail-state recovery across writer reopens ───────────────────────────────


@pytest.mark.asyncio
async def test_seq_continues_across_writer_restart(tmp_path: Path) -> None:
    w1 = _writer(tmp_path)
    for i in range(3):
        await w1.write("EXECUTE_COMMAND", request_id=str(i), details={"i": i}, result={})
    # New writer instance over the same file (simulates daemon restart).
    w2 = AuditWriter(path=tmp_path / "audit.log", audit_key=KEY)
    await w2.write("EXECUTE_COMMAND", request_id="post", details={"i": 99}, result={})
    lines = (tmp_path / "audit.log").read_text().splitlines()
    last = json.loads(lines[-1])
    assert last["seq"] == 4
    # Chain still verifies clean across the restart boundary.
    assert AuditWriter.verify(tmp_path / "audit.log", KEY) == []


# ── HMAC sanity ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hmac_matches_canonical_bytes(tmp_path: Path) -> None:
    """Re-implement the HMAC computation against the persisted record to
    catch any drift between canonicalization and what the verifier uses."""
    w = _writer(tmp_path)
    await w.write("EXECUTE_COMMAND", request_id="x", details={"command": "ls"}, result={})
    rec = json.loads((tmp_path / "audit.log").read_text().strip())
    shadow = {k: v for k, v in rec.items() if k != "hmac"}
    expected = hmac.new(KEY, canonical_bytes(shadow), hashlib.sha256).hexdigest()
    assert expected == rec["hmac"]


# ── Permissions: file is 0600 on POSIX ──────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_log_is_owner_only_on_posix(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("0600 not enforced on Windows")
    w = _writer(tmp_path)
    await w.write("EXECUTE_COMMAND", request_id="x", details={}, result={})
    mode = (tmp_path / "audit.log").stat().st_mode & 0o777
    assert mode == 0o600


# ── Concurrent writes never interleave ──────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_writes_serialize_via_internal_lock(tmp_path: Path) -> None:
    """Bursting writes from multiple tasks must still produce a clean chain."""
    w = _writer(tmp_path)
    await asyncio.gather(
        *[
            w.write("EXECUTE_COMMAND", request_id=str(i), details={"i": i}, result={})
            for i in range(20)
        ]
    )
    assert AuditWriter.verify(w.path, KEY) == []
    lines = w.path.read_text().splitlines()
    assert len(lines) == 20


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file locks and modes")
@pytest.mark.asyncio
async def test_audit_verify_reads_the_log_under_the_writers_lock(
    tmp_path: Path, monkeypatch
) -> None:
    """A verify that read mid-append could call a valid log truncated."""
    import fcntl

    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    held: list[bool] = []

    def verify(_path: Path, _key: bytes) -> list:
        with open(tmp_path / ".audit.log.lock", "a+b") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                held.append(True)
            else:
                fcntl.flock(probe, fcntl.LOCK_UN)
                held.append(False)
        return []

    monkeypatch.setattr(AuditWriter, "verify", staticmethod(verify))

    assert cli._audit_verify(writer.path) == 0
    assert held == [True]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file modes")
@pytest.mark.asyncio
async def test_a_log_removed_between_writes_comes_back_owner_only(tmp_path: Path) -> None:
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    writer.path.unlink()
    previous = os.umask(0o022)
    try:
        await writer.write("EXECUTE_COMMAND", request_id="b", details={}, result={})
    finally:
        os.umask(previous)

    assert stat.S_IMODE(writer.path.stat().st_mode) == 0o600


def _fake_windows_locking(monkeypatch, failures: list[OSError]) -> list[int]:
    """Drive _file_lock's Windows branch: msvcrt.locking fails with each of
    `failures` in turn, then succeeds."""
    modes: list[int] = []

    def locking(_fd: int, mode: int, _nbytes: int) -> None:
        modes.append(mode)
        if mode == 2 and failures:
            raise failures.pop(0)

    fake = types.SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setattr(sys, "platform", "win32")
    return modes


def test_the_windows_lock_waits_out_another_holder(tmp_path: Path, monkeypatch) -> None:
    modes = _fake_windows_locking(
        monkeypatch, [OSError(errno.EACCES, "locked"), OSError(errno.EACCES, "locked")]
    )

    with audit_module._file_lock(tmp_path / ".audit.log.lock"):
        pass

    assert modes == [2, 2, 2, 0]


def test_the_windows_lock_raises_an_error_that_is_not_contention(
    tmp_path: Path, monkeypatch
) -> None:
    """Retrying a permanent failure forever would hang daemon startup."""
    modes = _fake_windows_locking(monkeypatch, [OSError(errno.EBADF, "bad file descriptor")])

    with pytest.raises(OSError, match="bad file descriptor"):
        with audit_module._file_lock(tmp_path / ".audit.log.lock"):
            pass

    assert modes == [2]


@pytest.mark.asyncio
async def test_a_rotation_whose_new_log_cannot_be_created_starts_a_fresh_chain(
    tmp_path: Path, monkeypatch
) -> None:
    """The old log is already moved aside when the empty one can't be created;
    the same write then makes the file, and it must start at seq 1."""
    monkeypatch.setattr("autogpt_local_executor.audit.ROTATE_MAX_BYTES", 1)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    real_create = audit_module._create_owner_only

    def create(path: Path) -> None:
        if path == writer.path:
            raise OSError("disk full")
        real_create(path)

    monkeypatch.setattr(audit_module, "_create_owner_only", create)
    await writer.write("EXECUTE_COMMAND", request_id="b", details={}, result={})

    first = json.loads(writer.path.read_text().splitlines()[0])
    assert (first["request_id"], first["seq"], first["prev_hmac"]) == ("b", 1, None)
    assert AuditWriter.verify(writer.path, KEY) == []


def _lock_is_held(lock: Path) -> bool:
    import fcntl

    with open(lock, "rb") as probe:
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(probe, fcntl.LOCK_UN)
        return False


_posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX file locks and modes")
_not_root = pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes"
)


@_posix_only
@_not_root
def test_a_reader_that_cannot_write_the_lock_file_still_takes_the_lock(tmp_path: Path) -> None:
    """A read-only lock file (another account's log, a read-only volume) still
    has to keep the reader out while a writer appends."""
    log = tmp_path / "audit.log"
    lock = tmp_path / ".audit.log.lock"
    lock.touch()
    lock.chmod(0o400)

    with audit_module.holding_log_lock(log):
        assert _lock_is_held(lock)


@_posix_only
@_not_root
def test_a_lock_file_the_reader_cannot_open_stops_the_read(tmp_path: Path) -> None:
    """A writer may hold a lock the reader can't open; reading anyway could
    meet a half-written record, so the read stops."""
    log = tmp_path / "audit.log"
    lock = tmp_path / ".audit.log.lock"
    lock.touch()
    lock.chmod(0)
    try:
        with pytest.raises(audit_module.AuditLockError):
            with audit_module.holding_log_lock(log):
                pytest.fail("read without the lock")
    finally:
        lock.chmod(0o600)


@_posix_only
@_not_root
def test_a_missing_lock_file_that_cannot_be_created_lets_the_read_go_ahead(
    tmp_path: Path,
) -> None:
    """Every writer creates the lock file before it appends, so when it
    doesn't exist and can't be made, no writer holds it."""
    logs = tmp_path / "logs"
    logs.mkdir()
    logs.chmod(0o500)
    entered = False
    try:
        with audit_module.holding_log_lock(logs / "audit.log"):
            entered = True
    finally:
        logs.chmod(0o700)

    assert entered


def test_audit_reads_report_a_lock_they_cannot_take(tmp_path: Path, monkeypatch, capsys) -> None:
    @contextlib.contextmanager
    def refuse(_log: Path):
        raise audit_module.AuditLockError("permission denied")
        yield

    monkeypatch.setattr(audit_module, "holding_log_lock", refuse)
    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    log = tmp_path / "audit.log"
    log.write_text("")
    # An expired rotated log, so prune has something to delete under the lock.
    (tmp_path / "audit.log.20200101-000000").write_text('{"seq": 1, "ts": 1.0}\n')
    config = types.SimpleNamespace(audit_log_path=log)

    for command in ("verify", "verify-all", "export", "prune", "show"):
        args = types.SimpleNamespace(
            audit_command=command, output=tmp_path / "out.zip", older_than="1d", seq=1
        )
        assert cli._cmd_audit(config, args) == 2
        assert "Could not lock the audit log" in capsys.readouterr().err


def _rotate_before_the_lock(monkeypatch, writer: AuditWriter) -> None:
    """A rotation that lands after a reader lists the files but before it
    gets the lock moves the records to a name it never listed."""
    real_lock = audit_module.holding_log_lock

    @contextlib.contextmanager
    def rotate_then_lock(log: Path):
        writer.force_rotate()
        with real_lock(log):
            yield

    monkeypatch.setattr(audit_module, "holding_log_lock", rotate_then_lock)


@pytest.mark.asyncio
async def test_verify_all_lists_the_files_under_the_lock(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    verified: list[Path] = []
    real_verify = AuditWriter.verify

    def verify(path: Path, key: bytes) -> list:
        verified.append(path)
        return real_verify(path, key)

    monkeypatch.setattr(AuditWriter, "verify", staticmethod(verify))
    _rotate_before_the_lock(monkeypatch, writer)

    assert cli._audit_verify_all(writer.path) == 0
    assert [p for p in verified if p != writer.path] == list_rotated_files(writer.path) != []


@pytest.mark.asyncio
async def test_export_lists_the_files_under_the_lock(tmp_path: Path, monkeypatch) -> None:
    import zipfile

    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    _rotate_before_the_lock(monkeypatch, writer)
    output = tmp_path / "export.zip"

    assert cli._audit_export(writer.path, output) == 0
    rotated = list_rotated_files(writer.path)
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
    assert rotated and {p.name for p in rotated} <= names


@_posix_only
@pytest.mark.asyncio
async def test_prune_deletes_under_the_lock(tmp_path: Path, monkeypatch) -> None:
    """An export reads the files it listed under the lock; a prune deleting
    one meanwhile would end it with a half-written zip."""
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    writer.force_rotate()
    [rotated] = list_rotated_files(writer.path)
    old = json.loads(rotated.read_text().splitlines()[-1])
    os.utime(rotated, (1, 1))
    held: list[bool] = []
    real_unlink = Path.unlink

    def unlink(self: Path, *args, **kwargs) -> None:
        if self == rotated:
            held.append(_lock_is_held(tmp_path / ".audit.log.lock"))
        real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr("time.time", lambda: old["ts"] + 10 * 86400)

    assert cli._audit_prune(writer.path, "1d") == 0
    assert held == [True]
    assert not rotated.exists()


@pytest.mark.parametrize("duration", ["-1d", "0d", "0h", "1.5d", "+1d", " 1d"])
def test_prune_refuses_a_duration_that_is_not_a_positive_whole_number(duration: str) -> None:
    """`-1d` put the cutoff a day in the future, so every rotated file was deleted."""
    assert cli._parse_duration(duration) is None


@pytest.mark.asyncio
async def test_export_refuses_to_write_over_an_audit_log(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """ZipFile(output, "w") truncates the log before it is read into the zip."""
    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    before = writer.path.read_bytes()

    assert cli._audit_export(writer.path, writer.path) == 2
    assert writer.path.read_bytes() == before
    assert "audit log" in capsys.readouterr().out


@pytest.mark.parametrize("line", [b"[]", b'"text"', b"42", b"null"])
def test_a_record_that_is_not_an_object_is_a_parse_error(tmp_path: Path, line: bytes) -> None:
    path = tmp_path / "audit.log"
    path.write_bytes(line + b"\n")

    [violation] = AuditWriter.verify(path, KEY)

    assert violation.kind == "parse_error"


@pytest.mark.parametrize("line", [b"[]", b'"text"'])
def test_the_audit_commands_skip_a_record_that_is_not_an_object(
    tmp_path: Path, line: bytes, capsys
) -> None:
    path = tmp_path / "audit.log"
    path.write_bytes(line + b"\n")
    rotated = tmp_path / "audit.log.20200101-000000"
    rotated.write_bytes(line + b"\n")

    cli._print_record_line(line + b"\n")
    assert "[unparseable]" in capsys.readouterr().out
    assert cli._audit_show(path, 1) == 1
    assert cli._audit_prune(path, "1d") == 0
    assert rotated.exists()


def test_a_writer_refuses_a_log_whose_last_record_is_not_an_object(tmp_path: Path) -> None:
    """It used to stop with an AttributeError from inside the verifier."""
    (tmp_path / "audit.log").write_bytes(b"[]\n")

    with pytest.raises(RuntimeError, match="failed verification"):
        _writer(tmp_path)


class _StopFollowing(Exception):
    pass


def test_tail_follows_the_new_log_after_a_rotation(tmp_path: Path) -> None:
    path = tmp_path / "audit.log"
    path.write_bytes(b'{"seq": 1}\n')
    events = iter(["rotate", "stop"])

    def wait() -> None:
        event = next(events)
        if event == "rotate":
            path.rename(tmp_path / "audit.log.20200101-000000")
            path.write_bytes(b'{"seq": 1, "new": true}\n')
        else:
            raise _StopFollowing

    seen: list[bytes] = []
    with pytest.raises(_StopFollowing):
        for line in cli._follow(path, wait):
            seen.append(line)

    assert seen == [b'{"seq": 1}\n', b'{"seq": 1, "new": true}\n']


def test_tail_waits_for_the_rest_of_a_record_being_written(tmp_path: Path) -> None:
    """A record caught mid-append used to print as two unparseable halves."""
    path = tmp_path / "audit.log"
    path.write_bytes(b'{"seq": 1, ')
    events = iter(["finish", "stop"])

    def wait() -> None:
        if next(events) == "finish":
            with open(path, "ab") as f:
                f.write(b'"op": "X"}\n')
        else:
            raise _StopFollowing

    seen: list[bytes] = []
    with pytest.raises(_StopFollowing):
        for line in cli._follow(path, wait):
            seen.append(line)

    assert seen == [b'{"seq": 1, "op": "X"}\n']


@_posix_only
@_not_root
def test_a_writer_that_appears_during_an_unlocked_read_fails_the_read(tmp_path: Path) -> None:
    """Without a lock file the read goes ahead unlocked; if a writer creates
    one meanwhile, what was read may have been half written."""
    logs = tmp_path / "logs"
    logs.mkdir()
    logs.chmod(0o500)
    try:
        with pytest.raises(audit_module.AuditLockError, match="writer"):
            with audit_module.holding_log_lock(logs / "audit.log"):
                logs.chmod(0o700)
                (logs / ".audit.log.lock").touch()
    finally:
        logs.chmod(0o700)


@pytest.mark.asyncio
async def test_export_refuses_an_output_named_like_a_rotated_log(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """verify-all would later read the zip as a rotated log and fail on it."""
    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    output = tmp_path / "audit.log.export.zip"

    assert cli._audit_export(writer.path, output) == 2
    assert not output.exists()
    assert "rotated" in capsys.readouterr().out


def test_prune_judges_a_file_by_a_last_record_longer_than_8_kib(tmp_path: Path) -> None:
    """The tail window was 8 KiB, so a longer final record (a long command)
    failed to parse and the file was kept without a word."""
    current = tmp_path / "audit.log"
    current.write_text("")
    rotated = tmp_path / "audit.log.20200101-000000"
    record = {"seq": 1, "ts": 1.0, "details": {"command": "x" * 20_000}}
    rotated.write_text(json.dumps(record) + "\n")

    assert cli._audit_prune(current, "1d") == 0
    assert not rotated.exists()


@_posix_only
@pytest.mark.asyncio
async def test_a_new_log_is_owner_only_from_the_start(tmp_path: Path, monkeypatch) -> None:
    """The log was created with the umask's mode and then chmodded, so a
    chmod that failed left it readable by others."""
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})

    def refuse(*_args, **_kwargs) -> None:
        raise OSError("chmod not supported here")

    previous = os.umask(0o022)
    try:
        monkeypatch.setattr(audit_module.os, "chmod", refuse)
        writer.force_rotate()
    finally:
        os.umask(previous)

    assert stat.S_IMODE(writer.path.stat().st_mode) == 0o600


@_posix_only
@pytest.mark.asyncio
async def test_an_export_is_owner_only(tmp_path: Path, monkeypatch) -> None:
    """The zip holds every audit record, but got the umask's mode."""
    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    output = tmp_path / "export.zip"
    previous = os.umask(0o022)
    try:
        assert cli._audit_export(writer.path, output) == 0
    finally:
        os.umask(previous)

    assert stat.S_IMODE(output.stat().st_mode) == 0o600


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
def test_a_record_with_a_non_finite_number_is_a_parse_error(tmp_path: Path, number: str) -> None:
    """json.loads takes NaN and Infinity, but canonicalizing them raised, so
    verify ended in a traceback instead of a violation."""
    path = tmp_path / "audit.log"
    path.write_text(f'{{"seq": 1, "ts": {number}, "hmac": "00"}}\n')

    [violation] = AuditWriter.verify(path, KEY)

    assert violation.kind == "parse_error"


@_posix_only
@pytest.mark.asyncio
async def test_export_writes_the_archive_after_releasing_the_lock(
    tmp_path: Path, monkeypatch
) -> None:
    """The archive was written under the lock, so a slow destination held off
    every audit write, and the requests waiting on them."""
    import zipfile

    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    writer.force_rotate()
    await writer.write("EXECUTE_COMMAND", request_id="b", details={}, result={})
    held: list[bool] = []
    real_writestr = zipfile.ZipFile.writestr

    def writestr(self, name, data, *args, **kwargs):
        held.append(_lock_is_held(tmp_path / ".audit.log.lock"))
        return real_writestr(self, name, data, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", writestr)
    output = tmp_path / "export.zip"

    assert cli._audit_export(writer.path, output) == 0
    assert held and not any(held)
    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        current = zf.read("audit.log")
    assert {"audit.log", "manifest.json", "manifest.sig"} <= names
    assert len(names) == 4
    assert json.loads(current.splitlines()[-1])["request_id"] == "b"


@_posix_only
def test_the_lock_file_is_created_owner_only_and_an_old_one_is_tightened(tmp_path: Path) -> None:
    """The lock file got the umask's mode (0644). Another account that could
    open it could hold the lock, and every audit write would wait for good."""
    import stat

    from autogpt_local_executor.audit import _lock_path

    fresh = tmp_path / "fresh" / "audit.log"
    old = tmp_path / "old" / "audit.log"
    old.parent.mkdir()
    _lock_path(old).touch()
    _lock_path(old).chmod(0o644)
    umask = os.umask(0o022)
    try:
        AuditWriter(path=fresh, audit_key=KEY)
        AuditWriter(path=old, audit_key=KEY)
    finally:
        os.umask(umask)

    assert stat.S_IMODE(_lock_path(fresh).stat().st_mode) == 0o600
    assert stat.S_IMODE(_lock_path(old).stat().st_mode) == 0o600


def test_tail_starts_over_when_the_log_is_truncated_in_place(tmp_path: Path) -> None:
    """Truncation keeps the inode, so the follower sat at its old offset and
    printed nothing until the log grew past it again."""
    path = tmp_path / "audit.log"
    path.write_bytes(b'{"seq": 1, "op": "a long first record"}\n')
    events = iter(["truncate", "stop"])

    def wait() -> None:
        if next(events) == "truncate":
            with open(path, "r+b") as f:
                f.truncate(0)
                f.write(b'{"seq": 1}\n')
        else:
            raise _StopFollowing

    seen: list[bytes] = []
    with pytest.raises(_StopFollowing):
        for line in cli._follow(path, wait):
            seen.append(line)

    assert seen == [b'{"seq": 1, "op": "a long first record"}\n', b'{"seq": 1}\n']


def test_a_locked_keychain_is_not_replaced_by_the_file_key(monkeypatch) -> None:
    """With the passphrase set, a keychain that was still locked made the
    shim switch to the file's key, a different one, so the log it had been
    signing no longer verified."""
    import keyring
    import keyring.errors

    from autogpt_local_executor.audit import AuditKeyUnavailableError

    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    with (
        patch.object(keyring, "get_password", side_effect=keyring.errors.KeyringLocked("locked")),
        patch("autogpt_local_executor.audit._fallback_audit_key") as fallback,
    ):
        with pytest.raises(AuditKeyUnavailableError):
            get_or_create_audit_key()
    fallback.assert_not_called()


@pytest.mark.parametrize("error", ["NoKeyringError", "InitError"])
def test_no_usable_keychain_still_falls_back_to_the_file_key(error: str) -> None:
    import keyring
    import keyring.errors

    with (
        patch.object(keyring, "get_password", side_effect=getattr(keyring.errors, error)("none")),
        patch("autogpt_local_executor.audit._fallback_audit_key", return_value=b"k" * 32),
    ):
        assert get_or_create_audit_key() == b"k" * 32


# ── Export: descriptors, a prune meanwhile, the default name ───────────


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="counts open files in /proc")
@pytest.mark.asyncio
async def test_export_keeps_one_rotated_log_open_at_a_time(tmp_path: Path, monkeypatch) -> None:
    """Every rotated log stayed open until the archive was written. The shim
    never prunes on its own, so with enough of them open() hit the descriptor
    limit, and the export ended in a traceback with no archive."""
    import zipfile

    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    for index in range(40):
        await writer.write("EXECUTE_COMMAND", request_id=str(index), details={}, result={})
        writer.force_rotate()
    await writer.write("EXECUTE_COMMAND", request_id="last", details={}, result={})
    baseline = len(os.listdir("/proc/self/fd"))
    open_while_writing: list[int] = []
    real_writestr = zipfile.ZipFile.writestr

    def writestr(self, name, data, *args, **kwargs):
        open_while_writing.append(len(os.listdir("/proc/self/fd")))
        return real_writestr(self, name, data, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "writestr", writestr)

    assert cli._audit_export(writer.path, tmp_path / "export.zip") == 0
    assert max(open_while_writing) < baseline + 10


@pytest.mark.asyncio
async def test_an_export_whose_rotated_log_is_pruned_meanwhile_writes_nothing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """The set is listed under the lock and read after it; a prune in
    between must fail the export, not leave a file out of it."""
    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    writer.force_rotate()
    await writer.write("EXECUTE_COMMAND", request_id="b", details={}, result={})
    [rotated] = list_rotated_files(writer.path)
    real_lock = audit_module.holding_log_lock

    @contextlib.contextmanager
    def lock_then_prune(path):
        with real_lock(path):
            yield
        rotated.unlink()  # a prune that ran the moment the lock was free

    monkeypatch.setattr(audit_module, "holding_log_lock", lock_then_prune)
    output = tmp_path / "export.zip"

    assert cli._audit_export(writer.path, output) != 0
    assert "pruned" in capsys.readouterr().out
    assert not output.exists()
    assert [p.name for p in tmp_path.iterdir() if "export" in p.name] == []


@pytest.mark.asyncio
async def test_a_second_export_in_the_same_second_keeps_the_first(
    tmp_path: Path, monkeypatch
) -> None:
    """The default name has one-second precision, and a second export in that
    second truncated the first one's archive."""
    import datetime as real_datetime
    import zipfile

    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    exports = tmp_path / "exports"
    exports.mkdir()
    monkeypatch.chdir(exports)
    moment = real_datetime.datetime(2026, 11, 4, 3, 0, 0)

    class FrozenDatetime(real_datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return moment if tz is None else moment.replace(tzinfo=tz)

    monkeypatch.setattr(real_datetime, "datetime", FrozenDatetime)

    assert cli._audit_export(writer.path, None) == 0
    [first] = list(exports.iterdir())
    assert cli._audit_export(writer.path, None) == 0

    archives = sorted(exports.iterdir())
    assert len(archives) == 2 and first in archives
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            assert "audit.log" in zf.namelist()


# ── Prune: scan outside the lock, delete under it ──────────────────────


@_posix_only
@pytest.mark.asyncio
async def test_prune_reads_the_rotated_logs_without_holding_off_writers(
    tmp_path: Path, monkeypatch
) -> None:
    """Prune read every line of every rotated log under the writers' lock,
    and rotated logs are kept until pruned, so audit writes waited on it."""
    import builtins

    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    writer.force_rotate()
    [rotated] = list_rotated_files(writer.path)
    old = json.loads(rotated.read_text().splitlines()[-1])
    monkeypatch.setattr("time.time", lambda: old["ts"] + 10 * 86400)
    held_while_reading: list[bool] = []
    real_open = builtins.open

    def open_(file, *args, **kwargs):
        if str(file) == str(rotated):
            held_while_reading.append(_lock_is_held(tmp_path / ".audit.log.lock"))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_)

    assert cli._audit_prune(writer.path, "1d") == 0
    assert held_while_reading and not any(held_while_reading)
    assert not rotated.exists()


@pytest.mark.asyncio
async def test_prune_keeps_a_rotated_log_replaced_after_it_was_read(
    tmp_path: Path, monkeypatch
) -> None:
    """The scan runs outside the lock, so the file deleted must still be the
    one that was read."""
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    writer.force_rotate()
    [rotated] = list_rotated_files(writer.path)
    old = json.loads(rotated.read_text().splitlines()[-1])
    monkeypatch.setattr("time.time", lambda: old["ts"] + 10 * 86400)
    real_lock = audit_module.holding_log_lock
    newer = rotated.read_text().replace(str(old["ts"]), str(old["ts"] + 10 * 86400))

    @contextlib.contextmanager
    def replace_then_lock(path):
        rotated.unlink()
        rotated.write_text(newer + "\n")
        with real_lock(path):
            yield

    monkeypatch.setattr(audit_module, "holding_log_lock", replace_then_lock)

    assert cli._audit_prune(writer.path, "1d") == 0
    assert rotated.exists()


def test_a_record_with_an_integer_past_pythons_limit_is_a_parse_error(
    tmp_path: Path, capsys
) -> None:
    """Integers past 4,300 digits raise a plain ValueError rather than a
    JSONDecodeError, so verify, show and tail ended in a traceback."""
    path = tmp_path / "audit.log"
    huge = '{"seq": ' + "9" * 5000 + ', "ts": 1.0}'
    path.write_text(huge + "\n")

    [violation] = AuditWriter.verify(path, KEY)
    assert violation.kind == "parse_error"
    assert cli._audit_show(path, 1) != 0
    cli._print_record_line(huge.encode() + b"\n")
    assert "[unparseable]" in capsys.readouterr().out


@pytest.mark.parametrize(
    "record",
    ['{"seq": null, "op": "X"}', '{"seq": 1, "op": ["X"]}', '{"seq": {"a": 1}, "op": null}'],
)
def test_tail_prints_a_record_whose_seq_or_op_is_not_a_plain_value(capsys, record: str) -> None:
    """`{seq:>5}` and `{op:<22}` raise TypeError for null, a list or an
    object, so one such record ended audit tail."""
    cli._print_record_line(record.encode() + b"\n")

    assert "seq=" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_a_path_that_is_not_valid_utf_8_is_still_audited(tmp_path: Path) -> None:
    """A POSIX path whose bytes aren't UTF-8 decodes to lone surrogates,
    which UTF-8 can't encode, so the record raised and the handler dropped it
    while the operation itself went ahead."""
    writer = _writer(tmp_path)

    await writer.write(
        "FILE_WRITE",
        request_id="req-\udcff",
        details={"path": "/home/alice/caf\udce9.txt", "argv": ["x\udcff"]},
        result={},
    )

    [line] = writer.path.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert record["details"]["path"] == "/home/alice/caf\\udce9.txt"
    assert record["details"]["argv"] == ["x\\udcff"]
    assert record["request_id"] == "req-\\udcff"
    assert AuditWriter.verify(writer.path, KEY) == []


@pytest.mark.parametrize("failure", ["corrupt", "unreadable"])
def test_rotate_reports_a_log_it_cannot_open_instead_of_crashing(
    tmp_path: Path, monkeypatch, capsys, failure: str
) -> None:
    """Opening the log for a rotation raised out of `audit rotate`: a log that
    fails verification, or one it can't open, ended in a traceback."""
    monkeypatch.setattr("autogpt_local_executor.audit.get_or_create_audit_key", lambda: KEY)
    log = tmp_path / "audit.log"
    log.write_text('{"seq": 1, "ts": 1.0, "op": "X", "hmac": "00"}\n')
    if failure == "unreadable":

        def refuse(**_kwargs) -> None:
            raise OSError(errno.EACCES, "Permission denied")

        monkeypatch.setattr(audit_module, "AuditWriter", refuse)

    assert cli._audit_rotate(log) == 1
    out = capsys.readouterr().out
    assert ("audit verify" in out) if failure == "corrupt" else ("Permission denied" in out)


@_posix_only
@pytest.mark.asyncio
async def test_audit_show_reads_the_logs_under_the_writers_lock(
    tmp_path: Path, monkeypatch
) -> None:
    """show listed the logs and read them unlocked, so a rotation in between
    moved records to a file it never listed, and it answered "No record"."""
    import builtins

    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    held: list[bool] = []
    real_open = builtins.open

    def open_(file, *args, **kwargs):
        if str(file) == str(writer.path):
            held.append(_lock_is_held(tmp_path / ".audit.log.lock"))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_)

    assert cli._audit_show(writer.path, 1) == 0
    assert held == [True]
