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
from typing import Any
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
    lines = w.path.read_text(encoding="utf-8").splitlines()
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
    lines = w.path.read_text(encoding="utf-8").splitlines(keepends=True)
    # Delete the third line — verifier should report a sequence gap on the
    # fourth line (which now sits where seq=3 should be).
    del lines[2]
    w.path.write_text("".join(lines), encoding="utf-8")
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
    if w.path.read_text(encoding="utf-8").strip():
        first_line = w.path.read_text(encoding="utf-8").splitlines()[0]
        assert json.loads(first_line)["seq"] == 1


@pytest.mark.asyncio
async def test_force_rotate_returns_path_and_resets_seq(tmp_path: Path) -> None:
    w = _writer(tmp_path)
    await w.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    rotated = w.force_rotate()
    assert rotated is not None and rotated.exists()
    # Next write goes to a fresh file at seq=1.
    await w.write("EXECUTE_COMMAND", request_id="b", details={}, result={})
    rec = json.loads(w.path.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["seq"] == 1
    assert rec["prev_hmac"] is None


@pytest.mark.asyncio
async def test_cli_rotation_while_the_daemon_runs_keeps_both_chains_valid(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """`autogpt-shim audit rotate` has its own writer; the daemon's must notice."""
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    daemon = _writer(tmp_path)
    await daemon.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    await daemon.write("EXECUTE_COMMAND", request_id="b", details={}, result={})

    assert cli._audit_rotate(daemon.path) == 0
    await daemon.write("EXECUTE_COMMAND", request_id="c", details={}, result={})

    [rotated] = list_rotated_files(daemon.path)
    assert AuditWriter.verify(rotated, KEY) == []
    assert AuditWriter.verify(daemon.path, KEY) == []
    first = json.loads(daemon.path.read_text(encoding="utf-8").splitlines()[0])
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
    # Set once the rotation asks for the lock the append holds: a rotation
    # still starting up when the check below ran would pass it without
    # taking the lock at all.
    rotation_asked = threading.Event()
    real_lock = audit_module._file_lock

    @contextlib.contextmanager
    def watched_lock(path: Path, **kwargs: bool):
        if appending.is_set():
            rotation_asked.set()
        with real_lock(path, **kwargs):
            yield

    monkeypatch.setattr(audit_module, "_file_lock", watched_lock)
    write = asyncio.create_task(
        daemon.write("EXECUTE_COMMAND", request_id="b", details={}, result={})
    )
    assert await asyncio.to_thread(appending.wait, 30)
    rotation = asyncio.create_task(asyncio.to_thread(other.force_rotate))
    assert await asyncio.to_thread(rotation_asked.wait, 30)
    await asyncio.sleep(0.2)
    rotated_during_the_append = rotation.done()
    release.set()
    await write
    rotated = await rotation

    assert not rotated_during_the_append
    assert rotated is not None
    assert [
        json.loads(line)["request_id"] for line in rotated.read_text(encoding="utf-8").splitlines()
    ] == [
        "a",
        "b",
    ]
    assert AuditWriter.verify(rotated, KEY) == []


class _DiskFullPartWay:
    """The audit log's file on a disk that fills part way through a record:
    the first half of what is written lands, then the write fails."""

    def __init__(self, file: Any) -> None:
        self._file = file

    def write(self, data: Any) -> int:
        self._file.write(data[: len(data) // 2])
        self._file.flush()
        raise OSError(errno.ENOSPC, "No space left on device")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._file, name)

    def __enter__(self) -> _DiskFullPartWay:
        return self

    def __exit__(self, *_exc: object) -> bool:
        self._file.close()
        return False


def _appends_fill_the_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = open

    def full_disk_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = real_open(file, mode, *args, **kwargs)
        # Only an append, which opens the log by its descriptor.
        return _DiskFullPartWay(handle) if isinstance(file, int) and "a" in mode else handle

    monkeypatch.setattr(audit_module, "open", full_disk_open, raising=False)


@pytest.mark.asyncio
async def test_a_record_a_full_disk_cut_short_is_cut_off_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write the disk filled part way through left the first part of its
    record at the end of the log. Every later write then failed to read the
    log's last record, and the next start refused the log as tampered with,
    until someone cut the part off by hand. It is cut off when the write
    fails, so the log still ends with its last whole record, and the next
    write and start go on."""
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    whole = writer.path.read_bytes()

    with monkeypatch.context() as full_disk:
        _appends_fill_the_disk(full_disk)
        with pytest.raises(OSError) as raised:
            await writer.write("EXECUTE_COMMAND", request_id="b", details={}, result={})

    assert raised.value.errno == errno.ENOSPC
    assert writer.path.read_bytes() == whole
    await writer.write("EXECUTE_COMMAND", request_id="c", details={}, result={})
    restarted = AuditWriter(path=writer.path, audit_key=KEY)
    await restarted.write("EXECUTE_COMMAND", request_id="d", details={}, result={})

    assert AuditWriter.verify(writer.path, KEY) == []
    records = [json.loads(line) for line in writer.path.read_text(encoding="utf-8").splitlines()]
    assert [(record["request_id"], record["seq"]) for record in records] == [
        ("a", 1),
        ("c", 2),
        ("d", 3),
    ]


@pytest.mark.asyncio
async def test_a_record_written_whole_is_kept_when_what_follows_its_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a record cut short is cut off: one written whole stays, though
    the flush to disk after it failed, as some filesystems' fsync does."""
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})

    def no_fsync(_fd: int) -> None:
        raise OSError(errno.EINVAL, "Invalid argument")

    with monkeypatch.context() as unsynced:
        unsynced.setattr(audit_module.os, "fsync", no_fsync)
        await writer.write("EXECUTE_COMMAND", request_id="b", details={}, result={})

    assert AuditWriter.verify(writer.path, KEY) == []
    assert [
        json.loads(line)["request_id"]
        for line in writer.path.read_text(encoding="utf-8").splitlines()
    ] == ["a", "b"]


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

    contents = config.audit_log_path.read_text(encoding="utf-8")
    assert "shouldnotappear" not in contents
    # But the env key is logged.
    assert "SECRET" in contents


def test_computer_tool_text_is_redacted_to_its_length() -> None:
    """What the agent typed, set or scripted is kept out of the audit log,
    inside run_steps batches too; long strings are cut short."""
    from autogpt_local_executor.handlers import _audit_arguments

    details = _audit_arguments(
        {
            "pid": 3,
            "text": "my-password",
            "steps": [{"tool": "set_value", "args": {"value": "hunter2", "pid": 3}}],
            "script": "await cua.call('type_text', {text: 'x'})",
            "query": "q" * 300,
        }
    )

    assert details["pid"] == 3
    assert details["text"] == {"redacted_chars": 11}
    assert details["steps"][0]["args"] == {"value": {"redacted_chars": 7}, "pid": 3}
    assert details["script"] == {"redacted_chars": 40}
    assert details["query"].startswith("q" * 200) and details["query"].endswith("(300 chars)")


def test_keys_pressed_one_character_at_a_time_are_redacted() -> None:
    """press_key takes any single character, so a secret typed a key at a
    time went into the audit log whole, a run_steps batch's in one record.
    A shift chord types a character too. Named keys and command chords
    stay readable."""
    from autogpt_local_executor.handlers import _audit_arguments

    details = _audit_arguments(
        {
            "key": "p",
            "steps": [
                {"tool": "press_key", "args": {"key": "a", "pid": 3}},
                {"press_key": {"key": "enter"}},
                {"tool": "hotkey", "args": {"keys": ["shift", "s"]}},
                {"tool": "hotkey", "args": {"keys": ["ctrl", "c"]}},
            ],
        }
    )

    assert details["key"] == {"redacted_chars": 1}
    steps = details["steps"]
    assert steps[0]["args"] == {"key": {"redacted_chars": 1}, "pid": 3}
    assert steps[1] == {"press_key": {"key": "enter"}}
    assert steps[2]["args"] == {"keys": ["shift", {"redacted_chars": 1}]}
    assert steps[3]["args"] == {"keys": ["ctrl", "c"]}


@pytest.mark.parametrize(
    ("platform", "chord", "types"),
    [
        # On a Mac, Option with a key types a character: Option+L is @ on a
        # German layout, Option+Shift+7 a backslash.
        ("darwin", ["option", "l"], True),
        ("darwin", ["alt", "shift", "7"], True),
        ("darwin", ["cmd", "option", "m"], False),
        # Windows reads Ctrl+Alt as AltGr: Ctrl+Alt+Q is @ on a German layout.
        ("win32", ["ctrl", "alt", "q"], True),
        ("win32", ["alt", "f"], False),  # a menu's access key
        ("linux", ["alt", "f"], False),
        ("linux", ["ctrl", "alt", "t"], False),
    ],
)
def test_a_chord_that_types_a_character_is_redacted(
    platform: str, chord: list[str], types: bool
) -> None:
    """Only a Shift chord's character was redacted, so a secret typed with
    Option on a Mac, or with Ctrl+Alt on Windows, went into the audit log a
    chord at a time, its characters there to read back from the keyboard
    layout. The modifiers, and command chords, stay readable."""
    from autogpt_local_executor.handlers import _audit_arguments

    details = _audit_arguments(
        {"keys": chord, "steps": [{"tool": "hotkey", "args": {"keys": chord}}]},
        platform=platform,
    )

    expected = (
        [{"redacted_chars": 1} if len(key) == 1 else key for key in chord] if types else chord
    )
    assert details["keys"] == expected
    assert details["steps"][0]["args"]["keys"] == expected


@pytest.mark.parametrize(
    ("platform", "chord", "expected"),
    [
        # cua-driver 0.38 holds a modifier only by a name it knows, lowercased
        # but not trimmed, and presses or skips any other name.
        ("darwin", [" shift ", "a"], [" shift ", {"redacted_chars": 1}]),
        ("darwin", [" cmd ", "c"], [" cmd ", {"redacted_chars": 1}]),
        ("darwin", ["capslock", "a"], ["capslock", {"redacted_chars": 1}]),
        ("win32", ["super", "a"], ["super", {"redacted_chars": 1}]),
        ("linux", ["logo", "a"], ["logo", {"redacted_chars": 1}]),
        # On a Mac it trims the key it presses, so " a " types a.
        ("darwin", ["shift", " a "], ["shift", {"redacted_chars": 3}]),
        # Command chords by the names it holds stay readable.
        ("darwin", ["Cmd", "c"], ["Cmd", "c"]),
        ("win32", ["windows", "r"], ["windows", "r"]),
        ("linux", ["super", "a"], ["super", "a"]),
    ],
)
def test_a_chord_whose_names_the_driver_holds_no_modifier_for_is_redacted(
    platform: str, chord: list[str], expected: list[Any]
) -> None:
    """A `keys` list counted every name longer than one character as a held
    modifier, so a chord the driver reads without one kept its character in
    the audit log. cua-driver matches modifier names untrimmed and presses
    the last other key, so [" shift ", "a"] types a."""
    from autogpt_local_executor.handlers import _audit_arguments

    details = _audit_arguments(
        {"keys": chord, "steps": [{"tool": "hotkey", "args": {"keys": chord}}]},
        platform=platform,
    )

    assert details["keys"] == expected
    assert details["steps"][0]["args"]["keys"] == expected


@pytest.mark.parametrize(
    ("platform", "arguments", "expected"),
    [
        ("linux", {"key": "shift+a"}, {"key": ["shift", {"redacted_chars": 1}]}),
        ("linux", {"key": " Shift + A "}, {"key": ["Shift", {"redacted_chars": 1}]}),
        ("linux", {"key": " p "}, {"key": {"redacted_chars": 3}}),
        ("linux", {"keys": "shift+a"}, {"keys": ["shift", {"redacted_chars": 1}]}),
        ("linux", {"keys": "a"}, {"keys": [{"redacted_chars": 1}]}),
        ("darwin", {"key": "option+l"}, {"key": ["option", {"redacted_chars": 1}]}),
        ("win32", {"keys": "ctrl+alt+q"}, {"keys": ["ctrl", "alt", {"redacted_chars": 1}]}),
        # Command chords, named keys, and what the driver takes whole stay.
        ("linux", {"key": "ctrl+c"}, {"key": "ctrl+c"}),
        ("linux", {"keys": "cmd+s"}, {"keys": "cmd+s"}),
        ("linux", {"key": "shift+Right"}, {"key": "shift+Right"}),
        ("win32", {"keys": "alt+f"}, {"keys": "alt+f"}),
        ("linux", {"key": "a+"}, {"key": "a+"}),
    ],
)
def test_a_chord_written_as_one_string_is_redacted(
    platform: str, arguments: dict[str, str], expected: dict[str, Any]
) -> None:
    """cua-driver 0.38 takes a chord written as one string, splitting it at
    "+" (`press_key` key "shift+a", `hotkey` keys "shift+a"), and trims a
    key. The audit log redacted only a `keys` list and a `key` of exactly
    one character, so those kept the character they typed: a secret typed
    with Shift went into the log a chord at a time, inside run_steps too."""
    from autogpt_local_executor.handlers import _audit_arguments

    details = _audit_arguments(
        {**arguments, "steps": [{"tool": "press_key", "args": dict(arguments)}]},
        platform=platform,
    )

    assert {name: details[name] for name in arguments} == expected
    assert details["steps"][0]["args"] == expected


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
    raw = tampered_rotated.read_text(encoding="utf-8")
    rec = json.loads(raw.strip().splitlines()[0])
    rec["details"]["i"] = 999
    tampered_rotated.write_text(json.dumps(rec) + "\n", encoding="utf-8")

    # Verify each file individually — only the tampered one should report.
    assert AuditWriter.verify(clean_rotated, KEY) == []
    violations = AuditWriter.verify(tampered_rotated, KEY)
    assert violations and violations[0].kind == "hmac_mismatch"


# ── Keyring round-trip (mocked) ──────────────────────────────────────────────


def test_get_or_create_audit_key_round_trips_via_keyring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))  # the key's lock
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
    path.write_text(blob, encoding="utf-8")

    with pytest.raises(RuntimeError, match="audit key"):
        _fallback_audit_key()

    assert path.read_text(encoding="utf-8") == blob


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
    lines = (tmp_path / "audit.log").read_text(encoding="utf-8").splitlines()
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
    rec = json.loads((tmp_path / "audit.log").read_text(encoding="utf-8").strip())
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
    lines = w.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file locks and modes")
@pytest.mark.asyncio
async def test_audit_verify_reads_under_the_writers_lock_and_checks_after_it(
    tmp_path: Path, monkeypatch
) -> None:
    """A verify that read mid-append could call a valid log truncated. One
    that also checked every record under the lock held the daemon's audit
    writes, and the requests waiting on them, for the whole check."""
    import fcntl

    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    held: dict[str, list[bool]] = {"read": [], "check": []}

    def lock_held() -> bool:
        with open(tmp_path / ".audit.log.lock", "a+b") as probe:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            fcntl.flock(probe, fcntl.LOCK_UN)
            return False

    real_read = cli._read_audit_file
    real_check = AuditWriter.verify_records

    def read(path: Path) -> bytes | None:
        held["read"].append(lock_held())
        return real_read(path)

    def check(data: bytes, key: bytes) -> list:
        held["check"].append(lock_held())
        return real_check(data, key)

    monkeypatch.setattr(cli, "_read_audit_file", read)
    monkeypatch.setattr(AuditWriter, "verify_records", staticmethod(check))

    assert cli._audit_verify(writer.path) == 0
    assert held == {"read": [True], "check": [False]}


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

    first = json.loads(writer.path.read_text(encoding="utf-8").splitlines()[0])
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
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    log = tmp_path / "audit.log"
    log.write_text("", encoding="utf-8")
    # An expired rotated log, so prune has something to delete under the lock.
    (tmp_path / "audit.log.20200101-000000").write_text('{"seq": 1, "ts": 1.0}\n', encoding="utf-8")
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
async def test_verify_all_lists_the_files_under_the_lock(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    _rotate_before_the_lock(monkeypatch, writer)

    assert cli._audit_verify_all(writer.path) == 0
    out = capsys.readouterr().out
    rotated = list_rotated_files(writer.path)
    assert rotated and all(f"{path.name}: OK" in out for path in rotated)


@_posix_only
@pytest.mark.asyncio
async def test_verify_all_checks_records_without_holding_the_lock(
    tmp_path: Path, monkeypatch
) -> None:
    """verify-all held the writers' lock while it parsed and HMACed every
    record of every retained file, so each audit write the daemon made
    meanwhile waited for the whole scan."""
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    writer.force_rotate()
    await writer.write("EXECUTE_COMMAND", request_id="b", details={}, result={})
    held: list[bool] = []
    real_canonical = audit_module.canonical_bytes

    def canonical(record):
        held.append(_lock_is_held(tmp_path / ".audit.log.lock"))
        return real_canonical(record)

    monkeypatch.setattr(audit_module, "canonical_bytes", canonical)

    assert cli._audit_verify_all(writer.path) == 0
    assert held and not any(held)


@pytest.mark.asyncio
async def test_verify_all_skips_a_rotated_file_pruned_while_it_runs(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    writer.force_rotate()
    await writer.write("EXECUTE_COMMAND", request_id="b", details={}, result={})
    [rotated] = list_rotated_files(writer.path)
    real_lock = audit_module.holding_log_lock
    holds = 0

    @contextlib.contextmanager
    def prune_after_the_first_hold(log: Path):
        nonlocal holds
        holds += 1
        if holds == 2:
            rotated.unlink()
        with real_lock(log):
            yield

    monkeypatch.setattr(audit_module, "holding_log_lock", prune_after_the_first_hold)

    assert cli._audit_verify_all(writer.path) == 0
    out = capsys.readouterr().out
    assert f"{rotated.name}: removed while the check ran" in out
    assert "All 1 files verified clean." in out


@pytest.mark.asyncio
async def test_export_lists_the_files_under_the_lock(tmp_path: Path, monkeypatch) -> None:
    import zipfile

    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    _rotate_before_the_lock(monkeypatch, writer)
    output = tmp_path / "export.zip"

    assert cli._audit_export(writer.path, output) == 0
    rotated = list_rotated_files(writer.path)
    with zipfile.ZipFile(output) as zf:
        archived = {name: zf.read(name) for name in zf.namelist()}
    assert rotated and {p.name for p in rotated} <= archived.keys()
    # The bytes too: an export that packed an empty or wrong body passed.
    assert all(archived[p.name] == p.read_bytes() for p in rotated)


@_posix_only
@pytest.mark.asyncio
async def test_prune_deletes_under_the_lock(tmp_path: Path, monkeypatch) -> None:
    """An export reads the files it listed under the lock; a prune deleting
    one meanwhile would end it with a half-written zip."""
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    writer.force_rotate()
    [rotated] = list_rotated_files(writer.path)
    old = json.loads(rotated.read_text(encoding="utf-8").splitlines()[-1])
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


def test_prune_refuses_a_duration_past_what_a_cutoff_can_hold(tmp_path: Path, capsys) -> None:
    """A duration hundreds of digits long parsed, then the cutoff's
    subtraction raised OverflowError, and prune ended in a traceback
    instead of saying the duration was bad."""
    duration = "9" * 400 + "d"

    assert cli._parse_duration(duration) is None
    assert cli._audit_prune(tmp_path / "audit.log", duration) == 2
    assert "Could not parse duration" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_export_refuses_to_write_over_an_audit_log(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """ZipFile(output, "w") truncates the log before it is read into the zip."""
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    before = writer.path.read_bytes()

    assert cli._audit_export(writer.path, writer.path) == 2
    assert writer.path.read_bytes() == before
    assert "audit log" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_export_refuses_to_write_over_the_logs_lock(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """An export named after the writers' lock replaced it, so a writer that
    opened the lock afterwards locked another file than the daemon held, and
    both appended to the log."""
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    lock = tmp_path / ".audit.log.lock"
    before = lock.stat()

    assert cli._audit_export(writer.path, lock) == 2
    assert lock.stat().st_ino == before.st_ino
    assert lock.stat().st_size == before.st_size
    assert "lock" in capsys.readouterr().out


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


def test_ctrl_c_ends_audit_tail_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """audit tail follows the log until Ctrl+C, its only way out, which it
    didn't catch: every tail ended with KeyboardInterrupt's traceback."""
    import time

    path = tmp_path / "audit.log"
    path.write_bytes(b'{"seq": 1, "op": "FILE_READ", "result": {"ok": true}}\n')

    def ctrl_c(_seconds: float) -> None:
        raise KeyboardInterrupt  # pressed while tail waits for the next record

    monkeypatch.setattr(time, "sleep", ctrl_c)
    try:
        code = cli._audit_tail(path)
    except KeyboardInterrupt:
        pytest.fail("Ctrl+C ended audit tail with a traceback")

    assert code == 0
    assert "seq=    1  FILE_READ" in capsys.readouterr().out


def test_tail_reads_a_record_appended_just_before_a_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tail read to the end of the log and only then asked whether a
    rotation had moved it. A record appended, and the log rotated, between
    the two was in the old file it then stopped reading, and never printed."""
    path = tmp_path / "audit.log"
    path.write_bytes(b'{"seq": 1}\n')
    real_open = cli._open_for_following
    rotated = False

    class WriterBetweenTheSteps:
        def __init__(self, f) -> None:
            self._f = f

        def readline(self) -> bytes:
            nonlocal rotated
            chunk = self._f.readline()
            if not chunk and not rotated:
                rotated = True
                with open(path, "ab") as f:
                    f.write(b'{"seq": 2}\n')
                path.rename(tmp_path / "audit.log.20200101-000000")
                path.write_bytes(b'{"seq": 1, "new": true}\n')
            return chunk

        def __getattr__(self, name: str):
            return getattr(self._f, name)

    monkeypatch.setattr(cli, "_open_for_following", lambda p: WriterBetweenTheSteps(real_open(p)))
    waits = 0

    def wait() -> None:
        nonlocal waits
        waits += 1
        if waits > 3:
            raise _StopFollowing

    seen: list[bytes] = []
    with pytest.raises(_StopFollowing):
        for line in cli._follow(path, wait):
            seen.append(line)

    assert seen == [b'{"seq": 1}\n', b'{"seq": 2}\n', b'{"seq": 1, "new": true}\n']


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
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
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
    current.write_text("", encoding="utf-8")
    rotated = tmp_path / "audit.log.20200101-000000"
    record = {"seq": 1, "ts": 1.0, "details": {"command": "x" * 20_000}}
    rotated.write_text(json.dumps(record) + "\n", encoding="utf-8")

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
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
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
    path.write_text(f'{{"seq": 1, "ts": {number}, "hmac": "00"}}\n', encoding="utf-8")

    [violation] = AuditWriter.verify(path, KEY)

    assert violation.kind == "parse_error"


def test_a_record_that_parses_but_cannot_be_canonicalized_is_a_violation(
    tmp_path: Path,
) -> None:
    """The case below is too deep for the JSON parser itself, so it passed
    with the canonicalizer's RecursionError uncaught. Some hundreds of
    levels parse and only the canonicalizer, which recurses deeper per
    level, gives up (700 does on Python 3.11 to 3.13)."""
    depth = 700
    nested = '{"a":' * depth + "1" + "}" * depth
    record = f'{{"seq": 1, "ts": 1.0, "details": {nested}, "hmac": "00"}}'
    json.loads(record)  # this one parses
    path = tmp_path / "audit.log"
    path.write_text(record + "\n", encoding="utf-8")

    [violation] = AuditWriter.verify(path, KEY)

    assert violation.kind == "parse_error"
    assert "uncanonicalizable record" in violation.message


def test_a_record_nested_too_deeply_to_check_is_a_parse_error(tmp_path: Path) -> None:
    """A record nested some hundreds of levels deep parses, but canonicalizing
    it raised RecursionError, which the verifier didn't catch, so verify and
    verify-all ended in a traceback instead of a violation."""
    depth = 2000
    nested = '{"a":' * depth + "1" + "}" * depth
    path = tmp_path / "audit.log"
    path.write_text(
        f'{{"seq": 1, "ts": 1.0, "details": {nested}, "hmac": "00"}}\n', encoding="utf-8"
    )

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

    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
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


@pytest.mark.parametrize("name", ["manifest.json", "manifest.sig", "Manifest.JSON"])
@pytest.mark.asyncio
async def test_export_refuses_a_log_named_like_its_manifest(
    tmp_path: Path, monkeypatch, capsys, name: str
) -> None:
    """A log named manifest.json or manifest.sig went into the archive under
    the name the manifest then took too: two members of one name, where a
    reader by name gets the manifest, so the log's hash in it never matched.
    On a filesystem that ignores case, one name overwrites the other when
    the archive is unpacked."""
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    writer = AuditWriter(path=tmp_path / name, audit_key=KEY, machine_id="test-machine")
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    output = tmp_path / "export.zip"

    assert cli._audit_export(writer.path, output) == 2
    assert "manifest" in capsys.readouterr().out
    assert not output.exists()


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


class _PriorityRaises(type):
    @property
    def priority(cls) -> float:
        raise RuntimeError("The Secret Service daemon is neither running nor activatable")


class KeyringNotUpYet(metaclass=_PriorityRaises):
    """A backend keyring can load whose service isn't running yet, as
    SecretService is before the desktop session starts it."""


@pytest.mark.parametrize(
    ("backend", "for_good"),
    [
        ("keyring.backends.no_such_backend.Keyring", True),  # no such module
        ("keyring.backends.fail.NoSuchKeyring", True),  # no such class
        ("NoModuleKeyring", True),  # a name without its module
        (f"{__name__}.KeyringNotUpYet", False),
    ],
)
def test_a_keyring_backend_that_cant_be_loaded_is_not_waited_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, for_good: bool
) -> None:
    """A PYTHON_KEYRING_BACKEND naming a backend that isn't there raised
    ModuleNotFoundError, AttributeError or ValueError from the key read,
    which the shim took for a keychain still locked: the service exited to
    be started again, every few seconds, over a setting no restart fixes.
    A backend that is there but not running yet is still waited for."""
    import keyring.core

    from autogpt_local_executor.audit import AuditKeyError, AuditKeyUnavailableError

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", backend)
    # keyring picks its backend once per process; have it pick again.
    monkeypatch.setattr(keyring.core, "_keyring_backend", None)

    expected = AuditKeyError if for_good else AuditKeyUnavailableError
    with pytest.raises(expected) as raised:
        get_or_create_audit_key(tmp_path / "audit.log")

    assert ("PYTHON_KEYRING_BACKEND" in str(raised.value)) is for_good
    assert not (tmp_path / "audit-key.enc").exists()


@pytest.mark.parametrize("error", ["NoKeyringError", "InitError"])
def test_no_usable_keychain_still_falls_back_to_the_file_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    """With the passphrase set, and no audit log yet for another key to have
    signed, the file key stands in, and stays the key from then on."""
    import keyring
    import keyring.errors

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    with patch.object(keyring, "get_password", side_effect=getattr(keyring.errors, error)("none")):
        first = get_or_create_audit_key(tmp_path / "audit.log")
        (tmp_path / "audit.log").write_text("", encoding="utf-8")
        again = get_or_create_audit_key(tmp_path / "audit.log")

    assert len(first) == 32
    assert again == first


@pytest.mark.parametrize("error", ["NoKeyringError", "InitError"])
def test_a_keychain_missing_at_start_is_waited_for_without_the_file_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    """With no keychain and no passphrase the audit key failed for good, so
    the systemd unit stayed stopped when its Secret Service came up after
    it (one the desktop session starts later, say). The shim can't run
    without a keychain, which holds its sign-in, so a missing one at start
    is waited for."""
    import keyring
    import keyring.errors

    from autogpt_local_executor.audit import AuditKeyUnavailableError

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.delenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", raising=False)
    with patch.object(keyring, "get_password", side_effect=getattr(keyring.errors, error)("none")):
        with pytest.raises(AuditKeyUnavailableError, match="AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE"):
            get_or_create_audit_key(tmp_path / "audit.log")


@pytest.mark.parametrize("error", ["NoKeyringError", "InitError"])
def test_a_file_key_without_its_passphrase_waits_for_the_keychain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    """With no keychain up yet and a file key that can't be read without the
    passphrase, the audit key failed with a plain RuntimeError, which the
    daemon took for a failure no restart fixes: the service stayed stopped,
    though the keychain may hold the key its log was signed with (the file
    key is every deployment's, and may be another's). The start waits for
    the keychain, as it does without a file key, and says what to set."""
    import secrets

    import keyring
    import keyring.errors

    from autogpt_local_executor.config import ShimConfig
    from autogpt_local_executor.daemon import DaemonPreflightError, ShimDaemon

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.delenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", raising=False)
    fallback = tmp_path / "audit-key.enc"
    salt = secrets.token_bytes(16)
    ciphertext = audit_module._xor_encrypt("another deployment's", salt, secrets.token_bytes(32))
    fallback.write_text(f"{salt.hex()}:{ciphertext.hex()}", encoding="utf-8")
    file_key = fallback.read_bytes()
    log = tmp_path / "audit.log"
    config = ShimConfig(session_id="session-1", allowed_root=tmp_path, audit_log_path=log)

    not_up = getattr(keyring.errors, error)("not up yet")
    with patch.object(keyring, "get_password", side_effect=not_up):
        with pytest.raises(DaemonPreflightError) as raised:
            ShimDaemon._build_audit_writer(config)

    assert raised.value.transient is True
    assert "AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE" in str(raised.value)
    assert str(fallback) in str(raised.value)
    assert fallback.read_bytes() == file_key
    # The next start, with the keychain up, finds the key there.
    signing_key = secrets.token_bytes(32)
    with patch.object(keyring, "get_password", return_value=signing_key.hex()):
        assert get_or_create_audit_key(log) == signing_key


def test_a_keychain_missing_at_start_is_not_replaced_by_a_new_file_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the passphrase set, a keychain that wasn't up yet made the shim
    create a file key, a different one from the keychain's that had signed
    the log, so the log stopped verifying and the stray file key stayed."""
    import keyring
    import keyring.errors

    from autogpt_local_executor.audit import AuditKeyUnavailableError

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    log = tmp_path / "audit.log"
    log.write_text('{"seq": 1}\n', encoding="utf-8")
    with patch.object(keyring, "get_password", side_effect=keyring.errors.NoKeyringError("none")):
        with pytest.raises(AuditKeyUnavailableError, match="move it aside"):
            get_or_create_audit_key(log)

    assert not (tmp_path / "audit-key.enc").exists()


@pytest.mark.asyncio
async def test_an_audit_command_never_makes_a_file_key_for_a_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """With no keychain up and the passphrase set, `audit verify` made a file
    key for a log the keychain's key had signed: it reported the log corrupt,
    and the stray key then stood in for the keychain at the daemon's next
    start, which found its own log failing verification and stopped."""
    import keyring
    import keyring.errors

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    with patch.object(keyring, "get_password", side_effect=keyring.errors.NoKeyringError("none")):
        assert cli._audit_verify(writer.path) == 2
        assert cli._audit_verify_all(writer.path) == 2
        assert cli._audit_rotate(writer.path) == 2
        assert cli._audit_export(writer.path, tmp_path / "export.zip") == 2

    assert not (tmp_path / "audit-key.enc").exists()
    assert "Could not load audit key" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_an_audit_command_never_stores_a_keychain_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the keychain up and holding no key, `audit verify` stored a new
    one, which no existing log was signed with."""
    import keyring

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))  # no file key either
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    stored: list[str] = []
    with (
        patch.object(keyring, "get_password", return_value=None),
        patch.object(keyring, "set_password", side_effect=lambda *args: stored.append(args[2])),
    ):
        assert cli._audit_verify(writer.path) == 2

    assert stored == []


def test_an_empty_keychain_gets_no_new_key_for_a_signed_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the keychain up but its audit key gone, the daemon stored a new
    key, failed to verify its log with it, and stopped as if the log were
    corrupt, leaving the new key behind in place of the one that signed it."""
    import keyring

    from autogpt_local_executor.audit import AuditKeyError

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))  # the key's lock
    log = tmp_path / "audit.log"
    log.write_text('{"seq": 1}\n', encoding="utf-8")
    stored: list[str] = []
    with (
        patch.object(keyring, "get_password", return_value=None),
        patch.object(keyring, "set_password", side_effect=lambda *args: stored.append(args[2])),
    ):
        with pytest.raises(AuditKeyError, match="move the log aside"):
            get_or_create_audit_key(log)
        # A first start, with no log yet, makes one as before.
        assert len(get_or_create_audit_key(tmp_path / "new.log")) == 32

    assert len(stored) == 1


def test_no_keyring_package_gets_no_new_file_key_for_a_signed_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the keyring package, and with the passphrase set, the daemon
    made a file key for a log the keychain's key had signed, so the log no
    longer verified."""
    from autogpt_local_executor.audit import AuditKeyError

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    monkeypatch.setitem(sys.modules, "keyring", None)
    log = tmp_path / "audit.log"
    log.write_text('{"seq": 1}\n', encoding="utf-8")

    with pytest.raises(AuditKeyError, match="move the log aside"):
        get_or_create_audit_key(log)

    assert not (tmp_path / "audit-key.enc").exists()


def test_a_keychain_that_turns_up_later_keeps_the_file_key_that_signed_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first start without a keychain made a file key and signed the log
    with it. Once a keychain was available, holding no audit key, the daemon
    stored a new one there, the log no longer verified, and it stopped."""
    import keyring
    import keyring.errors

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    log = tmp_path / "audit.log"
    with patch.object(keyring, "get_password", side_effect=keyring.errors.NoKeyringError("none")):
        signing_key = get_or_create_audit_key(log)
    log.write_text('{"seq": 1}\n', encoding="utf-8")
    stored: list[str] = []
    with (
        patch.object(keyring, "get_password", return_value=None),
        patch.object(keyring, "set_password", side_effect=lambda *args: stored.append(args[2])),
    ):
        assert get_or_create_audit_key(log) == signing_key
        assert cli._audit_verify(log) != 2  # the audit commands read it too

    assert stored == []


def _start_meanwhile(log: Path) -> tuple[threading.Thread, list[object]]:
    """A second start, in a thread, finding its audit key for `log`: what
    get_or_create_audit_key returned or raised."""
    outcome: list[object] = []

    def start() -> None:
        try:
            outcome.append(get_or_create_audit_key(log))
        except Exception as exc:
            outcome.append(exc)

    return threading.Thread(target=start, daemon=True), outcome


def test_a_start_that_finds_no_key_while_another_makes_one_uses_that_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two starts at once with no key stored (the service, and a `start` run
    by hand) each made a key and stored it; the keychain kept the last one,
    and the log signed with the other never verified again. A key is made
    under a lock that every start takes, whatever its log, and one that
    waited for it looks again."""
    import keyring

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.delenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", raising=False)
    store: dict[tuple[str, str], str] = {}
    looked = threading.Event()

    def fake_get(service: str, user: str) -> str | None:
        looked.set()
        return store.get((service, user))

    def fake_set(service: str, user: str, value: str) -> None:
        store[(service, user)] = value

    the_other_starts = b"o" * 32
    start, outcome = _start_meanwhile(tmp_path / "another-deployment" / "audit.log")
    with (
        patch.object(keyring, "get_password", side_effect=fake_get),
        patch.object(keyring, "set_password", side_effect=fake_set),
    ):
        # The other start, for a log of its own, is making its key.
        with audit_module._file_lock(tmp_path / ".audit-key.lock"):
            start.start()
            assert looked.wait(timeout=30)
            start.join(timeout=1)
            assert start.is_alive(), f"made a key while another start made one: {outcome}"
            store[("autogpt-local-executor", "audit_key")] = the_other_starts.hex()
        start.join(timeout=30)

    assert outcome == [the_other_starts]
    assert store == {("autogpt-local-executor", "audit_key"): the_other_starts.hex()}


def test_a_file_key_is_made_and_read_under_the_same_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no keychain and the passphrase set, two first starts each wrote
    a file key of their own, the file kept the last, and one start could
    read the other's half written, which stopped it for good as malformed."""
    import keyring
    import keyring.errors

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    looked = threading.Event()

    def no_keychain(_service: str, _user: str) -> str | None:
        looked.set()
        raise keyring.errors.NoKeyringError("none")

    the_other_starts = b"o" * 32
    salt = b"s" * 16
    start, outcome = _start_meanwhile(tmp_path / "audit.log")
    with patch.object(keyring, "get_password", side_effect=no_keychain):
        with audit_module._file_lock(tmp_path / ".audit-key.lock"):
            start.start()
            assert looked.wait(timeout=30)
            start.join(timeout=1)
            assert start.is_alive(), f"made a file key while another start made one: {outcome}"
            ciphertext = audit_module._xor_encrypt("passphrase", salt, the_other_starts)
            (tmp_path / "audit-key.enc").write_text(
                f"{salt.hex()}:{ciphertext.hex()}", encoding="utf-8"
            )
        start.join(timeout=30)

    assert outcome == [the_other_starts]


def test_a_start_looks_for_the_key_again_before_it_refuses_for_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start that found no key refused for good when the log it is for
    existed, before taking the lock it looks again under. Another start
    could store the key and make the log just after this one looked, so the
    refusal came while the key that signed the log was stored, and a
    supervised start stayed stopped."""
    import secrets

    import keyring

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.delenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", raising=False)
    log = tmp_path / "audit.log"
    other_key = secrets.token_bytes(32)
    keychain: dict[tuple[str, str], str] = {}
    reads = 0

    def read(service: str, user: str) -> str | None:
        nonlocal reads
        reads += 1
        found = keychain.get((service, user))
        if reads == 1:
            # The other start stores its key and makes its log just after.
            keychain[(service, user)] = other_key.hex()
            log.write_text("", encoding="utf-8")
        return found

    stored: list[str] = []
    with (
        patch.object(keyring, "get_password", side_effect=read),
        patch.object(keyring, "set_password", side_effect=lambda *args: stored.append(args[2])),
    ):
        key = get_or_create_audit_key(log)

    assert key == other_key
    assert stored == []


def test_without_the_keyring_package_a_start_looks_for_the_file_key_again_before_it_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the keyring package, a start that found no file key refused
    for good when the log existed, before taking the lock the file key is
    read under, though another start could have made that key and signed
    the log with it just after this one looked."""
    import secrets

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    monkeypatch.setitem(sys.modules, "keyring", None)
    log = tmp_path / "audit.log"
    fallback = tmp_path / "audit-key.enc"
    other_key = secrets.token_bytes(32)
    real_exists = Path.exists
    looked = False

    def exists(self: Path, *args: Any, **kwargs: Any) -> bool:
        nonlocal looked
        found = real_exists(self, *args, **kwargs)
        if self == fallback and not looked:
            looked = True
            # The other start makes the file key and its log just after.
            salt = secrets.token_bytes(16)
            ciphertext = audit_module._xor_encrypt("passphrase", salt, other_key)
            fallback.write_text(f"{salt.hex()}:{ciphertext.hex()}", encoding="utf-8")
            log.write_text("", encoding="utf-8")
        return found

    monkeypatch.setattr(Path, "exists", exists)

    assert get_or_create_audit_key(log) == other_key


@pytest.mark.parametrize("error", ["NoKeyringError", "InitError"])
def test_without_a_keychain_a_start_looks_for_the_file_key_again_before_it_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    """With no keychain up, a start that found no file key refused for the
    log (exit 75) before taking the lock the file key is read under, though
    another start could have made that key and signed the log with it just
    after this one looked: a start run by hand failed while the key that
    signed its log was there."""
    import secrets

    import keyring
    import keyring.errors

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    log = tmp_path / "audit.log"
    fallback = tmp_path / "audit-key.enc"
    other_key = secrets.token_bytes(32)
    real_exists = Path.exists
    looked = False

    def exists(self: Path, *args: Any, **kwargs: Any) -> bool:
        nonlocal looked
        found = real_exists(self, *args, **kwargs)
        if self == fallback and not looked:
            looked = True
            # The other start makes the file key and its log just after.
            salt = secrets.token_bytes(16)
            ciphertext = audit_module._xor_encrypt("passphrase", salt, other_key)
            fallback.write_text(f"{salt.hex()}:{ciphertext.hex()}", encoding="utf-8")
            log.write_text("", encoding="utf-8")
        return found

    monkeypatch.setattr(Path, "exists", exists)
    no_keychain = getattr(keyring.errors, error)("none")
    with patch.object(keyring, "get_password", side_effect=no_keychain):
        assert get_or_create_audit_key(log) == other_key


def test_without_the_keyring_package_an_audit_command_with_no_file_key_stops_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An audit command makes no key, so it has nothing to wait for the lock
    for: with no file key it stops before taking it, and makes no folder."""
    from autogpt_local_executor.audit import AuditKeyError

    fallback_dir = tmp_path / "fallback"
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(fallback_dir))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    monkeypatch.setitem(sys.modules, "keyring", None)

    with pytest.raises(AuditKeyError, match="none to read"):
        audit_module.load_audit_key()

    assert not fallback_dir.exists()


# ── Export: descriptors, a prune meanwhile, the default name ───────────


@pytest.mark.skipif(not Path("/proc/self/fd").is_dir(), reason="counts open files in /proc")
@pytest.mark.asyncio
async def test_export_keeps_one_rotated_log_open_at_a_time(tmp_path: Path, monkeypatch) -> None:
    """Every rotated log stayed open until the archive was written. The shim
    never prunes on its own, so with enough of them open() hit the descriptor
    limit, and the export ended in a traceback with no archive."""
    import zipfile

    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
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
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
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

    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
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
    old = json.loads(rotated.read_text(encoding="utf-8").splitlines()[-1])
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
    old = json.loads(rotated.read_text(encoding="utf-8").splitlines()[-1])
    monkeypatch.setattr("time.time", lambda: old["ts"] + 10 * 86400)
    real_lock = audit_module.holding_log_lock
    newer = rotated.read_text(encoding="utf-8").replace(str(old["ts"]), str(old["ts"] + 10 * 86400))

    @contextlib.contextmanager
    def replace_then_lock(path):
        rotated.unlink()
        rotated.write_text(newer + "\n", encoding="utf-8")
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
    path.write_text(huge + "\n", encoding="utf-8")

    [violation] = AuditWriter.verify(path, KEY)
    assert violation.kind == "parse_error"
    assert cli._audit_show(path, 1) != 0
    cli._print_record_line(huge.encode() + b"\n")
    assert "[unparseable]" in capsys.readouterr().out


def test_tail_marks_a_line_nested_too_deeply_to_parse(capsys) -> None:
    """json.loads raises RecursionError for a line nested past Python's
    limit, which ended `audit tail` with a traceback."""
    nested = '{"a":' * 100_000 + "1" + "}" * 100_000

    cli._print_record_line(nested.encode() + b"\n")

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


@pytest.mark.asyncio
async def test_an_escaped_key_keeps_both_values_when_it_spells_another_key(
    tmp_path: Path,
) -> None:
    """A lone surrogate key was escaped to `\\udcff`, which is also how a
    key a request carries can be spelled, so one of the two values was
    dropped from the record, a refused call's included."""
    writer = _writer(tmp_path)

    await writer.write(
        "COMPUTER_TOOL_CALL",
        request_id="req-1",
        details={"arguments": {"\udcff": "lone", "\\udcff": "spelled", "a\\b": "plain"}},
        result={},
    )

    [line] = writer.path.read_text(encoding="utf-8").splitlines()
    arguments = json.loads(line)["details"]["arguments"]
    assert sorted(arguments.values()) == ["lone", "plain", "spelled"]
    assert arguments["\\udcff"] == "lone"
    assert AuditWriter.verify(writer.path, KEY) == []


@pytest.mark.asyncio
async def test_keys_that_need_no_escape_are_written_as_they_are(tmp_path: Path) -> None:
    writer = _writer(tmp_path)

    await writer.write(
        "FILE_WRITE", request_id="req-1", details={"a\\b": 1, "path": "x"}, result={}
    )

    [line] = writer.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(line)["details"] == {"a\\b": 1, "path": "x"}


@pytest.mark.parametrize("failure", ["corrupt", "unreadable"])
def test_rotate_reports_a_log_it_cannot_open_instead_of_crashing(
    tmp_path: Path, monkeypatch, capsys, failure: str
) -> None:
    """Opening the log for a rotation raised out of `audit rotate`: a log that
    fails verification, or one it can't open, ended in a traceback."""
    monkeypatch.setattr("autogpt_local_executor.audit.load_audit_key", lambda: KEY)
    log = tmp_path / "audit.log"
    log.write_text('{"seq": 1, "ts": 1.0, "op": "X", "hmac": "00"}\n', encoding="utf-8")
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
    writer.force_rotate()
    await writer.write("EXECUTE_COMMAND", request_id="b", details={}, result={})
    held: list[bool] = []
    printed: list[bool] = []
    real_read = cli._read_audit_file
    real_print = builtins.print

    def read(path: Path) -> bytes | None:
        held.append(_lock_is_held(tmp_path / ".audit.log.lock"))
        return real_read(path)

    def print_(*args, **kwargs) -> None:
        printed.append(_lock_is_held(tmp_path / ".audit.log.lock"))
        real_print(*args, **kwargs)

    monkeypatch.setattr(cli, "_read_audit_file", read)
    monkeypatch.setattr(builtins, "print", print_)

    assert cli._audit_show(writer.path, 1) == 0
    assert held == [True, True]
    # A slow terminal or pager held the daemon's audit writes too.
    assert printed and not any(printed)


@pytest.mark.asyncio
async def test_audit_show_skips_a_rotated_log_pruned_while_it_runs(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    writer = _writer(tmp_path)
    await writer.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    writer.force_rotate()
    await writer.write("EXECUTE_COMMAND", request_id="b", details={}, result={})
    [rotated] = list_rotated_files(writer.path)
    real_lock = audit_module.holding_log_lock
    holds = 0

    @contextlib.contextmanager
    def prune_after_the_first_hold(log: Path):
        nonlocal holds
        holds += 1
        if holds == 2:
            rotated.unlink()
        with real_lock(log):
            yield

    monkeypatch.setattr(audit_module, "holding_log_lock", prune_after_the_first_hold)

    assert cli._audit_show(writer.path, 1) == 0
    out = capsys.readouterr().out
    assert f"# from {writer.path.name}" in out
    assert rotated.name not in out


@pytest.mark.asyncio
async def test_a_non_ascii_hmac_is_a_mismatch_not_a_crash(tmp_path: Path) -> None:
    """hmac.compare_digest raises TypeError for a non-ASCII str, so one
    damaged record stopped verify and verify-all with a traceback."""
    w = _writer(tmp_path)
    await w.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    record = json.loads(w.path.read_text(encoding="utf-8"))
    record["hmac"] = "\u00e9" * 64
    w.path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    violations = AuditWriter.verify(w.path, KEY)

    assert [v.kind for v in violations] == ["hmac_mismatch"]


def test_only_files_named_like_rotated_logs_are_rotated_logs(tmp_path: Path) -> None:
    """Any file named audit.log.* counted, so export packed a stray
    audit.log.private-notes into the archive, and prune could delete it.
    Collision suffixes also sorted as text, putting -10 before -2."""
    current = tmp_path / "audit.log"
    current.write_text("", encoding="utf-8")
    names = [
        "audit.log.private-notes",
        "audit.log.bak",
        "audit.log.20261104-0600",
        "audit.log.20261104-060000-10",
        "audit.log.20261104-060000",
        "audit.log.20261104-060000-2",
        "audit.log.20261103-235959",
    ]
    for name in names:
        (tmp_path / name).write_text("", encoding="utf-8")

    assert [p.name for p in list_rotated_files(current)] == [
        "audit.log.20261103-235959",
        "audit.log.20261104-060000",
        "audit.log.20261104-060000-2",
        "audit.log.20261104-060000-10",
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_a_link_or_folder_named_like_a_rotated_log_is_not_one(tmp_path: Path) -> None:
    """A symlink with a rotated log's name was listed, so export packed the
    bytes of whatever it pointed at, under the link's name."""
    logs = tmp_path / "logs"
    logs.mkdir()
    current = logs / "audit.log"
    current.write_text("", encoding="utf-8")
    elsewhere = tmp_path / "private.txt"
    elsewhere.write_text("not an audit log", encoding="utf-8")
    (logs / "audit.log.20261104-060000").symlink_to(elsewhere)
    (logs / "audit.log.20261104-060001").mkdir()
    (logs / "audit.log.20261104-060002").write_text("", encoding="utf-8")

    assert [p.name for p in list_rotated_files(current)] == ["audit.log.20261104-060002"]


def test_prune_keeps_a_file_holding_a_record_newer_than_its_last(tmp_path: Path) -> None:
    """A clock that went back while the file was written leaves its last
    record older than one before it. Prune judged the file by its last
    record, so it deleted the newer record with it."""
    import time

    current = tmp_path / "audit.log"
    current.write_text("", encoding="utf-8")
    rotated = tmp_path / "audit.log.20200101-000000"
    records = [{"seq": 1, "ts": time.time()}, {"seq": 2, "ts": 1.0}]
    rotated.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    assert cli._audit_prune(current, "1d") == 0

    assert rotated.exists()


@pytest.mark.parametrize(
    "line",
    [
        # math.isfinite raises OverflowError for an int past a float's range.
        '{"seq": 1, "ts": ' + "9" * 400 + "}",
        # json.loads raises RecursionError for a line nested this deep.
        "[" * 100_000 + "]" * 100_000,
    ],
)
def test_prune_keeps_a_log_it_cannot_judge_without_a_traceback(tmp_path: Path, line: str) -> None:
    current = tmp_path / "audit.log"
    current.write_text("", encoding="utf-8")
    rotated = tmp_path / "audit.log.20200101-000000"
    rotated.write_text(line + "\n", encoding="utf-8")

    assert cli._audit_prune(current, "1d") == 0

    assert rotated.exists()


@_posix_only
@pytest.mark.asyncio
async def test_an_existing_log_is_made_owner_only(tmp_path: Path) -> None:
    """Only a log the writer created got 0600, so one an older shim made, or
    one someone loosened, stayed readable by other accounts as it grew."""
    w = _writer(tmp_path)
    await w.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    w.path.chmod(0o644)

    AuditWriter(path=w.path, audit_key=KEY)

    assert stat.S_IMODE(w.path.stat().st_mode) == 0o600


@_posix_only
@pytest.mark.asyncio
async def test_a_log_that_cannot_be_made_owner_only_is_warned_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A chmod that failed (another account's file, a filesystem without
    POSIX modes) was logged at debug level, and the shim went on appending
    to a log other accounts may read."""
    w = _writer(tmp_path)
    await w.write("EXECUTE_COMMAND", request_id="a", details={}, result={})
    real_chmod = os.chmod

    def refuse(path, mode, *args, **kwargs):
        if Path(path) == w.path:
            raise PermissionError(errno.EPERM, "Operation not permitted")
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(audit_module.os, "chmod", refuse)

    with caplog.at_level("WARNING", logger=audit_module.__name__):
        AuditWriter(path=w.path, audit_key=KEY)

    assert any(str(w.path) in r.getMessage() for r in caplog.records if r.levelname == "WARNING")


def test_prune_refuses_a_duration_too_long_to_convert(tmp_path: Path, capsys) -> None:
    """int() refuses a string of more than 4300 digits, and the ValueError
    ended `audit prune` with a traceback instead of the duration message."""
    current = tmp_path / "audit.log"
    current.write_text("", encoding="utf-8")

    assert cli._audit_prune(current, "9" * 5000 + "d") == 2
    assert "Could not parse duration" in capsys.readouterr().out


def test_a_file_key_made_while_waiting_for_the_lock_is_the_one_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start that found no keychain key and no file key waited for the
    lock, then looked again only in the keychain. A start without a keychain
    that made the file key meanwhile, and signed the log with it, left this
    one storing a second key the log would never verify with."""
    import secrets

    import keyring

    from autogpt_local_executor import audit as audit_module

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    other_key = secrets.token_bytes(32)
    real_lock = audit_module._holding_key_lock

    @contextlib.contextmanager
    def after_the_other_start():
        with real_lock():
            salt = secrets.token_bytes(16)
            ciphertext = audit_module._xor_encrypt("passphrase", salt, other_key)
            audit_module._fallback_path().write_text(
                f"{salt.hex()}:{ciphertext.hex()}", encoding="utf-8"
            )
            yield

    monkeypatch.setattr(audit_module, "_holding_key_lock", after_the_other_start)
    stored: list[str] = []
    with (
        patch.object(keyring, "get_password", return_value=None),
        patch.object(keyring, "set_password", side_effect=lambda *args: stored.append(args[2])),
    ):
        key = get_or_create_audit_key(tmp_path / "audit.log")

    assert key == other_key
    assert stored == []


@pytest.mark.parametrize("when", ["before", "while-waiting"])
def test_a_file_key_that_cant_be_read_is_not_replaced_by_a_keychain_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, when: str
) -> None:
    """The file fallback's key signed a deployment's log while there was no
    keychain. Another deployment, starting with a new log and without the
    passphrase, skipped the file key it couldn't read and stored a new key
    in the keychain, which every start then finds first: the first
    deployment's log stopped verifying, and that start stopped for good."""
    import secrets

    import keyring

    from autogpt_local_executor.audit import AuditKeyError

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.delenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", raising=False)

    def make_file_key() -> None:
        salt = secrets.token_bytes(16)
        ciphertext = audit_module._xor_encrypt("passphrase", salt, secrets.token_bytes(32))
        audit_module._fallback_path().write_text(
            f"{salt.hex()}:{ciphertext.hex()}", encoding="utf-8"
        )

    if when == "before":
        make_file_key()
    else:
        real_lock = audit_module._holding_key_lock

        @contextlib.contextmanager
        def after_a_start_without_a_keychain():
            with real_lock():
                make_file_key()
                yield

        monkeypatch.setattr(audit_module, "_holding_key_lock", after_a_start_without_a_keychain)
    stored: list[str] = []
    with (
        patch.object(keyring, "get_password", return_value=None),
        patch.object(keyring, "set_password", side_effect=lambda *args: stored.append(args[2])),
    ):
        with pytest.raises(AuditKeyError, match="AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE"):
            get_or_create_audit_key(tmp_path / "another-deployment" / "audit.log")

    assert stored == []


@pytest.mark.parametrize("error", ["NoKeyringError", "InitError"])
def test_a_keychain_key_stored_while_waiting_for_the_lock_is_the_one_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    """A start whose keychain wasn't up yet waited for the lock, then made a
    file key without looking in the keychain again. A start that stored the
    keychain's key meanwhile left two keys: this start signed its log with
    the file key, and its next start, finding the keychain's key first,
    failed that log's verification and stopped for good."""
    import secrets

    import keyring
    import keyring.errors

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    other_key = secrets.token_bytes(32)
    keychain: dict[tuple[str, str], str] = {}
    real_lock = audit_module._holding_key_lock

    def read(service: str, user: str) -> str | None:
        if not keychain:
            # A Secret Service still starting, say: InitError, or no backend.
            raise getattr(keyring.errors, error)("not up yet")
        return keychain.get((service, user))

    @contextlib.contextmanager
    def after_the_other_start():
        with real_lock():
            keychain[(audit_module.KEYCHAIN_SERVICE, audit_module.KEYCHAIN_AUDIT_KEY_USERNAME)] = (
                other_key.hex()
            )
            yield

    monkeypatch.setattr(audit_module, "_holding_key_lock", after_the_other_start)
    with patch.object(keyring, "get_password", side_effect=read):
        key = get_or_create_audit_key(tmp_path / "audit.log")

    assert key == other_key
    assert not (tmp_path / "audit-key.enc").exists()


def test_a_keychain_that_comes_up_locked_while_waiting_makes_no_file_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A keychain that came up while this start waited for the lock, but
    locked, may hold a key another start stored: a file key made then would
    be a second one. The start waits for the keychain instead, as it does
    for one locked at its first look."""
    import keyring
    import keyring.errors

    from autogpt_local_executor.audit import AuditKeyUnavailableError

    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(tmp_path))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    reads = 0

    def read(_service: str, _user: str) -> str | None:
        nonlocal reads
        reads += 1
        if reads == 1:
            raise keyring.errors.InitError("not up yet")
        raise keyring.errors.KeyringLocked("the keychain is locked")

    with patch.object(keyring, "get_password", side_effect=read):
        with pytest.raises(AuditKeyUnavailableError, match="locked"):
            get_or_create_audit_key(tmp_path / "audit.log")

    assert reads == 2
    assert not (tmp_path / "audit-key.enc").exists()


def _keyring_found_no_keychain(monkeypatch: pytest.MonkeyPatch) -> None:
    """keyring as a start leaves it that found no keychain: on its failure
    backend, which it keeps for the rest of the process."""
    import keyring.core
    from keyring.backends import fail

    monkeypatch.setattr(keyring.core, "_keyring_backend", fail.Keyring())


def test_a_keychain_only_a_new_process_can_see_gets_no_file_key_beside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """keyring picks its backend once per process. A start that found no
    keychain kept the failure backend, so its second look under the lock
    could never see a Secret Service that came up meanwhile, holding the key
    another start stored: it made a file key, a second key, and signed the
    log with it, which the next start, finding the keychain's key first,
    refused. A new process looks instead, and a keychain it finds sends the
    start to be started again with it (exit 75)."""
    import textwrap

    from autogpt_local_executor.audit import AuditKeyUnavailableError
    from autogpt_local_executor.config import ShimConfig
    from autogpt_local_executor.daemon import DaemonPreflightError, ShimDaemon

    fallback_dir = tmp_path / "fallback"
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(fallback_dir))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    _keyring_found_no_keychain(monkeypatch)
    # What a new process finds: a keychain holding the other start's key.
    backends = tmp_path / "backends"
    backends.mkdir()
    (backends / "keychain_now_up.py").write_text(
        textwrap.dedent(
            """
            from keyring.backend import KeyringBackend


            class Keyring(KeyringBackend):
                priority = 1

                def get_password(self, service, username):
                    return "6f" * 32

                def set_password(self, service, username, password):
                    raise NotImplementedError
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(backends))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keychain_now_up.Keyring")

    with pytest.raises(AuditKeyUnavailableError, match="started again"):
        get_or_create_audit_key(tmp_path / "audit.log")
    config = ShimConfig(
        session_id="session-1", allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log"
    )
    with pytest.raises(DaemonPreflightError) as raised:
        ShimDaemon._build_audit_writer(config)

    assert raised.value.transient is True
    assert not (fallback_dir / "audit-key.enc").exists()
    assert not (tmp_path / "audit.log").exists()


def test_a_home_with_no_keychain_for_a_new_process_either_gets_the_file_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file fallback is for a home with no keychain at all: a new
    process that finds none either lets the start make the file key, which
    the next start, with the log it signed, reads."""
    fallback_dir = tmp_path / "fallback"
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(fallback_dir))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    _keyring_found_no_keychain(monkeypatch)
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    log = tmp_path / "audit.log"

    key = get_or_create_audit_key(log)
    log.write_text("", encoding="utf-8")

    assert (fallback_dir / "audit-key.enc").is_file()
    assert get_or_create_audit_key(log) == key


@pytest.mark.parametrize("outcome", ["locked", "crashed", "timed out", "unrunnable"])
def test_a_new_process_that_cant_say_whether_a_keychain_is_up_makes_no_file_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """A new process that couldn't finish its look (a keychain that is up
    but waits to be unlocked, say) may have had a keychain to tell of: the
    start waits, as it does for a keychain locked at its own first look.
    Only the look's own answer says there is none, not an exit code: 3 is
    also what Windows gives a process that aborted."""
    import subprocess

    from autogpt_local_executor.audit import AuditKeyUnavailableError

    fallback_dir = tmp_path / "fallback"
    monkeypatch.setenv("AUTOGPT_SHIM_FALLBACK_DIR", str(fallback_dir))
    monkeypatch.setenv("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE", "passphrase")
    _keyring_found_no_keychain(monkeypatch)
    looks: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        looks.append(argv)
        if outcome == "timed out":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if outcome == "unrunnable":
            raise OSError(errno.ENOENT, "No such file or directory", argv[0])
        if outcome == "crashed":
            return subprocess.CompletedProcess(argv, 3, b"", b"")
        return subprocess.CompletedProcess(argv, 1, b"", b"keyring.errors.KeyringLocked")

    monkeypatch.setattr(subprocess, "run", run)

    with pytest.raises(AuditKeyUnavailableError):
        get_or_create_audit_key(tmp_path / "audit.log")

    assert len(looks) == 1
    assert looks[0][0] == sys.executable
    assert not (fallback_dir / "audit-key.enc").exists()
