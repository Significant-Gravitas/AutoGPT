"""Tests for the audit log writer + verifier (docs/AUDIT_LOG.md)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

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
