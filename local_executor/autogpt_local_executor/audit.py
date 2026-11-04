"""
Audit log — JSONL records chained by HMAC-SHA256 over canonical JSON (RFC 8785
with Python's float spelling; see docs/AUDIT_LOG.md).

Format and verification algorithm live in docs/AUDIT_LOG.md. This module is a
faithful implementation; if you change behavior here, update the doc first.

Design notes:

- One record per write. Each `write()` rebuilds the JCS canonical form of the
  record (sans `hmac`), computes the HMAC, appends the full JSON line, fsyncs.
- The chain is keyed by a per-machine 32-byte secret stored in the OS keychain.
  A legacy audit-only fallback exists; OAuth tokens always require keyring.
- Rotation: 64 MiB hard cap or 30 days since the first record, whichever comes
  first. Rotated files keep their tail (`audit.log.{YYYYMMDD-HHMMSS}`); the
  new file starts at `seq=1` with `prev_hmac=null`.

The verification entry point (`AuditWriter.verify`) is intentionally
synchronous and dependency-free so an offline operator can run it without
spinning up the full shim.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import datetime as _dt
import errno
import hmac
import json
import logging
import os
import secrets
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# Rotation thresholds — match docs/AUDIT_LOG.md "Rotation" table.
ROTATE_MAX_BYTES = 64 * 1024 * 1024
ROTATE_MAX_AGE_SECONDS = 30 * 24 * 60 * 60

# Keychain/storage identifiers share the service id with the OAuth store.
KEYCHAIN_SERVICE = "autogpt-local-executor"
KEYCHAIN_AUDIT_KEY_USERNAME = "audit_key"


# ── JCS canonicalization ─────────────────────────────────────────────────────


def _canonical_number(n: int | float) -> str:
    """Number text for the canonical form: RFC 8785 for ints and bools,
    Python's `repr` for floats.

    `repr` is the shortest round-trippable form, but an integer-valued float
    keeps its `.0` (`1712345678.0`, where RFC 8785 writes `1712345678`) and
    exponents use Python's spelling. Existing chains are HMACed over this
    text, so it can only change with a versioned log format (AUDIT_LOG.md).
    """
    if isinstance(n, bool):  # bool is an int subclass; handle before int
        return "true" if n else "false"
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        if n != n:  # NaN
            raise ValueError("NaN cannot be JCS-canonicalized")
        if n in (float("inf"), float("-inf")):
            raise ValueError("Infinity cannot be JCS-canonicalized")
        # Python's repr already gives shortest round-trippable representation.
        return repr(n)
    raise TypeError(f"Not a number: {type(n)!r}")


def _canonicalize(value: Any) -> str:
    """Hand-rolled RFC 8785 JCS subset: sorted keys, no whitespace, ascii-safe.

    We avoid the optional `rfc8785` dependency to keep the wheel small. The
    inputs we canonicalize are strictly the JSON shapes documented in
    AUDIT_LOG.md (str/int/float/bool/None/list/dict), so the JCS subset is
    sufficient. If we ever need to canonicalize arbitrary user-supplied JSON
    (e.g., for inbound verification of foreign logs) we should swap to the
    `rfc8785` package.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _canonical_number(value)
    if isinstance(value, str):
        # json.dumps with ensure_ascii=False, separators set, gives a JCS-
        # compatible string encoding (RFC 8785 §3.2.2 references RFC 8259).
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ",".join(_canonicalize(v) for v in value) + "]"
    if isinstance(value, dict):
        # Keys are sorted by codepoint (RFC 8785 §3.2.3 — UTF-16 ordering;
        # for the BMP subset we actually use, codepoint sort is equivalent).
        items = sorted(value.items(), key=lambda kv: kv[0])
        return (
            "{"
            + ",".join(f"{json.dumps(k, ensure_ascii=False)}:{_canonicalize(v)}" for k, v in items)
            + "}"
        )
    raise TypeError(f"Cannot canonicalize {type(value)!r}")


def canonical_bytes(record: dict[str, Any]) -> bytes:
    """Public entry point — returns the JCS bytes used for HMAC input."""
    return _canonicalize(record).encode("utf-8")


# ── Verification result types ────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Violation:
    """One verification failure. `seq` is the (claimed) seq of the bad record,
    or None when the failure happens before we could parse any record."""

    kind: str  # one of: "parse_error", "hmac_mismatch", "chain_break",
    #                   "sequence_gap", "truncated", "missing_field",
    #                   "duplicate_seq"
    seq: int | None
    message: str


# ── Audit writer ─────────────────────────────────────────────────────────────


class AuditWriter:
    """Append-only JSONL writer for the audit log.

    Use `await writer.write(...)` from anywhere in the daemon's event loop.
    For shim-internal events prefer the typed convenience methods (e.g.,
    `await writer.shim_start(machine_id)`).
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        audit_key: bytes,
        *,
        shim_version: str | None = None,
        session_id: str | None = None,
        machine_id: str | None = None,
    ) -> None:
        if not audit_key or len(audit_key) < 16:
            raise ValueError("audit_key must be at least 16 bytes")
        self.path = Path(path)
        self.audit_key = audit_key
        self._shim_version = shim_version or _detect_shim_version()
        self._default_session_id = session_id
        self._default_machine_id = machine_id
        self._lock = asyncio.Lock()
        # The daemon and `autogpt-shim audit` commands each have a writer for
        # one log, so every read of the chain state, append and rotation also
        # holds this lock file. Its name keeps it out of list_rotated_files.
        self._file_lock_path = _lock_path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _file_lock(self._file_lock_path):
            if self.path.exists():
                try:
                    violations = self.verify(self.path, audit_key)
                except Exception as exc:
                    raise RuntimeError(
                        "Existing audit log cannot be verified; refusing to append"
                    ) from exc
                if violations:
                    raise RuntimeError("Existing audit log failed verification; refusing to append")
            self._seq, self._prev_hmac = self._load_tail_state(self.path)
            # 0600 on POSIX; on Windows the file inherits the user-only ACL.
            if not self.path.exists():
                # Create empty so first write can fsync against an existing inode
                # and we can set 0600 once instead of after every append.
                _create_owner_only(self.path)
            self._file_state = _file_identity(self.path)

    # ── Defaults wiring ──────────────────────────────────────────────────

    def set_session_id(self, session_id: str | None) -> None:
        """Updated after HELLO_ACK assigns a session."""
        self._default_session_id = session_id

    def set_machine_id(self, machine_id: str | None) -> None:
        self._default_machine_id = machine_id

    # ── Core write ───────────────────────────────────────────────────────

    async def write(
        self,
        op: str,
        *,
        session_id: str | None = None,
        machine_id: str | None = None,
        request_id: str | None,
        details: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> None:
        async with self._lock:
            await asyncio.to_thread(
                self._write_sync,
                op,
                session_id if session_id is not None else (self._default_session_id or "-"),
                machine_id if machine_id is not None else (self._default_machine_id or "-"),
                request_id,
                details,
                result or {"ok": True, "exit_code": None, "duration_ms": 0, "error_code": None},
            )

    def _write_sync(
        self,
        op: str,
        session_id: str,
        machine_id: str,
        request_id: str | None,
        details: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        with _file_lock(self._file_lock_path):
            # Another writer may have rotated or appended since this one last
            # wrote; chain onto what the file holds now.
            if _file_identity(self.path) != self._file_state:
                self._seq, self._prev_hmac = self._load_tail_state(self.path)
            # Rotate before writing if we'd cross a threshold. Rotation resets
            # the in-memory chain state.
            self._maybe_rotate()
            self._append_record(op, session_id, machine_id, request_id, details, result)
            self._file_state = _file_identity(self.path)

    def _append_record(
        self,
        op: str,
        session_id: str,
        machine_id: str,
        request_id: str | None,
        details: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        record = {
            "ts": _now_ts(),
            "session_id": session_id,
            "machine_id": machine_id,
            "shim_version": self._shim_version,
            "request_id": request_id,
            "op": op,
            "details": details,
            "result": _normalize_result(result),
            "seq": self._seq,
            "prev_hmac": self._prev_hmac,
        }
        hmac_hex = hmac.new(self.audit_key, canonical_bytes(record), sha256).hexdigest()
        record["hmac"] = hmac_hex
        self._append(json.dumps(record, ensure_ascii=False) + "\n")
        self._prev_hmac = hmac_hex
        self._seq += 1

    def _append(self, line: str) -> None:
        # Open-append-fsync per record. Slow for high throughput but the
        # audit log is one-line-per-shim-event, not a hot path. A log removed
        # since the last write is recreated owner-only, like the first one.
        flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_BINARY", 0)
        with open(os.open(self.path, flags, 0o600), "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                # Best-effort: some filesystems (procfs, etc.) reject fsync.
                logger.debug("fsync failed on audit log", exc_info=True)

    # ── Rotation ────────────────────────────────────────────────────────

    def _maybe_rotate(self) -> None:
        try:
            stat = self.path.stat()
        except OSError:
            return
        reason = None
        if stat.st_size >= ROTATE_MAX_BYTES:
            reason = "size"
        elif stat.st_size > 0:
            first_ts = _read_first_ts(self.path)
            if first_ts is not None and (_now_ts() - first_ts) > ROTATE_MAX_AGE_SECONDS:
                reason = "age"
        if reason is None:
            return
        try:
            self._rotate(reason=reason)
        except OSError as exc:
            # Keep appending to the current file; the next write retries.
            logger.warning("Audit rotation failed (%s): %s", reason, exc)

    def force_rotate(self) -> Path | None:
        """Caller-initiated rotation. Returns the rotated-file path, or None
        if the current file was empty (nothing to rotate). Raises OSError if
        the file could not be rotated."""
        with _file_lock(self._file_lock_path):
            with open(self.path, "rb") as f:
                f.seek(0, os.SEEK_END)
                if f.tell() == 0:
                    return None
            return self._rotate(reason="explicit")

    def _rotate(self, *, reason: str) -> Path:
        # If a file with the same timestamp suffix already exists (sub-second
        # double-rotation), tack on a `-N` counter so we never clobber a
        # rotated chain.
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        target = self.path.with_name(f"{self.path.name}.{stamp}")
        counter = 1
        while target.exists():
            target = self.path.with_name(f"{self.path.name}.{stamp}-{counter}")
            counter += 1
        os.replace(self.path, target)
        # The old chain now lives in `target`; whatever is written next starts
        # a new one, even if creating the empty file below fails.
        self._seq = 1
        self._prev_hmac = None
        _create_owner_only(self.path)
        self._file_state = _file_identity(self.path)
        logger.info("Rotated audit log (%s) → %s", reason, target.name)
        return target

    # ── Tail-state recovery ─────────────────────────────────────────────

    @staticmethod
    def _load_tail_state(path: Path) -> tuple[int, str | None]:
        """Read the last full line and return (next_seq, prev_hmac)."""
        if not path.is_file() or path.stat().st_size == 0:
            return 1, None
        try:
            last_line = b""
            with open(path, "rb") as file:
                for line in file:
                    if line.strip():
                        last_line = line
            if not last_line:
                return 1, None
            last = json.loads(last_line.decode("utf-8"))
            if not isinstance(last, dict):
                raise ValueError("the last record is not a JSON object")
            return int(last.get("seq", 0)) + 1, last.get("hmac")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("Existing audit log tail cannot be recovered") from exc

    # ── Shim-internal convenience methods ────────────────────────────────

    async def shim_start(self, machine_id: str) -> None:
        await self.write(
            "SHIM_START",
            machine_id=machine_id,
            request_id=None,
            details={"shim_version": self._shim_version},
            result={"ok": True, "exit_code": None, "duration_ms": 0, "error_code": None},
        )

    async def shim_stop(self, reason: str = "graceful") -> None:
        await self.write(
            "SHIM_STOP",
            request_id=None,
            details={"reason": reason},
        )

    async def ws_connected(self, url: str) -> None:
        await self.write(
            "WS_CONNECTED",
            request_id=None,
            details={"url": url},
        )

    async def ws_disconnected(self, reason: str) -> None:
        await self.write(
            "WS_DISCONNECTED",
            request_id=None,
            details={"reason": reason},
        )

    async def token_refreshed(self) -> None:
        await self.write(
            "TOKEN_REFRESHED",
            request_id=None,
            details={},
        )

    async def config_reloaded(self, granted_capabilities: list[str]) -> None:
        await self.write(
            "CONFIG_RELOADED",
            request_id=None,
            details={"granted_capabilities": granted_capabilities},
        )

    async def jail_violation(self, code: str, path: str, op: str | None = None) -> None:
        await self.write(
            "JAIL_VIOLATION",
            request_id=None,
            details={"code": code, "path": path, "attempted_op": op},
            result={
                "ok": False,
                "exit_code": None,
                "duration_ms": 0,
                "error_code": code,
            },
        )

    # ── Verification ────────────────────────────────────────────────────

    @classmethod
    def verify(
        cls,
        path: str | os.PathLike[str],
        audit_key: bytes,
    ) -> list[Violation]:
        """Chain-verify one audit log file. Returns [] if clean."""
        p = Path(path)
        if not p.is_file():
            return [Violation("truncated", None, f"audit log not found: {p}")]
        violations: list[Violation] = []
        prev_hmac: str | None = None
        expected_seq = 1
        with open(p, "rb") as f:
            data = f.read()
        if not data:
            return []
        # Detect mid-line truncation: file must end with a newline since we
        # always write `json + "\n"`.
        if not data.endswith(b"\n"):
            violations.append(
                Violation(
                    "truncated",
                    None,
                    "audit log does not end with newline — last record truncated",
                )
            )
        for raw in data.splitlines():
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                violations.append(Violation("parse_error", None, f"unparseable line: {exc}"))
                # Can't continue the chain past an unparseable record.
                return violations
            if not isinstance(rec, dict):
                violations.append(Violation("parse_error", None, "record is not a JSON object"))
                return violations
            seq = rec.get("seq")
            if not isinstance(seq, int):
                violations.append(Violation("missing_field", None, "record missing seq"))
                return violations
            if seq != expected_seq:
                violations.append(
                    Violation(
                        "sequence_gap",
                        seq,
                        f"expected seq={expected_seq}, got seq={seq}",
                    )
                )
            record_prev = rec.get("prev_hmac")
            if record_prev != prev_hmac:
                violations.append(
                    Violation(
                        "chain_break",
                        seq,
                        f"prev_hmac mismatch at seq={seq}",
                    )
                )
            stored_hmac = rec.get("hmac")
            if not isinstance(stored_hmac, str):
                violations.append(Violation("missing_field", seq, "record missing hmac"))
                return violations
            shadow = {k: v for k, v in rec.items() if k != "hmac"}
            try:
                canonical = canonical_bytes(shadow)
            except ValueError as exc:
                # json.loads takes NaN and Infinity; no record the writer
                # produced can hold one, and canonicalizing them raises.
                violations.append(Violation("parse_error", seq, f"uncanonicalizable record: {exc}"))
                return violations
            actual = hmac.new(audit_key, canonical, sha256).hexdigest()
            if not hmac.compare_digest(actual, stored_hmac):
                violations.append(
                    Violation(
                        "hmac_mismatch",
                        seq,
                        f"hmac mismatch at seq={seq}",
                    )
                )
            prev_hmac = stored_hmac
            expected_seq = seq + 1
        return violations


# ── Audit key storage ────────────────────────────────────────────────────────


class AuditKeyError(RuntimeError):
    pass


def get_or_create_audit_key() -> bytes:
    """Return the persisted audit key, generating + storing one on first use.

    Prefers the OS keychain. The legacy audit-only fallback requires
    AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE; OAuth tokens have no file fallback.
    """
    # Try keyring first.
    try:
        import keyring

        stored = keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_AUDIT_KEY_USERNAME)
        if stored is not None:
            try:
                key = bytes.fromhex(stored)
            except ValueError:
                raise AuditKeyError(
                    "Stored audit key is malformed; refusing to replace it"
                ) from None
            if len(key) != 32:
                raise AuditKeyError(
                    "Stored audit key has an invalid length; refusing to replace it"
                )
            return key
        key = secrets.token_bytes(32)
        keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_AUDIT_KEY_USERNAME, key.hex())
        return key
    except AuditKeyError:
        raise
    except Exception as exc:
        logger.warning("Keychain unavailable for audit key (%s); using fallback.", exc)
        return _fallback_audit_key()


def _fallback_audit_key() -> bytes:
    """Legacy audit-only obfuscation; the existing log authenticates the key.

    This format has no authentication tag or slow password KDF. OAuth tokens
    require an OS keychain, so this is not a headless token-storage solution.
    """
    fallback_path = _fallback_path()
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    passphrase = os.environ.get("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE")
    if not passphrase:
        raise RuntimeError(
            "No keychain available and AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE is not set. "
            "Either install a keyring backend or set the env var to enable the "
            "encrypted-file fallback."
        )
    if fallback_path.exists():
        try:
            blob = fallback_path.read_text(encoding="utf-8").strip()
            salt_hex, ct_hex = blob.split(":", 1)
            salt = bytes.fromhex(salt_hex)
            ciphertext = bytes.fromhex(ct_hex)
            if len(salt) != 16 or len(ciphertext) != 32:
                raise ValueError("Invalid audit key envelope length")
            return _xor_decrypt(passphrase, salt, ciphertext)
        except (OSError, ValueError) as exc:
            raise AuditKeyError(
                "Stored audit key fallback is malformed; refusing to replace it"
            ) from exc
    key = secrets.token_bytes(32)
    salt = secrets.token_bytes(16)
    ct = _xor_encrypt(passphrase, salt, key)
    fallback_path.write_text(f"{salt.hex()}:{ct.hex()}", encoding="utf-8")
    _chmod_owner_only(fallback_path)
    return key


def _fallback_path() -> Path:
    base = os.environ.get(
        "AUTOGPT_SHIM_FALLBACK_DIR",
        str(Path.home() / ".autogpt-local-executor"),
    )
    return Path(base) / "audit-key.enc"


def _kdf(passphrase: str, salt: bytes, length: int) -> bytes:
    """Legacy obfuscation stream, without slow password derivation."""
    out = b""
    counter = 0
    while len(out) < length:
        counter += 1
        out += hmac.new(
            passphrase.encode("utf-8"),
            salt + counter.to_bytes(4, "big"),
            sha256,
        ).digest()
    return out[:length]


def _xor_encrypt(passphrase: str, salt: bytes, plaintext: bytes) -> bytes:
    stream = _kdf(passphrase, salt, len(plaintext))
    return bytes(p ^ s for p, s in zip(plaintext, stream))


def _xor_decrypt(passphrase: str, salt: bytes, ciphertext: bytes) -> bytes:
    return _xor_encrypt(passphrase, salt, ciphertext)


class SessionAuditWriter:
    """Bind every write to one session while sharing a single HMAC chain."""

    def __init__(self, writer: AuditWriter, session_id: str) -> None:
        if not session_id:
            raise ValueError("session_id must be nonempty")
        self.writer = writer
        self.session_id = session_id

    @property
    def audit_key(self) -> bytes:
        return self.writer.audit_key

    def set_session_id(self, session_id: str | None) -> None:
        if session_id != self.session_id:
            raise ValueError("A session-bound audit writer cannot change sessions")

    def set_machine_id(self, machine_id: str | None) -> None:
        if machine_id is not None:
            self.writer.set_machine_id(machine_id)

    async def write(
        self,
        op: str,
        *,
        session_id: str | None = None,
        machine_id: str | None = None,
        request_id: str | None,
        details: dict[str, Any],
        result: dict[str, Any] | None = None,
    ) -> None:
        if session_id is not None and session_id != self.session_id:
            raise ValueError("Audit write attempted to cross session boundary")
        await self.writer.write(
            op,
            session_id=self.session_id,
            machine_id=machine_id,
            request_id=request_id,
            details=details,
            result=result,
        )

    async def shim_start(self, machine_id: str) -> None:
        await self.write(
            "SHIM_START",
            machine_id=machine_id,
            request_id=None,
            details={"shim_version": self.writer._shim_version},
        )

    async def shim_stop(self, reason: str = "graceful") -> None:
        await self.write("SHIM_STOP", request_id=None, details={"reason": reason})

    async def ws_connected(self, url: str) -> None:
        await self.write("WS_CONNECTED", request_id=None, details={"url": url})

    async def ws_disconnected(self, reason: str) -> None:
        await self.write("WS_DISCONNECTED", request_id=None, details={"reason": reason})

    async def token_refreshed(self) -> None:
        await self.write("TOKEN_REFRESHED", request_id=None, details={})

    async def config_reloaded(self, granted_capabilities: list[str]) -> None:
        await self.write(
            "CONFIG_RELOADED",
            request_id=None,
            details={"granted_capabilities": granted_capabilities},
        )

    async def jail_violation(self, code: str, path: str, op: str | None = None) -> None:
        await self.write(
            "JAIL_VIOLATION",
            request_id=None,
            details={"code": code, "path": path, "attempted_op": op},
            result={
                "ok": False,
                "exit_code": None,
                "duration_ms": 0,
                "error_code": code,
            },
        )


# ── Helpers ──────────────────────────────────────────────────────────────────


def _detect_shim_version() -> str:
    try:
        from . import __version__  # late import to avoid cycle on cold start

        return __version__
    except Exception:
        return "0.0.0"


def _now_ts() -> float:
    # Plain wall clock; the spec calls for unix epoch seconds. Tests can
    # monkeypatch this if they need determinism.
    import time

    return time.time()


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Ensure the four result fields per AUDIT_LOG.md exist with the right
    types so JCS canonicalization is stable across writers."""
    out = {
        "ok": bool(result.get("ok", True)),
        "exit_code": result.get("exit_code") if result.get("exit_code") is not None else None,
        "duration_ms": int(result.get("duration_ms", 0)),
        "error_code": result.get("error_code") if result.get("error_code") is not None else None,
    }
    # Coerce exit_code to int when set; the spec types it as int|null.
    if out["exit_code"] is not None:
        try:
            out["exit_code"] = int(out["exit_code"])
        except (TypeError, ValueError):
            out["exit_code"] = None
    return out


def _read_first_ts(path: Path) -> float | None:
    try:
        with open(path, "rb") as f:
            line = f.readline()
        if not line.strip():
            return None
        rec = json.loads(line.decode("utf-8"))
        ts = rec.get("ts") if isinstance(rec, dict) else None
        return float(ts) if ts is not None else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _create_owner_only(path: Path) -> None:
    """Create an empty file at `path` that only its owner can read: 0600 from
    the start, so neither the umask nor a failed chmod can leave it open."""
    flags = os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    os.close(os.open(path, flags, 0o600))
    _chmod_owner_only(path)


def _chmod_owner_only(path: Path) -> None:
    """0600 on POSIX; no-op on Windows (the file inherits the user ACL)."""
    if os.name == "nt":
        return
    try:
        os.chmod(path, 0o600)
    except OSError:
        logger.debug("chmod 0600 failed for %s", path, exc_info=True)


def _file_identity(path: Path) -> tuple[int, int, int] | None:
    """Which file is at `path` and how long it is, to notice another writer."""
    try:
        info = path.stat()
    except OSError:
        return None
    return info.st_dev, info.st_ino, info.st_size


class AuditLockError(RuntimeError):
    """A reader could not take the lock the audit log's writers hold."""


def _lock_path(log: Path) -> Path:
    # The leading dot keeps it out of list_rotated_files.
    return log.with_name(f".{log.name}.lock")


@contextmanager
def holding_log_lock(log: Path) -> Iterator[None]:
    """Hold the lock that every AuditWriter for `log` appends and rotates
    under, so a reader never sees a record half written.

    A reader that can't open the lock file for writing (a read-only volume,
    another account's log) takes the same lock through a read-only handle.
    The read goes ahead unlocked only when the lock file doesn't exist: every
    writer creates it before appending, so then no writer holds it. Any other
    failure raises AuditLockError rather than risk a half-written record.
    """
    lock = _lock_path(log)
    unlocked = False
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(_file_lock(lock))
        except OSError:
            try:
                stack.enter_context(_file_lock(lock, read_only=True))
            except FileNotFoundError:
                unlocked = True
            except OSError as exc:
                raise AuditLockError(f"{lock}: {exc}") from exc
        yield
    # A writer that started meanwhile created the lock file first, and may
    # have been mid-append while this read ran, so the read can't be trusted.
    if unlocked and lock.exists():
        raise AuditLockError(f"{lock}: a writer started during the read; run it again")


@contextmanager
def _file_lock(path: Path, *, read_only: bool = False) -> Iterator[None]:
    """Hold an exclusive lock on `path`, across processes, while the block runs.

    `read_only` opens an existing lock file without creating or writing it;
    flock and msvcrt.locking both lock through a read-only handle."""
    with open(path, "rb" if read_only else "a+b") as handle:
        if sys.platform == "win32":
            import msvcrt

            # LK_LOCK gives up after ten seconds, so poll LK_NBLCK, which
            # fails with EACCES while another process holds the lock. Any other
            # error is permanent and raised, rather than retried forever.
            while True:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EDEADLK):
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def list_rotated_files(current: Path) -> list[Path]:
    """Return rotated audit files (oldest first) next to `current`."""
    if not current.parent.is_dir():
        return []
    prefix = current.name + "."
    matches = [p for p in current.parent.iterdir() if p.name.startswith(prefix)]
    return sorted(matches, key=lambda p: p.name)


__all__ = [
    "AuditLockError",
    "AuditWriter",
    "ROTATE_MAX_AGE_SECONDS",
    "ROTATE_MAX_BYTES",
    "Violation",
    "SessionAuditWriter",
    "canonical_bytes",
    "get_or_create_audit_key",
    "holding_log_lock",
    "list_rotated_files",
]
