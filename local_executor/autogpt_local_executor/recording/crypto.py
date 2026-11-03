"""
At-rest encryption for the recording buffer (§9 "At-rest").

The buffered recording is encrypted on disk under a key DISTINCT from the
audit-chain key — a different trust boundary (§9). Stored in the OS keychain
under its own username, with the same encrypted-file fallback the audit key
uses when no keyring is available.

The cipher itself is the stdlib-only HMAC-stream XOR used elsewhere in the shim
(see audit.py): no extra deps, salt-per-blob, fine against a casual filesystem
snoop. The interpretation route controls whether pixels leave the machine;
structured review data still crosses the shim boundary. This at-rest layer
keeps the retained buffer from sitting in plaintext between capture and review.
"""

from __future__ import annotations

import hmac
import logging
import os
import secrets
from hashlib import sha256
from pathlib import Path

logger = logging.getLogger(__name__)

# Share the service id with the OAuth/audit stores; a DISTINCT username so the
# recording key is a separate secret from the audit-chain key (§9).
KEYCHAIN_SERVICE = "autogpt-local-executor"
KEYCHAIN_RECORDING_KEY_USERNAME = "recording_key"


def _kdf(key: bytes, salt: bytes, length: int) -> bytes:
    """HMAC-SHA256 stream KDF (same construction as audit.py)."""
    out = b""
    counter = 0
    while len(out) < length:
        counter += 1
        out += hmac.new(key, salt + counter.to_bytes(4, "big"), sha256).digest()
    return out[:length]


class RecordingCipher:
    """Symmetric stream cipher over the recording-buffer key.

    `encrypt` prepends a fresh 16-byte salt; `decrypt` reads it back. Both are
    pure functions of (key, salt, data).
    """

    def __init__(self, key: bytes) -> None:
        if len(key) < 16:
            raise ValueError("recording key must be at least 16 bytes")
        self._key = key

    @classmethod
    def create(cls) -> RecordingCipher:
        """Build a cipher over the persisted recording key (minting on first use)."""
        return cls(get_or_create_recording_key())

    def encrypt(self, plaintext: bytes) -> bytes:
        salt = secrets.token_bytes(16)
        stream = _kdf(self._key, salt, len(plaintext))
        ct = bytes(p ^ s for p, s in zip(plaintext, stream))
        return salt + ct

    def decrypt(self, blob: bytes) -> bytes:
        if len(blob) < 16:
            raise ValueError("ciphertext too short")
        salt, ct = blob[:16], blob[16:]
        stream = _kdf(self._key, salt, len(ct))
        return bytes(c ^ s for c, s in zip(ct, stream))


# ── Key storage (distinct from the audit key) ─────────────────────────────────


def get_or_create_recording_key() -> bytes:
    """Return the persisted recording-buffer key, minting one on first use.

    Prefers the OS keychain under a DISTINCT username from the audit key (§9).
    Falls back to an encrypted file gated by AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE,
    symmetric with the audit-key fallback.
    """
    try:
        import keyring

        stored = keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_RECORDING_KEY_USERNAME)
        if stored:
            try:
                return bytes.fromhex(stored)
            except ValueError:
                logger.warning("Stored recording key in keychain is not hex; regenerating.")
        key = secrets.token_bytes(32)
        keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_RECORDING_KEY_USERNAME, key.hex())
        return key
    except Exception as exc:
        logger.warning("Keychain unavailable for recording key (%s); using fallback.", exc)
        return _fallback_recording_key()


def _fallback_recording_key() -> bytes:
    fallback_path = _fallback_path()
    fallback_path.parent.mkdir(parents=True, exist_ok=True)
    passphrase = os.environ.get("AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE")
    if not passphrase:
        raise RuntimeError(
            "No keychain available and AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE is not set. "
            "Either install a keyring backend or set the env var to enable the "
            "encrypted-file fallback for the recording key."
        )
    if fallback_path.is_file():
        try:
            blob = fallback_path.read_text(encoding="utf-8").strip()
            salt_hex, ct_hex = blob.split(":", 1)
            salt = bytes.fromhex(salt_hex)
            stream = _kdf(passphrase.encode("utf-8"), salt, 32)
            return bytes(c ^ s for c, s in zip(bytes.fromhex(ct_hex), stream))
        except (OSError, ValueError) as exc:
            logger.warning("Recording-key fallback unreadable, regenerating: %s", exc)
    key = secrets.token_bytes(32)
    salt = secrets.token_bytes(16)
    stream = _kdf(passphrase.encode("utf-8"), salt, 32)
    ct = bytes(p ^ s for p, s in zip(key, stream))
    fallback_path.write_text(f"{salt.hex()}:{ct.hex()}", encoding="utf-8")
    if os.name != "nt":
        try:
            os.chmod(fallback_path, 0o600)
        except OSError:
            pass
    return key


def _fallback_path() -> Path:
    base = os.environ.get(
        "AUTOGPT_SHIM_FALLBACK_DIR",
        str(Path.home() / ".autogpt-local-executor"),
    )
    return Path(base) / "recording-key.enc"


__all__ = [
    "KEYCHAIN_RECORDING_KEY_USERNAME",
    "RecordingCipher",
    "get_or_create_recording_key",
]
