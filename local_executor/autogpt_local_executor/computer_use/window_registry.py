"""
WindowRegistry — UUID-to-native-handle map with re-verify-at-USE.

Per docs/COMPUTER_USE.md Q2: window_id is a shim-minted opaque UUID with
a `win_` prefix, never reused, wiped on every HELLO. The registry
records a fingerprint `(pid, class_name, creation_timestamp)` at LIST
time and checks it again at USE time. Mismatch → WindowStaleError; the
shim never silently re-binds.

The registry is OS-agnostic — backends supply their own
NativeWindowHandle subclass and `fingerprint()` helper.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WindowFingerprint:
    pid: int
    class_name: str | None
    creation_timestamp: float | None

    def matches(self, other: WindowFingerprint) -> bool:
        # All three fields must match. None on either side is considered a
        # "couldn't read" — we don't try to be clever and treat it as a match.
        return (
            self.pid == other.pid
            and self.class_name == other.class_name
            and self.creation_timestamp == other.creation_timestamp
        )


@dataclass
class WindowEntry:
    window_id: str
    native_handle: Any
    fingerprint: WindowFingerprint
    extra: dict[str, Any] = field(default_factory=dict)


class WindowRegistry:
    """Thread-safe registry of `win_<uuid>` → native window handle."""

    PREFIX = "win_"

    def __init__(self) -> None:
        self._entries: dict[str, WindowEntry] = {}
        self._lock = threading.Lock()

    def mint(
        self,
        native_handle: Any,
        fingerprint: WindowFingerprint,
        *,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Mint a fresh UUID for `native_handle` and store it."""
        wid = f"{self.PREFIX}{uuid.uuid4()}"
        with self._lock:
            self._entries[wid] = WindowEntry(
                window_id=wid,
                native_handle=native_handle,
                fingerprint=fingerprint,
                extra=dict(extra or {}),
            )
        return wid

    def lookup(self, window_id: str) -> WindowEntry | None:
        with self._lock:
            return self._entries.get(window_id)

    def verify(
        self,
        window_id: str,
        live_fingerprint_fn: Callable[[Any], WindowFingerprint | None],
    ) -> WindowEntry:
        """Look up `window_id`, then re-verify against the live OS.

        `live_fingerprint_fn(native_handle)` returns the *current*
        fingerprint of the OS window or `None` if it's gone. Raises
        WindowStaleError on any mismatch or absence; never falls back.
        """
        from .errors import WindowStaleError

        entry = self.lookup(window_id)
        if entry is None:
            raise WindowStaleError(window_id, message=f"unknown window_id {window_id!r}")
        live = live_fingerprint_fn(entry.native_handle)
        if live is None:
            self.drop(window_id)
            raise WindowStaleError(window_id)
        if not entry.fingerprint.matches(live):
            # Drop so we don't keep returning a stale handle on repeat calls.
            self.drop(window_id)
            raise WindowStaleError(window_id)
        return entry

    def drop(self, window_id: str) -> None:
        with self._lock:
            self._entries.pop(window_id, None)

    def clear(self) -> None:
        """Called by daemon on HELLO/reconnect, per Q2."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


__all__ = ["WindowEntry", "WindowFingerprint", "WindowRegistry"]
