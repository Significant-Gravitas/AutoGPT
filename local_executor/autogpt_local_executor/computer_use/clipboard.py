"""
Clipboard sandbox primitives.

Implements docs/COMPUTER_USE.md Q3:
- Writeback registry: every CLIPBOARD_WRITE records (sha256, sequence,
  timestamp) so a subsequent CLIPBOARD_READ in writeback-only mode can
  confirm the contents are still ours and haven't aged past 30 s.
- ConcealedType / CF_PRIVATE marker detection lives here only as the
  shape used by callers; per-OS lookup is in the backends because each
  pasteboard API exposes it differently.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass

WRITEBACK_WINDOW_SECONDS = 30.0


@dataclass(frozen=True)
class WritebackRecord:
    sha256: str
    sequence: int | None
    written_at: float

    def age(self, now: float | None = None) -> float:
        return (now if now is not None else time.monotonic()) - self.written_at


def _hash_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class ClipboardWritebackRegistry:
    """Single-slot writeback cache.

    We only ever track *the most recent* write. If the platform writes
    A then B within 30 s, a read after that returns B (matches live
    clipboard) and a read after a third party overwrote B returns
    CLIPBOARD_CONCEALED with reason `writeback_overwritten`.
    """

    def __init__(self, window_seconds: float = WRITEBACK_WINDOW_SECONDS) -> None:
        self._window = window_seconds
        self._record: WritebackRecord | None = None
        self._lock = threading.Lock()

    @property
    def window_seconds(self) -> float:
        return self._window

    def record_write(self, content: str, sequence: int | None = None) -> WritebackRecord:
        record = WritebackRecord(
            sha256=_hash_text(content),
            sequence=sequence,
            written_at=time.monotonic(),
        )
        with self._lock:
            self._record = record
        return record

    def check(
        self,
        *,
        live_content: str | None,
        live_sequence: int | None,
        now: float | None = None,
    ) -> WritebackRecord | None:
        """Return the active writeback record if the live clipboard still
        matches what we wrote within the window; otherwise None.

        Callers that get back None should raise
        ClipboardConcealedError with `reason="writeback_only"` or
        `reason="writeback_overwritten"` depending on which check failed
        — they can interrogate `self.snapshot()` to tell which.
        """
        with self._lock:
            record = self._record
        if record is None:
            return None
        if record.age(now) > self._window:
            return None
        if live_content is None:
            return None
        if record.sha256 != _hash_text(live_content):
            return None
        if record.sequence is not None and live_sequence is not None:
            if record.sequence != live_sequence:
                return None
        return record

    def snapshot(self) -> WritebackRecord | None:
        with self._lock:
            return self._record

    def clear(self) -> None:
        with self._lock:
            self._record = None


__all__ = [
    "WRITEBACK_WINDOW_SECONDS",
    "ClipboardWritebackRegistry",
    "WritebackRecord",
]
