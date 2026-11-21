"""Computer-use failures the shim reports as wire ERROR envelopes."""

from __future__ import annotations

from typing import Any

from ..protocol import ErrorCode


class ComputerUseError(Exception):
    """A refusal or failure with the wire error code it maps to."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


class ClipboardConcealedError(ComputerUseError):
    """The clipboard holds content this session may not read: text the shim
    didn't write (without --enable-clipboard-read-foreign), or content a
    password manager marked private."""

    def __init__(
        self,
        reason: str,
        *,
        message: str | None = None,
        marker: str | None = None,
        writeback_age_seconds: float | None = None,
    ) -> None:
        details: dict[str, Any] = {"reason": reason}
        if marker is not None:
            details["marker"] = marker
        if writeback_age_seconds is not None:
            details["writeback_age_seconds"] = writeback_age_seconds
        super().__init__(
            ErrorCode.CLIPBOARD_CONCEALED,
            message or f"clipboard contents not readable: {reason}",
            details=details,
        )


__all__ = ["ClipboardConcealedError", "ComputerUseError"]
