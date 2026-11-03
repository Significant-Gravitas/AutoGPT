"""
Structured backend errors → wire ERROR envelopes.

Each backend method raises one of these on a recoverable failure; the
dispatcher in `computer_use.backend` converts to an `ErrorMessage` with
the matching `ErrorCode` and a `details` payload the platform can
inspect (display rects, window_id hints, etc). See docs/COMPUTER_USE.md
Q1-Q5 for the per-code shape.
"""

from __future__ import annotations

from typing import Any

from ..protocol import ErrorCode


class BackendError(Exception):
    """Base class for structured backend errors."""

    code: ErrorCode

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class WindowStaleError(BackendError):
    code = ErrorCode.WINDOW_STALE

    def __init__(self, window_id: str, *, message: str | None = None) -> None:
        super().__init__(
            message or f"window_id {window_id} no longer maps to a live window",
            details={"window_id": window_id, "hint": "re-list windows"},
        )


class PermissionPendingError(BackendError):
    code = ErrorCode.PERMISSION_PENDING

    def __init__(
        self,
        permission: str,
        platform: str,
        *,
        message: str | None = None,
        hint: str | None = None,
    ) -> None:
        details: dict[str, Any] = {"permission": permission, "platform": platform}
        if hint:
            details["hint"] = hint
        super().__init__(
            message or f"{permission} permission required but not granted",
            details=details,
        )


class FeatureNotSupportedError(BackendError):
    code = ErrorCode.FEATURE_NOT_SUPPORTED

    def __init__(self, feature: str, *, reason: str | None = None) -> None:
        details: dict[str, Any] = {"feature": feature}
        if reason:
            details["reason"] = reason
        super().__init__(
            f"feature {feature!r} is not supported on this backend"
            + (f": {reason}" if reason else ""),
            details=details,
        )


class ClipboardConcealedError(BackendError):
    code = ErrorCode.CLIPBOARD_CONCEALED

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
            message or f"clipboard contents not readable: {reason}",
            details=details,
        )


class InputOutOfBoundsError(BackendError):
    code = ErrorCode.INPUT_OUT_OF_BOUNDS

    def __init__(
        self,
        coordinate: tuple[int, int],
        displays: list[dict[str, Any]],
    ) -> None:
        super().__init__(
            f"coordinate {tuple(coordinate)} is outside the union of connected display bounds",
            details={
                "requested_coordinate": list(coordinate),
                "displays": displays,
            },
        )


__all__ = [
    "BackendError",
    "ClipboardConcealedError",
    "FeatureNotSupportedError",
    "InputOutOfBoundsError",
    "PermissionPendingError",
    "WindowStaleError",
]
