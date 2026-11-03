"""
Computer-use backend layer.

Per-OS backends live under `computer_use.backends.*`; the dispatch in
`computer_use.backend.get_backend()` picks the right one at runtime.
"""

from .backend import (
    ClipboardReadResult,
    ComputerUseBackend,
    NullBackend,
    ScreenshotResult,
    get_backend,
)
from .errors import (
    BackendError,
    ClipboardConcealedError,
    FeatureNotSupportedError,
    InputOutOfBoundsError,
    PermissionPendingError,
    WindowStaleError,
)

__all__ = [
    "BackendError",
    "ClipboardConcealedError",
    "ClipboardReadResult",
    "ComputerUseBackend",
    "FeatureNotSupportedError",
    "InputOutOfBoundsError",
    "NullBackend",
    "PermissionPendingError",
    "ScreenshotResult",
    "WindowStaleError",
    "get_backend",
]
