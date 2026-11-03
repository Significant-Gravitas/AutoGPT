"""
ComputerUseBackend — the per-OS abstract base class.

Each backend method takes raw values (not message envelopes) and returns
raw values. The dispatcher in `handlers.ComputerUseHandler` wraps in
wire envelopes / converts BackendError → ErrorMessage. Keeps backends
testable without the protocol layer in the way.

Capability advertisement: every backend exposes
`coarse_features()` and `features()` — the daemon reads these at HELLO
time into `HelloPayload.computer_use_features` /
`computer_use_features_coarse`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, NoReturn

from ..config import ShimConfig
from ..protocol import (
    AppInfo,
    DisplayMonitor,
    WindowInfo,
)
from .clipboard import ClipboardWritebackRegistry
from .window_registry import WindowRegistry


@dataclass(frozen=True)
class ScreenshotResult:
    image_bytes: bytes
    mime_type: str
    width: int
    height: int
    monitor: int
    region: tuple[int, int, int, int] | None
    display_scale: float
    logical_size: tuple[int, int] | None
    origin: tuple[int, int]
    display_id: int


@dataclass(frozen=True)
class ClipboardReadResult:
    format: str
    content: str
    size_bytes: int


class ComputerUseBackend(ABC):
    """Per-OS backend interface.

    All methods may raise BackendError subclasses (WindowStaleError,
    PermissionPendingError, FeatureNotSupportedError, etc.) which the
    dispatcher converts to wire ERROR envelopes. Implementations should
    NOT catch and swallow these; let them propagate.
    """

    def __init__(self, config: ShimConfig) -> None:
        self.config = config
        self.window_registry = WindowRegistry()
        self.clipboard_writeback = ClipboardWritebackRegistry()

    # ── Capability advertisement ──────────────────────────────────────

    @abstractmethod
    def coarse_features(self) -> list[str]:
        """Subset of {"screenshot", "input", "windows", "apps",
        "clipboard", "permissions"} this backend supports given current
        config + OS state."""

    @abstractmethod
    def features(self) -> list[str]:
        """Fine-grained list per COMPUTER_USE.md `computer_use_features`."""

    # ── Screenshot ───────────────────────────────────────────────────

    @abstractmethod
    def screenshot(
        self,
        *,
        monitor: int = 0,
        quality: int = 75,
        region: tuple[int, int, int, int] | None = None,
        window_id: str | None = None,
        format: str = "jpeg",
        include_cursor: bool = True,
    ) -> ScreenshotResult: ...

    # ── Input ────────────────────────────────────────────────────────

    @abstractmethod
    def input_action(
        self,
        action: str,
        *,
        coordinate: tuple[int, int] | None = None,
        text: str | None = None,
        key: str | None = None,
        direction: str | None = None,
        clicks: int | None = None,
        button: str | None = None,
        modifiers: list[str] | None = None,
        scroll_amount: int | None = None,
        scroll_direction: str | None = None,
        duration_ms: int | None = None,
        path: list[tuple[int, int]] | None = None,
        paste: bool = False,
        preserve_clipboard: bool = False,
    ) -> None: ...

    # ── Cursor / display ─────────────────────────────────────────────

    @abstractmethod
    def cursor_position(self) -> tuple[int, int, int]:
        """Returns (x, y, monitor_index)."""

    @abstractmethod
    def display_info(self) -> list[DisplayMonitor]: ...

    # ── Windows ──────────────────────────────────────────────────────

    @abstractmethod
    def window_list(
        self,
        *,
        app_bundle_id: str | None = None,
        include_minimized: bool = False,
        include_offscreen: bool = False,
    ) -> list[WindowInfo]: ...

    @abstractmethod
    def window_focus(self, window_id: str, *, raise_: bool = True) -> None: ...

    # ── Apps ─────────────────────────────────────────────────────────

    @abstractmethod
    def app_list(self, *, include_background: bool = False) -> list[AppInfo]: ...

    @abstractmethod
    def app_launch(
        self,
        *,
        bundle_id: str | None = None,
        executable_path: str | None = None,
        args: list[str] | None = None,
        activate: bool = True,
    ) -> int:
        """Returns the launched pid."""

    # ── Clipboard ────────────────────────────────────────────────────

    @abstractmethod
    def clipboard_read(self, *, format: str = "text") -> ClipboardReadResult: ...

    @abstractmethod
    def clipboard_write(self, *, format: str = "text", content: str) -> None: ...

    # ── Permissions ──────────────────────────────────────────────────

    @abstractmethod
    def permissions_check(self, permissions: list[str]) -> dict[str, str]: ...

    # ── Lifecycle ────────────────────────────────────────────────────

    def on_hello(self) -> None:
        """Called by daemon on (re)connect. Per Q2: wipe window IDs."""
        self.window_registry.clear()
        # NOTE: we deliberately do NOT clear clipboard writeback here.
        # The 30 s window is process-lifetime, not session-lifetime;
        # a reconnect mid-paste shouldn't lose our just-written value.


# ── Backend selection ────────────────────────────────────────────────────


def get_backend(config: ShimConfig) -> ComputerUseBackend:
    """Return the per-OS backend for the current process.

    Falls back to a NullBackend if no per-OS implementation is suitable
    (e.g., Wayland for input ops, WSL2 with no display). The null
    backend advertises an empty coarse feature set and rejects every
    op with FEATURE_NOT_SUPPORTED.
    """
    from .. import platform_info

    plat = platform_info.detect_platform()
    if plat == "darwin":
        from .backends.macos import MacOSBackend

        return MacOSBackend(config)
    if plat == "windows":
        from .backends.windows import WindowsBackend

        return WindowsBackend(config)
    if plat == "linux":
        from .backends.linux import LinuxBackend

        return LinuxBackend(config)
    # WSL2 / anything else — no display reachable.
    return NullBackend(config)


class NullBackend(ComputerUseBackend):
    """Backend that refuses every op with FEATURE_NOT_SUPPORTED.

    Used on WSL2 (no host display) and as a fallback for unhandled
    platforms. Advertises empty coarse + fine features so the platform
    side never tries to call us.
    """

    def coarse_features(self) -> list[str]:
        return []

    def features(self) -> list[str]:
        return []

    def _refuse(self, feature: str) -> NoReturn:
        from .errors import FeatureNotSupportedError

        raise FeatureNotSupportedError(feature, reason="no display reachable")

    def screenshot(self, **kw: Any) -> ScreenshotResult:  # type: ignore[override]
        self._refuse("screenshot")

    def input_action(self, action: str, **kw: Any) -> None:  # type: ignore[override]
        self._refuse("input")

    def cursor_position(self) -> tuple[int, int, int]:
        self._refuse("cursor.position")

    def display_info(self) -> list[DisplayMonitor]:
        self._refuse("display.info")

    def window_list(self, **kw: Any) -> list[WindowInfo]:  # type: ignore[override]
        self._refuse("window.list")

    def window_focus(self, window_id: str, *, raise_: bool = True) -> None:
        self._refuse("window.focus")

    def app_list(self, **kw: Any) -> list[AppInfo]:  # type: ignore[override]
        self._refuse("app.list")

    def app_launch(self, **kw: Any) -> int:  # type: ignore[override]
        self._refuse("app.launch")

    def clipboard_read(self, *, format: str = "text") -> ClipboardReadResult:
        self._refuse("clipboard.read")

    def clipboard_write(self, *, format: str = "text", content: str) -> None:
        self._refuse("clipboard.write")

    def permissions_check(self, permissions: list[str]) -> dict[str, str]:
        return {p: "not_applicable" for p in permissions}


__all__ = [
    "ClipboardReadResult",
    "ComputerUseBackend",
    "NullBackend",
    "ScreenshotResult",
    "get_backend",
]
