"""
Linux backend — pyautogui (X11) + xclip + mss.

Implements docs/COMPUTER_USE.md against the X11 / Wayland surfaces:
- Screen capture: mss (X11 OK; Wayland blocked unless using a portal).
- Input: pyautogui (X11). Wayland: input ops raise FEATURE_NOT_SUPPORTED.
- Window list: best-effort. Without `wmctrl` or an AT-SPI bridge we'd be
  rebuilding the wheel, so the current implementation returns
  FEATURE_NOT_SUPPORTED with a reason if wmctrl isn't on PATH. Same
  story for window focus.
- App list: walks /proc for command-line + executable path. Good enough
  to answer "is Slack running" without binding to GTK or AT-SPI.
- App launch: subprocess.Popen with `start_new_session=True`. Honors
  `.desktop` files via gtk-launch / dex if present.
- Clipboard: xclip / wl-clipboard.
- Permissions: returns `not_applicable` on Linux for TCC keys (no
  equivalent); display_server hint is reported separately.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from typing import Any

from ...config import ShimConfig
from ...protocol import AppInfo, DisplayMonitor, WindowInfo
from ..backend import ClipboardReadResult, ComputerUseBackend, ScreenshotResult
from ..display import displays_to_bounds_list, displays_via_mss
from ..errors import (
    ClipboardConcealedError,
    FeatureNotSupportedError,
    InputOutOfBoundsError,
)
from ..window_registry import WindowFingerprint
from ._common import (
    PASTE_THRESHOLD_CHARS,
    image_to_base64,
    in_any_rect,
    normalize_displays_for_error,
)

logger = logging.getLogger(__name__)


def _is_wayland() -> bool:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" or bool(
        os.environ.get("WAYLAND_DISPLAY")
    )


class LinuxBackend(ComputerUseBackend):
    def __init__(self, config: ShimConfig) -> None:
        super().__init__(config)
        self._clipboard_enabled = getattr(config, "enable_clipboard", False)
        self._clipboard_read_foreign = getattr(config, "enable_clipboard_read_foreign", False)
        self._wayland = _is_wayland()
        self._has_wmctrl = shutil.which("wmctrl") is not None
        self._has_xclip = shutil.which("xclip") is not None
        self._has_wlcopy = shutil.which("wl-copy") is not None

    # ── Capability advertisement ──────────────────────────────────────

    def coarse_features(self) -> list[str]:
        if self._wayland:
            return ["screenshot"]
        feats = ["screenshot", "input", "apps", "permissions"]
        if self._has_wmctrl:
            feats.append("windows")
        if self._clipboard_enabled and (self._has_xclip or self._has_wlcopy):
            feats.append("clipboard")
        return feats

    def features(self) -> list[str]:
        if self._wayland:
            return [
                "screenshot.region",
                "display.info",
                "app.list",
                "app.launch",
                "permissions.check",
            ]
        feats = [
            "screenshot.region",
            "input.click.modifiers",
            "input.click.button",
            "input.drag.path",
            "input.key.hold",
            "input.mouse.down_up",
            "input.scroll.amount",
            "input.wait",
            "cursor.position",
            "display.info",
            "app.list",
            "app.launch",
            "permissions.check",
        ]
        if self._has_wmctrl:
            feats.extend(["window.list", "window.focus"])
        if self._clipboard_enabled and (self._has_xclip or self._has_wlcopy):
            feats.extend(["clipboard.read", "clipboard.write"])
        return feats

    # ── Screenshot ───────────────────────────────────────────────────

    def screenshot(
        self,
        *,
        monitor: int = 0,
        quality: int = 75,
        region: tuple[int, int, int, int] | None = None,
        window_id: str | None = None,
        format: str = "jpeg",
        include_cursor: bool = True,
    ) -> ScreenshotResult:
        if region is not None and window_id is not None:
            raise FeatureNotSupportedError(
                "screenshot.region+window",
                reason="region and window_id are mutually exclusive",
            )
        if window_id is not None:
            raise FeatureNotSupportedError(
                "screenshot.window",
                reason="window capture on Linux requires per-WM cooperation; not in v1",
            )
        try:
            import mss  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError as exc:
            raise FeatureNotSupportedError("screenshot", reason=str(exc)) from exc
        if self._wayland:
            # mss on Wayland sometimes errors, sometimes returns black; we
            # try and surface the error rather than guessing.
            try:
                with mss.mss() as _:
                    pass
            except Exception as exc:
                raise FeatureNotSupportedError(
                    "screenshot",
                    reason=f"Wayland session: {exc} — use a portal+pipewire backend",
                ) from exc

        displays = self.display_info()
        if not displays:
            raise FeatureNotSupportedError("screenshot", reason="no displays detected")
        if monitor < 0 or monitor >= len(displays):
            monitor = 0
        target = displays[monitor]

        with mss.mss() as sct:
            mon = sct.monitors[monitor + 1]
            if region is not None:
                rx1, ry1, rx2, ry2 = region
                mx, my, mw, mh = mon["left"], mon["top"], mon["width"], mon["height"]
                cx1 = max(rx1, mx)
                cy1 = max(ry1, my)
                cx2 = min(rx2, mx + mw)
                cy2 = min(ry2, my + mh)
                if cx2 <= cx1 or cy2 <= cy1:
                    raise FeatureNotSupportedError(
                        "screenshot.region",
                        reason=f"region {region} does not intersect monitor {monitor}",
                    )
                raw = sct.grab({"left": cx1, "top": cy1, "width": cx2 - cx1, "height": cy2 - cy1})
                origin = (cx1, cy1)
            else:
                raw = sct.grab(mon)
                origin = (mon["left"], mon["top"])
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

        img_bytes, mime = image_to_base64(img, format=format, quality=quality)
        return ScreenshotResult(
            image_bytes=img_bytes,
            mime_type=mime,
            width=img.width,
            height=img.height,
            monitor=target.index,
            region=region,
            display_scale=target.scale,
            logical_size=target.logical_size,
            origin=origin,
            display_id=target.index,
        )

    # ── Input ────────────────────────────────────────────────────────

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
    ) -> None:
        if action == "wait":
            ms = max(0, min(int(duration_ms or 0), 5000))
            time.sleep(ms / 1000.0)
            return

        if self._wayland:
            raise FeatureNotSupportedError(
                "input", reason="Wayland sessions block input injection; switch to X11"
            )

        if coordinate is not None:
            self._check_bounds(coordinate)
        if path:
            for pt in path:
                self._check_bounds(pt)

        try:
            import pyautogui  # type: ignore[import-untyped]
        except ImportError as exc:
            raise FeatureNotSupportedError("input", reason=str(exc)) from exc

        pyautogui.FAILSAFE = True

        if action == "mouse_move" and coordinate:
            pyautogui.moveTo(coordinate[0], coordinate[1])
            return
        if action in ("left_click", "right_click", "double_click", "middle_click", "triple_click"):
            self._click(action, coordinate, modifiers)
            return
        if action == "mouse_down" and coordinate:
            pyautogui.mouseDown(coordinate[0], coordinate[1], button=button or "left")
            return
        if action == "mouse_up" and coordinate:
            pyautogui.mouseUp(coordinate[0], coordinate[1], button=button or "left")
            return
        if action == "drag" and path:
            btn = button or "left"
            pyautogui.moveTo(path[0][0], path[0][1])
            pyautogui.mouseDown(button=btn)
            try:
                # X11 may glitch if path is > 100 points; sub-sample.
                step = max(1, len(path) // 100)
                for pt in path[1::step]:
                    pyautogui.moveTo(pt[0], pt[1], duration=0.01)
                pyautogui.moveTo(path[-1][0], path[-1][1], duration=0.01)
            finally:
                pyautogui.mouseUp(button=btn)
            return
        if action == "type" and text is not None:
            self._type(text, paste=paste, preserve_clipboard=preserve_clipboard)
            return
        if action == "key" and key:
            pyautogui.hotkey(*self._chord(key))
            return
        if action == "hold_key" and key:
            duration = max(0.0, min((duration_ms or 0) / 1000.0, 5.0))
            for k in self._chord(key):
                pyautogui.keyDown(k)
            try:
                time.sleep(duration)
            finally:
                for k in reversed(self._chord(key)):
                    pyautogui.keyUp(k)
            return
        if action == "scroll":
            amount = scroll_amount if scroll_amount is not None else (clicks or 3)
            d = scroll_direction or direction or "down"
            sign = 1 if d in ("up", "right") else -1
            if coordinate:
                pyautogui.scroll(sign * amount, x=coordinate[0], y=coordinate[1])
            else:
                pyautogui.scroll(sign * amount)
            return
        raise FeatureNotSupportedError(action, reason=f"unknown action {action!r}")

    def _click(
        self,
        action: str,
        coordinate: tuple[int, int] | None,
        modifiers: list[str] | None,
    ) -> None:
        import pyautogui  # type: ignore[import-untyped]

        if coordinate is None:
            raise FeatureNotSupportedError(action, reason="coordinate is required")
        mods = list(modifiers or [])
        for m in mods:
            pyautogui.keyDown(self._modifier(m))
        try:
            if action == "left_click":
                pyautogui.click(coordinate[0], coordinate[1])
            elif action == "right_click":
                pyautogui.rightClick(coordinate[0], coordinate[1])
            elif action == "double_click":
                pyautogui.doubleClick(coordinate[0], coordinate[1])
            elif action == "middle_click":
                pyautogui.middleClick(coordinate[0], coordinate[1])
            elif action == "triple_click":
                # X11 has no native triple-click; emit three clicks with
                # the same coord ~50 ms apart.
                pyautogui.click(coordinate[0], coordinate[1])
                time.sleep(0.05)
                pyautogui.click(coordinate[0], coordinate[1])
                time.sleep(0.05)
                pyautogui.click(coordinate[0], coordinate[1])
        finally:
            for m in reversed(mods):
                pyautogui.keyUp(self._modifier(m))

    @staticmethod
    def _modifier(name: str) -> str:
        return {"super": "win", "ctrl": "ctrl", "alt": "alt", "shift": "shift"}.get(name, name)

    @staticmethod
    def _chord(key: str) -> list[str]:
        return [k.strip().lower() for k in key.split("+") if k.strip()]

    def _check_bounds(self, coordinate: tuple[int, int]) -> None:
        displays = self.display_info()
        bounds = displays_to_bounds_list(displays)
        if not in_any_rect(coordinate, bounds):
            raise InputOutOfBoundsError(coordinate, normalize_displays_for_error(bounds))

    # ── Cursor / display ─────────────────────────────────────────────

    def cursor_position(self) -> tuple[int, int, int]:
        if self._wayland:
            raise FeatureNotSupportedError("cursor.position", reason="Wayland session")
        try:
            import pyautogui  # type: ignore[import-untyped]
        except ImportError as exc:
            raise FeatureNotSupportedError("cursor.position", reason=str(exc)) from exc
        pos = pyautogui.position()
        x, y = int(pos[0]), int(pos[1])
        idx = 0
        for d in self.display_info():
            ox, oy = d.origin
            w, h = d.physical_size
            if ox <= x < ox + w and oy <= y < oy + h:
                idx = d.index
                break
        return x, y, idx

    def display_info(self) -> list[DisplayMonitor]:
        ms = displays_via_mss()
        if ms:
            return ms
        raise FeatureNotSupportedError("display.info", reason="mss not available")

    # ── Windows ──────────────────────────────────────────────────────

    def window_list(
        self,
        *,
        app_bundle_id: str | None = None,
        include_minimized: bool = False,
        include_offscreen: bool = False,
    ) -> list[WindowInfo]:
        if self._wayland or not self._has_wmctrl:
            raise FeatureNotSupportedError(
                "window.list",
                reason=("Wayland session" if self._wayland else "wmctrl not installed"),
            )
        try:
            out_raw = subprocess.check_output(["wmctrl", "-l", "-G", "-p"], text=True, timeout=5)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise FeatureNotSupportedError("window.list", reason=str(exc)) from exc

        self.window_registry.clear()
        out: list[WindowInfo] = []
        for line in out_raw.splitlines():
            parts = line.split(None, 8)
            # wmctrl -lGp: <winid> <desktop> <pid> <x> <y> <w> <h> <host> <title...>
            if len(parts) < 8:
                continue
            try:
                native_id_str, _desktop, pid_s, x_s, y_s, w_s, h_s = parts[:7]
                title = parts[8] if len(parts) >= 9 else ""
                native_id = int(native_id_str, 16)
                pid = int(pid_s)
                x, y, w, h = int(x_s), int(y_s), int(w_s), int(h_s)
            except ValueError:
                continue
            fp = WindowFingerprint(pid=pid, class_name=None, creation_timestamp=None)
            wid = self.window_registry.mint(native_id, fp, extra={"bounds": (x, y, x + w, y + h)})
            out.append(
                WindowInfo(
                    window_id=wid,
                    pid=pid,
                    app_name=None,
                    app_bundle_id=None,
                    title=title or None,
                    bounds=(x, y, x + w, y + h),
                    monitor=0,
                    is_focused=False,
                    is_minimized=False,
                    is_fullscreen=False,
                )
            )
        return out

    def _live_fingerprint(self, native_id: Any) -> WindowFingerprint | None:
        if not self._has_wmctrl:
            return None
        try:
            out = subprocess.check_output(["wmctrl", "-l", "-p"], text=True, timeout=5)
            for line in out.splitlines():
                parts = line.split(None, 4)
                if len(parts) < 4:
                    continue
                if int(parts[0], 16) == int(native_id):
                    return WindowFingerprint(
                        pid=int(parts[2]), class_name=None, creation_timestamp=None
                    )
        except Exception:
            return None
        return None

    def window_focus(self, window_id: str, *, raise_: bool = True) -> None:
        if self._wayland or not self._has_wmctrl:
            raise FeatureNotSupportedError(
                "window.focus",
                reason=("Wayland session" if self._wayland else "wmctrl not installed"),
            )
        entry = self.window_registry.verify(window_id, self._live_fingerprint)
        try:
            subprocess.run(
                ["wmctrl", "-i", "-a", hex(int(entry.native_handle))], check=True, timeout=5
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise FeatureNotSupportedError(
                "window.focus", reason=f"wmctrl -a failed: {exc}"
            ) from exc

    # ── Apps ─────────────────────────────────────────────────────────

    def app_list(self, *, include_background: bool = False) -> list[AppInfo]:
        out: list[AppInfo] = []
        try:
            entries = os.listdir("/proc")
        except OSError as exc:
            raise FeatureNotSupportedError("app.list", reason=str(exc)) from exc
        for entry in entries:
            if not entry.isdigit():
                continue
            pid = int(entry)
            try:
                with open(f"/proc/{pid}/comm", encoding="utf-8") as f:
                    name = f.read().strip()
            except OSError:
                continue
            exe = None
            try:
                exe = os.readlink(f"/proc/{pid}/exe")
            except OSError:
                pass
            out.append(
                AppInfo(
                    pid=pid,
                    name=name,
                    bundle_id=None,
                    executable_path=exe,
                    is_frontmost=False,
                    window_count=0,
                )
            )
        return out

    def app_launch(
        self,
        *,
        bundle_id: str | None = None,
        executable_path: str | None = None,
        args: list[str] | None = None,
        activate: bool = True,
    ) -> int:
        if not executable_path:
            raise FeatureNotSupportedError(
                "app.launch", reason="executable_path is required on Linux"
            )
        # .desktop files: prefer gtk-launch / dex if available.
        if executable_path.endswith(".desktop"):
            for helper in ("gtk-launch", "dex"):
                if shutil.which(helper):
                    cmd = [helper, os.path.basename(executable_path).removesuffix(".desktop")]
                    proc = subprocess.Popen(cmd, start_new_session=True)
                    return proc.pid
            raise FeatureNotSupportedError(
                "app.launch", reason=".desktop launch needs gtk-launch or dex"
            )
        proc = subprocess.Popen([executable_path, *(args or [])], start_new_session=True)
        return proc.pid

    # ── Clipboard ────────────────────────────────────────────────────

    def clipboard_read(self, *, format: str = "text") -> ClipboardReadResult:
        if not self._clipboard_enabled:
            raise FeatureNotSupportedError("clipboard.read", reason="--enable-clipboard not set")
        if format != "text":
            raise FeatureNotSupportedError(
                "clipboard.read", reason=f"format {format!r} not supported"
            )
        text, seq = self._read_clipboard_text()
        if self._clipboard_read_foreign:
            content = text or ""
            return ClipboardReadResult(
                format="text", content=content, size_bytes=len(content.encode("utf-8"))
            )

        record = self.clipboard_writeback.check(live_content=text, live_sequence=seq)
        if record is None:
            snap = self.clipboard_writeback.snapshot()
            if snap is None:
                raise ClipboardConcealedError("writeback_only")
            age = snap.age()
            if age > self.clipboard_writeback.window_seconds:
                raise ClipboardConcealedError("writeback_only", writeback_age_seconds=age)
            raise ClipboardConcealedError("writeback_overwritten")
        content = text or ""
        return ClipboardReadResult(
            format="text", content=content, size_bytes=len(content.encode("utf-8"))
        )

    def clipboard_write(self, *, format: str = "text", content: str) -> None:
        if not self._clipboard_enabled:
            raise FeatureNotSupportedError("clipboard.write", reason="--enable-clipboard not set")
        if format != "text":
            raise FeatureNotSupportedError(
                "clipboard.write", reason=f"format {format!r} not supported"
            )
        seq = self._write_clipboard_text(content)
        self.clipboard_writeback.record_write(content, sequence=seq)

    def _read_clipboard_text(self) -> tuple[str | None, int | None]:
        if self._wayland and self._has_wlcopy and shutil.which("wl-paste"):
            try:
                out = subprocess.check_output(["wl-paste", "--no-newline"], timeout=2)
                return out.decode("utf-8", errors="replace"), None
            except Exception:
                return None, None
        if self._has_xclip:
            try:
                out = subprocess.check_output(["xclip", "-selection", "clipboard", "-o"], timeout=2)
                return out.decode("utf-8", errors="replace"), None
            except Exception:
                return None, None
        return None, None

    def _write_clipboard_text(self, content: str) -> int | None:
        if self._wayland and self._has_wlcopy:
            try:
                subprocess.run(["wl-copy"], input=content.encode("utf-8"), check=True, timeout=2)
                return None
            except Exception:
                return None
        if self._has_xclip:
            try:
                subprocess.run(
                    ["xclip", "-selection", "clipboard", "-i"],
                    input=content.encode("utf-8"),
                    check=True,
                    timeout=2,
                )
                return None
            except Exception:
                return None
        return None

    def _type(self, text: str, *, paste: bool, preserve_clipboard: bool) -> None:
        import pyautogui  # type: ignore[import-untyped]

        if (
            paste
            and len(text) >= PASTE_THRESHOLD_CHARS
            and self._clipboard_enabled
            and (self._has_xclip or self._has_wlcopy)
        ):
            self._do_paste(text, preserve_clipboard=preserve_clipboard)
            return
        pyautogui.write(text, interval=0.02)

    def _do_paste(self, text: str, *, preserve_clipboard: bool) -> None:
        import pyautogui  # type: ignore[import-untyped]

        snapshot: tuple[str | None, int | None] | None = None
        if preserve_clipboard:
            snapshot = self._read_clipboard_text()
        self._write_clipboard_text(text)
        self.clipboard_writeback.record_write(text)
        pyautogui.hotkey("ctrl", "v")
        if preserve_clipboard and snapshot is not None and snapshot[0] is not None:
            time.sleep(0.05)
            # We don't have a reliable sequence number on X11/Wayland from
            # the CLI utilities; the changeCount-skip-restore protection
            # falls back to "restore unconditionally" here. Documented in
            # COMPUTER_USE.md Q4 as a race window.
            self._write_clipboard_text(snapshot[0])
            self.clipboard_writeback.record_write(snapshot[0])

    # ── Permissions ──────────────────────────────────────────────────

    def permissions_check(self, permissions: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for p in permissions:
            if p in ("screen_recording", "accessibility", "input_monitoring"):
                if self._wayland:
                    out[p] = "denied"
                else:
                    out[p] = "granted"
            else:
                out[p] = "not_applicable"
        return out


__all__ = ["LinuxBackend"]
