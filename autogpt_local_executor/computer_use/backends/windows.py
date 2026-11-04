"""
Windows backend — pywin32 + pyautogui + mss + pyperclip.

Implements docs/COMPUTER_USE.md against the Win32 APIs:
- Screen capture: mss
- Input: pyautogui (SendInput)
- Window list / focus: EnumWindows + SetForegroundWindow
- App list / launch: EnumProcesses + ShellExecute
- Clipboard: OpenClipboard / GetClipboardData / SetClipboardData; CF_PRIVATE
  detection
- Permissions: process elevation level

Every win32 symbol is imported lazily so this module loads on non-Windows
for tests / type-checking.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from typing import Any

from ...config import ShimConfig
from ...protocol import AppInfo, DisplayMonitor, WindowInfo
from ..backend import ClipboardReadResult, ComputerUseBackend, ScreenshotResult
from ..display import displays_to_bounds_list, displays_via_mss
from ..errors import (
    ClipboardConcealedError,
    ClipboardWriteError,
    FeatureNotSupportedError,
    InputOutOfBoundsError,
)
from ..window_registry import WindowFingerprint
from ._common import (
    capture_fields,
    chord_keys,
    ensure_typeable,
    image_to_base64,
    in_any_rect,
    normalize_displays_for_error,
    paste_is_worth_it,
)

logger = logging.getLogger(__name__)

# The documented modifier names, in pyautogui's: super (and its alias meta)
# is the Windows logo key.
_KEY_ALIASES = {"super": "win", "meta": "win"}
# mouse_event's wheel flags, and one notch of the wheel in its units.
_MOUSEEVENTF_WHEEL = 0x0800
_MOUSEEVENTF_HWHEEL = 0x1000
_WHEEL_DELTA = 120

# CF_PRIVATEFIRST..CF_PRIVATELAST — apps that hold "do not paste this" data
# set a format in this range alongside CF_TEXT/CF_UNICODETEXT.
CF_PRIVATEFIRST = 0x0200
CF_PRIVATELAST = 0x02FF
# What a text-only restore rebuilds: CF_TEXT, CF_OEMTEXT, CF_UNICODETEXT and
# CF_LOCALE (Windows synthesizes the others from any one of them).
_TEXT_CLIPBOARD_FORMATS = frozenset({1, 7, 13, 16})


class WindowsBackend(ComputerUseBackend):
    def __init__(self, config: ShimConfig) -> None:
        super().__init__(config)
        self._clipboard_enabled = getattr(config, "enable_clipboard", False)
        self._clipboard_read_foreign = getattr(config, "enable_clipboard_read_foreign", False)

    # ── Capability advertisement ──────────────────────────────────────

    def coarse_features(self) -> list[str]:
        feats = ["screenshot", "input", "windows", "apps", "permissions"]
        if self._clipboard_enabled:
            feats.append("clipboard")
        return feats

    def features(self) -> list[str]:
        feats = [
            "screenshot.region",
            "screenshot.window",
            "input.click.modifiers",
            "input.click.button",
            "input.drag.path",
            "input.key.hold",
            "input.mouse.down_up",
            "input.scroll.amount",
            "input.wait",
            "cursor.position",
            "display.info",
            "window.list",
            "window.focus",
            "app.list",
            "app.launch",
            "permissions.check",
        ]
        if self._clipboard_enabled:
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
        try:
            import mss  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError as exc:
            raise FeatureNotSupportedError("screenshot", reason=str(exc)) from exc

        if window_id is not None:
            return self._screenshot_window(
                window_id, quality=quality, format=format, include_cursor=include_cursor
            )

        displays = self.display_info()
        if not displays:
            raise FeatureNotSupportedError("screenshot", reason="no displays detected")
        # PROTOCOL.md: monitor -1 captures every monitor stitched together.
        stitched = monitor == -1
        if not stitched and not 0 <= monitor < len(displays):
            monitor = 0

        with mss.mss() as sct:
            mon = sct.monitors[0 if stitched else monitor + 1]
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
                grabbed_width = cx2 - cx1
            else:
                raw = sct.grab(mon)
                origin = (mon["left"], mon["top"])
                grabbed_width = mon["width"]
            img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        index, scale, logical_size = capture_fields(
            displays,
            monitor,
            stitched=stitched,
            virtual_desktop=mon,
            grabbed_width=grabbed_width,
            image_width=img.width,
        )

        if include_cursor:
            try:
                self._composite_cursor(img, origin)
            except Exception:
                logger.debug("cursor composite failed", exc_info=True)

        img_bytes, mime = image_to_base64(img, format=format, quality=quality)
        return ScreenshotResult(
            image_bytes=img_bytes,
            mime_type=mime,
            width=img.width,
            height=img.height,
            monitor=index,
            region=region,
            display_scale=scale,
            logical_size=logical_size,
            origin=origin,
            display_id=index,
        )

    def _composite_cursor(self, img: Any, origin: tuple[int, int]) -> None:
        try:
            import win32api  # type: ignore[import-not-found]
            from PIL import ImageDraw  # type: ignore[import-untyped]
        except ImportError:
            return
        try:
            x, y = win32api.GetCursorPos()
            cx, cy = x - origin[0], y - origin[1]
            if 0 <= cx < img.width and 0 <= cy < img.height:
                d = ImageDraw.Draw(img)
                d.polygon(
                    [(cx, cy), (cx + 12, cy + 4), (cx + 5, cy + 5), (cx + 4, cy + 12)],
                    fill=(0, 0, 0),
                )
        except Exception:
            return

    def _screenshot_window(
        self,
        window_id: str,
        *,
        quality: int,
        format: str,
        include_cursor: bool = False,
    ) -> ScreenshotResult:
        import ctypes

        try:
            import win32gui  # type: ignore[import-not-found]
            import win32ui  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-untyped]
        except ImportError as exc:
            raise FeatureNotSupportedError("screenshot.window", reason=str(exc)) from exc
        entry = self.window_registry.verify(window_id, self._live_fingerprint)
        hwnd = int(entry.native_handle)
        if not win32gui.IsWindow(hwnd):
            from ..errors import WindowStaleError

            raise WindowStaleError(window_id)
        left, top, right, bot = win32gui.GetWindowRect(hwnd)
        w, h = right - left, bot - top
        if w <= 0 or h <= 0:
            raise FeatureNotSupportedError("screenshot.window", reason="zero-size window")
        # PrintWindow asks the window to paint itself and waits for it, so on
        # a window that has stopped responding it would never return.
        try:
            hung = ctypes.windll.user32.IsHungAppWindow(hwnd)  # type: ignore[attr-defined]
        except Exception as exc:
            raise FeatureNotSupportedError(
                "screenshot.window", reason=f"IsHungAppWindow failed: {exc}"
            ) from exc
        if hung:
            raise FeatureNotSupportedError(
                "screenshot.window", reason="the window is not responding"
            )
        # Acquired inside the try, so whatever was acquired is released even
        # when a later step (the bitmap, say) fails.
        hwnd_dc = mfc_dc = save_dc = bmp = previous = None
        bitmap_made = False
        try:
            hwnd_dc = win32gui.GetWindowDC(hwnd)
            mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
            save_dc = mfc_dc.CreateCompatibleDC()
            bmp = win32ui.CreateBitmap()
            bmp.CreateCompatibleBitmap(mfc_dc, w, h)
            bitmap_made = True
            # The DC's own bitmap, which goes back in before the DC is deleted.
            previous = save_dc.SelectObject(bmp)
            # PrintWindow with PW_RENDERFULLCONTENT (0x02) is the modern flag.
            # It returns 0 when the window did not render into the bitmap.
            try:
                printed = ctypes.windll.user32.PrintWindow(  # type: ignore[attr-defined]
                    hwnd, save_dc.GetSafeHdc(), 0x02
                )
            except Exception as exc:
                raise FeatureNotSupportedError(
                    "screenshot.window", reason=f"PrintWindow failed: {exc}"
                ) from exc
            if not printed:
                raise FeatureNotSupportedError(
                    "screenshot.window", reason="PrintWindow could not capture the window"
                )
            info = bmp.GetInfo()
            bits = bmp.GetBitmapBits(True)
            img = Image.frombuffer(
                "RGB",
                (info["bmWidth"], info["bmHeight"]),
                bits,
                "raw",
                "BGRX",
                0,
                1,
            )
        finally:
            # GDI does not delete a bitmap still selected into a DC, so the
            # DC gets its own bitmap back, then the DCs go, then the bitmap.
            if save_dc is not None and previous is not None:
                try:
                    save_dc.SelectObject(previous)
                except Exception:
                    logger.debug("could not restore the DC's own bitmap", exc_info=True)
            if save_dc is not None:
                save_dc.DeleteDC()
            if mfc_dc is not None:
                mfc_dc.DeleteDC()
            if bitmap_made and bmp is not None:
                win32gui.DeleteObject(bmp.GetHandle())
            if hwnd_dc is not None:
                win32gui.ReleaseDC(hwnd, hwnd_dc)

        if include_cursor:
            try:
                self._composite_cursor(img, (left, top))
            except Exception:
                logger.debug("cursor composite failed", exc_info=True)
        img_bytes, mime = image_to_base64(img, format=format, quality=quality)
        return ScreenshotResult(
            image_bytes=img_bytes,
            mime_type=mime,
            width=img.width,
            height=img.height,
            monitor=int(entry.extra.get("monitor", 0)),
            region=None,
            display_scale=1.0,
            logical_size=(img.width, img.height),
            origin=(left, top),
            display_id=int(entry.extra.get("monitor", 0)),
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
                for pt in path[1:]:
                    pyautogui.moveTo(pt[0], pt[1], duration=0.01)
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
                # The wheel turns the window under the cursor.
                pyautogui.moveTo(coordinate[0], coordinate[1])
            else:
                pyautogui.failSafeCheck()
            self._turn_wheel(sign * amount, horizontal=d in ("left", "right"))
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
                pyautogui.tripleClick(coordinate[0], coordinate[1])
        finally:
            for m in reversed(mods):
                pyautogui.keyUp(self._modifier(m))

    @staticmethod
    def _modifier(name: str) -> str:
        return _KEY_ALIASES.get(name.lower(), name)

    @staticmethod
    def _chord(key: str) -> list[str]:
        return chord_keys(key, _KEY_ALIASES)

    @staticmethod
    def _turn_wheel(notches: int, *, horizontal: bool) -> None:
        """Turn the wheel at the cursor. pyautogui sends its amount as the raw
        wheel delta, a 120th of a notch here, and its hscroll turns the
        vertical wheel on Windows, so the event goes out directly."""
        import ctypes

        flag = _MOUSEEVENTF_HWHEEL if horizontal else _MOUSEEVENTF_WHEEL
        ctypes.windll.user32.mouse_event(  # type: ignore[attr-defined]
            flag, 0, 0, notches * _WHEEL_DELTA, 0
        )

    def _check_bounds(self, coordinate: tuple[int, int]) -> None:
        displays = self.display_info()
        bounds = displays_to_bounds_list(displays)
        if not in_any_rect(coordinate, bounds):
            raise InputOutOfBoundsError(coordinate, normalize_displays_for_error(bounds))

    # ── Cursor / display ─────────────────────────────────────────────

    def cursor_position(self) -> tuple[int, int, int]:
        try:
            import win32api  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FeatureNotSupportedError("cursor.position", reason=str(exc)) from exc
        x, y = win32api.GetCursorPos()
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
        try:
            import win32gui  # type: ignore[import-not-found]
            import win32process  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FeatureNotSupportedError("window.list", reason=str(exc)) from exc

        self.window_registry.clear()
        out: list[WindowInfo] = []
        focus_hwnd = win32gui.GetForegroundWindow()

        def _enum(hwnd: int, _: Any) -> None:
            try:
                # A minimized window keeps WS_VISIBLE, so this leaves out only
                # hidden ones; IsIconic tells minimized ones apart.
                if not win32gui.IsWindowVisible(hwnd):
                    return
                title = win32gui.GetWindowText(hwnd)
                cls = win32gui.GetClassName(hwnd)
                if not title:
                    return
                minimized = bool(win32gui.IsIconic(hwnd))
                if minimized and not include_minimized:
                    return
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid == 0:
                    return
                left, top, right, bot = win32gui.GetWindowRect(hwnd)
                if (right - left) <= 0 or (bot - top) <= 0:
                    if not include_offscreen:
                        return
                fp = WindowFingerprint(pid=pid, class_name=cls, creation_timestamp=None)
                wid = self.window_registry.mint(hwnd, fp, extra={"bounds": (left, top, right, bot)})
                out.append(
                    WindowInfo(
                        window_id=wid,
                        pid=pid,
                        app_name=cls,
                        app_bundle_id=None,
                        title=title or None,
                        bounds=(left, top, right, bot),
                        monitor=0,
                        is_focused=(hwnd == focus_hwnd),
                        is_minimized=minimized,
                        is_fullscreen=False,
                    )
                )
            except Exception:
                logger.debug("skipping window in enum", exc_info=True)

        win32gui.EnumWindows(_enum, None)
        return out

    def _live_fingerprint(self, hwnd: Any) -> WindowFingerprint | None:
        try:
            import win32gui  # type: ignore[import-not-found]
            import win32process  # type: ignore[import-not-found]

            if not win32gui.IsWindow(int(hwnd)):
                return None
            cls = win32gui.GetClassName(int(hwnd))
            _, pid = win32process.GetWindowThreadProcessId(int(hwnd))
            return WindowFingerprint(pid=pid, class_name=cls, creation_timestamp=None)
        except Exception:
            return None

    def window_focus(self, window_id: str, *, raise_: bool = True) -> None:
        try:
            import win32gui  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FeatureNotSupportedError("window.focus", reason=str(exc)) from exc
        entry = self.window_registry.verify(window_id, self._live_fingerprint)
        try:
            win32gui.SetForegroundWindow(int(entry.native_handle))
        except Exception as exc:
            # SetForegroundWindow honors foreground-lock rules; this is
            # best-effort.
            raise FeatureNotSupportedError(
                "window.focus", reason=f"SetForegroundWindow failed: {exc}"
            ) from exc

    # ── Apps ─────────────────────────────────────────────────────────

    def app_list(self, *, include_background: bool = False) -> list[AppInfo]:
        try:
            import win32gui  # type: ignore[import-not-found]
            import win32process  # type: ignore[import-not-found]
        except ImportError as exc:
            raise FeatureNotSupportedError("app.list", reason=str(exc)) from exc

        focus_hwnd = win32gui.GetForegroundWindow()
        pids_with_window: dict[int, tuple[str, int]] = {}
        # (name, window_count)

        def _enum(hwnd: int, _: Any) -> None:
            try:
                if not win32gui.IsWindowVisible(hwnd) and not include_background:
                    return
                title = win32gui.GetWindowText(hwnd)
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if pid == 0:
                    return
                existing = pids_with_window.get(pid, ("", 0))
                pids_with_window[pid] = (existing[0] or title or "", existing[1] + 1)
            except Exception:
                pass

        win32gui.EnumWindows(_enum, None)
        out: list[AppInfo] = []
        for pid, (name, wcount) in pids_with_window.items():
            exe = self._exe_for_pid(pid)
            out.append(
                AppInfo(
                    pid=pid,
                    name=name or (os.path.basename(exe) if exe else ""),
                    bundle_id=None,
                    executable_path=exe,
                    is_frontmost=(
                        pid
                        == (
                            win32process.GetWindowThreadProcessId(focus_hwnd)[1]
                            if focus_hwnd
                            else -1
                        )
                    ),
                    window_count=wcount,
                )
            )
        return out

    def _exe_for_pid(self, pid: int) -> str | None:
        try:
            import win32api  # type: ignore[import-not-found]
            import win32con  # type: ignore[import-not-found]
            import win32process  # type: ignore[import-not-found]

            h = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            try:
                return win32process.GetModuleFileNameEx(h, 0)
            finally:
                win32api.CloseHandle(h)
        except Exception:
            return None

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
                "app.launch", reason="executable_path is required on Windows"
            )
        proc = subprocess.Popen(
            [executable_path, *(args or [])],
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        return proc.pid

    # ── Clipboard ────────────────────────────────────────────────────

    def clipboard_read(self, *, format: str = "text") -> ClipboardReadResult:
        if not self._clipboard_enabled:
            raise FeatureNotSupportedError("clipboard.read", reason="--enable-clipboard not set")
        if format != "text":
            raise FeatureNotSupportedError(
                "clipboard.read", reason=f"format {format!r} not supported"
            )

        snapshot = self._clipboard_snapshot()
        if snapshot is None:
            raise ClipboardConcealedError(
                "concealed_type",
                marker="unknown",
                message="the clipboard could not be read consistently",
            )
        text, seq, formats = snapshot
        # Concealed check first; formats that can't be read count as private.
        if formats is None or any(CF_PRIVATEFIRST <= fmt <= CF_PRIVATELAST for fmt in formats):
            raise ClipboardConcealedError("concealed_type", marker="CF_PRIVATE")
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
        try:
            import win32clipboard  # type: ignore[import-not-found]
            import win32con  # type: ignore[import-not-found]

            seq = self._clipboard_sequence()
            win32clipboard.OpenClipboard()
            try:
                if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                    data = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
                    return (str(data) if data is not None else None), seq
                if win32clipboard.IsClipboardFormatAvailable(win32con.CF_TEXT):
                    data = win32clipboard.GetClipboardData(win32con.CF_TEXT)
                    if isinstance(data, bytes):
                        return data.decode("utf-8", errors="replace"), seq
                    return (str(data) if data is not None else None), seq
                return None, seq
            finally:
                win32clipboard.CloseClipboard()
        except Exception:
            return None, None

    def _write_clipboard_text(self, content: str) -> int | None:
        """Write `content` and return the new sequence number; raise if the
        text did not land, so nothing acknowledges or pastes a stale clipboard."""
        try:
            import win32clipboard  # type: ignore[import-not-found]
            import win32con  # type: ignore[import-not-found]

            win32clipboard.OpenClipboard()
            try:
                win32clipboard.EmptyClipboard()
                win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, content)
            finally:
                win32clipboard.CloseClipboard()
        except Exception as exc:
            raise ClipboardWriteError(str(exc)) from exc
        return self._clipboard_sequence()

    def _clipboard_sequence(self) -> int | None:
        try:
            import ctypes

            return int(
                ctypes.windll.user32.GetClipboardSequenceNumber()  # type: ignore[attr-defined]
            )
        except Exception:
            return None

    def _clipboard_formats(self) -> set[int] | None:
        """Every format on the clipboard, or None if they can't be read."""
        try:
            import win32clipboard  # type: ignore[import-not-found]

            win32clipboard.OpenClipboard()
            try:
                formats: set[int] = set()
                fmt = win32clipboard.EnumClipboardFormats(0)
                while fmt != 0:
                    formats.add(fmt)
                    fmt = win32clipboard.EnumClipboardFormats(fmt)
                return formats
            finally:
                win32clipboard.CloseClipboard()
        except Exception:
            return None

    def _type(self, text: str, *, paste: bool, preserve_clipboard: bool) -> None:
        import pyautogui  # type: ignore[import-untyped]

        if paste and paste_is_worth_it(text) and self._clipboard_enabled:
            if not preserve_clipboard:
                self._do_paste(text, restore=None)
                return
            # The restore puts back text only, so paste only with that text in
            # hand. A clipboard holding anything else (an image, a CF_PRIVATE
            # secret), one that can't be read, or an empty one (clearing it
            # after the paste could beat a slow app to the text) is left alone
            # and the text typed.
            snapshot = self._clipboard_snapshot()
            if snapshot is not None:
                previous, _, formats = snapshot
                if (
                    formats is not None
                    and formats <= _TEXT_CLIPBOARD_FORMATS
                    and previous is not None
                ):
                    self._do_paste(text, restore=previous)
                    return
        ensure_typeable(text)
        pyautogui.write(text, interval=0.02)

    def _clipboard_snapshot(self) -> tuple[str | None, int | None, set[int] | None] | None:
        """The text, sequence number and formats as one reading. They come
        from separate clipboard sessions, so the sequence number and text
        must read the same after the formats as before them, or a copy
        landed in between; retried a few times, then None."""
        for _ in range(3):
            text, seq = self._read_clipboard_text()
            formats = self._clipboard_formats()
            again, seq_again = self._read_clipboard_text()
            # A sequence number that couldn't be read proves nothing, and two
            # failed reads would otherwise look alike.
            if seq is not None and seq == seq_again and text == again:
                return text, seq, formats
        return None

    def _do_paste(self, text: str, *, restore: str | None) -> None:
        """Paste `text`, then put `restore` back unless something else has
        written the clipboard since. With `restore=None` the text stays."""
        import pyautogui  # type: ignore[import-untyped]

        try:
            written_seq = self._write_clipboard_text(text)
        except ClipboardWriteError:
            # EmptyClipboard may have gone through before the write failed,
            # taking the user's text with it.
            if restore is not None:
                try:
                    self._write_clipboard_text(restore)
                except ClipboardWriteError:
                    logger.warning(
                        "Could not restore the clipboard after a failed write", exc_info=True
                    )
            raise
        self.clipboard_writeback.record_write(text, sequence=written_seq)
        try:
            pyautogui.hotkey("ctrl", "v")
        finally:
            # Also when the hotkey raised (pyautogui's fail-safe fires before
            # any key goes out): the user's text is already overwritten.
            if restore is not None:
                self._restore_after_paste(text, restore, written_seq)

    def _restore_after_paste(self, text: str, restore: str, written_seq: int | None) -> None:
        time.sleep(0.05)
        current, current_seq = self._read_clipboard_text()
        # Restore only if the clipboard still holds exactly our write. A
        # changed sequence number means the user copied during the paste;
        # EmptyClipboard and SetClipboardData each advance it, so compare with
        # what our write left behind. That number is read after CloseClipboard,
        # so a copy landing in between counts as ours: the text check catches
        # it, whenever Windows advances the number.
        if written_seq is not None and current_seq == written_seq and current == text:
            # The restored contents are the user's, not ours, so they are
            # not recorded as a writeback: a writeback-only read refuses them.
            try:
                self._write_clipboard_text(restore)
            except ClipboardWriteError:
                logger.warning("Could not restore the clipboard after pasting", exc_info=True)

    # ── Permissions ──────────────────────────────────────────────────

    def permissions_check(self, permissions: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        elevated = self._is_elevated()
        for p in permissions:
            if p in ("screen_recording", "accessibility", "input_monitoring"):
                # No Windows-equivalent TCC keys; reflect process elevation
                # which is the closest "can we drive this window" gate.
                out[p] = "granted" if elevated else "unknown"
            else:
                out[p] = "not_applicable"
        return out

    def _is_elevated(self) -> bool:
        try:
            import ctypes

            return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
        except Exception:
            return False


__all__ = ["WindowsBackend"]
