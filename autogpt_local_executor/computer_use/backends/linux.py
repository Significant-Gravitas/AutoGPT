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

# The documented modifier names, in pyautogui's: its X11 keys call Super
# (and its alias meta) `win`.
_KEY_ALIASES = {"super": "win", "meta": "win"}

# Clipboard targets holding plain text, which is all a text-only restore
# rebuilds. GTK and Qt list COMPOUND_TEXT with the others for any text copy.
# Kept in lowercase, as targets are compared: a MIME type's case doesn't count.
_TEXT_TARGETS = frozenset(
    {"utf8_string", "string", "text", "compound_text", "text/plain", "text/plain;charset=utf-8"}
)
# X11 selection owners list these whatever they hold; they carry no content.
_SELECTION_PROTOCOL_TARGETS = frozenset({"targets", "timestamp", "multiple", "save_targets"})


def _is_wayland() -> bool:
    return os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" or bool(
        os.environ.get("WAYLAND_DISPLAY")
    )


def _application_dirs() -> list[str]:
    """The XDG application directories, in the order gtk-launch searches."""
    data_home = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    data_dirs = os.environ.get("XDG_DATA_DIRS") or "/usr/local/share:/usr/share"
    return [
        os.path.join(base, "applications")
        for base in [data_home, *data_dirs.split(":")]
        if os.path.isabs(base)
    ]


def _desktop_entry_value(path: str, key: str) -> str | None:
    """A key's value in a .desktop file's [Desktop Entry] group, or None."""
    try:
        with open(path, encoding="utf-8", errors="replace") as entry:
            in_group = False
            for raw in entry:
                line = raw.strip()
                if line.startswith("[") and line.endswith("]"):
                    in_group = line == "[Desktop Entry]"
                elif in_group and not line.startswith("#") and "=" in line:
                    name, _, value = line.partition("=")
                    if name.strip() == key:
                        return value.strip()
    except OSError:
        return None
    return None


def _program_found(program: str) -> bool:
    """Whether `program` (a path, or a name looked up on PATH) can be run."""
    if os.path.isabs(program):
        return os.path.isfile(program) and os.access(program, os.X_OK)
    return shutil.which(program) is not None


def _desktop_file_id(path: str) -> str | None:
    """The desktop file ID of a file in an XDG application directory: its
    path below that directory, with "/" turned into "-", so
    applications/kde4/kcalc.desktop is kde4-kcalc.desktop. None for a file
    outside them."""
    for directory in _application_dirs():
        for base in dict.fromkeys((os.path.abspath(directory), os.path.realpath(directory))):
            relative = os.path.relpath(os.path.abspath(path), base)
            if relative != os.pardir and not relative.startswith(os.pardir + os.sep):
                return relative.replace(os.sep, "-")
    return None


def _files_with_desktop_id(directory: str, desktop_id: str) -> list[str]:
    """The files in `directory` the ID names: the file of that name, and
    files in its subdirectories, where any "-" in the ID can stand for a
    "/"."""
    found = []
    candidate = os.path.join(directory, desktop_id)
    if os.path.isfile(candidate):
        found.append(candidate)
    for index, char in enumerate(desktop_id):
        if char == "-":
            subdirectory = os.path.join(directory, desktop_id[:index])
            if os.path.isdir(subdirectory):
                found.extend(_files_with_desktop_id(subdirectory, desktop_id[index + 1 :]))
    return found


def _gtk_launch_id(path: str) -> str | None:
    """The desktop ID that gtk-launch starts this very file by, or None.
    gtk-launch starts the first file with the ID in the application
    directories, so one in an earlier directory shadows it, two in the same
    directory leave it to the order the directory lists them in, and a file
    outside them can't be reached by an ID at all."""
    desktop_id = _desktop_file_id(path)
    if desktop_id is None:
        return None
    for directory in _application_dirs():
        found = _files_with_desktop_id(directory, desktop_id)
        if found:
            try:
                same = all(os.path.samefile(candidate, path) for candidate in found)
            except OSError:
                return None
            return desktop_id if same else None
    return None


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
        if self._clipboard_enabled and self._has_clipboard_tool():
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
        if self._clipboard_enabled and self._has_clipboard_tool():
            if self._has_clipboard_read_tool():
                feats.append("clipboard.read")
            feats.append("clipboard.write")
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
            scroll = pyautogui.hscroll if d in ("left", "right") else pyautogui.scroll
            if coordinate:
                scroll(sign * amount, x=coordinate[0], y=coordinate[1])
            else:
                scroll(sign * amount)
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
        return _KEY_ALIASES.get(name.lower(), name)

    @staticmethod
    def _chord(key: str) -> list[str]:
        import pyautogui  # type: ignore[import-untyped]

        return chord_keys(key, _KEY_ALIASES, known=getattr(pyautogui, "KEY_NAMES", None))

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
    ) -> int | None:
        if not executable_path:
            raise FeatureNotSupportedError(
                "app.launch", reason="executable_path is required on Linux"
            )
        if executable_path.endswith(".desktop"):
            if args:
                raise FeatureNotSupportedError(
                    "app.launch", reason="args can't be passed to a .desktop file"
                )
            return self._launch_desktop_file(executable_path)
        proc = subprocess.Popen([executable_path, *(args or [])], start_new_session=True)
        return proc.pid

    @staticmethod
    def _launch_desktop_file(path: str) -> None:
        """Start a .desktop file with gtk-launch when its desktop ID leads to
        this very file, else with dex, which takes the file's path. Either
        helper exits once it has started the app, so there is no app pid to
        report: the helper's names a process already gone."""
        # dex can exit 0 for a file that can't start, so what can be checked
        # is checked first.
        if not os.path.isfile(path):
            raise FeatureNotSupportedError(
                "app.launch", reason=f"there is no .desktop file at {path}"
            )
        try_exec = _desktop_entry_value(path, "TryExec")
        if try_exec and not _program_found(try_exec):
            raise FeatureNotSupportedError(
                "app.launch", reason=f"its TryExec program {try_exec!r} isn't installed"
            )
        has_gtk_launch = shutil.which("gtk-launch") is not None
        desktop_id = _gtk_launch_id(path) if has_gtk_launch else None
        if desktop_id is not None:
            command = ["gtk-launch", desktop_id.removesuffix(".desktop")]
        elif shutil.which("dex"):
            command = ["dex", path]
        elif has_gtk_launch:
            raise FeatureNotSupportedError(
                "app.launch",
                reason="gtk-launch finds a desktop file by its ID in the XDG application "
                "directories, where this one's ID leads to another file or to none; "
                "install dex to launch it by its path",
            )
        else:
            raise FeatureNotSupportedError(
                "app.launch", reason=".desktop launch needs gtk-launch or dex"
            )
        proc = subprocess.Popen(command, start_new_session=True)
        # The helper exits once it has started the app, with a failing status
        # if it couldn't.
        try:
            status = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            return None  # still starting it: not a failure
        if status != 0:
            raise FeatureNotSupportedError(
                "app.launch", reason=f"{' '.join(command)} exited with status {status}"
            )
        return None

    # ── Clipboard ────────────────────────────────────────────────────

    def clipboard_read(self, *, format: str = "text") -> ClipboardReadResult:
        if not self._clipboard_enabled:
            raise FeatureNotSupportedError("clipboard.read", reason="--enable-clipboard not set")
        if format != "text":
            raise FeatureNotSupportedError(
                "clipboard.read", reason=f"format {format!r} not supported"
            )
        if not self._has_clipboard_read_tool():
            raise FeatureNotSupportedError(
                "clipboard.read", reason="reading the clipboard needs xclip, or wl-paste on Wayland"
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

    def _has_clipboard_tool(self) -> bool:
        """Whether the tool clipboard reads and writes go through is installed:
        wl-copy in a Wayland session, xclip otherwise. wl-copy alone can't
        reach an X11 clipboard."""
        return (self._wayland and self._has_wlcopy) or self._has_xclip

    def _has_clipboard_read_tool(self) -> bool:
        """Whether a tool _read_clipboard_text reads with is installed:
        wl-paste in a Wayland session, else xclip."""
        wl_paste = self._wayland and self._has_wlcopy and shutil.which("wl-paste") is not None
        return wl_paste or self._has_xclip

    def _write_clipboard_text(self, content: str) -> int | None:
        """Write `content`; raise if it did not land, so nothing acknowledges
        or pastes a stale clipboard. The CLI tools expose no sequence number."""
        if self._wayland and self._has_wlcopy:
            command = ["wl-copy"]
        elif self._has_xclip:
            command = ["xclip", "-selection", "clipboard", "-i"]
        else:
            raise ClipboardWriteError("neither wl-copy nor xclip is available")
        try:
            subprocess.run(command, input=content.encode("utf-8"), check=True, timeout=2)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ClipboardWriteError(str(exc)) from exc
        return None

    def _type(self, text: str, *, paste: bool, preserve_clipboard: bool) -> None:
        import pyautogui  # type: ignore[import-untyped]

        if (
            paste
            and paste_is_worth_it(text)
            and self._clipboard_enabled
            and self._has_clipboard_tool()
        ):
            if not preserve_clipboard:
                self._do_paste(text, restore=None)
                return
            # The restore puts back text only, so paste only with that text in
            # hand and nothing else on the clipboard. An empty clipboard, one
            # the tools can't read as text, or one also holding HTML or an
            # image is left alone and the text typed.
            previous, _ = self._read_clipboard_text()
            if previous is not None and self._clipboard_holds_only_text():
                self._do_paste(text, restore=previous)
                return
        ensure_typeable(text)
        pyautogui.write(text, interval=0.02)

    def _clipboard_holds_only_text(self) -> bool:
        """Whether every target the clipboard offers is text. False when the
        targets can't be listed."""
        if self._wayland and self._has_wlcopy and shutil.which("wl-paste"):
            command = ["wl-paste", "--list-types"]
        elif self._has_xclip:
            command = ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"]
        else:
            return False
        try:
            out = subprocess.check_output(command, timeout=2)
        except (OSError, subprocess.SubprocessError):
            return False
        targets = {
            "".join(line.split()).lower()
            for line in out.decode("utf-8", errors="replace").splitlines()
        }
        targets -= _SELECTION_PROTOCOL_TARGETS | {""}
        return bool(targets) and targets <= _TEXT_TARGETS

    def _do_paste(self, text: str, *, restore: str | None) -> None:
        """Paste `text`, then put `restore` back. With `restore=None` the
        text stays."""
        import pyautogui  # type: ignore[import-untyped]

        self._write_clipboard_text(text)
        self.clipboard_writeback.record_write(text)
        try:
            pyautogui.hotkey("ctrl", "v")
        finally:
            # Also when the hotkey raised (pyautogui's fail-safe fires before
            # any key goes out): the user's text is already overwritten.
            if restore is not None:
                self._restore_after_paste(text, restore)

    def _restore_after_paste(self, pasted: str, restore: str) -> None:
        time.sleep(0.05)
        # The CLI tools expose no sequence number, so the text stands in for
        # it: a clipboard that no longer holds what was pasted has had
        # something copied to it since, and keeps that. (Copying the very
        # same text again can't be told apart.)
        current, _ = self._read_clipboard_text()
        if current != pasted:
            return
        # The restored contents are the user's, not ours, so they are not
        # recorded as a writeback: a writeback-only read refuses them.
        try:
            self._write_clipboard_text(restore)
        except ClipboardWriteError:
            logger.warning("Could not restore the clipboard after pasting", exc_info=True)

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
