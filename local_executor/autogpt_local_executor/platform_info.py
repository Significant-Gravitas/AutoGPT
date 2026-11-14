"""
Per-OS detection helpers.

Implements the tables in docs/CROSS_PLATFORM.md:
- "HELLO.platform" detection
- "HELLO.arch" detection / normalization
- "Machine ID source"
- "Default allowed_root"
- "Audit log location"
- Capability probes (cua-driver / ollama / pyserial / pyusb / RPi.GPIO)

Everything in this module is sync — the daemon calls it once at startup.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import logging
import os
import platform as _platform
import shutil
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Platform / arch ──────────────────────────────────────────────────────────


def detect_platform() -> str:
    """Returns one of: darwin | linux | windows | wsl2."""
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "linux":
        try:
            with open("/proc/version", encoding="utf-8", errors="replace") as f:
                if "microsoft" in f.read().lower():
                    return "wsl2"
        except OSError:
            pass
        return "linux"
    # Anything else — surface honestly so callers can fail loudly.
    return sys.platform


def detect_arch() -> str:
    """Normalize platform.machine() to x86_64 | arm64.

    Raises ValueError on anything else (the platform side will reject the
    HELLO with UNSUPPORTED_ARCH, per docs/PROTOCOL.md).
    """
    raw = _platform.machine()
    low = raw.lower()
    if low in ("x86_64", "amd64"):
        return "x86_64"
    if low in ("arm64", "aarch64"):
        return "arm64"
    raise ValueError(f"Unsupported architecture: {raw!r}")


# ── Capabilities ─────────────────────────────────────────────────────────────


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def detect_capabilities(
    *,
    enable_shell: bool = False,
    enable_computer_use: bool = False,
    enable_local_llm: bool = False,
    enable_hardware: bool = False,
    enable_recording: bool = False,
    cua_driver_path: str | None = None,
    driver_found: bool | None = None,
) -> list[str]:
    """Return the list of capabilities to advertise in HELLO.

    Defaults gate optional capabilities behind config flags so a shim never
    advertises something the user didn't opt in to. `driver_found` is
    can_do_computer_use's.
    """
    caps: list[str] = []
    if enable_shell:
        caps.append("shell")
    caps.append("files")  # always present
    if enable_computer_use and can_do_computer_use(cua_driver_path, driver_found=driver_found):
        caps.append("computer_use")
    if enable_hardware:
        if _module_available("serial"):
            caps.append("hardware_serial")
        if _module_available("usb"):
            caps.append("hardware_usb")
        if _module_available("RPi.GPIO"):
            caps.append("hardware_gpio")
    # Workflow recording remains a design preview. Its capture and
    # interpretation pipeline isn't safe to expose end to end, so this
    # capability is never advertised even when the preview flag is set.
    return caps


def available_recording_channels() -> list[str]:
    """Return no channels until recording is production-ready end to end."""
    return []


def can_do_computer_use(
    cua_driver_path: str | None = None, *, driver_found: bool | None = None
) -> bool:
    """Whether computer use can be offered: cua-driver is installed and, on
    Linux, has a display it can drive (linux_display_problem). WSL2 has none
    for Windows apps. The daemon advertises it only once the driver starts.
    `driver_found` says whether the driver is installed, for a caller that
    looked for it itself (HELLO, which bounds the look); otherwise it is
    looked for here."""
    from .computer_use.driver import find_driver_binary

    plat = detect_platform()
    if plat == "wsl2":
        return False
    if driver_found is None:
        driver_found = find_driver_binary(cua_driver_path) is not None
    if not driver_found:
        return False
    if plat == "linux":
        session_display.follow()
        return linux_display_problem() is None
    return True


def computer_use_waits_for_a_display() -> bool:
    """Whether computer use is off only until the desktop exports its
    display: the Linux systemd service, started without one, that has none
    yet (SessionDisplay). The daemon then says HELLO again once it is up."""
    return (
        detect_platform() == "linux"
        and session_display.following()
        and linux_display_problem() is not None
    )


def linux_display_problem(environ: Mapping[str, str] | None = None) -> str | None:
    """Why cua-driver can't drive this Linux session's display, or None.

    cua-driver 0.38 drives X11 through DISPLAY, which XWayland provides on a
    Wayland desktop. It uses its own Wayland backend only when
    CUA_DRIVER_RS_ENABLE_WAYLAND is set to anything but empty, 0 or false
    (the driver inherits it), so WAYLAND_DISPLAY alone is not enough."""
    env = os.environ if environ is None else environ
    if env.get("DISPLAY") or linux_native_wayland(env):
        return None
    if not env.get("WAYLAND_DISPLAY"):
        return "neither DISPLAY nor WAYLAND_DISPLAY is set in the shim's environment"
    return (
        "WAYLAND_DISPLAY is set but DISPLAY isn't: cua-driver needs XWayland's DISPLAY, "
        "or CUA_DRIVER_RS_ENABLE_WAYLAND=1 for its experimental Wayland backend"
    )


def linux_native_wayland(environ: Mapping[str, str] | None = None) -> bool:
    """Whether cua-driver uses its own Wayland backend: a Wayland session
    with CUA_DRIVER_RS_ENABLE_WAYLAND set to anything but empty, 0 or false.
    Otherwise it drives X11 alone, through DISPLAY."""
    env = os.environ if environ is None else environ
    opt_in = env.get("CUA_DRIVER_RS_ENABLE_WAYLAND", "").strip()
    return (
        bool(env.get("WAYLAND_DISPLAY"))
        and bool(opt_in)
        and opt_in != "0"
        and opt_in.lower() != "false"
    )


# ── The desktop's display, for the systemd service ──────────────────────────

# What a desktop session exports to the systemd user manager once it is up
# (GNOME and KDE do, and so do Xsession and xinit on most distributions), and
# what cua-driver reads to find and drive the display.
SESSION_DISPLAY_VARIABLES = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XAUTHORITY",
    "XDG_SESSION_TYPE",
    "XDG_CURRENT_DESKTOP",
    "XDG_SESSION_DESKTOP",
    "DESKTOP_SESSION",
    "SWAYSOCK",
    "HYPRLAND_INSTANCE_SIGNATURE",
)
_MANAGER_READ_EVERY_SECONDS = 10.0
_MANAGER_TIMEOUT_SECONDS = 2.0
# How long `SessionDisplay.follow` waits for the read under way: systemctl's
# run and a second more. A read that takes longer (systemctl looked up on a
# stalled PATH mount, say) goes on in its own thread, and the caller with
# the display as it is.
MANAGER_WAIT_SECONDS = _MANAGER_TIMEOUT_SECONDS + 1.0


class SessionDisplay:
    """The desktop's display for the shim's systemd service.

    The user manager starts the service at login, before the desktop has
    exported its display to the manager, and a service's environment is
    fixed when it starts: the shim had no DISPLAY or WAYLAND_DISPLAY for as
    long as it ran, so computer use stayed off. When the service started
    without either, `follow` takes them, and the session variables that go
    with them, from the manager's environment as it is now, and follows them
    after that: a new login brings a new display and, on GNOME, a new
    XAUTHORITY; a session that ended takes them away. The session's values
    stand in for any the service started with (an XAUTHORITY an earlier
    session left doesn't open the new display), which come back once the
    session ends, and so does a session's lack of an XAUTHORITY: Xlib then
    reads ~/.Xauthority. A shim run by hand, or one that started with a
    display, keeps its own environment.

    The manager is read in a thread of its own, one read at a time and
    apart from the lock (refresh), and each caller waits for the read under
    way no longer than it can: a read that stalls holds no caller past its
    wait, and no thread but its own."""

    def __init__(
        self,
        environ: MutableMapping[str, str] | None = None,
        *,
        read: Callable[[], Mapping[str, str] | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._env = os.environ if environ is None else environ
        self._read = read or _manager_environment
        self._clock = clock
        self._lock = threading.Lock()
        # The session variables this process started with, and those it took.
        self._own: dict[str, str] | None = None
        self._taken: set[str] = set()
        self._read_at: float | None = None
        # The manager's read under way, or the last (refresh).
        self._reading: concurrent.futures.Future[bool] | None = None

    def _started_with(self) -> dict[str, str]:
        return {key: value for key in SESSION_DISPLAY_VARIABLES if (value := self._env.get(key))}

    def following(self) -> bool:
        """Whether `follow` takes this process's display from the manager:
        the systemd service, started without one."""
        with self._lock:
            if self._env.get("AUTOGPT_SHIM_SUPERVISOR") != "systemd":
                return False
            own = self._own if self._own is not None else self._started_with()
            return not {"DISPLAY", "WAYLAND_DISPLAY"} & own.keys()

    def follow(self, timeout: float | None = None) -> bool:
        """Bring this process's display variables in line with the
        manager's, reading them at most every 10 s. Returns whether any
        changed. Waits for the read (refresh) at most `timeout` seconds,
        MANAGER_WAIT_SECONDS if not given; past that it returns False and
        leaves the read to finish."""
        try:
            return self.refresh().result(MANAGER_WAIT_SECONDS if timeout is None else timeout)
        except TimeoutError:
            return False

    def refresh(self) -> concurrent.futures.Future[bool]:
        """The manager's read `follow` waits for, as a future of whether any
        variable changed: the read under way, else a new one when one is
        due, else an answer already in (False).

        The read runs in a thread of its own, one at a time, apart from the
        lock. It was made under the lock, systemctl's lookup on PATH with
        it, and a stalled mount on PATH (NFS, FUSE) holds that lookup for as
        long as it stalls: every later caller waited for the lock, HELLO's
        `following` too, each in one of the worker threads every session's
        file and audit work shares, until they were all held. A caller now
        waits for the one read as long as it can, one on the event loop
        with asyncio.wrap_future, in no thread at all, and the read goes on
        without it."""
        with self._lock:
            if self._env.get("AUTOGPT_SHIM_SUPERVISOR") != "systemd":
                return _answered(False)
            if self._own is None:
                self._own = self._started_with()
            if {"DISPLAY", "WAYLAND_DISPLAY"} & self._own.keys():
                return _answered(False)
            reading = self._reading
            if reading is not None and not reading.done():
                return reading
            now = self._clock()
            if self._read_at is not None and now - self._read_at < _MANAGER_READ_EVERY_SECONDS:
                return _answered(False)
            self._read_at = now
            reading = self._reading = concurrent.futures.Future()
            # Running from here on, so a caller that stops waiting
            # (asyncio.wrap_future passes its cancel on) can't cancel the
            # read for the others.
            reading.set_running_or_notify_cancel()
        try:
            threading.Thread(
                target=self._read_into,
                args=(reading,),
                name="autogpt-shim-session-display",
                daemon=True,
            ).start()
        except RuntimeError as exc:
            # No thread to read in: nothing changed, and the next read is
            # tried in 10 s.
            logger.warning("Could not read the systemd user manager's environment: %s", exc)
            reading.set_result(False)
        return reading

    def _read_into(self, reading: concurrent.futures.Future[bool]) -> None:
        """Read the manager and take its display variables, in the read's
        thread, answering `reading` whatever happens: a read never answered
        would be waited for by every caller, and none would start again."""
        try:
            manager = self._read()
            with self._lock:
                changed = self._take(manager)
        except BaseException as exc:
            reading.set_exception(exc)
        else:
            reading.set_result(changed)

    def _take(self, manager: Mapping[str, str] | None) -> bool:
        """Take the display variables from `manager`, the manager's
        environment (None if it couldn't be read). Returns whether any
        changed. Called with the lock held."""
        if manager is None or self._own is None:
            return False
        usable = {
            key: value for key in SESSION_DISPLAY_VARIABLES if _usable(value := manager.get(key))
        }
        if "DISPLAY" not in usable and "WAYLAND_DISPLAY" not in usable:
            usable = {}
        changed = False
        for key in SESSION_DISPLAY_VARIABLES:
            if key in usable:
                # The session's, even over one this process started
                # with: an XAUTHORITY an earlier session left doesn't
                # open this session's display.
                value = usable[key]
                self._taken.add(key)
            elif key == "XAUTHORITY" and usable:
                # Nor for a session without one: Xlib reads only the
                # file XAUTHORITY names, and ~/.Xauthority without it.
                value = None
                self._taken.add(key)
            elif key in self._taken:
                # Gone with the session: this process's own comes back.
                value = self._own.get(key)
                self._taken.discard(key)
            else:
                continue
            if self._env.get(key) != value:
                if value is None:
                    self._env.pop(key, None)
                else:
                    self._env[key] = value
                changed = True
        if changed:
            logger.info(
                "Display from the systemd user manager: DISPLAY=%s WAYLAND_DISPLAY=%s",
                self._env.get("DISPLAY"),
                self._env.get("WAYLAND_DISPLAY"),
            )
        return changed


def _answered(changed: bool) -> concurrent.futures.Future[bool]:
    """A SessionDisplay read's answer, already in: no read was due."""
    answer: concurrent.futures.Future[bool] = concurrent.futures.Future()
    answer.set_result(changed)
    return answer


def session_display_values(environ: Mapping[str, str] | None = None) -> tuple[str | None, ...]:
    """This process's display variables, to tell when they have changed."""
    env = os.environ if environ is None else environ
    return tuple(env.get(key) for key in SESSION_DISPLAY_VARIABLES)


def _usable(value: str | None) -> bool:
    if not value:
        return False
    return len(value) <= 4096 and "\x00" not in value and "\n" not in value


# systemctl's path, once found on PATH (_systemctl).
_systemctl_path: str | None = None


def _systemctl() -> str | None:
    """systemctl's path, looked up on PATH until it is found and then kept:
    shutil.which stats a file in each PATH folder, and a stalled mount there
    (NFS, FUSE) holds it for as long as it stalls. The service's PATH stays
    as it started."""
    global _systemctl_path
    if _systemctl_path is None:
        _systemctl_path = shutil.which("systemctl")
    return _systemctl_path


def _manager_environment() -> dict[str, str] | None:
    """The systemd user manager's environment, or None."""
    systemctl = _systemctl()
    if systemctl is None:
        return None
    try:
        completed = subprocess.run(
            [systemctl, "--user", "show-environment"],
            capture_output=True,
            timeout=_MANAGER_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    env: dict[str, str] = {}
    for line in completed.stdout.decode("utf-8", "replace").splitlines():
        key, sep, value = line.partition("=")
        decoded = _manager_value(value) if sep and key else None
        if decoded is not None:
            env[key] = decoded
    return env


# What a backslash and the character after it stand for in a value
# systemctl quotes.
_QUOTED_ESCAPES = {
    "a": 7,
    "b": 8,
    "f": 12,
    "n": 10,
    "r": 13,
    "t": 9,
    "v": 11,
    "\\": 92,
    "'": 39,
    '"': 34,
}


def _manager_value(value: str) -> str | None:
    """A value as `systemctl show-environment` prints it: as it is, or, when
    it holds spaces or shell characters (an XAUTHORITY under a home folder
    with a space in its name, say), quoted as $'...' (systemd's
    shell_maybe_quote). Inside, a backslash escapes `\\`, `'` and control
    characters, and three octal digits stand for any other byte. Decoded
    here, without a shell; None for quoting systemctl doesn't write."""
    if not value.startswith("$'"):
        return value
    if len(value) < 3 or not value.endswith("'"):
        return None
    body = value[2:-1]
    out = bytearray()
    i = 0
    while i < len(body):
        char = body[i]
        if char == "'":
            return None
        if char != "\\":
            out += char.encode("utf-8")
            i += 1
            continue
        escaped = body[i + 1 : i + 2]
        octal = body[i + 1 : i + 4]
        if escaped and escaped in _QUOTED_ESCAPES:
            out.append(_QUOTED_ESCAPES[escaped])
            i += 2
        elif len(octal) == 3 and octal[0] in "0123" and all(d in "01234567" for d in octal):
            out.append(int(octal, 8))
            i += 4
        else:
            return None
    # As Linux reads an environment's bytes, whatever the OS running this.
    return bytes(out).decode("utf-8", "surrogateescape")


session_display = SessionDisplay()


# ── Machine ID ───────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def detect_machine_id() -> str:
    """Best-effort stable identifier. Per CROSS_PLATFORM.md table."""
    plat = detect_platform()
    try:
        if plat == "darwin":
            out = subprocess.check_output(
                ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            for line in out.splitlines():
                if "IOPlatformUUID" in line:
                    # Line looks like: "IOPlatformUUID" = "ABCDEF12-..."
                    parts = line.split("=", 1)
                    if len(parts) == 2:
                        val = parts[1].strip().strip('"').lower()
                        return val
        elif plat in ("linux", "wsl2"):
            for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
                try:
                    with open(candidate, encoding="utf-8") as f:
                        val = f.read().strip()
                    if val:
                        # /etc/machine-id is a bare 32-char hex; format as UUID.
                        return _format_as_uuid(val)
                except OSError:
                    continue
        elif plat == "windows":
            try:
                import winreg  # type: ignore[import-not-found]

                with winreg.OpenKey(  # type: ignore[attr-defined]
                    winreg.HKEY_LOCAL_MACHINE,  # type: ignore[attr-defined]
                    r"SOFTWARE\Microsoft\Cryptography",
                ) as key:
                    val, _ = winreg.QueryValueEx(key, "MachineGuid")  # type: ignore[attr-defined]
                    return str(val).lower()
            except OSError:
                pass
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        pass
    # Last resort: hostname + persisted random UUID per home dir.
    fallback_path = Path.home() / ".autogpt-local-executor-machine-id"
    try:
        if fallback_path.is_file():
            return fallback_path.read_text(encoding="utf-8").strip()
        val = f"{socket.gethostname()}-{uuid.uuid4()}"
        fallback_path.write_text(val, encoding="utf-8")
        return val
    except OSError:
        return f"{socket.gethostname()}-{uuid.uuid4()}"


def _format_as_uuid(hex_str: str) -> str:
    """Format a 32-char hex string as a UUID-with-dashes string."""
    h = hex_str.replace("-", "").lower()
    if len(h) == 32 and all(c in "0123456789abcdef" for c in h):
        return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"
    return hex_str


# ── Default paths ────────────────────────────────────────────────────────────


def default_allowed_root() -> Path:
    """Per CROSS_PLATFORM.md "Default allowed_root" table."""
    plat = detect_platform()
    if plat == "darwin":
        return Path.home() / "Documents" / "autogpt-workspace"
    if plat == "windows":
        return Path.home() / "autogpt-workspace"
    # linux / wsl2
    return Path.home() / "autogpt-workspace"


def default_audit_log_path() -> Path:
    """Per CROSS_PLATFORM.md "Audit log location" table."""
    plat = detect_platform()
    if plat == "darwin":
        return Path.home() / "Library" / "Logs" / "autogpt-local-executor" / "audit.log"
    if plat == "windows":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "autogpt-local-executor" / "logs" / "audit.log"
    # linux / wsl2
    xdg_state = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state) if xdg_state else Path.home() / ".local" / "state"
    return base / "autogpt-local-executor" / "audit.log"


# ── Filesystem case-sensitivity probe ────────────────────────────────────────


def is_case_insensitive_fs(path: str | os.PathLike[str]) -> bool:
    """Probe the filesystem that contains `path`.

    Per CROSS_PLATFORM.md: macOS / Windows default to insensitive, Linux
    to sensitive, but all three have toggles per volume / per directory.
    One of the folder's own entries is looked up again with its letters'
    case swapped, which reads only (_folds_case_by_lookup). Where no entry
    serves (an empty folder, say), the OS's default answers.

    Nothing is written to tell. A folder with no entry to look up got a
    probe file made in it for a moment, which sync clients and file
    watchers saw, though FILE_LIST, the caller, is a pure read
    (PROTOCOL.md). Nor is an answer kept: an empty folder's default would
    stand after the folder had an entry to look up, where the probe file
    had told, and looking again costs an entry or two.
    """
    p = Path(path)
    # Walk up until we find something that exists, so we can probe its parent.
    probe_dir: Path
    if p.exists():
        probe_dir = p if p.is_dir() else p.parent
    else:
        probe_dir = p
        while probe_dir != probe_dir.parent and not probe_dir.exists():
            probe_dir = probe_dir.parent
        if not probe_dir.exists():
            # Fall back to OS default if we can't probe anything.
            return sys.platform in ("win32", "darwin")
    folds = _folds_case_by_lookup(probe_dir)
    if folds is None:
        return sys.platform in ("win32", "darwin")
    return folds


# Swaps the case of ASCII letters alone, which every filesystem that folds
# case folds alike.
_ASCII_CASE_SWAP = str.maketrans(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
)
# The most entries of a folder looked at for one to look up again.
_CASE_LOOKUP_ENTRIES = 100


def _folds_case_by_lookup(folder: Path) -> bool | None:
    """Whether `folder`'s filesystem folds case, from one of its entries
    looked up again with its letters' case swapped: the same file where it
    folds, none (or another file) where it keeps case. None if no entry
    serves: none has an ASCII letter in its name, or a hard link could hold
    both names, or no file identity can be read."""
    try:
        with os.scandir(folder) as entries:
            for looked_at, entry in enumerate(entries):
                if looked_at >= _CASE_LOOKUP_ENTRIES:
                    return None
                swapped = entry.name.translate(_ASCII_CASE_SWAP)
                if swapped == entry.name:
                    continue
                try:
                    # os.stat, not the entry's: Windows leaves its inode out.
                    own = os.stat(entry.path, follow_symlinks=False)
                except OSError:
                    continue
                if not own.st_ino or (not stat.S_ISDIR(own.st_mode) and own.st_nlink > 1):
                    continue
                try:
                    other = os.stat(os.path.join(folder, swapped), follow_symlinks=False)
                except FileNotFoundError:
                    return False
                except OSError:
                    continue
                return (other.st_dev, other.st_ino) == (own.st_dev, own.st_ino)
    except OSError:
        return None
    return None


# ── Shell detection (used by EXECUTE_COMMAND handler) ────────────────────────


def resolve_shell(shell: str) -> tuple[str, list[str]] | None:
    """Resolve a wire `shell` value to (executable, prefix_args).

    Returns None if the requested shell isn't available on this OS — the
    handler turns that into a SHELL_NOT_AVAILABLE error.

    `shell` is one of: auto, bash, sh, zsh, pwsh, powershell, cmd.
    """
    plat = detect_platform()
    candidates: dict[str, tuple[str, list[str]]]
    if plat == "windows":
        candidates = {
            "auto": ("cmd.exe", ["/C"]),
            "cmd": ("cmd.exe", ["/C"]),
            "bash": ("bash", ["-c"]),  # Git Bash / WSL
            "pwsh": ("pwsh", ["-NoLogo", "-NoProfile", "-Command"]),
            "powershell": (
                "powershell.exe",
                ["-NoLogo", "-NoProfile", "-Command"],
            ),
        }
    else:
        candidates = {
            "auto": ("bash", ["-c"]),
            "bash": ("bash", ["-c"]),
            "sh": ("sh", ["-c"]),
            "zsh": ("zsh", ["-c"]),
            "pwsh": ("pwsh", ["-NoLogo", "-NoProfile", "-Command"]),
        }

    if shell == "auto" and plat != "windows":
        # bash if available else sh
        bash = shutil.which("bash")
        if bash:
            return (bash, ["-c"])
        sh = shutil.which("sh")
        if sh:
            return (sh, ["-c"])
        return None

    entry = candidates.get(shell)
    if entry is None:
        return None
    exe, args = entry
    found = shutil.which(exe)
    if found is None:
        return None
    return (found, args)


__all__ = [
    "MANAGER_WAIT_SECONDS",
    "SESSION_DISPLAY_VARIABLES",
    "SessionDisplay",
    "can_do_computer_use",
    "available_recording_channels",
    "default_allowed_root",
    "default_audit_log_path",
    "detect_arch",
    "detect_capabilities",
    "detect_machine_id",
    "detect_platform",
    "is_case_insensitive_fs",
    "linux_display_problem",
    "session_display",
    "session_display_values",
    "linux_native_wayland",
    "resolve_shell",
]
