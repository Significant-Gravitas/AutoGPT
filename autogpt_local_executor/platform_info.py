"""
Per-OS detection helpers.

Implements the tables in docs/CROSS_PLATFORM.md:
- "HELLO.platform" detection
- "HELLO.arch" detection / normalization
- "Machine ID source"
- "Default allowed_root"
- "Audit log location"
- Capability probes (pyautogui / ollama / pyserial / pyusb / RPi.GPIO)

Everything in this module is sync — the daemon calls it once at startup.
"""

from __future__ import annotations

import importlib.util
import os
import platform as _platform
import shutil
import socket
import subprocess
import sys
import uuid
from functools import lru_cache
from pathlib import Path

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
) -> list[str]:
    """Return the list of capabilities to advertise in HELLO.

    Defaults gate optional capabilities behind config flags so a shim never
    advertises something the user didn't opt in to.
    """
    caps: list[str] = []
    if enable_shell:
        caps.append("shell")
    caps.append("files")  # always present
    if enable_computer_use and _can_do_computer_use():
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


# The third-party modules each computer-use backend imports. The
# computer-use extra installs them, and tests/test_computer_use_requirements.py
# keeps the backends, this list and the extra in step.
_COMPUTER_USE_MODULES = ("pyautogui", "PIL", "mss")
_COMPUTER_USE_PLATFORM_MODULES = {
    "darwin": ("Quartz", "AppKit", "Cocoa", "ApplicationServices"),
    "windows": (
        "win32api",
        "win32clipboard",
        "win32con",
        "win32gui",
        "win32process",
        "win32ui",
    ),
}


def computer_use_modules(platform_name: str) -> tuple[str, ...]:
    return (*_COMPUTER_USE_MODULES, *_COMPUTER_USE_PLATFORM_MODULES.get(platform_name, ()))


def _can_do_computer_use() -> bool:
    """Advertise computer_use only when every module its backend imports is
    installed and, per CROSS_PLATFORM.md, not on Wayland, which blocks input
    injection.
    """
    plat = detect_platform()
    if not all(_module_available(name) for name in computer_use_modules(plat)):
        return False
    if plat in ("linux",):
        # Computer use needs an X display, and Wayland sessions can't inject
        # input even into XWayland apps — see CROSS_PLATFORM.md row.
        wayland = os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland" or bool(
            os.environ.get("WAYLAND_DISPLAY")
        )
        if wayland or not os.environ.get("DISPLAY"):
            return False
        if not _x11_display_opens():
            return False
    if plat == "wsl2":
        return False
    return True


def _x11_display_opens() -> bool:
    """Whether the X server DISPLAY names accepts a connection. A set DISPLAY
    can still name a server that has gone, such as a closed SSH forward.
    pyautogui connects through python-xlib, so the probe does too."""
    try:
        from Xlib.display import Display  # type: ignore[import-not-found]

        Display().close()
    except Exception:
        return False
    return True


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


@lru_cache(maxsize=64)
def is_case_insensitive_fs(path: str | os.PathLike[str]) -> bool:
    """Probe the filesystem that contains `path`.

    Per CROSS_PLATFORM.md: macOS / Windows default to insensitive, Linux
    to sensitive, but all three have toggles per volume / per directory.
    The cheapest reliable check is to probe `os.path.exists` of the
    case-toggled basename.
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
    try:
        # Create a temporary probe file; check whether its upper-cased name
        # resolves to the same inode.
        import tempfile

        with tempfile.NamedTemporaryFile(
            dir=probe_dir, prefix=".aple_case_probe_", delete=False
        ) as f:
            probe_path = Path(f.name)
        try:
            upper = probe_path.parent / probe_path.name.upper()
            return upper.exists()
        finally:
            try:
                probe_path.unlink()
            except OSError:
                pass
    except (OSError, PermissionError):
        return sys.platform in ("win32", "darwin")


# ── Screen resolution ────────────────────────────────────────────────────────


def detect_screen_resolution() -> tuple[int, int] | None:
    """Return (width, height) when computer_use is supported, else None."""
    if not _can_do_computer_use():
        return None
    try:
        import pyautogui  # type: ignore[import-untyped]

        size = pyautogui.size()
        return int(size[0]), int(size[1])
    except Exception:
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
    "available_recording_channels",
    "default_allowed_root",
    "default_audit_log_path",
    "detect_arch",
    "detect_capabilities",
    "detect_machine_id",
    "detect_platform",
    "detect_screen_resolution",
    "is_case_insensitive_fs",
    "resolve_shell",
]
