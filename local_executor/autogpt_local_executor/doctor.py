"""
`autogpt-shim doctor` — preflight checks for the shim.

Per docs/COMPUTER_USE.md Q5:
- Probes the keychain, directory browser (or legacy allowed_root), display,
  and (on macOS) calls
  AXIsProcessTrustedWithOptions with the prompt option **on first
  invocation** so the consent dialog appears proactively.
- Returns exit 0 when everything required is healthy, 78
  (EX_CONFIG) when computer-use is requested but a prerequisite OS
  permission is missing.

Output is plain ASCII (no emoji) — the markers are:
  OK    everything good
  WARN  degraded but won't block (e.g. computer-use not requested)
  FAIL  required and missing
  INFO  context-only

Cross-OS: Linux/Windows variants reach the same exit codes; macOS is
where the heavy lifting (AXIsProcessTrusted + Screen Recording probe)
lives.
"""

from __future__ import annotations

import os
import secrets
import threading
from dataclasses import dataclass

from . import platform_info
from .config import ShimConfig
from .directory_browser import DirectoryBrowser

EX_CONFIG = 78


@dataclass
class CheckResult:
    marker: str  # "OK" | "WARN" | "FAIL" | "INFO"
    label: str
    detail: str
    blocking: bool = False  # True → contributes to non-zero exit


def run_doctor(config: ShimConfig) -> int:
    """Run all checks, print results, return shell exit code."""
    checks: list[CheckResult] = []
    if config.session_id and config.session_id.strip():
        checks.append(_check_allowed_root(config))
    else:
        checks.append(_check_directory_browser())
    checks.append(_check_keychain())
    plat = platform_info.detect_platform()
    if plat == "darwin":
        checks.extend(_check_macos(config))
    elif plat == "windows":
        checks.extend(_check_windows(config))
    elif plat in ("linux", "wsl2"):
        checks.extend(_check_linux(config, is_wsl2=(plat == "wsl2")))
    else:
        checks.append(
            CheckResult("WARN", "platform", f"unknown platform {plat!r}; doctor cannot probe")
        )
    checks.append(_check_display(config))
    _print_checks(checks)
    if any(c.blocking for c in checks):
        return EX_CONFIG
    return 0


def _print_checks(checks: list[CheckResult]) -> None:
    width = max(len(c.label) for c in checks) if checks else 0
    for c in checks:
        print(f"[{c.marker:<4}] {c.label.ljust(width)}  {c.detail}")


def _check_allowed_root(config: ShimConfig) -> CheckResult:
    p = config.allowed_root
    try:
        p.mkdir(parents=True, exist_ok=True)
        if not os.access(p, os.W_OK):
            return CheckResult(
                "FAIL",
                "allowed_root",
                f"{p} exists but is not writable",
                blocking=True,
            )
    except OSError as exc:
        return CheckResult("FAIL", "allowed_root", f"{p}: {exc}", blocking=True)
    return CheckResult("OK", "allowed_root", f"{p} exists, writable")


def _check_directory_browser() -> CheckResult:
    try:
        browser = DirectoryBrowser()
        listing = browser.list_directories(None, None)
        browser.reset()
        if not listing.entries:
            return CheckResult(
                "FAIL",
                "folder_browser",
                "no accessible host directory roots found",
                blocking=True,
            )
        return CheckResult(
            "OK",
            "folder_browser",
            f"{len(listing.entries)} accessible host root(s)",
        )
    except Exception as exc:
        return CheckResult("FAIL", "folder_browser", str(exc), blocking=True)


# A locked keychain can block on an unlock prompt; doctor must not hang on it.
KEYCHAIN_PROBE_SECONDS = 10.0


def _check_keychain() -> CheckResult:
    # Finding the backend can block as well (the Secret Service check makes
    # D-Bus calls), so the whole probe runs under the time limit.
    outcome: list[CheckResult] = []
    probe = threading.Thread(target=lambda: outcome.append(_probe_keychain()), daemon=True)
    probe.start()
    probe.join(KEYCHAIN_PROBE_SECONDS)
    if not outcome:
        return CheckResult(
            "WARN",
            "keychain",
            f"the keychain did not answer within {KEYCHAIN_PROBE_SECONDS:.0f} s; it may be locked",
        )
    return outcome[0]


def _probe_keychain() -> CheckResult:
    try:
        import keyring  # type: ignore[import-not-found]

        backend = keyring.get_keyring()
    except Exception as exc:
        return CheckResult(
            "WARN",
            "keychain",
            f"unavailable: {exc}. `autogpt-shim auth` needs a keychain to store "
            "sign-in tokens; only the audit key can fall back to a file, and only "
            "when AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE is set",
        )
    name = f"{type(backend).__module__}.{type(backend).__name__}"
    # keyring falls back to these when it finds no real keychain; they store
    # nothing.
    if name.startswith(("keyring.backends.fail.", "keyring.backends.null.")):
        return CheckResult(
            "WARN",
            "keychain",
            f"no usable keychain ({name}). `autogpt-shim auth` needs one to store sign-in tokens",
        )
    # Naming a backend says nothing about whether it stores anything, so store,
    # read and delete a probe.
    return _round_trip(keyring, name)


def _round_trip(keyring, name: str) -> CheckResult:
    from .auth import KEYCHAIN_SERVICE

    key = f"doctor-probe-{secrets.token_hex(4)}"
    try:
        keyring.set_password(KEYCHAIN_SERVICE, key, "ok")
        try:
            stored = keyring.get_password(KEYCHAIN_SERVICE, key)
        finally:
            try:
                keyring.delete_password(KEYCHAIN_SERVICE, key)
            except Exception:
                pass
    except Exception as exc:
        return CheckResult("WARN", "keychain", f"{name} could not store a value: {exc}")
    if stored != "ok":
        return CheckResult("WARN", "keychain", f"{name} did not return a value just stored in it")
    return CheckResult("OK", "keychain", name)


def _check_display(config: ShimConfig) -> CheckResult:
    try:
        from .computer_use import get_backend

        backend = get_backend(config)
        displays = backend.display_info()
        if not displays:
            return CheckResult("WARN", "display", "no displays detected")
        desc = " + ".join(f"{d.physical_size[0]}x{d.physical_size[1]}" for d in displays)
        return CheckResult("OK", "display", f"{len(displays)} display(s): {desc}")
    except Exception as exc:
        return CheckResult("WARN", "display", f"could not enumerate: {exc}")


# ── macOS ────────────────────────────────────────────────────────────────


def _check_macos(config: ShimConfig) -> list[CheckResult]:
    out: list[CheckResult] = []
    ax = _ax_trusted_with_prompt()
    if ax is None:
        out.append(
            CheckResult(
                "WARN",
                "Accessibility",
                "pyobjc/ApplicationServices not installed; cannot probe",
                blocking=config.enable_computer_use,
            )
        )
    elif ax:
        out.append(CheckResult("OK", "Accessibility", "granted"))
    else:
        out.append(
            CheckResult(
                "FAIL",
                "Accessibility",
                "not granted — System Settings → Privacy & Security → Accessibility, "
                "enable autogpt-shim",
                blocking=config.enable_computer_use,
            )
        )

    sc = _screen_recording_granted()
    if sc is None:
        out.append(
            CheckResult(
                "WARN",
                "Screen Recording",
                "Quartz not installed; cannot probe",
                blocking=config.enable_computer_use,
            )
        )
    elif sc:
        out.append(CheckResult("OK", "Screen Recording", "granted"))
    else:
        out.append(
            CheckResult(
                "FAIL",
                "Screen Recording",
                "not granted — System Settings → Privacy & Security → Screen Recording",
                blocking=config.enable_computer_use,
            )
        )

    if config.enable_computer_use:
        if ax and sc:
            out.append(CheckResult("OK", "computer_use", "ready"))
        else:
            missing = []
            if not ax:
                missing.append("Accessibility")
            if not sc:
                missing.append("Screen Recording")
            out.append(
                CheckResult(
                    "WARN",
                    "computer_use",
                    f"degraded — missing: {', '.join(missing)}",
                )
            )
    else:
        out.append(CheckResult("INFO", "computer_use", "not requested in config"))
    return out


def _ax_trusted_with_prompt() -> bool | None:
    """Call AXIsProcessTrustedWithOptions({"AXTrustedCheckOptionPrompt": True}).

    Proactively surfaces the consent dialog on first invocation. Returns
    None when pyobjc isn't installed so the caller can produce a WARN.
    """
    try:
        from ApplicationServices import (  # type: ignore[import-not-found]
            AXIsProcessTrustedWithOptions,
            kAXTrustedCheckOptionPrompt,
        )
        from CoreFoundation import (  # type: ignore[import-not-found]
            CFDictionaryCreate,
            kCFBooleanTrue,
        )
    except ImportError:
        # Fallback: the non-prompting variant (no Accessibility install
        # but pyobjc-core present is unusual; we still want a probe).
        try:
            from ApplicationServices import (  # type: ignore[import-not-found]
                AXIsProcessTrusted,
            )

            return bool(AXIsProcessTrusted())
        except ImportError:
            return None
    try:
        # CFDictionary with a single { kAXTrustedCheckOptionPrompt: True }
        keys = (kAXTrustedCheckOptionPrompt,)
        vals = (kCFBooleanTrue,)
        opts = CFDictionaryCreate(None, keys, vals, 1, None, None)
        return bool(AXIsProcessTrustedWithOptions(opts))
    except Exception:
        return None


def _screen_recording_granted() -> bool | None:
    try:
        from Quartz import (  # type: ignore[import-not-found]
            CGPreflightScreenCaptureAccess,
        )

        return bool(CGPreflightScreenCaptureAccess())
    except ImportError:
        return None


# ── Linux ────────────────────────────────────────────────────────────────


def _check_linux(config: ShimConfig, *, is_wsl2: bool) -> list[CheckResult]:
    out: list[CheckResult] = []
    if is_wsl2:
        out.append(
            CheckResult(
                "WARN",
                "display_server",
                "WSL2 has no display server reachable to the Windows host; computer_use disabled",
                blocking=config.enable_computer_use,
            )
        )
        return out
    from .computer_use.backends.linux import _is_wayland

    # The backend's own rule, so the two agree: it refuses input whenever
    # WAYLAND_DISPLAY is set, whatever XDG_SESSION_TYPE says.
    session = "wayland" if _is_wayland() else os.environ.get("XDG_SESSION_TYPE", "").lower()
    # An unset or unrecognized session type (e.g. "tty" in a shell attached to
    # a running desktop) says nothing about the display; ask DISPLAY.
    if session not in ("wayland", "x11") and os.environ.get("DISPLAY"):
        session = "x11"
    if session == "wayland":
        out.append(
            CheckResult(
                "WARN",
                "display_server",
                "Wayland — input injection is blocked; switch to X11 or use a "
                "portal+pipewire workflow for screenshots",
                blocking=config.enable_computer_use,
            )
        )
    elif session == "x11" and os.environ.get("DISPLAY"):
        # HELLO's own probe, so doctor never reports what HELLO leaves out.
        if platform_info._x11_display_opens():
            out.append(CheckResult("OK", "display_server", "X11 — full computer_use available"))
        else:
            out.append(
                CheckResult(
                    "WARN",
                    "display_server",
                    f"X11 session, but the X server at DISPLAY={os.environ['DISPLAY']} "
                    "doesn't accept a connection; computer_use is unavailable",
                    blocking=config.enable_computer_use,
                )
            )
    elif session == "x11":
        out.append(
            CheckResult(
                "WARN",
                "display_server",
                "X11 session, but DISPLAY is unset; computer_use can't open the display",
                blocking=config.enable_computer_use,
            )
        )
    else:
        out.append(
            CheckResult(
                "WARN",
                "display_server",
                "no display found (DISPLAY and WAYLAND_DISPLAY are unset); "
                "computer_use needs an X11 session",
                blocking=config.enable_computer_use,
            )
        )

    import shutil

    if shutil.which("wmctrl"):
        out.append(CheckResult("OK", "wmctrl", "installed"))
    else:
        out.append(
            CheckResult(
                "WARN",
                "wmctrl",
                "not installed — WINDOW_LIST / WINDOW_FOCUS will be unavailable",
            )
        )
    if config.enable_clipboard:
        if shutil.which("xclip") or shutil.which("wl-copy"):
            out.append(CheckResult("OK", "clipboard tools", "xclip or wl-copy present"))
        else:
            out.append(
                CheckResult(
                    "WARN",
                    "clipboard tools",
                    "neither xclip nor wl-copy on PATH — clipboard ops will fail",
                )
            )
    return out


# ── Windows ──────────────────────────────────────────────────────────────


def _check_windows(config: ShimConfig) -> list[CheckResult]:
    out: list[CheckResult] = []
    try:
        import ctypes

        elevated = bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except Exception:
        elevated = False
    if elevated:
        out.append(
            CheckResult(
                "OK",
                "UAC",
                "running elevated — can inject into UAC-elevated windows",
            )
        )
    else:
        out.append(
            CheckResult(
                "INFO",
                "UAC",
                "not elevated — non-elevated shim cannot inject into elevated "
                "windows. Re-launch as Administrator for full computer_use.",
            )
        )
    try:
        import win32clipboard  # type: ignore[import-not-found]  # noqa: F401

        out.append(CheckResult("OK", "pywin32", "installed"))
    except Exception as exc:
        out.append(
            CheckResult(
                "WARN",
                "pywin32",
                f"not importable: {exc} — install with `pip install pywin32`",
                blocking=config.enable_computer_use,
            )
        )
    return out


__all__ = ["EX_CONFIG", "CheckResult", "run_doctor"]
