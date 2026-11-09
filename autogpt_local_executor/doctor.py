"""
`autogpt-shim doctor` — preflight checks for the shim.

Probes the keychain and the directory browser (or a legacy allowed_root).
With computer use on, it runs cua-driver's own `doctor`, starts the driver
the way the daemon does, and on macOS reports the Accessibility and Screen
Recording grants of the app that runs the shim, asking macOS to show its
prompts when they are missing. See docs/CUA_DRIVER.md.

Returns exit 0 when everything required is healthy, 78 (EX_CONFIG) when
computer use is requested but a prerequisite is missing.

Output is plain ASCII (no emoji) — the markers are:
  OK    everything good
  WARN  degraded but won't block (e.g. computer-use not requested)
  FAIL  required and missing
  INFO  context-only
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from . import platform_info
from .computer_use import CuaDriverBridge, DriverUnavailable, find_driver_binary, offered_tools
from .computer_use.catalog import REVIEWED_DRIVER_VERSION, same_minor
from .computer_use.driver import COMPUTER_USE_INSTALL, driver_environment, ensure_executable
from .computer_use.mcp_client import McpError
from .config import ShimConfig
from .directory_browser import DirectoryBrowser

EX_CONFIG = 78
# How long one probe of the running driver may take.
PROBE_SECONDS = 30.0


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
    if config.enable_computer_use:
        checks.extend(_check_computer_use(config, plat))
    else:
        checks.append(CheckResult("INFO", "computer_use", "not requested in config"))
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
# The probe's name in the keychain. One name, not one per run: a probe the
# keychain stores after doctor stopped waiting may outlive doctor, which
# doesn't wait for its thread to delete it, and the next run's probe then
# overwrites and deletes it rather than leaving it there for good.
KEYCHAIN_PROBE_NAME = "doctor-probe"


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

    try:
        keyring.set_password(KEYCHAIN_SERVICE, KEYCHAIN_PROBE_NAME, "ok")
        try:
            stored = keyring.get_password(KEYCHAIN_SERVICE, KEYCHAIN_PROBE_NAME)
        finally:
            try:
                keyring.delete_password(KEYCHAIN_SERVICE, KEYCHAIN_PROBE_NAME)
            except Exception:
                pass
    except Exception as exc:
        return CheckResult("WARN", "keychain", f"{name} could not store a value: {exc}")
    if stored != "ok":
        return CheckResult("WARN", "keychain", f"{name} did not return a value just stored in it")
    return CheckResult("OK", "keychain", name)


# ── Computer use (cua-driver) ────────────────────────────────────────────

# How long cua-driver's own `doctor` and `permissions grant` may take.
DRIVER_COMMAND_SECONDS = 60.0
# The cua-driver doctor probes that only describe the install.
_INFO_PROBES = frozenset({"binary", "install dir", "home dir"})


def _check_computer_use(config: ShimConfig, plat: str) -> list[CheckResult]:
    if plat == "wsl2":
        return [
            CheckResult(
                "FAIL",
                "computer_use",
                "WSL2 has no display server reachable to the Windows host; run the shim "
                "on Windows itself",
                blocking=True,
            )
        ]
    binary = find_driver_binary(config.cua_driver_path)
    if binary is None:
        return [
            CheckResult(
                "FAIL",
                "cua-driver",
                f"not installed — reinstall with `{COMPUTER_USE_INSTALL}`",
                blocking=True,
            )
        ]
    out = _driver_doctor(binary)
    if any(check.blocking for check in out):
        return out
    if plat == "linux":
        # The daemon's own test (can_do_computer_use): without it, HELLO
        # leaves computer use out, whatever cua-driver's doctor said.
        problem = platform_info.linux_display_problem()
        if problem is not None:
            out.append(CheckResult("FAIL", "display", problem, blocking=True))
            return out
    out.extend(asyncio.run(_probe_driver(config, plat)))
    if plat == "linux" and config.enable_clipboard and config.enable_clipboard_read_foreign:
        out.append(_check_linux_clipboard_types())
    if plat == "windows":
        out.append(_check_windows_elevation())
    return out


def _driver_doctor(binary: Path) -> list[CheckResult]:
    """cua-driver's own probes: version, display server, AT-SPI, telemetry,
    and on macOS other copies of CuaDriver.app."""
    ensure_executable(binary)
    try:
        completed = subprocess.run(
            [str(binary), "doctor", "--json"],
            capture_output=True,
            text=True,
            timeout=DRIVER_COMMAND_SECONDS,
            env=driver_environment(None),
            check=False,
        )
        report = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        return [CheckResult("FAIL", "cua-driver", f"`{binary} doctor` failed: {exc}", True)]
    if not isinstance(report, dict):
        return [
            CheckResult("FAIL", "cua-driver", f"`{binary} doctor` printed no report object", True)
        ]
    build = report.get("build")
    version = str(build.get("version") or "") if isinstance(build, dict) else ""
    if not same_minor(version):
        return [
            CheckResult(
                "FAIL",
                "cua-driver",
                f"{binary} is {version or 'an unknown version'}; this shim works with "
                f"cua-driver {REVIEWED_DRIVER_VERSION}",
                blocking=True,
            )
        ]
    out = [CheckResult("OK", "cua-driver", f"{version} at {binary}")]
    probes = report.get("probes") or []
    readable = (
        [probe for probe in probes if isinstance(probe, dict)] if isinstance(probes, list) else []
    )
    if not isinstance(probes, list) or len(readable) < len(probes):
        out.append(
            CheckResult(
                "FAIL",
                "cua-driver probes",
                f"`{binary} doctor` reported probes doctor can't read",
                blocking=True,
            )
        )
    for probe in readable:
        label = str(probe.get("label") or "probe")
        if label in _INFO_PROBES:
            continue
        status = probe.get("status")
        message = str(probe.get("message") or "")
        detail = str(probe.get("detail") or "")
        text = message + (f" ({detail})" if detail and status != "ok" else "")
        marker = {"ok": "OK", "warn": "WARN"}.get(str(status), "FAIL")
        out.append(CheckResult(marker, f"cua-driver {label}", text, blocking=marker == "FAIL"))
    return out


async def _probe_driver(config: ShimConfig, plat: str) -> list[CheckResult]:
    """Start the driver as the daemon would and ask it what it can see."""
    bridge = CuaDriverBridge(
        tools=offered_tools(enable_clipboard=config.enable_clipboard),
        binary=config.cua_driver_path,
    )
    out: list[CheckResult] = []
    try:
        try:
            snapshot = await bridge.snapshot()
        except DriverUnavailable as exc:
            return [CheckResult("FAIL", "cua-driver start", exc.message, blocking=True)]
        out.append(
            CheckResult(
                "OK",
                "cua-driver start",
                f"{snapshot.info.version}, {len(snapshot.tools)} tools offered",
            )
        )
        out.append(await _check_screen(bridge))
        if plat == "darwin":
            out.extend(await _check_macos_permissions(bridge))
    finally:
        await bridge.close()
    return out


async def _check_screen(bridge: CuaDriverBridge) -> CheckResult:
    try:
        raw = await bridge.call_tool("get_screen_size", {}, timeout=PROBE_SECONDS)
    except (DriverUnavailable, McpError, TimeoutError) as exc:
        return CheckResult("WARN", "display", f"cua-driver could not read the screen: {exc}")
    structured = raw.get("structuredContent")
    size = structured if isinstance(structured, dict) else {}
    width, height, scale = size.get("width"), size.get("height"), size.get("scale_factor", 1)
    if raw.get("isError") or not all(_is_number(value) for value in (width, height, scale)):
        content = raw.get("content")
        text = " ".join(
            str(block.get("text") or "")
            for block in (content if isinstance(content, list) else [])
            if isinstance(block, dict)
        )
        return CheckResult("WARN", "display", f"cua-driver could not read the screen: {text}")
    return CheckResult("OK", "display", f"{width:g}x{height:g} (scale {scale:g})")


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


async def _check_macos_permissions(bridge: CuaDriverBridge) -> list[CheckResult]:
    granted = await _macos_grants(bridge)
    if granted is None:
        return [
            CheckResult(
                "FAIL",
                "permissions",
                "cua-driver could not report the Accessibility and Screen Recording grants",
                blocking=True,
            )
        ]
    missing = [name for name, ok in granted.items() if not ok]
    if missing:
        # The driver runs as this process's child and leaves prompting to its
        # host, so the shim asks, as it did before cua-driver.
        _request_macos_grants(missing)
    out = []
    for name, label in (
        ("accessibility", "Accessibility"),
        ("screen_recording", "Screen Recording"),
    ):
        if granted.get(name):
            out.append(CheckResult("OK", label, "granted"))
        else:
            out.append(
                CheckResult(
                    "FAIL",
                    label,
                    f"not granted — System Settings → Privacy & Security → {label}, enable "
                    f"{MACOS_GRANT_HOLDER}, then restart the local executor",
                    blocking=True,
                )
            )
    return out


# Who holds the macOS grants: the driver is a child of the shim, so macOS
# asks for the app that started the shim.
MACOS_GRANT_HOLDER = (
    "the app that runs the local executor (your terminal, or for the autostart "
    "service, the Python that runs it)"
)


def _request_macos_grants(missing: list[str]) -> None:
    """Have macOS show its prompts for this process's app (best effort)."""
    if "accessibility" in missing:
        with contextlib.suppress(Exception):
            from ApplicationServices import (  # type: ignore[import-not-found]
                AXIsProcessTrustedWithOptions,
                kAXTrustedCheckOptionPrompt,
            )

            AXIsProcessTrustedWithOptions({kAXTrustedCheckOptionPrompt: True})
    if "screen_recording" in missing:
        with contextlib.suppress(Exception):
            from Quartz import CGRequestScreenCaptureAccess  # type: ignore[import-not-found]

            CGRequestScreenCaptureAccess()


async def _macos_grants(bridge: CuaDriverBridge) -> dict[str, bool] | None:
    try:
        raw = await bridge.call_tool("check_permissions", {"prompt": False}, timeout=PROBE_SECONDS)
    except (DriverUnavailable, McpError, TimeoutError):
        return None
    structured = raw.get("structuredContent")
    if raw.get("isError") or not isinstance(structured, dict):
        return None
    return {name: structured.get(name) is True for name in ("accessibility", "screen_recording")}


def _check_linux_clipboard_types() -> CheckResult:
    """Reading clipboard content the shim didn't write first checks it for
    a password manager's marker, with wl-paste or xclip."""
    wayland = bool(os.environ.get("WAYLAND_DISPLAY"))
    tool = "wl-paste" if wayland else "xclip"
    if shutil.which(tool):
        return CheckResult("OK", "clipboard types", f"{tool} present")
    return CheckResult(
        "WARN",
        "clipboard types",
        f"{tool} not on PATH — clipboard_read refuses content the shim didn't write, "
        "since it can't check it for a password manager's marker",
    )


def _check_windows_elevation() -> CheckResult:
    try:
        import ctypes

        elevated = bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except Exception:
        elevated = False
    if elevated:
        return CheckResult("OK", "UAC", "running elevated — can drive UAC-elevated windows")
    return CheckResult(
        "INFO",
        "UAC",
        "not elevated — computer use cannot drive windows of apps running as Administrator",
    )


__all__ = ["EX_CONFIG", "CheckResult", "run_doctor"]
