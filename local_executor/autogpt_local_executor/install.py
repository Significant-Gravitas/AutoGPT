"""
Install / uninstall the autostart entry for the shim daemon.

Per docs/CROSS_PLATFORM.md "Autostart / background daemon" row:
- macOS  : ~/Library/LaunchAgents/net.autogpt.shim.plist  (launchd LaunchAgent)
- Linux  : ~/.config/systemd/user/autogpt-shim.service    (systemd user unit)
- Windows: %LOCALAPPDATA%\\autogpt-local-executor\\autogpt-shim.xml
           (Task Scheduler XML; registered via `schtasks /Create /XML ...`)
- WSL2   : same path as Linux (systemd user unit). Note `enable-linger` for
           start-without-login.

We never invoke the OS service managers (launchctl, systemctl, schtasks) —
we only write/remove files and print the commands the user should run.
This module is import-safe on every OS; the OS-specific writers are only
called via dispatch from `install_autostart()` / `uninstall_autostart()`.
"""

from __future__ import annotations

import os
import platform as _platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from . import platform_info
from .fsops import temporary_sibling

_TEMPLATES_DIR = Path(__file__).parent / "templates"

LAUNCHD_LABEL = "net.autogpt.shim"
SYSTEMD_UNIT_NAME = "autogpt-shim.service"
TASKSCHED_TASK_NAME = r"\AutoGPT\Shim"


# ── Rendering ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AutostartContext:
    """Values substituted into the templates at install time."""

    # The program that starts the shim, and any arguments before its own.
    shim_command: tuple[str, ...]
    # The config file the service names with --config; None when it names
    # none, and the shim finds its default itself.
    config_path: str | None
    stdout_log: str
    stderr_log: str
    # Other global options `install` was given (--platform-url, say), which
    # `start` needs as well.
    global_args: tuple[str, ...] = ()

    @property
    def start_argv(self) -> tuple[str, ...]:
        config = ("--config", self.config_path) if self.config_path is not None else ()
        return (*self.shim_command, *config, *self.global_args, "start")


def _read_template(name: str) -> str:
    return (_TEMPLATES_DIR / name).read_text(encoding="utf-8")


def render_template(name: str, ctx: AutostartContext) -> str:
    """Fill a template, each value quoted for that file's own syntax.

    KeyError surfaces unknown placeholders.
    """
    for value in (*ctx.start_argv, ctx.stdout_log, ctx.stderr_log):
        if _CONTROL_CHARACTER.search(value):
            raise ValueError(f"Autostart value {value!r} contains a control character")
    return _read_template(name).format(**_TEMPLATE_VALUES[name](ctx))


_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")
# Characters an ExecStart word may hold without quotes.
_SYSTEMD_PLAIN = re.compile(r"[\w@+=:,./-]+")


def _launchd_values(ctx: AutostartContext) -> dict[str, str]:
    return {
        "program_arguments": "\n".join(
            f"        <string>{_xml_escape(arg)}</string>" for arg in ctx.start_argv
        ),
        "stdout_log": _xml_escape(ctx.stdout_log),
        "stderr_log": _xml_escape(ctx.stderr_log),
    }


def _systemd_values(ctx: AutostartContext) -> dict[str, str]:
    return {
        "exec_start": " ".join(_systemd_word(arg) for arg in ctx.start_argv),
        # Specifiers are expanded in these paths; variables are not.
        "stdout_log": ctx.stdout_log.replace("%", "%%"),
        "stderr_log": ctx.stderr_log.replace("%", "%%"),
    }


def _systemd_word(arg: str) -> str:
    """One ExecStart argument, which systemd reads back exactly as given."""
    text = arg.replace("%", "%%").replace("$", "$$")
    if _SYSTEMD_PLAIN.fullmatch(arg):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _task_scheduler_values(ctx: AutostartContext) -> dict[str, str]:
    program, *arguments = ctx.start_argv
    return {
        "command": _xml_escape(subprocess.list2cmdline([program])),
        "arguments": _xml_escape(subprocess.list2cmdline(arguments)),
    }


_TEMPLATE_VALUES: dict[str, Callable[[AutostartContext], dict[str, str]]] = {
    "launchd.plist.template": _launchd_values,
    "systemd.service.template": _systemd_values,
    "taskscheduler.xml.template": _task_scheduler_values,
}


# ── Path resolution ──────────────────────────────────────────────────────────


def _absolute_as_given(path: str | os.PathLike[str]) -> str:
    """`path` made absolute for a service that starts in another working
    directory, with its links and `..`s left for the OS to follow. abspath()
    drops `link/..` without looking at the disk, which on Linux and macOS
    names another file where `link` is a symlink. Windows drops `link\\..`
    itself, as abspath does, and abspath reads a drive-relative path
    (`C:conf.toml`), which Path.absolute() gets wrong before Python 3.12."""
    if sys.platform == "win32":
        return os.path.abspath(path)
    return str(Path(path).absolute())


def resolve_shim_command() -> tuple[str, ...]:
    """The command that starts the shim, as separate arguments.

    Prefer the `autogpt-shim` console script on PATH, then the one beside
    this interpreter (a pipx or venv install run by absolute path), then
    `python -m autogpt_local_executor`.
    """
    found = shutil.which("autogpt-shim") or shutil.which(
        "autogpt-shim", path=str(Path(sys.executable).parent)
    )
    if found:
        # A relative PATH entry gives a relative path, which a service
        # started from another working directory couldn't run.
        return (_absolute_as_given(found),)
    return (sys.executable, "-m", "autogpt_local_executor")


def default_stdout_log() -> Path:
    plat = platform_info.detect_platform()
    if plat == "darwin":
        return Path.home() / "Library" / "Logs" / "autogpt-local-executor" / "shim.out.log"
    if plat == "windows":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "autogpt-local-executor" / "logs" / "shim.out.log"
    xdg_state = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state) if xdg_state else Path.home() / ".local" / "state"
    return base / "autogpt-local-executor" / "shim.out.log"


def default_stderr_log() -> Path:
    return default_stdout_log().with_name("shim.err.log")


def default_config_path() -> Path:
    # Mirror config._default_config_path() without importing the heavy module.
    plat = platform_info.detect_platform()
    if plat == "windows":
        base = Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
        return base / "autogpt-local-executor" / "config.toml"
    if plat == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "autogpt-local-executor"
            / "config.toml"
        )
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "autogpt-local-executor" / "config.toml"


def build_context(
    config_path: Path | None, *, global_args: tuple[str, ...] = ()
) -> AutostartContext:
    # A --config that names no file is an error at start, so the default
    # location is named only when a config file is there. The CLI has already
    # checked that an explicit one exists.
    cfg = config_path if config_path is not None else default_config_path()
    named = config_path is not None or cfg.expanduser().is_file()
    return AutostartContext(
        shim_command=resolve_shim_command(),
        # The service manager starts the shim in its own working directory.
        # Not resolve(): a config that is a symlink stays one.
        config_path=_absolute_as_given(cfg.expanduser()) if named else None,
        stdout_log=str(default_stdout_log()),
        stderr_log=str(default_stderr_log()),
        global_args=global_args,
    )


# ── Per-OS target paths ──────────────────────────────────────────────────────


def macos_launchagent_path(home: Path | None = None) -> Path:
    home = home or Path.home()
    return home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def linux_systemd_unit_path(home: Path | None = None) -> Path:
    home = home or Path.home()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else home / ".config"
    return base / "systemd" / "user" / SYSTEMD_UNIT_NAME


def windows_task_xml_path(home: Path | None = None) -> Path:
    home = home or Path.home()
    base = Path(os.environ.get("LOCALAPPDATA", str(home / "AppData" / "Local")))
    return base / "autogpt-local-executor" / "autogpt-shim.xml"


# ── Per-OS writers (small, testable) ─────────────────────────────────────────


def _write_atomic(target: Path, content: str, *, encoding: str = "utf-8") -> Path:
    """Write `content` beside `target` and rename it over, so a write that
    fails part way leaves the old file whole."""
    target.parent.mkdir(parents=True, exist_ok=True)
    # A name of its own, created exclusively, so two installs at once can't
    # write into each other's file.
    temp = temporary_sibling(target)
    try:
        with open(temp, "x", encoding=encoding) as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, target)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    return target


def write_macos_launchagent(
    ctx: AutostartContext,
    target: Path | None = None,
) -> Path:
    target = target or macos_launchagent_path()
    return _write_atomic(target, render_template("launchd.plist.template", ctx))


def write_linux_systemd_unit(
    ctx: AutostartContext,
    target: Path | None = None,
) -> Path:
    target = target or linux_systemd_unit_path()
    return _write_atomic(target, render_template("systemd.service.template", ctx))


def write_windows_task_xml(
    ctx: AutostartContext,
    target: Path | None = None,
) -> Path:
    target = target or windows_task_xml_path()
    # The template declares UTF-16, the encoding Task Scheduler exports.
    return _write_atomic(
        target, render_template("taskscheduler.xml.template", ctx), encoding="utf-16"
    )


# ── Status / uninstall helpers ───────────────────────────────────────────────


@dataclass(frozen=True)
class AutostartStatus:
    platform: str
    target_path: Path
    installed: bool
    enable_command: str | None
    disable_command: str | None
    notes: tuple[str, ...] = ()


def _current_platform_for_install() -> str:
    """Single source of truth for the install/uninstall/status dispatch.

    We do not call platform_info.detect_platform() directly here because
    tests mock platform.system(); centralising the read keeps the seam
    obvious.
    """
    sys_name = _platform.system().lower()
    if sys_name == "darwin":
        return "darwin"
    if sys_name == "windows":
        return "windows"
    if sys_name == "linux":
        # Distinguish WSL2 so we can surface the enable-linger note.
        try:
            with open("/proc/version", encoding="utf-8", errors="replace") as f:
                if "microsoft" in f.read().lower():
                    return "wsl2"
        except OSError:
            pass
        return "linux"
    return sys_name


def target_path_for(platform_name: str, home: Path | None = None) -> Path:
    if platform_name == "darwin":
        return macos_launchagent_path(home)
    if platform_name == "windows":
        return windows_task_xml_path(home)
    if platform_name in ("linux", "wsl2"):
        return linux_systemd_unit_path(home)
    raise ValueError(f"Unsupported platform for autostart install: {platform_name!r}")


def status_for(platform_name: str, home: Path | None = None) -> AutostartStatus:
    target = target_path_for(platform_name, home)
    installed = target.is_file()
    if platform_name == "darwin":
        return AutostartStatus(
            platform=platform_name,
            target_path=target,
            installed=installed,
            enable_command=f"launchctl load {target}",
            disable_command=f"launchctl unload {target}",
            notes=(
                "RunAtLoad + KeepAlive are set; launchd starts the shim at "
                "login and restarts it if it dies.",
            ),
        )
    if platform_name in ("linux", "wsl2"):
        notes: tuple[str, ...] = (
            "Enable with: systemctl --user daemon-reload && "
            "systemctl --user enable --now autogpt-shim.service",
        )
        if platform_name == "wsl2":
            notes = notes + (
                "WSL2: also run `loginctl enable-linger $USER` to keep the "
                "shim alive when no shell is open.",
            )
        else:
            notes = notes + (
                "For start-without-login on headless boxes: `loginctl enable-linger $USER`.",
            )
        return AutostartStatus(
            platform=platform_name,
            target_path=target,
            installed=installed,
            enable_command="systemctl --user enable --now autogpt-shim.service",
            disable_command="systemctl --user disable --now autogpt-shim.service",
            notes=notes,
        )
    if platform_name == "windows":
        return AutostartStatus(
            platform=platform_name,
            target_path=target,
            installed=installed,
            enable_command=(f'schtasks /Create /TN "{TASKSCHED_TASK_NAME}" /XML "{target}"'),
            disable_command=f'schtasks /Delete /TN "{TASKSCHED_TASK_NAME}" /F',
            notes=(
                "Task Scheduler XML is written but NOT registered. Run the "
                "schtasks command above (elevated cmd / PowerShell) to "
                "register it.",
            ),
        )
    return AutostartStatus(
        platform=platform_name,
        target_path=target,
        installed=installed,
        enable_command=None,
        disable_command=None,
        notes=(f"No autostart wiring for platform {platform_name!r}.",),
    )


# ── Public install / uninstall entry points ──────────────────────────────────


_WRITERS: dict[str, Callable[[AutostartContext, Path | None], Path]] = {
    "darwin": write_macos_launchagent,
    "linux": write_linux_systemd_unit,
    "wsl2": write_linux_systemd_unit,
    "windows": write_windows_task_xml,
}


def install_autostart(
    *,
    at_login: bool = True,
    config_path: Path | None = None,
    home: Path | None = None,
    platform_name: str | None = None,
    global_args: tuple[str, ...] = (),
) -> AutostartStatus:
    """Write the autostart file for this OS.

    `at_login` is accepted for forward compatibility; v0 only supports
    "at login" triggers (matching CROSS_PLATFORM.md), so we currently
    raise if the caller sets it to False.
    """
    if not at_login:
        raise NotImplementedError(
            "v0 install only supports start-at-login. Use --at-login (default)."
        )
    plat = platform_name or _current_platform_for_install()
    writer = _WRITERS.get(plat)
    if writer is None:
        raise ValueError(f"Unsupported platform for autostart install: {plat!r}")
    ctx = build_context(config_path, global_args=global_args)
    target = target_path_for(plat, home)
    # Ensure parent log dirs exist; templates reference them.
    Path(ctx.stdout_log).parent.mkdir(parents=True, exist_ok=True)
    writer(ctx, target)
    return status_for(plat, home)


def uninstall_autostart(
    *,
    home: Path | None = None,
    platform_name: str | None = None,
) -> AutostartStatus:
    """Remove the autostart file for this OS.

    Returns the post-removal status. Idempotent: no-op if already absent.
    """
    plat = platform_name or _current_platform_for_install()
    target = target_path_for(plat, home)
    if target.is_file():
        target.unlink()
    return status_for(plat, home)


__all__ = [
    "AutostartContext",
    "AutostartStatus",
    "build_context",
    "default_config_path",
    "install_autostart",
    "linux_systemd_unit_path",
    "macos_launchagent_path",
    "render_template",
    "resolve_shim_command",
    "status_for",
    "target_path_for",
    "uninstall_autostart",
    "windows_task_xml_path",
    "write_linux_systemd_unit",
    "write_macos_launchagent",
    "write_windows_task_xml",
]
