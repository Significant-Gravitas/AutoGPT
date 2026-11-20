"""
Install / uninstall the autostart entry for the shim daemon.

Per docs/CROSS_PLATFORM.md "Autostart / background daemon" row:
- macOS  : ~/Library/LaunchAgents/co.agpt.shim.plist  (launchd LaunchAgent)
- Linux  : ~/.config/systemd/user/autogpt-shim.service    (systemd user unit)
- Windows: %LOCALAPPDATA%\\autogpt-local-executor\\autogpt-shim.xml
           (Task Scheduler XML; registered via `schtasks /Create /XML ...`)
- WSL2   : same path as Linux (systemd user unit). Note `enable-linger` for
           start-without-login.

`install` only writes the file and prints the command that registers it with
the OS service manager (launchctl, systemctl, schtasks): starting the shim at
every login stays the person's own step. `uninstall` does run the service
manager: it stops the service and unregisters it, and only then removes the
file (`uninstall_autostart`). On macOS both also remove a service left under
the label the LaunchAgent had before (`_forget_former_launchagent`).

Each entry is one account's own: the LaunchAgent and the systemd unit are in
the account's home folder, and the Windows task names the account that ran
`install` (`_windows_account_sid`).
This module is import-safe on every OS; the OS-specific writers are only
called via dispatch from `install_autostart()` / `uninstall_autostart()`.
"""

from __future__ import annotations

import csv
import os
import platform as _platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from . import platform_info
from .fsops import temporary_sibling
from .supervise import SUPERVISE_OPTION

_TEMPLATES_DIR = Path(__file__).parent / "templates"

# A launchd label is the reversed name of a domain its author owns: agpt.co.
LAUNCHD_LABEL = "co.agpt.shim"
SYSTEMD_UNIT_NAME = "autogpt-shim.service"
TASKSCHED_TASK_NAME = r"\AutoGPT\Shim"
# What stops the service, unregisters it and removes its autostart entry, on
# every OS and whether or not the entry's file is still there.
UNINSTALL_COMMAND = "autogpt-shim uninstall"


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
    # The account the Windows task belongs to: its SID, or its name. The
    # LaunchAgent and the systemd unit name none, because launchd and
    # systemd keep each account's services apart themselves.
    user_id: str | None = None

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
    for value in (*ctx.start_argv, ctx.stdout_log, ctx.stderr_log, ctx.user_id or ""):
        if _CONTROL_CHARACTER.search(value):
            raise ValueError(f"Autostart value {value!r} contains a control character")
    return _read_template(name).format(**_TEMPLATE_VALUES[name](ctx))


_CONTROL_CHARACTER = re.compile(r"[\x00-\x1f\x7f]")
# Characters an ExecStart word may hold without quotes.
_SYSTEMD_PLAIN = re.compile(r"[\w@+=:,./-]+")


def _launchd_values(ctx: AutostartContext) -> dict[str, str]:
    return {
        "label": _xml_escape(LAUNCHD_LABEL),
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
    if not ctx.user_id:
        # A logon trigger and a principal that name no account make a task
        # for every account's logon, which only an administrator may
        # register, and which then runs as whoever registered it.
        raise ValueError("The Windows task has to name the account it belongs to")
    program, *arguments = ctx.start_argv
    # Task Scheduler doesn't start a program again after it exits, so under
    # the task the shim does that itself (supervise). launchd and systemd
    # restart it, and get the plain `start`.
    arguments.append(SUPERVISE_OPTION)
    return {
        "command": _xml_escape(subprocess.list2cmdline([program])),
        "arguments": _xml_escape(subprocess.list2cmdline(arguments)),
        "user_id": _xml_escape(ctx.user_id),
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
    config_path: Path | None, *, global_args: tuple[str, ...] = (), user_id: str | None = None
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
        user_id=user_id,
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
    # Registers the entry with the service manager. It names the entry's
    # file, so it works only while the entry is installed.
    enable_command: str | None
    # Stops the service and removes the entry: `autogpt-shim uninstall`,
    # which works with or without the entry's file. The service managers'
    # own commands mostly don't: `launchctl unload` reads the plist and
    # `systemctl disable` the unit file.
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
            # Quoted where it needs it: a home folder with a space in its
            # name (`/Volumes/User Homes/alex`) is otherwise two arguments.
            enable_command=f"launchctl load {shlex.quote(str(target))}",
            disable_command=UNINSTALL_COMMAND,
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
            disable_command=UNINSTALL_COMMAND,
            notes=notes,
        )
    if platform_name == "windows":
        return AutostartStatus(
            platform=platform_name,
            target_path=target,
            installed=installed,
            enable_command=(f'schtasks /Create /TN "{TASKSCHED_TASK_NAME}" /XML "{target}"'),
            disable_command=UNINSTALL_COMMAND,
            notes=(
                "The task is written but NOT registered. Run the schtasks command "
                "above to register it. It is this account's own task, for its own "
                "logon, so it takes no elevated prompt: run it in a prompt of this "
                "account's.",
                "Task Scheduler keeps one task of this name for the whole computer. "
                "If another account on this computer has registered it, schtasks "
                "refuses this one, and `autogpt-shim start` is how this account "
                "runs the shim.",
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
    run: CommandRunner | None = None,
) -> AutostartStatus:
    """Write the autostart file for this OS.

    `at_login` is accepted for forward compatibility; v0 only supports
    "at login" triggers (matching CROSS_PLATFORM.md), so we currently
    raise if the caller sets it to False.

    Raises AutostartError when an entry that would work can't be written:
    on Windows, when the account the task is for can't be read.
    """
    if not at_login:
        raise NotImplementedError(
            "v0 install only supports start-at-login. Use --at-login (default)."
        )
    plat = platform_name or _current_platform_for_install()
    writer = _WRITERS.get(plat)
    if writer is None:
        raise ValueError(f"Unsupported platform for autostart install: {plat!r}")
    # Before anything is written: without it there is no task to write.
    user_id = _windows_account_sid(run or run_command) if plat == "windows" else None
    ctx = build_context(config_path, global_args=global_args, user_id=user_id)
    target = target_path_for(plat, home)
    # Ensure parent log dirs exist; templates reference them.
    Path(ctx.stdout_log).parent.mkdir(parents=True, exist_ok=True)
    writer(ctx, target)
    former_notes: tuple[str, ...] = ()
    if plat == "darwin":
        # Once the new entry is written: with the earlier service unloaded
        # first, a write that then failed (a full disk) left the Mac with
        # no service for the next login.
        former = _forget_former_launchagent(home, run or run_command)
        former_notes = former.notes
        if former.stuck is not None:
            former_notes += (
                "A service under this one's earlier label may still be running: "
                f"{former.stuck.problem}. Stop it before you load this one, or two shims "
                f"start for this computer: {former.stuck.stop_command}",
            )
        if former.entry_problem is not None:
            former_notes += (
                f"Could not remove the earlier autostart entry {former.entry_problem}. Remove "
                "it, or it loads the earlier service again at the next login and two shims "
                "start for this computer.",
            )
    status = status_for(plat, home)
    return replace(status, notes=status.notes + former_notes)


# ── Stopping the service ─────────────────────────────────────────────────────

# How long one question to a service manager may take.
_ASK_SECONDS = 10.0
# How long each service manager gets to stop the service: a little longer
# than it gives the shim itself before it kills it (launchd 20 s, systemd
# 90 s by default; Task Scheduler ends a task at once).
_LAUNCHD_STOP_SECONDS = 30.0
_SYSTEMD_STOP_SECONDS = 100.0
_TASK_SCHEDULER_STOP_SECONDS = 30.0
# How often launchd is asked whether the service has gone.
_LAUNCHD_POLL_SECONDS = 0.2
# systemd's states for a unit that isn't running and isn't about to.
_SYSTEMD_NOT_RUNNING = ("inactive", "failed")

_monotonic = time.monotonic
_sleep = time.sleep


@dataclass(frozen=True)
class CommandResult:
    """How one service-manager command ended."""

    # Its exit status; None when it didn't run to its end.
    returncode: int | None
    # What it wrote, stdout and stderr together.
    output: str = ""
    # Why it didn't run to its end: not found, or still running at its
    # time limit.
    failure: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0

    def said(self) -> str:
        """Why it didn't succeed, in its own words where it gave any."""
        if self.failure is not None:
            return self.failure
        text = " ".join(self.output.split())
        return text or f"exit status {self.returncode}"


CommandRunner = Callable[[Sequence[str], float], CommandResult]


def run_command(argv: Sequence[str], timeout: float) -> CommandResult:
    """Run one service-manager command to its end, for at most `timeout`
    seconds, with no input and its output kept."""
    shown = " ".join(argv)
    try:
        done = subprocess.run(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return CommandResult(None, failure=f"`{argv[0]}` was not found on PATH")
    except subprocess.TimeoutExpired:
        return CommandResult(None, failure=f"`{shown}` did not finish within {timeout:g} s")
    except OSError as exc:
        return CommandResult(None, failure=f"`{shown}` could not be run: {exc}")
    # schtasks writes in the console's code page, the others UTF-8.
    encoding = "oem" if sys.platform == "win32" else "utf-8"
    return CommandResult(done.returncode, done.stdout.decode(encoding, errors="replace"))


class AutostartError(Exception):
    """`install` couldn't write an autostart entry that would work."""


_SID = re.compile(r"S-1-\d+(?:-\d+)+")


def _windows_account_sid(run: CommandRunner) -> str:
    """The SID of the account this process runs as: the one the Windows
    task is for.

    `whoami` reads it from this process's own token, so it is the account
    whose home folder and Credential Manager this `install` uses, also in a
    prompt that was opened with another account's credentials. Both of the
    task's `UserId`s take an account's name or its SID; the SID needs no
    lookup, stays the account's when the account is renamed, and holds
    nothing XML has to escape.

    `whoami /user /fo csv /nh` prints one line, the name and then the SID,
    each in quotes. The SID is read from its own column: a name may hold
    anything, something shaped like a SID included."""
    asked = run(["whoami", "/user", "/fo", "csv", "/nh"], _ASK_SECONDS)
    if not asked.succeeded:
        raise AutostartError(
            f"the account the task is for couldn't be read: `whoami /user` said: {asked.said()}"
        )
    rows = [row for row in csv.reader(asked.output.splitlines()) if row]
    sid = rows[0][-1].strip() if len(rows) == 1 else ""
    if not _SID.fullmatch(sid):
        raise AutostartError(
            "the account the task is for couldn't be read: `whoami /user` gave no SID"
        )
    return sid


@dataclass(frozen=True)
class ServiceStop:
    """What the service manager had of the service, and what was done."""

    # Whether the service manager had the service: loaded (launchd), active
    # (systemd) or registered (Task Scheduler). None when it couldn't be
    # asked, or couldn't tell.
    found: bool | None
    # What was found and done, when all of it went through.
    said: str = ""
    # What couldn't be asked or done, else None.
    problem: str | None = None
    # The service manager's own command that stops the service by hand. It
    # needs no autostart file, which `uninstall` removes either way.
    stop_command: str = ""
    # Where that command has to be run, when not just anywhere.
    stop_where: str = ""


def _user_id() -> int:
    return os.getuid()  # type: ignore[attr-defined,unused-ignore]


def _is_elevated() -> bool:
    """Whether this process sees and may change every scheduled task: one
    run from an administrator's elevated prompt."""
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return False


def _stop_launchd(run: CommandRunner, *, label: str, uid: int) -> ServiceStop:
    """Unload the LaunchAgent from launchd, which stops it.

    `launchctl bootout <domain>/<label>` takes the label, so it needs no
    plist; `launchctl unload <plist>` reads the file. A LaunchAgent is in
    the person's login session (gui/<uid>) when it was loaded from a
    terminal there or at login, and in their background session
    (user/<uid>) when it was loaded over SSH, so both are looked in.

    Whether it is loaded, before and after, is `launchctl print`'s exit
    status. bootout's own status isn't the answer: launchd gives a service
    time to exit before it kills it (20 s by default), and bootout can come
    back sooner, with a failure of its own for a service still on its way
    out (36, "Operation now in progress")."""
    by_hand = f"launchctl bootout gui/$(id -u)/{label}"
    unloaded: list[str] = []
    for domain, by_name in ((f"gui/{uid}", "gui/$(id -u)"), (f"user/{uid}", "user/$(id -u)")):
        target = f"{domain}/{label}"
        loaded = run(["launchctl", "print", target], _ASK_SECONDS)
        if loaded.failure is not None:
            return ServiceStop(
                found=True if unloaded else None, problem=loaded.failure, stop_command=by_hand
            )
        if not loaded.succeeded:
            continue  # not loaded there, or there is no such session
        by_hand = f"launchctl bootout {by_name}/{label}"
        deadline = _monotonic() + _LAUNCHD_STOP_SECONDS
        bootout = run(["launchctl", "bootout", target], _LAUNCHD_STOP_SECONDS)
        while True:
            still = run(["launchctl", "print", target], _ASK_SECONDS)
            if still.failure is not None:
                return ServiceStop(found=True, problem=still.failure, stop_command=by_hand)
            if not still.succeeded:
                break
            if _monotonic() >= deadline:
                refusal = (
                    "" if bootout.succeeded else f" (`launchctl bootout` said: {bootout.said()})"
                )
                return ServiceStop(
                    found=True,
                    problem=f"launchd still has {target} loaded {_LAUNCHD_STOP_SECONDS:g} s "
                    f"after it was told to unload it{refusal}",
                    stop_command=by_hand,
                )
            _sleep(_LAUNCHD_POLL_SECONDS)
        unloaded.append(domain)
    if unloaded:
        return ServiceStop(
            found=True,
            said=f"Stopped the service: launchd unloaded {label} from {' and '.join(unloaded)}.",
            stop_command=by_hand,
        )
    return ServiceStop(
        found=False,
        said=f"The service wasn't running: launchd has no {label} loaded.",
        stop_command=by_hand,
    )


def _stop_systemd(run: CommandRunner, *, unit: str, entry_installed: bool) -> ServiceStop:
    """Stop the user unit and disable it.

    `systemctl disable` reads the unit file: once the file is gone it
    fails, `--now` stops nothing, and the link that enables the unit stays.
    So this runs while the file is there. With no unit file (one removed by
    hand, or by a shim that removed it first), `systemctl stop` still stops
    a unit that is running, by its name."""
    by_hand = f"systemctl --user stop {unit}"
    state = run(["systemctl", "--user", "show", "--property=ActiveState", unit], _ASK_SECONDS)
    if not state.succeeded:
        return ServiceStop(
            found=None,
            problem=f"systemd's user manager couldn't be asked about {unit}: {state.said()}",
            stop_command=by_hand,
        )
    states = [
        line.partition("=")[2].strip()
        for line in state.output.splitlines()
        if line.startswith("ActiveState=")
    ]
    if not states:
        return ServiceStop(
            found=None,
            problem=f"`systemctl --user show` didn't say whether {unit} is running",
            stop_command=by_hand,
        )
    running = states[0] not in _SYSTEMD_NOT_RUNNING
    if entry_installed:
        done = run(["systemctl", "--user", "disable", "--now", unit], _SYSTEMD_STOP_SECONDS)
        said = (
            f"Stopped the service: systemd stopped and disabled {unit}."
            if running
            else f"The service wasn't running; systemd disabled {unit}."
        )
    elif running:
        done = run(["systemctl", "--user", "stop", unit], _SYSTEMD_STOP_SECONDS)
        said = f"Stopped the service: systemd stopped {unit}."
    else:
        return ServiceStop(
            found=False,
            said=f"The service wasn't running: systemd has no active {unit}.",
            stop_command=by_hand,
        )
    if not done.succeeded:
        return ServiceStop(
            found=running,
            problem=f"systemd didn't {'stop' if running else 'disable'} {unit}: {done.said()}",
            stop_command=by_hand,
        )
    return ServiceStop(found=running, said=said, stop_command=by_hand)


def _stop_scheduled_task(run: CommandRunner, *, task: str, elevated: bool) -> ServiceStop:
    """End the task's running instance, then delete the task.

    In that order: deleting a task leaves what it started running, and
    with the task gone nothing can end it by name. So a task that can't be
    ended is kept, for a prompt that may end it.

    `schtasks /End` succeeds for a task with no instance running (one whose
    shim exited to stop, or that was registered and never run), so such a
    task goes on to be deleted. A failure of it is a task that couldn't be
    ended, not one with nothing to end: the real Task Scheduler ends and
    deletes a task that never ran in tests/test_task_owner.py.

    The task is the account's own (`_windows_account_sid`), registered
    from its own prompt, so a prompt that isn't elevated sees it, ends it
    and deletes it.

    schtasks exits 1 for every failure and words them in the system's
    language, so a task this prompt can't see looks like one that isn't
    there. A prompt that isn't elevated doesn't see another account's
    task, which isn't this account's service, and may not see or change
    one that was registered from an elevated prompt. Seeing none is
    therefore no failure, and the answer says what such a prompt can't
    see; a task that is seen and can't be ended or deleted is one."""
    # `&&`, not `&`: by hand too, the task is deleted only once it is ended.
    by_hand = f'cmd /c "schtasks /End /TN {task} && schtasks /Delete /TN {task} /F"'
    where = "" if elevated else "from an elevated prompt"

    def failed(found: bool | None, problem: str) -> ServiceStop:
        return ServiceStop(found=found, problem=problem, stop_command=by_hand, stop_where=where)

    seen = run(["schtasks", "/Query", "/TN", task], _ASK_SECONDS)
    if seen.failure is not None:
        return failed(None, seen.failure)
    if not seen.succeeded:
        if elevated:
            return ServiceStop(
                found=False,
                said=f"The service wasn't set to start: no task {task} is registered.",
                stop_command=by_hand,
            )
        return ServiceStop(
            found=False,
            said=f"The service wasn't set to start: this account has no task {task} "
            "registered that a prompt without elevation sees. If the shim still starts "
            "at logon, the task was registered from an elevated prompt: run "
            f"`{UNINSTALL_COMMAND}` from one.",
            stop_command=by_hand,
        )
    ended = run(["schtasks", "/End", "/TN", task], _TASK_SCHEDULER_STOP_SECONDS)
    if not ended.succeeded:
        return failed(True, f"Task Scheduler didn't end the task {task}: {ended.said()}")
    deleted = run(["schtasks", "/Delete", "/TN", task, "/F"], _ASK_SECONDS)
    if not deleted.succeeded:
        return failed(
            True,
            f"Task Scheduler ended the task {task} but didn't delete it, so it starts "
            f"again at the next logon: {deleted.said()}",
        )
    return ServiceStop(
        found=True,
        said=f"Stopped the service: Task Scheduler ended and deleted the task {task}.",
        stop_command=by_hand,
    )


def stop_service(
    platform_name: str, *, entry_installed: bool, run: CommandRunner | None = None
) -> ServiceStop:
    """Stop the service this OS's autostart entry starts and unregister it,
    through the OS's service manager, waiting a bounded time for each step.

    Only that service: a shim someone started by hand in a terminal isn't
    the service manager's, and isn't touched."""
    run = run or run_command
    if platform_name == "darwin":
        return _stop_launchd(run, label=LAUNCHD_LABEL, uid=_user_id())
    if platform_name in ("linux", "wsl2"):
        return _stop_systemd(run, unit=SYSTEMD_UNIT_NAME, entry_installed=entry_installed)
    if platform_name == "windows":
        return _stop_scheduled_task(run, task=TASKSCHED_TASK_NAME, elevated=_is_elevated())
    raise ValueError(f"Unsupported platform for autostart install: {platform_name!r}")


def _forget_systemd_unit(
    target: Path, service: ServiceStop, removed: bool, run: CommandRunner
) -> tuple[str, ...]:
    """Finish a systemd uninstall once the unit file is gone: remove the
    link that enabled the unit if `systemctl disable` couldn't (it would
    enable a later install unasked), and have systemd read its units
    again, so that it forgets this one."""
    wanted = target.parent / "default.target.wants" / target.name
    # Only a link to a unit file that is no longer there: another unit of
    # this name, kept elsewhere, is not this entry's.
    unlinked = wanted.is_symlink() and not wanted.exists()
    if unlinked:
        wanted.unlink()
    if service.found is None or not (removed or unlinked or service.found):
        return ()
    reloaded = run(["systemctl", "--user", "daemon-reload"], _ASK_SECONDS)
    if reloaded.succeeded:
        return ()
    return (
        "systemd was not told to read its units again "
        f"({reloaded.said()}); it does at the next login.",
    )


# ── The LaunchAgent's earlier label ──────────────────────────────────────────
# REMOVE this section, the lines that call it in install_autostart and
# uninstall_autostart, and the tests of it in tests/test_launchd_label.py,
# once no Mac has a service under the earlier label.

# The label the LaunchAgent had before it was named for a domain the project
# owns. This line is the one place the package spells it
# (tests/test_packaging.py).
FORMER_LAUNCHD_LABEL = "net.autogpt.shim"


@dataclass(frozen=True)
class _FormerLaunchAgent:
    """What was found and done of a LaunchAgent under the earlier label."""

    # What to tell the person: nothing when there was no such service.
    notes: tuple[str, ...] = ()
    # The stop that failed, when the service may still be running: a loaded
    # one that didn't unload, or an entry whose service launchd couldn't be
    # asked about.
    stuck: ServiceStop | None = None
    # Its file and why it is still there, when it couldn't be removed: it
    # loads the earlier service again at the next login.
    entry_problem: str | None = None


def _forget_former_launchagent(home: Path | None, run: CommandRunner) -> _FormerLaunchAgent:
    """Stop, unload and remove a LaunchAgent left under the earlier label.

    Left alone it would start a second shim for the same computer at every
    login, beside the one under LAUNCHD_LABEL, and the platform lets one
    connection per computer stand."""
    plist = (home or Path.home()) / "Library" / "LaunchAgents" / f"{FORMER_LAUNCHD_LABEL}.plist"
    entry_installed = plist.is_file()
    service = _stop_launchd(run, label=FORMER_LAUNCHD_LABEL, uid=_user_id())
    notes: list[str] = []
    entry_problem = None
    if service.found and service.problem is None:
        notes.append(f"A service under this one's earlier label was loaded. {service.said}")
    if entry_installed:
        try:
            plist.unlink()
        except OSError as exc:
            entry_problem = f"{plist}: {exc}"
        else:
            notes.append(f"Removed the earlier autostart entry: {plist}")
    may_still_run = service.problem is not None and (
        service.found or (service.found is None and entry_installed)
    )
    return _FormerLaunchAgent(tuple(notes), service if may_still_run else None, entry_problem)


@dataclass(frozen=True)
class Uninstalled:
    """What `uninstall_autostart` found and did."""

    # The autostart entry as it is now.
    status: AutostartStatus
    # Whether the entry's file was there.
    entry_installed: bool
    service: ServiceStop
    # Why the entry's file couldn't be removed, when it couldn't.
    entry_problem: str | None = None
    notes: tuple[str, ...] = ()
    # macOS, for a LaunchAgent under the earlier label
    # (_forget_former_launchagent): `service` is its stop, which failed with
    # the service perhaps still running, whatever the entry under the new
    # label was; and why its file is still there, when it couldn't be
    # removed.
    former_may_still_run: bool = False
    former_entry_problem: str | None = None

    @property
    def may_still_run(self) -> bool:
        """Whether the service may still be running: it was there and
        something about stopping it failed, or the service manager couldn't
        be asked about an entry that was installed. With no entry installed
        and a service manager that couldn't be asked, there is no sign of a
        service."""
        if self.former_may_still_run:
            return True
        if self.service.problem is None:
            return False
        if self.service.found is None:
            return self.entry_installed
        return self.service.found


def uninstall_autostart(
    *,
    home: Path | None = None,
    platform_name: str | None = None,
    run: CommandRunner | None = None,
) -> Uninstalled:
    """Stop the service this OS's autostart entry starts, unregister it,
    then remove the entry's file.

    In that order. Removing the file first left nothing for the service
    manager's own commands to read (`launchctl unload <plist>`, `systemctl
    disable`), so the shim ran on, connected, until the next logout. The
    file is removed whatever the stop came to, and the result says what
    couldn't be done. Safe to run again: with nothing installed and
    nothing running it changes nothing.
    """
    plat = platform_name or _current_platform_for_install()
    target = target_path_for(plat, home)
    run = run or run_command
    installed = target.is_file()
    service = stop_service(plat, entry_installed=installed, run=run)
    entry_problem = None
    if installed:
        try:
            target.unlink()
        except OSError as exc:
            entry_problem = str(exc)
    notes: tuple[str, ...] = ()
    if plat in ("linux", "wsl2"):
        notes = _forget_systemd_unit(target, service, installed and entry_problem is None, run)
    former_may_still_run = False
    former_entry_problem = None
    if plat == "darwin":
        former = _forget_former_launchagent(home, run)
        notes = former.notes
        former_entry_problem = former.entry_problem
        if former.stuck is not None:
            former_may_still_run = True
            this_one_may_still_run = service.problem is not None and (
                service.found or (service.found is None and installed)
            )
            if this_one_may_still_run:
                notes += (
                    f"A service under this one's earlier label may still be running too: "
                    f"{former.stuck.problem}. Stop it with: {former.stuck.stop_command}",
                )
            else:
                # The exit status and the command to stop it by hand are
                # the earlier service's, and what went through for this
                # one is still said.
                if service.problem is None:
                    notes += (service.said,)
                service = former.stuck
    return Uninstalled(
        status=status_for(plat, home),
        entry_installed=installed,
        service=service,
        entry_problem=entry_problem,
        notes=notes,
        former_may_still_run=former_may_still_run,
        former_entry_problem=former_entry_problem,
    )


__all__ = [
    "FORMER_LAUNCHD_LABEL",
    "LAUNCHD_LABEL",
    "AutostartContext",
    "AutostartError",
    "AutostartStatus",
    "CommandResult",
    "ServiceStop",
    "Uninstalled",
    "build_context",
    "default_config_path",
    "install_autostart",
    "linux_systemd_unit_path",
    "macos_launchagent_path",
    "render_template",
    "resolve_shim_command",
    "run_command",
    "status_for",
    "stop_service",
    "target_path_for",
    "uninstall_autostart",
    "windows_task_xml_path",
    "write_linux_systemd_unit",
    "write_macos_launchagent",
    "write_windows_task_xml",
]
