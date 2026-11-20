"""Runs cua-driver for the shim.

One `cua-driver mcp` process serves every session of a shim process. It
starts on first use, restarts after it stops (within a limit), and stops
with the shim. The shim launches it with a YAML permission policy that
allows only the catalog's tools, in cua-driver's `standard` permission mode,
with its telemetry and update checks off. See docs/CUA_DRIVER.md.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

from pydantic import ValidationError

from .. import platform_info
from ..protocol import ComputerDriverInfo, ErrorCode
from .catalog import REVIEWED_DRIVER_VERSION, policy_document, same_minor
from .mcp_client import McpClosedError, McpError, StdioMcpClient

logger = logging.getLogger(__name__)

DRIVER_NAME = "cua-driver"
# What the computer-use extra brings, as pyproject.toml lists it: everywhere,
# and on macOS as well. tests/test_install_hint.py holds the two together.
COMPUTER_USE_PACKAGES = (f"{DRIVER_NAME}=={REVIEWED_DRIVER_VERSION}",)
COMPUTER_USE_PACKAGES_MACOS = (
    "pyobjc-framework-ApplicationServices>=10.0",
    "pyobjc-framework-Cocoa>=10.0",
    "pyobjc-framework-Quartz>=10.0",
)


def computer_use_install(platform: str | None = None) -> str:
    """How to add computer use to a shim that is installed without it, as
    the platform's setup card installs it: the extra's packages are added
    to the install that is there. pip leaves an installed package as it
    is, whatever it was installed from (a Git revision, an index, a file),
    and fetches only what it is asked for. So the shim keeps its revision,
    which is the one its deployment's platform speaks the protocol of, and
    a download that fails leaves it as it was.

    The packages are named one by one and added with `pipx inject`, which
    records them as added to the shim's environment and leaves alone what
    pipx recorded the shim was installed from. pipx installs from that
    record again at `pipx reinstall` and `pipx upgrade`, and adds the
    injected packages back after it. Naming the extra instead (`pipx runpip
    <shim> install "<shim>[computer-use]"`) left the shim's files alone as
    well, but pipx then took `<shim>[computer-use]` for the shim's source,
    in place of the revision: the next reinstall or upgrade would have
    looked for the shim by name on the package index. Before that the hint
    was `pipx install --force` on the repository with no revision, which
    replaced the shim with whatever was on the default branch; `pipx
    install` without --force does nothing for a package pipx already has.
    docs/COMPUTER_USE.md says the same, and tests/test_install_hint.py
    holds the hint to it.

    Double quotes, which cmd.exe, PowerShell and POSIX shells all take
    (the text has no $, backtick or !, and `>` and `=` inside them are
    text to all three); cmd.exe would split single-quoted text at its
    spaces."""
    packages: tuple[str, ...] = COMPUTER_USE_PACKAGES
    if (sys.platform if platform is None else platform) == "darwin":
        packages += COMPUTER_USE_PACKAGES_MACOS
    return "pipx inject autogpt-local-executor " + " ".join(f'"{package}"' for package in packages)


# The hint for the computer this runs on.
COMPUTER_USE_INSTALL = computer_use_install()
START_TIMEOUT_SECONDS = 30.0
LIST_TIMEOUT_SECONDS = 30.0
# A driver that keeps stopping is given up on until the window has passed.
MAX_STARTS_PER_WINDOW = 3
START_WINDOW_SECONDS = 300.0
# How long a Linux driver's start waits for the look at the person's home
# folder (CuaDriverBridge._working_folder) before it goes on in the shim's
# own folder: a home on a mount that isn't answering holds the look for as
# long as it stalls.
HOME_FOLDER_SECONDS = 2.0
# How long a start waits for the look for the driver's binary (find_binary),
# holding the bridge's lock: as long as HELLO waits for it
# (handlers.DRIVER_LOOKUP_SECONDS).
RESTART_LOOKUP_SECONDS = 5.0
# How long the answer of a look that outlasted a caller's wait is kept for
# the callers that come next (CuaDriverBridge.find_binary): long enough for
# the request after the one that gave up, short enough that what it found
# is still where it was.
LATE_ANSWER_SECONDS = 60.0

# The cua settings (CUA*) the driver may inherit from the shim's environment:
# the ones cua's own SDK hands a driver it embeds (cua-driver-sdk's
# `allowed_environment_name`: logging, the driver's home, accessibility and
# Wayland switches, input pacing), and an administrator's managed policy,
# which can only narrow what the driver allows. cua-driver reads well over a
# hundred others, and among them some turn a protection off
# (CUA_DRIVER_BROWSER_NO_SANDBOX, CUA_DRIVER_DISABLE_KEY_FOCUS_GUARD), open a
# port (CUA_DRIVER_RS_MCP_HTTP_PORT), move what the driver owns
# (CUA_DRIVER_BROWSER_PROFILE_ROOT), approve on the person's behalf
# (CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS, a capability manifest) or exist
# for cua's own tests. Those never reach the shim's driver.
_INHERITED_CUA_SETTINGS = frozenset(
    {
        "CUA_LOG",
        "CUA_DRIVER_RS_HOME",
        "CUA_DRIVER_RS_ENABLE_WAYLAND",
        "CUA_DRIVER_RS_DISABLE_A11Y_ADVERTISE",
        "CUA_DRIVER_RS_A11Y_ADVERTISE_MODE",
        "CUA_DRIVER_WINDOW_CHANGE_TIMEOUT_MS",
        "CUA_DRIVER_WINDOW_CHANGE_POLL_MS",
        "CUA_DRIVER_KEY_GAP_MS",
        "CUA_DRIVER_BATCH_PACING",
        "CUA_DRIVER_MANAGED_POLICY_FILE",
    }
)

# The shim's own settings, its configuration and the keychain passphrase.
_SHIM_SETTINGS_PREFIX = "AUTOGPT_SHIM_"

Runtime = Literal["direct", "service"]


class CallNotStarted(TimeoutError):
    """A call's deadline passed before the driver took it: nothing ran."""


class DriverUnavailable(Exception):
    """cua-driver could not serve a request. `code` is the wire error."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode = ErrorCode.INTERNAL_ERROR,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


@dataclass(frozen=True)
class DriverSnapshot:
    """What the running driver offers, filtered to the catalog."""

    info: ComputerDriverInfo
    instructions: str
    tools: tuple[dict[str, Any], ...]

    def tool(self, name: str) -> dict[str, Any] | None:
        return next((tool for tool in self.tools if tool.get("name") == name), None)


def _executable_name() -> str:
    return "cua-driver.exe" if sys.platform == "win32" else "cua-driver"


def bundled_binary() -> Path | None:
    """The binary inside the installed `cua-driver` wheel, which puts it at
    `cua_driver/bin/` in a folder Python imports from: the first folder of
    sys.path that holds it there, as an absolute path.

    Looked for in those folders directly, not through the import system.
    importlib.util.find_spec holds the interpreter's import lock while each
    of its finders looks, and a finder stats its folder. With a folder of
    sys.path on a mount that stopped answering, this look, in its thread,
    held that lock for as long as the mount stalled, and the first import of
    any module on any other thread waits for that lock. The event loop's
    thread is one of them: the loop stopped, and no caller's bound on the
    look could run out. A folder that stalls still holds this look, and
    nothing else.

    Nor is the package imported: its __init__ loads a native library."""
    name = _executable_name()
    for folder in list(sys.path):
        if not isinstance(folder, str):
            continue
        # An empty entry is the current folder, as Python reads it.
        candidate = Path(folder or os.curdir, "cua_driver", "bin", name)
        try:
            mode = os.stat(candidate).st_mode
        except (OSError, ValueError):
            # Not there, or the entry is no folder (an archive), or it
            # can't be read.
            continue
        if stat.S_ISREG(mode):
            return candidate.absolute()
    return None


def find_driver_binary(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """The cua-driver to run: an explicit path, else the bundled wheel's,
    else one on PATH. An explicit one is made absolute: Path drops the `./`
    of `./cua-driver`, and the OS looks a bare name up on PATH, which would
    run another install, or none, rather than the file checked here. So is
    one found on PATH, where shutil.which keeps a relative entry (`bin`):
    `bin/cua-driver` names the driver only from the shim's own folder, and
    a Linux driver starts in the person's home folder (home_folder)."""
    if explicit:
        try:
            path = Path(explicit).expanduser().absolute()
        except RuntimeError as exc:
            # `~someone` with no such account, or a home Python can't find:
            # a driver that isn't there, which every caller reports.
            logger.warning("Can't expand the cua-driver path %r: %s", str(explicit), exc)
            return None
        return path if path.is_file() else None
    bundled = bundled_binary()
    if bundled is not None:
        return bundled
    found = shutil.which(_executable_name())
    return Path(found).absolute() if found else None


def ensure_executable(path: Path) -> None:
    """pip can install the wheel's binary without its execute bit."""
    if sys.platform == "win32":
        return
    with contextlib.suppress(OSError):
        mode = path.stat().st_mode
        if not mode & stat.S_IXUSR:
            path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _runnable_driver(explicit: str | os.PathLike[str] | None) -> Path | None:
    """find_driver_binary's driver, made executable (ensure_executable)."""
    binary = find_driver_binary(explicit)
    if binary is not None:
        ensure_executable(binary)
    return binary


def _account_home() -> str | None:
    """This account's home folder in the user database, or None when the
    database has no entry for the account, or the OS has no such database."""
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_dir
    except (ImportError, KeyError):
        return None


def home_folder() -> Path | None:
    """The person's home folder, for a Linux driver to start in: `HOME`, as a
    shell's `~` is, or without one the account's own in the user database.
    None when it can't be used: neither names one, or what is named isn't
    an absolute path to a folder this process can enter. It looks at the
    disk, where a home on a mount that isn't answering holds it."""
    home = os.environ.get("HOME") or _account_home()
    if not home:
        return None
    path = Path(home)
    if not path.is_absolute() or not os.path.isdir(path) or not os.access(path, os.X_OK):
        return None
    return path


def driver_runtime(platform: str | None = None) -> Runtime:
    """Always `direct`: the driver process the shim starts owns the runtime.

    On macOS, cua-driver's default is to relay to CuaDriver.app's service.
    The shim never does: that daemon is installed and updated by the person,
    not pinned with the shim (it can be another release than the one the
    catalog was reviewed against, which the relay can't see), it is shared
    with their other cua-driver clients, and it runs with whatever
    permission mode and policy it was started with, so the calls a script
    makes inside it wouldn't be bound by the shim's policy. Owning the
    runtime keeps all of that the shim's. The macOS grants then belong to
    the app that starts the shim, as they did before cua-driver. On Linux
    and Windows, `mcp` relays to such a daemon too once the person has let
    the driver's Computer History preview in (a preference file cua-driver
    0.38 reads at start), so the shim says `--direct` there as well."""
    return "direct"


def driver_argv(binary: Path, runtime: Runtime, platform: str | None = None) -> list[str]:
    argv = [str(binary), "mcp"]
    if runtime == "direct":
        # Without it, macOS relays to CuaDriver.app, and Linux and Windows to
        # a daemon once the Computer History preview is let in.
        argv.append("--direct")
    return argv


def driver_environment(
    policy_path: Path | None, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The driver's environment: the shim's own, which the driver and the
    apps it launches need (the display, the session bus, the locale), with
    only the reviewed cua settings, telemetry and update checks off and, for
    `mcp`, the shim's policy. The shim's own settings (`AUTOGPT_SHIM_*`, the
    keychain passphrase among them) stay out: nothing the driver starts
    needs them, and an app it launches by name would inherit them, where a
    terminal shows its environment to anyone who looks."""
    env = {
        name: value
        for name, value in (os.environ if base is None else base).items()
        if not name.upper().startswith(_SHIM_SETTINGS_PREFIX)
        and (not name.upper().startswith("CUA") or name.upper() in _INHERITED_CUA_SETTINGS)
    }
    env.update(
        {
            "CUA_DRIVER_PERMISSION_MODE": "standard",
            "CUA_DRIVER_DISABLE_UNRESTRICTED": "1",
            # Content-free product telemetry is on by default; DO_NOT_TRACK
            # outranks every other setting.
            "DO_NOT_TRACK": "1",
            "CUA_DRIVER_RS_TELEMETRY_ENABLED": "false",
            "CUA_DRIVER_RS_UPDATE_CHECK": "false",
        }
    )
    if policy_path is not None:
        env["CUA_DRIVER_POLICY_FILE"] = str(policy_path)
    return env


ClientFactory = Callable[..., StdioMcpClient]


class CuaDriverBridge:
    """The shim's one cua-driver process."""

    def __init__(
        self,
        *,
        tools: frozenset[str],
        binary: str | os.PathLike[str] | None = None,
        platform: str | None = None,
        client_factory: ClientFactory = StdioMcpClient,
        client_version: str = "0",
    ) -> None:
        self._offered = tools
        self._explicit_binary = binary
        self._platform = sys.platform if platform is None else platform
        self._client_factory = client_factory
        self._client_version = client_version
        self._client: StdioMcpClient | None = None
        self._snapshot: DriverSnapshot | None = None
        self._lock = asyncio.Lock()
        # A start a call waits for up to its deadline, then leaves running.
        self._start: asyncio.Task[StdioMcpClient] | None = None
        # Held for each tool call: the driver runs one at a time anyway.
        self._turn = asyncio.Lock()
        # Stops of drivers whose call overran, which the caller didn't wait
        # for (call_tool): each holds the turn until its driver is gone.
        self._stops: set[asyncio.Future[None]] = set()
        self._starts: list[float] = []
        self._last_failure: str | None = None
        self._policy_dir: Path | None = None
        self._closed = False
        # The display variables the running driver started with.
        self._display: tuple[str | None, ...] | None = None
        # The look for the driver's binary under way, or the last (find_binary).
        self._look: _Look | None = None
        # The look at the person's home folder under way, or the last
        # (_working_folder).
        self._home_looking: asyncio.Future[Path | None] | None = None
        # Whether a session's preflight has found the macOS grants computer
        # use needs: HELLO offers computer use on macOS only once one has.
        self.permissions_granted = False

    @property
    def platform(self) -> str:
        """The OS the driver runs on, as sys.platform names it."""
        return self._platform

    @property
    def offered_tools(self) -> frozenset[str]:
        return self._offered

    @property
    def info(self) -> ComputerDriverInfo | None:
        """The running driver, if one has started."""
        return self._snapshot.info if self._snapshot is not None else None

    @property
    def last_snapshot(self) -> DriverSnapshot | None:
        """What the driver offered when it last started, if it has: kept
        while it restarts, since a start lists the same tools."""
        return self._snapshot

    async def find_binary(self, *, to_start: bool = False) -> Path | None:
        """The driver this bridge runs, made executable, or None when it
        isn't installed: find_driver_binary and ensure_executable, in a
        thread of their own. They stat files in the folders on sys.path, for
        the wheel's driver, and on PATH, and a stalled mount (NFS, FUSE)
        holds a stat for as long as it stalls; on the event loop that held
        every session's work and keepalive.

        One look at a time: a caller that gives up on a stalled look (a
        HELLO whose connection closed, a chat's start undone, a start past
        RESTART_LOOKUP_SECONDS) leaves it to finish, and the next one waits
        for that look rather than start another, which would stall too. So
        a stall holds one thread, however many callers come.

        The answer of a look that a caller gave up on is kept when it comes,
        for LATE_ANSWER_SECONDS: the callers of that time take it without
        another look. With a look that takes longer than its callers wait,
        every time (a slow share, not a dead one), each caller gave up, the
        answer came with nobody waiting, and the next caller began a look
        that took as long: the driver was there and never started. A start
        (`to_start`) uses a kept answer up, so the start after it looks
        again and finds a driver that was moved or upgraded; a caller that
        only asks whether the driver is there (HELLO, the preflight, the
        display watch) leaves it for the start that follows. A look that
        every caller waited out, the usual one, keeps nothing: the next
        caller looks afresh.

        That thread is a daemon thread, not one of the event loop's workers
        (_in_a_thread_of_its_own): a look still held when the shim is told
        to stop doesn't keep it from stopping. Every caller bounds its own
        wait."""
        look = self._look
        if look is None or (look.answer.done() and not look.kept()):
            explicit = self._explicit_binary
            look = self._look = _Look(lambda: _runnable_driver(explicit))
        try:
            return await asyncio.shield(look.answer)
        except asyncio.CancelledError:
            # This caller stopped waiting, so the answer is the next one's.
            look.outlasted_a_caller = True
            to_start = False
            raise
        finally:
            if to_start:
                look.used_by_a_start = True

    async def snapshot(self, *, deadline: float | None = None) -> DriverSnapshot:
        """What the driver offers, starting it if need be. With `deadline`
        (loop time), a start still running then raises CallNotStarted."""
        if deadline is None:
            await self._ensure_client()
        else:
            await self._client_by(deadline, "the tool list")
        assert self._snapshot is not None
        return self._snapshot

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        timeout: float,
        start_within: float | None = None,
        wait_for_stop: bool = True,
    ) -> dict[str, Any]:
        """One tool call, which waits for its turn and runs within `timeout`
        seconds in all. Raises CallNotStarted when the deadline passes before
        the call reaches the driver. `start_within` bounds the wait for the
        turn and the driver's start alone, for a caller that would rather not
        run than wait long; the run then has what is left of `timeout`.

        cua-driver 0.38 runs one request at a time and drops MCP's cancel
        notice, so a call past its deadline, or one its caller gave up on,
        would go on acting on the machine until it ended, and every request
        sent after it would wait. The bridge sends one call at a time, so the
        one the driver runs is the one that overran, and stops the driver to
        end it; the next call starts a new one. A stop can take seconds (a
        driver slow to go is killed after KILL_GRACE_SECONDS), and the call
        waits for it unless `wait_for_stop` is false, for a caller that must
        be done by its deadline (HELLO's screen-size probe): the stop then
        goes on, keeping the turn until the driver is gone, so the next call
        still waits for it."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        start_by = deadline if start_within is None else min(deadline, loop.time() + start_within)
        try:
            async with asyncio.timeout_at(start_by):
                await self._turn.acquire()
        except TimeoutError as exc:
            raise CallNotStarted(f"{name} waited as long as it could for the driver") from exc
        # A stop still under way when the caller gives up: the turn is let go
        # only once it is done (below).
        stopping: asyncio.Future[None] | None = None
        try:
            client = await self._client_by(start_by, name, call_deadline=deadline)
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise CallNotStarted(f"{name}'s deadline passed while the driver started")
            try:
                return await client.call_tool(name, arguments, timeout=remaining)
            except McpClosedError as exc:
                self._last_failure = str(exc)
                raise DriverUnavailable(f"cua-driver stopped during the call: {exc}") from exc
            except (TimeoutError, asyncio.CancelledError):
                stopping = asyncio.ensure_future(self._stop_unfinished(client))
                self._stops.add(stopping)
                stopping.add_done_callback(self._stops.discard)
                if wait_for_stop:
                    await asyncio.shield(stopping)
                raise
        finally:
            if stopping is not None and not stopping.done():
                # Cancelled again while the driver stops: the next call would
                # start another driver while this one still acts on the desktop.
                stopping.add_done_callback(lambda _stopped: self._turn.release())
            else:
                self._turn.release()

    async def close(self) -> None:
        start = self._start
        if start is not None and not start.done():
            # Shutting down: a start no call waits for any more is stopped.
            start.cancel()
            with contextlib.suppress(BaseException):
                await start
        if self._stops:
            # A driver still being stopped is no longer the bridge's client,
            # which the stop below ends: it would outlive the shim.
            await asyncio.gather(*self._stops, return_exceptions=True)
        async with self._lock:
            self._closed = True
            await self._stop_client()
            if self._policy_dir is not None:
                shutil.rmtree(self._policy_dir, ignore_errors=True)
                self._policy_dir = None

    # ── Starting ─────────────────────────────────────────────────────

    async def _client_by(
        self, deadline: float, name: str, *, call_deadline: float | None = None
    ) -> StdioMcpClient:
        """The running driver, started if need be, by `deadline` (loop time).

        A start still running at the deadline goes on without the caller,
        who gets CallNotStarted: stopping it part way would count against
        the restart budget, and the next call finds the driver ready. Every
        caller in the meantime waits for that one start.

        The look at the desktop's display first (systemctl, under the
        systemd service) is no wait for anyone else, so it ends at the
        call's own deadline, `call_deadline` (`deadline` if not given): one
        that hangs can't hold the driver's turn past it, and a call whose
        start window went on waiting for the turn still reaches a driver
        that is running. Every call waits for the one look under way
        (_display_followed)."""
        try:
            await _display_followed(deadline if call_deadline is None else call_deadline)
        except TimeoutError:
            raise CallNotStarted(
                f"{name}'s deadline passed while the desktop's display was checked"
            ) from None
        client = self._client
        if client is not None and client.alive and self._same_display():
            return client
        start = self._start
        if start is None or start.done():
            start = self._start = asyncio.ensure_future(self._ensure_client())
            start.add_done_callback(_retrieved)
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining > 0:
            await asyncio.wait({start}, timeout=remaining)
        if not start.done():
            raise CallNotStarted(f"{name}'s deadline passed while the driver started")
        if start.cancelled():
            raise DriverUnavailable("the local executor is stopping")
        return start.result()

    async def _ensure_client(self) -> StdioMcpClient:
        # The look at the display under way, for MANAGER_WAIT_SECONDS at
        # most: HELLO waits for the owner's first start in full. Past that
        # the driver starts with the display as it is, and one the look
        # brings later restarts it at the next call (_same_display).
        with contextlib.suppress(TimeoutError):
            await _display_followed(
                asyncio.get_running_loop().time() + platform_info.MANAGER_WAIT_SECONDS
            )
        async with self._lock:
            if self._client is not None and self._client.alive:
                if self._same_display():
                    return self._client
                # A new login brought another display (on GNOME, another
                # XAUTHORITY too), and a driver keeps the environment it
                # started with. Stopping it is the shim's own doing, so its
                # start comes off the restart budget.
                logger.info("The desktop's display changed; restarting cua-driver")
                await self._stop_client()
                if self._starts:
                    self._starts.pop()
            if self._closed:
                raise DriverUnavailable("the local executor is stopping")
            if self._client is not None:
                self._last_failure = self._client.stderr_tail(5) or self._last_failure
                await self._stop_client()
            self._check_restart_budget()
            # A fresh look at each start, so a driver that was upgraded or
            # moved is found, by the usual order (the configured path, the
            # wheel's, PATH). It is the bridge's one look, waited for no
            # longer than RESTART_LOOKUP_SECONDS: this start holds the lock,
            # and with no limit a folder on PATH or sys.path on a mount that
            # stopped answering held it for good, and every call, stop and
            # close() behind it. Past the limit the start fails, the look
            # goes on in its thread, and the next start waits for that same
            # look, or takes its answer if it has come since (find_binary):
            # the one start that doesn't look afresh. Nothing here counts
            # against the restart budget: no driver was started.
            try:
                async with asyncio.timeout(RESTART_LOOKUP_SECONDS):
                    binary = await self.find_binary(to_start=True)
            except TimeoutError:
                raise DriverUnavailable(
                    f"Looking for cua-driver took longer than {RESTART_LOOKUP_SECONDS:g} s: "
                    "a folder on PATH, or one Python imports from, may be on a disk that "
                    "isn't answering. Nothing was started; try again once it answers.",
                ) from None
            if binary is None:
                raise DriverUnavailable(
                    "cua-driver is not installed. Add computer use's packages to this "
                    "install, which leaves the local executor itself as it is: "
                    f"{COMPUTER_USE_INSTALL}",
                    code=ErrorCode.DEPENDENCY_MISSING,
                    details={"dependency": DRIVER_NAME},
                )
            runtime = driver_runtime(self._platform)
            working_folder = await self._working_folder()
            try:
                policy = self._write_policy()
            except OSError as exc:
                # A full or unwritable temporary directory: without its policy
                # the driver can't run, and HELLO goes on without computer use.
                self._last_failure = str(exc)
                raise DriverUnavailable(
                    f"could not write the cua-driver policy file: {exc}",
                    code=ErrorCode.INTERNAL_ERROR,
                ) from exc
            self._display = platform_info.session_display_values()
            env = driver_environment(policy)
            if working_folder is not None:
                # As a shell has it after `cd`: a program that goes by PWD
                # would otherwise take the shim's folder for its own.
                env["PWD"] = str(working_folder)
            client = self._client_factory(
                driver_argv(binary, runtime, self._platform),
                env=env,
                cwd=working_folder,
                client_version=self._client_version,
            )
            self._starts.append(time.monotonic())
            try:
                initialized = await client.start(timeout=START_TIMEOUT_SECONDS)
                if not same_minor(initialized.server_version):
                    # A driver of another version didn't crash: its start
                    # comes off the restart budget, which would otherwise
                    # report stops in place of the version it needs.
                    self._starts.pop()
                    raise DriverUnavailable(
                        f"This local executor works with cua-driver {REVIEWED_DRIVER_VERSION}; "
                        f"found {initialized.server_version or 'an unknown version'} at "
                        f"{binary}.",
                        code=ErrorCode.DEPENDENCY_MISSING,
                        details={
                            "dependency": DRIVER_NAME,
                            "found_version": initialized.server_version,
                            "supported_version": REVIEWED_DRIVER_VERSION,
                        },
                    )
                listed = await client.list_tools(timeout=LIST_TIMEOUT_SECONDS)
                try:
                    info = ComputerDriverInfo(
                        name=initialized.server_name or DRIVER_NAME,
                        version=initialized.server_version,
                        runtime=runtime,
                    )
                except ValidationError as exc:
                    # A name or version HELLO can't carry, from a build the
                    # shim wasn't made for: like another version, off the
                    # restart budget, and stopped by the cleanup below.
                    self._starts.pop()
                    raise DriverUnavailable(
                        "cua-driver reported a name or version this local executor can't "
                        f"pass on: {initialized.server_name[:80]!r}, "
                        f"{initialized.server_version[:80]!r}",
                        code=ErrorCode.DEPENDENCY_MISSING,
                        details={"dependency": DRIVER_NAME},
                    ) from exc
            except (DriverUnavailable, asyncio.CancelledError):
                # The bridge doesn't hold this client yet, so close() can't
                # stop it: a cancelled start (shutdown) stops it here.
                await client.close()
                raise
            except (OSError, McpError, McpClosedError, TimeoutError) as exc:
                tail = client.stderr_tail(5)
                await client.close()
                self._last_failure = tail or str(exc)
                missing = isinstance(exc, (FileNotFoundError, PermissionError))
                if missing:
                    # A driver that can't be run didn't crash either.
                    self._starts.pop()
                raise DriverUnavailable(
                    f"cua-driver did not start: {exc}" + (f" ({tail})" if tail else ""),
                    code=ErrorCode.DEPENDENCY_MISSING if missing else ErrorCode.INTERNAL_ERROR,
                ) from exc
            tools = tuple(
                tool
                for tool in listed
                if isinstance(tool.get("name"), str) and tool["name"] in self._offered
            )
            self._snapshot = DriverSnapshot(
                info=info,
                instructions=initialized.instructions,
                tools=tools,
            )
            self._client = client
            logger.info(
                "cua-driver %s started (%s runtime, pid %s, %d tools)",
                initialized.server_version,
                runtime,
                client.pid,
                len(tools),
            )
            return client

    def _same_display(self) -> bool:
        return self._display == platform_info.session_display_values()

    def _check_restart_budget(self) -> None:
        now = time.monotonic()
        self._starts = [t for t in self._starts if now - t < START_WINDOW_SECONDS]
        if len(self._starts) >= MAX_STARTS_PER_WINDOW:
            detail = f": {self._last_failure}" if self._last_failure else ""
            raise DriverUnavailable(
                f"cua-driver stopped {len(self._starts)} times in "
                f"{int(START_WINDOW_SECONDS // 60)} minutes{detail}. "
                "Run `autogpt-shim doctor` on this machine.",
            )

    async def _working_folder(self) -> Path | None:
        """The folder the driver starts in, or None for the shim's own.

        On Linux it is the person's home folder (home_folder), however the
        shim was started. The apps cua-driver starts there inherit its
        folder: a terminal opens in it, and a save dialog starts in it. The
        shim's own folder is the home folder under the systemd service, but
        for a shim started from a terminal it is whatever folder that
        terminal was in. On macOS and Windows the driver keeps the shim's.

        A home folder that can't be used doesn't fail the start: the driver
        starts in the shim's own folder then. So it does when the look at
        the home folder isn't back within HOME_FOLDER_SECONDS (a mount that
        isn't answering). The look runs in a thread of its own, one at a
        time: a start that went on without it leaves it to finish, and the
        next start waits for that look rather than hold another thread in
        one of its own. That thread is a daemon thread, not one of the event
        loop's workers (_in_a_thread_of_its_own): a look still held when the
        shim is told to stop doesn't keep it from stopping. A home folder
        that goes between the look and the start fails that start; the next
        one looks again."""
        if not self._platform.startswith("linux"):
            return None
        looking = self._home_looking
        if looking is None or looking.done():
            looking = self._home_looking = _in_a_thread_of_its_own(home_folder, "home-folder")
            looking.add_done_callback(_retrieved)
        done, _ = await asyncio.wait({looking}, timeout=HOME_FOLDER_SECONDS)
        if not done:
            logger.warning(
                "Looking at the home folder took longer than %g s; cua-driver, and the apps "
                "it starts, start in the local executor's own folder instead",
                HOME_FOLDER_SECONDS,
            )
            return None
        folder = looking.result()
        if folder is None:
            logger.info(
                "No home folder to start in (HOME names no folder this account can enter); "
                "cua-driver, and the apps it starts, start in the local executor's own folder"
            )
        return folder

    def _write_policy(self) -> Path:
        """Write the policy in a new private folder for each driver start.
        The driver reads it the first time it authorizes a call, not at
        startup, so a folder lives as long as its driver, and close() removes
        the last one. A folder made once, at the first start, was gone after
        a temporary-file cleaner purged it while the shim ran, and its name,
        once free, could be another user's folder, so none is reused."""
        previous, self._policy_dir = (
            self._policy_dir,
            Path(tempfile.mkdtemp(prefix="autogpt-cua-driver-")),
        )
        if previous is not None:
            shutil.rmtree(previous, ignore_errors=True)
        path = self._policy_dir / "policy.yaml"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(policy_document(self._offered))
        return path

    async def _stop_client(self) -> None:
        """Stop the bridge's driver, if it has one, and wait until it is gone.

        The stop is a task of its own, which a cancel doesn't cut short, and
        one close() waits for (_stops). A display-change restart that
        close() cancelled while the old driver stopped ended the stop in its
        grace period, before the driver was terminated, and the driver, no
        longer the bridge's client, ran on after close(). A caller cancelled
        meanwhile still waits for the stop, then lets the cancel go on."""
        client, self._client = self._client, None
        if client is None:
            return
        stopping = asyncio.ensure_future(client.close())
        stopping.add_done_callback(_retrieved)
        self._stops.add(stopping)
        stopping.add_done_callback(self._stops.discard)
        cancelled = False
        while not stopping.done():
            try:
                await asyncio.shield(stopping)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                # A driver whose stop failed is let go of all the same.
                pass
        if cancelled:
            raise asyncio.CancelledError

    async def _stop_unfinished(self, client: StdioMcpClient) -> None:
        """Stop a driver still running a call nobody waits for. Stopping it
        is the shim's own doing, not the driver failing, so its start is
        taken off the restart budget."""
        async with self._lock:
            if self._client is client:
                self._client = None
                if self._starts:
                    self._starts.pop()
        logger.info("Stopping cua-driver (pid %s) to end a call that overran", client.pid)
        with contextlib.suppress(Exception):
            await client.close(now=True)


async def _display_followed(deadline: float) -> None:
    """Wait until `deadline` (loop time) for the look at the desktop's
    display (SessionDisplay.refresh): the one under way, else a new one when
    one is due. Past the deadline, TimeoutError, and the look goes on in its
    own thread. Nothing waits for it in a thread: a call that gave up on a
    look that hung (systemctl looked up on a stalled PATH mount) left a
    worker thread running it, and each later call another, waiting for the
    first, until the threads every session's file and audit work shares
    were all held."""
    async with asyncio.timeout_at(deadline):
        await asyncio.wrap_future(platform_info.session_display.refresh())


_T = TypeVar("_T")


def _in_a_thread_of_its_own(look: Callable[[], _T], name: str) -> asyncio.Future[_T]:
    """What `look()` returns or raises, run in a daemon thread of its own,
    as a future of the running loop.

    Not in one of the event loop's worker threads (asyncio.to_thread):
    asyncio.run waits for those before it returns, without a limit on Python
    3.11, and the interpreter waits for them again as it exits. A look held
    by a mount that isn't answering, in one of them, kept the shim from
    stopping when it was told to, and its supervisor from starting it again.
    Neither waits for a daemon thread."""
    done: concurrent.futures.Future[_T] = concurrent.futures.Future()
    # Running from here on: a waiter that gives up can't cancel it.
    done.set_running_or_notify_cancel()

    def work() -> None:
        try:
            result = look()
        except BaseException as exc:
            done.set_exception(exc)
        else:
            done.set_result(result)

    threading.Thread(target=work, name=f"autogpt-shim-{name}", daemon=True).start()
    return asyncio.wrap_future(done)


class _Look:
    """One look for the driver's binary (CuaDriverBridge.find_binary): its
    answer, when that came, and what its callers did with it."""

    def __init__(self, find: Callable[[], Path | None]) -> None:
        # When the answer came (time.monotonic), written by the look's
        # thread before the answer is, so whoever sees the answer sees it.
        self.answered_at: float | None = None
        # Whether a caller stopped waiting before it had the answer.
        self.outlasted_a_caller = False
        # Whether a start has had the answer.
        self.used_by_a_start = False

        def look() -> Path | None:
            try:
                return find()
            finally:
                self.answered_at = time.monotonic()

        self.answer = _in_a_thread_of_its_own(look, "driver-lookup")
        self.answer.add_done_callback(_retrieved)

    def kept(self) -> bool:
        """Whether a caller that comes now takes this look's answer rather
        than look again: a caller gave up on the look, no start has had
        its answer, and the answer is no older than LATE_ANSWER_SECONDS."""
        return (
            self.outlasted_a_caller
            and not self.used_by_a_start
            and self.answered_at is not None
            and time.monotonic() - self.answered_at < LATE_ANSWER_SECONDS
        )


def _retrieved(task: asyncio.Future[Any]) -> None:
    """Mark a start's failure seen, a look's (find_binary, _working_folder)
    or a stop's: the callers that waited for it may all have given up, and
    the next starts again."""
    if not task.cancelled():
        task.exception()


__all__ = [
    "COMPUTER_USE_INSTALL",
    "COMPUTER_USE_PACKAGES",
    "COMPUTER_USE_PACKAGES_MACOS",
    "computer_use_install",
    "DRIVER_NAME",
    "CallNotStarted",
    "CuaDriverBridge",
    "DriverSnapshot",
    "DriverUnavailable",
    "bundled_binary",
    "driver_argv",
    "driver_environment",
    "driver_runtime",
    "find_driver_binary",
    "home_folder",
]
