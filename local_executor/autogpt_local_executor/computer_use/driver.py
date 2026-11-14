"""Runs cua-driver for the shim.

One `cua-driver mcp` process serves every session of a shim process. It
starts on first use, restarts after it stops (within a limit), and stops
with the shim. The shim launches it with a YAML permission policy that
allows only the catalog's tools, in cua-driver's `standard` permission mode,
with its telemetry and update checks off. See docs/CUA_DRIVER.md.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import logging
import os
import shutil
import stat
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from .. import platform_info
from ..protocol import ComputerDriverInfo, ErrorCode
from .catalog import REVIEWED_DRIVER_VERSION, policy_document, same_minor
from .mcp_client import McpClosedError, McpError, StdioMcpClient

logger = logging.getLogger(__name__)

DRIVER_NAME = "cua-driver"
# How to add computer use to a shim installed from Git, as the platform's
# setup card and the README install it. pipx leaves an installed package
# as it is without --force, extra or not. docs/COMPUTER_USE.md says the same.
# Double quotes, which cmd.exe, PowerShell and POSIX shells all take (the
# text has no $, backtick or !); cmd.exe would split single-quoted text at
# its spaces.
COMPUTER_USE_INSTALL = (
    'pipx install --force "autogpt-local-executor[computer-use] @ '
    'git+https://github.com/Significant-Gravitas/autogpt-local-executor.git"'
)
START_TIMEOUT_SECONDS = 30.0
LIST_TIMEOUT_SECONDS = 30.0
# A driver that keeps stopping is given up on until the window has passed.
MAX_STARTS_PER_WINDOW = 3
START_WINDOW_SECONDS = 300.0

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
    """The binary inside the installed `cua-driver` wheel. The package is
    found without importing it: its __init__ loads a native library."""
    try:
        spec = importlib.util.find_spec("cua_driver")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "bin" / _executable_name()
        if candidate.is_file():
            return candidate
    return None


def find_driver_binary(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """The cua-driver to run: an explicit path, else the bundled wheel's,
    else one on PATH. An explicit one is made absolute: Path drops the `./`
    of `./cua-driver`, and the OS looks a bare name up on PATH, which would
    run another install, or none, rather than the file checked here. So is
    one found on PATH, where shutil.which keeps a relative entry (`bin`):
    a Linux driver starts in an empty folder of its own, where
    `bin/cua-driver` names nothing."""
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
        # The empty folders Linux drivers started in (_new_working_folder).
        self._working_folders: list[Path] = []
        self._closed = False
        # The display variables the running driver started with.
        self._display: tuple[str | None, ...] | None = None
        # The look for the driver's binary under way, or the last (find_binary).
        self._looking: asyncio.Future[Path | None] | None = None
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

    async def find_binary(self) -> Path | None:
        """The driver this bridge runs, made executable, or None when it
        isn't installed: find_driver_binary and ensure_executable, in a
        worker thread. They stat files in the folders on sys.path, for the
        wheel's driver, and on PATH, and a stalled mount (NFS, FUSE) holds a
        stat for as long as it stalls; on the event loop that held every
        session's work and keepalive.

        One look at a time: a caller that gives up on a stalled look (a
        HELLO whose connection closed, a chat's start undone) leaves it to
        finish, and the next one waits for that look rather than start
        another, which would stall too, until the threads every session's
        file work shares were all held."""
        looking = self._looking
        if looking is None or looking.done():
            looking = self._looking = asyncio.ensure_future(
                asyncio.to_thread(_runnable_driver, self._explicit_binary)
            )
            looking.add_done_callback(_retrieved)
        return await asyncio.shield(looking)

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
            for folder in self._working_folders:
                # Removed only while empty: whatever is in one isn't the
                # shim's to delete. Writable again first, which Windows wants
                # of a folder it removes.
                with contextlib.suppress(OSError):
                    os.chmod(folder, stat.S_IRWXU)
                    os.rmdir(folder)
            self._working_folders.clear()

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
            binary = await self.find_binary()
            if binary is None:
                raise DriverUnavailable(
                    "cua-driver is not installed. Reinstall the local executor with its "
                    f"computer-use extra: {COMPUTER_USE_INSTALL}",
                    code=ErrorCode.DEPENDENCY_MISSING,
                    details={"dependency": DRIVER_NAME},
                )
            runtime = driver_runtime(self._platform)
            try:
                policy = self._write_policy()
                working_folder = self._new_working_folder()
            except OSError as exc:
                # A full or unwritable temporary directory: without its policy
                # the driver can't run, and HELLO goes on without computer use.
                self._last_failure = str(exc)
                raise DriverUnavailable(
                    f"could not make cua-driver's policy file or working folder: {exc}",
                    code=ErrorCode.INTERNAL_ERROR,
                ) from exc
            self._display = platform_info.session_display_values()
            env = driver_environment(policy)
            if working_folder is not None:
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

    def _new_working_folder(self) -> Path | None:
        """A new empty folder for a Linux driver to start in, which nothing
        can be written to; None elsewhere, where the driver starts in the
        shim's own folder.

        On Linux cua-driver hands a launch_app `name` that matches no
        installed app to xdg-open when a file or folder of that name is in
        its working folder, which was the shim's: the person's home under
        systemd. The shim lets a name through only when an installed app has
        it, but the driver looks again, and an app removed between the two
        left a file of that name open to xdg-open, past the path jail and the
        approval `urls` needs. In this folder no name finds anything. The
        apps the driver starts start here too, and a relative path in an
        approved argument names nothing here. Each start gets its own, as
        each gets its own policy folder, and they are kept until close(): an
        app the driver started may still be in one."""
        if not self._platform.startswith("linux"):
            return None
        folder = Path(tempfile.mkdtemp(prefix="autogpt-cua-driver-cwd-"))
        self._working_folders.append(folder)
        # Read-only, so it stays empty: an app the driver starts can't save
        # a file there, for a later name to open.
        os.chmod(folder, stat.S_IRUSR | stat.S_IXUSR)
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


def _retrieved(task: asyncio.Future[Any]) -> None:
    """Mark a start's failure seen, a look's (find_binary) or a stop's: the
    callers that waited for it may all have given up, and the next starts
    again."""
    if not task.cancelled():
        task.exception()


__all__ = [
    "COMPUTER_USE_INSTALL",
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
]
