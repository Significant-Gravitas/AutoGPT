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
    else one on PATH."""
    if explicit:
        path = Path(explicit).expanduser()
        return path if path.is_file() else None
    bundled = bundled_binary()
    if bundled is not None:
        return bundled
    found = shutil.which(_executable_name())
    return Path(found) if found else None


def ensure_executable(path: Path) -> None:
    """pip can install the wheel's binary without its execute bit."""
    if sys.platform == "win32":
        return
    with contextlib.suppress(OSError):
        mode = path.stat().st_mode
        if not mode & stat.S_IXUSR:
            path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


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
    the app that starts the shim, as they did before cua-driver."""
    return "direct"


def driver_argv(binary: Path, runtime: Runtime, platform: str | None = None) -> list[str]:
    platform = sys.platform if platform is None else platform
    argv = [str(binary), "mcp"]
    if platform == "darwin" and runtime == "direct":
        # macOS's default would relay to CuaDriver.app.
        argv.append("--direct")
    return argv


def driver_environment(
    policy_path: Path | None, base: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The driver's environment: the shim's own, which the driver and the
    apps it launches need (the display, the session bus, the locale), with
    only the reviewed cua settings, telemetry and update checks off and, for
    `mcp`, the shim's policy."""
    env = {
        name: value
        for name, value in (os.environ if base is None else base).items()
        if not name.upper().startswith("CUA") or name.upper() in _INHERITED_CUA_SETTINGS
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
        # Held for each tool call: the driver runs one at a time anyway.
        self._turn = asyncio.Lock()
        self._starts: list[float] = []
        self._last_failure: str | None = None
        self._policy_dir: Path | None = None
        self._closed = False

    @property
    def offered_tools(self) -> frozenset[str]:
        return self._offered

    @property
    def info(self) -> ComputerDriverInfo | None:
        """The running driver, if one has started."""
        return self._snapshot.info if self._snapshot is not None else None

    async def snapshot(self) -> DriverSnapshot:
        await self._ensure_client()
        assert self._snapshot is not None
        return self._snapshot

    async def call_tool(
        self, name: str, arguments: dict[str, Any], *, timeout: float
    ) -> dict[str, Any]:
        """One tool call, which waits for its turn and runs within `timeout`
        seconds in all. Raises CallNotStarted when the deadline passes before
        the call reaches the driver.

        cua-driver 0.38 runs one request at a time and drops MCP's cancel
        notice, so a call past its deadline, or one its caller gave up on,
        would go on acting on the machine until it ended, and every request
        sent after it would wait. The bridge sends one call at a time, so the
        one the driver runs is the one that overran, and stops the driver to
        end it; the next call starts a new one."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        try:
            async with asyncio.timeout_at(deadline):
                await self._turn.acquire()
        except TimeoutError as exc:
            raise CallNotStarted(f"{name} waited its whole deadline for the driver") from exc
        try:
            client = await self._ensure_client()
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise CallNotStarted(f"{name}'s deadline passed while the driver started")
            try:
                return await client.call_tool(name, arguments, timeout=remaining)
            except McpClosedError as exc:
                self._last_failure = str(exc)
                raise DriverUnavailable(f"cua-driver stopped during the call: {exc}") from exc
            except (TimeoutError, asyncio.CancelledError):
                await asyncio.shield(self._stop_unfinished(client))
                raise
        finally:
            self._turn.release()

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            await self._stop_client()
            if self._policy_dir is not None:
                shutil.rmtree(self._policy_dir, ignore_errors=True)
                self._policy_dir = None

    # ── Starting ─────────────────────────────────────────────────────

    async def _ensure_client(self) -> StdioMcpClient:
        async with self._lock:
            if self._client is not None and self._client.alive:
                return self._client
            if self._closed:
                raise DriverUnavailable("the local executor is stopping")
            if self._client is not None:
                self._last_failure = self._client.stderr_tail(5) or self._last_failure
                await self._stop_client()
            self._check_restart_budget()
            binary = find_driver_binary(self._explicit_binary)
            if binary is None:
                raise DriverUnavailable(
                    "cua-driver is not installed. Reinstall the local executor with its "
                    f"computer-use extra: {COMPUTER_USE_INSTALL}",
                    code=ErrorCode.DEPENDENCY_MISSING,
                    details={"dependency": DRIVER_NAME},
                )
            ensure_executable(binary)
            runtime = driver_runtime(self._platform)
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
            client = self._client_factory(
                driver_argv(binary, runtime, self._platform),
                env=driver_environment(policy),
                client_version=self._client_version,
            )
            self._starts.append(time.monotonic())
            try:
                initialized = await client.start(timeout=START_TIMEOUT_SECONDS)
                if not same_minor(initialized.server_version):
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
            except (DriverUnavailable, asyncio.CancelledError):
                # The bridge doesn't hold this client yet, so close() can't
                # stop it: a cancelled start (shutdown) stops it here.
                await client.close()
                raise
            except (OSError, McpError, McpClosedError, TimeoutError) as exc:
                tail = client.stderr_tail(5)
                await client.close()
                self._last_failure = tail or str(exc)
                code = (
                    ErrorCode.DEPENDENCY_MISSING
                    if isinstance(exc, (FileNotFoundError, PermissionError))
                    else ErrorCode.INTERNAL_ERROR
                )
                raise DriverUnavailable(
                    f"cua-driver did not start: {exc}" + (f" ({tail})" if tail else ""),
                    code=code,
                ) from exc
            tools = tuple(
                tool
                for tool in listed
                if isinstance(tool.get("name"), str) and tool["name"] in self._offered
            )
            self._snapshot = DriverSnapshot(
                info=ComputerDriverInfo(
                    name=initialized.server_name or DRIVER_NAME,
                    version=initialized.server_version,
                    runtime=runtime,
                ),
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

    def _write_policy(self) -> Path:
        """The policy file lives as long as the bridge: the driver reads it
        the first time it authorizes a call, not at startup."""
        if self._policy_dir is None:
            self._policy_dir = Path(tempfile.mkdtemp(prefix="autogpt-cua-driver-"))
        path = self._policy_dir / "policy.yaml"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(policy_document(self._offered))
        return path

    async def _stop_client(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            with contextlib.suppress(Exception):
                await client.close()

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
