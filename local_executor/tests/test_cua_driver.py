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
            await asyncio.wait_for(client.list_tools(timeout=0.5), timeout=5)
    finally:
        await client.close()
    assert time.monotonic() - started < 3  # its own deadline, not the guard's


async def test_a_request_past_its_deadline_is_cancelled_at_the_server(fake_log: Path) -> None:
    client = _fake_client([str(FAKE), "mcp"])
    await client.start(timeout=10)
    try:
        with pytest.raises(TimeoutError):
            await client.call_tool("drag", {"pid": -2}, timeout=0.3)
        for _ in range(200):
            if _records(fake_log, "slow_done"):
                break
            await asyncio.sleep(0.02)
    finally:
        await client.close()

    assert _records(fake_log, "cancelled")
    assert _records(fake_log, "slow_done") == [{"event": "slow_done", "cancelled": True}]


async def test_a_driver_that_exits_fails_the_waiting_request(fake_log: Path) -> None:
    client = _fake_client([str(FAKE), "mcp"])
    await client.start(timeout=10)
    try:
        with pytest.raises(McpClosedError, match="exit code 3"):
            await client.call_tool("drag", {"pid": -3}, timeout=10)
        assert not client.alive
    finally:
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
        ("launch_app", {"name": "Calculator", "urls": ["https://example.com"]}),
    ],
)
def test_ordinary_arguments_pass(tool: str, arguments: dict) -> None:
    catalog.check_arguments(tool, arguments)


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
        "run_script": ["script"],
    }


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
    make_handler, fake_log, tmp_path
) -> None:
    handler = make_handler()

    response = await handler.handle(_call("drag", {"pid": -2}, timeout_seconds=1))

    assert isinstance(response, ErrorMessage)
    assert response.payload.code == ErrorCode.COMMAND_TIMEOUT
    assert response.payload.details == {"tool": "drag", "timeout_seconds": 1}
    for _ in range(200):
        if _records(fake_log, "cancelled"):
            break
        await asyncio.sleep(0.02)
    assert _records(fake_log, "cancelled")


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

    def look() -> concealed.ClipboardState:
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


# ── Daemon ───────────────────────────────────────────────────────────────


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
