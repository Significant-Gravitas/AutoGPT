"""The reviewed catalog against the cua-driver this package pins, run for
real. Skipped where the computer-use extra isn't installed; CI installs it
on every OS."""

from __future__ import annotations

import importlib.util
import re
import sys

import pytest

from autogpt_local_executor import platform_info
from autogpt_local_executor.computer_use import catalog, driver
from autogpt_local_executor.computer_use.driver import CuaDriverBridge
from autogpt_local_executor.computer_use.mcp_client import StdioMcpClient

# Skipped only when the cua-driver package isn't there at all. Skipping
# whenever its binary was missing let a wheel that installed without one
# turn CI green with none of these run.
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cua_driver") is None,
    reason="the computer-use extra is not installed",
)

OFFERED = catalog.offered_tools(enable_clipboard=True)
# Reviewed tools only the Windows driver has, all of them off.
WINDOWS_ONLY = frozenset({"debug_window_info"})


@pytest.fixture
async def bridge():
    bridge = CuaDriverBridge(tools=OFFERED)
    try:
        yield bridge
    finally:
        await bridge.close()


def test_the_extra_brings_the_driver_binary() -> None:
    assert driver.bundled_binary() is not None


async def test_the_pinned_driver_is_the_reviewed_one(bridge: CuaDriverBridge) -> None:
    snapshot = await bridge.snapshot()

    assert snapshot.info.version == catalog.REVIEWED_DRIVER_VERSION
    assert snapshot.instructions


async def test_the_driver_offers_the_catalog(bridge: CuaDriverBridge) -> None:
    names = {tool["name"] for tool in (await bridge.snapshot()).tools}

    if sys.platform.startswith("linux"):
        # Reviewed against the Linux tool list; another OS may lack a tool,
        # which the shim then simply doesn't offer there.
        assert names == OFFERED


async def test_every_tool_the_driver_has_was_reviewed() -> None:
    """The catalog test compared the bridge's tools with the catalog, but the
    bridge and the shim's policy both drop any tool the catalog doesn't
    list, so a release that added one still passed. Listed without the
    policy, every tool must be offered or named as not offered."""
    binary = driver.bundled_binary()
    assert binary is not None
    plat = platform_info.detect_platform()
    client = StdioMcpClient(
        driver.driver_argv(binary, driver.driver_runtime(plat), plat),
        env=driver.driver_environment(None),
        client_version="contract-test",
    )
    await client.start(timeout=30)
    try:
        names = {tool["name"] for tool in await client.list_tools(timeout=30)}
    finally:
        await client.close()

    reviewed = OFFERED | catalog.NOT_OFFERED_TOOLS
    assert names <= reviewed, sorted(names - reviewed)
    assert not OFFERED & catalog.NOT_OFFERED_TOOLS
    if sys.platform.startswith("linux"):
        assert names == reviewed - WINDOWS_ONLY


async def test_the_driver_enforces_the_shims_policy(bridge: CuaDriverBridge) -> None:
    result = await bridge.call_tool("kill_app", {"pid": 1}, timeout=60)

    assert result.get("isError") is True
    assert (result.get("structuredContent") or {}).get("code") == "permission_denied"


async def test_every_approval_argument_is_one_the_driver_takes(bridge: CuaDriverBridge) -> None:
    """An approval check on an argument the driver renamed would stop
    applying without anyone noticing."""
    system = "linux" if sys.platform.startswith("linux") else sys.platform
    snapshot = await bridge.snapshot()
    for tool, arguments in catalog.APPROVAL_ARGUMENTS.items():
        spec = snapshot.tool(tool)
        if spec is None:
            continue
        properties = (spec.get("inputSchema") or {}).get("properties") or {}
        here = {
            argument
            for argument in arguments
            if system in catalog.APPROVAL_ARGUMENT_PLATFORMS.get((tool, argument), {system})
        }
        assert here <= set(properties), tool


async def test_batches_reach_only_catalog_tools(bridge: CuaDriverBridge) -> None:
    """run_steps and run_script call other tools inside the driver, where
    the shim can't look; the driver limits them to these, all reviewed."""
    snapshot = await bridge.snapshot()
    steps = snapshot.tool("run_steps")
    assert steps is not None
    step_tools = steps["inputSchema"]["properties"]["steps"]["items"]["properties"]["tool"]
    assert set(step_tools["enum"]) <= OFFERED

    result = await bridge.call_tool(
        "run_script",
        {"script": "try { await cua.call('kill_app', {pid: 1}); } catch (e) { return String(e); }"},
        timeout=60,
    )
    value = str((result.get("structuredContent") or {}).get("value") or "")
    match = re.search(r"allowed: ([a-z_, ]+)", value)
    assert match, value
    script_tools = {name.strip() for name in match.group(1).split(",")}
    assert script_tools and script_tools <= OFFERED
