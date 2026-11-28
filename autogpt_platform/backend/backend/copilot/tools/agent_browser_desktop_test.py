from unittest.mock import AsyncMock, Mock

import pytest

from backend.copilot.capabilities.sources import tool_entries
from backend.copilot.model import ChatSession
from backend.copilot.tools import agent_browser
from backend.copilot.tools.models import ErrorResponse

TOOLS = (
    agent_browser.BrowserNavigateTool,
    agent_browser.BrowserActTool,
    agent_browser.BrowserScreenshotTool,
)


@pytest.mark.parametrize(
    "mode,available",
    [
        (None, True),
        ("0", True),
        ("false", True),
        ("1", False),
        ("true", False),
        ("unexpected", False),
    ],
)
def test_desktop_browser_is_hidden_even_with_host_binary(monkeypatch, mode, available):
    if mode is None:
        monkeypatch.delenv("AUTOGPT_DESKTOP", raising=False)
    else:
        monkeypatch.setenv("AUTOGPT_DESKTOP", mode)
    monkeypatch.setattr(
        agent_browser.shutil, "which", lambda _name: "host-agent-browser"
    )
    tools = {tool.name: tool for tool in (cls() for cls in TOOLS)}
    assert all(tool.is_available is available for tool in tools.values())
    assert bool(tool_entries(tools, {})) is available


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_class", TOOLS)
async def test_desktop_refuses_direct_browser_invocation_before_io(
    monkeypatch, tool_class
):
    monkeypatch.setenv("AUTOGPT_DESKTOP", "1")
    run, restore, validate = (
        AsyncMock(return_value=(1, "", "mock launch")),
        AsyncMock(),
        AsyncMock(),
    )
    workspace = Mock()
    monkeypatch.setattr(agent_browser, "_run", run)
    monkeypatch.setattr(agent_browser, "_ensure_session", restore)
    monkeypatch.setattr(agent_browser, "validate_url_host", validate)
    monkeypatch.setattr(agent_browser, "get_workspace_manager", workspace)
    session = Mock(spec=ChatSession, session_id="desktop-browser-test")
    response = await tool_class()._execute(
        "owner", session, url="https://example.com", action="snapshot"
    )
    assert isinstance(response, ErrorResponse)
    assert response.error == "desktop_browser_unavailable"
    run.assert_not_awaited()
    restore.assert_not_awaited()
    validate.assert_not_awaited()
    workspace.assert_not_called()


@pytest.mark.asyncio
async def test_desktop_refuses_browser_subprocess_and_proxy_start(monkeypatch):
    monkeypatch.setenv("AUTOGPT_DESKTOP", "1")
    proxy, launch = Mock(), AsyncMock()
    monkeypatch.setattr(agent_browser, "egress_proxy_url", proxy)
    monkeypatch.setattr(agent_browser.asyncio, "create_subprocess_exec", launch)
    code, output, error = await agent_browser._run(
        "session", "open", "https://example.com"
    )
    assert code != 0 and not output
    assert "desktop" in error.lower()
    proxy.assert_not_called()
    launch.assert_not_awaited()
