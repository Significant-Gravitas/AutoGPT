"""Integration tests for agent-browser + system chromium.

These tests actually invoke the agent-browser binary via subprocess and require:
  - agent-browser installed (npm install -g agent-browser)
  - AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium (set in Docker)

Run with:
    poetry run test

Or to run only this file:
    poetry run pytest backend/copilot/tools/agent_browser_integration_test.py -v -p no:autogpt_platform

Skipped automatically when agent-browser binary is not found.
Tests that hit external sites are marked ``integration`` and skipped by default
in CI (use ``-m integration`` to include them).

Two test tiers:
  - CLI tests: call agent-browser subprocess directly (no backend imports needed)
  - Tool class tests: call BrowserNavigateTool/BrowserActTool._execute() directly
    with user_id=None (skips workspace/DB interactions — no Postgres/RabbitMQ needed)
"""

import concurrent.futures
import os
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse

import pytest

from backend.copilot.model import ChatSession
from backend.copilot.tools.agent_browser import BrowserActTool, BrowserNavigateTool
from backend.copilot.tools.models import (
    BrowserActResponse,
    BrowserNavigateResponse,
    ErrorResponse,
)

pytestmark = pytest.mark.skipif(
    shutil.which("agent-browser") is None,
    reason="agent-browser binary not found",
)

_SESSION = "integration-test-session"


def _agent_browser(
    *args: str, session: str = _SESSION, timeout: int = 30
) -> tuple[int, str, str]:
    """Run agent-browser for the given session, return (rc, stdout, stderr)."""
    result = subprocess.run(
        ["agent-browser", "--session", session, "--session-name", session, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def _close_session(session: str, timeout: int = 5) -> None:
    """Best-effort close for a browser session; never raises on failure."""
    try:
        subprocess.run(
            ["agent-browser", "--session", session, "--session-name", session, "close"],
            capture_output=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


@pytest.fixture(autouse=True)
def _teardown():
    """Close the shared test session after each test (best-effort)."""
    yield
    _close_session(_SESSION)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chromium_executable_env_is_set():
    """AGENT_BROWSER_EXECUTABLE_PATH must be set and point to an executable binary."""
    exe = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH", "")
    assert exe, "AGENT_BROWSER_EXECUTABLE_PATH is not set"
    assert os.path.isfile(exe), f"Chromium binary not found at {exe}"
    assert os.access(exe, os.X_OK), f"Chromium binary at {exe} is not executable"


@pytest.mark.integration
def test_navigate_returns_success():
    """agent-browser can open a public URL using system chromium."""
    rc, _, stderr = _agent_browser("open", "https://example.com")
    assert rc == 0, f"open failed (rc={rc}): {stderr}"


@pytest.mark.integration
def test_get_title_after_navigate():
    """get title returns the page title after navigation."""
    rc, _, _ = _agent_browser("open", "https://example.com")
    assert rc == 0

    rc, stdout, stderr = _agent_browser("get", "title", timeout=10)
    assert rc == 0, f"get title failed: {stderr}"
    assert "example" in stdout.lower()


@pytest.mark.integration
def test_get_url_after_navigate():
    """get url returns the navigated URL."""
    rc, _, _ = _agent_browser("open", "https://example.com")
    assert rc == 0

    rc, stdout, stderr = _agent_browser("get", "url", timeout=10)
    assert rc == 0, f"get url failed: {stderr}"
    assert urlparse(stdout.strip()).netloc == "example.com"


@pytest.mark.integration
def test_snapshot_returns_interactive_elements():
    """snapshot -i -c lists interactive elements on the page."""
    rc, _, _ = _agent_browser("open", "https://example.com")
    assert rc == 0

    rc, stdout, stderr = _agent_browser("snapshot", "-i", "-c", timeout=15)
    assert rc == 0, f"snapshot failed: {stderr}"
    assert len(stdout.strip()) > 0, "snapshot returned empty output"


@pytest.mark.integration
def test_screenshot_produces_valid_png():
    """screenshot saves a non-empty, valid PNG file."""
    rc, _, _ = _agent_browser("open", "https://example.com")
    assert rc == 0

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        rc, _, stderr = _agent_browser("screenshot", tmp, timeout=15)
        assert rc == 0, f"screenshot failed: {stderr}"
        size = os.path.getsize(tmp)
        assert size > 1000, f"PNG too small ({size} bytes) — likely blank or corrupt"
        with open(tmp, "rb") as f:
            assert f.read(4) == b"\x89PNG", "Output is not a valid PNG"
    finally:
        os.unlink(tmp)


@pytest.mark.integration
def test_scroll_down():
    """scroll down succeeds without error."""
    rc, _, _ = _agent_browser("open", "https://example.com")
    assert rc == 0

    rc, _, stderr = _agent_browser("scroll", "down", timeout=10)
    assert rc == 0, f"scroll failed: {stderr}"


@pytest.mark.integration
def test_fill_form_field():
    """fill writes text into an input field."""
    rc, _, _ = _agent_browser("open", "https://httpbin.org/forms/post")
    assert rc == 0

    rc, _, stderr = _agent_browser(
        "fill", "input[name=custname]", "IntegrationTestUser", timeout=10
    )
    assert rc == 0, f"fill failed: {stderr}"


@pytest.mark.integration
def test_concurrent_independent_sessions():
    """Two independent sessions can navigate in parallel without interference."""
    session_a = "integration-concurrent-a"
    session_b = "integration-concurrent-b"

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(
                _agent_browser, "open", "https://example.com", session=session_a
            )
            fut_b = pool.submit(
                _agent_browser, "open", "https://httpbin.org/html", session=session_b
            )
            rc_a, _, err_a = fut_a.result(timeout=40)
            rc_b, _, err_b = fut_b.result(timeout=40)
        assert rc_a == 0, f"session_a open failed: {err_a}"
        assert rc_b == 0, f"session_b open failed: {err_b}"

        rc_ua, url_a, err_ua = _agent_browser(
            "get", "url", session=session_a, timeout=10
        )
        rc_ub, url_b, err_ub = _agent_browser(
            "get", "url", session=session_b, timeout=10
        )
        assert rc_ua == 0, f"session_a get url failed: {err_ua}"
        assert rc_ub == 0, f"session_b get url failed: {err_ub}"
        assert urlparse(url_a.strip()).netloc == "example.com"
        assert urlparse(url_b.strip()).netloc == "httpbin.org"
    finally:
        _close_session(session_a)
        _close_session(session_b)


@pytest.mark.integration
def test_close_session():
    """close shuts down the browser daemon cleanly."""
    rc, _, _ = _agent_browser("open", "https://example.com")
    assert rc == 0

    rc, _, stderr = _agent_browser("close", timeout=10)
    assert rc == 0, f"close failed: {stderr}"


# ---------------------------------------------------------------------------
# Python tool class integration tests
#
# These tests exercise the actual BrowserNavigateTool / BrowserActTool Python
# classes (not just the CLI binary) to verify the full call path — URL
# validation, subprocess dispatch, response parsing — works with system
# chromium.  user_id=None skips workspace/DB interactions so no Postgres or
# RabbitMQ is needed.
# ---------------------------------------------------------------------------

_TOOL_SESSION_ID = "integration-tool-test-session"
_TEST_SESSION = ChatSession(
    session_id=_TOOL_SESSION_ID,
    user_id="test-user",
    messages=[],
    usage=[],
    started_at=datetime.now(timezone.utc),
    updated_at=datetime.now(timezone.utc),
)


@pytest.fixture(autouse=False)
def _close_tool_session():
    """Tear down the tool-test browser session after each tool test."""
    yield
    _close_session(_TOOL_SESSION_ID)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_navigate_returns_response(_close_tool_session):
    """BrowserNavigateTool._execute returns a BrowserNavigateResponse with real content."""
    tool = BrowserNavigateTool()
    resp = await tool._execute(
        user_id=None, session=_TEST_SESSION, url="https://example.com"
    )
    assert isinstance(
        resp, BrowserNavigateResponse
    ), f"Expected BrowserNavigateResponse, got: {resp}"
    assert urlparse(resp.url).netloc == "example.com"
    assert resp.title, "Expected non-empty page title"
    assert resp.snapshot, "Expected non-empty accessibility snapshot"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ssrf_url",
    [
        "http://169.254.169.254/",  # AWS/GCP/Azure metadata endpoint
        "http://127.0.0.1/",  # IPv4 loopback
        "http://10.0.0.1/",  # RFC-1918 private range
        "http://[::1]/",  # IPv6 loopback
        "http://0.0.0.0/",  # Wildcard / INADDR_ANY
    ],
)
async def test_tool_navigate_blocked_url(ssrf_url: str, _close_tool_session):
    """BrowserNavigateTool._execute rejects internal/private URLs (SSRF guard)."""
    tool = BrowserNavigateTool()
    resp = await tool._execute(user_id=None, session=_TEST_SESSION, url=ssrf_url)
    assert isinstance(
        resp, ErrorResponse
    ), f"Expected ErrorResponse for SSRF URL {ssrf_url!r}, got: {resp}"
    assert resp.error == "blocked_url"


@pytest.mark.asyncio
async def test_tool_navigate_missing_url(_close_tool_session):
    """BrowserNavigateTool._execute returns an error when url is empty."""
    tool = BrowserNavigateTool()
    resp = await tool._execute(user_id=None, session=_TEST_SESSION, url="")
    assert isinstance(resp, ErrorResponse)
    assert resp.error == "missing_url"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_act_scroll(_close_tool_session):
    """BrowserActTool._execute can scroll after a navigate."""
    nav = BrowserNavigateTool()
    nav_resp = await nav._execute(
        user_id=None, session=_TEST_SESSION, url="https://example.com"
    )
    assert isinstance(nav_resp, BrowserNavigateResponse)

    act = BrowserActTool()
    resp = await act._execute(
        user_id=None, session=_TEST_SESSION, action="scroll", direction="down"
    )
    assert isinstance(
        resp, BrowserActResponse
    ), f"Expected BrowserActResponse, got: {resp}"
    assert resp.action == "scroll"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_act_fill_and_click(_close_tool_session):
    """BrowserActTool._execute can fill a form field."""
    nav = BrowserNavigateTool()
    nav_resp = await nav._execute(
        user_id=None, session=_TEST_SESSION, url="https://httpbin.org/forms/post"
    )
    assert isinstance(nav_resp, BrowserNavigateResponse)

    act = BrowserActTool()
    resp = await act._execute(
        user_id=None,
        session=_TEST_SESSION,
        action="fill",
        target="input[name=custname]",
        value="ToolIntegrationTest",
    )
    assert isinstance(resp, BrowserActResponse), f"fill failed: {resp}"


@pytest.mark.asyncio
async def test_tool_act_missing_action(_close_tool_session):
    """BrowserActTool._execute returns an error when action is missing."""
    act = BrowserActTool()
    resp = await act._execute(user_id=None, session=_TEST_SESSION, action="")
    assert isinstance(resp, ErrorResponse)
    assert resp.error == "missing_action"


@pytest.mark.asyncio
async def test_tool_act_missing_target(_close_tool_session):
    """BrowserActTool._execute returns an error when click target is missing."""
    act = BrowserActTool()
    resp = await act._execute(
        user_id=None, session=_TEST_SESSION, action="click", target=""
    )
    assert isinstance(resp, ErrorResponse)
    assert resp.error == "missing_target"
