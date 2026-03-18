"""Integration tests for agent-browser + system chromium.

These tests actually invoke the agent-browser binary via subprocess and require:
  - agent-browser installed (npm install -g agent-browser)
  - AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium (set in Docker)

Run with:
    poetry run pytest backend/copilot/tools/agent_browser_integration_test.py -v -p no:autogpt_platform

Skipped automatically when agent-browser binary is not found.

NOTE: These tests call agent-browser directly via subprocess to avoid importing
backend modules (which would trigger service initialization requiring Postgres/RabbitMQ).
"""

import os
import shutil
import subprocess
import tempfile
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("agent-browser") is None,
    reason="agent-browser binary not found",
)

_SESSION = "integration-test-session"


def _ab(*args: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run agent-browser for the shared test session, return (rc, stdout, stderr)."""
    result = subprocess.run(
        ["agent-browser", "--session", _SESSION, "--session-name", _SESSION, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def _ab_session(session: str, *args: str, timeout: int = 30) -> tuple[int, str, str]:
    """Run agent-browser for an explicitly named session."""
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


def test_navigate_returns_success():
    """agent-browser can open a public URL using system chromium."""
    rc, _, stderr = _ab("open", "https://example.com")
    assert rc == 0, f"open failed (rc={rc}): {stderr}"


def test_get_title_after_navigate():
    """get title returns the page title after navigation."""
    rc, _, _ = _ab("open", "https://example.com")
    assert rc == 0

    rc, stdout, stderr = _ab("get", "title", timeout=10)
    assert rc == 0, f"get title failed: {stderr}"
    assert "example" in stdout.lower()


def test_get_url_after_navigate():
    """get url returns the navigated URL."""
    rc, _, _ = _ab("open", "https://example.com")
    assert rc == 0

    rc, stdout, stderr = _ab("get", "url", timeout=10)
    assert rc == 0, f"get url failed: {stderr}"
    assert urlparse(stdout.strip()).netloc == "example.com"


def test_snapshot_returns_interactive_elements():
    """snapshot -i -c lists interactive elements on the page."""
    rc, _, _ = _ab("open", "https://example.com")
    assert rc == 0

    rc, stdout, stderr = _ab("snapshot", "-i", "-c", timeout=15)
    assert rc == 0, f"snapshot failed: {stderr}"
    assert len(stdout.strip()) > 0, "snapshot returned empty output"


def test_screenshot_produces_valid_png():
    """screenshot saves a non-empty, valid PNG file."""
    rc, _, _ = _ab("open", "https://example.com")
    assert rc == 0

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp = f.name
    try:
        rc, _, stderr = _ab("screenshot", tmp, timeout=15)
        assert rc == 0, f"screenshot failed: {stderr}"
        size = os.path.getsize(tmp)
        assert size > 1000, f"PNG too small ({size} bytes) — likely blank or corrupt"
        with open(tmp, "rb") as f:
            assert f.read(4) == b"\x89PNG", "Output is not a valid PNG"
    finally:
        os.unlink(tmp)


def test_scroll_down():
    """scroll down succeeds without error."""
    rc, _, _ = _ab("open", "https://example.com")
    assert rc == 0

    rc, _, stderr = _ab("scroll", "down", timeout=10)
    assert rc == 0, f"scroll failed: {stderr}"


def test_fill_form_field():
    """fill writes text into an input field."""
    rc, _, _ = _ab("open", "https://httpbin.org/forms/post")
    assert rc == 0

    rc, _, stderr = _ab(
        "fill", "input[name=custname]", "IntegrationTestUser", timeout=10
    )
    assert rc == 0, f"fill failed: {stderr}"


def test_concurrent_independent_sessions():
    """Two independent sessions can navigate in parallel without interference."""
    import concurrent.futures

    session_a = "integration-concurrent-a"
    session_b = "integration-concurrent-b"

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_ab_session, session_a, "open", "https://example.com")
            fut_b = pool.submit(
                _ab_session, session_b, "open", "https://httpbin.org/html"
            )
            fut_a.result(timeout=40)
            fut_b.result(timeout=40)

        _, url_a, _ = _ab_session(session_a, "get", "url", timeout=10)
        _, url_b, _ = _ab_session(session_b, "get", "url", timeout=10)
        assert urlparse(url_a.strip()).netloc == "example.com"
        assert urlparse(url_b.strip()).netloc == "httpbin.org"
    finally:
        _close_session(session_a)
        _close_session(session_b)


def test_close_session():
    """close shuts down the browser daemon cleanly."""
    rc, _, _ = _ab("open", "https://example.com")
    assert rc == 0

    rc, _, stderr = _ab("close", timeout=10)
    assert rc == 0, f"close failed: {stderr}"
