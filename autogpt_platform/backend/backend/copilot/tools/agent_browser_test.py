"""Unit tests for agent_browser tools.

All subprocess calls are mocked — no agent-browser binary or real browser needed.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSession

from .agent_browser import BrowserActTool, BrowserNavigateTool, BrowserScreenshotTool
from .models import (
    BrowserActResponse,
    BrowserNavigateResponse,
    BrowserScreenshotResponse,
    ErrorResponse,
    ResponseType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_session(session_id: str = "test-session-123") -> ChatSession:
    return MagicMock(spec=ChatSession, session_id=session_id)


def _run_result(rc: int = 0, stdout: str = "", stderr: str = "") -> tuple:
    return (rc, stdout, stderr)


# ---------------------------------------------------------------------------
# SSRF protection via shared validate_url (backend.util.request)
# ---------------------------------------------------------------------------

# Patch target: validate_url is imported directly into agent_browser's module scope.
_VALIDATE_URL = "backend.copilot.tools.agent_browser.validate_url"


class TestSsrfViaValidateUrl:
    """Verify that browser_navigate uses validate_url for SSRF protection.

    We mock validate_url itself (not the low-level socket) so these tests
    exercise the integration point, not the internals of request.py
    (which has its own thorough test suite in request_test.py).
    """

    def setup_method(self):
        self.tool = BrowserNavigateTool()
        self.session = make_session()

    @pytest.mark.asyncio
    async def test_blocked_ip_returns_blocked_url_error(self):
        """validate_url raises ValueError → tool returns blocked_url ErrorResponse."""
        with patch(_VALIDATE_URL, new_callable=AsyncMock) as mock_validate:
            mock_validate.side_effect = ValueError(
                "Access to blocked IP 10.0.0.1 is not allowed."
            )
            result = await self.tool._execute(
                user_id="user1", session=self.session, url="http://internal.corp"
            )
        assert isinstance(result, ErrorResponse)
        assert result.error == "blocked_url"
        assert "10.0.0.1" in result.message

    @pytest.mark.asyncio
    async def test_invalid_scheme_returns_blocked_url_error(self):
        with patch(_VALIDATE_URL, new_callable=AsyncMock) as mock_validate:
            mock_validate.side_effect = ValueError("Scheme 'ftp' is not allowed.")
            result = await self.tool._execute(
                user_id="user1", session=self.session, url="ftp://example.com"
            )
        assert isinstance(result, ErrorResponse)
        assert result.error == "blocked_url"

    @pytest.mark.asyncio
    async def test_unresolvable_host_returns_blocked_url_error(self):
        with patch(_VALIDATE_URL, new_callable=AsyncMock) as mock_validate:
            mock_validate.side_effect = ValueError(
                "Unable to resolve IP address for hostname bad.invalid"
            )
            result = await self.tool._execute(
                user_id="user1", session=self.session, url="https://bad.invalid"
            )
        assert isinstance(result, ErrorResponse)
        assert result.error == "blocked_url"

    @pytest.mark.asyncio
    async def test_validate_url_called_with_empty_trusted_origins(self):
        """Confirms no trusted-origins bypass is granted — all URLs are validated."""
        with patch(_VALIDATE_URL, new_callable=AsyncMock) as mock_validate:
            mock_validate.return_value = (object(), False, ["1.2.3.4"])
            with patch(
                "backend.copilot.tools.agent_browser._run",
                new_callable=AsyncMock,
                return_value=_run_result(rc=0),
            ):
                with patch(
                    "backend.copilot.tools.agent_browser._snapshot",
                    new_callable=AsyncMock,
                    return_value="",
                ):
                    await self.tool._execute(
                        user_id="user1",
                        session=self.session,
                        url="https://example.com",
                    )
        mock_validate.assert_called_once_with("https://example.com", trusted_origins=[])


# ---------------------------------------------------------------------------
# BrowserNavigateTool metadata
# ---------------------------------------------------------------------------


class TestAgentBrowserAvailability:
    """is_available reflects whether the agent-browser CLI binary is installed."""

    def test_available_when_binary_found(self):
        with patch("shutil.which", return_value="/usr/local/bin/agent-browser"):
            assert BrowserNavigateTool().is_available is True
            assert BrowserActTool().is_available is True
            assert BrowserScreenshotTool().is_available is True

    def test_unavailable_when_binary_missing(self):
        with patch("shutil.which", return_value=None):
            assert BrowserNavigateTool().is_available is False
            assert BrowserActTool().is_available is False
            assert BrowserScreenshotTool().is_available is False


class TestBrowserNavigateMetadata:
    def setup_method(self):
        self.tool = BrowserNavigateTool()

    def test_name(self):
        assert self.tool.name == "browser_navigate"

    def test_requires_auth(self):
        assert self.tool.requires_auth is True

    def test_parameters_has_url_required(self):
        params = self.tool.parameters
        assert "url" in params["properties"]
        assert "url" in params["required"]

    def test_parameters_has_wait_for(self):
        params = self.tool.parameters
        assert "wait_for" in params["properties"]

    def test_as_openai_tool(self):
        ot = self.tool.as_openai_tool()
        assert ot["type"] == "function"
        assert ot["function"]["name"] == "browser_navigate"


# ---------------------------------------------------------------------------
# BrowserNavigateTool execution
# ---------------------------------------------------------------------------


class TestBrowserNavigateExecute:
    def setup_method(self):
        self.tool = BrowserNavigateTool()
        self.session = make_session()

    @pytest.mark.asyncio
    async def test_missing_url_returns_error(self):
        result = await self.tool._execute(user_id="user1", session=self.session, url="")
        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_url"

    @pytest.mark.asyncio
    async def test_ssrf_blocked_url_returns_error(self):
        with patch(_VALIDATE_URL, new_callable=AsyncMock) as mock_validate:
            mock_validate.side_effect = ValueError(
                "Access to blocked IP 10.0.0.1 is not allowed."
            )
            result = await self.tool._execute(
                user_id="user1", session=self.session, url="http://internal.corp"
            )
        assert isinstance(result, ErrorResponse)
        assert result.error == "blocked_url"

    @pytest.mark.asyncio
    async def test_navigation_failure_returns_error(self):
        with patch(
            _VALIDATE_URL,
            new_callable=AsyncMock,
            return_value=(object(), False, ["1.2.3.4"]),
        ):
            with patch(
                "backend.copilot.tools.agent_browser._run",
                new_callable=AsyncMock,
            ) as mock_run:
                mock_run.return_value = _run_result(rc=1, stderr="timeout")
                result = await self.tool._execute(
                    user_id="user1",
                    session=self.session,
                    url="https://example.com",
                )
        assert isinstance(result, ErrorResponse)
        assert result.error == "navigation_failed"

    @pytest.mark.asyncio
    async def test_successful_navigation(self):
        with patch(
            _VALIDATE_URL,
            new_callable=AsyncMock,
            return_value=(object(), False, ["1.2.3.4"]),
        ):
            with patch(
                "backend.copilot.tools.agent_browser._run",
                new_callable=AsyncMock,
            ) as mock_run:
                # open → success; wait → success; get title; get url
                mock_run.side_effect = [
                    _run_result(rc=0),  # open
                    _run_result(rc=0),  # wait
                    _run_result(rc=0, stdout="Example Domain"),  # get title
                    _run_result(rc=0, stdout="https://example.com"),  # get url
                ]
                with patch(
                    "backend.copilot.tools.agent_browser._snapshot",
                    new_callable=AsyncMock,
                    return_value="[snapshot]",
                ):
                    result = await self.tool._execute(
                        user_id="user1",
                        session=self.session,
                        url="https://example.com",
                    )

        assert isinstance(result, BrowserNavigateResponse)
        assert result.type == ResponseType.BROWSER_NAVIGATE
        assert result.url == "https://example.com"
        assert result.title == "Example Domain"
        assert result.snapshot == "[snapshot]"

    @pytest.mark.asyncio
    async def test_session_id_used_as_session_name(self):
        """Ensure session.session_id is passed as --session-name."""
        session = make_session("my-unique-session")
        captured_calls = []

        async def fake_run(session_name, *args, **kwargs):
            captured_calls.append((session_name, args))
            return _run_result(rc=0)

        with patch(
            _VALIDATE_URL,
            new_callable=AsyncMock,
            return_value=(object(), False, ["1.2.3.4"]),
        ):
            with patch(
                "backend.copilot.tools.agent_browser._run", side_effect=fake_run
            ):
                with patch(
                    "backend.copilot.tools.agent_browser._snapshot",
                    new_callable=AsyncMock,
                    return_value="",
                ):
                    await self.tool._execute(
                        user_id="user1",
                        session=session,
                        url="https://example.com",
                    )

        for session_name, _ in captured_calls:
            assert session_name == "my-unique-session"


# ---------------------------------------------------------------------------
# BrowserActTool metadata
# ---------------------------------------------------------------------------


class TestBrowserActMetadata:
    def setup_method(self):
        self.tool = BrowserActTool()

    def test_name(self):
        assert self.tool.name == "browser_act"

    def test_requires_auth(self):
        assert self.tool.requires_auth is True

    def test_parameters_has_action_required(self):
        params = self.tool.parameters
        assert "action" in params["properties"]
        assert "action" in params["required"]

    def test_action_enum_values(self):
        actions = self.tool.parameters["properties"]["action"]["enum"]
        expected = {
            "click",
            "dblclick",
            "fill",
            "type",
            "scroll",
            "hover",
            "press",
            "check",
            "uncheck",
            "select",
            "wait",
            "back",
            "forward",
            "reload",
        }
        assert set(actions) == expected


# ---------------------------------------------------------------------------
# BrowserActTool execution — input validation
# ---------------------------------------------------------------------------


class TestBrowserActValidation:
    def setup_method(self):
        self.tool = BrowserActTool()
        self.session = make_session()

    @pytest.mark.asyncio
    async def test_missing_action_returns_error(self):
        result = await self.tool._execute(
            user_id="user1", session=self.session, action=""
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_action"

    @pytest.mark.asyncio
    async def test_click_without_target_returns_error(self):
        result = await self.tool._execute(
            user_id="user1", session=self.session, action="click", target=""
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_target"

    @pytest.mark.asyncio
    async def test_fill_without_value_returns_error(self):
        result = await self.tool._execute(
            user_id="user1",
            session=self.session,
            action="fill",
            target="@e1",
            value="",
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_params"

    @pytest.mark.asyncio
    async def test_press_without_value_returns_error(self):
        result = await self.tool._execute(
            user_id="user1", session=self.session, action="press", value=""
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_value"

    @pytest.mark.asyncio
    async def test_unsupported_action_returns_error(self):
        result = await self.tool._execute(
            user_id="user1", session=self.session, action="teleport"
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "invalid_action"


# ---------------------------------------------------------------------------
# BrowserActTool execution — successful actions
# ---------------------------------------------------------------------------


class TestBrowserActExecute:
    def setup_method(self):
        self.tool = BrowserActTool()
        self.session = make_session()

    async def _act(self, **kwargs):
        with patch(
            "backend.copilot.tools.agent_browser._run",
            new_callable=AsyncMock,
        ) as mock_run:
            mock_run.return_value = _run_result(rc=0, stdout="https://example.com")
            with patch(
                "backend.copilot.tools.agent_browser._snapshot",
                new_callable=AsyncMock,
                return_value="[updated-snapshot]",
            ):
                result = await self.tool._execute(
                    user_id="user1", session=self.session, **kwargs
                )
        return result

    @pytest.mark.asyncio
    async def test_click_success(self):
        result = await self._act(action="click", target="@e3")
        assert isinstance(result, BrowserActResponse)
        assert result.type == ResponseType.BROWSER_ACT
        assert result.action == "click"
        assert result.snapshot == "[updated-snapshot]"

    @pytest.mark.asyncio
    async def test_fill_success(self):
        result = await self._act(action="fill", target="@e1", value="hello@test.com")
        assert isinstance(result, BrowserActResponse)
        assert "fill" in result.message
        # value must NOT appear in the message — it may contain credentials
        assert "hello@test.com" not in result.message

    @pytest.mark.asyncio
    async def test_back_success(self):
        result = await self._act(action="back")
        assert isinstance(result, BrowserActResponse)

    @pytest.mark.asyncio
    async def test_scroll_default_direction(self):
        result = await self._act(action="scroll")
        assert isinstance(result, BrowserActResponse)

    @pytest.mark.asyncio
    async def test_scroll_up(self):
        result = await self._act(action="scroll", direction="up")
        assert isinstance(result, BrowserActResponse)

    @pytest.mark.asyncio
    async def test_press_enter(self):
        result = await self._act(action="press", value="Enter")
        assert isinstance(result, BrowserActResponse)

    @pytest.mark.asyncio
    async def test_action_failure_returns_error(self):
        with patch(
            "backend.copilot.tools.agent_browser._run",
            new_callable=AsyncMock,
            return_value=_run_result(rc=1, stderr="element not found"),
        ):
            result = await self.tool._execute(
                user_id="user1", session=self.session, action="click", target="@e99"
            )
        assert isinstance(result, ErrorResponse)
        assert result.error == "action_failed"

    @pytest.mark.asyncio
    async def test_select_success(self):
        result = await self._act(action="select", target="@e5", value="option1")
        assert isinstance(result, BrowserActResponse)

    @pytest.mark.asyncio
    async def test_check_success(self):
        result = await self._act(action="check", target="@e2")
        assert isinstance(result, BrowserActResponse)

    @pytest.mark.asyncio
    async def test_uncheck_success(self):
        result = await self._act(action="uncheck", target="@e2")
        assert isinstance(result, BrowserActResponse)

    @pytest.mark.asyncio
    async def test_dblclick_success(self):
        result = await self._act(action="dblclick", target="@e4")
        assert isinstance(result, BrowserActResponse)
        assert result.action == "dblclick"

    @pytest.mark.asyncio
    async def test_type_success(self):
        """type appends text without clearing the field first (unlike fill)."""
        result = await self._act(action="type", target="@e1", value="hello")
        assert isinstance(result, BrowserActResponse)
        assert result.action == "type"

    @pytest.mark.asyncio
    async def test_wait_selector_success(self):
        """wait with a CSS selector waits for the element to appear."""
        result = await self._act(action="wait", target=".my-element")
        assert isinstance(result, BrowserActResponse)
        assert result.action == "wait"

    @pytest.mark.asyncio
    async def test_wait_ms_success(self):
        """wait with a numeric string pauses for the given milliseconds."""
        result = await self._act(action="wait", target="1000")
        assert isinstance(result, BrowserActResponse)

    @pytest.mark.asyncio
    async def test_dblclick_without_target_returns_error(self):
        result = await self.tool._execute(
            user_id="user1", session=self.session, action="dblclick", target=""
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_target"

    @pytest.mark.asyncio
    async def test_type_without_value_returns_error(self):
        result = await self.tool._execute(
            user_id="user1",
            session=self.session,
            action="type",
            target="@e1",
            value="",
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_params"

    @pytest.mark.asyncio
    async def test_wait_without_target_returns_error(self):
        result = await self.tool._execute(
            user_id="user1", session=self.session, action="wait", target=""
        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "missing_target"


# ---------------------------------------------------------------------------
# BrowserScreenshotTool metadata
# ---------------------------------------------------------------------------


class TestBrowserScreenshotMetadata:
    def setup_method(self):
        self.tool = BrowserScreenshotTool()

    def test_name(self):
        assert self.tool.name == "browser_screenshot"

    def test_requires_auth(self):
        assert self.tool.requires_auth is True

    def test_parameters_are_optional(self):
        params = self.tool.parameters
        # No required fields — both annotate and filename are optional
        assert "required" not in params or len(params.get("required", [])) == 0


# ---------------------------------------------------------------------------
# BrowserScreenshotTool execution
# ---------------------------------------------------------------------------


class TestBrowserScreenshotExecute:
    def setup_method(self):
        self.tool = BrowserScreenshotTool()
        self.session = make_session()

    @pytest.mark.asyncio
    async def test_screenshot_failure_returns_error(self):

        fake_fd, fake_path = 5, "/tmp/fake.png"
        with patch("tempfile.mkstemp", return_value=(fake_fd, fake_path)):
            with patch("os.close"):
                with patch(
                    "backend.copilot.tools.agent_browser._run",
                    new_callable=AsyncMock,
                    return_value=_run_result(rc=1, stderr="no browser"),
                ):
                    with patch("os.unlink"):
                        result = await self.tool._execute(
                            user_id="user1",
                            session=self.session,
                        )
        assert isinstance(result, ErrorResponse)
        assert result.error == "screenshot_failed"

    @pytest.mark.asyncio
    async def test_screenshot_success(self):
        import os
        import tempfile

        from .workspace_files import WorkspaceWriteResponse

        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        # Build a real WorkspaceWriteResponse to satisfy the isinstance check
        write_resp = WorkspaceWriteResponse(
            message="ok",
            file_id="file-abc-123",
            name="test.png",
            path="/workspace/test.png",
            mime_type="image/png",
            size_bytes=len(png_bytes),
            download_url="workspace://file-abc-123#image/png",
            session_id="test-session-123",
        )

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(png_bytes)

        try:
            with patch("tempfile.mkstemp", return_value=(fd, path)):
                with patch("os.close"):
                    with patch(
                        "backend.copilot.tools.agent_browser._run",
                        new_callable=AsyncMock,
                        return_value=_run_result(rc=0),
                    ):
                        # Patch WriteWorkspaceFileTool in the workspace_files module
                        # (the lazy import resolves to this target)
                        with patch(
                            "backend.copilot.tools.workspace_files.WriteWorkspaceFileTool._execute",
                            new_callable=AsyncMock,
                            return_value=write_resp,
                        ):
                            with patch("os.unlink"):
                                result = await self.tool._execute(
                                    user_id="user1",
                                    session=self.session,
                                    filename="test.png",
                                    annotate=True,
                                )
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass  # Best-effort test cleanup; not critical if temp file lingers.

        assert isinstance(result, BrowserScreenshotResponse)
        assert result.type == ResponseType.BROWSER_SCREENSHOT
        assert result.file_id == "file-abc-123"
        assert result.filename == "test.png"


# ---------------------------------------------------------------------------
# BaseTool auth check (inherits to all browser tools)
# ---------------------------------------------------------------------------


class TestAuthCheck:
    @pytest.mark.asyncio
    async def test_navigate_requires_login_if_no_user(self):
        tool = BrowserNavigateTool()
        session = make_session()
        # BaseTool.execute checks requires_auth before calling _execute
        result = await tool.execute(
            user_id=None,
            session=session,
            tool_call_id="call-1",
            url="https://example.com",
        )
        assert isinstance(result.output, str)
        data = json.loads(result.output)
        assert data["type"] == ResponseType.NEED_LOGIN

    @pytest.mark.asyncio
    async def test_act_requires_login_if_no_user(self):
        tool = BrowserActTool()
        session = make_session()
        result = await tool.execute(
            user_id=None,
            session=session,
            tool_call_id="call-2",
            action="click",
            target="@e1",
        )
        assert isinstance(result.output, str)
        data = json.loads(result.output)
        assert data["type"] == ResponseType.NEED_LOGIN

    @pytest.mark.asyncio
    async def test_screenshot_requires_login_if_no_user(self):
        tool = BrowserScreenshotTool()
        session = make_session()
        result = await tool.execute(
            user_id=None,
            session=session,
            tool_call_id="call-3",
        )
        assert isinstance(result.output, str)
        data = json.loads(result.output)
        assert data["type"] == ResponseType.NEED_LOGIN


# ---------------------------------------------------------------------------
# _snapshot helper
# ---------------------------------------------------------------------------


class TestSnapshot:
    @pytest.mark.asyncio
    async def test_snapshot_truncation(self):
        from .agent_browser import _MAX_SNAPSHOT_CHARS, _snapshot

        big_text = "x" * (_MAX_SNAPSHOT_CHARS + 1000)
        with patch(
            "backend.copilot.tools.agent_browser._run",
            new_callable=AsyncMock,
            return_value=_run_result(rc=0, stdout=big_text),
        ):
            result = await _snapshot("test-session")
        assert len(result) <= _MAX_SNAPSHOT_CHARS
        assert "truncated" in result.lower()

    @pytest.mark.asyncio
    async def test_snapshot_failure(self):
        from .agent_browser import _snapshot

        with patch(
            "backend.copilot.tools.agent_browser._run",
            new_callable=AsyncMock,
            return_value=_run_result(rc=1, stderr="daemon died"),
        ):
            result = await _snapshot("test-session")
        assert "snapshot failed" in result

    @pytest.mark.asyncio
    async def test_snapshot_short_content_not_truncated(self):
        from .agent_browser import _snapshot

        short = "button @e1 Submit"
        with patch(
            "backend.copilot.tools.agent_browser._run",
            new_callable=AsyncMock,
            return_value=_run_result(rc=0, stdout=short),
        ):
            result = await _snapshot("test-session")
        assert result == short
        assert "truncated" not in result


# ---------------------------------------------------------------------------
# _run helper (timeout and FileNotFoundError)
# ---------------------------------------------------------------------------


class TestRunHelper:
    @pytest.mark.asyncio
    async def test_run_timeout(self):
        from .agent_browser import _run

        mock_proc = MagicMock()
        mock_proc.returncode = None  # Simulate a process that is still running
        mock_proc.kill = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
                rc, stdout, stderr = await _run(
                    "sess", "open", "https://x.com", timeout=1
                )

        assert rc == 1
        assert "timed out" in stderr
        mock_proc.kill.assert_called_once()
        mock_proc.communicate.assert_called()

    @pytest.mark.asyncio
    async def test_run_agent_browser_not_installed(self):
        from .agent_browser import _run

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("No such file"),
        ):
            rc, stdout, stderr = await _run("sess", "snapshot", "-i")

        assert rc == 1
        assert "agent-browser" in stderr
        assert "not installed" in stderr
