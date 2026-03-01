"""Agent-browser tools — multi-step browser automation for the Copilot.

Uses the agent-browser CLI (https://github.com/vercel-labs/agent-browser)
which runs a local Chromium instance managed by a persistent daemon.

Why agent-browser instead of Stagehand/Browserbase:
- Runs locally — no cloud account required
- Full interaction support: click, fill, scroll, login flows, multi-step
- Session persistence via --session-name: cookies/auth carry across tool calls
  within the same Copilot session, enabling login → navigate → extract workflows
- Screenshot with --annotate overlays @ref labels, saved to workspace for user
- The Claude Agent SDK's multi-turn loop handles orchestration — each tool call
  is one browser action; the LLM chains them naturally

SSRF protection:
  Uses the shared validate_url() from backend.util.request, which is the same
  guard used by HTTP blocks and web_fetch. It resolves ALL DNS answers (not just
  the first), blocks RFC 1918, loopback, link-local, 0.0.0.0/8, multicast,
  and all relevant IPv6 ranges, and applies IDNA encoding to prevent Unicode
  domain attacks.

Requires:
  npm install -g agent-browser
  agent-browser install   (downloads Chromium, one-time per machine)
"""

import asyncio
import base64
import logging
import os
import shutil
import tempfile
from typing import Any

from backend.copilot.model import ChatSession
from backend.util.request import validate_url

from .base import BaseTool
from .models import (
    BrowserActResponse,
    BrowserNavigateResponse,
    BrowserScreenshotResponse,
    ErrorResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

# Per-command timeout (seconds). Navigation + networkidle wait can be slow.
_CMD_TIMEOUT = 45
# Accessibility tree can be very large; cap it to keep LLM context manageable.
_MAX_SNAPSHOT_CHARS = 20_000


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------


async def _run(
    session_name: str,
    *args: str,
    timeout: int = _CMD_TIMEOUT,
) -> tuple[int, str, str]:
    """Run agent-browser for the given session and return (rc, stdout, stderr).

    Uses both:
      --session <name>       → isolated Chromium context (no shared history/cookies
                               with other Copilot sessions — prevents cross-session
                               browser state leakage)
      --session-name <name>  → persist cookies/localStorage across tool calls within
                               the same session (enables login → navigate flows)
    """
    cmd = [
        "agent-browser",
        "--session",
        session_name,
        "--session-name",
        session_name,
        *args,
    ]
    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return proc.returncode or 0, stdout.decode(), stderr.decode()
    except asyncio.TimeoutError:
        # Kill the orphaned subprocess so it does not linger in the process table.
        if proc is not None and proc.returncode is None:
            proc.kill()
            try:
                await proc.communicate()
            except Exception:
                pass  # Best-effort reap; ignore errors during cleanup.
        return 1, "", f"Command timed out after {timeout}s."
    except FileNotFoundError:
        return (
            1,
            "",
            "agent-browser is not installed (run: npm install -g agent-browser && agent-browser install).",
        )


async def _snapshot(session_name: str) -> str:
    """Return the current page's interactive accessibility tree, truncated."""
    rc, stdout, stderr = await _run(session_name, "snapshot", "-i")
    if rc != 0:
        return f"[snapshot failed: {stderr[:300]}]"
    text = stdout.strip()
    if len(text) > _MAX_SNAPSHOT_CHARS:
        suffix = "\n\n[Snapshot truncated — use browser_act to navigate further]"
        keep = max(0, _MAX_SNAPSHOT_CHARS - len(suffix))
        text = text[:keep] + suffix
    return text


# ---------------------------------------------------------------------------
# Tool: browser_navigate
# ---------------------------------------------------------------------------


class BrowserNavigateTool(BaseTool):
    """Navigate to a URL and return the page's interactive elements.

    The browser session persists across tool calls within this Copilot session
    (keyed to session_id), so cookies and auth state carry over. This enables
    full login flows: navigate to login page → browser_act to fill credentials
    → browser_act to submit → browser_navigate to the target page.
    """

    @property
    def name(self) -> str:
        return "browser_navigate"

    @property
    def description(self) -> str:
        return (
            "Navigate to a URL using a real browser. Returns an accessibility "
            "tree snapshot listing the page's interactive elements with @ref IDs "
            "(e.g. @e3) that can be used with browser_act. "
            "Session persists — cookies and login state carry over between calls. "
            "Use this (with browser_act) for multi-step interaction: login flows, "
            "form filling, button clicks, or anything requiring page interaction. "
            "For one-shot content extraction from JS pages with no interaction needed, "
            "prefer browse_web (if available) — it's simpler and faster. "
            "For plain static pages, prefer web_fetch — no browser overhead. "
            "For authenticated pages: navigate to the login page first, use browser_act "
            "to fill credentials and submit, then navigate to the target page."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The HTTP/HTTPS URL to navigate to.",
                },
                "wait_for": {
                    "type": "string",
                    "enum": ["networkidle", "load", "domcontentloaded"],
                    "default": "networkidle",
                    "description": "When to consider navigation complete. Use 'networkidle' for SPAs (default).",
                },
            },
            "required": ["url"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return shutil.which("agent-browser") is not None

    async def _execute(
        self,
        user_id: str | None,  # noqa: ARG002
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        url: str = (kwargs.get("url") or "").strip()
        wait_for: str = kwargs.get("wait_for") or "networkidle"
        session_name = session.session_id

        if not url:
            return ErrorResponse(
                message="Please provide a URL to navigate to.",
                error="missing_url",
                session_id=session_name,
            )

        try:
            await validate_url(url, trusted_origins=[])
        except ValueError as e:
            return ErrorResponse(
                message=str(e),
                error="blocked_url",
                session_id=session_name,
            )

        # Navigate
        rc, _, stderr = await _run(session_name, "open", url)
        if rc != 0:
            logger.warning(
                "[browser_navigate] open failed for %s: %s", url, stderr[:300]
            )
            return ErrorResponse(
                message="Failed to navigate to URL.",
                error="navigation_failed",
                session_id=session_name,
            )

        # Wait for page to settle (best-effort: some SPAs never reach networkidle)
        wait_rc, _, wait_err = await _run(session_name, "wait", "--load", wait_for)
        if wait_rc != 0:
            logger.warning(
                "[browser_navigate] wait(%s) failed: %s", wait_for, wait_err[:300]
            )

        # Get current title and URL in parallel
        (_, title_out, _), (_, url_out, _) = await asyncio.gather(
            _run(session_name, "get", "title"),
            _run(session_name, "get", "url"),
        )

        snapshot = await _snapshot(session_name)

        return BrowserNavigateResponse(
            message=f"Navigated to {url}",
            url=url_out.strip() or url,
            title=title_out.strip(),
            snapshot=snapshot,
            session_id=session_name,
        )


# ---------------------------------------------------------------------------
# Tool: browser_act
# ---------------------------------------------------------------------------

_NO_TARGET_ACTIONS = frozenset({"back", "forward", "reload"})
_SCROLL_ACTIONS = frozenset({"scroll"})
_TARGET_ONLY_ACTIONS = frozenset({"click", "dblclick", "hover", "check", "uncheck"})
_TARGET_VALUE_ACTIONS = frozenset({"fill", "type", "select"})
# wait <selector|ms>: waits for a DOM element or a fixed delay (e.g. "1000" for 1 s)
_WAIT_ACTIONS = frozenset({"wait"})


class BrowserActTool(BaseTool):
    """Perform an action on the current browser page and return the updated snapshot.

    Use @ref IDs from the snapshot returned by browser_navigate (e.g. '@e3').
    The LLM orchestrates multi-step flows by chaining browser_navigate and
    browser_act calls across turns of the Claude Agent SDK conversation.
    """

    @property
    def name(self) -> str:
        return "browser_act"

    @property
    def description(self) -> str:
        return (
            "Interact with the current browser page. Use @ref IDs from the "
            "snapshot (e.g. '@e3') to target elements. Returns an updated snapshot. "
            "Supported actions: click, dblclick, fill, type, scroll, hover, press, "
            "check, uncheck, select, wait, back, forward, reload. "
            "fill clears the field before typing; type appends without clearing. "
            "wait accepts a CSS selector (waits for element) or milliseconds string (e.g. '1000'). "
            "Example login flow: fill @e1 with email → fill @e2 with password → "
            "click @e3 (submit) → browser_navigate to the target page."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
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
                    ],
                    "description": "The action to perform.",
                },
                "target": {
                    "type": "string",
                    "description": (
                        "Element to target. Use @ref from snapshot (e.g. '@e3'), "
                        "a CSS selector, or a text description. "
                        "Required for: click, dblclick, fill, type, hover, check, uncheck, select. "
                        "For wait: a CSS selector to wait for, or milliseconds as a string (e.g. '1000')."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": (
                        "For fill/type: the text to enter. "
                        "For press: key name (e.g. 'Enter', 'Tab', 'Control+a'). "
                        "For select: the option value to select."
                    ),
                },
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "default": "down",
                    "description": "For scroll: direction to scroll.",
                },
            },
            "required": ["action"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return shutil.which("agent-browser") is not None

    async def _execute(
        self,
        user_id: str | None,  # noqa: ARG002
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        action: str = (kwargs.get("action") or "").strip()
        target: str = (kwargs.get("target") or "").strip()
        value: str = (kwargs.get("value") or "").strip()
        direction: str = (kwargs.get("direction") or "down").strip()
        session_name = session.session_id

        if not action:
            return ErrorResponse(
                message="Please specify an action.",
                error="missing_action",
                session_id=session_name,
            )

        # Build the agent-browser command args
        if action in _NO_TARGET_ACTIONS:
            cmd_args = [action]

        elif action in _SCROLL_ACTIONS:
            cmd_args = ["scroll", direction]

        elif action == "press":
            if not value:
                return ErrorResponse(
                    message="'press' requires a 'value' (key name, e.g. 'Enter').",
                    error="missing_value",
                    session_id=session_name,
                )
            cmd_args = ["press", value]

        elif action in _TARGET_ONLY_ACTIONS:
            if not target:
                return ErrorResponse(
                    message=f"'{action}' requires a 'target' element.",
                    error="missing_target",
                    session_id=session_name,
                )
            cmd_args = [action, target]

        elif action in _TARGET_VALUE_ACTIONS:
            if not target or not value:
                return ErrorResponse(
                    message=f"'{action}' requires both 'target' and 'value'.",
                    error="missing_params",
                    session_id=session_name,
                )
            cmd_args = [action, target, value]

        elif action in _WAIT_ACTIONS:
            if not target:
                return ErrorResponse(
                    message=(
                        "'wait' requires a 'target': a CSS selector to wait for, "
                        "or milliseconds as a string (e.g. '1000')."
                    ),
                    error="missing_target",
                    session_id=session_name,
                )
            cmd_args = ["wait", target]

        else:
            return ErrorResponse(
                message=f"Unsupported action: {action}",
                error="invalid_action",
                session_id=session_name,
            )

        rc, _, stderr = await _run(session_name, *cmd_args)
        if rc != 0:
            logger.warning("[browser_act] %s failed: %s", action, stderr[:300])
            return ErrorResponse(
                message=f"Action '{action}' failed.",
                error="action_failed",
                session_id=session_name,
            )

        # Allow the page to settle after interaction (best-effort: SPAs may not idle)
        settle_rc, _, settle_err = await _run(
            session_name, "wait", "--load", "networkidle"
        )
        if settle_rc != 0:
            logger.warning(
                "[browser_act] post-action wait failed: %s", settle_err[:300]
            )

        snapshot = await _snapshot(session_name)
        _, url_out, _ = await _run(session_name, "get", "url")

        return BrowserActResponse(
            message=f"Performed '{action}'" + (f" on '{target}'" if target else ""),
            action=action,
            current_url=url_out.strip(),
            snapshot=snapshot,
            session_id=session_name,
        )


# ---------------------------------------------------------------------------
# Tool: browser_screenshot
# ---------------------------------------------------------------------------


class BrowserScreenshotTool(BaseTool):
    """Capture a screenshot of the current browser page and save it to the workspace."""

    @property
    def name(self) -> str:
        return "browser_screenshot"

    @property
    def description(self) -> str:
        return (
            "Take a screenshot of the current browser page and save it to the workspace. "
            "IMPORTANT: After calling this tool, immediately call read_workspace_file "
            "with the returned file_id to display the image inline to the user — "
            "the screenshot is not visible until you do this. "
            "With annotate=true (default), @ref labels are overlaid on interactive "
            "elements, making it easy to see which @ref ID maps to which element on screen."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "annotate": {
                    "type": "boolean",
                    "default": True,
                    "description": "Overlay @ref labels on interactive elements (default: true).",
                },
                "filename": {
                    "type": "string",
                    "default": "screenshot.png",
                    "description": "Filename to save in the workspace.",
                },
            },
        }

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return shutil.which("agent-browser") is not None

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        raw_annotate = kwargs.get("annotate", True)
        if isinstance(raw_annotate, str):
            annotate = raw_annotate.strip().lower() in {"1", "true", "yes", "on"}
        else:
            annotate = bool(raw_annotate)
        filename: str = (kwargs.get("filename") or "screenshot.png").strip()
        session_name = session.session_id

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".png")
        os.close(tmp_fd)
        try:
            cmd_args = ["screenshot"]
            if annotate:
                cmd_args.append("--annotate")
            cmd_args.append(tmp_path)

            rc, _, stderr = await _run(session_name, *cmd_args)
            if rc != 0:
                logger.warning("[browser_screenshot] failed: %s", stderr[:300])
                return ErrorResponse(
                    message="Failed to take screenshot.",
                    error="screenshot_failed",
                    session_id=session_name,
                )

            with open(tmp_path, "rb") as f:
                png_bytes = f.read()

        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass  # Best-effort temp file cleanup; not critical if it fails.

        # Upload to workspace so the user can view it
        png_b64 = base64.b64encode(png_bytes).decode()

        # Import here to avoid circular deps — workspace_files imports from .models
        from .workspace_files import (  # noqa: PLC0415
            WorkspaceWriteResponse,
            WriteWorkspaceFileTool,
        )

        write_resp = await WriteWorkspaceFileTool()._execute(
            user_id=user_id,
            session=session,
            filename=filename,
            content_base64=png_b64,
        )

        if not isinstance(write_resp, WorkspaceWriteResponse):
            return ErrorResponse(
                message="Screenshot taken but failed to save to workspace.",
                error="workspace_write_failed",
                session_id=session_name,
            )

        return BrowserScreenshotResponse(
            message=f"Screenshot saved to workspace as '{filename}'. Use read_workspace_file with file_id='{write_resp.file_id}' to retrieve it.",
            file_id=write_resp.file_id,
            filename=filename,
            session_id=session_name,
        )
