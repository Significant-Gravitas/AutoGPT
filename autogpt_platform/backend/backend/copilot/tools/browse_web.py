"""Web browsing tool — navigate real browser sessions to extract page content.

Uses Stagehand + Browserbase for cloud-based browser execution. Handles
JS-rendered pages, SPAs, and dynamic content that web_fetch cannot reach.

Requires environment variables:
    STAGEHAND_API_KEY     — Browserbase API key
    STAGEHAND_PROJECT_ID  — Browserbase project ID
    ANTHROPIC_API_KEY     — LLM key used by Stagehand for extraction
"""

import logging
import os
import threading
from typing import Any

from backend.copilot.model import ChatSession
from backend.util.request import validate_url

from .base import BaseTool
from .models import BrowseWebResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)

# Stagehand uses the LLM internally for natural-language extraction/actions.
_STAGEHAND_MODEL = "anthropic/claude-sonnet-4-5-20250929"
# Hard cap on extracted content returned to the LLM context.
_MAX_CONTENT_CHARS = 50_000
# Explicit timeouts for Stagehand browser operations (milliseconds).
_GOTO_TIMEOUT_MS = 30_000  # page navigation
_EXTRACT_TIMEOUT_MS = 60_000  # LLM extraction

# ---------------------------------------------------------------------------
# Thread-safety patch for Stagehand signal handlers (applied lazily, once).
#
# Stagehand calls signal.signal() during __init__, which raises ValueError
# when called from a non-main thread (e.g. the CoPilot executor thread pool).
# We patch _register_signal_handlers to be a no-op outside the main thread.
# The patch is applied exactly once per process via double-checked locking.
# ---------------------------------------------------------------------------
_stagehand_patched = False
_patch_lock = threading.Lock()


def _patch_stagehand_once() -> None:
    """Monkey-patch Stagehand signal handler registration to be thread-safe.

    Must be called after ``import stagehand.main`` has succeeded.
    Safe to call from multiple threads — applies the patch at most once.
    """
    global _stagehand_patched
    if _stagehand_patched:
        return
    with _patch_lock:
        if _stagehand_patched:
            return
        import stagehand.main  # noqa: PLC0415

        _original = stagehand.main.Stagehand._register_signal_handlers

        def _safe_register(self: Any) -> None:
            if threading.current_thread() is threading.main_thread():
                _original(self)

        stagehand.main.Stagehand._register_signal_handlers = _safe_register
        _stagehand_patched = True


class BrowseWebTool(BaseTool):
    """Navigate a URL with a real browser and extract its content.

    Use this instead of ``web_fetch`` when the page requires JavaScript
    to render (SPAs, dashboards, paywalled content with JS checks, etc.).
    """

    @property
    def name(self) -> str:
        return "browse_web"

    @property
    def description(self) -> str:
        return (
            "Navigate to a URL using a real browser and extract content. "
            "Handles JavaScript-rendered pages and dynamic content that "
            "web_fetch cannot reach. "
            "Specify exactly what to extract via the `instruction` parameter."
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
                "instruction": {
                    "type": "string",
                    "description": (
                        "What to extract from the page. Be specific — e.g. "
                        "'Extract all pricing plans with features and prices', "
                        "'Get the main article text and author', "
                        "'List all navigation links'. "
                        "Defaults to extracting the main page content."
                    ),
                    "default": "Extract the main content of this page.",
                },
            },
            "required": ["url"],
        }

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def is_available(self) -> bool:
        return bool(
            os.environ.get("STAGEHAND_API_KEY")
            and os.environ.get("STAGEHAND_PROJECT_ID")
            and os.environ.get("ANTHROPIC_API_KEY")
        )

    async def _execute(
        self,
        user_id: str | None,  # noqa: ARG002
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
        """Navigate to a URL with a real browser and return extracted content."""
        url: str = (kwargs.get("url") or "").strip()
        instruction: str = (
            kwargs.get("instruction") or "Extract the main content of this page."
        )
        session_id = session.session_id if session else None

        if not url:
            return ErrorResponse(
                message="Please provide a URL to browse.",
                error="missing_url",
                session_id=session_id,
            )

        # Full SSRF guard: resolves all DNS IPs, blocks RFC-1918/loopback/link-local/IPv6
        # (same guard used by HTTP blocks and browser_navigate).
        try:
            await validate_url(url, trusted_origins=[])
        except ValueError as e:
            return ErrorResponse(
                message=str(e),
                error="invalid_url",
                session_id=session_id,
            )

        api_key = os.environ.get("STAGEHAND_API_KEY")
        project_id = os.environ.get("STAGEHAND_PROJECT_ID")
        model_api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key or not project_id:
            return ErrorResponse(
                message=(
                    "Web browsing is not configured on this platform. "
                    "STAGEHAND_API_KEY and STAGEHAND_PROJECT_ID are required."
                ),
                error="not_configured",
                session_id=session_id,
            )

        if not model_api_key:
            return ErrorResponse(
                message=(
                    "Web browsing is not configured: ANTHROPIC_API_KEY is required "
                    "for Stagehand's extraction model."
                ),
                error="not_configured",
                session_id=session_id,
            )

        # Lazy import — Stagehand is an optional heavy dependency.
        # Importing here scopes any ImportError to this tool only, so other
        # tools continue to register and work normally if Stagehand is absent.
        try:
            from stagehand import Stagehand  # noqa: PLC0415
        except ImportError:
            return ErrorResponse(
                message="Web browsing is not available: Stagehand is not installed.",
                error="not_configured",
                session_id=session_id,
            )

        # Apply the signal handler patch now that we know stagehand is present.
        _patch_stagehand_once()

        client: Any | None = None
        try:
            client = Stagehand(
                api_key=api_key,
                project_id=project_id,
                model_name=_STAGEHAND_MODEL,
                model_api_key=model_api_key,
            )
            await client.init()

            page = client.page
            if page is None:
                raise RuntimeError("Stagehand page is not initialized")
            await page.goto(url, timeoutMs=_GOTO_TIMEOUT_MS)
            result = await page.extract(instruction, timeoutMs=_EXTRACT_TIMEOUT_MS)

            # Extract the text content from the Pydantic result model.
            raw = result.model_dump().get("extraction", "")
            content = str(raw) if raw else ""

            truncated = len(content) > _MAX_CONTENT_CHARS
            if truncated:
                suffix = "\n\n[Content truncated]"
                keep = max(0, _MAX_CONTENT_CHARS - len(suffix))
                content = content[:keep] + suffix

            return BrowseWebResponse(
                message=f"Browsed {url}",
                url=url,
                content=content,
                truncated=truncated,
                session_id=session_id,
            )

        except Exception:
            logger.exception("[browse_web] Failed for %s", url)
            return ErrorResponse(
                message="Failed to browse URL.",
                error="browse_failed",
                session_id=session_id,
            )
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass  # Best-effort cleanup; close failure must not mask the original result.
