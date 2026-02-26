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
import signal
import threading
from contextlib import contextmanager
from typing import Any, Generator

import stagehand.main
from stagehand import Stagehand

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import BrowseWebResponse, ErrorResponse, ToolResponseBase

logger = logging.getLogger(__name__)

# Stagehand uses the LLM internally for natural-language extraction/actions.
_STAGEHAND_MODEL = "anthropic/claude-sonnet-4-5-20250929"
# Hard cap on extracted content returned to the LLM context.
_MAX_CONTENT_CHARS = 50_000

# ---------------------------------------------------------------------------
# Thread-safety patch for Stagehand signal handlers
# Stagehand tries to register signal handlers on init. In worker threads
# (e.g. the CoPilot executor thread pool) this raises a ValueError because
# signal.signal() is only allowed in the main thread.
# ---------------------------------------------------------------------------
_original_register_signal_handlers = stagehand.main.Stagehand._register_signal_handlers


def _safe_register_signal_handlers(self: Any) -> None:
    if threading.current_thread() is threading.main_thread():
        _original_register_signal_handlers(self)


stagehand.main.Stagehand._register_signal_handlers = _safe_register_signal_handlers


@contextmanager
def _thread_safe_signal() -> Generator[None, None, None]:
    """Suppress signal.signal() calls when not in the main thread."""
    if threading.current_thread() is not threading.main_thread():
        original = signal.signal
        signal.signal = lambda *_: None  # type: ignore[assignment]
        try:
            yield
        finally:
            signal.signal = original
    else:
        yield


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

    async def _execute(
        self,
        user_id: str | None,  # noqa: ARG002
        session: ChatSession,
        **kwargs: Any,
    ) -> ToolResponseBase:
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

        if not url.startswith(("http://", "https://")):
            return ErrorResponse(
                message="Only HTTP/HTTPS URLs are supported.",
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

        client: Stagehand | None = None
        try:
            with _thread_safe_signal():
                client = Stagehand(
                    api_key=api_key,
                    project_id=project_id,
                    model_name=_STAGEHAND_MODEL,
                    model_api_key=model_api_key,
                )
                await client.init()

            page = client.page
            assert page is not None, "Stagehand page is not initialized"
            await page.goto(url)
            result = await page.extract(instruction)

            # Extract the text content from the Pydantic result model.
            raw = result.model_dump().get("extraction", "")
            content = str(raw) if raw else ""

            truncated = len(content) > _MAX_CONTENT_CHARS
            if truncated:
                content = content[:_MAX_CONTENT_CHARS] + "\n\n[Content truncated]"

            return BrowseWebResponse(
                message=f"Browsed {url}",
                url=url,
                content=content,
                truncated=truncated,
                session_id=session_id,
            )

        except Exception as e:
            logger.warning("[browse_web] Failed for %s: %s", url, e)
            return ErrorResponse(
                message=f"Failed to browse URL: {e}",
                error="browse_failed",
                session_id=session_id,
            )
        finally:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
