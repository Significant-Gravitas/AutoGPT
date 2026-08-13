"""Web fetch tool — safely retrieve public web page content."""

import logging
from typing import Any

import aiohttp
import html2text

from backend.copilot.model import ChatSession
from backend.util.request import Requests

from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase, WebFetchResponse

logger = logging.getLogger(__name__)

# Limits
_MAX_CONTENT_BYTES = 102_400  # 100 KB download cap
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=15)

# Content types we'll read as text
_TEXT_CONTENT_TYPES = {
    "text/html",
    "text/plain",
    "text/xml",
    "text/csv",
    "text/markdown",
    "application/json",
    "application/xml",
    "application/xhtml+xml",
    "application/rss+xml",
    "application/atom+xml",
    # RFC 7807 — JSON problem details; used by many REST APIs for error responses
    "application/problem+json",
    "application/problem+xml",
    "application/ld+json",
}


def _is_text_content(content_type: str) -> bool:
    base = content_type.split(";")[0].strip().lower()
    return base in _TEXT_CONTENT_TYPES or base.startswith("text/")


def _html_to_text(html: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    return h.handle(html)


class WebFetchTool(BaseTool):
    """Safely fetch content from a public URL using SSRF-protected HTTP."""

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch a public web page. Public URLs only — internal addresses blocked. Returns readable text from HTML by default."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Public HTTP/HTTPS URL.",
                },
                "extract_text": {
                    "type": "boolean",
                    "description": "Extract text from HTML (default: true).",
                    "default": True,
                },
            },
            "required": ["url"],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        url: str = "",
        extract_text: bool = True,
        **kwargs: Any,
    ) -> ToolResponseBase:
        url = url.strip()
        session_id = session.session_id if session else None

        if not url:
            return ErrorResponse(
                message="Please provide a URL to fetch.",
                error="missing_url",
                session_id=session_id,
            )

        try:
            client = Requests(raise_for_status=False, retry_max_attempts=1)
            response = await client.get(url, timeout=_REQUEST_TIMEOUT)
        except ValueError as e:
            # validate_url raises ValueError for SSRF / blocked IPs
            return ErrorResponse(
                message=f"URL blocked: {e}",
                error="url_blocked",
                session_id=session_id,
            )
        except Exception as e:
            logger.warning(f"[web_fetch] Request failed for {url}: {e}")
            return ErrorResponse(
                message=f"Failed to fetch URL: {e}",
                error="fetch_failed",
                session_id=session_id,
            )

        content_type = response.headers.get("content-type", "")
        if not _is_text_content(content_type):
            return ErrorResponse(
                message=f"Non-text content type: {content_type.split(';')[0]}",
                error="unsupported_content_type",
                session_id=session_id,
            )

        raw = response.content[:_MAX_CONTENT_BYTES]
        text = raw.decode("utf-8", errors="replace")

        if extract_text and "html" in content_type.lower():
            text = _html_to_text(text)

        return WebFetchResponse(
            message=f"Fetched {url}",
            url=response.url,
            status_code=response.status,
            content_type=content_type.split(";")[0].strip(),
            content=text,
            truncated=False,
            session_id=session_id,
        )
