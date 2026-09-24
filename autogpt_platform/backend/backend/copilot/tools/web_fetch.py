"""Web fetch tool — safely retrieve public web page content."""

import logging
import re
from html import escape, unescape
from html.parser import HTMLParser
from typing import Any

import aiohttp
import html2text

from backend.copilot.model import ChatSession
from backend.util.request import Requests

from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase, WebFetchResponse

logger = logging.getLogger(__name__)

# Limits
_MAX_DOWNLOAD_BYTES = 2_097_152  # 2 MB response body cap to avoid OOM / stream DOS
_MAX_TEXT_CHARS = 100_000  # 100K characters text budget for the model
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

# SPA container and fallback signals
_SPA_SHELL_PATTERNS = re.compile(
    r"""(?<![-a-zA-Z0-9_])id\s*=\s*['"]?(?:root|__next|app)['"]?(?=[\s>/]|$)""",
    re.IGNORECASE,
)
_SPA_FALLBACK_TEXT = "you need to enable javascript to run this app"


class _HTMLCleaner(HTMLParser):
    """Filter out script, style, noscript, and svg elements from HTML before text extraction."""

    _DROP_TAGS = frozenset({"script", "style", "noscript", "svg"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._skip_depth = 0
        self._pieces: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._DROP_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0:
            attr_str = "".join(
                f' {k}="{escape(v, quote=True)}"' if v is not None else f" {k}"
                for k, v in attrs
            )
            self._pieces.append(f"<{tag}{attr_str}>")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._DROP_TAGS:
            if self._skip_depth > 0:
                self._skip_depth -= 1
                if self._skip_depth == 0:
                    self._pieces.append(" ")
        elif self._skip_depth == 0:
            self._pieces.append(f"</{tag}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in self._DROP_TAGS:
            if self._skip_depth == 0:
                self._pieces.append(" ")
        elif self._skip_depth == 0:
            attr_str = "".join(
                f' {k}="{escape(v, quote=True)}"' if v is not None else f" {k}"
                for k, v in attrs
            )
            self._pieces.append(f"<{tag}{attr_str} />")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._pieces.append(data)

    def handle_entityref(self, name: str) -> None:
        if self._skip_depth == 0:
            self._pieces.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if self._skip_depth == 0:
            self._pieces.append(f"&#{name};")

    def get_cleaned_html(self) -> str:
        return "".join(self._pieces)


def _is_text_content(content_type: str) -> bool:
    base = content_type.split(";")[0].strip().lower()
    return base in _TEXT_CONTENT_TYPES or base.startswith("text/")


def _html_to_text(html: str) -> str:
    cleaner = _HTMLCleaner()
    try:
        cleaner.feed(html)
        cleaned_html = cleaner.get_cleaned_html()
    except Exception as err:
        logger.debug(
            "HTML cleaner encountered an error, falling back to raw html: %s", err
        )
        cleaned_html = html
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.body_width = 0
    return h.handle(cleaned_html).strip()


_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _extract_title(html: str) -> str | None:
    match = _TITLE_RE.search(html)
    if not match:
        return None
    title = re.sub(r"\s+", " ", unescape(match.group(1))).strip()
    return title or None


def _is_client_rendered_shell(raw_html: str, extracted_text: str) -> bool:
    """Detect whether a page returned only an empty shell requiring JavaScript."""
    if len(extracted_text.strip()) >= 600:
        return False
    html_lower = raw_html.lower()
    has_spa_marker = bool(_SPA_SHELL_PATTERNS.search(raw_html)) or (
        _SPA_FALLBACK_TEXT in html_lower
    )
    has_script_payload = (
        len(raw_html) > 2048
        and len(extracted_text.strip()) < 300
        and "<script" in html_lower
    )
    return has_spa_marker or has_script_payload


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

        raw_bytes = response.content[:_MAX_DOWNLOAD_BYTES]
        raw_text = raw_bytes.decode("utf-8", errors="replace")

        title = None
        is_html = "html" in content_type.lower()
        raw_truncated = len(response.content) > _MAX_DOWNLOAD_BYTES
        text_truncated = False

        if is_html:
            title = _extract_title(raw_text)
            if extract_text:
                text = _html_to_text(raw_text)
            else:
                text = raw_text
        else:
            text = raw_text

        # Enforce character budget on the extracted text
        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS]
            text_truncated = True

        truncated = raw_truncated or text_truncated

        message = f"Fetched {url}"

        # Detect JavaScript-rendered SPA shells and surface an actionable hint
        if is_html and extract_text and _is_client_rendered_shell(raw_text, text):
            hint = (
                "[Notice: Content not rendered. This page appears to require JavaScript "
                "to render its content. Use the 'browser_navigate' tool instead.]"
            )
            text = f"{hint}\n\n{text}".strip()
            message = (
                f"Fetched {url} — warning: content not rendered (use browser_navigate)"
            )

        if text_truncated and raw_truncated:
            message += f" (download capped at {_MAX_DOWNLOAD_BYTES:,} bytes, text truncated to {_MAX_TEXT_CHARS:,} chars)"
            text += (
                f"\n\n[Content truncated — response exceeded {_MAX_DOWNLOAD_BYTES:,} bytes "
                f"and text was capped at {_MAX_TEXT_CHARS:,} characters]"
            )
        elif text_truncated:
            message += f" (truncated to {_MAX_TEXT_CHARS:,} chars)"
            text += f"\n\n[Content truncated — limit of {_MAX_TEXT_CHARS:,} characters reached]"
        elif raw_truncated:
            message += f" (raw content truncated at {_MAX_DOWNLOAD_BYTES:,} bytes)"
            text += f"\n\n[Content truncated — response body exceeded network cap of {_MAX_DOWNLOAD_BYTES:,} bytes]"

        return WebFetchResponse(
            message=message,
            url=response.url,
            status_code=response.status,
            content_type=content_type.split(";")[0].strip(),
            content=text,
            title=title,
            content_length=len(response.content),
            truncated=truncated,
            session_id=session_id,
        )
