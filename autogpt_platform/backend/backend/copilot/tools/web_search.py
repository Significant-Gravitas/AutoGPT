"""Web search tool — wraps Anthropic's server-side ``web_search`` beta.

Single entry point for web search on both SDK and baseline paths.  The
``web_search_20250305`` tool is server-side on Anthropic, so we call
the Messages API directly regardless of which LLM invoked the copilot
tool — OpenRouter can't proxy server-side tool execution.
"""

import logging
from typing import Any

from anthropic import AsyncAnthropic

from backend.copilot.model import ChatSession
from backend.copilot.token_tracking import persist_and_record_usage
from backend.util.settings import Settings

from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase, WebSearchResponse, WebSearchResult

logger = logging.getLogger(__name__)

_WEB_SEARCH_DISPATCH_MODEL = "claude-haiku-4-5"
_MAX_DISPATCH_TOKENS = 512
_DEFAULT_MAX_RESULTS = 5
_HARD_MAX_RESULTS = 20


class WebSearchTool(BaseTool):
    """Search the public web and return cited results."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for live info (news, recent docs). Returns "
            "{title, url, snippet}; use web_fetch to deep-dive a URL. "
            "Prefer one targeted query over many reformulations."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": (
                        f"Max results (default {_DEFAULT_MAX_RESULTS}, "
                        f"cap {_HARD_MAX_RESULTS})."
                    ),
                    "default": _DEFAULT_MAX_RESULTS,
                },
            },
            "required": ["query"],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    @property
    def is_available(self) -> bool:
        return bool(Settings().secrets.anthropic_api_key)

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        query: str = "",
        max_results: int = _DEFAULT_MAX_RESULTS,
        **kwargs: Any,
    ) -> ToolResponseBase:
        query = (query or "").strip()
        session_id = session.session_id if session else None
        if not query:
            return ErrorResponse(
                message="Please provide a non-empty search query.",
                error="missing_query",
                session_id=session_id,
            )

        try:
            max_results = int(max_results)
        except (TypeError, ValueError):
            max_results = _DEFAULT_MAX_RESULTS
        max_results = max(1, min(max_results, _HARD_MAX_RESULTS))

        api_key = Settings().secrets.anthropic_api_key
        if not api_key:
            return ErrorResponse(
                message=(
                    "Web search is unavailable — the deployment has no "
                    "Anthropic API key configured."
                ),
                error="web_search_not_configured",
                session_id=session_id,
            )

        client = AsyncAnthropic(api_key=api_key)
        try:
            resp = await client.messages.create(
                model=_WEB_SEARCH_DISPATCH_MODEL,
                max_tokens=_MAX_DISPATCH_TOKENS,
                tools=[
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": 1,
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Use the web_search tool exactly once with the "
                            f"query {query!r} and then stop.  Do not "
                            f"summarise — the caller parses the raw "
                            f"tool_result."
                        ),
                    }
                ],
            )
        except Exception as exc:
            logger.warning(
                "[web_search] Anthropic call failed for query=%r: %s", query, exc
            )
            return ErrorResponse(
                message=f"Web search failed: {exc}",
                error="web_search_failed",
                session_id=session_id,
            )

        results, search_requests = _extract_results(resp, limit=max_results)

        cost_usd = _estimate_cost_usd(resp, search_requests=search_requests)
        try:
            usage = getattr(resp, "usage", None)
            await persist_and_record_usage(
                session=session,
                user_id=user_id,
                prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
                completion_tokens=getattr(usage, "output_tokens", 0) or 0,
                log_prefix="[web_search]",
                cost_usd=cost_usd,
                model=_WEB_SEARCH_DISPATCH_MODEL,
                provider="anthropic",
            )
        except Exception as exc:
            logger.warning("[web_search] usage tracking failed: %s", exc)

        return WebSearchResponse(
            message=f"Found {len(results)} result(s) for {query!r}.",
            query=query,
            results=results,
            search_requests=search_requests,
            session_id=session_id,
        )


def _extract_results(resp: Any, *, limit: int) -> tuple[list[WebSearchResult], int]:
    """Pull results + server-side request count from an Anthropic response."""
    results: list[WebSearchResult] = []
    search_requests = 0

    for block in getattr(resp, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "web_search_tool_result":
            content = getattr(block, "content", []) or []
            for item in content:
                if getattr(item, "type", None) != "web_search_result":
                    continue
                if len(results) >= limit:
                    break
                # Anthropic's ``web_search_result`` exposes only
                # ``title``/``url``/``page_age`` plus an opaque
                # ``encrypted_content`` blob that is meant for citation
                # round-tripping, not for display — it is base64-ish
                # binary and would show as gibberish if surfaced to the
                # model or the frontend.  There is no plain-text snippet
                # field in the current beta; callers get the readable
                # text via the model's ``text`` blocks with citations,
                # not via this list.  Leave ``snippet`` empty.
                results.append(
                    WebSearchResult(
                        title=getattr(item, "title", "") or "",
                        url=getattr(item, "url", "") or "",
                        snippet="",
                        page_age=getattr(item, "page_age", None),
                    )
                )

    usage = getattr(resp, "usage", None)
    server_tool_use = getattr(usage, "server_tool_use", None) if usage else None
    if server_tool_use is not None:
        search_requests = getattr(server_tool_use, "web_search_requests", 0) or 0

    return results, search_requests


# Update when Anthropic revises pricing.
_COST_PER_SEARCH_USD = 0.010  # $10 per 1,000 web_search requests
_HAIKU_INPUT_USD_PER_MTOK = 1.0
_HAIKU_OUTPUT_USD_PER_MTOK = 5.0


def _estimate_cost_usd(resp: Any, *, search_requests: int) -> float:
    """Per-search fee × count + Haiku dispatch tokens."""
    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
    output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

    search_cost = search_requests * _COST_PER_SEARCH_USD
    inference_cost = (input_tokens / 1_000_000) * _HAIKU_INPUT_USD_PER_MTOK + (
        output_tokens / 1_000_000
    ) * _HAIKU_OUTPUT_USD_PER_MTOK
    return round(search_cost + inference_cost, 6)
