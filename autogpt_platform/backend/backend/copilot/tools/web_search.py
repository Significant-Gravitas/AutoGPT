"""Web search tool — Perplexity Sonar via OpenRouter.

One provider, two tiers, one billing path:

* ``deep=False`` (default) — ``perplexity/sonar``.  Searches the web
  natively and returns citation annotations in a single inference pass.
* ``deep=True`` — ``perplexity/sonar-deep-research``.  Multi-step
  agentic research; slower and costlier.

Why Sonar and not the ``openrouter:web_search`` server tool + dispatch
model?  The server tool feeds all search-result page content back into
the dispatch model for a second inference pass — one observed call was
74K input tokens at Gemini Flash rates, billing $0.072.  Sonar
searches natively in one pass, returns annotations typed as
``ChatCompletionMessage.annotations`` in ``openai.types``, and at
$1 / MTok base pricing lands ~$0.01 / call at our default shape.

``resp.usage.cost`` carries the real billed value via OpenRouter's
``include: true`` extension; the value flows through
``persist_and_record_usage(provider='open_router')`` into the daily /
weekly microdollar rate-limit counter on the same rails as every other
OpenRouter turn — no separate provider ledger line, no estimation
drift.  ``_extract_cost_usd`` mirrors the baseline service's
``_extract_usage_cost`` logic; keep the two in sync if one changes.
"""

import logging
import math
from typing import Any

from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatSession
from backend.copilot.token_tracking import persist_and_record_usage

from .base import BaseTool
from .models import ErrorResponse, ToolResponseBase, WebSearchResponse, WebSearchResult

logger = logging.getLogger(__name__)

_chat_config = ChatConfig()

_QUICK_MODEL = "perplexity/sonar"
# Sonar base can emit up to ~4K output; cap at the provider ceiling so the
# model stops when the answer is complete rather than when our budget trips.
_QUICK_MAX_TOKENS = 4096

_DEEP_MODEL = "perplexity/sonar-deep-research"
# Deep runs can produce long structured writeups — ~4x the quick ceiling
# is enough headroom for multi-source comparisons without uncapping.
_DEEP_MAX_TOKENS = _QUICK_MAX_TOKENS * 4

_DEFAULT_MAX_RESULTS = 5
_HARD_MAX_RESULTS = 20
_SNIPPET_MAX_CHARS = 500

# OpenRouter-specific extra_body flag that embeds the real generation
# cost into the response usage object.  Same dict shape the baseline
# service uses — keep the two aligned.
_OPENROUTER_INCLUDE_USAGE_COST: dict[str, Any] = {"usage": {"include": True}}


class WebSearchTool(BaseTool):
    """Search the public web and return cited results."""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for live info (news, recent docs). Returns a "
            "synthesised answer grounded in fresh page content plus "
            "{title, url, snippet} citations — read the answer first "
            "before reaching for web_fetch. Set deep=true when the user "
            "asks for research / comparison / in-depth analysis; leave "
            "deep=false for quick fact lookups. Prefer one targeted "
            "query over many reformulations."
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
                "deep": {
                    "type": "boolean",
                    "description": (
                        "Only set true when the user EXPLICITLY asks for "
                        "research, comparison, or in-depth investigation "
                        "across many sources — it is ~100x more expensive "
                        "and much slower than a normal search. Default "
                        "false; do not flip it for ordinary fact lookups "
                        "or fresh-news questions."
                    ),
                    "default": False,
                },
            },
            "required": ["query"],
        }

    @property
    def requires_auth(self) -> bool:
        return False

    @property
    def is_available(self) -> bool:
        return bool(_chat_config.api_key and _chat_config.base_url)

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        query: str = "",
        max_results: int = _DEFAULT_MAX_RESULTS,
        deep: bool = False,
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

        if not _chat_config.api_key or not _chat_config.base_url:
            return ErrorResponse(
                message=(
                    "Web search is unavailable — the deployment has no "
                    "OpenRouter credentials configured."
                ),
                error="web_search_not_configured",
                session_id=session_id,
            )

        client = AsyncOpenAI(
            api_key=_chat_config.api_key, base_url=_chat_config.base_url
        )
        model_used = _DEEP_MODEL if deep else _QUICK_MODEL
        max_tokens = _DEEP_MAX_TOKENS if deep else _QUICK_MAX_TOKENS

        try:
            resp = await client.chat.completions.create(
                model=model_used,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": query}],
                extra_body=_OPENROUTER_INCLUDE_USAGE_COST,
            )
        except Exception as exc:
            logger.warning(
                "[web_search] OpenRouter call failed (deep=%s) for query=%r: %s",
                deep,
                query,
                exc,
            )
            return ErrorResponse(
                message=f"Web search failed: {exc}",
                error="web_search_failed",
                session_id=session_id,
            )

        answer = _extract_answer(resp)
        results = _extract_results(resp, limit=max_results)
        cost_usd = _extract_cost_usd(resp.usage)

        try:
            await persist_and_record_usage(
                session=session,
                user_id=user_id,
                prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
                log_prefix="[web_search]",
                cost_usd=cost_usd,
                model=model_used,
                provider="open_router",
            )
        except Exception as exc:
            logger.warning("[web_search] usage tracking failed: %s", exc)

        return WebSearchResponse(
            message=f"Found {len(results)} result(s) for {query!r}.",
            query=query,
            answer=answer,
            results=results,
            search_requests=1 if results else 0,
            session_id=session_id,
        )


def _extract_answer(resp: ChatCompletion) -> str:
    """Return the synthesised answer text from Sonar's response.

    Sonar reads every page it cites and writes a web-grounded synthesis
    into ``choices[0].message.content`` on the same call we pay for.
    Surfacing it saves the agent from re-fetching citation URLs — many
    are bot-protected and ``web_fetch`` can't reach them.
    """
    if not resp.choices:
        return ""
    content = resp.choices[0].message.content
    return content or ""


def _extract_results(resp: ChatCompletion, *, limit: int) -> list[WebSearchResult]:
    """Pull ``url_citation`` annotations from the response.

    Shared across both tiers — OpenRouter normalises the annotation
    schema across Perplexity's sonar models into
    ``Annotation.url_citation`` (typed in ``openai.types.chat``).  The
    ``content`` snippet is an OpenRouter extension on the otherwise-
    typed ``AnnotationURLCitation``; pydantic stashes unknown fields in
    ``model_extra``, which we read there rather than via ``getattr``.
    """
    if not resp.choices:
        return []
    annotations = resp.choices[0].message.annotations or []
    out: list[WebSearchResult] = []
    for ann in annotations:
        if len(out) >= limit:
            break
        if ann.type != "url_citation":
            continue
        citation = ann.url_citation
        extras = citation.model_extra or {}
        snippet_raw = extras.get("content")
        snippet = (snippet_raw or "")[:_SNIPPET_MAX_CHARS] if snippet_raw else ""
        out.append(
            WebSearchResult(
                title=citation.title,
                url=citation.url,
                snippet=snippet,
                page_age=None,
            )
        )
    return out


def _extract_cost_usd(usage: CompletionUsage | None) -> float | None:
    """Return the provider-reported USD cost off the response usage.

    OpenRouter piggybacks a ``cost`` field on the OpenAI-compatible
    usage object when the request body includes
    ``usage: {"include": True}``.  The OpenAI SDK's typed
    ``CompletionUsage`` does not declare it, so we read it off
    ``model_extra`` (the pydantic v2 container for extras) to keep
    access fully typed — no ``getattr``.  Mirrors the baseline service
    ``_extract_usage_cost``; keep the two in sync.

    Returns ``None`` when the field is absent, null, non-numeric,
    non-finite, or negative.  Invalid values log at error level because
    they indicate a provider bug worth chasing; plain absences are
    silent so the caller can dedupe the "missing cost" warning.
    """
    if usage is None:
        return None
    extras = usage.model_extra or {}
    if "cost" not in extras:
        return None
    raw = extras["cost"]
    if raw is None:
        logger.error("[web_search] usage.cost is present but null")
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        logger.error("[web_search] usage.cost is not numeric: %r", raw)
        return None
    if not math.isfinite(val) or val < 0:
        logger.error("[web_search] usage.cost is non-finite or negative: %r", val)
        return None
    return val
