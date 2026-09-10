"""Web-fact-check tool — verifies a stale-fact candidate against the web.

The actual search backend choice (Brave / Tavily / Perplexity / …) is
deferred to product per ``dream/p0-spec.md`` §P0.5 open item #3, so
this module ships the *shape* of the tool only:

* :class:`WebFactCheckResult` — what the orchestrator hook receives
  back per candidate.
* :class:`SearchBackend` — protocol that any future provider adapter
  must satisfy. Keeping the surface narrow (one async ``verify``
  call) means swapping providers is a one-file change.
* :class:`_NullSearchBackend` — default backend that returns
  ``error="no_search_backend_configured"`` so the tool is callable
  end-to-end with no env config.
* :class:`WebFactCheckTool` — caller-facing wrapper that swallows
  backend exceptions into the result's ``error`` field; one bad
  lookup must not abort the dream pass.
"""

from __future__ import annotations

import logging
from typing import Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebFactCheckResult(BaseModel):
    """Per-fact verification result handed back to the dream pipeline."""

    fact_text: str
    verified: bool = False
    contradicted: bool = False
    sources: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    error: str | None = None


class SearchBackend(Protocol):
    """Pluggable search backend the tool delegates verification to.

    Returning a populated :class:`WebFactCheckResult` (rather than raw
    search hits) keeps backend-specific judging logic next to the
    backend that owns the search API and rate-limit budget.
    """

    async def verify(self, fact_text: str) -> WebFactCheckResult: ...


class _NullSearchBackend:
    """Default backend used when no provider has been wired in yet."""

    async def verify(self, fact_text: str) -> WebFactCheckResult:
        return WebFactCheckResult(
            fact_text=fact_text,
            verified=False,
            contradicted=False,
            sources=[],
            confidence=0.0,
            error="no_search_backend_configured",
        )


class WebFactCheckTool:
    """Caller-facing wrapper around a :class:`SearchBackend`.

    The backend choice is injected so tests and future provider PRs
    can swap implementations without touching the orchestrator. When
    no backend is passed, :class:`_NullSearchBackend` is used so the
    tool stays callable end-to-end in a fresh checkout.
    """

    def __init__(self, backend: SearchBackend | None = None) -> None:
        self._backend: SearchBackend = backend or _NullSearchBackend()

    async def verify(self, fact_text: str) -> WebFactCheckResult:
        try:
            return await self._backend.verify(fact_text)
        except Exception as exc:
            logger.warning(
                "web_fact_check backend raised for fact=%r: %s",
                fact_text[:80],
                exc,
            )
            return WebFactCheckResult(
                fact_text=fact_text,
                verified=False,
                contradicted=False,
                sources=[],
                confidence=0.0,
                error=f"backend_error:{type(exc).__name__}",
            )
