"""Orchestrator hook — verify a batch of stale-fact candidates.

Called from the dream orchestrator before phase 3 sanitization
issues demotions, so the sanitizer prompt can see "the web also
contradicts this" / "the web confirms this" alongside the internal
heuristic score. The actual wiring is follow-up; this module just
exposes the function shape and the concurrency guarantees the
orchestrator needs:

* ``enabled=False`` → no-op, returns empty dict.
* No tool configured → no-op, returns empty dict.
* Otherwise, all candidates are verified concurrently via
  ``asyncio.gather``; per-item exceptions are caught and surfaced
  on the result's ``error`` field so one bad lookup cannot kill
  the batch.

The return map is keyed by ``fact.uuid`` so the orchestrator can
join verification results back onto its candidate list without
re-matching on fact text.
"""

from __future__ import annotations

import asyncio
import logging

from backend.copilot.dream.fetch import FactRow
from backend.copilot.dream.webcheck.tool import WebFactCheckResult, WebFactCheckTool

logger = logging.getLogger(__name__)


async def verify_stale_candidates(
    candidates: list[FactRow],
    *,
    enabled: bool,
    tool: WebFactCheckTool | None = None,
) -> dict[str, WebFactCheckResult]:
    if not enabled or tool is None or not candidates:
        return {}

    async def _verify_one(candidate: FactRow) -> tuple[str, WebFactCheckResult]:
        fact_text = candidate.fact or ""
        try:
            result = await tool.verify(fact_text)
        except Exception as exc:
            logger.warning(
                "web_fact_check hook caught exception for uuid=%s: %s",
                candidate.uuid,
                exc,
            )
            result = WebFactCheckResult(
                fact_text=fact_text,
                verified=False,
                contradicted=False,
                sources=[],
                confidence=0.0,
                error=f"hook_error:{type(exc).__name__}",
            )
        return candidate.uuid, result

    pairs = await asyncio.gather(*(_verify_one(c) for c in candidates))
    return dict(pairs)
