"""Decide whether a dream pass runs via Anthropic batch or sync baseline.

Per ``dream/p0-spec.md`` §13, the dream pass must work on every
transport (openrouter / subscription / openai_compat / local). Batch
is only available when we have a direct Anthropic API key AND the
batch flag is on. Anything else falls back to the synchronous baseline
path through the existing ``copilot_executor``.

Slice 1 of P-0 ships sync-baseline only. The batch path is a follow-up
PR; this resolver is here from day one so adding batch is a one-line
flip, not a routing-redesign.
"""

from __future__ import annotations

from typing import Literal

ExecutionPath = Literal["batch", "sync_baseline"]


def resolve_dream_execution_path(
    *,
    has_anthropic_key: bool,
    batch_processing_enabled: bool,
) -> ExecutionPath:
    """Pick the dream pass execution path.

    ``batch`` requires both a direct Anthropic API key (separate from
    the OpenRouter / Claude SDK creds) and the ``batch_processing_enabled``
    config / flag being on. Otherwise we run synchronously through the
    standard copilot executor — slower and pricier per pass, but works
    on every transport.
    """
    if has_anthropic_key and batch_processing_enabled:
        return "batch"
    return "sync_baseline"
