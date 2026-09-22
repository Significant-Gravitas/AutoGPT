"""Decide which execution path a dream pass runs through.

Per ``dream/p0-spec.md`` §13, the dream pass must work on every
transport (openrouter / subscription / openai_compat / local). Two
batch APIs are supported (both Anthropic and OpenAI offer a 50%
discount); we route to whichever has a direct API key present + the
batch flag on. Anything else falls back to the synchronous baseline
path through the existing ``copilot_executor``.

Slice 1 of P-0 ships sync-baseline only. Both batch paths land in
task #33 (P0.1). This resolver is wired from day one so adding the
batch executors is just flipping the conditional below.
"""

from __future__ import annotations

from typing import Literal

from backend.copilot.config import TransportName

# Kept in sync with ``model_pricing.ExecutionPath`` so the discount
# table in model_pricing.py reads our chosen path directly. If you add
# a new path here, mirror it there.
ExecutionPath = Literal["sync_baseline", "anthropic_batch", "openai_batch"]


# Transports that cannot honour a batch path even when an Anthropic /
# OpenAI key is configured separately.
#
# - ``"local"`` — local backends (Ollama, vLLM, …) don't expose batch
#   endpoints; even if the operator set ``ANTHROPIC_API_KEY`` for some
#   unrelated reason, the dream pass should stay on the local LLM that
#   the rest of the install uses (and that ``CHAT_USE_LOCAL=true``
#   explicitly opted in to).
# - ``"subscription"`` — the chat layer authenticates via Claude Code
#   OAuth (no API key); dual-billing the user's Anthropic API key for
#   the dream pass would surprise the operator and split the cost-log
#   audit trail across two surfaces. Force sync so dream + chat
#   continue to roll up under the same provider label
#   (``cost_log_provider="anthropic"``).
_TRANSPORTS_FORCE_SYNC: frozenset[TransportName] = frozenset({"local", "subscription"})


def resolve_dream_execution_path(
    *,
    has_anthropic_key: bool,
    has_openai_key: bool = False,
    batch_processing_enabled: bool = False,
    transport_name: TransportName | None = None,
) -> ExecutionPath:
    """Pick the dream pass execution path.

    Both batch paths require their respective provider API key AND the
    ``batch_processing_enabled`` flag. Anthropic batch is preferred
    when both keys are present (better cache pricing for the consolidation
    + sanitization phases that share context). Falls back to sync_baseline
    when no batch path is viable — slower and pricier per pass, but
    works on every transport.

    ``transport_name`` (when provided) takes precedence over the key
    flags: ``"local"`` and ``"subscription"`` always force
    ``sync_baseline`` regardless of which API keys happen to be set
    (see ``_TRANSPORTS_FORCE_SYNC`` above for the rationale).
    Defaults to ``None`` so callers that don't know their transport
    still get the historical key-driven behaviour.
    """
    if transport_name in _TRANSPORTS_FORCE_SYNC:
        return "sync_baseline"
    if batch_processing_enabled:
        if has_anthropic_key:
            return "anthropic_batch"
        if has_openai_key:
            return "openai_batch"
    return "sync_baseline"
