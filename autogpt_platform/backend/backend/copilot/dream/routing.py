"""Decide which execution path a dream pass runs through.

Per ``dream/p0-spec.md`` §13, the dream pass must work on every
transport (openrouter / subscription / openai_compat / local). The one
batch path is Anthropic's Message Batches API (50% discount); it is
chosen when a direct Anthropic key is present and the batch flag is on.
Anything else falls back to the synchronous baseline path through the
existing ``copilot_executor``.
"""

from __future__ import annotations

from typing import Literal

from backend.copilot.config import TransportName

# The single source of truth for execution paths; ``billing``,
# ``orchestrator`` and ``schemas`` import it from here.
ExecutionPath = Literal["sync_baseline", "anthropic_batch"]

# Anthropic's Message Batches API bills half the synchronous list price.
_ANTHROPIC_BATCH_DISCOUNT = 0.5


# Transports that cannot honour the batch path even when an Anthropic key
# is configured separately.
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
    batch_processing_enabled: bool = False,
    transport_name: TransportName | None = None,
) -> ExecutionPath:
    """Pick the dream pass execution path.

    The batch path requires a direct Anthropic API key AND the
    ``batch_processing_enabled`` flag. Falls back to sync_baseline
    otherwise — slower and pricier per pass, but works on every
    transport.

    ``transport_name`` (when provided) takes precedence over the key
    flags: ``"local"`` and ``"subscription"`` always force
    ``sync_baseline`` regardless of which API keys happen to be set
    (see ``_TRANSPORTS_FORCE_SYNC`` above for the rationale).
    Defaults to ``None`` so callers that don't know their transport
    still get the historical key-driven behaviour.
    """
    if transport_name in _TRANSPORTS_FORCE_SYNC:
        return "sync_baseline"
    if batch_processing_enabled and has_anthropic_key:
        return "anthropic_batch"
    return "sync_baseline"


def batch_discount(path: ExecutionPath) -> float:
    """The share of the list price the provider takes off on *path*: half
    on Anthropic's Message Batches API, nothing on the sync path.

    Priced into the dream's cost rows and recorded next to them as
    ``discount_applied``.
    """
    return _ANTHROPIC_BATCH_DISCOUNT if path == "anthropic_batch" else 0.0
